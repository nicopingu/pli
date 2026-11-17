"""PLI (16/11/2026): PLI's own background tasks, run in the quiet moments of service.

Outside pli/additions on purpose: those files enter the checkpoints' key (additions/code_key.py); the tasks never compute a checkpoint's bits, so a pose
that changes them keeps the saved conversations.

Decision (15/11/2026 and 16/11/2026): PLI runs its background work itself (pruning, the drafter's
changes, later FALCON B's adjustment), in the gaps between requests, without cutting the service; the external timers
that drove it from outside go away.

PLI_TASKS=name[,name...] switches tasks on. Empty (the default): no task, the server behaves exactly as before.
Known tasks: ``prune`` (session snapshots, pli/tasks/prune.py), ``drafter`` (the DFlash2 drafter changed
between two requests and watched in service, pli/tasks/drafter.py) and ``falcon`` (FALCON B's adjustment of
the drafter on the service's own tokens, with the served 27B, pli/tasks/falcon.py).

When: the engine thread calls :meth:`Tasks.tick` where it would otherwise wait for a request (no stream live, no prompt
open, nothing queued). A step runs only after PLI_TASKS_CALM_S seconds (30) without engine activity and while no
generation child (image, voice, video) is working; one step a tick, the due tasks in turn. A step is short: a request
that arrives waits at most for the end of the step in progress, then the engine serves it before any other step.

Memory: a task tells what its next step needs (``need_bytes``); the step waits while MLX's active memory plus that need
passes PLI_TASKS_MEMORY_FRACTION (0.9) of the GPU's recommended working set.

Errors: a step that raises is logged; PLI_TASKS_MAX_ERRORS (3) in a row switch the task off until the server restarts.
A task never makes a request fail.

Journal: PLI_TASKS_LOG=<file>, one JSON line per event (t, task, event, then the event's fields), moved to <file>.1
past 20 MB. State: PLI_TASKS_STATE=<dir>, one JSON file per task (its progress), read back at start, written atomically.
"""

from __future__ import annotations

import json
import os
import threading
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Iterable

LOG_MAX_BYTES = 20 * 1024 * 1024


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return float(default)


def mlx_memory_ok(need: int, fraction: float) -> bool:
    """Whether MLX's active memory plus ``need`` bytes stays under ``fraction`` of the recommended working set."""

    if need <= 0:
        return True
    try:
        import mlx.core as mx

        info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
        limit = float(info.get("max_recommended_working_set_size", 0) or 0)
        active = float(mx.get_active_memory())
    except Exception:  # noqa: BLE001 - no reading: no step that needs memory
        return False
    return limit > 0 and active + need <= fraction * limit


class Task:
    """One background task: ``due`` says whether it has work now, ``step`` does a short piece of it."""

    name = ""

    def __init__(self, tasks: "Tasks", context: dict[str, Any]) -> None:
        self.tasks = tasks
        self.context = context
        self.state: dict[str, Any] = tasks.load_state(self.name)
        self.errors = 0          # in a row
        self.off = False
        self.steps = 0

    def due(self, now: float) -> bool:
        return False

    def need_bytes(self) -> int:
        return 0

    def step(self, engine: Any) -> None:
        raise NotImplementedError

    def observe(self, **request: Any) -> None:
        """A finished request (prompt, cached, generation, drafted, accepted): for tasks that watch the service."""

    def summary(self) -> dict[str, Any]:
        return {}

    def save(self) -> None:
        self.tasks.save_state(self.name, self.state)

    def log(self, event: str, **fields: Any) -> None:
        self.tasks.log(self.name, event, **fields)


class Tasks:
    """The background tasks of one server, stepped by its engine thread."""

    def __init__(self, tasks: Iterable[Task] = (), *, calm_s: float = 30.0, log_path: str = "", state_dir: str = "",
                 memory_fraction: float = 0.9, max_errors: int = 3, busy: Callable[[], bool] | None = None,
                 clock: Callable[[], float] = time.monotonic, wall: Callable[[], float] = time.time,
                 memory_ok: Callable[[int, float], bool] = mlx_memory_ok) -> None:
        self.calm_s = float(calm_s)
        self.log_path = log_path
        self.state_dir = Path(state_dir) if state_dir else None
        self.memory_fraction = float(memory_fraction)
        self.max_errors = max(1, int(max_errors))
        self.busy = busy
        self.clock, self.wall, self.memory_ok = clock, wall, memory_ok
        self._lock = threading.Lock()
        self._next = 0
        self.tasks: list[Task] = list(tasks)
        self.waits_memory = 0

    # -- the engine thread --------------------------------------------------------------------------------------
    def tick(self, engine: Any, last_activity: float) -> bool:
        """One step of the next due task when the engine has been quiet for ``calm_s``; True when a step ran."""

        now = self.clock()
        if now - last_activity < self.calm_s:
            return False
        live = [t for t in self.tasks if not t.off]
        if not live:
            return False
        if self.busy is not None:
            try:
                if self.busy():
                    return False
            except Exception:  # noqa: BLE001 - an unreadable child state counts as busy
                return False
        for k in range(len(live)):
            task = live[(self._next + k) % len(live)]
            try:
                if not task.due(self.wall()):
                    continue
                need = int(task.need_bytes() or 0)
            except Exception as exc:  # noqa: BLE001
                self._failed(task, exc, "due")
                continue
            if need and not self.memory_ok(need, self.memory_fraction):
                self.waits_memory += 1
                continue
            self._next = (self._next + k + 1) % len(live)
            started = time.perf_counter()
            try:
                task.step(engine)
            except Exception as exc:  # noqa: BLE001 - a task never stops the server
                self._failed(task, exc, "step")
                return True
            task.errors = 0
            task.steps += 1
            spent = time.perf_counter() - started
            if spent > 2.0:
                task.log("slow_step", seconds=round(spent, 2))
            return True
        return False

    def _failed(self, task: Task, exc: BaseException, where: str) -> None:
        task.errors += 1
        task.log("error", where=where, error=f"{type(exc).__name__}: {exc}",
                 trace="".join(traceback.format_exception(exc))[-1500:], in_a_row=task.errors)
        if task.errors >= self.max_errors:
            task.off = True
            task.log("off", reason=f"{task.errors} errors in a row")
            print(f"[PLI] background task {task.name} switched off after {task.errors} errors in a row", flush=True)

    # -- request threads ----------------------------------------------------------------------------------------
    def observe(self, **request: Any) -> None:
        for task in self.tasks:
            if task.off:
                continue
            try:
                task.observe(**request)
            except Exception as exc:  # noqa: BLE001 - never the request
                self._failed(task, exc, "observe")

    def summary(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for task in self.tasks:
            try:
                extra = task.summary()
            except Exception:  # noqa: BLE001
                extra = {}
            out[task.name] = {"off": task.off, "steps": task.steps, "errors": task.errors, **extra}
        return out

    # -- journal and state ----------------------------------------------------------------------------------------
    def log(self, task: str, event: str, **fields: Any) -> None:
        if not self.log_path:
            return
        line = json.dumps({"t": round(self.wall(), 3), "task": task, "event": event, **fields}, ensure_ascii=False,
                          default=str)
        try:
            with self._lock:
                path = Path(self.log_path)
                if path.exists() and path.stat().st_size > LOG_MAX_BYTES:
                    path.replace(path.with_name(path.name + ".1"))
                with open(path, "a", encoding="utf-8") as handle:
                    handle.write(line + "\n")
        except OSError:
            pass

    def load_state(self, name: str) -> dict[str, Any]:
        if self.state_dir is None:
            return {}
        try:
            return dict(json.loads((self.state_dir / f"{name}.json").read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError):
            return {}

    def save_state(self, name: str, state: dict[str, Any]) -> None:
        if self.state_dir is None:
            return
        try:
            self.state_dir.mkdir(parents=True, exist_ok=True)
            tmp = self.state_dir / f".{name}.json.tmp"
            tmp.write_text(json.dumps(state, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
            tmp.replace(self.state_dir / f"{name}.json")
        except OSError:
            pass


def _registry() -> dict[str, Callable[[Tasks, dict[str, Any]], Task]]:
    from pli.tasks import drafter as tasks_drafter, falcon as tasks_falcon, prune as tasks_prune

    return {"prune": tasks_prune.Prune, "drafter": tasks_drafter.DrafterSwap, "falcon": tasks_falcon.Falcon}


def names_from_env() -> list[str]:
    return [n.strip() for n in os.environ.get("PLI_TASKS", "").split(",") if n.strip()]


def from_env(context: dict[str, Any], *, busy: Callable[[], bool] | None = None) -> Tasks | None:
    """The tasks PLI_TASKS asks for, or None (the default: none). An unknown name is said and skipped."""

    names = names_from_env()
    if not names:
        return None
    tasks = Tasks(calm_s=_env_float("PLI_TASKS_CALM_S", 30.0), log_path=os.environ.get("PLI_TASKS_LOG", "").strip(),
                  state_dir=os.environ.get("PLI_TASKS_STATE", "").strip(),
                  memory_fraction=_env_float("PLI_TASKS_MEMORY_FRACTION", 0.9),
                  max_errors=int(_env_float("PLI_TASKS_MAX_ERRORS", 3)), busy=busy)
    registry = _registry()
    for name in names:
        make = registry.get(name)
        if make is None:
            print(f"[PLI] PLI_TASKS: unknown task {name!r} (known: {', '.join(sorted(registry))})", flush=True)
            continue
        try:
            task = make(tasks, context)
        except Exception as exc:  # noqa: BLE001 - a task that cannot start is left out, the server starts
            print(f"[PLI] background task {name} not started: {type(exc).__name__}: {exc}", flush=True)
            continue
        tasks.tasks.append(task)
        tasks.log(name, "start", calm_s=tasks.calm_s)
    if not tasks.tasks:
        return None
    print(f"[PLI] background tasks: {', '.join(t.name for t in tasks.tasks)} after {tasks.calm_s:g} s of calm",
          flush=True)
    return tasks
