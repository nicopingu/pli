"""PLI: step-by-step trace of requests, to find the slow steps. Built into the engine, switched on when needed.

Two ways to switch it on:
  ``PLI_TRACE_DIR=<dir>``          trace ready, off at start. It switches on without restarting the engine while
                                   ``<dir>/trace.on`` exists (checked every second), and writes to
                                   ``<dir>/trace-<model>.jsonl``. Switch:
                                   ``python -m tensorfold.pli.trace on|off|status --dir <dir> [--duration 24h]``.
                                   With ``--duration``, the trace switches itself off at that time.
  ``PLI_TRACE=<file.jsonl>``       on from the start, into that file (benches, tests).
With neither, every call returns at once.

One JSON line per step: t (Unix time, 6 decimals), e (step), req (request) when known, then the step's
fields. Touches neither the computation nor the order of requests: no GPU synchronisation added, same
text produced. Durations are those the code already measures (perf_counter) or gaps between two Python
points; "GPU wait" is the time spent in the read that forces the computation, not the GPU's own time.
Writing is done by a separate thread: the engine loop only queues a dict. An error is swallowed,
never the request. The file moves to ``<file>.1`` past PLI_TRACE_MAX_MB (500 MB by default).

Steps: request, open, prefix_resume, chunk, sub_chunk, checkpoints (prefill), round, round_p (decode),
draft, ev.* (actions of the event log), trace (on, off). Reading:
``python -m tensorfold.pli.trace_analysis <trace.jsonl>`` (it also reads traces written before 3.7.0, with French names).
"""

from __future__ import annotations

import argparse
import atexit
import json
import os
import queue
import sys
import threading
import time
from typing import Any

_FORCE = os.environ.get("PLI_TRACE", "").strip()
_DIR = os.environ.get("PLI_TRACE_DIR", "").strip()
try:
    _MAX_BYTES = max(1, int(os.environ.get("PLI_TRACE_MAX_MB", "500"))) * 1024 * 1024
except ValueError:
    _MAX_BYTES = 500 * 1024 * 1024
_T0 = time.perf_counter()
_W0 = time.time()
_QUEUE: "queue.SimpleQueue[dict[str, Any]]" = queue.SimpleQueue()
_WRITER: threading.Thread | None = None
_WATCHER: threading.Thread | None = None
_LOCK = threading.Lock()
_ACTIVE = bool(_FORCE)
_PATH = _FORCE
SWITCH = "trace.on"


def active() -> bool:
    if _DIR and _WATCHER is None:
        _watch()
    return _ACTIVE


def wall_time(t: float) -> float:
    """Unix time of a ``perf_counter`` instant."""

    return _W0 + (t - _T0)


def pt(step: str, req: Any = None, t: float | None = None, **fields: Any) -> None:
    """Queue a step. ``t``: perf_counter instant of the step's start (default: now)."""

    if not _ACTIVE:
        return
    try:
        d: dict[str, Any] = {"t": round(wall_time(time.perf_counter() if t is None else t), 6), "e": step}
        if req is not None:
            d["req"] = str(req)
        for key, value in fields.items():
            if value is None:
                continue
            d[key] = round(value, 3) if isinstance(value, float) else value
        _start()
        _QUEUE.put(d)
    except Exception:  # noqa: BLE001 - the trace never makes a request fail
        pass


def line(d: dict[str, Any]) -> None:
    """Queue a ready-made line (event log); "t" in Unix time."""

    if not _ACTIVE:
        return
    try:
        _start()
        _QUEUE.put(dict(d))
    except Exception:  # noqa: BLE001
        pass


# -- switch ------------------------------------------------------------------------------------------------

def _model_name() -> str:
    """Served name (--name), else the model's folder, read from the command line of "tensorfold serve"."""

    args = sys.argv
    try:
        if "--name" in args:
            return args[args.index("--name") + 1]
        if "serve" in args:
            return os.path.basename(str(args[args.index("serve") + 1]).rstrip("/"))
    except (IndexError, ValueError):
        pass
    return "engine"


def _deadline(path: str) -> float | None:
    """Unix time of switching off written in the switch ("until=<time>", or "jusqu=" before 3.7.0), else None."""

    try:
        with open(path, encoding="utf-8") as f:
            for word in f.read().split():
                if word.startswith("until="):
                    return float(word[6:])
                if word.startswith("jusqu="):
                    return float(word[6:])
    except (OSError, ValueError):
        pass
    return None


def _watch() -> None:
    global _WATCHER, _PATH
    with _LOCK:
        if _WATCHER is not None:
            return
        name = "".join(c if c.isalnum() or c in "-_." else "_" for c in _model_name())
        _PATH = os.path.join(_DIR, f"trace-{name}.jsonl")
        _WATCHER = threading.Thread(target=_watcher, name="pli-trace-switch", daemon=True)
        _WATCHER.start()


def _watcher() -> None:
    global _ACTIVE
    folder = _DIR
    switch = os.path.join(folder, SWITCH)
    while _DIR == folder and not _FORCE:      # module reloaded otherwise, or trace forced: nothing to watch
        try:
            present = os.path.exists(switch)
            end = _deadline(switch) if present else None
            if present and end is not None and time.time() >= end:
                try:
                    os.remove(switch)
                except OSError:
                    pass
                present = False
            if present != _ACTIVE:
                if not present:
                    pt("trace", state="off")
                _ACTIVE = present
                if present:
                    pt("trace", state="on", pid=os.getpid(), until=end)
        except Exception:  # noqa: BLE001
            pass
        time.sleep(1.0)


# -- writing -----------------------------------------------------------------------------------------------

def _start() -> None:
    global _WRITER
    if _WRITER is not None:
        return
    with _LOCK:
        if _WRITER is None:
            _WRITER = threading.Thread(target=_write, name="pli-trace", daemon=True)
            _WRITER.start()
            atexit.register(flush)


def _write() -> None:
    batch: list[str] = []
    last = time.monotonic()
    while True:
        try:
            d: dict[str, Any] | None = _QUEUE.get(timeout=0.5)
        except queue.Empty:
            d = None
        if d is not None:
            try:
                batch.append(json.dumps(d, ensure_ascii=False, separators=(",", ":")))
            except (TypeError, ValueError):
                pass
        if batch and (len(batch) >= 2000 or time.monotonic() - last >= 0.5 or d is None):
            _flush_batch(batch)
            batch = []
            last = time.monotonic()


def _flush_batch(batch: list[str]) -> None:
    try:
        try:
            if os.path.getsize(_PATH) > _MAX_BYTES:
                os.replace(_PATH, _PATH + ".1")
        except OSError:
            pass
        with open(_PATH, "a", encoding="utf-8") as f:
            f.write("\n".join(batch) + "\n")
    except Exception:  # noqa: BLE001
        pass


def flush(timeout_s: float = 2.0) -> None:
    """Wait until the queue is written (end of process, tests)."""

    if _WRITER is None:
        return
    end = time.monotonic() + timeout_s
    while not _QUEUE.empty() and time.monotonic() < end:
        time.sleep(0.02)
    time.sleep(0.6)     # the last batch leaves at most 0.5 s after its last line


# -- command -----------------------------------------------------------------------------------------------

def _duration_s(text: str) -> float:
    units = {"s": 1, "min": 60, "m": 60, "h": 3600, "j": 86400, "d": 86400}
    for u in sorted(units, key=len, reverse=True):
        if text.endswith(u):
            return float(text[: -len(u)]) * units[u]
    return float(text)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tensorfold.pli.trace",
                                description="Switch of the PLI trace (engines started with PLI_TRACE_DIR).")
    p.add_argument("action", choices=["on", "off", "status", "etat"], metavar="{on,off,status}")   # etat: name before 3.7.0
    p.add_argument("--dir", dest="folder", default=_DIR, help="directory of PLI_TRACE_DIR (default: the variable)")
    p.add_argument("--dossier", dest="folder", default=argparse.SUPPRESS, help=argparse.SUPPRESS)   # name before 3.7.0
    p.add_argument("--duration", dest="duration", help="switch off automatically after this time (e.g. 30min, 24h, 2d)")
    p.add_argument("--duree", dest="duration", default=argparse.SUPPRESS, help=argparse.SUPPRESS)   # name before 3.7.0
    a = p.parse_args(argv)
    if not a.folder:
        p.error("--dir or PLI_TRACE_DIR required")
    switch = os.path.join(a.folder, SWITCH)
    if a.action == "on":
        os.makedirs(a.folder, exist_ok=True)
        content = f"since={time.time():.0f}"
        if a.duration:
            content += f" until={time.time() + _duration_s(a.duration):.0f}"
        with open(switch + ".tmp", "w", encoding="utf-8") as f:
            f.write(content + "\n")
        os.replace(switch + ".tmp", switch)
    elif a.action == "off":
        try:
            os.remove(switch)
        except FileNotFoundError:
            pass
    present = os.path.exists(switch)
    end = _deadline(switch) if present else None
    traces = sorted(n for n in os.listdir(a.folder) if n.startswith("trace-") and n.endswith(".jsonl")) \
        if os.path.isdir(a.folder) else []
    state = "on" if present else "off"
    if end is not None:
        state += f" until {time.strftime('%d/%m %H:%M', time.localtime(end))}"
    print(f"trace {state} (engines pick it up within 1 s); files: "
          + (", ".join(f"{n} ({os.path.getsize(os.path.join(a.folder, n)) // 1024} KiB)" for n in traces) or "none"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
