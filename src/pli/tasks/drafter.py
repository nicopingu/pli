"""PLI (16/11/2026): background task ``drafter``, the DFlash2 drafter changed between two requests and judged in service.

Decision (15/11/2026): the adjusted drafter of FALCON B is put in service without a
separate bench and without restarting the server, then judged by watching the service itself over several days, "without
a hasty conclusion". The text never changes: the 27B checks every draft; only the speed does.

PLI_TASKS=drafter switches it on. PLI_TASKS_DRAFTER_FILE=<file>: one line, the folder of the drafter to serve (written by
FALCON B when a night's drafter is better). When the file names another folder than the one served, and the served one
has been watched for PLI_TASKS_DRAFTER_MIN_DAYS days (3; no wait at the very first change), the engine thread loads it in
a quiet moment, in place of the served one (same object, so every stream and checkpoint keeps its reference; loading
takes a few seconds, once). A folder of another shape (layers read, block, mask, window) is refused.

After a restart on the configured drafter (--drafter), the drafter the state says was served comes back at the first
quiet moment when the file still names it, and its watch goes on where it was.

Watching: every finished request adds, for the served drafter and the request's size (prompt tokens: under 4K, 4K to
16K, 16K to 64K, 64K and more), its generated tokens and the tokens that came from drafts. The share of generated tokens
that came from drafts is the measure (the more, the fewer rounds a reply takes).

The guard: once the new drafter has been served PLI_TASKS_DRAFTER_MIN_DAYS days, it is compared with the previous one,
size by size, on the sizes where both have at least PLI_TASKS_DRAFTER_MIN_REQUESTS requests (50), weighted by the new
one's requests. Below the previous one by more than PLI_TASKS_DRAFTER_DROP (0.03, relative), the previous drafter comes
back and the new folder is marked refused (the file naming it again is ignored). Otherwise the new one is kept and
becomes the reference of the next change. A summary line per day goes to the journal.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Any

from pli.tasks import Task, _env_float

BUCKETS = ((4096, "lt4k"), (16384, "4k-16k"), (65536, "16k-64k"), (1 << 62, "ge64k"))
DAY_S = 86400.0


def bucket(prompt: int) -> str:
    for limit, name in BUCKETS:
        if prompt < limit:
            return name
    return BUCKETS[-1][1]


def share(stats: dict[str, list[float]], name: str) -> tuple[float, int]:
    """(drafted share of the generated tokens, requests) of one size."""

    n, generated, accepted = (list(stats.get(name) or []) + [0, 0, 0])[:3]
    return (float(accepted) / float(generated) if generated else 0.0), int(n)


def compare(new: dict[str, list[float]], old: dict[str, list[float]], min_requests: int) -> dict[str, Any]:
    """The new drafter's share against the old one's, on the sizes both measured enough, weighted by the new one's."""

    sizes, weight, new_sum, old_sum = [], 0, 0.0, 0.0
    for _, name in BUCKETS:
        s_new, n_new = share(new, name)
        s_old, n_old = share(old, name)
        if n_new >= min_requests and n_old >= min_requests:
            sizes.append({"size": name, "new": round(s_new, 4), "old": round(s_old, 4), "n_new": n_new, "n_old": n_old})
            weight += n_new
            new_sum += n_new * s_new
            old_sum += n_new * s_old
    if not weight or old_sum <= 0:
        return {"sizes": sizes, "ratio": None}
    return {"sizes": sizes, "ratio": round(new_sum / old_sum, 4)}


def _resolved(path: Any) -> str:
    return str(Path(str(path)).expanduser().resolve())


class DrafterSwap(Task):
    name = "drafter"

    def __init__(self, tasks: Any, context: dict[str, Any]) -> None:
        super().__init__(tasks, context)
        self.drafter = context.get("drafter")
        self.reload = context.get("reload_drafter")
        if self.drafter is None or self.reload is None:
            raise ValueError("no DFlash2 drafter in this server (--drafter)")
        self._lock = threading.RLock()
        self.file = os.environ.get("PLI_TASKS_DRAFTER_FILE", "").strip()
        self.min_days = _env_float("PLI_TASKS_DRAFTER_MIN_DAYS", 3.0)
        self.min_requests = int(_env_float("PLI_TASKS_DRAFTER_MIN_REQUESTS", 50))
        self.drop = _env_float("PLI_TASKS_DRAFTER_DROP", 0.03)
        self._wanted_cache: tuple[float, str] = (-1.0, "")
        now = time.time()
        st = self.state
        for key, value in (("stats", {}), ("history", {}), ("refused", []), ("previous", None), ("judged", True),
                           ("day_stats", {})):
            st.setdefault(key, value)
        st.setdefault("day", time.strftime("%Y-%m-%d", time.localtime(now)))
        configured = _resolved(self.drafter.path)
        self.restore = ""
        if not st.get("served"):
            st["served"], st["since"], st["first"] = configured, now, True
        elif st["served"] != configured:
            if self.wanted() == st["served"] and st["served"] not in st["refused"]:
                self.restore = st["served"]          # brought back at the first quiet moment, its watch goes on
            else:
                self._archive(st["served"])
                st.update(served=configured, since=now, stats={}, previous=None, judged=True)
        self.save()
        self.log("serving", drafter=configured, restore=self.restore or None)

    def save(self) -> None:
        with self._lock:
            super().save()

    # -- what is wanted -------------------------------------------------------------------------------------------
    def wanted(self) -> str:
        if not self.file:
            return ""
        try:
            mtime = os.stat(self.file).st_mtime
        except OSError:
            return ""
        if mtime != self._wanted_cache[0]:
            try:
                line = Path(self.file).read_text(encoding="utf-8").strip().splitlines()[0].strip()
                path = _resolved(line) if line else ""
            except (OSError, IndexError):
                path = ""
            self._wanted_cache = (mtime, path)
        return self._wanted_cache[1]

    def _served_days(self, now: float) -> float:
        return (now - float(self.state.get("since", now) or now)) / DAY_S

    def _change_due(self, now: float) -> str:
        path = self.wanted()
        st = self.state
        if self.restore:
            return self.restore if path == self.restore else ""
        if not path or path == st["served"] or path in st["refused"]:
            return ""
        if not st.get("first") and self._served_days(now) < self.min_days:
            return ""                                        # the served one is still being watched
        if not (Path(path) / "config.json").is_file():
            return ""
        return path

    def _guard_due(self, now: float) -> bool:
        st = self.state
        return not self.restore and bool(st.get("previous")) and not st.get("judged") \
            and self._served_days(now) >= self.min_days

    def _new_day(self, now: float) -> bool:
        return self.state.get("day") != time.strftime("%Y-%m-%d", time.localtime(now))

    def due(self, now: float) -> bool:
        if self.restore and self.wanted() != self.restore:
            self._give_up_restore(now)
        return self._new_day(now) or bool(self._change_due(now)) or self._guard_due(now)

    def need_bytes(self) -> int:
        return 6 * 2**30 if self._change_due(time.time()) else 0      # a bf16 drafter read (3.6 GiB) before it is quantized

    # -- the engine thread ----------------------------------------------------------------------------------------
    def step(self, engine: Any) -> None:
        now = time.time()
        if self._new_day(now):
            self._close_day(now)
            return
        if self._guard_due(now):
            self._judge(now)
            return
        path = self._change_due(now)
        if path:
            self._swap(path, now, reason="restore" if path == self.restore else "new")

    def _swap(self, path: str, now: float, *, reason: str) -> None:
        st = self.state
        old_path = st["served"] if reason != "restore" else _resolved(self.drafter.path)
        started = time.perf_counter()
        try:
            self.reload(self.drafter, path)
        except Exception as exc:  # noqa: BLE001 - the served drafter stays
            with self._lock:
                st["refused"].append(path)
                if reason == "restore":
                    self.restore = ""
                    self._archive(st["served"])
                    st.update(served=old_path, since=now, stats={}, previous=None, judged=True)
            self.save()
            self.log("refused", drafter=path, error=f"{type(exc).__name__}: {exc}")
            return
        with self._lock:
            if reason == "restore":
                self.restore = ""                    # same watch, same stats, same previous
            else:
                self._archive(old_path)
                st.update(served=path, since=now, stats={}, first=False,
                          previous=old_path if reason == "new" else None, judged=reason != "new")
        self.save()
        self.log("swap", reason=reason, drafter=path, previous=old_path, seconds=round(time.perf_counter() - started, 2))
        print(f"[PLI] drafter changed between two requests ({reason}): {path}", flush=True)

    def _give_up_restore(self, now: float) -> None:
        with self._lock:
            st = self.state
            self._archive(st["served"])
            st.update(served=_resolved(self.drafter.path), since=now, stats={}, previous=None, judged=True)
            self.restore = ""
        self.save()

    def _archive(self, path: str | None) -> None:
        if not path:
            return
        with self._lock:
            st = self.state
            past = st["history"].setdefault(path, {})
            for size, row in (st.get("stats") or {}).items():
                acc = past.setdefault(size, [0, 0, 0, 0])
                for i, v in enumerate(list(row)[:4]):
                    acc[i] += v
            # only the served, the previous and this one are worth keeping
            keep = {st.get("served"), st.get("previous"), path}
            for other in list(st["history"]):
                if other not in keep:
                    del st["history"][other]

    def _judge(self, now: float) -> None:
        with self._lock:
            st = self.state
            result = compare(dict(st.get("stats") or {}), dict(st["history"].get(st["previous"], {})),
                             self.min_requests)
            st["judged"] = True
        ratio = result["ratio"]
        if ratio is not None and ratio < 1.0 - self.drop:
            refused = st["served"]
            with self._lock:
                st["refused"].append(refused)
            self.log("guard", verdict="back", ratio=ratio, sizes=result["sizes"], drafter=refused,
                     previous=st["previous"])
            self._swap(st["previous"], now, reason="guard")
            return
        self.save()
        self.log("guard", verdict="kept" if ratio is not None else "not_enough", ratio=ratio, sizes=result["sizes"],
                 drafter=st["served"], previous=st["previous"])

    def _close_day(self, now: float) -> None:
        with self._lock:
            st = self.state
            sizes = {name: {"requests": int(r[0]), "share": round(r[2] / r[1], 4) if r[1] else None}
                     for name, r in (st.get("day_stats") or {}).items()}
            day = st.get("day")
            st["day"], st["day_stats"] = time.strftime("%Y-%m-%d", time.localtime(now)), {}
        self.save()
        self.log("day", day=day, drafter=self.state["served"], sizes=sizes)

    # -- request threads ------------------------------------------------------------------------------------------
    def observe(self, *, prompt: int = 0, generation: int = 0, drafted: int = 0, accepted: int = 0,
                **_: Any) -> None:
        if generation <= 0 or self.restore:
            return
        name = bucket(int(prompt))
        with self._lock:
            for key in ("stats", "day_stats"):
                row = self.state.setdefault(key, {}).setdefault(name, [0, 0, 0, 0])
                row[0] += 1
                row[1] += int(generation)
                row[2] += int(accepted)
                row[3] += int(drafted)
            self._unsaved = getattr(self, "_unsaved", 0) + 1
            flush = self._unsaved >= 20
            if flush:
                self._unsaved = 0
        if flush:
            self.save()

    def summary(self) -> dict[str, Any]:
        with self._lock:
            st = self.state
            return {"served": st.get("served"), "since": st.get("since"), "previous": st.get("previous"),
                    "judged": st.get("judged"), "refused": list(st.get("refused") or []), "restore": self.restore or None,
                    "share": {name: round(share(st.get("stats") or {}, name)[0], 4) for _, name in BUCKETS}}
