"""PLI (16/11/2026): background task ``prune``, the session snapshots pruned by PLI itself (PLI_TASKS=prune).

Takes over an external cron (``elagage-instantanes-pli``, 27/10/2026), with
the same rules: the engine spills evicted conversations to the session-snapshot folder (one .safetensors per conversation,
shared by the models) and bounds only the files of the model it serves (--spill-gib); the index (DiskBlocks) touches a
file when it reads it back, so its modification time is its last use. A deleted file is read again from the prompt the
next time (a cost in reading, never an error).

Rules: (1) a file unused for PLI_TASKS_PRUNE_DAYS days (7) is deleted, whatever its model; (2) past PLI_TASKS_PRUNE_CAP_GB
(60) for the whole folder, the least recently used go; (3) a .partial.safetensors older than a day goes; (4) nothing
modified in the last 10 minutes is touched (a write in progress). A pass every PLI_TASKS_PRUNE_EVERY_S seconds (86,400, once a day as the cron):
one step lists the folder, then each step deletes at most 32 files.
PLI_TASKS_PRUNE_CAP_PERCENT (16/11/2026): the cap follows the disk, PCT % of the space the folder may
use (the disk's free space plus the folder's own size), measured at each pass; with PLI_TASKS_PRUNE_CAP_GB also set, the
smaller of the two. The engine's own bound (--spill-gib, files of the model it serves) still applies on top. The folder is the server's session-snapshot folder
(--snapshot-dir's sibling ``session-snapshots``), or PLI_TASKS_PRUNE_DIR.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any

from pli.tasks import Task, _env_float

FRESH_S = 600.0
PARTIAL_S = 86400.0
PER_STEP = 32


def plan(files: list[tuple[float, int, Path]], now: float, days: float, cap_bytes: float) -> list[tuple[Path, int, str]]:
    """The files to delete, with the reason: ``files`` are (mtime, size, path); same order and rules as the cron script."""

    out: list[tuple[Path, int, str]] = []
    total = 0
    for mtime, size, path in sorted(files, key=lambda x: -x[0]):          # the most recently used first
        age = now - mtime
        if age < FRESH_S:
            total += size
            continue
        if path.name.endswith(".partial.safetensors"):
            if age > PARTIAL_S:
                out.append((path, size, "partial"))
            continue
        if age > days * 86400:
            out.append((path, size, "unused"))
            continue
        total += size
        if total > cap_bytes:
            out.append((path, size, "cap"))
    return out


class Prune(Task):
    name = "prune"

    def __init__(self, tasks: Any, context: dict[str, Any]) -> None:
        super().__init__(tasks, context)
        folder = os.environ.get("PLI_TASKS_PRUNE_DIR", "").strip() or context.get("session_dir") or ""
        if not folder:
            raise ValueError("no session-snapshot folder (--snapshot-dir or PLI_TASKS_PRUNE_DIR)")
        self.folder = Path(folder).expanduser()
        self.days = _env_float("PLI_TASKS_PRUNE_DAYS", 7.0)
        self.cap_bytes = _env_float("PLI_TASKS_PRUNE_CAP_GB", 60.0) * 2**30
        self.cap_percent = _env_float("PLI_TASKS_PRUNE_CAP_PERCENT", 0.0)
        self.cap_gb_set = bool(os.environ.get("PLI_TASKS_PRUNE_CAP_GB", "").strip())
        self.every_s = _env_float("PLI_TASKS_PRUNE_EVERY_S", 86400.0)   # once a day, as the cron it replaces
        self.pending: list[tuple[Path, int, str]] = []
        self.pass_stats: dict[str, Any] = {}

    def due(self, now: float) -> bool:
        return bool(self.pending) or now - float(self.state.get("last_pass", 0) or 0) >= self.every_s

    def step(self, engine: Any) -> None:
        if not self.pending:
            self._list()
            return
        batch, self.pending = self.pending[:PER_STEP], self.pending[PER_STEP:]
        for path, size, reason in batch:
            try:
                st = path.stat()
                if time.time() - st.st_mtime < FRESH_S:          # read back since the listing: kept
                    self.pass_stats["kept_fresh"] = self.pass_stats.get("kept_fresh", 0) + 1
                    continue
                path.unlink()
            except FileNotFoundError:
                continue
            except OSError as exc:
                self.pass_stats["failed"] = self.pass_stats.get("failed", 0) + 1
                self.log("delete_failed", file=path.name, error=str(exc))
                continue
            key = f"deleted_{reason}"
            self.pass_stats[key] = self.pass_stats.get(key, 0) + 1
            self.pass_stats["freed_bytes"] = self.pass_stats.get("freed_bytes", 0) + int(size)
        if not self.pending:
            self._close()

    def _list(self) -> None:
        now = time.time()
        files: list[tuple[float, int, Path]] = []
        if self.folder.is_dir():
            for path in self.folder.glob("*.safetensors"):
                try:
                    st = path.stat()
                except OSError:
                    continue
                files.append((st.st_mtime, st.st_size, path))
        size = sum(s for _, s, _ in files)
        cap = self.cap_bytes
        if self.cap_percent > 0:
            try:
                st = os.statvfs(self.folder)
                room = st.f_bavail * st.f_frsize + size
                cap = room * self.cap_percent / 100 if not self.cap_gb_set else min(cap, room * self.cap_percent / 100)
            except OSError:
                pass                                   # no reading of the disk: the fixed cap
        self.pending = plan(files, now, self.days, cap)
        self.pass_stats = {"files": len(files), "bytes": size, "cap_bytes": int(cap), "to_delete": len(self.pending)}
        if not self.pending:
            self._close()

    def _close(self) -> None:
        self.state["last_pass"] = time.time()
        self.state["last"] = dict(self.pass_stats)
        self.save()
        self.log("pass", folder=str(self.folder), **self.pass_stats)

    def summary(self) -> dict[str, Any]:
        return {"folder": str(self.folder), "last_pass": self.state.get("last_pass"), "last": self.state.get("last", {}),
                "pending": len(self.pending)}
