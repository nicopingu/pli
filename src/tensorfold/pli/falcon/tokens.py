"""PLI (11/11/2026): FALCON B, the tokens of each request kept on disk for a few days.

The DFlash2 draft reads states of the 27B (layers 5, 19, 33, 47, 61). Storing them during service would write about
23 GB a day (option A). Here only the tokens are kept: a night job replays them in the 27B, recomputes those states and
adjusts the draft on them. The prompt is needed, not only the reply: the states at a position depend on the whole prefix.

Nothing changes in the computation: the tokens are already on the host when the request ends, a separate thread
writes them, a full queue drops the record (counted) instead of waiting.

PLI_FALCON_TOKENS=<dir> switches it on (off by default). One file per request, <dir>/jetons-<time>-<pid>-<n>.npz:
prompt (int32), output (int32), meta (JSON: t, model, conv, request, cached, reason, vision). Files older than
PLI_FALCON_TOKENS_DAYS days (7) are deleted, checked at most every 10 minutes. The directory is made with mode 700:
prompts hold mail, invoices and client text; they stay on this machine.
"""

from __future__ import annotations

import json
import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Sequence

FOLDER = os.environ.get("PLI_FALCON_TOKENS", "").strip()
DAYS = float(os.environ.get("PLI_FALCON_TOKENS_DAYS", "7") or 7)
PURGE_EVERY = 600.0
count = {"records": 0, "files": 0, "tokens": 0, "lost": 0, "errors": 0, "purged": 0}
_said: set[str] = set()
_queue: "queue.Queue | None" = None
_lock = threading.Lock()
_last_purge = [0.0]
_n = [0]


def _say(key: str, text: str) -> None:
    if key not in _said:
        _said.add(key)
        import sys
        print(f"[PLI] falcon tokens: {text}", file=sys.stderr, flush=True)


def active() -> bool:
    return bool(FOLDER)


def purge(folder: Path, days: float, now: float | None = None) -> int:
    """Delete the files older than ``days`` days; return how many."""

    now = time.time() if now is None else now
    gone = 0
    for f in folder.glob("jetons-*.npz"):
        try:
            if now - f.stat().st_mtime > days * 86400:
                f.unlink(missing_ok=True)
                gone += 1
        except OSError:
            pass
    count["purged"] += gone
    return gone


def _folder() -> Path:
    d = Path(FOLDER)
    if not d.exists():
        d.mkdir(parents=True, exist_ok=True)
    try:
        os.chmod(d, 0o700)
    except OSError:
        pass
    return d


def write(prompt: Any, output: Any, meta: dict) -> Path:
    """One record to disk (temporary name, then rename: a reader never sees half a file)."""

    import numpy as np

    d = _folder()
    _n[0] += 1
    name = d / f"jetons-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{_n[0]:05d}.npz"
    tmp = name.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, prompt=np.asarray(prompt, dtype=np.int32), output=np.asarray(output, dtype=np.int32),
                        meta=np.array(json.dumps(meta, ensure_ascii=False)))
    os.chmod(tmp, 0o600)
    tmp.rename(name)
    count["files"] += 1
    if time.time() - _last_purge[0] > PURGE_EVERY:
        _last_purge[0] = time.time()
        purge(d, DAYS)
    return name


def _writer() -> None:
    while True:
        item = _queue.get()
        if item is None:
            return
        try:
            name = write(*item)
            _say("file", f"first file written: {name} (kept {DAYS:g} days)")
        except Exception as e:
            count["errors"] += 1
            _say("write", f"write failed: {type(e).__name__}: {e}")


def record(prompt: Sequence[int], output: Sequence[int], **meta: Any) -> None:
    """A finished request: its prompt tokens and its reply tokens, with ``meta`` (conv, request, cached...)."""

    global _queue
    if not FOLDER:
        return
    try:
        with _lock:
            if _queue is None:
                _queue = queue.Queue(maxsize=64)
                threading.Thread(target=_writer, daemon=True, name="pli-falcon-tokens").start()
        meta = {"t": round(time.time(), 3), **meta}
        _queue.put_nowait((list(prompt), list(output), meta))
        count["records"] += 1
        count["tokens"] += len(prompt) + len(output)
    except queue.Full:
        count["lost"] += 1
        _say("full", "queue full, a record dropped")
    except Exception as e:
        count["errors"] += 1
        _say("record", f"record failed: {type(e).__name__}: {e}")


def read(path: Path) -> tuple[Any, Any, dict]:
    """(prompt, output, meta) of one file."""

    import numpy as np

    with np.load(path) as z:
        return z["prompt"], z["output"], json.loads(str(z["meta"]))


__all__ = ["active", "record", "read", "purge", "count"]
