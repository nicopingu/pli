"""PLI (10/11/2026): evicted conversations written to disk by a thread of their own.

Before: ``spill_conversation`` wrote the file on the engine thread. During a write no round was made, every request
waited, not only the one whose checkpoint was evicted (production journal 05/11 -> 10/11: engine stopped by the disk
777 s a day, 723 s of them spill writes; 343 s a day on 04/11).

Now the engine thread only prepares: the KVD5 reduction, the layer export, contiguous and evaluated arrays and a view of
their bytes (``prepare``). A writer thread writes the safetensors file from those bytes, without calling MLX, renames
the partial file and prunes this model's files by bytes. The engine waits for a write only:
- when a prompt could resume a conversation still being written (``settle``, before the disk is searched);
- when the bytes waiting to be written would pass ``PLI_SPILL_BACKGROUND_MAX_GIB`` (they stay in memory until written);
- at shutdown (``drain``), before the conversations are saved.
Reading a file and pruning share one lock: the pruning never removes a file being read.

PLI_SPILL_BACKGROUND=1: on. 0 (default): off, previous behaviour (written on the engine thread).
Text unchanged: the file holds the same arrays (same names, dtypes, shapes, bytes, metadata) as ``mx.save_safetensors``.
"""

from __future__ import annotations

import json
import os
import queue
import struct
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import mlx.core as mx

from tensorfold.pli import events as _ev

ENABLED = os.environ.get("PLI_SPILL_BACKGROUND", "0") == "1"
MAX_PENDING_BYTES = int(float(os.environ.get("PLI_SPILL_BACKGROUND_MAX_GIB", "8") or 8) * 1024**3)

_DTYPES = {
    mx.float32: "F32", mx.bfloat16: "BF16", mx.float16: "F16", mx.int64: "I64", mx.int32: "I32", mx.int16: "I16",
    mx.int8: "I8", mx.uint64: "U64", mx.uint32: "U32", mx.uint16: "U16", mx.uint8: "U8", mx.bool_: "BOOL",
    mx.complex64: "C64",
}


@dataclass(eq=False)
class Job:
    """One prepared spill: the file, its header and the bytes of its arrays (kept alive by ``arrays``)."""

    target: Path
    tokens: list[int]
    nbytes: int                       # the checkpoint's size in memory (the store's count, as the journal gives it)
    header: bytes
    views: list[memoryview]
    arrays: dict[str, Any]
    file_bytes: int
    engine_seconds: float = 0.0
    done: threading.Event = field(default_factory=threading.Event)
    ok: bool = False


def header_and_views(arrays: dict[str, Any], meta: dict[str, str]) -> tuple[bytes, list[memoryview], int]:
    """The safetensors header and the byte views of ``arrays`` (contiguous, evaluated); raises on an unknown dtype."""

    parent: dict[str, Any] = {}
    views: list[memoryview] = []
    offset = 0
    for name, array in arrays.items():
        dtype = _DTYPES.get(array.dtype)
        if dtype is None:
            raise TypeError(f"no safetensors dtype for {array.dtype}")
        view = memoryview(array).cast("B") if array.nbytes else memoryview(b"")
        if view.nbytes != array.nbytes:
            raise ValueError(f"{name}: {view.nbytes} bytes in view, {array.nbytes} in the array")
        parent[name] = {"dtype": dtype, "shape": list(array.shape), "data_offsets": [offset, offset + array.nbytes]}
        views.append(view)
        offset += array.nbytes
    parent["__metadata__"] = dict(meta)
    header = json.dumps(parent, separators=(",", ":"), ensure_ascii=True).encode()
    return struct.pack("<Q", len(header)) + header, views, offset


def write_file(target: Path, header: bytes, views: Sequence[memoryview]) -> None:
    """Write the file as a partial of this process, then rename it (no MLX call: any thread)."""

    from tensorfold.engine.prefix_snapshots import partial_path

    partial = partial_path(target)
    try:
        with open(partial, "wb") as handle:
            handle.write(header)
            for view in views:
                if view.nbytes:
                    handle.write(view)
        partial.rename(target)
    except BaseException:
        partial.unlink(missing_ok=True)   # a full disk would otherwise keep these bytes, outside every byte budget
        raise


def prepare(entry: Any, directory: Path, model_id: str) -> Job | None:
    """On the engine thread: everything that touches MLX. None when the file is already there (touched)."""

    from tensorfold.engine.prefix_snapshots import prepare_snapshot
    from tensorfold.pli import checkpoint_bits as _cb

    prepared = prepare_snapshot(directory, model_id, entry.tokens, _cb.for_disk(entry.cache))
    if prepared is None:
        return None
    target, arrays, meta = prepared
    arrays = {name: mx.contiguous(array) for name, array in arrays.items()}
    mx.eval(list(arrays.values()))
    header, views, size = header_and_views(arrays, meta)
    return Job(target=target, tokens=list(entry.tokens), nbytes=int(entry.nbytes), header=header, views=views,
               arrays=arrays, file_bytes=len(header) + size)


class SpillWriter:
    """The writer thread of evicted conversations, with the list of writes in progress."""

    def __init__(self, directory: Path, model_id: str, *, limit_bytes: int,
                 max_pending_bytes: int = MAX_PENDING_BYTES) -> None:
        self.directory = Path(directory)
        self.model_id = model_id
        self.limit_bytes = int(limit_bytes)
        self.max_pending_bytes = int(max_pending_bytes)
        self.disk_lock = threading.Lock()          # file reads and pruning
        self._cv = threading.Condition()
        self._pending: list[Job] = []
        self._queue: "queue.Queue[Job | None]" = queue.Queue()
        self.written = 0
        self.failed = 0
        self.waited_seconds = 0.0
        self._closed = False                      # after close: written on the calling thread, as before
        self._thread = threading.Thread(target=self._run, name="pli-spill-writer", daemon=True)
        self._thread.start()

    # --- engine thread -------------------------------------------------------------------------------------------
    def submit(self, entry: Any) -> bool:
        """The store's ``on_evict``: prepare on this thread, write on the writer's. False: nothing goes to disk."""

        if entry.nbytes > self.limit_bytes:
            return False
        started = time.perf_counter()
        try:
            job = prepare(entry, self.directory, self.model_id)
        except Exception as exc:  # noqa: BLE001 - a failed spill costs a later prefill, never the request
            print(f"[tensorfold] conversation spill failed: {type(exc).__name__}: {exc}", flush=True)
            return False
        if job is None:
            return True
        if self._closed:
            return self._write(job, started)
        with self._cv:
            waited = time.perf_counter()
            while self._pending and self.pending_bytes() + job.file_bytes > self.max_pending_bytes:
                self._cv.wait()                    # bytes held in memory until written: bounded
            self.waited_seconds += time.perf_counter() - waited
            self._pending.append(job)
        job.engine_seconds = time.perf_counter() - started
        self._queue.put(job)
        return True

    def pending_bytes(self) -> int:
        return sum(job.file_bytes for job in self._pending)

    def settle(self, prompt: Sequence[int]) -> float:
        """Wait for the writes a prompt could resume (strict prefixes of it); return the seconds waited."""

        with self._cv:
            waits = [job for job in self._pending
                     if len(job.tokens) < len(prompt) and list(prompt[:len(job.tokens)]) == job.tokens]
        if not waits:
            return 0.0
        started = time.perf_counter()
        for job in waits:
            job.done.wait()
        seconds = time.perf_counter() - started
        self.waited_seconds += seconds
        _ev.ev("spill_wait", conv=_ev.conv_of_tokens(prompt), tokens=max(len(job.tokens) for job in waits),
               duration=seconds)
        return seconds

    def drain(self, timeout: float | None = None) -> bool:
        """Wait until every write is done (shutdown); False on timeout."""

        deadline = None if timeout is None else time.monotonic() + timeout
        with self._cv:
            while self._pending:
                left = None if deadline is None else deadline - time.monotonic()
                if left is not None and left <= 0:
                    return False
                self._cv.wait(left)
        return True

    def close(self, timeout: float | None = 120.0) -> bool:
        done = self.drain(timeout)
        self._closed = True
        self._queue.put(None)
        self._thread.join(timeout=5.0)
        return done

    # --- writer thread -------------------------------------------------------------------------------------------
    def _run(self) -> None:
        while True:
            job = self._queue.get()
            if job is None:
                return
            self._write(job, time.perf_counter())

    def _write(self, job: Job, started: float) -> bool:
        from tensorfold.server.checkpoints import prune_conversations

        background = threading.current_thread() is self._thread
        try:
            write_file(job.target, job.header, job.views)
            written = time.perf_counter() - started
            with self.disk_lock:
                prune_conversations(self.directory, self.model_id, self.limit_bytes)
            job.ok = True
            self.written += 1
            _ev.ev("spill", conv=_ev.conv_of_tokens(job.tokens), tokens=len(job.tokens),
                   gib=job.nbytes / 1024**3, duration=time.perf_counter() - started, write=written,
                   engine=job.engine_seconds if background else time.perf_counter() - started, background=background)
            print(f"[tensorfold] spilled conversation tokens={len(job.tokens)} ({job.nbytes / 1024**3:.1f} GiB) "
                  f"in {written:.2f}s {'on the writer thread' if background else 'after the writer closed'} "
                  f"(engine {job.engine_seconds:.2f}s)", flush=True)
        except Exception as exc:  # noqa: BLE001 - a full disk costs a later prefill, never the request
            self.failed += 1
            print(f"[tensorfold] conversation spill failed: {type(exc).__name__}: {exc}", flush=True)
        finally:
            job.views = []
            job.arrays = {}
            with self._cv:
                if job in self._pending:
                    self._pending.remove(job)
                self._cv.notify_all()
            job.done.set()
        return job.ok

__all__ = ["ENABLED", "Job", "MAX_PENDING_BYTES", "SpillWriter", "header_and_views", "prepare", "write_file"]
