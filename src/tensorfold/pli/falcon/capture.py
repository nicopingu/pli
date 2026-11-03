"""PLI (24/10/2026, FALCON R0): capture of the draft's training examples during service.

At each verification round, for the kept rows of a request: the row's final state (input of the MTP head),
the token the model chose there, and its K best logits (the distribution to imitate). Nothing changes in the computation:
the arrays are read one round later, when the GPU has already returned them, then written to disk by a separate thread.

PLI_FALCON=<dir> switches the capture on (off by default). Files <dir>/tampon-<time>-<n>.npz (the on-disk names below are
kept from the French version: the training tools read them): flux (int64, number of the request in this process), pos
(position of the row), h (uint16: bits of the bf16, [n, D]), suiv (int32: token chosen by the model at this row), ind (int32 [n, K])
and val (float16 [n, K]): the K best logits (ind = -1: no target this round, see PLI_FALCON_TARGETS_EVERY). A row (pos p, next token t) and the next one of the same stream (p + 1) make an example for the MTP head:
input (h_p, t), target: logits of row p + 1. PLI_FALCON_K (20), PLI_FALCON_BATCH (4096 rows per file),
PLI_FALCON_MAX_GB (2: beyond, the oldest files are deleted).
Added on the evening of 24/10 (prompt context): lu (uint8, 1 = prompt row read by the head, without target except the last one),
and the links of a request to the previous one of its conversation: lien_flux, lien_pos (rows reused from the cache) and
lien_jetons (int32 [n, 32]: the 32 prompt tokens before lien_pos, -1 if fewer); training finds the previous request
(same tokens at the same positions) and takes from it the context the head had in service. PLI_FALCON_PROMPT (2,048):
rows of the end of the prompt kept. The written tokens and those of the end of the prompt are kept (not the rest of the text).
"""

from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path
from typing import Any, Sequence

FOLDER = os.environ.get("PLI_FALCON", "").strip()
K = int(os.environ.get("PLI_FALCON_K", "20") or 20)
BATCH = int(os.environ.get("PLI_FALCON_BATCH", "4096") or 4096)
MAX_GB = float(os.environ.get("PLI_FALCON_MAX_GB", "2") or 2)
# targets (the K best logits) computed one round in TARGETS_EVERY only; the other rounds keep the states (context
# of the head at training) with ind = -1. Sorting the 248,320 logits is the cost of the capture (+ 2.8 % per round).
TARGETS_EVERY = max(1, int(os.environ.get("PLI_FALCON_TARGETS_EVERY", "2") or 2))
PROMPT = int(os.environ.get("PLI_FALCON_PROMPT", "2048") or 2048)
# PLI (27/10/2026, space-precision balance): prompt rows kept in 8 bits per row (scale = max |x| / 127) instead of 16:
# half the space (82 % of the captured rows are context); measured gate (pli-falcon-c8c): 3.529 against 3.513. 0: 16 bits.
CONTEXT8 = os.environ.get("PLI_FALCON_CONTEXT8", "1") != "0"
ANCHOR = 32
_rounds: dict[str, int] = {}
count = {"calls": 0, "rows": 0, "files": 0, "lost": 0, "errors": 0, "flushes": 0}
_said: set[str] = set()


def _say(key: str, text: str) -> None:
    """One line in the engine log, once per kind (first capture, first error of each place)."""

    if key not in _said:
        _said.add(key)
        import sys
        print(f"[PLI] falcon: {text}", file=sys.stderr, flush=True)

_pending: list[tuple] = []          # rounds captured at the previous round, read at the next (already computed by the GPU)
_numbers: dict[str, int] = {}
_queue: "queue.Queue | None" = None
_flushed = threading.Event()
_stream: Any = None
_last = [0.0]


def _at_exit() -> None:
    """At engine exit: the last rounds written (5 s at most)."""

    if _queue is None:
        return
    try:
        if _pending:
            _read_pending()
        _flushed.clear()
        _queue.put(None, timeout=1.0)
        _flushed.wait(5.0)
    except Exception:
        pass


def active() -> bool:
    return bool(FOLDER)


def _writer() -> None:
    import numpy as np

    batch: list[tuple] = []
    batch8: list[tuple] = []
    links: list[tuple] = []
    n = 0

    def flush() -> None:
        nonlocal batch, batch8, links, n
        count["flushes"] += 1
        if not batch and not batch8 and not links:
            return
        try:
            d = Path(FOLDER)
            d.mkdir(parents=True, exist_ok=True)
            n += 1
            name = d / f"tampon-{time.strftime('%Y%m%d-%H%M%S')}-{os.getpid()}-{n:05d}.npz"
            arrays = {}
            if batch:
                arrays = dict(flux=np.concatenate([x[0] for x in batch]), pos=np.concatenate([x[1] for x in batch]),
                              h=np.concatenate([x[2] for x in batch]), suiv=np.concatenate([x[3] for x in batch]),
                              ind=np.concatenate([x[4] for x in batch]), val=np.concatenate([x[5] for x in batch]),
                              lu=np.concatenate([x[6] for x in batch]))
            if batch8:                               # prompt rows in 8 bits: h8 (int8 [n, D]) and e8 (float32 scale [n])
                arrays.update(flux8=np.concatenate([x[0] for x in batch8]), pos8=np.concatenate([x[1] for x in batch8]),
                              h8=np.concatenate([x[2] for x in batch8]), suiv8=np.concatenate([x[3] for x in batch8]),
                              ind8=np.concatenate([x[4] for x in batch8]), val8=np.concatenate([x[5] for x in batch8]),
                              lu8=np.concatenate([x[6] for x in batch8]), e8=np.concatenate([x[7] for x in batch8]))
            if links:
                arrays.update(lien_flux=np.array([x[0] for x in links], dtype=np.int64),
                              lien_pos=np.array([x[1] for x in links], dtype=np.int64),
                              lien_jetons=np.stack([x[2] for x in links]).astype(np.int32))
            np.savez(name.with_suffix(".tmp.npz"), **arrays)
            name.with_suffix(".tmp.npz").rename(name)
            count["files"] += 1
            _say("file", f"first file written: {name} ({count})")
            files = sorted(d.glob("tampon-*.npz"), key=lambda f: f.stat().st_mtime)
            total = sum(f.stat().st_size for f in files)
            while files and total > MAX_GB * 2**30:
                oldest = files.pop(0)
                total -= oldest.stat().st_size
                oldest.unlink(missing_ok=True)
        except Exception as e:
            count["errors"] += 1
            _say("write", f"write failed: {type(e).__name__}: {e}")
        batch = []
        batch8 = []
        links = []

    while True:
        try:
            item = _queue.get(timeout=30.0)
        except queue.Empty:
            flush()                                  # a pause in the traffic: the current batch is written
            continue
        if item is None:                             # flush request (the engine, every 20 s, or the exit)
            flush()
            _flushed.set()
            continue
        if item[0] == "link":
            links.append(item[1:])
            continue
        (batch8 if item[0] == "rows8" else batch).append(item[1:])
        if sum(len(x[1]) for x in batch) + sum(len(x[1]) for x in batch8) >= BATCH:
            flush()


def _read_pending() -> None:
    """The previous round's rounds in numpy (the GPU returned them), to the writing thread."""

    import mlx.core as mx
    import numpy as np

    global _pending
    ready, _pending = _pending, []
    for flux, pos, h, ind, val, suiv, lu, *rest in ready:
        try:
            if flux == "link":                       # ("link", number, rows reused, anchor tokens)
                _queue.put_nowait(("link", pos, h, ind))
                continue
            suiv = np.array(suiv).astype(np.int32).reshape(-1)
            item = ("rows", np.full(len(suiv), flux, dtype=np.int64), np.arange(pos, pos + len(suiv), dtype=np.int64),
                    np.array(h), suiv, np.array(ind), np.array(val), np.full(len(suiv), lu, dtype=np.uint8))
            if rest and rest[0] is not None:          # 8 bits: h is the int8 array, rest[0] the scales
                item = ("rows8",) + item[1:] + (np.array(rest[0]).astype(np.float32).reshape(-1),)
            _queue.put_nowait(item)
            count["rows"] += len(suiv)
        except queue.Full:
            count["lost"] += len(suiv) if suiv is not None else 0
        except Exception as e:
            count["errors"] += 1
            _say("read", f"read failed: {type(e).__name__}: {e}")


_seen: dict[str, int] = {}


def seen(place: str, kind: str, cut: Any, prefix: bool) -> None:
    """Diagnostic: each concluded round that passes the capture (place, kind, cut, prefix path)."""

    if not FOLDER:
        return
    key = f"{place}:{kind}:{'cut' if cut is not None else '-'}:{'prefix' if prefix else 'tree'}"
    _seen[key] = _seen.get(key, 0) + 1
    total = sum(_seen.values())
    if total in (1, 10, 50, 200, 1000):
        _say(f"seen{total}", f"{total} concluded rounds seen: {_seen}")


def capture(flux: str, position: int, hidden: Any, logits: Any, rows: Sequence[int], nexts: Sequence[int]) -> None:
    """Kept rows ``rows`` of a round (in ``hidden`` [R, D] and ``logits`` [R, V]); row i at position + i."""

    if not FOLDER or not rows:
        return
    count["calls"] += 1
    if count["calls"] in (1, 10, 100, 1000, 10000):
        _say(f"calls{count['calls']}", f"{count['calls']} rounds captured ({count})")
    try:
        import mlx.core as mx

        with mx.stream(_prepare()):
            _capture_on_stream(flux, position, hidden, logits, rows, nexts)
    except Exception as e:
        count["errors"] += 1
        _say("capture", f"capture failed: {type(e).__name__}: {e}")


def _prepare() -> Any:
    """Writing thread started, earlier rounds read, flush every 20 s; return the capture's GPU stream."""

    global _queue, _stream
    import mlx.core as mx

    if True:
        if _queue is None:
            import atexit

            _queue = queue.Queue(maxsize=256)
            threading.Thread(target=_writer, daemon=True, name="pli-falcon-capture").start()
            atexit.register(_at_exit)
            _last[0] = time.time()
        if _pending:
            _read_pending()
        if time.time() - _last[0] > 20.0:                    # a partial batch written at least every 20 s
            _last[0] = time.time()
            try:
                _queue.put_nowait(None)
            except queue.Full:
                pass
        if _stream is None:
            # PLI: a separate GPU stream for the capture; the next round's drafts (default stream) do not wait for it
            _stream = mx.new_stream(mx.gpu) if os.environ.get("PLI_FALCON_STREAM", "1") != "0" else mx.default_stream(mx.gpu)
        return _stream


def begin(flux: str, reused: int, prompt: Sequence[int]) -> None:
    """Start of a prompt's prefill: number of the request, and if ``reused`` rows come from the cache, the anchor."""

    if not FOLDER:
        return
    try:
        import numpy as np

        number = _numbers.setdefault(flux, len(_numbers) + 1)
        _rounds.pop(flux, None)
        if reused > 0:
            tokens = np.full(ANCHOR, -1, dtype=np.int32)
            tail = [int(t) for t in prompt[max(0, reused - ANCHOR):reused]]
            tokens[ANCHOR - len(tail):] = tail
            _prepare()
            _pending.append(("link", number, int(reused), tokens, None, None, 0))
    except Exception as e:
        count["errors"] += 1
        _say("begin", f"begin failed: {type(e).__name__}: {e}")


def context(flux: str, position: int, hidden: Any, nexts: Any, prompt_end: int, logits: Any = None) -> None:
    """Prompt rows the head reads (``hidden`` [n, D], row i at ``position`` + i, next token ``nexts``),
    kept if they are among the last PLI_FALCON_PROMPT; targets only if ``logits`` [n, V] is given."""

    if not FOLDER:
        return
    try:
        import mlx.core as mx

        n = int(hidden.shape[0])
        skip = max(0, prompt_end - PROMPT - position)
        if skip >= n:
            return
        with mx.stream(_prepare()):
            scale = None
            if CONTEXT8:
                xf = hidden[skip:].astype(mx.bfloat16).astype(mx.float32)        # the bits of the served bf16
                scale = mx.maximum(mx.max(mx.abs(xf), axis=1, keepdims=True), 1e-30) / 127.0
                h = mx.clip(mx.round(xf / scale), -127, 127).astype(mx.int8)
                scale = scale.reshape(-1)
            else:
                h = hidden[skip:].astype(mx.bfloat16).view(mx.uint16)
            k = n - skip
            if logits is not None:
                lg = logits[skip:]
                ind = mx.argpartition(-lg, kth=K - 1, axis=-1)[:, :K]
                val = mx.take_along_axis(lg, ind, axis=-1).astype(mx.float16)
                ind = ind.astype(mx.int32)
            else:
                ind = mx.full((k, K), -1, dtype=mx.int32)
                val = mx.zeros((k, K), dtype=mx.float16)
            suiv = nexts[skip:] if isinstance(nexts, mx.array) else mx.array([int(t) for t in nexts[skip:]],
                                                                             dtype=mx.int32)
            mx.async_eval(h, ind, val, suiv, *([] if scale is None else [scale]))
        number = _numbers.setdefault(flux, len(_numbers) + 1)
        _pending.append((number, int(position) + skip, h, ind, val, suiv, 1, scale))
        count["context"] = count.get("context", 0) + k
        _say("context", f"first prompt rows captured: {k} from {int(position) + skip} (end {prompt_end})")
    except Exception as e:
        count["errors"] += 1
        _say("context", f"context failed: {type(e).__name__}: {e}")


def _capture_on_stream(flux: str, position: int, hidden: Any, logits: Any, rows: Sequence[int], nexts: Sequence[int]) -> None:
    import mlx.core as mx

    if True:
        idx = mx.array([int(r) for r in rows], dtype=mx.int32)
        # the uint16 view computed with the rest: when read, numpy finds no operation left to launch (otherwise
        # a GPU synchronisation at each round: + 1.8 % per round)
        h = mx.take(hidden, idx, axis=0).astype(mx.bfloat16).view(mx.uint16)
        rnd = _rounds.get(flux, 0)
        _rounds[flux] = rnd + 1
        if rnd % TARGETS_EVERY == 0:
            lg = mx.take(logits, idx, axis=0)
            ind = mx.argpartition(-lg, kth=K - 1, axis=-1)[:, :K]
            val = mx.take_along_axis(lg, ind, axis=-1).astype(mx.float16)
            ind = ind.astype(mx.int32)
        else:                                            # no target this round: the row serves as context
            ind = mx.full((len(rows), K), -1, dtype=mx.int32)
            val = mx.zeros((len(rows), K), dtype=mx.float16)
        mx.async_eval(h, ind, val)
        number = _numbers.setdefault(flux, len(_numbers) + 1)
        _pending.append((number, int(position), h, ind, val, [int(t) for t in nexts], 0))
        _say("begin", f"capture to {FOLDER} (K={K}, batches of {BATCH} rows, {MAX_GB} GB at most, separate stream); "
                      f"h {tuple(h.shape)}")


__all__ = ["active", "capture", "count", "context", "begin"]
