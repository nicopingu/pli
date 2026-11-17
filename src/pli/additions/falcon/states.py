"""PLI (16/11/2026): FALCON B, the 27B's states read by the served model itself, in short pieces.

FALCON B adjusts the DFlash2 drafter on the states it reads in service: the outputs of the 27B's layers 5, 19, 33, 47
and 61 (the drafter's taps) and the 27B's greedy choice at each position. Until 15/11/2026, the night job
(~/dev/dflash2-ajuste/ajuste.py) loaded a second copy of the 27B to recompute them (about 44 GB with the training, hence
a cut of the service of 20 to 40 minutes a night). Here the served model reads them, through the same forward as a prompt
chunk (``family.prefill``: the 27B's caches, kernels and the drafter's layer hooks), a piece of ``chunk`` tokens a call,
so that a background task can interleave the reading with the service and stop between two pieces.

:func:`read_states` yields, piece by piece, (begin, end, taps [rows, 5 * hidden] bf16, greedy [rows] int32), the greedy
choice at row i being the 27B's next token after tokens[: begin + i + 1]. Nothing is kept by the engine: the cache is the
caller's (a fresh one per sequence), the taps are released after each piece. Called on the engine thread, no stream live.
"""

from __future__ import annotations

from typing import Any, Iterator, Sequence


def drafter_of(family: Any) -> Any:
    head = getattr(family, "head_drafts", None)
    drafter = getattr(head, "drafter", None)
    if drafter is None or not hasattr(drafter, "taps"):
        raise ValueError("the served model has no DFlash2 drafter: no taps to read")
    return drafter


def read_states(family: Any, tokens: Sequence[int], *, chunk: int = 512, cache: list[Any] | None = None,
                start: int = 0) -> Iterator[tuple[int, int, Any, Any]]:
    import mlx.core as mx

    from pli.engine.family_common import cache_arrays

    drafter = drafter_of(family)
    feed = getattr(family, "prefill", None) or family.hidden
    work = family.make_cache() if cache is None else cache
    layers = work[:-1] if work and type(work[-1]).__name__ == "DraftSlot" else work
    for i, layer in enumerate(layers):           # as a request's prefill (family_prefill._family_prefill_steps)
        try:
            layer.pli_layer = i
            layer.pli_request_start = len(tokens)
            layer.pli_prefill_start = int(start)
            layer.pli_anchors = {}
        except Exception:  # noqa: BLE001 - a cache without those attributes reads as before
            pass
    tokens = [int(t) for t in tokens]
    for begin in range(int(start), len(tokens), max(1, int(chunk))):
        rows = tokens[begin:begin + int(chunk)]
        drafter.release_taps()
        hidden = feed(mx.array([rows], dtype=mx.uint32), work)
        taps = drafter.taps()
        if taps is None:
            raise RuntimeError("the forward left no taps (the drafter's layer hooks are missing)")
        greedy = mx.argmax(family.head(hidden)[0], axis=-1).astype(mx.int32)
        taps = taps[0].astype(mx.bfloat16)
        mx.eval(taps, greedy, *cache_arrays(work))
        drafter.release_taps()
        del hidden
        yield begin, begin + len(rows), taps, greedy
