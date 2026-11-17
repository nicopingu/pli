"""PLI (29/10/2026, extended list, PLI_EXTENDED_DRAFT_LIST=<file>, off by default): tokens added to the drafts' lists.

The drafts' reduced heads never propose a token outside their list (DFlash2: ids 0 to 98,303 and the special ones;
MTP head: 79,591 ids of ``draft_vocab.txt``): each token written outside the list is a forced rejection that cuts the chain.
Measured on 709,177 tokens written in service (29/10, the author's traffic, mostly French): 3.89 % of the written tokens are outside DFlash2's list,
5.6 % outside the MTP head's, mostly French words ("modèles", "octobre", "fichiers", "mémoire"...).
A list of 2,048 ids per draft, learned on the first half of the captures, brings the second half down to 0.69 % (DFlash2)
and 1.20 % (MTP head); merged, the two lists hold 2,515 ids. The gain on other traffic is not measured.

The file (one id per line) extends both lists once, at load time: rows of lm_head added to the reduced head,
no extra product per round. Only the drafts change: the verification stays exact (same text, V2).
The list shipped with PLI is ``extended-list-2048.txt`` (``liste-plus-2048.txt`` before 4.0.0: the old file name is still found).
"""
from __future__ import annotations

import os
from typing import Any

FILE = os.environ.get("PLI_EXTENDED_DRAFT_LIST", "").strip()
_ids: "list[int] | None" = None
OLD_PREFIX, NEW_PREFIX = "liste-plus-", "extended-list-"


def active() -> bool:
    return bool(FILE)


def _path(name: str) -> str:
    """``name`` (a bare file name is looked up next to this module, where PLI ships its lists), or the same list under
    its new file name when an old name (``liste-plus-*.txt``) is given and missing."""

    if not os.path.dirname(name) and not os.path.exists(name):
        name = os.path.join(os.path.dirname(os.path.abspath(__file__)), name)   # a bare file name: the list shipped with PLI
    base = os.path.basename(name)
    if base.startswith(OLD_PREFIX) and not os.path.exists(name):
        new = os.path.join(os.path.dirname(name), NEW_PREFIX + base[len(OLD_PREFIX):])
        if os.path.exists(new):
            return new
    return name


def ids() -> list[int]:
    """The file's ids, sorted and without duplicates (empty list if off or unreadable)."""

    global _ids
    if _ids is None:
        _ids = []
        if FILE:
            try:
                with open(_path(FILE), encoding="utf-8") as f:
                    _ids = sorted({int(x) for x in f.read().split() if x.strip()})
            except (OSError, ValueError) as exc:
                print(f"[tensorfold] PLI extended list: {FILE} unreadable ({exc}); draft lists unchanged",
                      flush=True)
                _ids = []
    return _ids


def outside(spans: Any, n: int, multiple: int = 1) -> list[int]:
    """The file's ids outside the intervals ``spans`` and under ``n``, padded to a multiple of ``multiple``
    with the smallest ids outside the intervals and outside the file (no duplicate column)."""

    inside = lambda t: any(a <= t < b for a, b in spans)  # noqa: E731
    out = [t for t in ids() if 0 <= t < n and not inside(t)]
    if out and multiple > 1:
        taken, t = set(out), 0
        while len(out) % multiple and t < n:
            if not inside(t) and t not in taken:
                out.append(t)
                taken.add(t)
            t += 1
        out.sort()
    return out


def head_rows(head: Any, rows: list[int]) -> tuple[Any, Any, Any]:
    """The rows ``rows`` of an MLX quantized head, in MLX's format (weights laid out in tiles put back in order)."""

    import mlx.core as mx

    idx = mx.array(rows, dtype=mx.int32)
    w = head["weight"]
    if getattr(head, "_lane_tiled", False):
        nt = int(getattr(head, "_lane_nt", 32))
        n, kw = int(w.shape[0]), int(w.shape[1])
        word = int(head.group_size) * int(head.bits) // 32
        w4 = w.reshape(n // nt, kw // word, nt, word)                              # [tile][group][row][words]
        tiles = mx.take(w4, mx.array([i // nt for i in rows], dtype=mx.int32), axis=0)      # [m, kw/word, nt, word]
        r = mx.array([i % nt for i in rows], dtype=mx.int32).reshape(-1, 1, 1, 1)
        w = mx.take_along_axis(tiles, mx.broadcast_to(r, (len(rows), kw // word, 1, word)), axis=2).reshape(len(rows), kw)
    else:
        w = mx.take(w, idx, axis=0)
    s = mx.take(head["scales"], idx, axis=0)
    b = mx.take(head["biases"], idx, axis=0)
    return mx.contiguous(w), s, b


def say(who: str, m: int) -> None:
    print(f"[tensorfold] PLI extended list ({os.path.basename(FILE)}): {m} ids added to the list of {who}", flush=True)


__all__ = ["active", "head_rows", "ids", "outside", "say"]
