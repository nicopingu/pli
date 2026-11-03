"""PLI (28/10/2026): checkpoints without a copy of the KV cache, in Kapitsa's manner (a checkpoint keeps the length).

With MLX, a checkpoint that shares the arrays of the live conversation makes the whole KV buffer be copied at the first append
(copy on write), that is 6 to 7 GiB more at 96,000 tokens. Here, a checkpoint keeps, for each attention layer, the conversation's
cache OBJECT and a length n (``KVLen``), never a reference to the arrays: the conversation writes in place, and the first n
positions do not change while the conversation moves on. The other layers (DeltaNet states, MTP head, draft) are copied as before
(their arrays are replaced, never written in place).

Rules:
- a live attention layer (``pli_live`` = 1) is never given to another request: that one gets a copy of the n positions;
- a free layer is given without a copy to the request that resumes; if it was longer than n, it is brought back to n and the
  checkpoints that relied on more than n positions are removed;
- a "real" checkpoint (arrays of its own: end of a request, read from disk) becomes a length checkpoint when it is resumed;
  when a real checkpoint is evicted, the length checkpoints on the same layers leave with it (its memory is then freed).
PLI_KV_SHARING=1 turns it on. Off by default.
"""

from __future__ import annotations

import os
from typing import Any, Callable, Sequence

ACTIVE = os.environ.get("PLI_KV_SHARING", "0") == "1"


def is_kv(item: Any) -> bool:
    from mlx_lm.models.cache import KVCache

    return type(item) is KVCache


class KVLen:
    """Attention layer of a checkpoint: the conversation's layer and the kept length."""

    keys = None                      # no array of its own (memory accounting, cache_arrays)
    values = None
    stored = True

    def __init__(self, owner: Any, n: int) -> None:
        self.owner = owner
        self.n = int(n)

    @property
    def offset(self) -> int:
        return self.n

    def valid(self) -> bool:
        return self.owner.keys is not None and int(self.owner.offset) >= self.n

    def export(self) -> Any:
        """An ordinary KV layer on views of the n positions (writing to disk)."""

        from mlx_lm.models.cache import KVCache

        c = KVCache()
        r = self.owner
        c.keys, c.values, c.offset = r.keys[..., : self.n, :], r.values[..., : self.n, :], self.n
        return c

    def copy(self) -> Any:
        """A KV layer of its own, a copy of the n positions (the layer is live elsewhere)."""

        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        c = KVCache()
        r = self.owner
        c.keys = mx.contiguous(r.keys[..., : self.n, :])
        c.values = mx.contiguous(r.values[..., : self.n, :])
        c.offset = self.n
        mx.eval(c.keys, c.values)
        return c


def _free(r: Any) -> bool:
    return not getattr(r, "pli_live", 0)


def owner(item: Any) -> Any:
    return item.owner if isinstance(item, KVLen) else item


def view(cache: Sequence[Any], copier: Callable[[list[Any]], list[Any]]) -> list[Any]:
    """Checkpoint taken on a live conversation (prompt read): KV layers by length, the rest copied."""

    if not any(is_kv(item) for item in cache):
        return copier(list(cache))
    out = []
    for item in cache:
        if is_kv(item) and item.keys is not None:
            item.pli_live = 1
            out.append(KVLen(item, int(item.offset)))
        else:
            out.append(copier([item])[0])
    return out


def lighten(copy: Sequence[Any], live: Sequence[Any]) -> list[Any]:
    """A copy already made (and brought back, PLI_MERGE_TAIL): its KV layers become lengths on the conversation."""

    out = []
    for c, w in zip(copy, live):
        if is_kv(c) and is_kv(w) and w.keys is not None and int(c.offset) <= int(w.offset):
            w.pli_live = 1
            out.append(KVLen(w, int(c.offset)))
        else:
            out.append(c)
    return out


def retain(cache: Sequence[Any], copier: Callable[[list[Any]], list[Any]]) -> list[Any]:
    """End of a request: its KV layers become free and form the real checkpoint (same objects, no copy)."""

    if not any(is_kv(item) for item in cache):
        return copier(list(cache))
    out = []
    for item in cache:
        if is_kv(item):
            item.pli_live = 0
            out.append(item)
        else:
            out.append(copier([item])[0])
    return out


def release(cache: Sequence[Any]) -> None:
    if not isinstance(cache, (list, tuple)):
        return
    for item in cache:
        if is_kv(item):
            item.pli_live = 0


def _invalidate(entries: list[Any], r: Any, n: int) -> None:
    """Removes the checkpoints that relied on more than n positions of layer r (it is about to be rewritten)."""

    kept = []
    for e in entries:
        bad = False
        for item in e.cache:
            if isinstance(item, KVLen) and item.owner is r and item.n > n:
                bad = True
            elif item is r and int(r.offset) > n:
                bad = True
        if not bad:
            kept.append(e)
    entries[:] = kept


def resume(cache: Sequence[Any], n: int, copier: Callable[[list[Any]], list[Any]], entries: list[Any],
           take: bool = False) -> list[Any]:
    """The cache of a request that resumes an n-token checkpoint: free KV layers given without a copy, copied otherwise.
    ``take`` (memory does not allow a copy: the checkpoint is removed): everything is given as is, except a length layer
    whose conversation is still live, which is copied."""

    if not any(isinstance(item, KVLen) or is_kv(item) for item in cache):
        return list(cache) if take else copier(list(cache))         # no KV layer: as before
    out = []
    for item in cache:
        if isinstance(item, KVLen) or is_kv(item):
            r = owner(item)
            if r.keys is not None and _free(r) and int(r.offset) >= n:
                if int(r.offset) > n:
                    _invalidate(entries, r, n)
                    r.trim(int(r.offset) - n)
                r.pli_live = 1
                out.append(r)
            elif isinstance(item, KVLen):
                out.append(item.copy())
            else:
                out.append(item if take else copier([item])[0])
        else:
            out.append(item if take else copier([item])[0])
    return out


def as_lengths(cache: Sequence[Any], given: Sequence[Any], n: int) -> list[Any]:
    """The checkpoint after ``resume``: the layers given to the request become lengths in it."""

    out = []
    for item, d in zip(cache, given):
        if (isinstance(item, KVLen) or is_kv(item)) and owner(item) is d:
            out.append(KVLen(d, n))
        else:
            out.append(item)
    return out


def evict(entries: list[Any], gone: Sequence[Any], keep: Any = None) -> list[Any]:
    """The length checkpoints that depend on an evicted real layer leave with it (except ``keep``); returned."""

    real = {id(item) for e in gone for item in e.cache if is_kv(item)}
    if not real:
        return []
    left = [e for e in entries if e is not keep and any(isinstance(i, KVLen) and id(i.owner) in real and _free(i.owner)
                                                        for i in e.cache)]
    if left:
        entries[:] = [e for e in entries if e not in left]
    return left


__all__ = ["ACTIVE", "KVLen", "as_lengths", "evict", "is_kv", "lighten", "owner", "release", "resume", "retain", "view"]
