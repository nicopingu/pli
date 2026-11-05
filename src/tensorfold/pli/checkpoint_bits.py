"""PLI (04/11/2026, KVD5): checkpoints kept with K and V in 8 bits (or 4 bits after a rotation).

The live cache of a request stays in 16 bits. A checkpoint is reduced only when it has to be:
- evicted from the store for the budget or for memory: before evicting, the store reduces the oldest checkpoint that has a
  real copy of its KV layers (a free layer that no K5 length checkpoint holds), then counts again;
- written to disk (spill, save at shutdown): the file carries the reduced layers;
- read back from disk: it stays reduced in memory.
On resume (``match``), the reduced layers are restored to 16 bits (a new KV layer, given to the request).
A length checkpoint (K5) has no copy of its own: it is never reduced. DeltaNet state, MTP head, draft: unchanged.

PLI_CHECKPOINT_BITS=0 (default): off, previous behaviour. 8: affine 8 bits in groups of 64 (mx.quantize).
4: 4 bits, orthonormal Hadamard rotation over the head dimension (mx.hadamard_transform) before quantization.
The setting enters the file key (cli.py, "|points=..."): a reduced checkpoint is not resumed by an engine where it is off.
Text can change: a resumed answer may differ from the 16-bit one (in production on an M1 Ultra since 2026-11-04: 4 bits).
"""

from __future__ import annotations

import os
from typing import Any, Callable, Sequence

BITS = int(os.environ.get("PLI_CHECKPOINT_BITS", "0") or 0)
GROUP = int(os.environ.get("PLI_CHECKPOINT_GROUP", "64") or 64)
if BITS not in (0, 4, 8):
    raise ValueError("PLI_CHECKPOINT_BITS must be 0, 4 or 8")


def key_suffix(bits: int | None = None) -> str:
    """The file-key part; empty when off (previous key, bit for bit)."""

    b = BITS if bits is None else int(bits)
    return "" if b == 0 else f"|points=q{b}{'h' if b == 4 else ''}g{GROUP}"


class QuantizedKV:
    """An attention layer of a checkpoint, K and V quantized (n positions)."""

    stored = True
    keys = None                      # no 16-bit arrays: neither a KV layer nor a memory probe
    values = None

    def __init__(self, wk: Any, sk: Any, bk: Any, wv: Any, sv: Any, bv: Any, offset: int, bits: int, group: int,
                 rotated: bool) -> None:
        self.wk, self.sk, self.bk = wk, sk, bk
        self.wv, self.sv, self.bv = wv, sv, bv
        self.offset = int(offset)
        self.bits = int(bits)
        self.group = int(group)
        self.rotated = bool(rotated)

    @property
    def state(self) -> list[Any]:
        return [self.wk, self.sk, self.bk, self.wv, self.sv, self.bv]

    @property
    def nbytes(self) -> int:
        return sum(int(a.nbytes) for a in self.state)

    def size(self) -> int:
        return self.offset

    def restore(self) -> Any:
        """A 16-bit KV layer of its own (same shape and dtype as before the reduction)."""

        import mlx.core as mx
        from mlx_lm.models.cache import KVCache

        k = mx.dequantize(self.wk, self.sk, self.bk, group_size=self.group, bits=self.bits)
        v = mx.dequantize(self.wv, self.sv, self.bv, group_size=self.group, bits=self.bits)
        if self.rotated:
            k, v = _rotate(k), _rotate(v)
        c = KVCache()
        c.keys, c.values, c.offset = k, v, self.offset
        mx.eval(c.keys, c.values)
        return c


def _rotate(x: Any) -> Any:
    """Orthonormal Hadamard rotation over the last dimension (its own inverse), computed in f32, dtype kept."""

    import mlx.core as mx

    d = int(x.shape[-1])
    return mx.hadamard_transform(x.astype(mx.float32), scale=d ** -0.5).astype(x.dtype)


def is_quantized(item: Any) -> bool:
    return type(item) is QuantizedKV


def reducible(item: Any) -> bool:
    from tensorfold.pli import kv_sharing as _kvs

    return _kvs.is_kv(item) and item.keys is not None and int(item.offset) > 0


def reduce(item: Any, bits: int | None = None) -> QuantizedKV:
    """A KV layer (or a length layer, exported over its n positions) reduced, evaluated."""

    import mlx.core as mx

    b = BITS if bits is None else int(bits)
    export = getattr(item, "export", None)
    if callable(export):
        item = export()
    n = int(item.offset)
    k, v = item.keys[..., :n, :], item.values[..., :n, :]
    rotated = b == 4
    if rotated:
        k, v = _rotate(k), _rotate(v)
    wk, sk, bk = mx.quantize(k, group_size=GROUP, bits=b)
    wv, sv, bv = mx.quantize(v, group_size=GROUP, bits=b)
    mx.eval(wk, sk, bk, wv, sv, bv)
    return QuantizedKV(wk, sk, bk, wv, sv, bv, n, b, GROUP, rotated)


def reduce_cache(cache: Sequence[Any], bits: int | None = None) -> list[Any]:
    """A checkpoint's cache with its KV layers reduced (length layers included); the rest as is."""

    from tensorfold.pli import kv_sharing as _kvs

    out = []
    for item in cache:
        if isinstance(item, _kvs.KVLen) or reducible(item):
            out.append(reduce(item, bits))
        else:
            out.append(item)
    return out


def has_reduced(cache: Sequence[Any]) -> bool:
    return any(is_quantized(item) for item in cache)


def restore_cache(cache: Sequence[Any], copier: Callable[[list[Any]], list[Any]] | None = None) -> list[Any]:
    """The cache given to a request: reduced layers restored to 16 bits, the rest copied by ``copier`` (or as is)."""

    out = []
    for item in cache:
        if is_quantized(item):
            out.append(item.restore())
        else:
            out.append(item if copier is None else copier([item])[0])
    return out


def for_disk(cache: Sequence[Any]) -> list[Any]:
    """What is written of a checkpoint when on: its KV layers reduced (already reduced: as they are)."""

    return list(cache) if BITS == 0 else reduce_cache(cache)


__all__ = ["BITS", "GROUP", "QuantizedKV", "for_disk", "has_reduced", "is_quantized", "key_suffix", "reduce",
           "reduce_cache", "reducible", "restore_cache"]
