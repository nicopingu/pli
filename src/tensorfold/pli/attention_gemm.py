"""PLI, rank 2, second path: causal attention through MLX's matrix products, exact row by row.

Read in MLX 0.32.3 (mlx/backend/metal/matmul.cpp, kernel "steel_gemm_fused"): for a batched product (more than one
key/value head), MLX always takes the "fused" kernel, where each element of the result is ONE 32-bit accumulation
chain, k increasing, the same whatever the number of rows or columns; the result is rounded once to bf16.
The "splitk" kernel, whose split depends on the sizes, serves only without a batch. From this comes an attention where a row has
the same bits alone or in a window:

  scores  S = K x Q^T      [kv heads, keys, queries]   one chain over the 256 dimensions per element
  weights P = exp((S - max) x scale), zero for the masked keys (maximum: exact, without order)
  output  O = P^T x V      one chain over the keys, in order; the masked keys add exact zeros
  sum     l = P^T x 1      same kernel, same rounded weights
The number of keys is rounded up to the next multiple of 16 (rows of the cache buffer past the context, masked):
otherwise MLX handles the tail of the keys first, and the order would depend on the context length.
The scores are rounded to bf16, as in MLX's prompt prefill (MLX's decode kernel keeps them in 32 bits).
Chains only (no tree): row t of a window of T sees keys 0 .. L - T + t.
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx

MAX_QUERIES = 128
ROWS = int(os.environ.get("PLI_ATTN_ROWS", "16"))     # rows per pass (working memory); no effect on the bits
_ones: dict[tuple[int, int], mx.array] = {}


def _slice(q: mx.array, kb: mx.array, vb: mx.array, scale: float, L: int, T: int, t0: int, n: int) -> mx.array:
    """Rows t0 .. t0 + n - 1 of a window of T: q [HKV, G, n, D], buffers [HKV, L16, D]."""

    HKV, G, _, D = (int(s) for s in q.shape)
    L16 = int(kb.shape[1])
    m = G * n
    qt = mx.swapaxes(q.reshape(HKV, m, D), 1, 2)                       # [HKV, D, m], column g n + t
    s = kb @ qt                                                        # [HKV, L16, m] bf16
    la = max(0, (L - T) // 16 * 16)                                    # before: keys seen by all rows
    seen = mx.arange(la, L16)[:, None] < (L - T + 1 + t0 + mx.arange(m) % n)[None, :]
    s = mx.concatenate([s[:, :la], mx.where(seen[None], s[:, la:], -mx.inf)], axis=1)
    p = mx.exp((s.astype(mx.float32) - mx.max(s, axis=1, keepdims=True).astype(mx.float32)) * scale).astype(mx.bfloat16)
    pt = mx.swapaxes(p, 1, 2)                                          # [HKV, m, L16]
    o = (pt @ vb).astype(mx.float32)                                   # [HKV, m, D]
    one = _ones.get((HKV, L16))
    if one is None:
        if len(_ones) > 8:
            _ones.clear()
        one = _ones[(HKV, L16)] = mx.ones((HKV, L16, 8), dtype=mx.float32)
    l = (pt.astype(mx.float32) @ one)[..., :1]                         # [HKV, m, 1]
    return (o / l).astype(mx.bfloat16).reshape(HKV, G, n, D)


def sdpa(queries: mx.array, keys: mx.array, values: mx.array, scale: float, cache: Any = None, **_: Any) -> mx.array:
    """Causal attention: queries [1, H, T, D], keys and values [1, Hkv, L, D]. ``cache``: its buffer avoids a copy."""

    _, H, T, D = (int(s) for s in queries.shape)
    HKV, L = int(keys.shape[1]), int(keys.shape[2])
    if HKV < 2 or H % HKV or T > MAX_QUERIES or T > L or D % 16:
        raise ValueError(f"pli attention: shape not handled q={queries.shape} k={keys.shape}")
    G = H // HKV
    L16 = -(-L // 16) * 16
    kb = getattr(cache, "keys", None)
    vb = getattr(cache, "values", None)
    if kb is not None and vb is not None and int(kb.shape[2]) >= L16 and int(vb.shape[2]) >= L16:
        kb, vb = kb[0, :, :L16], vb[0, :, :L16]                        # views of the buffer: no copy
    elif L16 == L:
        kb, vb = keys[0], values[0]
    else:                                                              # no buffer: padded with zeros (copy)
        z = mx.zeros((HKV, L16 - L, D), dtype=keys.dtype)
        kb, vb = mx.concatenate([keys[0], z], axis=1), mx.concatenate([values[0], z], axis=1)
    q = queries.reshape(HKV, G, T, D)
    outs = [_slice(q[:, :, t0:t0 + ROWS], kb, vb, float(scale), L, T, t0, min(ROWS, T - t0)) for t0 in range(0, T, ROWS)]
    out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)
    return out.reshape(1, H, T, D)


def sources() -> dict[str, str]:
    import inspect

    return {"gemm": inspect.getsource(_slice), "mlx": mx.__version__}


__all__ = ["MAX_QUERIES", "sdpa", "sources"]
