"""Bound fallback prompt attention to two causal query parts in flight."""

from __future__ import annotations

import os

import mlx.core as mx

ROWS = 128
# PLI (12/10): PLI_ATTENTION_GROUPED=1 sends the query heads that share a key head in a single call (rows laid out
# head by head) with an explicit causal mask; same bits as the head-by-head causal call (trial on M1 Pro; M1 Ultra probe to follow).
# Only past 16 rows (below, MLX takes another kernel). 0 = original.
GROUPED = os.environ.get("PLI_ATTENTION_GROUPED", "0") == "1"


def _causal(queries: mx.array, keys: mx.array, values: mx.array, scale: float) -> mx.array:
    rows = int(queries.shape[2])
    b, hq, _, d = (int(v) for v in queries.shape)
    hk = int(keys.shape[1])
    if not GROUPED or rows <= 16 or hq == hk:
        return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask="causal")
    g, total = hq // hk, int(keys.shape[2])
    pos = mx.arange(total - rows, total)
    mask = mx.arange(total)[None, :] <= pos[:, None]
    mask = mx.broadcast_to(mask[None], (g, rows, total)).reshape(1, 1, g * rows, total)
    out = mx.fast.scaled_dot_product_attention(queries.reshape(b, hk, g * rows, d), keys, values, scale=scale, mask=mask)
    return out.reshape(b, hq, rows, d)
# PLI (12/10): PLI_ATTENTION_GEMM=1 does the two products (Q.Kt with the causal mask in the store, P.V) of a part with MLX's steel kernel
# instantiated with other tiles (prompt_attention_gemm.py); same bits as MLX's unfused path; falls back to MLX's SDPA when a condition is missing. 0 = original.
GEMM = os.environ.get("PLI_ATTENTION_GEMM", "0") == "1"
# PLI (13/10): PLI_ATTENTION_F16=1 takes the prefill attention (past FROM_KEYS keys, parts of more than ROWS rows) in f16 by groups of key heads
# (prompt_attention_f16.py): CLASS V3, the bits change; falls back to the original path when a condition is missing. 0 = original.
F16 = os.environ.get("PLI_ATTENTION_F16", "0") == "1"
FROM_KEYS = 4096
KEY_BLOCK = 16          # MLX 0.32.2's key block before M5: a part ending inside one rounds unlike one whole call


def _part(queries: mx.array, keys: mx.array, values: mx.array, scale: float) -> mx.array:
    if GEMM:
        from tensorfold.kernels.qwen.dense.v1 import prompt_attention_gemm as _g

        if _g.applicable(queries, keys, values):
            return _g.part(queries, keys, values, scale)
    return _causal(queries, keys, values, scale)


def attend(queries: mx.array, keys: mx.array, values: mx.array, scale: float) -> mx.array:
    """Causal attention of the last ``queries.shape[2]`` of ``keys.shape[2]`` positions."""

    rows, total = int(queries.shape[2]), int(keys.shape[2])
    if total <= FROM_KEYS or rows <= ROWS:
        return _causal(queries, keys, values, scale)
    if F16:
        from tensorfold.kernels.qwen.dense.v1 import prompt_attention_f16 as _f

        if _f.usable(queries, keys, values):
            return _f.attend(queries, keys, values, scale)
    from tensorfold.families.qwen3_5 import tensor_units

    block = 1 if tensor_units() else KEY_BLOCK
    outs: list[mx.array] = []
    begin = 0
    while begin < rows:
        end = min(rows, begin + ROWS)
        if end < rows:
            end -= (total - rows + end) % block                   # its keys end on a key block
        # A short tail would select vector attention and change the full prompt's bits.
        if 0 < rows - end <= 16:
            end = rows
        visible = total - rows + end
        part = _part(queries[:, :, begin:end], keys[:, :, :visible], values[:, :, :visible], scale)
        mx.async_eval(part)
        if outs:
            mx.eval(outs[-1])
        outs.append(part)
        begin = end
    return mx.concatenate(outs, axis=2)


__all__ = ["FROM_KEYS", "KEY_BLOCK", "ROWS", "attend"]
