"""PLI (13/10/2026): PREFILL attention of the 27B (parts of more than 16 rows, D 256, bf16) with the products in f16, K, V and Q converted by groups of
HC key heads. CLASS V3: the bits change (measure: computation error against an f32 reference divided by 2 to 3 compared with 2.7.0).
Wired into prompt_attention.py, enabled by PLI_ATTENTION_F16=1 (0 = original).

Same sequence as prompt_attention_gemm.py (two products of MLX's steel_gemm_fused kernel, MIT licence Apple, causal mask applied in the store of Q.Kt, MLX's
softmax in precise mode, parts of 128 rows, two parts in flight) with three differences:
  - the inputs of both products are in f16 (K and V of the key-head group converted before its parts, Q scaled in f16): f16 runs faster than bf16 on
    M1 (bf16 emulated). The bf16 -> f16 conversion is exact only between 6.1e-5 and 65504 in absolute value: above, f16 gives infinity (then NaN in the softmax);
    below, the values lose digits (subnormals) then become 0. PLI_F16_CLAMP=1 bounds Q, K and V to +-65280 (largest bf16 that f16 represents)
    before the conversion: same bits as long as nothing overflows. Scores and probabilities in f16 (probabilities under 6.1e-5 less precise at very long context);
  - the key heads are handled by groups of HC (default 2): the f16 copy of K and V covers only HC heads out of 4, and the scores 6 x HC heads (lower memory peak);
  - the output is returned in bf16.
Conditions (otherwise the caller keeps the original path): bf16, b = 1, D = 256, more than 16 rows, number of visible keys a multiple of 16 and at least MIN_KEYS (a part that
does not fill them takes MLX's SDPA in f16), buffers with inner stride 1 and row stride 256.
"""

from __future__ import annotations

import os

import mlx.core as mx

from tensorfold.kernels.qwen.dense.v1 import prompt_attention_gemm as _g

ROWS = 128
KEY_BLOCK = 16          # MLX 0.32.2's key block before M5: a part ending inside one rounds unlike one whole call
HC = int(os.environ.get("PLI_ATTENTION_F16_HEADS", "2"))
QK_CFG = tuple(int(x) for x in os.environ.get("PLI_ATTENTION_F16_QK", "64,64,16,2,2").split(","))
PV_CFG = tuple(int(x) for x in os.environ.get("PLI_ATTENTION_F16_PV", "32,64,32,2,2").split(","))
MIN_KEYS = int(os.environ.get("PLI_ATTENTION_F16_MIN", "16384"))
CLAMP = os.environ.get("PLI_F16_CLAMP", "0") == "1"
BF16_MAX_F16 = 65280.0   # largest bfloat16 that float16 represents (the next one, 65536, overflows)

_SOURCE = _g._SOURCE.replace("typedef bfloat16_t T;", "typedef half T;")
assert _SOURCE != _g._SOURCE
_HEADER = None
_KER: dict = {}
_EX = None


def _kernel(cfg, tb, mask):
    global _HEADER
    if _HEADER is None:
        _HEADER = _g._expand_includes("mlx/backend/metal/kernels/steel/gemm/gemm.h", {"mlx/backend/metal/kernels/steel/gemm/gemm.h"}) + "\n"
    key = (cfg, tb, mask)
    if key not in _KER:
        bm, bn, bk, wm, wn = cfg
        src = (_SOURCE.replace("@BM@", str(bm)).replace("@BN@", str(bn)).replace("@BK@", str(bk)).replace("@WM@", str(wm)).replace("@WN@", str(wn))
               .replace("@TB@", "true" if tb else "false").replace("@MASK@", "true" if mask else "false"))
        _KER[key] = mx.fast.metal_kernel(name="pli_attf16_%d_%d_%d_%d_%d_%d_%d" % (bm, bn, bk, wm, wn, int(tb), int(mask)),
                                         input_names=["A", "B", "prm", "ex"], output_names=["D"], source=src, header=_HEADER,
                                         ensure_row_contiguous=False)
    return _KER[key]


def _gemm(a, b, batch, m, n, k, off, grp, cfg, tb, mask):
    global _EX
    if _EX is None:
        _EX = mx.array([float(mx.finfo(mx.float16).min)], dtype=mx.float16)
    bm, bn, _bk, wm, wn = cfg
    prm = mx.array([m, n, k, off, grp], dtype=mx.int32)
    tn, tm = -(-n // bn), -(-m // bm)
    return _kernel(cfg, tb, mask)(inputs=[a, b, prm, _EX], output_shapes=[(batch, m, n)], output_dtypes=[mx.float16],
                                 grid=(tn * 32, tm * wn, batch * wm), threadgroup=(32, wn, wm))[0]


def _applicable(queries: mx.array, keys: mx.array, values: mx.array) -> bool:
    b, hq, r, d = (int(v) for v in queries.shape)
    hk, vis = int(keys.shape[1]), int(keys.shape[2])
    return (b == 1 and d == 256 and r > 16 and hq % hk == 0 and vis % 16 == 0 and vis >= MIN_KEYS
            and queries.dtype == mx.float16 and keys.dtype == mx.float16 and values.dtype == mx.float16
            and _g._layout_ok(keys) and _g._layout_ok(values))


def _part(queries: mx.array, keys: mx.array, values: mx.array, scale: float) -> mx.array:
    """Causal attention of ``queries`` (the last rows of the ``keys.shape[2]`` positions); f16 inputs, f16 output."""
    b, hq, r, d = (int(v) for v in queries.shape)
    hk, vis = int(keys.shape[1]), int(keys.shape[2])
    g = hq // hk
    qs = (mx.array(scale, queries.dtype) * queries).reshape(hq, r, d)
    s = _gemm(qs, keys, hq, r, vis, d, vis - r, g, QK_CFG, True, True)
    p = mx.softmax(s, axis=-1, precise=True)
    o = _gemm(p, values, hq, r, d, vis, 0, g, PV_CFG, False, False)
    return o.reshape(1, hq, r, d)


def usable(queries: mx.array, keys: mx.array, values: mx.array) -> bool:
    """Is the f16 path suitable? (bf16, b 1, D 256, query heads a multiple of the key heads; no device matrix tiles.)"""
    from tensorfold.families.qwen3_5 import tensor_units

    b, hq, _r, d = (int(v) for v in queries.shape)
    hk = int(keys.shape[1])
    return (b == 1 and d == 256 and hq % hk == 0 and not tensor_units()
            and queries.dtype == mx.bfloat16 and keys.dtype == mx.bfloat16 and values.dtype == mx.bfloat16)


def attend(queries: mx.array, keys: mx.array, values: mx.array, scale: float) -> mx.array:
    """Causal attention of the ``queries.shape[2]`` last of the ``keys.shape[2]`` positions (parts of 128 rows like prompt_attention.attend), bf16 output."""
    rows, total = int(queries.shape[2]), int(keys.shape[2])
    hq, hk = int(queries.shape[1]), int(keys.shape[1])
    g = hq // hk
    bounds, begin = [], 0
    while begin < rows:
        end = min(rows, begin + ROWS)
        if end < rows:
            end -= (total - rows + end) % KEY_BLOCK                 # its keys end on a key block
        # A short tail would select vector attention and change the full prompt's bits.
        if 0 < rows - end <= 16:
            end = rows
        bounds.append((begin, end, total - rows + end))
        begin = end
    chunks: list[mx.array] = []
    prev = None
    for h0 in range(0, hk, HC):
        h1 = min(hk, h0 + HC)
        kh, vh, qh = keys[:, h0:h1], values[:, h0:h1], queries[:, h0 * g:h1 * g]
        if CLAMP:
            kh, vh, qh = (mx.clip(x, -BF16_MAX_F16, BF16_MAX_F16) for x in (kh, vh, qh))
        kh, vh, qh = kh.astype(mx.float16), vh.astype(mx.float16), qh.astype(mx.float16)
        outs: list[mx.array] = []
        for begin, end, visible in bounds:
            qq, kk, vv = qh[:, :, begin:end], kh[:, :, :visible], vh[:, :, :visible]
            if _applicable(qq, kk, vv):
                part = _part(qq, kk, vv, scale)
            else:
                part = mx.fast.scaled_dot_product_attention(qq, kk, vv, scale=scale, mask="causal")
            mx.async_eval(part)
            if prev is not None:
                mx.eval(prev)
            prev = part
            outs.append(part)
        chunks.append(mx.concatenate(outs, axis=2))
    return mx.concatenate(chunks, axis=1).astype(mx.bfloat16)


__all__ = ["attend", "usable"]
