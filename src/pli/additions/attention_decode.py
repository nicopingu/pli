"""Full decode attention of the 27B at equal bits: a single launch for all the rows of the round (pli-attec, 12/10/2026).

Copies the arithmetic of MLX 0.32.3's ``sdpa_vector_2pass`` (M1 Ultra, family "d", D 256, 24 query heads, 4 key heads):
pass 1 by blocks (key i goes to block i % blocks), q = scale x query in fp32, dot product per thread in order then a tree
of ``simd_sum``, running maximum, ``fast::exp``, partials in bf16; pass 2 merges them. Arithmetic text as is ("txt" mode:
the compiler contracts as for MLX). Each simdgroup handles RT = 2 rows with a single read of each key and value.
Measured at equal bits (``mx.array_equal``) on 528 cases. ``sdpa`` returns None when the shape does not fit what
was checked: the caller then takes the usual path.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

D, H, HKV = 256, 24, 4
G = H // HKV
RT = 2
MIN_KEYS = 1100                                  # smallest number of keys checked (sdpa_vector_2pass from 1024)
MAX_ROWS = 16
FMIN = "-3.4028234663852886e+38f"

_P1 = r"""
  constexpr int D = 256, G = 6, RT = 2;
  const uint lane = thread_index_in_simdgroup;
  const int g = int(simdgroup_index_in_threadgroup);
  const int blk = int(threadgroup_position_in_grid.y);
  const int z = int(threadgroup_position_in_grid.z);
  const int RG = RGA[0];
  const int kv = z / RG, rg = z % RG;
  const int qh = kv * G + g;
  const int NK = dims[0], T = dims[1], B = dims[2];
  constexpr int DPL = D / 32;
  const float FMIN = __FMIN__;
  const float sc = scale[0];
  float q[RT][DPL], o[RT][DPL], mxs[RT], sm[RT];
  int nrow[RT];
#pragma clang loop unroll(full)
  for (int r = 0; r < RT; r++) {
    const int row = rg * RT + r;
    nrow[r] = row < T ? NK - T + row + 1 : 0;
    mxs[r] = FMIN; sm[r] = 0.0f;
#pragma clang loop unroll(full)
    for (int j = 0; j < DPL; j++) {
      q[r][j] = row < T ? sc * float(Q[size_t(qh) * Q_strides[1] + size_t(row) * Q_strides[2] + int(lane) * DPL + j]) : 0.0f;
      o[r][j] = 0.0f;
    }
  }
  const device bfloat* kp = K + size_t(kv) * K_strides[1] + size_t(blk) * D + int(lane) * DPL;
  const device bfloat* vp = V + size_t(kv) * V_strides[1] + size_t(blk) * D + int(lane) * DPL;
  const size_t step = size_t(B) * D;
  for (int i = blk; i < NK; i += B) {
    float kk[DPL], vv[DPL];
#pragma clang loop unroll(full)
    for (int j = 0; j < DPL; j++) { kk[j] = float(kp[j]); vv[j] = float(vp[j]); }
#pragma clang loop unroll(full)
    for (int r = 0; r < RT; r++) {
      if (i < nrow[r]) {
        float score = 0.0f;
#pragma clang loop unroll(full)
        for (int j = 0; j < DPL; j++) { score += q[r][j] * kk[j]; }
        score = simd_sum(score);
        const float nm = max(mxs[r], score);
        const float factor = fast::exp(mxs[r] - nm);
        const float es = fast::exp(score - nm);
        mxs[r] = nm;
        sm[r] = sm[r] * factor + es;
#pragma clang loop unroll(full)
        for (int j = 0; j < DPL; j++) { o[r][j] = o[r][j] * factor + es * vv[j]; }
      }
    }
    kp += step; vp += step;
  }
#pragma clang loop unroll(full)
  for (int r = 0; r < RT; r++) {
    const int row = rg * RT + r;
    if (row < T) {
      const size_t slot = (size_t(qh) * T + row) * B + blk;
      if (lane == 0) { PL[slot] = sm[r]; PM[slot] = mxs[r]; }
#pragma clang loop unroll(full)
      for (int j = 0; j < DPL; j++) PO[slot * D + int(lane) * DPL + j] = bfloat(o[r][j]);
    }
  }
""".replace("__FMIN__", FMIN)

_P2 = r"""
  constexpr int D = 256;
  const uint lid = thread_index_in_simdgroup;
  const uint gid = simdgroup_index_in_threadgroup;
  const int head = int(threadgroup_position_in_grid.y);
  const int row = int(threadgroup_position_in_grid.z);
  const int T = dims[1], B = dims[2];
  constexpr int DPL = D / 32;
  const float FMIN = __FMIN__;
  float o[DPL];
  for (int i = 0; i < DPL; i++) o[i] = 0.0f;
  threadgroup float outputs[1024];
  const size_t qo = size_t(head) * T + row;
  const device bfloat* partials = PO + qo * B * D + gid * D + lid * DPL;
  const device float* sums = PL + qo * B;
  const device float* maxs = PM + qo * B;
  float sum_exp_score = 0.0f;
  float max_score = FMIN;
  for (int b = 0; b < B / 32; ++b) max_score = max(max_score, maxs[lid + 32 * b]);
  max_score = simd_max(max_score);
  for (int b = 0; b < B / 32; ++b) {
    float factor = fast::exp(maxs[lid + 32 * b] - max_score);
    sum_exp_score += factor * sums[lid + 32 * b];
  }
  sum_exp_score = simd_sum(sum_exp_score);
  for (int b = 0; b < B / 32; ++b) {
    float factor = fast::exp(maxs[gid] - max_score);
    for (int i = 0; i < DPL; i++) { o[i] += factor * float(partials[i]); }
    maxs += 32; sums += 32; partials += 32 * D;
  }
  for (int i = 0; i < DPL; i++) {
    outputs[lid * 32 + gid] = o[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    o[i] = simd_sum(outputs[gid * 32 + lid]);
    o[i] = sum_exp_score == 0 ? o[i] : (o[i] / sum_exp_score);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (lid == 0) {
    for (int i = 0; i < DPL; i++) OUT[qo * D + gid * DPL + i] = bfloat(o[i]);
  }
""".replace("__FMIN__", FMIN)

_K: list = []
_CONST: dict = {}
_DIMS: list = [None, None]


def _devc() -> str:
    info = mx.device_info() if hasattr(mx, "device_info") else mx.metal.device_info()
    return str(info["architecture"])[-1]


def _blocks(n: int) -> int:
    """Number of blocks of sdpa_vector_2pass (MLX 0.32.3, family "d", 6 simdgroups) for n keys; 0 if the family is not checked."""

    if _devc() != "d":
        return 0
    return 512 if 16384 <= n < 65536 else 1024 if n >= 65536 else 128


def _kernels():
    if not _K:
        _K.append(mx.fast.metal_kernel(name="pli_attec_p1", input_names=["Q", "K", "V", "scale", "dims", "RGA"],
                                       output_names=["PM", "PL", "PO"], source=_P1, ensure_row_contiguous=False))
        _K.append(mx.fast.metal_kernel(name="pli_attec_p2", input_names=["PM", "PL", "PO", "dims"], output_names=["OUT"], source=_P2))
    return _K


def _small(key, make):
    v = _CONST.get(key)
    if v is None:
        v = _CONST[key] = make()
    return v


def sdpa(queries: mx.array, bk: mx.array, bv: mx.array, L: int, scale: float) -> mx.array | None:
    """Attention of the T rows of ``queries`` [1, 24, T, 256] over the whole buffers ``bk``, ``bv`` [1, 4, capacity, 256]; row t sees
    L - T + t + 1 keys. Same bits as T one-row calls of ``mx.fast.scaled_dot_product_attention``. None if outside what was checked."""

    if queries.ndim != 4 or bk.ndim != 4 or queries.dtype != mx.bfloat16 or bk.dtype != mx.bfloat16 or bv.dtype != mx.bfloat16:
        return None
    if queries.shape[0] != 1 or queries.shape[1] != H or queries.shape[3] != D or bk.shape[1] != HKV or bk.shape[3] != D \
            or bv.shape[:2] != bk.shape[:2] or bv.shape[3] != D or int(bk.shape[2]) < L or int(bv.shape[2]) != int(bk.shape[2]):
        return None
    T = int(queries.shape[2])
    if T < 1 or T > MAX_ROWS or L - T + 1 < MIN_KEYS:
        return None
    b0, b1 = _blocks(L - T + 1), _blocks(L)
    if b0 == 0 or b0 != b1:
        return None                                   # regime change within the round (16384 or 65536): usual path
    RG = -(-T // RT)
    if _DIMS[0] != (L, T, b0):
        _DIMS[0], _DIMS[1] = (L, T, b0), mx.array([L, T, b0, 0, 0, 0, 0, 0], dtype=mx.int32)
    sa = _small(("s", float(scale)), lambda: mx.array([float(scale)] * 8, dtype=mx.float32))
    rga = _small(("rg", RG), lambda: mx.array([RG] * 8, dtype=mx.int32))
    k1, k2 = _kernels()
    pm, pl, po = k1(inputs=[queries, bk, bv, sa, _DIMS[1], rga], grid=(32 * G, b0, HKV * RG), threadgroup=(32 * G, 1, 1),
                    output_shapes=[(H * T * b0,), (H * T * b0,), (H * T * b0 * D,)],
                    output_dtypes=[mx.float32, mx.float32, mx.bfloat16])
    return k2(inputs=[pm, pl, po, _DIMS[1]], grid=(1024, H, T), threadgroup=(1024, 1, 1), output_shapes=[(1, H, T, D)],
              output_dtypes=[queries.dtype])[0]


__all__ = ["sdpa"]
