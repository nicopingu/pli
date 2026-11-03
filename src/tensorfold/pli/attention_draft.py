"""Attention of the DFlash2 draft (PLI_DRAFT_ATTN=1, off; 31/10/2026). CLASS V3: bits other than MLX's (the draft does not need
MLX's bits: its tokens are verified by the 27B).

Served shape: 1 to 8 query rows (the lattice's block), query heads H = G x HKV, D 128, bf16, keys = context (sliding window)
then block, non-causal: row i (position CTX + i) sees the context keys at position k with CTX + i - k < W, and all those of the block.
Measured before (dbench2, dbench3, M1 Ultra): MLX sdpa 0.27 to 0.30 ms per layer for 8 x 32 heads over 2,057 keys (8.4 MB read).

Two passes. P1: one threadgroup per (slice of SL keys, kv head); the slice is loaded into threadgroup memory; each simdgroup
takes RPS rows (query head, position); a lane holds 4 of the 128 dimensions; score = simd_sum; online softmax over the slice;
returns (m, l, O) in f32. P2: one lane per 4 dimensions, slices merged in order; output (1, L, H * D), ready for o_proj.
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx

ENABLED = os.environ.get("PLI_DRAFT_ATTN", "0") == "1"
D = 128
SL = 128          # keys per slice
NSG = 8           # simdgroups per threadgroup

_P1 = r"""
  // v7: every loop over the matrices unrolled (otherwise the matrix arrays go to memory); v6: Q K products interleaved (dd outside, 4 independent sums), loops unrolled; v4: layout by half4 and fragments by simdgroup_load; v3: 8 x 8 simdgroup matrices (f16, f32 sums): one simdgroup per tile of 8 rows (query head, position) of the kv head;
  // keys and values of a block of BK keys in threadgroup memory (stride of D + 8 half-words: no bank conflicts), online softmax
  constexpr int D = 128, BK = 32, NJ = BK / 8, DD = D / 8, PITCH = D + 8;
  constexpr int ROWS = G * L, NT = (ROWS + 7) / 8;
  threadgroup half Ks[BK * PITCH];
  threadgroup half Vs[BK * PITCH];
  const int sl = int(threadgroup_position_in_grid.x), hk = int(threadgroup_position_in_grid.y);
  const int tid = int(thread_position_in_threadgroup.x);
  const int sg = int(simdgroup_index_in_threadgroup);
  const int lane = int(thread_index_in_simdgroup);
  const int S = dims[0], CTX = dims[1], W = dims[2], NSL = dims[3];
  const float sc = scale[0];
  const int qid = lane >> 2;
  const int fm = (qid & 4) | ((lane >> 1) & 3);
  const int fn = ((qid & 2) << 1) | ((lane & 1) << 1);
  const int r = sg * 8 + fm;
  const bool rvalid = r < ROWS;
  const int h = hk * G + (rvalid ? r / L : 0), i = rvalid ? r % L : 0;
  const int qpos = CTX + i;
  simdgroup_matrix<half, 8, 8> Qf[DD];
  simdgroup_matrix<float, 8, 8> O[DD];
  const device bfloat* qp = Q + (size_t(h) * L + i) * D + fn;
#pragma clang loop unroll(full)
  for (int dd = 0; dd < DD; dd++) {
    half a = 0, b = 0;
    if (rvalid) { a = half(float(qp[dd * 8]) * sc); b = half(float(qp[dd * 8 + 1]) * sc); }
    Qf[dd].thread_elements()[0] = a; Qf[dd].thread_elements()[1] = b;
    O[dd] = simdgroup_matrix<float, 8, 8>(0.0f);
  }
  float m = -INFINITY, l = 0.0f;
  const int k0 = sl * SL, k1 = min(k0 + SL, S);
  // v8: keys and values of the cache (CTX rows) and of the block (L rows) read in their own arrays, without concatenation
  const device uint4* K4 = (const device uint4*)(K) + size_t(hk) * size_t(CTX) * (D / 8);
  const device uint4* V4 = (const device uint4*)(V) + size_t(hk) * size_t(CTX) * (D / 8);
  const device uint4* KB4 = (const device uint4*)(KB) + size_t(hk) * size_t(L) * (D / 8);
  const device uint4* VB4 = (const device uint4*)(VB) + size_t(hk) * size_t(L) * (D / 8);
  for (int kb = k0; kb < k1; kb += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int u = tid; u < BK * (D / 8); u += NT * 32) {
      const int key = u / (D / 8), c8 = u % (D / 8);
      const int kg = kb + key;
      const bool cb = kg < CTX;
      const size_t ro = size_t(cb ? kg : kg - CTX) * (D / 8) + c8;
      const uint4 wk = kg < k1 ? (cb ? K4[ro] : KB4[ro]) : uint4(0);
      const uint4 wv = kg < k1 ? (cb ? V4[ro] : VB4[ro]) : uint4(0);
      threadgroup half4* kd = (threadgroup half4*)(Ks + key * PITCH + c8 * 8);
      threadgroup half4* vd = (threadgroup half4*)(Vs + key * PITCH + c8 * 8);
      kd[0] = half4(float4(as_type<float>(wk.x << 16), as_type<float>(wk.x & 0xFFFF0000u), as_type<float>(wk.y << 16), as_type<float>(wk.y & 0xFFFF0000u)));
      kd[1] = half4(float4(as_type<float>(wk.z << 16), as_type<float>(wk.z & 0xFFFF0000u), as_type<float>(wk.w << 16), as_type<float>(wk.w & 0xFFFF0000u)));
      vd[0] = half4(float4(as_type<float>(wv.x << 16), as_type<float>(wv.x & 0xFFFF0000u), as_type<float>(wv.y << 16), as_type<float>(wv.y & 0xFFFF0000u)));
      vd[1] = half4(float4(as_type<float>(wv.z << 16), as_type<float>(wv.z & 0xFFFF0000u), as_type<float>(wv.w << 16), as_type<float>(wv.w & 0xFFFF0000u)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    simdgroup_matrix<float, 8, 8> Sm[NJ];
#pragma clang loop unroll(full)
    for (int jt = 0; jt < NJ; jt++) Sm[jt] = simdgroup_matrix<float, 8, 8>(0.0f);
#pragma clang loop unroll(full)
    for (int dd = 0; dd < DD; dd++) {
#pragma clang loop unroll(full)
      for (int jt = 0; jt < NJ; jt++) {
        simdgroup_matrix<half, 8, 8> kf;
        simdgroup_load(kf, Ks + (jt * 8) * PITCH + dd * 8, PITCH, ulong2(0, 0), true);
        simdgroup_multiply_accumulate(Sm[jt], Qf[dd], kf, Sm[jt]);
      }
    }
    float sv[NJ][2];
    float mb = -INFINITY;
#pragma clang loop unroll(full)
    for (int jt = 0; jt < NJ; jt++) {
#pragma clang loop unroll(full)
      for (int e = 0; e < 2; e++) {
        const int kg = kb + jt * 8 + fn + e;
        const bool vis = rvalid && kg < k1 && (kg >= CTX || qpos - kg < W);
        const float x = vis ? Sm[jt].thread_elements()[e] : -INFINITY;
        sv[jt][e] = x;
        mb = max(mb, x);
      }
    }
    mb = max(mb, simd_shuffle_xor(mb, ushort(1)));
    mb = max(mb, simd_shuffle_xor(mb, ushort(8)));
    const float mn = max(m, mb);
    const float c = (mn == -INFINITY) ? 1.0f : metal::exp(m - mn);
    float ps = 0.0f;
    half pv[NJ][2];
#pragma clang loop unroll(full)
    for (int jt = 0; jt < NJ; jt++) {
#pragma clang loop unroll(full)
      for (int e = 0; e < 2; e++) {
        const float p = (sv[jt][e] == -INFINITY) ? 0.0f : metal::exp(sv[jt][e] - mn);
        ps += p; pv[jt][e] = half(p);
      }
    }
    ps += simd_shuffle_xor(ps, ushort(1));
    ps += simd_shuffle_xor(ps, ushort(8));
    l = l * c + ps;
    m = mn;
#pragma clang loop unroll(full)
    for (int dd = 0; dd < DD; dd++) {
      O[dd].thread_elements()[0] *= c; O[dd].thread_elements()[1] *= c;
    }
#pragma clang loop unroll(full)
    for (int jt = 0; jt < NJ; jt++) {
      simdgroup_matrix<half, 8, 8> P;
      P.thread_elements()[0] = pv[jt][0]; P.thread_elements()[1] = pv[jt][1];
#pragma clang loop unroll(full)
      for (int dd = 0; dd < DD; dd++) {
        simdgroup_matrix<half, 8, 8> vf;
        simdgroup_load(vf, Vs + (jt * 8) * PITCH + dd * 8, PITCH);
        simdgroup_multiply_accumulate(O[dd], P, vf, O[dd]);
      }
    }
  }
  if (rvalid) {
    const size_t at = (size_t(hk) * ROWS + r) * NSL + sl;
    if ((lane & 9) == 0) { PM[at] = m; PL[at] = l; }
    device float* po = PO + at * D + fn;
#pragma clang loop unroll(full)
    for (int dd = 0; dd < DD; dd++) { po[dd * 8] = O[dd].thread_elements()[0]; po[dd * 8 + 1] = O[dd].thread_elements()[1]; }
  }
"""

_P2 = r"""
  constexpr int D = 128;
  const int row = int(threadgroup_position_in_grid.x);     // h * L + i
  const int lane = int(thread_index_in_simdgroup);
  const int NSL = dims[3];
  float M = -INFINITY;
  for (int s = 0; s < NSL; s++) M = max(M, PM[row * NSL + s]);
  float lsum = 0.0f;
  float4 o = float4(0.0f);
  for (int s = 0; s < NSL; s++) {
    const size_t at = size_t(row) * NSL + s;
    const float w = (PL[at] > 0.0f) ? metal::exp(PM[at] - M) : 0.0f;
    lsum += w * PL[at];
    const device float* po = PO + at * D + lane * 4;
    o += w * float4(po[0], po[1], po[2], po[3]);
  }
  o /= lsum;
  const int h = row / L, i = row % L;
  device bfloat* out = OUT + (size_t(i) * H + h) * D + lane * 4;
  out[0] = bfloat(o.x); out[1] = bfloat(o.y); out[2] = bfloat(o.z); out[3] = bfloat(o.w);
"""

_kernels: dict[tuple, Any] = {}


def _kernel(name: str, G: int, L: int) -> Any:
    """The kernel for (G, L), constants in the source (no template: threads.reserve bounds the pipeline's threads; otherwise the M1
    cuts the registers and the kernel spills to memory)."""

    from tensorfold.kernels import threads

    key = (name, G, L, SL)
    k = _kernels.get(key)
    if k is None:
        if name == "p1":
            nt = -(-(G * L) // 8)
            src = f"  constexpr int G = {G}, L = {L}, SL = {SL};\n" + _P1
            k = mx.fast.metal_kernel(name=f"pli_draft_attn_p1b_g{G}_l{L}_s{SL}", input_names=["Q", "K", "V", "KB", "VB", "dims", "scale"],
                                     output_names=["PM", "PL", "PO"], source=src, header=threads.reserve(nt * 32))
        _kernels[key] = k
    return k


def _kernel_p2(H: int, L: int) -> Any:
    from tensorfold.kernels import threads

    key = ("p2", H, L)
    k = _kernels.get(key)
    if k is None:
        src = f"  constexpr int H = {H}, L = {L};\n" + _P2
        k = _kernels[key] = mx.fast.metal_kernel(name=f"pli_draft_attn_p2_h{H}_l{L}", input_names=["PM", "PL", "PO", "dims"],
                                                 output_names=["OUT"], source=src, header=threads.reserve(32))
    return k


def sdpa(queries: mx.array, keys: mx.array, values: mx.array, ctx_len: int, window: int, scale: float,
         block_keys: mx.array | None = None, block_values: mx.array | None = None) -> mx.array | None:
    """queries (1, H, L, D); keys and values: (1, HKV, ctx_len + L, D), or those of the cache (1, HKV, ctx_len, D) and of the block
    (block_keys, block_values: (1, HKV, L, D)); bf16 -> (1, L, H * D) bf16, or None outside the served shape."""

    B, H, L, d = (int(s) for s in queries.shape)
    _, HKV, S, _ = (int(s) for s in keys.shape)
    if block_keys is None:
        if S != ctx_len + L:
            return None
        block_keys = keys[:, :, ctx_len:]
        block_values = values[:, :, ctx_len:]
        keys, values = keys[:, :, :ctx_len], values[:, :, :ctx_len]
    elif S != ctx_len or tuple(block_keys.shape) != (1, HKV, L, D) or tuple(block_values.shape) != (1, HKV, L, D):
        return None
    if ctx_len == 0:                      # nothing in the cache: the kernel reads only the block
        keys, values = block_keys, block_values
    S = ctx_len + L
    if (B != 1 or d != D or not 1 <= L <= 8 or H % HKV or queries.dtype != mx.bfloat16 or keys.dtype != mx.bfloat16
            or values.dtype != mx.bfloat16 or block_keys.dtype != mx.bfloat16 or block_values.dtype != mx.bfloat16
            or tuple(values.shape) != tuple(keys.shape)):
        return None
    G = H // HKV
    NT = -(-(G * L) // 8)
    NSL = -(-S // SL)
    dims = mx.array([S, int(ctx_len), int(window), NSL], dtype=mx.int32)
    sc = mx.array([float(scale)], dtype=mx.float32)
    pm, pl, po = _kernel("p1", G, L)(inputs=[queries, keys, values, block_keys, block_values, dims, sc],
                                     grid=(NSL * NT * 32, HKV, 1), threadgroup=(NT * 32, 1, 1),
                                     output_shapes=[(H * L * NSL,), (H * L * NSL,), (H * L * NSL * D,)],
                                     output_dtypes=[mx.float32, mx.float32, mx.float32])
    (out,) = _kernel_p2(H, L)(inputs=[pm, pl, po, dims], grid=(H * L * 32, 1, 1),
                              threadgroup=(32, 1, 1), output_shapes=[(1, L, H * D)], output_dtypes=[mx.bfloat16])
    return out
