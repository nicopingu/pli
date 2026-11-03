"""PLI (28/10/2026, PLI_V8=1, off by default): Kapitsa's 8-row verify GEMM for 4-bit, group-64 weights.

Ported from nace-ai/kapitsa v0.2.23 (Apache-2.0), src/kernels/verify8.metal: the X8 input table (x8_prep) and the dense
4-bit GEMM qmm8snb_b4_nf2 (codes as exact half subnormals, two 64-groups per step), with Kapitsa's M1 tiling (kplan 3,
src/model/qwen35_v8.cpp v8_plan). Changes for PLI: an MLX metal_kernel body, a variant that reads PLI's "rg" weight
layout (pli/relayout.py), row tiles of 8 on grid y for up to 128 rows, bf16 outputs.

Every row is computed from its own column of the 8x8 MMAs and the zero padding of the tile, so a row's bits do not depend
on the other rows or on the row count (row-exact, as the lane decoder requires); they differ from simd_qmm's (V3).
Measured on the M1 Ultra (28/10, 8 rows, 27B shapes): the projections of one round 44.7 ms instead of 53.6 (mmaf_rg).
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx

ENABLED = os.environ.get("PLI_V8", "0") == "1"
MAX_ROWS = 128           # rows a call takes here (the lane decoder's windows); longer calls keep simd_qmm
GROUP = 64
fallback: set[tuple[int, int]] = set()   # (N, K) shapes whose check failed: simd_qmm keeps them

_HEADER = r"""
#define NI_INLINE inline __attribute__((always_inline))
constant constexpr uint kV8 = 8;
constant constexpr float kV8Sub = 16777216.0f;
NI_INLINE uint2 v8_lane(uint lane) {
  const uint q = lane >> 2;
  return uint2((q & 4) | ((lane >> 1) & 3), ((q & 2) << 1) | ((lane & 1) << 1));
}
NI_INLINE uint v8_lane_of(uint fm, uint fn) {
  return ((fm >> 2) << 4) | ((fn >> 2) << 3) | ((fm & 3) << 1) | ((fn >> 1) & 1);
}
NI_INLINE uint2 v8_fj(uint kk) { return uint2(4 * ((kk >> 3) & 1) + (kk & 3), 2 * (kk >> 4) + ((kk >> 2) & 1)); }
NI_INLINE uint v8_tab(uint g, uint kk, uint t) {
  const uint2 fj = v8_fj(kk);
  return (g * 32 + v8_lane_of(fj.y, t & 6u)) * 16 + 2 * fj.x + (t & 1u);
}
NI_INLINE void v8_mma(thread float2& c, half2 a, half2 b) {
  simdgroup_half8x8 A, B;
  simdgroup_float8x8 C;
  A.thread_elements()[0] = a.x;
  A.thread_elements()[1] = a.y;
  B.thread_elements()[0] = b.x;
  B.thread_elements()[1] = b.y;
  C.thread_elements()[0] = c.x;
  C.thread_elements()[1] = c.y;
  simdgroup_multiply_accumulate(C, A, B, C);
  c = float2(C.thread_elements()[0], C.thread_elements()[1]);
}
NI_INLINE half2 v8_codes_sn4(uint2 w, int f) {
  const uint word = f < 4 ? w.x : w.y;
  return as_type<half2>((word >> (4 * (f & 3))) & 0x000F000Fu);
}
NI_INLINE float v8_bf(uint b) { return as_type<float>((b & 0xffffu) << 16); }
// One 64-group of NF fragments: codes w, scale s and bias b (bf16 bits), this lane's table words t0, t1 and group sums sx.
template <int NF>
NI_INLINE void v8_group(thread const uint2* w, thread const uint* s, thread const uint* b, uint4 t0, uint4 t1, float2 sx,
                        thread float2* acc) {
  half2 x[8];
  _Pragma("clang loop unroll(full)")
  for (int j = 0; j < 4; ++j) {
    x[j] = as_type<half2>(t0[j]);
    x[j + 4] = as_type<half2>(t1[j]);
  }
  _Pragma("clang loop unroll(full)")
  for (int f = 0; f < NF; ++f) {
    float2 dot = float2(0.0f);
    _Pragma("clang loop unroll(full)")
    for (int j = 0; j < 8; ++j) v8_mma(dot, v8_codes_sn4(w[f], j), x[j]);
    acc[f] = fma(dot, float2(v8_bf(s[f]) * kV8Sub), acc[f]);
    acc[f] = fma(sx, float2(v8_bf(b[f])), acc[f]);
  }
}
"""

# Weights, scale and bias of group g for fragment f, as plain words: two layouts.
_LOAD_MLX = r"""
#define V8_LOAD(f, g, W, S, B) { W = *(const device uint2*)(wp[f] + (g) * 8); S = sp[f][g]; B = bp[f][g]; }
"""
_LOAD_RG = r"""
#define V8_LOAD(f, g, W, S, B) { W = *(const device uint2*)(wp[f] + (g) * 256); const uint wd_ = sbp[f][(g) * 32]; \
                                 S = wd_ & 0xffffu; B = wd_ >> 16; }
"""

_BODY = r"""
  const uint tg = threadgroup_position_in_grid.x, tile = threadgroup_position_in_grid.y;
  const uint R = rr[0];
  const uint sg = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  threadgroup float2 red[8 * NF * 32];
  const uint kslice = sg % KS;
  const uint n0 = (tg * CS + sg / KS) * 8 * NF;
  const uint gpr = K / 64;
  const uint2 rc = v8_lane(lane);
  const uint fm = rc.x, fn = rc.y, q = fn / 2;
  const uint gper = (gpr + KS - 1) / KS, g0 = min(kslice * gper, gpr), g1 = min(g0 + gper, gpr);
  const device half* tab = table + tile * (gpr * 512);
  const device float* sms = sums + tile * (gpr * 8);
  PTRS
  float2 acc[NF];
  for (int f = 0; f < NF; ++f) {
    const uint row = min(n0 + 8 * f + fm, uint(N) - 1);
    SETUP
    acc[f] = float2(0.0f);
  }
  for (uint g = g0; g < g1; ++g) {
    const device uint4* tb = (const device uint4*)(tab + (g * 32 + lane) * 16);
    const uint4 t0 = tb[0], t1 = tb[1];
    const float2 sx = *(const device float2*)(sms + g * kV8 + fn);
    uint2 w[NF];
    uint s[NF], b[NF];
    for (int f = 0; f < NF; ++f) V8_LOAD(f, g, w[f], s[f], b[f]);
    v8_group<NF>(w, s, b, t0, t1, sx, acc);
  }
  if (KS > 1) {
    if (kslice)
      for (int f = 0; f < NF; ++f) red[(sg * NF + f) * 32 + lane] = acc[f];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kslice) return;
    for (uint j = 1; j < KS; ++j)
      for (int f = 0; f < NF; ++f) acc[f] += red[((sg + j) * NF + f) * 32 + lane];
  }
  for (int f = 0; f < NF; ++f) {
    const uint n = n0 + 8 * f + fm;
    if (n >= uint(N)) continue;
    for (int e = 0; e < 2; ++e) {
      const uint t = 8 * tile + fn + e;
      if (t < uint(R)) y[t * N + n] = bfloat16_t(acc[f][e]);
    }
  }
"""

_PTRS_MLX = "const device uint32_t* wp[NF]; const device ushort* sp[NF]; const device ushort* bp[NF];"
_SETUP_MLX = ("wp[f] = w + row * (K / 8) + q * 2; sp[f] = (const device ushort*)scales + row * gpr; "
              "bp[f] = (const device ushort*)biases + row * gpr;")
_PTRS_RG = "const device uint32_t* wp[NF]; const device uint32_t* sbp[NF];"
_SETUP_RG = ("const uint p = row >> 5, t4 = (row >> 3) & 3u, fr = row & 7u; "
             "wp[f] = w + (((p * gpr) * 32 + fr * 4 + q) * 4 + t4) * 2; sbp[f] = sb + ((p * gpr) * 8 + fr) * 4 + t4;")

_PREP = r"""
  const uint t = threadgroup_position_in_grid.x % 8, tile = threadgroup_position_in_grid.x / 8;
  const uint R = rr[0];
  const uint sg = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint row = 8 * tile + t;
  const bool live = row < uint(R);
  const uint G = K / 64;
  device half* tab = table + tile * (G * 512);
  device float* sms = sums + tile * (G * 8);
  for (uint g = sg; g < G; g += 8) {
    const uint kk = 2 * lane;
    const float2 v = live ? float2(float(x[row * K + g * 64 + kk]), float(x[row * K + g * 64 + kk + 1])) : float2(0.0f);
    const half2 h = half2(v);
    tab[v8_tab(g, kk, t)] = h.x;
    tab[v8_tab(g, kk + 1, t)] = h.y;
    const float sm = simd_sum(float(h.x) + float(h.y));
    if (lane == 0) sms[g * kV8 + t] = sm;
  }
"""

# PLI (31/10/2026, fbench: preparing the tables costs 12 to 26 us per product, 4.3 ms per 8-row round, with only 8 threadgroups):
# PLI_V8_PREP2=1 (off) does the same computation per (row, group of 64) as _PREP, each group on one simdgroup as before,
# but spread over ceil(G / 8) threadgroups per row instead of one: same values, same sums (same simd_sum), same addresses.
PREP2 = os.environ.get("PLI_V8_PREP2", "0") == "1"
_PREP2 = r"""
  const uint NGB = (K / 64 + 7) / 8;
  const uint i = threadgroup_position_in_grid.x;
  const uint t = i % 8, gb = (i / 8) % NGB, tile = (i / 8) / NGB;
  const uint R = rr[0];
  const uint sg = simdgroup_index_in_threadgroup;
  const uint lane = thread_index_in_simdgroup;
  const uint row = 8 * tile + t;
  const bool live = row < uint(R);
  const uint G = K / 64;
  const uint g = gb * 8 + sg;
  device half* tab = table + tile * (G * 512);
  device float* sms = sums + tile * (G * 8);
  if (g < G) {
    const uint kk = 2 * lane;
    const float2 v = live ? float2(float(x[row * K + g * 64 + kk]), float(x[row * K + g * 64 + kk + 1])) : float2(0.0f);
    const half2 h = half2(v);
    tab[v8_tab(g, kk, t)] = h.x;
    tab[v8_tab(g, kk + 1, t)] = h.y;
    const float sm = simd_sum(float(h.x) + float(h.y));
    if (lane == 0) sms[g * kV8 + t] = sm;
  }
"""

_kernels: dict[str, Any] = {}


def _kernel(kind: str) -> Any:
    k = _kernels.get(kind)
    if k is None:
        if kind == "prep":
            k = mx.fast.metal_kernel(name="pli_v8_prep", input_names=["x", "rr"], output_names=["table", "sums"],
                                     source=_PREP, header=_HEADER)
        elif kind == "prep2":
            k = mx.fast.metal_kernel(name="pli_v8_prep2", input_names=["x", "rr"], output_names=["table", "sums"],
                                     source=_PREP2, header=_HEADER)
        elif kind == "mlx":
            k = mx.fast.metal_kernel(name="pli_v8_qmm", input_names=["w", "scales", "biases", "table", "sums", "rr"],
                                     output_names=["y"], header=_HEADER + _LOAD_MLX,
                                     source=_BODY.replace("PTRS", _PTRS_MLX).replace("SETUP", _SETUP_MLX))
        else:
            k = mx.fast.metal_kernel(name="pli_v8_qmm_rg", input_names=["w", "sb", "table", "sums", "rr"], output_names=["y"],
                                     header=_HEADER + _LOAD_RG,
                                     source=_BODY.replace("PTRS", _PTRS_RG).replace("SETUP", _SETUP_RG))
        _kernels[kind] = k
    return k


def plan(N: int, K: int, nf: int = 2) -> tuple[int, int, int, int]:
    """Kapitsa's v8_plan with kplan 3 (its M1 setting): nf fragments, cs column blocks, ks k-slices, threadgroups."""

    blocks = -(-N // (8 * nf))
    gpr = K // 64
    ks = 1
    while ks < 8 and gpr // (ks * 2) >= 16:
        ks *= 2
    while ks > 1 and blocks * ks > 16384:
        ks //= 2
    cs = 2 if ks == 1 else 1
    cs = max(1, min(cs, 8 // ks))
    return nf, cs, ks, -(-blocks // cs)


def fits(weight: mx.array, scales: mx.array, biases: mx.array, group_size: int, bits: int = 4) -> bool:
    n, k = int(weight.shape[0]), int(weight.shape[1]) * 8
    if bits != 4 or group_size != GROUP or k % GROUP or (n, k) in fallback:
        return False
    if scales.dtype == mx.uint32:                       # rg layout: packs of 32 outputs
        return n % 32 == 0
    return scales.dtype == mx.bfloat16 and biases.dtype == mx.bfloat16 and n % 8 == 0


def qmm(x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array) -> mx.array:
    """x [R, K] (R <= MAX_ROWS) @ W.T as bf16 [R, N], row-exact in R."""

    rows, K = int(x.shape[0]), int(x.shape[1])
    N = int(weight.shape[0])
    tiles = -(-rows // 8)
    G = K // GROUP
    rr = mx.array([rows], dtype=mx.uint32)                # rows at run time: one compiled kernel per shape
    if PREP2:
        table, sums = _kernel("prep2")(inputs=[x, rr], template=[("K", K)], grid=(8 * ((G + 7) // 8) * tiles * 256, 1, 1),
                                       threadgroup=(256, 1, 1), output_shapes=[(tiles * G * 512,), (tiles * G * 8,)],
                                       output_dtypes=[mx.float16, mx.float32])
    else:
        table, sums = _kernel("prep")(inputs=[x, rr], template=[("K", K)], grid=(8 * tiles * 256, 1, 1),
                                      threadgroup=(256, 1, 1), output_shapes=[(tiles * G * 512,), (tiles * G * 8,)],
                                      output_dtypes=[mx.float16, mx.float32])
    nf, cs, ks, ntg = plan(N, K)
    rg = scales.dtype == mx.uint32
    inputs = [weight, scales, table, sums, rr] if rg else [weight, scales, biases, table, sums, rr]
    (y,) = _kernel("rg" if rg else "mlx")(
        inputs=inputs, template=[("NF", nf), ("CS", cs), ("KS", ks), ("K", K), ("N", N)],
        grid=(ntg * 32 * cs * ks, tiles, 1), threadgroup=(32 * cs * ks, 1, 1), output_shapes=[(rows, N)],
        output_dtypes=[mx.bfloat16])
    return y


def check(weight: mx.array, scales: mx.array, biases: mx.array, *, seed: int = 0) -> bool:
    """Row-exactness at load: a row alone, in 8 rows and in 24 rows (3 tiles) gives the same bits; else the shape falls back."""

    K = int(weight.shape[1]) * 8
    key = mx.random.key(seed)
    x = (mx.random.normal((24, K), key=key) * 0.5).astype(mx.bfloat16)
    one, eight, many = qmm(x[:1], weight, scales, biases), qmm(x[:8], weight, scales, biases), qmm(x, weight, scales, biases)
    mx.eval(one, eight, many)
    ok = bool(mx.all(one[0] == eight[0])) and bool(mx.all(eight == many[:8])) and bool(mx.all(mx.isfinite(many.astype(mx.float32))))
    if not ok:
        fallback.add((int(weight.shape[0]), K))
    return ok
