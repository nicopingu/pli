"""Custom DECODE attention of the 27B, by tiles of 8 x 8 simdgroup matrices (pli-attm, 13/10/2026). CLASS V3: the bits are NOT those of MLX.

Shape served: 1 to 32 query rows (1, 2 and 17 to 32 rows since pli-attmd, 13/10/2026: a row has the same bits whatever the round; 32 = widest decode window of the engine), 24 query heads, 4 kv heads, D 256, bf16, cache = a larger buffer whose start the round reads;
row t of a round of T rows sees L - T + t + 1 keys. ``sdpa`` returns None outside what was measured: the caller then takes the usual path
(attention_decode if on, else MLX).

Design (measured on M1 Ultra):
 - one kv head = 6 query heads x T rows, laid out r = 6 t + g, in tiles of 8 rows (one tile = a set of 2 simdgroups); one threadgroup
   carries all tiles of the kv head (up to 6 tiles = 12 simdgroups, 2 groups at 16 rows): the keys and values of a block of 16 keys are
   read ONCE, converted to f16 (exact for bf16 between 6.1e-5 and 65504 in absolute value; past 65504: infinity, see PLI_F16_CLAMP) and laid out in threadgroup memory (columns permuted by xor: no bank conflicts);
 - S = Q Kt by 8 x 8 f16 matrices with f32 accumulation (each simdgroup takes half of the 256 dimensions, the two halves are added in order through
   threadgroup memory), online softmax per block of 16 keys (exp2), P in f16, O += P V (f32 accumulation, each simdgroup keeps half of the columns);
 - keys cut into slices of SLEN keys WITH FIXED BOUNDS (absolute position of the key): a row depends neither on T, nor on L, nor on its tile's place
   (measured: same bits of a row between rounds of 1 to 8 rows and of different lengths); each slice returns (m, l, O) in f32, merged
   in slice order by a second pass (bf16 output).
Measured (33,000 keys, 8 rows, M1 Ultra): 1.24 ms per layer against 3.00 for attention_decode; error against an f32 reference smaller than MLX's.
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx

# PLI (13/10/2026): PLI_F16_CLAMP=1 bounds the values of Q, K and V to ±65504 before their conversion to f16 (beyond, f16 gives infinity then NaN
# in the softmax). Identity for any value f16 already represents: same bits as long as no value overflows. 0 = original.
CLAMP = os.environ.get("PLI_F16_CLAMP", "0") == "1"

D, H, HKV, G = 256, 24, 4, 6
BK = 16                                          # keys per block (part of the arithmetic)
NS = 2                                           # simdgroups per tile of 8 rows (part of the arithmetic: order of the sum of S)
SLEN = 512                                       # keys per slice (part of the arithmetic: fixed bounds by absolute position)
# PLI (21/10/2026): PLI_ATTM_SMALL_KEYS=S1 lays out the first S1 keys in slices of SMALL keys (128): at 3,300 and 6,000 keys, 16 layers
# go from 7.5 to 3.9 and 5.0 ms (the GPU had only 28 to 48 threadgroups for 64 cores). Bits other than SLEN alone (V3); same bits between 1 and 8 rows.
SMALL_KEYS = max(0, int(os.environ.get("PLI_ATTM_SMALL_KEYS", "0") or 0))
SMALL = 128
# PLI (21/10/2026, measured: slices of 128 mixed with those of 512 cost 0.4 ms more per layer at 22K):
# PLI_ATTM_SHORT=N: slices of SMALL keys for any call of N keys at most (short context, and arrays of the reduced prefill past 20K),
# of SLEN beyond. V3: a position read under N keys in one round and over in another does not have the same bits (only around N).
SHORT = max(0, int(os.environ.get("PLI_ATTM_SHORT", "0") or 0))
MIN_KEYS = 1   # 13/10: every decode row goes through attm, the kernel choice never depends on the round
MIN_ROWS = 1
MAX_ROWS = 32
# PLI (30/10/2026, layer profile: the attention core carries the cost of one more row): PLI_ATTM_LOAD=1 lays out the block's keys and
# values without permutation, with a stride of LDK = 264 half-words per key, and loads the 8 x 8 fragments with simdgroup_load (instead of
# two reads per thread at permuted addresses). Same products in the same order: same bits as the original (checked on two Macs).
LOAD = os.environ.get("PLI_ATTM_LOAD", "0") == "1"
LDK = 264
TPG_MAX = 6                                      # tiles of 8 rows per threadgroup
TPG_LOAD: dict = {1: 3, 2: 3}                            # number of 8-row tiles -> tiles of the threadgroup if more than needed: the extra tiles do not compute
                                                 # (active false) but load the keys and values with the others: same operations per row, same bits (measured: 1 row 1.089 -> 0.549 ms at 33,000 keys with 3 tiles)

_HEADER = r"""
inline float bfx(uint4 v, int e) {
  const uint w = v[e >> 1];
  return as_type<float>((e & 1) ? (w & 0xFFFF0000u) : (w << 16));
}
inline half h16(float x) { return half(clamp(x, -65504.0f, 65504.0f)); }
"""

_P1 = r"""
  constexpr int D = 256, G = 6, NJ = BK / 8, NTH = 32 * NSG, TPG = NSG / NS, DH = 32 / NS, NLD = (BK * 32 + NTH - 1) / NTH;
  const float FMIN = -3.4028234663852886e+38f;
  const float LOG2E = 1.4426950408889634f;
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int tid = int(thread_position_in_threadgroup.x);
  const int sl = int(threadgroup_position_in_grid.y);
  const int z = int(threadgroup_position_in_grid.z);
  const int kv = z / RGN, rg = z % RGN;
  const int NK = dims[0], T = dims[1], NSL = dims[2], SLEN = dims[3];
  const float c2 = scale[0] * LOG2E;
  const int qid = int(lane) >> 2;
  const int fm = (qid & 4) | ((int(lane) >> 1) & 3);
  const int fn = ((qid & 2) << 1) | ((int(lane) & 1) << 1);
  const int tl = sg / NS, ds = sg % NS;
  const int tile = rg * TPG + tl;
  const int r = tile * 8 + fm;
  const bool rvalid = r < G * T;
  const int rt = rvalid ? r / G : 0;
  const int qh = kv * G + (rvalid ? r % G : 0);
  const int nrow = rvalid ? NK - T + rt + 1 : 0;
  const bool active = tile * 8 < G * T;
  const int d0 = ds * DH;
  const device bfloat* qpb = Q + size_t(qh) * Q_strides[1] + size_t(rt) * Q_strides[2] + fn;

  simdgroup_matrix<half, 8, 8> Qf[DH];
  simdgroup_matrix<float, 8, 8> O[DH];
#pragma clang loop unroll(full)
  for (int j = 0; j < DH; j++) {
    half a = 0, b = 0; if (rvalid) { a = half(float(qpb[(d0 + j) * 8])); b = half(float(qpb[(d0 + j) * 8 + 1])); }
Qf[j].thread_elements()[0] = a; Qf[j].thread_elements()[1] = b;
    O[j] = simdgroup_matrix<float, 8, 8>(0.0f);
  }
  float m = FMIN, l = 0.0f;

  threadgroup half KV[NBUF * 2 * BK * 256];
  threadgroup float SX[NS > 1 ? NBUF * TPG * NS * NJ * 64 : 1];
  int cur = 0;
  // PLI (21/10/2026, PLI_ATTM_SMALL_KEYS): the first S1 keys in slices of SP keys (fixed bounds by absolute position, like SLEN):
  // more threadgroups when the context is short; S1 = SP = 0: all slices of SLEN (original, same bits).
  const int S1 = dims[4], SP = dims[5];
  // the long slices (keys past S1) first: the most loaded threadgroups start first
  const int NB = (SP > 0 && NK > S1) ? (NK - S1 + SLEN - 1) / SLEN : 0;
  const bool long_slice = SP == 0 || sl < NB;
  const int kbeg = SP == 0 ? sl * SLEN : (long_slice ? S1 + sl * SLEN : (sl - NB) * SP);
  const int kend = SP == 0 ? min(kbeg + SLEN, NK) : (long_slice ? min(kbeg + SLEN, NK) : min(kbeg + SP, min(S1, NK)));
  const size_t kb0 = size_t(kv) * K_strides[1], vb0 = size_t(kv) * V_strides[1];
  const size_t ks2 = K_strides[2], vs2 = V_strides[2];
  uint4 rk[NLD], rv[NLD];
  
#pragma clang loop unroll(full)
    for (int u = 0; u < NLD; u++) { const int i = tid + u * NTH; if (i < BK * 32) {
      const int kk = i / 32, c8 = i % 32; const int key = min((kbeg) + kk, NK - 1);
      rk[u] = *(const device uint4*)(K + kb0 + size_t(key) * ks2 + c8 * 8);
      rv[u] = *(const device uint4*)(V + vb0 + size_t(key) * vs2 + c8 * 8); } }

  for (int kt = kbeg; kt < kend; kt += BK) {
    threadgroup_barrier(mem_flags::mem_threadgroup);

#pragma clang loop unroll(full)
    for (int u = 0; u < NLD; u++) { const int i = tid + u * NTH; if (i < BK * 32) {
      const int kk = i / 32, c8 = i % 32; const int pc = c8 ^ (kk & 7);
      {
        half4 h0 = half4(half(bfx(rk[u], 0)), half(bfx(rk[u], 1)), half(bfx(rk[u], 2)), half(bfx(rk[u], 3)));
        half4 h1 = half4(half(bfx(rk[u], 4)), half(bfx(rk[u], 5)), half(bfx(rk[u], 6)), half(bfx(rk[u], 7)));
        *(threadgroup half4*)(&(KV)[kk * 256 + pc * 8]) = h0; *(threadgroup half4*)(&(KV)[kk * 256 + pc * 8 + 4]) = h1;
      }
      {
        half4 h0 = half4(half(bfx(rv[u], 0)), half(bfx(rv[u], 1)), half(bfx(rv[u], 2)), half(bfx(rv[u], 3)));
        half4 h1 = half4(half(bfx(rv[u], 4)), half(bfx(rv[u], 5)), half(bfx(rv[u], 6)), half(bfx(rv[u], 7)));
        *(threadgroup half4*)(&(KV)[BK * 256 + kk * 256 + pc * 8]) = h0; *(threadgroup half4*)(&(KV)[BK * 256 + kk * 256 + pc * 8 + 4]) = h1;
      }
    } }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (kt + BK < kend) {
#pragma clang loop unroll(full)
    for (int u = 0; u < NLD; u++) { const int i = tid + u * NTH; if (i < BK * 32) {
      const int kk = i / 32, c8 = i % 32; const int key = min((kt + BK) + kk, NK - 1);
      rk[u] = *(const device uint4*)(K + kb0 + size_t(key) * ks2 + c8 * 8);
      rv[u] = *(const device uint4*)(V + vb0 + size_t(key) * vs2 + c8 * 8); } }
}

    {
    threadgroup float* SXc = SX + ((NBUF == 2 ? cur : 0) * (TPG * NS * NJ * 64));
    threadgroup half* Ks = (KV);
    threadgroup half* Vs = (KV) + BK * 256;
    simdgroup_matrix<float, 8, 8> S[NJ][NACC];
#pragma clang loop unroll(full)
    for (int jt = 0; jt < NJ; jt++) {
#pragma clang loop unroll(full)
      for (int a = 0; a < NACC; a++) S[jt][a] = simdgroup_matrix<float, 8, 8>(0.0f);
    }
    if (active) {
#pragma clang loop unroll(full)
      for (int dd = 0; dd < DH; dd++) {
        
#pragma clang loop unroll(full)
        for (int jt = 0; jt < NJ; jt++) {
          simdgroup_matrix<half, 8, 8> kf;
          const int ka = jt * 8 + fn;
          const int dj = d0 + dd;
          kf.thread_elements()[0] = Ks[ka * 256 + (dj ^ (ka & 7)) * 8 + fm];
          kf.thread_elements()[1] = Ks[(ka + 1) * 256 + (dj ^ ((ka + 1) & 7)) * 8 + fm];
          simdgroup_multiply_accumulate(S[jt][dd % NACC], Qf[dd], kf, S[jt][dd % NACC]);
        }
      }
    }
    float sa[NJ][2];
#pragma clang loop unroll(full)
    for (int jt = 0; jt < NJ; jt++) {
      sa[jt][0] = S[jt][0].thread_elements()[0]; sa[jt][1] = S[jt][0].thread_elements()[1];
#pragma clang loop unroll(full)
      for (int a = 1; a < NACC; a++) { sa[jt][0] += S[jt][a].thread_elements()[0]; sa[jt][1] += S[jt][a].thread_elements()[1]; }
    }
    if (NS > 1) {
      if (active) {
#pragma clang loop unroll(full)
        for (int jt = 0; jt < NJ; jt++) {
          SXc[((tl * NS + ds) * NJ + jt) * 64 + int(lane) * 2] = sa[jt][0];
          SXc[((tl * NS + ds) * NJ + jt) * 64 + int(lane) * 2 + 1] = sa[jt][1];
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (active) {
      const bool full = (kt + BK) <= (NK - T + 1);
      float s[NJ][2];
      float x = FMIN;
#pragma clang loop unroll(full)
      for (int jt = 0; jt < NJ; jt++) {
        float a, b;
        if (NS > 1) {
          a = SXc[((tl * NS) * NJ + jt) * 64 + int(lane) * 2]; b = SXc[((tl * NS) * NJ + jt) * 64 + int(lane) * 2 + 1];
#pragma clang loop unroll(full)
          for (int w = 1; w < NS; w++) {
            a += SXc[((tl * NS + w) * NJ + jt) * 64 + int(lane) * 2]; b += SXc[((tl * NS + w) * NJ + jt) * 64 + int(lane) * 2 + 1];
          }
        } else { a = sa[jt][0]; b = sa[jt][1]; }
        if (full) { s[jt][0] = a * c2; s[jt][1] = b * c2; }
        else {
          const int k0 = kt + jt * 8 + fn;
          s[jt][0] = (k0 < nrow) ? a * c2 : FMIN;
          s[jt][1] = (k0 + 1 < nrow) ? b * c2 : FMIN;
        }
        x = max(x, max(s[jt][0], s[jt][1]));
      }
      x = max(x, simd_shuffle_xor(x, ushort(1)));
      x = max(x, simd_shuffle_xor(x, ushort(8)));
      const float nm = max(m, x);
      const float f = fast::exp2(m - nm);
      float ps = 0.0f;
      simdgroup_matrix<half, 8, 8> Pf[NJ];
#pragma clang loop unroll(full)
      for (int jt = 0; jt < NJ; jt++) {
#pragma clang loop unroll(full)
        for (int e = 0; e < 2; e++) {
          const float p = (s[jt][e] == FMIN) ? 0.0f : fast::exp2(s[jt][e] - nm);
          const half pq = half(p);
          ps += float(pq);
          Pf[jt].thread_elements()[e] = pq;
        }
      }
      l = l * f + ps; m = nm;
      if (simd_any(f != 1.0f)) {
#pragma clang loop unroll(full)
        for (int j = 0; j < DH; j++) { O[j].thread_elements()[0] *= f; O[j].thread_elements()[1] *= f; }
      }
#pragma clang loop unroll(full)
      for (int jt = 0; jt < NJ; jt++) {
#pragma clang loop unroll(full)
        for (int cj = 0; cj < DH; cj++) {
          simdgroup_matrix<half, 8, 8> vf;
          const half2 v = *(threadgroup half2*)(&Vs[(jt * 8 + fm) * 256 + ((d0 + cj) ^ fm) * 8 + fn]);
          vf.thread_elements()[0] = v.x; vf.thread_elements()[1] = v.y;
          simdgroup_multiply_accumulate(O[cj], Pf[jt], vf, O[cj]);
        }
      }
    }
    }
  }

  if (active && rvalid) {
    l += simd_shuffle_xor(l, ushort(1));
    l += simd_shuffle_xor(l, ushort(8));
    const size_t slot = (size_t(qh) * T + rt) * NSL + sl;
    if (ds == 0 && (int(lane) & 9) == 0) { PM[slot] = m; PL[slot] = l; }
#pragma clang loop unroll(full)
    for (int cj = 0; cj < DH; cj++)
      *(device float2*)(PO + slot * D + (d0 + cj) * 8 + fn) = float2(O[cj].thread_elements()[0], O[cj].thread_elements()[1]);
  }
"""

_P2 = r"""
  const float FMIN = -3.4028234663852886e+38f;
  const uint lane = thread_index_in_simdgroup;
  const int h = int(threadgroup_position_in_grid.y), t = int(threadgroup_position_in_grid.z);
  const int T = dims[1], NSL = dims[2];
  const size_t slot0 = (size_t(h) * T + t) * NSL;
  float mm = FMIN;
  for (int i = int(lane); i < NSL; i += 32) mm = max(mm, PM[slot0 + i]);
  mm = simd_max(mm);
  float L = 0.0f;
  for (int i = int(lane); i < NSL; i += 32) L += fast::exp2(PM[slot0 + i] - mm) * PL[slot0 + i];
  L = simd_sum(L);
  float o[8];
  for (int j = 0; j < 8; j++) o[j] = 0.0f;
  for (int i = 0; i < NSL; i++) {
    const float w = fast::exp2(PM[slot0 + i] - mm);
    const device float4* p = (const device float4*)(PO + (slot0 + i) * 256 + lane * 8);
    const float4 a = p[0], b = p[1];
    o[0] += w * a.x; o[1] += w * a.y; o[2] += w * a.z; o[3] += w * a.w;
    o[4] += w * b.x; o[5] += w * b.y; o[6] += w * b.z; o[7] += w * b.w;
  }
  for (int j = 0; j < 8; j++) OUT[(size_t(h) * T + t) * 256 + lane * 8 + j] = bfloat(o[j] / L);
"""

def _with_load(src: str) -> str:
    """Source of _P1 with the blocks laid out at stride LDK and the fragments loaded by simdgroup_load (PLI_ATTM_LOAD)."""

    for a, b in (
        ("threadgroup half KV[NBUF * 2 * BK * 256];", "threadgroup half KV[NBUF * 2 * BK * LDK];"),
        ("*(threadgroup half4*)(&(KV)[kk * 256 + pc * 8]) = h0; *(threadgroup half4*)(&(KV)[kk * 256 + pc * 8 + 4]) = h1;",
         "*(threadgroup half4*)(&(KV)[kk * LDK + c8 * 8]) = h0; *(threadgroup half4*)(&(KV)[kk * LDK + c8 * 8 + 4]) = h1;"),
        ("*(threadgroup half4*)(&(KV)[BK * 256 + kk * 256 + pc * 8]) = h0; *(threadgroup half4*)(&(KV)[BK * 256 + kk * 256 + pc * 8 + 4]) = h1;",
         "*(threadgroup half4*)(&(KV)[BK * LDK + kk * LDK + c8 * 8]) = h0; *(threadgroup half4*)(&(KV)[BK * LDK + kk * LDK + c8 * 8 + 4]) = h1;"),
        ("threadgroup half* Vs = (KV) + BK * 256;", "threadgroup half* Vs = (KV) + BK * LDK;"),
        ("""          const int ka = jt * 8 + fn;
          const int dj = d0 + dd;
          kf.thread_elements()[0] = Ks[ka * 256 + (dj ^ (ka & 7)) * 8 + fm];
          kf.thread_elements()[1] = Ks[(ka + 1) * 256 + (dj ^ ((ka + 1) & 7)) * 8 + fm];
""", """          simdgroup_load(kf, Ks + (jt * 8) * LDK + (d0 + dd) * 8, LDK, ulong2(0, 0), true);
"""),
        ("""          const half2 v = *(threadgroup half2*)(&Vs[(jt * 8 + fm) * 256 + ((d0 + cj) ^ fm) * 8 + fn]);
          vf.thread_elements()[0] = v.x; vf.thread_elements()[1] = v.y;
""", """          simdgroup_load(vf, Vs + (jt * 8) * LDK + (d0 + cj) * 8, LDK);
"""),
    ):
        if src.count(a) != 1:
            raise RuntimeError("attm: PLI_ATTM_LOAD does not find " + a[:50])
        src = src.replace(a, b)
    return src


_K: dict = {}
_CONST: dict = {}
_DIMS: list = [None, None]


def _small(key, make):
    v = _CONST.get(key)
    if v is None:
        v = _CONST[key] = make()
    return v


def _kernels(nsg: int, rgn: int):
    key = (nsg, rgn)
    if key not in _K:
        pre = "".join("  constexpr int %s = %d;\n" % kv for kv in (("BK", BK), ("NS", NS), ("NSG", nsg), ("RGN", rgn), ("NACC", 1), ("NBUF", 1)))
        src = _P1.replace("half(bfx(", "h16(bfx(").replace("half(float(qpb", "h16(float(qpb") if CLAMP else _P1
        if LOAD:
            pre += "  constexpr int LDK = %d;\n" % LDK
            src = _with_load(src)
        k1 = mx.fast.metal_kernel(name=("pli_attm_p1e_%d_%d" if CLAMP else "pli_attm_p1_%d_%d") % key + ("_c" if LOAD else ""),
                                  input_names=["Q", "K", "V", "scale", "dims"], output_names=["PM", "PL", "PO"],
                                  source=pre + src, header=_HEADER, ensure_row_contiguous=False)
        k2 = _K.get("p2")
        if k2 is None:
            k2 = _K["p2"] = mx.fast.metal_kernel(name="pli_attm_p2", input_names=["PM", "PL", "PO", "dims"], output_names=["OUT"], source=_P2)
        _K[key] = (k1, k2)
    return _K[key]


def sdpa(queries: mx.array, bk: mx.array, bv: mx.array, L: int, scale: float) -> mx.array | None:
    """Attention of the T rows of ``queries`` [1, 24, T, 256] over the whole buffers ``bk``, ``bv`` [1, 4, capacity, 256]; row t sees
    L - T + t + 1 keys. Bits specific to this kernel (V3). None if outside what was measured."""

    if queries.ndim != 4 or bk.ndim != 4 or queries.dtype != mx.bfloat16 or bk.dtype != mx.bfloat16 or bv.dtype != mx.bfloat16:
        return None
    if queries.shape[0] != 1 or queries.shape[1] != H or queries.shape[3] != D or bk.shape[1] != HKV or bk.shape[3] != D \
            or bv.shape[:2] != bk.shape[:2] or bv.shape[3] != D or int(bk.shape[2]) < L or int(bv.shape[2]) != int(bk.shape[2]):
        return None
    T = int(queries.shape[2])
    if T < MIN_ROWS or T > MAX_ROWS or L - T + 1 < MIN_KEYS:
        return None
    nrt = -(-(G * T) // 8)
    tpg = max(min(nrt, TPG_MAX), TPG_LOAD.get(nrt, 0))
    nsg = tpg * NS
    rgn = -(-nrt // tpg)
    slen = SMALL if (SHORT and L <= SHORT) else SLEN
    s1 = (SMALL_KEYS // SMALL) * SMALL
    sp = SMALL if s1 else 0
    nsl = (-(-min(L, s1) // sp) + -(-max(0, L - s1) // slen)) if s1 else -(-L // slen)
    if _DIMS[0] != (L, T, s1, slen):
        _DIMS[0], _DIMS[1] = (L, T, s1, slen), mx.array([L, T, nsl, slen, s1, sp, 0, 0], dtype=mx.int32)
    sa = _small(("s", float(scale)), lambda: mx.array([float(scale)] * 8, dtype=mx.float32))
    k1, k2 = _kernels(nsg, rgn)
    pm, pl, po = k1(inputs=[queries, bk, bv, sa, _DIMS[1]], grid=(32 * nsg, nsl, HKV * rgn), threadgroup=(32 * nsg, 1, 1),
                    output_shapes=[(H * T * nsl,), (H * T * nsl,), (H * T * nsl * D,)], output_dtypes=[mx.float32, mx.float32, mx.float32])
    return k2(inputs=[pm, pl, po, _DIMS[1]], grid=(32, H, T), threadgroup=(32, 1, 1), output_shapes=[(1, H, T, D)],
              output_dtypes=[queries.dtype])[0]


# PLI (16/10/2026, D2: draft trees): variant of the kernel for a TREE window. Row t sees the whole context (keys < L - T) and,
# among the T keys of the window, those of its ancestors and itself (mask AN[t], bit j = row j of the window). A chain keeps the
# original kernel (sdpa): its bits do not change. A chain passed here (AN[t] = 2^(t+1) - 1) sees the same keys.
_P1_TREE = (_P1.replace('      const bool full = (kt + BK) <= (NK - T + 1);\n', "      const bool full = (kt + BK) <= (NK - T);\n")
             .replace('          s[jt][0] = (k0 < nrow) ? a * c2 : FMIN;\n          s[jt][1] = (k0 + 1 < nrow) ? b * c2 : FMIN;\n', "          s[jt][0] = (rvalid && (k0 < base_ || (k0 < NK && ((an_ >> uint(k0 - base_)) & 1u)))) ? a * c2 : FMIN;\n          s[jt][1] = (rvalid && (k0 + 1 < base_ || (k0 + 1 < NK && ((an_ >> uint(k0 + 1 - base_)) & 1u)))) ? b * c2 : FMIN;\n")
             .replace('  const bool active = tile * 8 < G * T;\n', '  const bool active = tile * 8 < G * T;\n  const int base_ = NK - T;\n  const uint an_ = rvalid ? AN[rt] : 0u;\n  '))


def _kernels_tree(nsg: int, rgn: int):
    key = ("tree", nsg, rgn)
    if key not in _K:
        pre = "".join("  constexpr int %s = %d;\n" % kv for kv in (("BK", BK), ("NS", NS), ("NSG", nsg), ("RGN", rgn), ("NACC", 1), ("NBUF", 1)))
        src = _P1_TREE.replace("half(bfx(", "h16(bfx(").replace("half(float(qpb", "h16(float(qpb") if CLAMP else _P1_TREE
        k1 = mx.fast.metal_kernel(name=("pli_attm_tree_p1e_%d_%d" if CLAMP else "pli_attm_tree_p1_%d_%d") % (nsg, rgn),
                                  input_names=["Q", "K", "V", "scale", "dims", "AN"], output_names=["PM", "PL", "PO"],
                                  source=pre + src, header=_HEADER, ensure_row_contiguous=False)
        k2 = _K.get("p2")
        if k2 is None:
            k2 = _K["p2"] = mx.fast.metal_kernel(name="pli_attm_p2", input_names=["PM", "PL", "PO", "dims"], output_names=["OUT"], source=_P2)
        _K[key] = (k1, k2)
    return _K[key]


def ancestors(parents) -> tuple[int, ...]:
    """Mask of the ancestors of each row of the window (itself included); parents[t] < t, -1 for a root."""
    an: list[int] = []
    for t, p in enumerate(parents):
        if p >= t:
            raise ValueError("attm tree: a parent must come before its row")
        an.append((1 << t) | (an[p] if p >= 0 else 0))
    return tuple(an)


def sdpa_tree(queries: mx.array, bk: mx.array, bv: mx.array, L: int, scale: float, parents) -> mx.array | None:
    """Like ``sdpa``, for a tree window of T rows (T <= 32): context keys [0, L - T) and, within the window, the row's ancestors."""

    if queries.ndim != 4 or bk.ndim != 4 or queries.dtype != mx.bfloat16 or bk.dtype != mx.bfloat16 or bv.dtype != mx.bfloat16:
        return None
    if queries.shape[0] != 1 or queries.shape[1] != H or queries.shape[3] != D or bk.shape[1] != HKV or bk.shape[3] != D \
            or bv.shape[:2] != bk.shape[:2] or bv.shape[3] != D or int(bk.shape[2]) < L or int(bv.shape[2]) != int(bk.shape[2]):
        return None
    T = int(queries.shape[2])
    if T < MIN_ROWS or T > MAX_ROWS or len(parents) != T or L - T < 0:
        return None
    an = ancestors(parents)
    nrt = -(-(G * T) // 8)
    tpg = max(min(nrt, TPG_MAX), TPG_LOAD.get(nrt, 0))
    nsg = tpg * NS
    rgn = -(-nrt // tpg)
    # PLI (item D, 24/10/2026): same slices as sdpa (PLI_ATTM_SHORT: slices of SMALL keys under SHORT keys), otherwise a node does not have
    # the bits of the same row in a chain (tree_anchor_trial B1); PLI_ATTM_SMALL_KEYS is not taken over (S1 = SP = 0 here as in production).
    slen = SMALL if (SHORT and L <= SHORT) else SLEN
    nsl = -(-L // slen)
    dims = _small(("dims", L, T, slen), lambda: mx.array([L, T, nsl, slen, 0, 0, 0, 0], dtype=mx.int32))
    if len(_CONST) > 4096:
        _CONST.clear()
    an_a = _small(("an", an), lambda: mx.array(list(an), dtype=mx.uint32))
    sa = _small(("s", float(scale)), lambda: mx.array([float(scale)] * 8, dtype=mx.float32))
    k1, k2 = _kernels_tree(nsg, rgn)
    pm, pl, po = k1(inputs=[queries, bk, bv, sa, dims, an_a], grid=(32 * nsg, nsl, HKV * rgn), threadgroup=(32 * nsg, 1, 1),
                    output_shapes=[(H * T * nsl,), (H * T * nsl,), (H * T * nsl * D,)], output_dtypes=[mx.float32, mx.float32, mx.float32])
    return k2(inputs=[pm, pl, po, dims], grid=(32, H, T), threadgroup=(32, 1, 1), output_shapes=[(1, H, T, D)],
              output_dtypes=[queries.dtype])[0]


def _p1(queries: mx.array, bk: mx.array, bv: mx.array, nk: int, nsl: int, slen: int, scale: float):
    """First pass of the chain kernel over the first nsl slices of slen keys: (PM, PL, PO) in [H, T, nsl] and [H, T, nsl, D]."""

    T = int(queries.shape[2])
    nrt = -(-(G * T) // 8)
    tpg = max(min(nrt, TPG_MAX), TPG_LOAD.get(nrt, 0))
    nsg = tpg * NS
    rgn = -(-nrt // tpg)
    dims = _small(("dimsx", nk, T, nsl, slen), lambda: mx.array([nk, T, nsl, slen, 0, 0, 0, 0], dtype=mx.int32))
    sa = _small(("s", float(scale)), lambda: mx.array([float(scale)] * 8, dtype=mx.float32))
    k1, _ = _kernels(nsg, rgn)
    pm, pl, po = k1(inputs=[queries, bk, bv, sa, dims], grid=(32 * nsg, nsl, HKV * rgn), threadgroup=(32 * nsg, 1, 1),
                    output_shapes=[(H * T * nsl,), (H * T * nsl,), (H * T * nsl * D,)], output_dtypes=[mx.float32, mx.float32, mx.float32])
    return pm.reshape(H, T, nsl), pl.reshape(H, T, nsl), po.reshape(H, T, nsl, D)


def _sdpa_tree_exact_loop(queries: mx.array, bk: mx.array, bv: mx.array, L: int, scale: float, parents) -> mx.array | None:
    """PLI (item D, 24/10/2026, PLI_TREE_EXACT): like ``sdpa_tree``, but each node has the bits of the same row in a chain. The slices
    of the context that end before the window are computed for all nodes in one pass (they are full for all rows);
    the slice where the window starts is redone for each node over its chain keys: end of the context then its ancestors in a row,
    at the same places as in a chain. The second pass merges the slices in chain order (empty slices counted as zero)."""

    if queries.ndim != 4 or bk.ndim != 4 or queries.dtype != mx.bfloat16 or bk.dtype != mx.bfloat16 or bv.dtype != mx.bfloat16:
        return None
    if queries.shape[0] != 1 or queries.shape[1] != H or queries.shape[3] != D or bk.shape[1] != HKV or bk.shape[3] != D \
            or bv.shape[:2] != bk.shape[:2] or bv.shape[3] != D or int(bk.shape[2]) < L or int(bv.shape[2]) != int(bk.shape[2]):
        return None
    T = int(queries.shape[2])
    if T < MIN_ROWS or T > MAX_ROWS or len(parents) != T or L - T < 0:
        return None
    base = L - T
    slen = SMALL if (SHORT and L <= SHORT) else SLEN
    ts = (base // slen) * slen
    nsl0 = ts // slen
    paths = []
    for t, p in enumerate(parents):
        p = int(p)
        if p >= t:
            return None
        paths.append((paths[p] if p >= 0 else []) + [t])
    if nsl0:
        pm0, pl0, po0 = _p1(queries, bk, bv, ts + T - 1, nsl0, slen, scale)
    tails = []
    for t in range(T):
        ix = mx.array([base + j for j in paths[t]], dtype=mx.int32)
        kq = mx.concatenate([bk[:, :, ts:base], mx.take(bk, ix, axis=2)], axis=2)
        vq = mx.concatenate([bv[:, :, ts:base], mx.take(bv, ix, axis=2)], axis=2)
        n = base - ts + len(paths[t])
        tails.append(_p1(queries[:, :, t:t + 1], kq, vq, n, -(-n // slen), slen, scale))
    nq = max(int(q[0].shape[2]) for q in tails)
    pms, pls, pos = [], [], []
    for t, (pm, pl, po) in enumerate(tails):
        k = int(pm.shape[2])
        if k < nq:                                   # empty slices: PM at the lowest, PL and PO zero (exp2(PM - max) = 0)
            pm = mx.concatenate([pm, mx.full((H, 1, nq - k), -3.4028234663852886e+38, dtype=mx.float32)], axis=2)
            pl = mx.concatenate([pl, mx.zeros((H, 1, nq - k), dtype=mx.float32)], axis=2)
            po = mx.concatenate([po, mx.zeros((H, 1, nq - k, D), dtype=mx.float32)], axis=2)
        if nsl0:
            pm = mx.concatenate([pm0[:, t:t + 1], pm], axis=2)
            pl = mx.concatenate([pl0[:, t:t + 1], pl], axis=2)
            po = mx.concatenate([po0[:, t:t + 1], po], axis=2)
        pms.append(pm); pls.append(pl); pos.append(po)
    nsl = nsl0 + nq
    PM = mx.concatenate(pms, axis=1).reshape(-1)
    PL = mx.concatenate(pls, axis=1).reshape(-1)
    PO = mx.concatenate(pos, axis=1).reshape(-1)
    dims = _small(("dims2", T, nsl), lambda: mx.array([L, T, nsl, slen, 0, 0, 0, 0], dtype=mx.int32))
    _, k2 = _kernels(*(lambda nrt: (max(min(nrt, TPG_MAX), TPG_LOAD.get(nrt, 0)) * NS, 1))(-(-(G * T) // 8)))
    return k2(inputs=[PM, PL, PO, dims], grid=(32, H, T), threadgroup=(32, 1, 1), output_shapes=[(1, H, T, D)],
              output_dtypes=[queries.dtype])[0]


# PLI (28/10/2026, Kapitsa's "tree zone" method): ``sdpa_tree_exact`` without a loop over nodes. The full slices stay one shared
# pass (_p1); the tail of each node (end of the context then its ancestors at the chain's places) goes through ONE call of the same kernel,
# one tile per (kv head, node), the keys read through an indirection table (PATH) instead of being copied. Same arithmetic per row as the
# call per node (one row, same slices, same key blocks, same order), so the same bits as ``sdpa_tree_exact``.
_P1_TAIL = (_P1
    .replace("  const int kv = z / RGN, rg = z % RGN;\n",
             "  const int NN_ = dims[6];\n  const int kv = z / NN_, node_ = z % NN_, rg = 0;\n")
    .replace("  const int NK = dims[0], T = dims[1], NSL = dims[2], SLEN = dims[3];\n",
             "  const int TS_ = dims[0], BASE_ = dims[1], NSL = dims[2], SLEN = dims[3];\n  const int T = 1;\n"
             "  const int NK = BASE_ - TS_ + int(PATH[node_ * 33]);\n"
             "#define PHYS_(kk) ((kk) < (BASE_ - TS_) ? (TS_ + (kk)) : (BASE_ + int(PATH[node_ * 33 + 1 + ((kk) - (BASE_ - TS_))])))\n")
    .replace("size_t(rt) * Q_strides[2]", "size_t(node_) * Q_strides[2]")
    .replace("K + kb0 + size_t(key) * ks2", "K + kb0 + size_t(PHYS_(key)) * ks2")
    .replace("V + vb0 + size_t(key) * vs2", "V + vb0 + size_t(PHYS_(key)) * vs2")
    .replace("    const size_t slot = (size_t(qh) * T + rt) * NSL + sl;\n",
             "    const size_t slot = (size_t(qh) * NN_ + node_) * NSL + sl;\n"))
assert _P1_TAIL.count("PHYS_(key)") == 4 and "node_) * Q_strides" in _P1_TAIL and "NN_ + node_" in _P1_TAIL


def _kernel_tail(nsg: int):
    key = ("tail", nsg)
    if key not in _K:
        pre = "".join("  constexpr int %s = %d;\n" % kv for kv in (("BK", BK), ("NS", NS), ("NSG", nsg), ("RGN", 1), ("NACC", 1), ("NBUF", 1)))
        src = _P1_TAIL.replace("half(bfx(", "h16(bfx(").replace("half(float(qpb", "h16(float(qpb") if CLAMP else _P1_TAIL
        _K[key] = mx.fast.metal_kernel(name=("pli_attm_tail_e_%d" if CLAMP else "pli_attm_tail_%d") % nsg,
                                       input_names=["Q", "K", "V", "scale", "dims", "PATH"], output_names=["PM", "PL", "PO"],
                                       source=pre + src, header=_HEADER, ensure_row_contiguous=False)
    return _K[key]


# Second pass over two sources (shared full slices, then tails per node): the slice index i runs over nsl0 then nq exactly
# as in the concatenated array, so the same bits as _P2 on the concatenation, without copying it.
_P2_TWO = r"""
  const float FMIN = -3.4028234663852886e+38f;
  const uint lane = thread_index_in_simdgroup;
  const int h = int(threadgroup_position_in_grid.y), t = int(threadgroup_position_in_grid.z);
  const int T = dims[1], N0 = dims[2], NQ = dims[3], NSL = N0 + NQ;
  const size_t s0 = (size_t(h) * T + t) * N0, sq = (size_t(h) * T + t) * NQ;
#define XM_(i) ((i) < N0 ? PM0[s0 + (i)] : PMQ[sq + (i) - N0])
#define XL_(i) ((i) < N0 ? PL0[s0 + (i)] : PLQ[sq + (i) - N0])
  float mm = FMIN;
  for (int i = int(lane); i < NSL; i += 32) mm = max(mm, XM_(i));
  mm = simd_max(mm);
  float L = 0.0f;
  for (int i = int(lane); i < NSL; i += 32) L += fast::exp2(XM_(i) - mm) * XL_(i);
  L = simd_sum(L);
  float o[8];
  for (int j = 0; j < 8; j++) o[j] = 0.0f;
  for (int i = 0; i < NSL; i++) {
    const float w = fast::exp2(XM_(i) - mm);
    const device float4* p = (const device float4*)(i < N0 ? PO0 + (s0 + i) * 256 + lane * 8 : POQ + (sq + i - N0) * 256 + lane * 8);
    const float4 a = p[0], b = p[1];
    o[0] += w * a.x; o[1] += w * a.y; o[2] += w * a.z; o[3] += w * a.w;
    o[4] += w * b.x; o[5] += w * b.y; o[6] += w * b.z; o[7] += w * b.w;
  }
  for (int j = 0; j < 8; j++) OUT[(size_t(h) * T + t) * 256 + lane * 8 + j] = bfloat(o[j] / L);
"""


def _kernel_p2_two():
    k = _K.get("p2two")
    if k is None:
        k = _K["p2two"] = mx.fast.metal_kernel(name="pli_attm_p2two", input_names=["PM0", "PL0", "PO0", "PMQ", "PLQ", "POQ", "dims"],
                                                output_names=["OUT"], source=_P2_TWO)
    return k


def sdpa_tree_exact2(queries: mx.array, bk: mx.array, bv: mx.array, L: int, scale: float, parents) -> mx.array | None:
    """``sdpa_tree_exact`` (same bits) in three kernel calls instead of five operations per node."""

    if queries.ndim != 4 or bk.ndim != 4 or queries.dtype != mx.bfloat16 or bk.dtype != mx.bfloat16 or bv.dtype != mx.bfloat16:
        return None
    if queries.shape[0] != 1 or queries.shape[1] != H or queries.shape[3] != D or bk.shape[1] != HKV or bk.shape[3] != D \
            or bv.shape[:2] != bk.shape[:2] or bv.shape[3] != D or int(bk.shape[2]) < L or int(bv.shape[2]) != int(bk.shape[2]):
        return None
    T = int(queries.shape[2])
    if T < MIN_ROWS or T > MAX_ROWS or len(parents) != T or L - T < 0:
        return None
    base = L - T
    slen = SMALL if (SHORT and L <= SHORT) else SLEN
    ts = (base // slen) * slen
    nsl0 = ts // slen
    key = ("paths", tuple(int(p) for p in parents))
    plan = _CONST.get(key)
    if plan is None:
        paths: list[list[int]] = []
        for t, p in enumerate(parents):
            p = int(p)
            if p >= t:
                return None
            paths.append((paths[p] if p >= 0 else []) + [t])
        table = []
        for c in paths:
            table += [len(c), *c, *([0] * (32 - len(c)))]
        if len(_CONST) > 4096:
            _CONST.clear()
        plan = _CONST[key] = (tuple(len(c) for c in paths), mx.array(table, dtype=mx.uint32))
    lengths, path_a = plan
    nq = max(-(-(base - ts + n) // slen) for n in lengths)
    sa = _small(("s", float(scale)), lambda: mx.array([float(scale)] * 8, dtype=mx.float32))
    dq = _small(("dimsq", ts, base, nq, slen, T), lambda: mx.array([ts, base, nq, slen, 0, 0, T, 0], dtype=mx.int32))
    nsg = max(min(1, TPG_MAX), TPG_LOAD.get(1, 0)) * NS          # one row of 6 heads: as _p1 for T = 1
    kq = _kernel_tail(nsg)
    qm, ql, qo = kq(inputs=[queries, bk, bv, sa, dq, path_a], grid=(32 * nsg, nq, HKV * T), threadgroup=(32 * nsg, 1, 1),
                    output_shapes=[(H * T * nq,), (H * T * nq,), (H * T * nq * D,)], output_dtypes=[mx.float32, mx.float32, mx.float32])
    if nsl0:
        pm0, pl0, po0 = _p1(queries, bk, bv, ts + T - 1, nsl0, slen, scale)
        d2 = _small(("dimsd", T, nsl0, nq), lambda: mx.array([L, T, nsl0, nq, 0, 0, 0, 0], dtype=mx.int32))
        return _kernel_p2_two()(inputs=[pm0.reshape(-1), pl0.reshape(-1), po0.reshape(-1), qm, ql, qo, d2], grid=(32, H, T),
                                 threadgroup=(32, 1, 1), output_shapes=[(1, H, T, D)], output_dtypes=[queries.dtype])[0]
    qm, ql, qo = qm.reshape(H, T, nq), ql.reshape(H, T, nq), qo.reshape(H, T, nq, D)
    nsl = nsl0 + nq
    dims = _small(("dims2", T, nsl), lambda: mx.array([L, T, nsl, slen, 0, 0, 0, 0], dtype=mx.int32))
    _, k2 = _kernels(*(lambda nrt: (max(min(nrt, TPG_MAX), TPG_LOAD.get(nrt, 0)) * NS, 1))(-(-(G * T) // 8)))
    return k2(inputs=[qm.reshape(-1), ql.reshape(-1), qo.reshape(-1), dims], grid=(32, H, T), threadgroup=(32, 1, 1),
              output_shapes=[(1, H, T, D)], output_dtypes=[queries.dtype])[0]


# PLI (28/10/2026): ``sdpa_tree_exact`` goes through the one-kernel version (same bits, checked on 120 cases on an M1 Pro);
# PLI_TREE_EXACT2=0 keeps the old loop over nodes.
_EXACT2 = os.environ.get("PLI_TREE_EXACT2", "1") != "0"


def sdpa_tree_exact(queries: mx.array, bk: mx.array, bv: mx.array, L: int, scale: float, parents) -> mx.array | None:
    if _EXACT2:
        out = sdpa_tree_exact2(queries, bk, bv, L, scale, parents)
        if out is not None:
            return out
    return _sdpa_tree_exact_loop(queries, bk, bv, L, scale, parents)


__all__ = ["sdpa", "sdpa_tree", "sdpa_tree_exact", "sdpa_tree_exact2", "ancestors"]
