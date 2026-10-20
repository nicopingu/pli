"""Row-exact 4-bit matmul uses fixed per-group FMA chains and shape-dependent chunk reductions, with scalar/MMA equivalence checked at install."""

from __future__ import annotations

import hashlib
from typing import Any, NamedTuple, Sequence

import mlx.core as mx

from tensorfold.kernels import threads

MAX_ROWS = 1 << 16   # rows a call routed here (prompt chunks included); every row's bits are its one-row bits
# PLI (lot S6, sonde pli-rt du 10/10) : PLI_PAVES_PAR_GROUPE règle RT_MAX (2 = origine ; 1 mesuré plus rapide de 9 à 64 lignes, mêmes bits).
RT_MAX = max(1, int(__import__("os").environ.get("PLI_PAVES_PAR_GROUPE", "2") or 2))   # 8-row tiles a threadgroup: more rows than 8 RT_MAX spread over the grid's y axis
MMA_SGS = 16         # physical simdgroups at most (fewer where a pipeline takes fewer threads): same chunks
GROUP = 64
# PLI (lot P2, palier a, 15/10/2026) : PLI_XF_BF16=1 range les fragments d'entrée du noyau à 2..8 lignes (mmaf) en paires de bf16
# (un mot de 32 bits) au lieu de paires de fp32. La valeur x * 2^-4s d'une entrée bf16 tient exactement en bf16 (même exposant
# que fp32, mantisse de 8 bits inchangée) : mêmes valeurs, mêmes bits, deux fois moins d'octets relus. Seule exception : un
# résultat sous 2^-126 (|x| < 2^-98), où fp32 et bf16 n'ont pas la même mantisse dénormale. Éteint par défaut.
XF_BF16 = __import__("os").environ.get("PLI_XF_BF16", "0") == "1"

_HEADER = r"""
#define PRAGMA_UNROLL _Pragma("clang loop unroll(full)")
// the bf16 at index e (0..7) of 8 packed bf16 as fp32
inline float bf8(uint4 v, int e) {
  const uint w = v[e / 2];
  return as_type<float>((e % 2) ? (w & 0xFFFF0000u) : (w << 16));
}
// a row's 8 inputs summed left to right
inline float sum8(uint4 v, float one) {
  float t = bf8(v, 0);
  for (int e = 1; e < 8; e++) t = fma(bf8(v, e), one, t);
  return t;
}
// 2^-4s
inline float pre(int s) { return as_type<float>(uint(127 - 4 * s) << 23); }
// PLI (P2 b+c, rangement « rg ») : poids par paquets de 32 sorties [paquet][groupe][fm * 4 + quart][tuile 0..3] (uint2),
// échelle (bf16 bas) et décalage (bf16 haut) réunis en un mot [paquet][groupe][fm][tuile]. n = 32 paquet + 8 tuile + fm.
inline size_t rg_w(int n, int g, int q, int G) { return ((size_t(n / 32) * G + g) * 32 + (n % 8) * 4 + q) * 4 + (n / 8) % 4; }
inline size_t rg_sb(int n, int g, int G) { return ((size_t(n / 32) * G + g) * 8 + (n % 8)) * 4 + (n / 8) % 4; }
inline float rg_sc(uint w) { return as_type<float>(w << 16); }
inline float rg_bi(uint w) { return as_type<float>(w & 0xFFFF0000u); }
// un groupe de 64 poids d'une ligne n, moitié h (0 ou 1) : les mots 4h .. 4h + 3 du rangement d'origine
inline uint4 rg_half(const device uint2* W2, int n, int g, int h, int G) {
  const uint2 a = W2[rg_w(n, g, 2 * h, G)], b = W2[rg_w(n, g, 2 * h + 1, G)];
  return uint4(a.x, a.y, b.x, b.y);
}
// PLI (P2 a) : deux valeurs exactes en bf16 (16 bits bas de leur fp32 nuls) rangées dans un mot, a en bas, b en haut
inline uint pack_bf2(float a, float b) { return (as_type<uint>(a) >> 16) | (as_type<uint>(b) & 0xFFFF0000u); }
"""

# PLI (lot E4, entiers d'abord) : mêmes noyaux, entrée en entiers de 16 bits signés (8 par mot de 128 bits, à la place des 8 bf16)
# et sortie en 32 bits. Tant que |x| <= ENT_MAX, la somme d'un groupe de 64 produits x * q (q de 0 à 15) reste sous 2^24 : elle est
# exacte en flottant 32 bits, quel que soit l'ordre des additions. Le reste (une FMA d'échelle et une de décalage par groupe, par
# morceaux, puis arbre) est l'ordre écrit ci-dessous, que le noyau CPU pli_qmm_e15 reprend à l'identique.
ENT_MAX = 16383
_HEADER_ENT = _HEADER.replace("""  const uint w = v[e / 2];
  return as_type<float>((e % 2) ? (w & 0xFFFF0000u) : (w << 16));""", """  const uint w = v[e / 2];
  return float(int(short((e % 2) ? (w >> 16) : (w & 0xFFFFu))));""")
assert _HEADER_ENT != _HEADER

_SCALAR = r"""
  // RS rows (1 to 4). Lane (chunk c = lane % S, slot j = lane / S) runs chunk c of NR outputs n0 + j + (32 / S) u,
  // a whole group (WPG words) of each in registers, once a row; the threadgroup stages XB groups of each row's
  // inputs pre-scaled in chain order. A row's chain is the same at any RS.
  constexpr int WPG = GS / 8, NS = 8 / WPG;     // words a group; nibble stride of an MMA step
  constexpr int XP = GS == 64 ? 76 : 44;        // floats a staged group: GS inputs, WPG sums, pad (bank spread)
  threadgroup float xs[RS * XB * XP];
  const uint lane = thread_index_in_simdgroup;
  const int tid = int(simdgroup_index_in_threadgroup) * 32 + int(lane);
  const int c = int(lane) % S;
  constexpr int SLOTS = 32 / S;
  const int n0 = (int(threadgroup_position_in_grid.x) * SGS + int(simdgroup_index_in_threadgroup)) * (SLOTS * NR)
                 + int(lane) / S;
  constexpr int G = K / GS;
  const float one = ONE[0];
  const device uint4* wr[NR];
  const device bfloat* sr[NR];
  const device bfloat* br[NR];
  float acc[NR][RS];
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++) {
    const int nn = min(n0 + SLOTS * u, N - 1);
    wr[u] = (const device uint4*)(W + size_t(nn) * (K / 8));
    sr[u] = SC + size_t(nn) * G;
    br[u] = BI + size_t(nn) * G;
    PRAGMA_UNROLL
    for (int r = 0; r < RS; r++) acc[u][r] = 0.0f;
  }
  uint4 nw[NR][WPG / 4];
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++) for (int h = 0; h < WPG / 4; h++) nw[u][h] = c < G ? wr[u][(WPG / 4) * c + h] : uint4(0);
  for (int b0 = 0; b0 < G; b0 += XB) {
    const int nbk = min(XB, G - b0);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int idx = tid; idx < RS * nbk * WPG; idx += SGS * 32) {
      const int r = RS == 1 ? 0 : idx / (nbk * WPG);
      const int gl = (RS == 1 ? idx : idx - r * (nbk * WPG)) / WPG, j = idx % WPG;
      const uint4 v = LOAD8(r, WPG * (b0 + gl) + j);
      threadgroup float* xr = xs + r * (XB * XP) + gl * XP;
      PRAGMA_UNROLL
      for (int e = 0; e < 8; e++) xr[8 * (e / NS) + NS * j + e % NS] = bf8(v, e) * pre(e);
      xr[GS + j] = sum8(v, one);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int g = b0 + c; g < b0 + nbk; g += S) {
      uint4 wv[NR][WPG / 4];
      PRAGMA_UNROLL
      for (int u = 0; u < NR; u++) for (int h = 0; h < WPG / 4; h++) wv[u][h] = nw[u][h];
      if (g + S < G) {
        PRAGMA_UNROLL
        for (int u = 0; u < NR; u++) for (int h = 0; h < WPG / 4; h++) nw[u][h] = wr[u][(WPG / 4) * (g + S) + h];
      }
      float xsum[RS];
      float P[NR][RS];
      PRAGMA_UNROLL
      for (int r = 0; r < RS; r++) {
        const threadgroup float* xg = xs + r * (XB * XP) + (g - b0) * XP;
        const float4 p0 = *(const threadgroup float4*)(xg + GS);
        xsum[r] = fma(fma(p0.w, one, p0.z), one, fma(p0.y, one, p0.x));
        if (GS == 64) {
          const float4 p1 = *(const threadgroup float4*)(xg + GS + 4);
          xsum[r] = fma(fma(fma(p1.w, one, p1.z), one, fma(p1.y, one, p1.x)), one, xsum[r]);
        }
        PRAGMA_UNROLL
        for (int u = 0; u < NR; u++) P[u][r] = 0.0f;
      }
      PRAGMA_UNROLL
      for (int s = 0; s < WPG; s++) {
        float xq[RS][8];
        PRAGMA_UNROLL
        for (int r = 0; r < RS; r++) {
          const threadgroup float* xg = xs + r * (XB * XP) + (g - b0) * XP + 8 * s;
          const float4 lo = *(const threadgroup float4*)(xg), hi = *(const threadgroup float4*)(xg + 4);
          xq[r][0] = lo.x; xq[r][1] = lo.y; xq[r][2] = lo.z; xq[r][3] = lo.w;
          xq[r][4] = hi.x; xq[r][5] = hi.y; xq[r][6] = hi.z; xq[r][7] = hi.w;
        }
        PRAGMA_UNROLL
        for (int u = 0; u < NR; u++)
          PRAGMA_UNROLL
          for (int i = 0; i < 8; i++) {
            const float q = float(wv[u][i / NS / 4][(i / NS) % 4] & (0xFu << (4 * (NS * s + i % NS))));
            PRAGMA_UNROLL
            for (int r = 0; r < RS; r++) P[u][r] = fma(xq[r][i], q, P[u][r]);
          }
      }
      PRAGMA_UNROLL
      for (int u = 0; u < NR; u++) {
        const float sc = float(sr[u][g]), bi = float(br[u][g]);
        PRAGMA_UNROLL
        for (int r = 0; r < RS; r++) {
          acc[u][r] = fma(sc, P[u][r], acc[u][r]);
          acc[u][r] = fma(bi, xsum[r], acc[u][r]);
        }
      }
    }
  }
  PRAGMA_UNROLL
  for (int u = 0; u < NR; u++)
    PRAGMA_UNROLL
    for (int r = 0; r < RS; r++) {
      float v = acc[u][r];
      PRAGMA_UNROLL
      for (int m = 1; m < S; m <<= 1) v = fma(simd_shuffle_xor(v, ushort(m)), one, v);
      const int n = n0 + SLOTS * u;
      if (n < N && c == 0) OUT[size_t(r) * N + n] = bfloat(v);
    }
"""

_MMA = r"""
  // R rows: threadgroup (x, y) takes rows 8 RT y .. 8 RT y + 8 RT - 1 in RT tiles of 8 (rows >= R read row R - 1;
  // their results are dropped). SGS physical simdgroups compute all S arithmetic chunks in order.
  const uint lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  const int R = X_shape[0];
  constexpr int G = K / GS, WPG = GS / 8, NS = 8 / WPG;   // groups; words a group; nibble stride
  const float one = ONE[0];
  const int nb = int(threadgroup_position_in_grid.x) * (8 * NT);
  const int rb = int(threadgroup_position_in_grid.y) * (8 * RT);
  threadgroup float red[S > 1 ? S * RT * NT * 64 : 1];
  const device uint2* W2 = (const device uint2*)W;
  const device uint* W1 = (const device uint*)W;
  int wrow[NT];
  for (int t = 0; t < NT; t++) wrow[t] = min(nb + 8 * t + fm, N - 1);
  int xr0[RT], xr1[RT];
  for (int rt = 0; rt < RT; rt++) { xr0[rt] = min(rb + 8 * rt + fn, R - 1); xr1[rt] = min(rb + 8 * rt + fn + 1, R - 1); }
  for (int c = sg; c < S; c += SGS) {
    float acc[RT][NT][2];
    for (int rt = 0; rt < RT; rt++)
      for (int t = 0; t < NT; t++) { acc[rt][t][0] = 0.0f; acc[rt][t][1] = 0.0f; }
    for (int g = c; g < G; g += S) {
      uint2 wv[NT];
      PRAGMA_UNROLL
      for (int t = 0; t < NT; t++)
        if (GS == 64) wv[t] = W2[size_t(wrow[t]) * (K / 16) + 4 * g + fn / 2];
        else wv[t] = uint2(W1[size_t(wrow[t]) * (K / 8) + 4 * g + fn / 2]);
      uint4 xa[RT], xb[RT];
      float xs0[RT], xs1[RT];
      PRAGMA_UNROLL
      for (int rt = 0; rt < RT; rt++) {
        xa[rt] = LOAD8(xr0[rt], WPG * g + fm / NS);
        xb[rt] = LOAD8(xr1[rt], WPG * g + fm / NS);
        float v = sum8(xa[rt], one), u = sum8(xb[rt], one);
        if (NS == 1) { v = fma(simd_shuffle_xor(v, ushort(2)), one, v); u = fma(simd_shuffle_xor(u, ushort(2)), one, u); }
        v = fma(simd_shuffle_xor(v, ushort(4)), one, v); u = fma(simd_shuffle_xor(u, ushort(4)), one, u);
        v = fma(simd_shuffle_xor(v, ushort(16)), one, v); u = fma(simd_shuffle_xor(u, ushort(16)), one, u);
        xs0[rt] = v; xs1[rt] = u;
      }
      simdgroup_matrix<float, 8, 8> P[RT][NT];
      PRAGMA_UNROLL
      for (int rt = 0; rt < RT; rt++)
        for (int t = 0; t < NT; t++) P[rt][t] = simdgroup_matrix<float, 8, 8>(0.0f);
      PRAGMA_UNROLL
      for (int s = 0; s < WPG; s++) {
        const int e = NS * s + fm % NS;          // this lane's input of the step (its MMA-k is fm)
        const float ps = pre(e);
        const uint mask = 0xFu << (4 * NS * s), mask1 = 0xFu << (4 * (NS * s + NS - 1));
        simdgroup_matrix<float, 8, 8> bm[RT];
        PRAGMA_UNROLL
        for (int rt = 0; rt < RT; rt++) {
          bm[rt].thread_elements()[0] = bf8(xa[rt], e) * ps;
          bm[rt].thread_elements()[1] = bf8(xb[rt], e) * ps;
        }
        PRAGMA_UNROLL
        for (int t = 0; t < NT; t++) {
          simdgroup_matrix<float, 8, 8> am;
          am.thread_elements()[0] = float(wv[t].x & mask);
          am.thread_elements()[1] = float(wv[t].y & mask1);
          PRAGMA_UNROLL
          for (int rt = 0; rt < RT; rt++) simdgroup_multiply_accumulate(P[rt][t], am, bm[rt], P[rt][t]);
        }
      }
      PRAGMA_UNROLL
      for (int t = 0; t < NT; t++) {
        const float sc = float(SC[size_t(wrow[t]) * G + g]);
        const float bi = float(BI[size_t(wrow[t]) * G + g]);
        PRAGMA_UNROLL
        for (int rt = 0; rt < RT; rt++) {
          acc[rt][t][0] = fma(bi, xs0[rt], fma(sc, P[rt][t].thread_elements()[0], acc[rt][t][0]));
          acc[rt][t][1] = fma(bi, xs1[rt], fma(sc, P[rt][t].thread_elements()[1], acc[rt][t][1]));
        }
      }
    }
    if (S == 1) {
      for (int rt = 0; rt < RT; rt++)
        for (int t = 0; t < NT; t++)
          for (int e = 0; e < 2; e++) {
            const int row = rb + 8 * rt + fn + e, n = nb + 8 * t + fm;
            if (row < R && n < N) OUT[size_t(row) * N + n] = bfloat(acc[rt][t][e]);
          }
      return;
    }
    for (int rt = 0; rt < RT; rt++)
      for (int t = 0; t < NT; t++)
        for (int e = 0; e < 2; e++) red[((c * RT + rt) * NT + t) * 64 + int(lane) * 2 + e] = acc[rt][t][e];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (int idx = sg * 32 + int(lane); idx < RT * NT * 64; idx += SGS * 32) {
    float v[S];
    for (int k = 0; k < S; k++) v[k] = red[k * (RT * NT * 64) + idx];
    for (int w = 1; w < S; w *= 2)
      for (int k = 0; k + w < S; k += 2 * w) v[k] = fma(v[k + w], one, v[k]);
    const int rt = idx / (NT * 64), t = (idx / 64) % NT, l = (idx % 64) / 2, e = idx % 2;
    const int lq = l / 4;
    const int row = rb + 8 * rt + (lq & 2) * 2 + (l % 2) * 2 + e, n = nb + 8 * t + (lq & 4) + ((l / 2) % 4);
    if (row < R && n < N) OUT[size_t(row) * N + n] = bfloat(v[0]);
  }
"""

def _scalar_rg(src: str) -> str:
    """PLI (P2 b+c) : le noyau scalaire lit le rangement rg (mêmes mots, mêmes valeurs, autres adresses)."""

    out = src
    for old, new in (
        ("""  const device uint4* wr[NR];
  const device bfloat* sr[NR];
  const device bfloat* br[NR];""", """  const device uint2* W2 = (const device uint2*)W;
  int wn[NR];"""),
        ("""    wr[u] = (const device uint4*)(W + size_t(nn) * (K / 8));
    sr[u] = SC + size_t(nn) * G;
    br[u] = BI + size_t(nn) * G;""", """    wn[u] = nn;"""),
        ("""nw[u][h] = c < G ? wr[u][(WPG / 4) * c + h] : uint4(0);""", """nw[u][h] = c < G ? rg_half(W2, wn[u], c, h, G) : uint4(0);"""),
        ("""nw[u][h] = wr[u][(WPG / 4) * (g + S) + h];""", """nw[u][h] = rg_half(W2, wn[u], g + S, h, G);"""),
        ("""        const float sc = float(sr[u][g]), bi = float(br[u][g]);""",
         """        const uint sbw = SC[rg_sb(wn[u], g, G)];
        const float sc = rg_sc(sbw), bi = rg_bi(sbw);"""),
    ):
        assert out.count(old) == 1, old[:60]
        out = out.replace(old, new)
    return out


def _mma_rg(src: str) -> str:
    """PLI (P2 b+c) : le noyau MMA lit le rangement rg ; avec 4 tuiles (paquet entier), lectures de 128 bits."""

    out = src
    for old, new in (
        ("""      uint2 wv[NT];
      PRAGMA_UNROLL
      for (int t = 0; t < NT; t++)
        if (GS == 64) wv[t] = W2[size_t(wrow[t]) * (K / 16) + 4 * g + fn / 2];
        else wv[t] = uint2(W1[size_t(wrow[t]) * (K / 8) + 4 * g + fn / 2]);""",
         """      uint2 wv[NT];
      uint sbw[NT];
      if (NT == 4) {
        const device uint4* W4 = (const device uint4*)W;
        const size_t b4 = ((size_t(nb / 32) * G + g) * 32 + fm * 4 + fn / 2) * 2;
        const uint4 w01 = W4[b4], w23 = W4[b4 + 1];
        const uint4 s4 = ((const device uint4*)SC)[(size_t(nb / 32) * G + g) * 8 + fm];
        PRAGMA_UNROLL
        for (int t = 0; t < NT; t++) {
          wv[t] = t == 0 ? w01.xy : (t == 1 ? w01.zw : (t == 2 ? w23.xy : w23.zw));
          sbw[t] = s4[t];
        }
      } else {
        PRAGMA_UNROLL
        for (int t = 0; t < NT; t++) {
          wv[t] = W2[rg_w(wrow[t], g, fn / 2, G)];
          sbw[t] = SC[rg_sb(wrow[t], g, G)];
        }
      }"""),
        ("""        const float sc = float(SC[size_t(wrow[t]) * G + g]);
        const float bi = float(BI[size_t(wrow[t]) * G + g]);""",
         """        const float sc = rg_sc(sbw[t]);
        const float bi = rg_bi(sbw[t]);"""),
    ):
        assert out.count(old) == 1, old[:60]
        out = out.replace(old, new)
    return out


_PREP = r"""
  // one simdgroup a (row tile, group): writes the MMA kernel's input fragments, pre-scaled
  // (XF[tile][g][s][lane] = x[8 tile + fn + 0 / 1][64 g + 8 fm + s] * 2^-4s, rows past R copy row R - 1), and each
  // row's x-sum of the group by the kernels' tree (XS[r][g])
  const uint lane = thread_index_in_simdgroup;
  const int unit = int(threadgroup_position_in_grid.x) * 4 + int(simdgroup_index_in_threadgroup);
  const int R = X_shape[0];
  constexpr int G = K / 64;
  const int T8 = (R + 7) / 8;
  if (unit >= T8 * G) return;
  const int tile = unit / G, g = unit % G;
  const int qid = int(lane) / 4;
  const int fm = (qid & 4) + ((int(lane) / 2) % 4);
  const int fn = (qid & 2) * 2 + (int(lane) % 2) * 2;
  const float one = ONE[0];
  const int r0 = min(8 * tile + fn, R - 1), r1 = min(8 * tile + fn + 1, R - 1);
  const uint4 xa = LOAD8(r0, 8 * g + fm), xb = LOAD8(r1, 8 * g + fm);
  device float2* xf = (device float2*)XF + (size_t(tile) * G + g) * 256 + lane;
  PRAGMA_UNROLL
  for (int s = 0; s < 8; s++) xf[32 * s] = float2(bf8(xa, s) * pre(s), bf8(xb, s) * pre(s));
  float v = sum8(xa, one), u = sum8(xb, one);
  v = fma(simd_shuffle_xor(v, ushort(2)), one, v); u = fma(simd_shuffle_xor(u, ushort(2)), one, u);
  v = fma(simd_shuffle_xor(v, ushort(4)), one, v); u = fma(simd_shuffle_xor(u, ushort(4)), one, u);
  v = fma(simd_shuffle_xor(v, ushort(16)), one, v); u = fma(simd_shuffle_xor(u, ushort(16)), one, u);
  if (fm == 0) {
    if (8 * tile + fn < R) XS[size_t(8 * tile + fn) * G + g] = v;
    if (8 * tile + fn + 1 < R) XS[size_t(8 * tile + fn + 1) * G + g] = u;
  }
"""


_PREP_BF16 = _PREP.replace('  device float2* xf = (device float2*)XF + (size_t(tile) * G + g) * 256 + lane;\n  PRAGMA_UNROLL\n  for (int s = 0; s < 8; s++) xf[32 * s] = float2(bf8(xa, s) * pre(s), bf8(xb, s) * pre(s));\n', """  device uint* xf = (device uint*)XF + (size_t(tile) * G + g) * 256 + lane;
  PRAGMA_UNROLL
  for (int s = 0; s < 8; s++) xf[32 * s] = pack_bf2(bf8(xa, s) * pre(s), bf8(xb, s) * pre(s));
""")
assert _PREP_BF16 != _PREP


def _fragment_source_bf16(mma: str) -> str:
    """PLI (P2 a) : le noyau mmaf lit les fragments rangés en paires de bf16 (mêmes valeurs que les paires de fp32)."""

    out = _fragment_source(mma)
    old = """          const float2 f = XF2[(size_t(min(rb / 8 + rt, T8 - 1)) * G + g) * 256 + 32 * s + lane];
          bm[rt].thread_elements()[0] = f.x;
          bm[rt].thread_elements()[1] = f.y;"""
    assert old in out
    out = out.replace(old, """          const uint f = XF1[(size_t(min(rb / 8 + rt, T8 - 1)) * G + g) * 256 + 32 * s + lane];
          bm[rt].thread_elements()[0] = as_type<float>(f << 16);
          bm[rt].thread_elements()[1] = as_type<float>(f & 0xFFFF0000u);""")
    old2 = "  const device float2* XF2 = (const device float2*)XF;"
    assert old2 in out
    return out.replace(old2, "  const device uint* XF1 = (const device uint*)XF;")


def _fragment_source(mma: str) -> str:
    """Read pre-scaled fragments and input sums with the same values and bits as the MMA kernel's direct inputs."""

    x_block = mma[mma.index("      uint4 xa[RT], xb[RT];"):mma.index("      simdgroup_matrix<float, 8, 8> P[RT][NT];")]
    out = mma.replace("  const int R = X_shape[0];", "  const int R = XS_shape[0];\n  const int T8 = (R + 7) / 8;\n"
                      "  const device float2* XF2 = (const device float2*)XF;")
    out = out.replace(x_block, """      float xs0[RT], xs1[RT];
      PRAGMA_UNROLL
      for (int rt = 0; rt < RT; rt++) { xs0[rt] = XS[size_t(xr0[rt]) * G + g]; xs1[rt] = XS[size_t(xr1[rt]) * G + g]; }
""")
    old_bm = """          bm[rt].thread_elements()[0] = bf8(xa[rt], e) * ps;
          bm[rt].thread_elements()[1] = bf8(xb[rt], e) * ps;"""
    assert old_bm in out
    out = out.replace(old_bm, """          const float2 f = XF2[(size_t(min(rb / 8 + rt, T8 - 1)) * G + g) * 256 + 32 * s + lane];
          bm[rt].thread_elements()[0] = f.x;
          bm[rt].thread_elements()[1] = f.y;""")
    return out


SGS = 2          # simdgroups a threadgroup in the scalar kernel
NR = 2           # outputs a lane in the scalar kernel
XB = 32          # scalar kernel: groups of inputs staged at a time (one row)
SCALAR_ROWS = 4  # rows the scalar kernel can take (each row its own FMA chains over the same weight loads)
_kernels: dict[tuple, Any] = {}
_plans: dict[tuple, Any] = {}
_BF16 = [mx.bfloat16]
_one: Any = None
# weights (n, k, group size) whose 1-4 row calls go through the MMA kernel (the scalar kernel's bits differ there)
mma_one_row: set[tuple[int, int, int]] = set()


class Prologue(NamedTuple):
    """Compute packed bf16 LOAD8(r, j) from X and ordered extra inputs, exactly matching the unfused stored values before unchanged matmul arithmetic."""

    name: str
    load8: str
    inputs: tuple[str, ...] = ()
    header: str = ""


_DEFAULT = Prologue("x", "(((const device uint4*)X)[size_t(r) * (K / 8) + (j)])")


def _compiled(kind: str, consts: tuple[tuple[str, int], ...], dep: bool = False, prologue: Prologue = _DEFAULT
              ) -> Any:
    """Compile per kind, constants and prologue, embedding constants in source to avoid MLX's per-call template regex."""

    key = (kind, consts, dep, prologue.name, XF_BF16 and kind in ("prep", "mmaf", "mmaf_rg"))
    kernel = _kernels.get(key)
    ent = kind.endswith("_ent")                      # PLI : entrée en entiers de 16 bits, sortie en 32 bits
    if kernel is None and ent:
        body = {"scalar_ent": _SCALAR, "mma_ent": _MMA}[kind].replace("bfloat(", "float(")
        source = ("".join(f"  constexpr int {k} = {v};\n" for k, v in consts)
                  + f"  #define LOAD8(r, j) ({prologue.load8})\n" + body + "  #undef LOAD8\n")
        name = f"simd_qmm_{kind}_" + hashlib.sha256((_HEADER_ENT + source).encode()).hexdigest()[:16]
        kernel = _kernels[key] = mx.fast.metal_kernel(name=name, input_names=["X", "W", "SC", "BI", "ONE"],
                                                      output_names=["OUT"], source=source, header=_HEADER_ENT)
    if kernel is None:
        if kind in ("scalar_rg", "mma_rg", "mmaf_rg"):        # PLI (P2 b+c) : rangement rg
            mma_rg = _mma_rg(_MMA)
            body = {"scalar_rg": _scalar_rg(_SCALAR), "mma_rg": mma_rg,
                    "mmaf_rg": (_fragment_source_bf16 if XF_BF16 else _fragment_source)(mma_rg)}[kind]
        elif XF_BF16 and kind in ("prep", "mmaf"):            # PLI (P2 a)
            body = {"prep": _PREP_BF16, "mmaf": _fragment_source_bf16(_MMA)}[kind]
        else:
            body = {"scalar": _SCALAR, "mma": _MMA, "prep": _PREP, "mmaf": _fragment_source(_MMA)}[kind]
        source = ("".join(f"  constexpr int {k} = {v};\n" for k, v in consts)
                  + f"  #define LOAD8(r, j) ({prologue.load8})\n" + body + "  #undef LOAD8\n")
        header = _HEADER + prologue.header
        name = (f"simd_qmm_{kind}_{prologue.name}_" + hashlib.sha256((header + source).encode()).hexdigest()[:16]
                + ("_dep" if dep else ""))
        if kind == "prep":
            inputs, outputs = ["X", "ONE", *prologue.inputs], ["XF", "XS"]
        elif kind in ("mmaf", "mmaf_rg"):
            inputs, outputs = ["XF", "XS", "W", "SC", "BI", "ONE"], ["OUT"]
        else:
            inputs, outputs = ["X", "W", "SC", "BI", "ONE", *prologue.inputs], ["OUT"]
        inputs = inputs + (["DEP"] if dep else [])
        kernel = _kernels[key] = mx.fast.metal_kernel(name=name, input_names=inputs, output_names=outputs,
                                                      source=source, header=header)
    return kernel


def splits(n: int, k: int) -> int:
    """Choose chunks by weight shape alone; stacking projections can change the reduction tree and therefore the bits."""

    # Use more chunks for small outputs to supply independent one-row FMA chains.
    return 32 if n <= 64 else (16 if n <= 6144 else 8)


def tiles(n: int, rows: int, s: int) -> int:
    """Choose output tiles within the split reduction's 16 KB threadgroup limit without changing arithmetic."""

    nt = 4 if n % 32 == 0 else (2 if n % 16 == 0 else 1)
    while nt > 1 and s * ((rows + 7) // 8) * nt * 64 * 4 > 16384:
        nt //= 2
    return nt


def scalar_block(rows: int, s: int, group: int = GROUP) -> int:
    """Choose staged input groups as a multiple of chunk count s within 20 KB, or return 0 if the scalar kernel cannot fit the rows."""

    if not 1 <= rows <= SCALAR_ROWS:
        return 0
    xb = XB if rows == 1 else max(s, XB // 2)
    return xb if xb % s == 0 and rows * xb * (76 if group == 64 else 44) * 4 <= 20480 else 0


def scalar_kind(rows: int, n: int, dims: int, group: int = GROUP) -> bool:
    """Select scalar or MMA by row count and shape; their bits agree where check passes."""

    limit = 3 if group == 32 and n > 6144 else 2
    return rows <= limit and bool(scalar_block(rows, splits(n, dims), group)) and (n, dims, group) not in mma_one_row


def rangee(scales: mx.array) -> bool:
    """PLI (P2 b+c) : une matrice au rangement rg porte, à la place des échelles, des mots de 32 bits (échelle + décalage)."""

    return scales.dtype == mx.uint32


def fits(module: Any) -> bool:
    """Require 4-bit affine weights, groups of 32 or 64, bf16 scales, inputs divisible by 64 and outputs divisible by 8."""

    weight = module["weight"]
    return (module.bits == 4 and module.group_size in (32, 64) and module["scales"].dtype == mx.bfloat16
            and weight.ndim == 2 and (int(weight.shape[1]) * 8) % module.group_size == 0 and int(weight.shape[0]) % 8 == 0
            and getattr(module, "mode", "affine") == "affine")


def _launch(kind: str, rows: int, n: int, dims: int, group: int = GROUP, most: int = MMA_SGS) -> tuple:
    """Return constants, grid, threadgroup and output shapes; an MMA launch uses up to ``most`` simdgroups."""

    s = splits(n, dims)
    if kind == "scalar":
        xb = scalar_block(rows, s, group)
        assert xb
        nr = NR if n > 2048 else 1
        # 16 outputs a threadgroup share one staging of the inputs (8 simdgroups for small outputs)
        sgs = max(1, 16 // ((32 // s) * nr)) if n > 2048 else 8
        per = sgs * (32 // s) * nr
        consts = (("K", dims), ("N", n), ("S", s), ("SGS", sgs), ("NR", nr), ("XB", xb), ("GS", group), ("RS", rows))
        return consts, (-(-n // per) * sgs * 32, 1, 1), (sgs * 32, 1, 1), [(rows, n)]
    rt = min(RT_MAX, (rows + 7) // 8)          # past 16 rows the grid's y axis takes more threadgroups of two tiles
    nt = tiles(n, rt * 8, s)
    sgs = min(s, most)
    consts = (("K", dims), ("N", n), ("S", s), ("SGS", sgs), ("NT", nt), ("RT", rt), ("GS", group))
    return consts, (-(-n // (8 * nt)) * sgs * 32, -(-rows // (8 * rt)), 1), (sgs * 32, 1, 1), [(rows, n)]


def _go(kind: str, plan: tuple, dep: bool, pro: Prologue, inputs: list) -> mx.array:
    consts, grid, tg, oshape = plan
    return _compiled(kind, consts, dep, pro)(inputs=inputs, grid=grid, threadgroup=tg, output_shapes=oshape,
                                            output_dtypes=[mx.float32] if kind.endswith("_ent") else _BF16)[0]


def _run(kind: str, rows: int, n: int, dims: int, group: int, dep: bool, pro: Prologue, inputs: list) -> mx.array:
    """Launch through the call's cached plan; a new MMA plan takes the physical simdgroups its pipeline allows here."""

    key = (kind, rows, n, dims, group, dep, pro.name, XF_BF16 and kind in ("mmaf", "mmaf_rg"))
    plan = _plans.get(key)
    if plan is not None:
        return _go(kind, plan, dep, pro, inputs)
    shape = "scalar" if kind in ("scalar", "scalar_ent", "scalar_rg") else "mma"
    if shape == "scalar":
        plan = _plans[key] = _launch(shape, rows, n, dims, group)
        return _go(kind, plan, dep, pro, inputs)
    consts = _launch(shape, rows, n, dims, group)[0]
    made: list[tuple] = []

    def launch(size: int) -> mx.array:
        made.append(_launch(shape, rows, n, dims, group, size // 32))
        return _go(kind, made[-1], dep, pro, inputs)

    pipeline = (f"simd_qmm {kind}", tuple(c for c in consts if c[0] != "SGS"), dep, pro.name,
                XF_BF16 and kind in ("mmaf", "mmaf_rg"))
    out = threads.fit(pipeline, [32 * g for g in (16, 8, 4, 2, 1) if g <= dict(consts)["SGS"]], launch, inputs)
    if threads.fitted(pipeline) is not None:
        _plans[key] = made[-1]
    return out


def qmm(x: mx.array, weight: mx.array, scales: mx.array, biases: mx.array, group_size: int = GROUP, *,
        kind: str | None = None, dep: mx.array | None = None, prologue: Prologue | None = None,
        extra: Sequence[mx.array] = ()) -> mx.array:
    """Return row-exact bf16 x @ W.T; kind forces a kernel, dep adds a dependency, and prologue reads X plus ordered extra inputs."""

    global _one
    assert group_size in (32, 64)
    if _one is None:
        _one = mx.array([1.0], dtype=mx.float32)
    shape = x.shape
    x2 = x.reshape(-1, shape[-1])
    rows, dims = int(x2.shape[0]), int(x2.shape[1])
    n = int(weight.shape[0])
    if kind is None:
        kind = "scalar" if scalar_kind(rows, n, dims, group_size) else "mma"
    if rangee(scales):                                   # PLI (P2 b+c) : mêmes calculs, lecture du rangement rg
        assert group_size == GROUP and n % 32 == 0 and kind in ("scalar", "mma")
        kind += "_rg"
    pro = prologue or _DEFAULT
    assert len(extra) == len(pro.inputs)
    inputs = [x2, weight, scales, biases, _one, *extra] + ([dep] if dep is not None else [])
    return _run(kind, rows, n, dims, group_size, dep is not None, pro, inputs).reshape(*shape[:-1], n)


def qmm_ent(xi: mx.array, weight: mx.array, scales: mx.array, biases: mx.array, *, kind: str | None = None) -> mx.array:
    """PLI (lot E4) : xi [lignes, K] en entiers de 16 bits (|x| <= ENT_MAX) ; rend xi @ W.T en 32 bits, groupes de 64.

    Mêmes bits pour une ligne quel que soit le nombre de lignes de l'appel, et mêmes bits que le noyau CPU ``pli_qmm_e15``."""

    global _one
    if _one is None:
        _one = mx.array([1.0], dtype=mx.float32)
    assert xi.dtype == mx.int16 and xi.ndim == 2
    rows, dims = int(xi.shape[0]), int(xi.shape[1])
    n = int(weight.shape[0])
    if kind is None:
        kind = "scalar" if scalar_kind(rows, n, dims, GROUP) else "mma"
    return _run(kind + "_ent", rows, n, dims, GROUP, False, _DEFAULT, [xi, weight, scales, biases, _one])


def fragments(x: mx.array, *, prologue: Prologue | None = None, extra: Sequence[mx.array] = ()
              ) -> tuple[mx.array, mx.array]:
    """Prepare bit-exact pre-scaled fp32 input pairs XF in MMA lane order and group sums XS using eight left-to-right sums followed by pairwise reduction."""

    global _one
    if _one is None:
        _one = mx.array([1.0], dtype=mx.float32)
    x2 = x.reshape(-1, x.shape[-1])
    rows, dims = int(x2.shape[0]), int(x2.shape[1])
    pro = prologue or _DEFAULT
    units = -(-rows // 8) * (dims // GROUP)
    xf, xs = _compiled("prep", (("K", dims),), False, pro)(
        inputs=[x2, _one, *extra], grid=(-(-units // 4) * 128, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(units * (256 if XF_BF16 else 512),), (rows, dims // GROUP)],
        output_dtypes=[mx.uint32 if XF_BF16 else mx.float32, mx.float32])
    return xf, xs


def qmm_fragments(frags: tuple[mx.array, mx.array], weight: mx.array, scales: mx.array, biases: mx.array, *,
                  dep: mx.array | None = None) -> mx.array:
    """Multiply two or more rows from fragments(x) with exactly the bits of qmm(x, ...)."""

    global _one
    if _one is None:
        _one = mx.array([1.0], dtype=mx.float32)
    xf, xs = frags
    rows, groups = int(xs.shape[0]), int(xs.shape[1])
    dims, n = groups * GROUP, int(weight.shape[0])
    inputs = [xf, xs, weight, scales, biases, _one] + ([dep] if dep is not None else [])
    return _run("mmaf_rg" if rangee(scales) else "mmaf", rows, n, dims, GROUP, dep is not None, _DEFAULT, inputs)


def check(weight: mx.array, scales: mx.array, biases: mx.array, *, seed: int = 0, group_size: int = GROUP) -> bool:
    """Check scalar calls of 1 through SCALAR_ROWS rows against the MMA kernel's bits for this weight."""

    k, n = int(weight.shape[1]) * 8, int(weight.shape[0])
    x = (mx.random.normal((8, k), key=mx.random.key(seed)) * 0.5).astype(mx.bfloat16)
    full = qmm(x, weight, scales, biases, group_size, kind="mma")
    calls = [(r, 1) for r in range(8)]
    calls += [(r, m) for m in range(2, SCALAR_ROWS + 1) if scalar_block(m, splits(n, k), group_size) for r in (0, 8 - m)]
    return all(bool(mx.array_equal(qmm(x[r:r + m], weight, scales, biases, group_size, kind="scalar"),
                                   full[r:r + m]).item()) for r, m in calls)


__all__ = ["ENT_MAX", "MAX_ROWS", "Prologue", "check", "fits", "fragments", "qmm", "qmm_ent", "qmm_fragments", "splits"]
