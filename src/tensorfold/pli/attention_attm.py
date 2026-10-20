"""Attention maison d'ÉCRITURE du 27B, par tuiles de matrices simdgroup 8 x 8 (pli-attm, 13/10/2026). CLASSE V3 : les bits ne sont PAS ceux de MLX.

Forme servie : 1 à 32 lignes de requête (1, 2 et 17 à 32 lignes depuis pli-attmd, 13/10/2026 : une ligne a les mêmes bits quel que soit le tour ; 32 = fenêtre d écriture la plus large du moteur), 24 têtes de requête, 4 têtes kv, D 256, bf16, cache = tampon plus grand dont le tour lit le début ;
la ligne t d'un tour de T lignes voit L - T + t + 1 clés. ``sdpa`` rend None hors de ce qui a été mesuré : l'appelant prend alors le chemin habituel
(attention_ecriture si allumée, sinon MLX).

Conception (mesurée dans les sondes pli-attm1 à 7, banc/criteres-tour.txt) :
 - une tête kv = 6 têtes de requête x T lignes, rangées r = 6 t + g, par tuiles de 8 lignes (une tuile = un jeu de 2 simdgroups) ; un groupe de fils
   porte toutes les tuiles de la tête kv (jusqu'à 6 tuiles = 12 simdgroups, 2 groupes à 16 lignes) : les clés et les valeurs d'un bloc de 16 clés sont
   lues UNE fois, converties en f16 (exact pour du bf16 entre 6,1e-5 et 65504 en valeur absolue ; au-delà de 65504 : infini, voir PLI_F16_ECRETE) et rangées dans la mémoire du groupe (colonnes permutées par xor : sans conflit de banques) ;
 - S = Q Kt par matrices 8 x 8 f16 à cumul f32 (chaque simdgroup prend la moitié des 256 dimensions, les deux moitiés sont additionnées dans l'ordre par la
   mémoire du groupe), softmax en ligne par bloc de 16 clés (exp2), P en f16, O += P V (cumul f32, chaque simdgroup garde la moitié des colonnes) ;
 - clés découpées en tranches de SLEN clés À BORNES FIXES (position absolue de la clé) : une ligne ne dépend ni de T, ni de L, ni de la place de sa tuile
   (mesuré pli-attm7 : mêmes bits d'une ligne entre tours de 1 à 8 lignes et de longueurs différentes) ; chaque tranche rend (m, l, O) en f32, fusionnés
   dans l'ordre des tranches par une seconde passe (sortie bf16).
Mesuré (33 000 clés, 8 lignes, M1 Ultra) : 1,24 ms par couche contre 3,00 pour attention_ecriture ; écart contre une référence f32 plus petit que celui de MLX.
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx

# PLI (13/10/2026) : PLI_F16_ECRETE=1 borne à ±65504 les valeurs de Q, K et V avant leur passage en f16 (au-delà, le f16 donne l'infini puis NaN
# dans la softmax). Identité pour toute valeur que le f16 représente déjà : mêmes bits tant qu'aucune valeur ne déborde. 0 = origine.
ECRETE = os.environ.get("PLI_F16_ECRETE", "0") == "1"

D, H, HKV, G = 256, 24, 4, 6
BK = 16                                          # clés par bloc (fait partie de l'arithmétique)
NS = 2                                           # simdgroups par tuile de 8 lignes (fait partie de l'arithmétique : ordre de la somme de S)
SLEN = 512                                       # clés par tranche (fait partie de l'arithmétique : bornes fixes par position absolue)
MIN_CLES = 1   # pli-attmd4 (13/10) : toute ligne d'écriture passe par attm, le choix du noyau ne dépend jamais du tour
MIN_LIGNES = 1
MAX_LIGNES = 32
TPG_MAX = 6                                      # tuiles de 8 lignes par groupe de fils
TPG_CHARGE: dict = {1: 3, 2: 3}                          # nombre de tuiles de 8 lignes -> tuiles du groupe de fils si plus que nécessaire : les tuiles en trop ne calculent pas
                                                 # (active faux) mais chargent les clés et les valeurs avec les autres : mêmes opérations par ligne, mêmes bits (pli-attmd2 : 1 ligne 1,089 -> 0,549 ms à 33 000 clés avec 3 tuiles)

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
  const int kbeg = sl * SLEN;
  const int kend = min(kbeg + SLEN, NK);
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

_K: dict = {}
_CONST: dict = {}
_DIMS: list = [None, None]


def _petit(cle, fabrique):
    v = _CONST.get(cle)
    if v is None:
        v = _CONST[cle] = fabrique()
    return v


def _kernels(nsg: int, rgn: int):
    cle = (nsg, rgn)
    if cle not in _K:
        pre = "".join("  constexpr int %s = %d;\n" % kv for kv in (("BK", BK), ("NS", NS), ("NSG", nsg), ("RGN", rgn), ("NACC", 1), ("NBUF", 1)))
        src = _P1.replace("half(bfx(", "h16(bfx(").replace("half(float(qpb", "h16(float(qpb") if ECRETE else _P1
        k1 = mx.fast.metal_kernel(name=("pli_attm_p1e_%d_%d" if ECRETE else "pli_attm_p1_%d_%d") % cle,
                                  input_names=["Q", "K", "V", "scale", "dims"], output_names=["PM", "PL", "PO"],
                                  source=pre + src, header=_HEADER, ensure_row_contiguous=False)
        k2 = _K.get("p2")
        if k2 is None:
            k2 = _K["p2"] = mx.fast.metal_kernel(name="pli_attm_p2", input_names=["PM", "PL", "PO", "dims"], output_names=["OUT"], source=_P2)
        _K[cle] = (k1, k2)
    return _K[cle]


def sdpa(queries: mx.array, bk: mx.array, bv: mx.array, L: int, scale: float) -> mx.array | None:
    """Attention des T lignes de ``queries`` [1, 24, T, 256] sur les tampons entiers ``bk``, ``bv`` [1, 4, capacité, 256] ; la ligne t voit
    L - T + t + 1 clés. Bits propres à ce noyau (V3). None si hors du mesuré."""

    if queries.ndim != 4 or bk.ndim != 4 or queries.dtype != mx.bfloat16 or bk.dtype != mx.bfloat16 or bv.dtype != mx.bfloat16:
        return None
    if queries.shape[0] != 1 or queries.shape[1] != H or queries.shape[3] != D or bk.shape[1] != HKV or bk.shape[3] != D \
            or bv.shape[:2] != bk.shape[:2] or bv.shape[3] != D or int(bk.shape[2]) < L or int(bv.shape[2]) != int(bk.shape[2]):
        return None
    T = int(queries.shape[2])
    if T < MIN_LIGNES or T > MAX_LIGNES or L - T + 1 < MIN_CLES:
        return None
    nrt = -(-(G * T) // 8)
    tpg = max(min(nrt, TPG_MAX), TPG_CHARGE.get(nrt, 0))
    nsg = tpg * NS
    rgn = -(-nrt // tpg)
    nsl = -(-L // SLEN)
    if _DIMS[0] != (L, T):
        _DIMS[0], _DIMS[1] = (L, T), mx.array([L, T, nsl, SLEN, 0, 0, 0, 0], dtype=mx.int32)
    sa = _petit(("s", float(scale)), lambda: mx.array([float(scale)] * 8, dtype=mx.float32))
    k1, k2 = _kernels(nsg, rgn)
    pm, pl, po = k1(inputs=[queries, bk, bv, sa, _DIMS[1]], grid=(32 * nsg, nsl, HKV * rgn), threadgroup=(32 * nsg, 1, 1),
                    output_shapes=[(H * T * nsl,), (H * T * nsl,), (H * T * nsl * D,)], output_dtypes=[mx.float32, mx.float32, mx.float32])
    return k2(inputs=[pm, pl, po, _DIMS[1]], grid=(32, H, T), threadgroup=(32, 1, 1), output_shapes=[(1, H, T, D)],
              output_dtypes=[queries.dtype])[0]


__all__ = ["sdpa"]
