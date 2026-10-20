"""Attention pleine d'écriture du 27B à bits égaux : un seul lancement pour toutes les lignes du tour (pli-attec, 12/10/2026).

Recopie l'arithmétique de ``sdpa_vector_2pass`` de MLX 0.32.3 (M1 Ultra, famille « d », D 256, 24 têtes de requêtes, 4 têtes de clés) :
passe 1 par blocs (la clé i va au bloc i % blocs), q = échelle x requête en fp32, produit scalaire par fil dans l'ordre puis arbre
de ``simd_sum``, maximum glissant, ``fast::exp``, partiels en bf16 ; passe 2 qui les fusionne. Texte d'arithmétique tel quel (mode « txt » :
le compilateur contracte comme pour MLX). Chaque simdgroup traite RT = 2 lignes avec une seule lecture de chaque clé et valeur.
Mesuré à bits égaux (``mx.array_equal``) sur 528 cas (sonde-attec1, partie C). ``sdpa`` rend None quand la forme ne rentre pas dans ce qui
a été vérifié : l'appelant prend alors le chemin habituel.
"""

from __future__ import annotations

from typing import Any

import mlx.core as mx

D, H, HKV = 256, 24, 4
G = H // HKV
RT = 2
MIN_CLES = 1100                                  # plus petit nombre de clés vérifié (sdpa_vector_2pass dès 1024)
MAX_LIGNES = 16
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


def _blocs(n: int) -> int:
    """Nombre de blocs de sdpa_vector_2pass (MLX 0.32.3, famille « d », 6 simdgroups) pour n clés ; 0 si la famille n'est pas vérifiée."""

    if _devc() != "d":
        return 0
    return 512 if 16384 <= n < 65536 else 1024 if n >= 65536 else 128


def _kernels():
    if not _K:
        _K.append(mx.fast.metal_kernel(name="pli_attec_p1", input_names=["Q", "K", "V", "scale", "dims", "RGA"],
                                       output_names=["PM", "PL", "PO"], source=_P1, ensure_row_contiguous=False))
        _K.append(mx.fast.metal_kernel(name="pli_attec_p2", input_names=["PM", "PL", "PO", "dims"], output_names=["OUT"], source=_P2))
    return _K


def _petit(cle, fabrique):
    v = _CONST.get(cle)
    if v is None:
        v = _CONST[cle] = fabrique()
    return v


def sdpa(queries: mx.array, bk: mx.array, bv: mx.array, L: int, scale: float) -> mx.array | None:
    """Attention des T lignes de ``queries`` [1, 24, T, 256] sur les tampons entiers ``bk``, ``bv`` [1, 4, capacité, 256] ; la ligne t voit
    L - T + t + 1 clés. Mêmes bits que T appels d'une ligne de ``mx.fast.scaled_dot_product_attention``. None si hors du vérifié."""

    if queries.ndim != 4 or bk.ndim != 4 or queries.dtype != mx.bfloat16 or bk.dtype != mx.bfloat16 or bv.dtype != mx.bfloat16:
        return None
    if queries.shape[0] != 1 or queries.shape[1] != H or queries.shape[3] != D or bk.shape[1] != HKV or bk.shape[3] != D \
            or bv.shape[:2] != bk.shape[:2] or bv.shape[3] != D or int(bk.shape[2]) < L or int(bv.shape[2]) != int(bk.shape[2]):
        return None
    T = int(queries.shape[2])
    if T < 1 or T > MAX_LIGNES or L - T + 1 < MIN_CLES:
        return None
    b0, b1 = _blocs(L - T + 1), _blocs(L)
    if b0 == 0 or b0 != b1:
        return None                                   # changement de régime dans le tour (16384 ou 65536) : chemin habituel
    RG = -(-T // RT)
    if _DIMS[0] != (L, T, b0):
        _DIMS[0], _DIMS[1] = (L, T, b0), mx.array([L, T, b0, 0, 0, 0, 0, 0], dtype=mx.int32)
    sa = _petit(("s", float(scale)), lambda: mx.array([float(scale)] * 8, dtype=mx.float32))
    rga = _petit(("rg", RG), lambda: mx.array([RG] * 8, dtype=mx.int32))
    k1, k2 = _kernels()
    pm, pl, po = k1(inputs=[queries, bk, bv, sa, _DIMS[1], rga], grid=(32 * G, b0, HKV * RG), threadgroup=(32 * G, 1, 1),
                    output_shapes=[(H * T * b0,), (H * T * b0,), (H * T * b0 * D,)],
                    output_dtypes=[mx.float32, mx.float32, mx.bfloat16])
    return k2(inputs=[pm, pl, po, _DIMS[1]], grid=(1024, H, T), threadgroup=(1024, 1, 1), output_shapes=[(1, H, T, D)],
              output_dtypes=[queries.dtype])[0]


__all__ = ["sdpa"]
