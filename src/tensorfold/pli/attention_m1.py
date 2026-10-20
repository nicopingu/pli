"""PLI, rang 2 : attention causale à lecture unique pour les puces sans « tensor units » (M1 à M4).

Reprend l'algorithme de ``kernels/qwen/dense/v1/lane_attention.py`` (écrit pour M5) avec les matrices 8 x 8 par simdgroup
du M1. Toutes les lignes d'une fenêtre passent ensemble : les requêtes d'une tête clé/valeur (G têtes x T lignes) sont
rangées par tuiles de 8, et chaque tuile parcourt les clés une seule fois.
Une tuile occupe SPLIT simdgroups qui se partagent les dimensions : peu de registres par fil (limite des M1).

Arithmétique fixée par la position absolue des clés (c'est ce qui rend une ligne identique seule ou dans une fenêtre) :
- morceaux de clés à bornes fixes (table MORCEAUX : petits au début, plus grands ensuite), découpés en tuiles de TK clés,
  elles-mêmes en blocs de 8 clés ;
- score d'une requête pour une clé : un élément d'un produit de matrices 8 x 8 enchaîné sur les D / 8 blocs de dimensions ;
- softmax en ligne par tuile de TK clés (maximum, poids, somme dans un ordre fixe), les clés masquées pèsent zéro ;
- sortie : P x V par blocs 8 x 8, accumulée en fp32 ; les morceaux sont fusionnés dans l'ordre par un second noyau.
Une ligne ne dépend ni des autres lignes de sa tuile (chaque élément d'un produit 8 x 8 a sa propre chaîne), ni du
nombre de lignes, ni de la longueur du contexte au moment du calcul.

Chaînes seulement (pas d'arbre de brouillons) : la ligne t d'une fenêtre de T voit les clés 0 .. L - T + t.
"""

from __future__ import annotations

import bisect
import hashlib
import os
from typing import Any

import mlx.core as mx

# Morceaux de clés : (taille, nombre) à la suite, la dernière taille sans fin. Bornes fixées par la position absolue
# (arithmétique). Mesuré le 09/10 (noyau seul) : 512 partout vaut ou bat les tables à tailles croissantes essayées.
MORCEAUX = tuple(tuple(int(x) for x in m.split("x")) for m in os.environ.get("PLI_ATTN_MORCEAUX", "512x0").split(","))
MAX_KEYS = 1 << 20
TILE = int(os.environ.get("PLI_ATTN_TK", "64"))        # clés par pas de softmax (fixe une fois choisi : arithmétique)
MAX_QUERIES = 128
# simdgroups par tuile de 8 requêtes : ils se partagent les D / 8 pas des produits, et leurs parts de score sont sommées
# dans l'ordre (fixe une fois choisi : arithmétique). 4 : chaque simdgroup garde 8 matrices de sortie en registres.
SPLIT = int(os.environ.get("PLI_ATTN_NS", "4"))
# tuiles de 8 requêtes par simdgroup (elles partagent la lecture des clés et des valeurs) ; sans effet sur les bits
ROW_TILES = int(os.environ.get("PLI_ATTN_RT", "1"))

_HEADER = r"""
// le bf16 d'indice e (0..7) parmi 8 bf16 rangés dans un uint4, en fp32
inline float bf8(uint4 v, int e) {
  const uint w = v[e / 2];
  return as_type<float>((e % 2) ? (w & 0xFFFF0000u) : (w << 16));
}
"""

_PARTIAL = r"""
  // Un groupe de fils par (tête clé/valeur, morceau de clés, RT tuiles de 8 requêtes). Ses NS simdgroups se partagent
  // les D / 8 pas des produits (JS pas chacun) ; chaque simdgroup traite les RT tuiles sur les mêmes clés et valeurs,
  // lues et converties une seule fois. RT ne change pas les bits ; NS et TK font partie de l'arithmétique.
  const ushort lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int tile0 = int(threadgroup_position_in_grid.z) * RT;            // première tuile de 8 requêtes
  const uint hk = threadgroup_position_in_grid.x;                        // tête clé/valeur
  const int c = int(threadgroup_position_in_grid.y);                     // morceau de clés (bornes dans CS)
  const int L = dims[0], NCH = dims[1], NQ = dims[2], T8 = dims[3];      // T8 : tuiles, multiple de RT
  const int RP = 8 * T8, RQ = G * NQ;
  constexpr int DJ = D / 8, JS = DJ / NS, JU = JS / 8;   // pas d'un produit ; pas par simdgroup ; uint4 (8 bf16) par plage
  // ce fil tient les éléments [fm][fn] et [fm][fn + 1] de chaque matrice 8 x 8 du simdgroup.
  // Dimension d'un élément : DJ * (indice intérieur) + (pas j) ; chaque fil lit donc des plages contiguës.
  const int qid = int(lane) >> 2;
  const int fm = (qid & 4) | ((int(lane) >> 1) & 3);
  const int fn = ((qid & 2) << 1) | ((int(lane) & 1) << 1);
  const int j0 = sg * JS;                                                // premier pas de ce simdgroup
  const device bfloat* Kh = K + int64_t(hk) * int64_t(K_strides[1]) + j0;
  const device bfloat* Vh = V + int64_t(hk) * int64_t(V_strides[1]) + j0;
  const int64_t ks = int64_t(K_strides[2]), vs = int64_t(V_strides[2]);
  threadgroup float SP[RT * NS * 8 * TK];
  int nq[RT];                                                            // clés que voit la requête de ce fil, par tuile
  float qf[RT][JS][2];
  simdgroup_matrix<float, 8, 8> O[RT][JS];
  float m[RT], l[RT];
  for (int t = 0; t < RT; t++) {
    const int r = (tile0 + t) * 8 + fm;                                  // requête t * G + g
    nq[t] = (r < RQ) ? L - NQ + r / G + 1 : 0;
    const device bfloat* qr = Qp + (int64_t(hk) * RQ + min(r, RQ - 1)) * D + j0;   // tuile incomplète : requête relue, masquée
    for (int u = 0; u < JU; u++) {
      const uint4 ya = ((const device uint4*)(qr + DJ * fn))[u], yb = ((const device uint4*)(qr + DJ * (fn + 1)))[u];
      for (int e = 0; e < 8; e++) { qf[t][8 * u + e][0] = bf8(ya, e); qf[t][8 * u + e][1] = bf8(yb, e); }
    }
    for (int j = 0; j < JS; j++) O[t][j] = simdgroup_matrix<float, 8, 8>(0.0f);
    m[t] = -INFINITY; l[t] = 0.0f;
  }
  const int kbeg = CS[c];
  const int kend = min(int(CS[c + 1]), L);
  for (int kt = kbeg; kt < kend; kt += TK) {
    for (int b = 0; b < TK / 8; b++) {
      const int k0 = kt + 8 * b;
      const device uint4* ka = (const device uint4*)(Kh + int64_t(min(k0 + fn, L - 1)) * ks + DJ * fm);
      const device uint4* kb = (const device uint4*)(Kh + int64_t(min(k0 + fn + 1, L - 1)) * ks + DJ * fm);
      simdgroup_matrix<float, 8, 8> S[RT];
      for (int t = 0; t < RT; t++) S[t] = simdgroup_matrix<float, 8, 8>(0.0f);
      for (int u = 0; u < JU; u++) {
        const uint4 xa = ka[u], xb = kb[u];
        for (int e = 0; e < 8; e++) {
          simdgroup_matrix<float, 8, 8> bm;
          bm.thread_elements()[0] = bf8(xa, e);
          bm.thread_elements()[1] = bf8(xb, e);
          for (int t = 0; t < RT; t++) {
            simdgroup_matrix<float, 8, 8> am;
            am.thread_elements()[0] = qf[t][8 * u + e][0];
            am.thread_elements()[1] = qf[t][8 * u + e][1];
            simdgroup_multiply_accumulate(S[t], am, bm, S[t]);
          }
        }
      }
      for (int t = 0; t < RT; t++) {
        SP[((t * NS + sg) * 8 + fm) * TK + 8 * b + fn] = S[t].thread_elements()[0];
        SP[((t * NS + sg) * 8 + fm) * TK + 8 * b + fn + 1] = S[t].thread_elements()[1];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float p[RT][TK / 4];
    for (int t = 0; t < RT; t++) {
      float s[TK / 4];
      for (int i = 0; i < TK / 4; i++) {                   // clé kt + 8 (i / 2) + fn + (i % 2) : les NS parts dans l'ordre
        const int key = 8 * (i / 2) + fn + (i % 2);
        float v = SP[(t * NS * 8 + fm) * TK + key];
        for (int w = 1; w < NS; w++) v += SP[((t * NS + w) * 8 + fm) * TK + key];
        s[i] = (kt + key < nq[t]) ? v * scale[0] : -INFINITY;
      }
      float x = -INFINITY;
      for (int i = 0; i < TK / 4; i++) x = max(x, s[i]);
      x = max(x, simd_shuffle_xor(x, ushort(1)));
      x = max(x, simd_shuffle_xor(x, ushort(8)));
      const float nm = max(m[t], x);
      const float f = (x == -INFINITY) ? 1.0f : precise::exp(m[t] - nm);
      for (int i = 0; i < TK / 4; i++) p[t][i] = (s[i] == -INFINITY) ? 0.0f : precise::exp(s[i] - nm);
      float y = 0.0f;
      for (int b = 0; b < TK / 8; b++) y += p[t][2 * b] + p[t][2 * b + 1];
      y += simd_shuffle_xor(y, ushort(1));
      y += simd_shuffle_xor(y, ushort(8));
      if (x != -INFINITY) { l[t] = l[t] * f + y; m[t] = nm; }
      for (int j = 0; j < JS; j++) { O[t][j].thread_elements()[0] *= f; O[t][j].thread_elements()[1] *= f; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int b = 0; b < TK / 8; b++) {
      const device bfloat* pv = Vh + int64_t(min(kt + 8 * b + fm, L - 1)) * vs;
      const device uint4* va = (const device uint4*)(pv + DJ * fn);
      const device uint4* vb = (const device uint4*)(pv + DJ * (fn + 1));
      for (int u = 0; u < JU; u++) {
        const uint4 xa = va[u], xb = vb[u];
        for (int e = 0; e < 8; e++) {
          simdgroup_matrix<float, 8, 8> vm;
          vm.thread_elements()[0] = bf8(xa, e);
          vm.thread_elements()[1] = bf8(xb, e);
          for (int t = 0; t < RT; t++) {
            simdgroup_matrix<float, 8, 8> pm;
            pm.thread_elements()[0] = p[t][2 * b];
            pm.thread_elements()[1] = p[t][2 * b + 1];
            simdgroup_multiply_accumulate(O[t][8 * u + e], pm, vm, O[t][8 * u + e]);
          }
        }
      }
    }
  }
  for (int t = 0; t < RT; t++) {
    const int64_t row = (int64_t(hk) * NCH + c) * RP + (tile0 + t) * 8 + fm;
    for (int j = 0; j < JS; j++) {
      PO[row * D + DJ * fn + j0 + j] = O[t][j].thread_elements()[0];
      PO[row * D + DJ * (fn + 1) + j0 + j] = O[t][j].thread_elements()[1];
    }
    if (sg == 0 && (int(lane) & 9) == 0) { PM[row] = m[t]; PL[row] = l[t]; }
  }
"""

_MERGE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint hk = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;              // requête t * G + g
  const int NCH = dims[1], NQ = dims[2], T8 = dims[3];
  const int RP = 8 * T8;
  constexpr int DP = D / 32;
  float m = -INFINITY, l = 0.0f, o[DP];
  for (int i = 0; i < DP; i++) o[i] = 0.0f;
  for (int c = 0; c < NCH; c++) {
    const int64_t row = (int64_t(hk) * NCH + c) * RP + r;
    const float mc = PM[row];
    if (mc == -INFINITY) continue;
    const float lc = PL[row];
    const float nm = max(m, mc);
    const float f1 = precise::exp(m - nm), f2 = precise::exp(mc - nm);
    l = l * f1 + lc * f2;
    for (int i = 0; i < DP; i++) o[i] = o[i] * f1 + PO[row * D + lane * DP + i] * f2;
    m = nm;
  }
  const int t = int(r) / G, g = int(r) % G, h = int(hk) * G + g;
  for (int i = 0; i < DP; i++) OUT[(int64_t(h) * NQ + t) * D + lane * DP + i] = static_cast<bfloat>(o[i] / l);
"""

_kernels: dict[str, Any] = {}
_table: list[Any] = []


def _morceaux() -> tuple[list[int], Any]:
    """Bornes des morceaux (liste croissante) et le même tableau pour le noyau."""

    if not _table:
        bornes = [0]
        for i, (taille, nombre) in enumerate(MORCEAUX):
            if taille % TILE or taille < TILE:
                raise ValueError(f"pli attention : morceau de {taille} clés, pas multiple de {TILE}")
            while bornes[-1] < MAX_KEYS and (nombre <= 0 or i == len(MORCEAUX) - 1 or nombre > 0):
                bornes.append(bornes[-1] + taille)
                nombre -= 1
                if nombre == 0 and i < len(MORCEAUX) - 1:
                    break
        _table.extend([bornes, mx.array(bornes, dtype=mx.int32)])
    return _table[0], _table[1]
_sg_cap: dict[int, int] = {}


def sources() -> dict[str, str]:
    """Le texte des noyaux et leurs constantes d'arithmétique (pour la version des instantanés)."""

    return {"header": _HEADER, "partial": _PARTIAL, "merge": _MERGE, "chunk": f"{MORCEAUX}/{TILE}/{SPLIT}"}


def _kernel(name: str, consts: tuple) -> Any:
    """Un noyau par jeu de constantes, écrites dans le texte (les arguments « template » de MLX coûtent une regex par appel)."""

    key = (name, consts)
    if key not in _kernels:
        source = "".join(f"  constexpr int {k} = {v};\n" for k, v in consts) + (_PARTIAL if name == "partial" else _MERGE)
        digest = hashlib.sha256((_HEADER + source).encode()).hexdigest()[:16]
        if name == "partial":
            _kernels[key] = mx.fast.metal_kernel(
                name=f"pli_attention_partial_{digest}", input_names=["Qp", "K", "V", "scale", "dims", "CS"],
                output_names=["PO", "PM", "PL"], source=source, header=_HEADER, ensure_row_contiguous=False)
        else:
            _kernels[key] = mx.fast.metal_kernel(
                name=f"pli_attention_merge_{digest}", input_names=["PO", "PM", "PL", "dims"], output_names=["OUT"],
                source=source)
    return _kernels[key]


def sdpa(queries: mx.array, keys: mx.array, values: mx.array, scale: float, split: int = 0, row_tiles: int = 0) -> mx.array:
    """Attention causale : requêtes [1, H, T, D], clés et valeurs [1, Hkv, L, D] (vues d'un tampon de cache admises)."""

    _, H, T, D = (int(s) for s in queries.shape)
    HKV, L = int(keys.shape[1]), int(keys.shape[2])
    if D % 64 or H % HKV or T > MAX_QUERIES or T > L or TILE % 8 or L > MAX_KEYS:
        raise ValueError(f"pli attention : forme non prise q={queries.shape} k={keys.shape}")
    if queries.dtype != mx.bfloat16 or keys.dtype != mx.bfloat16 or values.dtype != mx.bfloat16:
        raise ValueError("pli attention : bf16 seulement")
    G = H // HKV
    R = G * T
    RT = max(1, min(int(row_tiles) or ROW_TILES, (R + 7) // 8))
    T8 = RT * -(-R // (8 * RT))
    RP = 8 * T8
    NS = int(split) or SPLIT
    if (D // 8) % (8 * NS):
        raise ValueError(f"pli attention : {NS} simdgroups ne divisent pas les {D // 8} pas en plages de 8")
    # requêtes rangées [tête clé/valeur, ligne, tête du groupe] ; à une ligne c'est l'ordre d'origine, rien à copier
    qp = mx.contiguous(queries if T == 1 else queries.reshape(HKV, G, T, D).transpose(0, 2, 1, 3))
    bornes, cs = _morceaux()
    nch = bisect.bisect_left(bornes, L)
    dims = mx.array([L, nch, T, T8], dtype=mx.int32)
    po, pm, pl = _kernel("partial", (("G", G), ("D", D), ("NS", NS), ("RT", RT), ("TK", TILE)))(
        inputs=[qp, keys, values, mx.array([float(scale)], dtype=mx.float32), dims, cs],
        grid=(HKV * 32 * NS, nch, T8 // RT), threadgroup=(32 * NS, 1, 1),
        output_shapes=[(HKV * nch * RP * D,), (HKV * nch * RP,), (HKV * nch * RP,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32])
    return _kernel("merge", (("G", G), ("D", D)))(
        inputs=[po, pm, pl, dims],
        grid=(HKV * 32, R, 1), threadgroup=(32, 1, 1),
        output_shapes=[(1, H, T, D)], output_dtypes=[mx.bfloat16])[0]


__all__ = ["MAX_QUERIES", "MORCEAUX", "TILE", "sdpa", "sources"]
