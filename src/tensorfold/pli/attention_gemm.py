"""PLI, rang 2, seconde voie : attention causale par les produits de matrices de MLX, exacte ligne par ligne.

Lu dans MLX 0.32.3 (mlx/backend/metal/matmul.cpp, noyau « steel_gemm_fused ») : pour un produit par lots (plus d'une
tête clé/valeur), MLX prend toujours le noyau « fused », où chaque élément du résultat est UNE chaîne d'accumulation en
32 bits, k croissant, identique quel que soit le nombre de lignes ou de colonnes ; le résultat est arrondi une fois en bf16.
Le noyau « splitk », dont le découpage dépend des tailles, ne sert que sans lot. On en tire une attention où une ligne a
les mêmes bits seule ou dans une fenêtre :

  scores  S = K x Q^T      [têtes kv, clés, requêtes]   une chaîne sur les 256 dimensions par élément
  poids   P = exp((S - max) x échelle), nuls pour les clés masquées (maximum : exact, sans ordre)
  sortie  O = P^T x V      une chaîne sur les clés, dans l'ordre ; les clés masquées ajoutent des zéros exacts
  somme   l = P^T x 1      même noyau, mêmes poids arrondis
Le nombre de clés est arrondi au multiple de 16 supérieur (lignes du tampon du cache au-delà du contexte, masquées) :
sinon MLX traite la queue des clés en premier, et l'ordre dépendrait de la longueur du contexte.
Les scores sont arrondis en bf16, comme dans la lecture du prompt par MLX (le noyau d'écriture de MLX les garde en 32 bits).
Chaînes seulement (pas d'arbre) : la ligne t d'une fenêtre de T voit les clés 0 .. L - T + t.
"""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx

MAX_QUERIES = 128
ROWS = int(os.environ.get("PLI_ATTN_ROWS", "16"))     # lignes par passe (mémoire de travail) ; sans effet sur les bits
_ones: dict[tuple[int, int], mx.array] = {}


def _tranche(q: mx.array, kb: mx.array, vb: mx.array, scale: float, L: int, T: int, t0: int, n: int) -> mx.array:
    """Les lignes t0 .. t0 + n - 1 d'une fenêtre de T : q [HKV, G, n, D], tampons [HKV, L16, D]."""

    HKV, G, _, D = (int(s) for s in q.shape)
    L16 = int(kb.shape[1])
    m = G * n
    qt = mx.swapaxes(q.reshape(HKV, m, D), 1, 2)                       # [HKV, D, m], colonne g n + t
    s = kb @ qt                                                        # [HKV, L16, m] bf16
    la = max(0, (L - T) // 16 * 16)                                    # avant : clés vues par toutes les lignes
    vues = mx.arange(la, L16)[:, None] < (L - T + 1 + t0 + mx.arange(m) % n)[None, :]
    s = mx.concatenate([s[:, :la], mx.where(vues[None], s[:, la:], -mx.inf)], axis=1)
    p = mx.exp((s.astype(mx.float32) - mx.max(s, axis=1, keepdims=True).astype(mx.float32)) * scale).astype(mx.bfloat16)
    pt = mx.swapaxes(p, 1, 2)                                          # [HKV, m, L16]
    o = (pt @ vb).astype(mx.float32)                                   # [HKV, m, D]
    un = _ones.get((HKV, L16))
    if un is None:
        if len(_ones) > 8:
            _ones.clear()
        un = _ones[(HKV, L16)] = mx.ones((HKV, L16, 8), dtype=mx.float32)
    l = (pt.astype(mx.float32) @ un)[..., :1]                          # [HKV, m, 1]
    return (o / l).astype(mx.bfloat16).reshape(HKV, G, n, D)


def sdpa(queries: mx.array, keys: mx.array, values: mx.array, scale: float, cache: Any = None, **_: Any) -> mx.array:
    """Attention causale : requêtes [1, H, T, D], clés et valeurs [1, Hkv, L, D]. ``cache`` : son tampon évite une copie."""

    _, H, T, D = (int(s) for s in queries.shape)
    HKV, L = int(keys.shape[1]), int(keys.shape[2])
    if HKV < 2 or H % HKV or T > MAX_QUERIES or T > L or D % 16:
        raise ValueError(f"pli attention : forme non prise q={queries.shape} k={keys.shape}")
    G = H // HKV
    L16 = -(-L // 16) * 16
    kb = getattr(cache, "keys", None)
    vb = getattr(cache, "values", None)
    if kb is not None and vb is not None and int(kb.shape[2]) >= L16 and int(vb.shape[2]) >= L16:
        kb, vb = kb[0, :, :L16], vb[0, :, :L16]                        # vues du tampon : pas de copie
    elif L16 == L:
        kb, vb = keys[0], values[0]
    else:                                                              # pas de tampon : complété par des zéros (copie)
        z = mx.zeros((HKV, L16 - L, D), dtype=keys.dtype)
        kb, vb = mx.concatenate([keys[0], z], axis=1), mx.concatenate([values[0], z], axis=1)
    q = queries.reshape(HKV, G, T, D)
    outs = [_tranche(q[:, :, t0:t0 + ROWS], kb, vb, float(scale), L, T, t0, min(ROWS, T - t0)) for t0 in range(0, T, ROWS)]
    out = outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)
    return out.reshape(1, H, T, D)


def sources() -> dict[str, str]:
    import inspect

    return {"gemm": inspect.getsource(_tranche), "mlx": mx.__version__}


__all__ = ["MAX_QUERIES", "sdpa", "sources"]
