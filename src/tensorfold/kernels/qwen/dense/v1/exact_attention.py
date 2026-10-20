"""Match serial attention bits by grouping queries only when their one-query calls share MLX's kernel variant and key partition."""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx

from tensorfold.kernels.qwen.dense.v1 import prompt_attention

EXACT_MAX_QUERIES = 16
# PLI_ATTENTION_ATTEC=1 (PLI-U 2.6.0) : attention d'écriture (2 à 16 lignes, 1100 clés et plus) en un seul lancement à bits égaux, voir
# pli/attention_ecriture.py (sondes pli-attec1 à 5 ; banc pli-attec27a : textes identiques, écriture + 4,5 à + 13 %). Éteint par défaut.
ATTEC = os.environ.get("PLI_ATTENTION_ATTEC", "0") == "1"
# PLI_ATTENTION_ATTM=1 (PLI-U 3.0.0, V3) : attention d'écriture maison par tuiles 8 x 8 pour toute ligne d'écriture (1 à 32 lignes), voir
# pli/attention_attm.py (bancs pli-attm27a, pli-attmd27a : écriture + 27 %, même texte d'un passage à l'autre). Éteint par défaut.
ATTM = os.environ.get("PLI_ATTENTION_ATTM", "0") == "1"
# Group queries with identical one-query arithmetic; disable when MLX dispatch differs.
# PLI (fusion « calcul réparti », 10/10) : PLI_ATTENTION_LIGNE=1 fait un appel du noyau de MLX par ligne au lieu de paquets.
GROUP_QUERIES = os.environ.get("PLI_ATTENTION_LIGNE", "0") != "1"
# PLI (banc pli-ligne du 10/10) : PLI_ATTENTION_LIGNE=N (N > 1) ne passe à un appel par ligne qu'à partir de N clés de contexte
# (mesuré : l'appel par ligne gagne de 25 000 à 66 000 jetons et perd un peu en dessous de 7 000). Mêmes bits dans les deux cas.
try:
    LIGNE_SEUIL = int(os.environ.get("PLI_ATTENTION_LIGNE", "0") or 0)
except ValueError:
    LIGNE_SEUIL = 0
if LIGNE_SEUIL > 1:
    GROUP_QUERIES = True
_STOCK: Any = None

# MLX 0.31.2 skips masked tail keys without changing key slots; grouping is exact only within one kernel regime.
_REGIME_POINTS = (1024, 1025, 4096, 8193, 16384, 32769, 65536, 65537)

# PLI, rang 2 : PLI_ATTENTION fait passer les fenêtres de 1 à 128 lignes (pas d'une ligne compris) par une attention de PLI
# où une ligne a les mêmes bits seule ou dans une fenêtre (les bits diffèrent de ceux de MLX) :
#   gemm : produits de matrices de MLX en un seul régime (tensorfold/pli/attention_gemm.py) ;
#   m1   : noyau Metal à matrices 8 x 8 (tensorfold/pli/attention_m1.py).
PLI_ATTENTION = os.environ.get("PLI_ATTENTION", "").strip().lower()
if PLI_ATTENTION in ("0", "non", "off"):
    PLI_ATTENTION = ""
if PLI_ATTENTION and PLI_ATTENTION not in ("gemm", "m1"):
    raise ValueError(f"PLI_ATTENTION={PLI_ATTENTION} : gemm ou m1")


# PLI (fusion n° 2 de « calcul réparti », 10/10) : lecture réduite du vieux contexte avec carte faite par le GPU dans le tour.
# PLI_ATTENTION_REDUITE=20 : en écriture (1 à 16 lignes), au-delà de PLI_ATTENTION_REDUITE_SEUIL clés, le GPU calcule les scores
# de toutes les vieilles clés, garde les 20 % de pavés de 32 clés au plus grand score (par tête clé/valeur, maximum sur le pavé et
# sur les rangées), puis appelle le noyau habituel sur ces clés + la queue de 1 024 jetons. Les couches PLEINES restent complètes.
# Le texte change (classe V3) : éteint par défaut (0). La couche est lue sur le cache (attribut pli_couche, posé par la famille).
def _entier(nom: str, defaut: int) -> int:
    try:
        return int(os.environ.get(nom, str(defaut)) or defaut)
    except ValueError:
        return defaut


REDUITE = max(0, min(90, _entier("PLI_ATTENTION_REDUITE", 0)))
REDUITE_SEUIL = max(4096, _entier("PLI_ATTENTION_REDUITE_SEUIL", 20000))
REDUITE_PLEINES = tuple(int(x) for x in os.environ.get("PLI_ATTENTION_REDUITE_PLEINES", "3,7,59").split(",") if x.strip())
QUEUE, PAVE = 1024, 32
REDUITE_LIGNES = 16          # tours d'écriture seulement ; la lecture du prompt reste complète
REDUITE_MIN = max(1, _entier("PLI_ATTENTION_REDUITE_MIN", 2))   # reparti-red1 (10/10) : à 1 ligne la réduction coûte plus que la complète
COMPTE = {"reduites": 0, "completes": 0, "attm": 0}
REDUITE_ATTM = os.environ.get("PLI_ATTENTION_REDUITE_ATTM", "1") == "1"   # 0 : ancien chemin (sans attm) sur les couches réduites


def _reduire(queries: mx.array, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
    """Clés et valeurs gardées : pavés de tête du vieux contexte (dans l'ordre), puis la queue entière."""

    import math

    kvh, n, d = int(keys.shape[1]), int(keys.shape[2]), int(keys.shape[3])
    lignes = int(queries.shape[2])
    rangees = (int(queries.shape[1]) // kvh) * lignes
    pavés = (n - lignes - QUEUE) // PAVE
    debut = pavés * PAVE
    garde = max(1, math.ceil(REDUITE / 100 * pavés))
    # Tout se fait tête par tête (repris du correctif 55ab43ca du 10/10, dépôt de bancs) : dans le moteur les clés sont une tranche du tampon du
    # cache ; les aplatir recopiait tout le tableau à chaque appel (essai-reduite-pieces : 7,6 ms contre 0,75 ms par tête, mêmes bits).
    qh = queries.reshape(kvh, rangees, d)
    rang = mx.arange(PAVE)[None, :]
    kt, vt = [], []
    for h in range(kvh):
        scores = qh[h] @ mx.swapaxes(keys[0, h, :debut], -1, -2)                               # [rangées, vieilles clés]
        tete = mx.max(scores.reshape(rangees, pavés, PAVE), axis=(0, 2)).astype(mx.float32)
        choix = mx.sort(mx.argsort(tete)[pavés - garde:])
        ix = (choix[:, None] * PAVE + rang).reshape(-1)
        kt.append(mx.take(keys[0, h], ix, axis=0))
        vt.append(mx.take(values[0, h], ix, axis=0))
    return (mx.concatenate([mx.stack(kt)[None], keys[:, :, debut:]], axis=2),
            mx.concatenate([mx.stack(vt)[None], values[:, :, debut:]], axis=2))


def _one(queries: mx.array, keys: mx.array, values: mx.array, scale: float, mask: Any, t: int, T: int,
         L: int) -> mx.array:
    n = L - T + t + 1
    m = mask[..., t:t + 1, :n] if isinstance(mask, mx.array) else None
    return mx.fast.scaled_dot_product_attention(
        queries[:, :, t:t + 1], keys[:, :, :n], values[:, :, :n], scale=scale, mask=m)


# PLI (16/10/2026, bancs seulement) : PLI_ATTENTION_REGLAGE=/chemin/fichier change à chaud la part gardée (« reduite=N », 0 = attention
# complète), sans relancer le serveur ni relire le contexte ; le fichier est relu quand sa date change (repris du dépôt de bancs, 30fad843).
REGLAGE = os.environ.get("PLI_ATTENTION_REGLAGE", "")
_REGLAGE_VU = [0.0]


def _reglage() -> None:
    global REDUITE
    try:
        date = os.stat(REGLAGE).st_mtime
    except OSError:
        return
    if date == _REGLAGE_VU[0]:
        return
    _REGLAGE_VU[0] = date
    import sys

    try:
        v = dict(x.split("=", 1) for x in open(REGLAGE).read().split())
        REDUITE = max(0, min(90, int(v.get("reduite", REDUITE))))
    except Exception as erreur:  # noqa: BLE001
        print("[pli] réglage à chaud illisible : %s" % erreur, file=sys.stderr, flush=True)
        return
    print("[pli] réglage à chaud : attention réduite à %d %%" % REDUITE, file=sys.stderr, flush=True)


def exact_sdpa(queries: mx.array, keys: mx.array, values: mx.array, cache: Any, scale: float,
               mask: Any, sinks: Any = None) -> mx.array:
    T = int(queries.shape[2])
    if REGLAGE:
        _reglage()
    T = int(queries.shape[2])
    if REDUITE and REDUITE_MIN <= T <= REDUITE_LIGNES and sinks is None and not hasattr(cache, "bits") and int(queries.shape[0]) == 1 \
            and int(keys.shape[2]) >= REDUITE_SEUIL and (mask is None or (isinstance(mask, str) and mask == "causal")) \
            and int(queries.shape[1]) % int(keys.shape[1]) == 0:
        couche = getattr(cache, "pli_couche", None)
        if couche is not None and couche not in REDUITE_PLEINES:
            COMPTE["reduites"] += 1
            keys, values = _reduire(queries, keys, values)
            # PLI (16/10/2026, V3) : la réduction passe par attm quand attm est allumée. Avant, les clés raccourcies ne passaient plus le test
            # d'attm (cache.offset == longueur des clés) : la couche réduite perdait attm. Les clés gardées forment un tableau d'un seul tenant
            # qui finit par la queue (dont les T lignes du tour) : la ligne t voit L - T + t + 1 clés, comme attm l'attend.
            if ATTM and REDUITE_ATTM and T <= 32 and queries.dtype == mx.bfloat16 and keys.dtype == mx.bfloat16:
                from tensorfold.pli import attention_attm

                out = attention_attm.sdpa(queries, keys, values, int(keys.shape[2]), scale)
                if out is not None:
                    COMPTE["attm"] += 1
                    return out
        else:
            COMPTE["completes"] += 1
        if (COMPTE["reduites"] + COMPTE["completes"]) % 20000 == 0:
            import sys
            print("[pli] attention réduite à %d %% : %d appels réduits (%d par attm), %d laissés complets" % (REDUITE, COMPTE["reduites"],
                  COMPTE["attm"], COMPTE["completes"]), file=sys.stderr, flush=True)
    if PLI_ATTENTION and T <= 128 and sinks is None and not hasattr(cache, "bits") \
            and ((mask is None and T == 1) or (isinstance(mask, str) and mask == "causal")) and int(queries.shape[0]) == 1 \
            and int(queries.shape[-1]) % 64 == 0 and queries.dtype == mx.bfloat16 and keys.dtype == mx.bfloat16 \
            and values.dtype == mx.bfloat16:
        if PLI_ATTENTION == "gemm":
            from tensorfold.pli import attention_gemm

            return attention_gemm.sdpa(queries, keys, values, scale, cache=cache)
        from tensorfold.pli import attention_m1

        return attention_m1.sdpa(queries, keys, values, scale)
    if T > EXACT_MAX_QUERIES and int(queries.shape[-1]) == 256 \
            and isinstance(mask, str) and mask == "causal" and sinks is None \
            and not hasattr(cache, "bits"):
        return prompt_attention.attend(queries, keys, values, scale)      # prompt chunks: bounded memory
    # PLI_ATTENTION_ATTM : TOUTE ligne d'attention complète d'écriture (1 à 16 lignes) passe par la même arithmétique, y compris à 1 ligne (mask None)
    # et à 2 lignes : le choix ne dépend que des clés et de la forme, jamais du nombre de lignes du tour (pli-attmd, 13/10/2026).
    if ATTM and sinks is None and not hasattr(cache, "bits") and int(queries.shape[0]) == 1 and T <= EXACT_MAX_QUERIES \
            and ((isinstance(mask, str) and mask == "causal") or (T == 1 and mask is None)):
        bk, bv = getattr(cache, "keys", None), getattr(cache, "values", None)
        if bk is not None and bv is not None and getattr(cache, "offset", None) is not None and int(cache.offset) == int(keys.shape[2]) \
                and bk.ndim == 4 and bk.shape[:2] == keys.shape[:2] and bk.shape[3] == keys.shape[3]:
            from tensorfold.pli import attention_attm

            out = attention_attm.sdpa(queries, bk, bv, int(keys.shape[2]), scale)
            if out is not None:
                return out
    if T < 2 or T > EXACT_MAX_QUERIES or sinks is not None or hasattr(cache, "bits"):
        return _STOCK(queries, keys, values, cache, scale, mask, sinks)
    L = int(keys.shape[2])
    if ATTEC and isinstance(mask, str) and mask == "causal" and int(queries.shape[0]) == 1:
        bk, bv = getattr(cache, "keys", None), getattr(cache, "values", None)
        if bk is not None and bv is not None and getattr(cache, "offset", None) is not None and int(cache.offset) == L \
                and bk.ndim == 4 and bk.shape[:2] == keys.shape[:2] and bk.shape[3] == keys.shape[3]:
            from tensorfold.pli import attention_ecriture

            out = attention_ecriture.sdpa(queries, bk, bv, L, scale)
            if out is not None:
                return out
    heads, kv_heads = int(queries.shape[1]), int(keys.shape[1])
    group = 1
    if GROUP_QUERIES and not (LIGNE_SEUIL > 1 and L >= LIGNE_SEUIL) and not isinstance(mask, mx.array) and heads % kv_heads == 0:
        group = max(1, min(8, 32 // (heads // kv_heads)))
    outs = []
    t = 0
    while t < T:
        g = min(group, T - t)
        first, last = L - T + t + 1, L - T + t + g
        if g > 1 and not any(first < p <= last for p in _REGIME_POINTS):
            outs.append(mx.fast.scaled_dot_product_attention(
                queries[:, :, t:t + g], keys[:, :, :last], values[:, :, :last], scale=scale, mask="causal"))
        else:
            outs.extend(_one(queries, keys, values, scale, mask, j, T, L) for j in range(t, t + g))
        t += g
    return mx.concatenate(outs, axis=2)


def install() -> None:
    """Route the Qwen3-Next / Qwen3.5 full-attention call through ``exact_sdpa``. Idempotent."""

    global _STOCK
    import mlx_lm.models.qwen3_next as qn

    if _STOCK is None:
        _STOCK = qn.scaled_dot_product_attention
    qn.scaled_dot_product_attention = exact_sdpa


__all__ = ["EXACT_MAX_QUERIES", "exact_sdpa", "install"]
