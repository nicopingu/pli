"""PLI, P2 paliers b et c : poids et échelles du 27B rangés pour l'écriture (PLI_RANGE=1, éteint par défaut).

Sondes pli-range1 à 6 (M1 Ultra, 12/10/2026) : le noyau d'écriture à 8 lignes lit les poids de 4 lignes de sortie à la fois
(4 lectures de 64 bits) et l'échelle et le décalage dans deux tableaux séparés. Rangés par paquets de 32 sorties, avec
l'échelle et le décalage réunis dans un mot de 32 bits, un fil lit tout ce qu'il lui faut en 3 lectures de 128 bits :
- 16,6 % (gate + up), - 18 % (down), - 11 à - 12 % (in_proj, qkv) par produit à 8 lignes, mêmes bits. Calcul : - 9 ms par tour.

Rangement « rg » d'une matrice de N sorties (N multiple de 32), K entrées, groupes de 64 (G = K / 64), n = 32 p + 8 t + fm :
  poids   : uint2 [p][g][fm * 4 + q][t] = uint2 n° (4 g + q) de la ligne n (q = quart du groupe, 0 à 3)
  échelles : uint32 [p][g][fm][t] = bf16 de l'échelle (16 bits bas) | bf16 du décalage (16 bits hauts)
Les tableaux gardent leur forme ([N, K / 8] pour les poids, [N, G] pour les mots) et un paquet occupe 32 lignes contiguës :
une vue de lignes alignée sur 32 (membre d'une pile de row_matmul) est encore une matrice rangée. Le tableau des mots prend
la place des échelles ; le module garde ce même tableau en guise de décalages (rien ne les lit). Les noyaux reconnaissent le
rangement au type des échelles (uint32 au lieu de bf16 : simd_qmm.rangee).

Ce qui lit ces poids, et comment :
- écriture (row_matmul, simd_qmm) : noyaux scalaire, MMA et à fragments, variantes « _rg », mêmes calculs ;
- lecture des prompts, 512 lignes et plus (pli/deplie.py) : noyau de dépliage maison, variante rg, mêmes valeurs ;
- lecture des prompts, moins de 512 lignes, et projections non marquées : passent par le noyau d'écriture au lieu du produit
  de MLX. LES BITS CHANGENT LÀ (choix du 15/10/2026 : bascule du texte, mesurée sans et avec avant toute pose) ;
- non rangés : la tête (lm_head, lue aussi par le brouillon DFlash2), les plongements, le brouillon, les membres d'une pile
  non alignés sur 32 (in_proj_b, in_proj_a : 48 sorties), qui reçoivent leur propre copie au rangement d'origine.

Conditions (sinon rien n'est rangé, et un message dit pourquoi) : PLI_LECTURE_DEPLIE à f16 ou bf16 avec
PLI_LECTURE_DEPLIE_NOYAU=1 ; ni PLI_LECTURE_CPU, ni PLI_MLP_ENT (ils lisent les poids d'origine).

Chaque matrice est vérifiée avant d'être remplacée : 1, 2 et 8 lignes, noyau d'origine contre noyau rangé, mêmes bits exigés.
Une seule différence et cette matrice reste au rangement d'origine (le reste du modèle n'en dépend pas).
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from typing import Any

ACTIF = os.environ.get("PLI_RANGE", "0") == "1"
_conv: dict = {}
etat: dict[str, Any] = {"rangees": 0, "refusees": 0, "copies": 0, "octets": 0, "raison": "", "secondes": 0.0}


def _noyau_conversion(K: int) -> Any:
    import mlx.core as mx

    if K not in _conv:
        G = K // 64
        src = (f"  constexpr int K = {K}, G = {G};\n"
               "  const uint i = thread_position_in_grid.x;          // un fil par (ligne n, groupe g)\n"
               "  const int g = int(i % G), n = int(i / G);\n"
               "  const device uint2* W2 = (const device uint2*)W;\n"
               "  device uint2* R2 = (device uint2*)WR;\n"
               "  const size_t p = size_t(n / 32), t = size_t((n / 8) % 4), fm = size_t(n % 8);\n"
               "  for (int q = 0; q < 4; q++) R2[((p * G + g) * 32 + fm * 4 + q) * 4 + t] = W2[size_t(n) * (K / 16) + 4 * g + q];\n"
               "  const uint s16 = as_type<ushort>(SC[size_t(n) * G + g]), b16 = as_type<ushort>(BI[size_t(n) * G + g]);\n"
               "  SB[((p * G + g) * 8 + fm) * 4 + t] = s16 | (b16 << 16);\n")
        _conv[K] = mx.fast.metal_kernel(name="pli_rg_conv_" + hashlib.sha256(src.encode()).hexdigest()[:12],
                                        input_names=["W", "SC", "BI"], output_names=["WR", "SB"], source=src, header="")
    return _conv[K]


def ranger(w: Any, s: Any, b: Any) -> tuple[Any, Any]:
    """(poids, échelles, décalages) d'origine -> (poids rangés, mots échelle + décalage), mêmes formes, mêmes octets."""

    import mlx.core as mx

    N, K = int(w.shape[0]), int(w.shape[1]) * 8
    G = K // 64
    wr, sb = _noyau_conversion(K)(inputs=[w, s, b], grid=(N * G, 1, 1), threadgroup=(256, 1, 1),
                                  output_shapes=[(N, K // 8), (N, G)], output_dtypes=[mx.uint32, mx.uint32])
    return wr, sb


def convenable(w: Any, s: Any, b: Any) -> bool:
    import mlx.core as mx

    return (w.dtype == mx.uint32 and w.ndim == 2 and s.dtype == mx.bfloat16 and b.dtype == mx.bfloat16
            and int(w.shape[0]) % 32 == 0 and (int(w.shape[1]) * 8) % 64 == 0
            and tuple(s.shape) == (int(w.shape[0]), int(w.shape[1]) * 8 // 64) and tuple(b.shape) == tuple(s.shape))


def verifie(w: Any, s: Any, b: Any, wr: Any, sb: Any) -> bool:
    """Mêmes bits, noyau d'origine contre noyau rangé : 1 ligne (le noyau que l'appel choisirait), 2 lignes, 8 lignes."""

    import mlx.core as mx

    from tensorfold.kernels.qwen.dense.v1 import simd_qmm as Q

    K = int(w.shape[1]) * 8
    x = (mx.random.normal((8, K), key=mx.random.key(int(w.shape[0]) * 131 + K)) * 0.5).astype(mx.bfloat16)
    for rows in (1, 2, 8):
        xr = x[:rows]
        a = Q.qmm(xr, w, s, b, 64)
        c = Q.qmm(xr, wr, sb, sb, 64)
        if rows >= 2:
            a2 = Q.qmm_fragments(Q.fragments(xr), w, s, b)
            c2 = Q.qmm_fragments(Q.fragments(xr), wr, sb, sb)
            if not bool(mx.array_equal(a2, c2).item()):
                return False
        if not bool(mx.array_equal(a, c).item()):
            return False
    return True


def _refus() -> str:
    from tensorfold.pli import deplie

    if not deplie.actif():
        return "PLI_LECTURE_DEPLIE éteint : la lecture des prompts passerait par le produit de MLX"
    if not deplie.NOYAU:
        return "PLI_LECTURE_DEPLIE_NOYAU éteint : mx.dequantize ne lit pas le rangement"
    try:
        if float(os.environ.get("PLI_LECTURE_CPU", "0") or 0) > 0:
            return "PLI_LECTURE_CPU allumé : la lecture répartie lit les poids d'origine"
    except ValueError:
        return "PLI_LECTURE_CPU illisible"
    if os.environ.get("PLI_MLP_ENT", "0") == "1":
        return "PLI_MLP_ENT allumé : le MLP en entiers lit les poids d'origine"
    return ""


def _copie(a: Any) -> Any:
    """Une copie qui ne garde pas le tableau source en vie (une vue de lignes le garderait)."""

    import mlx.core as mx

    c = mx.array(a) + mx.zeros((), dtype=a.dtype)
    mx.eval(c)
    return c


def installer(model: Any) -> dict[str, Any]:
    """Range les projections des couches (piles et projections seules) ; rend ``etat``. À appeler après row_matmul.install."""

    if not ACTIF:
        return etat
    import mlx.core as mx
    import mlx.nn as nn

    from tensorfold.kernels.qwen.dense.v1 import row_matmul

    raison = _refus()
    if raison:
        etat["raison"] = raison
        print(f"PLI range : rien de rangé ({raison})", file=sys.stderr, flush=True)
        return etat
    debut = time.perf_counter()
    lm = getattr(model, "language_model", model)
    couches = lm.model.layers
    for couche in couches:
        dans_piles: set[int] = set()
        for _, module in couche.named_modules():
            for stack in module.__dict__.get(row_matmul._ATTR, {}).values():
                if not stack.valid():
                    continue
                dans_piles.update(id(m) for m in stack.members)
                if not convenable(stack.weight, stack.scales, stack.biases):
                    etat["refusees"] += 1
                    continue
                wr, sb = ranger(stack.weight, stack.scales, stack.biases)
                mx.eval(wr, sb)
                if not verifie(stack.weight, stack.scales, stack.biases, wr, sb):
                    etat["refusees"] += 1
                    continue
                # membres : vue rangée si alignée sur 32, sinon copie au rangement d'origine (petits membres seulement)
                o, vues = 0, []
                for m, n in zip(stack.members, stack.sizes):
                    if o % 32 == 0 and n % 32 == 0:
                        vues.append((m, wr[o:o + n], sb[o:o + n], sb[o:o + n], True))
                    else:
                        vues.append((m, _copie(stack.weight[o:o + n]), _copie(stack.scales[o:o + n]),
                                     _copie(stack.biases[o:o + n]), False))
                        etat["copies"] += 1
                    o += n
                stack.weight, stack.scales, stack.biases = wr, sb, sb
                for m, w2, s2, b2, rg in vues:
                    m.weight, m.scales, m.biases = w2, s2, b2
                    object.__setattr__(m, "_pli_range", rg)
                mx.eval([a for m in stack.members for a in (m["weight"], m["scales"], m["biases"])])
                stack.held = tuple(m["weight"] for m in stack.members)
                etat["rangees"] += 1
                etat["octets"] += int(wr.nbytes + sb.nbytes)
        for _, m in couche.named_modules():
            if not isinstance(m, nn.QuantizedLinear) or id(m) in dans_piles or getattr(m, "_pli_range", None) is not None:
                continue
            if int(m.bits) != 4 or int(m.group_size) != 64 or getattr(m, "mode", "affine") != "affine" or "bias" in m:
                continue
            w, s, b = m["weight"], m["scales"], m["biases"]
            if not convenable(w, s, b):
                continue
            wr, sb = ranger(w, s, b)
            mx.eval(wr, sb)
            if not verifie(w, s, b, wr, sb):
                etat["refusees"] += 1
                continue
            m.weight, m.scales, m.biases = wr, sb, sb
            object.__setattr__(m, "_pli_range", True)
            etat["rangees"] += 1
            etat["octets"] += int(wr.nbytes + sb.nbytes)
        mx.clear_cache()
    etat["secondes"] = round(time.perf_counter() - debut, 1)
    # les projections rangées passent par le crochet de dépliage dès maintenant (il aiguille vers les noyaux rangés)
    from tensorfold.pli import deplie

    deplie.installer(couches)
    print(f"PLI range : {etat['rangees']} matrices rangées ({etat['octets'] / 2**30:.2f} Gio), {etat['refusees']} refusées, "
          f"{etat['copies']} petits membres copiés, {etat['secondes']} s", file=sys.stderr, flush=True)
    return etat


__all__ = ["ACTIF", "convenable", "etat", "installer", "ranger", "verifie"]
