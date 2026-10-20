"""PLI (lecture) : poids 4 bits dépliés avant le produit sur les gros appels de la lecture (PLI_LECTURE_DEPLIE).

Sonde pli-deq1 (M1 Ultra, 12/10/2026) : à 2 048 lignes, « déplier puis produit » coûte 37 ms contre 43 pour quantized_matmul
(gate du 27B) en bf16, 26 ms en f16. Seule la lecture est touchée (appels d'au moins PLI_LECTURE_DEPLIE_LIGNES lignes) ;
l'écriture passe par le noyau à lignes exactes, jamais par ici. Les bits changent : bascule de référence, à générer sans
et avec avant toute pose.

  PLI_LECTURE_DEPLIE = 0 (défaut, éteint) | bf16 (déplier en bf16 partout) | f16 (f16 pour les projections dont l'entrée est
                       l'état normé : gate, up, in_proj_qkv, in_proj_z, q, k, v ; bf16 pour les autres)
  PLI_LECTURE_DEPLIE_LIGNES = 512 : nombre de lignes à partir duquel on déplie
  PLI_LECTURE_DEPLIE_CONTROLE = 1 : compte les sorties f16 infinies ou non numériques (force une évaluation : essai seulement)
  PLI_LECTURE_DEPLIE_MLP = 0 (défaut) | t : avec PLI_LECTURE_DEPLIE=f16, le MLP dense (Qwen3NextMLP) calculé transposé : gT = wg @ x.T,
                       hT = swiglu contigu, yT = wd @ hT. Mêmes opérations que le chemin f16 ci-dessus, mêmes bits (sondes pli-mlph1,
                       pli-mlph2b, pli-mlph3) ; seul le pavé du produit de down change (mlx 0.32.3, matmul.cpp : down passe du pavé
                       « nt grand K » 64 x 32 x 32 au pavé nn 32 x 64 x 16, la somme sur K se fait dans le même ordre) ; le passage
                       f16 -> bf16 des sorties de gate et up est fusionné au swiglu.
  PLI_LECTURE_DEPLIE_NOYAU = 0 (défaut, éteint) | 1 : le dépliage lui-même (mx.dequantize) est remplacé par un noyau Metal maison
                       à une passe, MÊMES VALEURS (sondes pli-range4 et pli-dqm1 : mx.array_equal) : en f16 comme en bf16 MLX calcule
                       en bf16 (échelle et décalage arrondis en bf16, scale * d + bias en bf16) puis convertit ; le noyau fait de même.
                       Environ 3 fois plus rapide (gate f16 à 2 048 lignes : 1,45 contre 0,50 ms, pli-range4). Repli sur mx.dequantize pour
                       toute forme non vérifiée (4 bits, group_size 64, affine, poids uint32 à 2 dimensions, échelles et décalages bf16 de
                       forme (N, K/64), K multiple de 64, sortie f16 ou bf16). Les entrées passent en lignes contiguës (garanti par
                       mx.fast.metal_kernel, ensure_row_contiguous) : une vue de lignes d'une pile de row_matmul est donc lue à son décalage.
  PLI_LECTURE_DEPLIE_NN = 0 (défaut) | 1 : (a) les projections dépliées en bf16 (out_proj, o_proj ; down si PLI_LECTURE_DEPLIE_MLP est
                       éteint) calculées en nn : y = (w @ contigu(x.T)).T recopié en lignes (sondes pli-mat1, pli-mat2 : out_proj 13,17 ->
                       11,77 ms à 2 048 lignes, bits égaux de 512 à 4 096 lignes ; mlx 0.32.3 : même pavé 64 x 64 x 16, même ordre sur K) ;
                       (b) le brouillon : fc (forme nn) et k_proj, v_proj (x @ w.T) dépliés en bf16 au lieu de quantized_matmul, hors
                       split-K (sondes pli-mat2, pli-mat3 : bits égaux à quantized_matmul dès que ceil(M/16) x ceil(N/16) > 2 048 ;
                       à 2 047 lignes fc 63,33 -> 48,06 ms, k, v des 5 couches 25,84 -> 20,28).
  PLI_LECTURE_DEPLIE_PILE = 0 (défaut) | 1 : avec PLI_LECTURE_DEPLIE=f16, in_proj_qkv + in_proj_z (GDN) et q_proj + k_proj + v_proj
                       (attention) en un seul produit nn sur la pile de row_matmul (vues, pas de mémoire en plus), sorties découpées après
                       (sonde pli-mat2 : 25,40 -> 24,94 et 22,36 -> 21,94 ms à 2 048 lignes, bits égaux).
                       Pas de pile quand un membre seul passerait par le split-K de MLX (_splitk : k_proj, v_proj à 512 lignes ; sonde pli-lotlec1 :
                       bits différents de la pile) : ce membre-là reste en produit séparé.
  Avec NOYAU allumé en même temps, le dépliage de ces chemins (nn, pile, brouillon) passe par le noyau maison (mêmes valeurs), puis le produit
  par la forme nn (lot pli-lotlec, 13/10/2026 : mat-nn.diff reporté à la main sur le dépliage maison).
"""

from __future__ import annotations

import os
import sys
from typing import Any

MODE = os.environ.get("PLI_LECTURE_DEPLIE", "0")
LIGNES = int(os.environ.get("PLI_LECTURE_DEPLIE_LIGNES", "512"))
CONTROLE = os.environ.get("PLI_LECTURE_DEPLIE_CONTROLE", "0") == "1"
MLP_T = os.environ.get("PLI_LECTURE_DEPLIE_MLP", "0") == "t"
NOYAU = os.environ.get("PLI_LECTURE_DEPLIE_NOYAU", "0") == "1"
NN = os.environ.get("PLI_LECTURE_DEPLIE_NN", "0") == "1"
PILE = os.environ.get("PLI_LECTURE_DEPLIE_PILE", "0") == "1"
F16 = ("gate_proj", "up_proj", "in_proj_qkv", "in_proj_z", "q_proj", "k_proj", "v_proj")
MIN_POIDS = 1 << 22          # pas les petites projections (in_proj_b, in_proj_a : 48 sorties)
_ATTR = "_pli_deplie"
_orig: Any = None
compte = {"appels": 0, "f16": 0, "bf16": 0, "hors_plage": 0, "mlp_t": 0, "noyau": 0, "repli_noyau": 0,
          "nn": 0, "brouillon": 0, "pile": 0}
_classes: dict = {}
_swc: Any = None


def actif() -> bool:
    return MODE in ("bf16", "f16")


_noyaux: dict = {}


def _noyau(T: str, K: int, rg: bool = False) -> Any:
    """Noyau Metal de dépliage 4 bits, un fil par (ligne, groupe de 64, quart de groupe) : 8 octets de poids, 16 valeurs.
    Expression de pli-range4 « BF » : échelle et décalage arrondis en bf16, scale * d + bias en bf16, puis conversion en T."""

    import hashlib

    import mlx.core as mx

    cle = (T, K, rg)
    if cle not in _noyaux:
        G = K // 64
        if rg:      # PLI (P2 b+c) : rangement rg (pli/range_poids.py), mêmes octets aux autres adresses, mêmes valeurs
            lecture = ("  const size_t p = row / 32, t = (row / 8) % 4, fm = row % 8;\n"
                       "  const uint2 wv = W2[((p * G + g) * 32 + fm * 4 + fnh) * 4 + t];\n"
                       "  const uint sbw = SC[((p * G + g) * 8 + fm) * 4 + t];\n"
                       "  const float scf = as_type<float>(sbw << 16), bif = as_type<float>(sbw & 0xFFFF0000u);\n")
        else:
            lecture = ("  const uint2 wv = W2[row * (K / 16) + 4 * g + fnh];\n"
                       "  const size_t sidx = row * G + g;\n  const float scf = float(SC[sidx]), bif = float(BI[sidx]);\n")
        src = (f"  constexpr int K = {K}, G = {G};\n  typedef {T} T;\n  const size_t i = thread_position_in_grid.x;\n"
               "  const size_t fnh = i & 3, g = (i >> 2) % G, row = i / (4 * G);\n"
               "  const device uint2* W2 = (const device uint2*)W;\n" + lecture +
               "  device T* o = OUT + row * K + g * 64 + fnh * 16;\n"
               "  const bfloat sb_ = bfloat(scf), bb_ = bfloat(bif);\n"
               "  for (int k = 0; k < 16; k++) {\n"
               "    const uchar d = uchar(((k < 8 ? wv.x : wv.y) >> (4 * (k & 7))) & 0xFu);\n"
               "    const bfloat v_ = sb_ * d + bb_;\n    o[k] = T(float(v_));\n  }\n")
        _noyaux[cle] = mx.fast.metal_kernel(name="pli_dqm_" + hashlib.sha256(src.encode()).hexdigest()[:12],
                                            input_names=["W", "SC", "BI"], output_names=["OUT"], source=src, header="")
    return _noyaux[cle]


def _deplier(m: Any, dt: Any) -> Any:
    """Poids de la projection m dépliés en dt : noyau maison si PLI_LECTURE_DEPLIE_NOYAU=1 et forme vérifiée, sinon mx.dequantize."""

    return _deplier_wsb(m["weight"], m["scales"], m["biases"], m.bits, m.group_size, getattr(m, "mode", "affine"), dt)


def _deplier_wsb(w: Any, s: Any, b: Any, bits: Any, group_size: Any, mode: str, dt: Any) -> Any:
    """Le même dépliage sur des tableaux (une pile de row_matmul n'est pas un module) : mêmes conditions, même repli."""

    import mlx.core as mx

    if s.dtype == mx.uint32:                       # PLI (P2 b+c) : matrice rangée, seul le noyau maison sait la lire
        N, K = int(w.shape[0]), int(w.shape[1]) * 8
        if not (NOYAU and dt in (mx.float16, mx.bfloat16) and int(bits) == 4 and int(group_size) == 64 and N % 32 == 0
                and K % 64 == 0 and N * (K // 64) * 4 < (1 << 31)):
            raise RuntimeError("PLI deplie : matrice rangée hors des formes du noyau maison")
        compte["noyau"] += 1
        return _noyau("half" if dt == mx.float16 else "bfloat", K, True)(
            inputs=[w, s, b], grid=(N * (K // 64) * 4, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(N, K)], output_dtypes=[dt])[0]
    if NOYAU:
        ok = (dt in (mx.float16, mx.bfloat16) and int(bits) == 4 and int(group_size) == 64
              and mode == "affine" and w.dtype == mx.uint32 and w.ndim == 2
              and s.dtype == mx.bfloat16 and b.dtype == mx.bfloat16)
        if ok:
            N, K = int(w.shape[0]), int(w.shape[1]) * 8
            ok = K % 64 == 0 and tuple(s.shape) == (N, K // 64) and tuple(b.shape) == (N, K // 64) and N * (K // 64) * 4 < (1 << 31)
        if ok:
            compte["noyau"] += 1
            return _noyau("half" if dt == mx.float16 else "bfloat", K)(
                inputs=[w, s, b], grid=(N * (K // 64) * 4, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(N, K)], output_dtypes=[dt])[0]
        compte["repli_noyau"] += 1
    return mx.dequantize(w, s, b, group_size=group_size, bits=bits, dtype=dt)


def _splitk(rows: int, n: int, k: int) -> bool:
    """Le produit (rows x k) @ (k x n) passerait par le split-K de MLX (mlx 0.32.3, matmul.cpp, appareil « d ») : là, quantized_matmul et
    « déplier puis produit » ne rendent pas les mêmes bits (sonde pli-mat3 : seuil observé exactement à cette règle, déduit)."""

    return ((rows + 15) // 16) * ((n + 15) // 16) <= 2048 and k // 16 >= 8 and k >= max(rows, n)


def _nn(x2: Any, w: Any) -> Any:
    """(w @ contigu(x2.T)).T recopié en lignes : mêmes bits que x2 @ w.T (même pavé, même ordre sur K), autre parcours."""

    import mlx.core as mx

    return mx.contiguous((w @ mx.contiguous(x2.T)).T)


def _rg_backend(self: Any, x: Any) -> Any:
    """PLI (P2 b+c) : une projection rangée hors dépliage (moins de LIGNES lignes, ou non marquée) passe par le noyau à lignes
    exactes de l'écriture, qui sait lire le rangement. Le produit de MLX ne le sait pas. Les bits diffèrent de ceux du produit
    de MLX (bascule acceptée le 15/10/2026, à mesurer sans et avec)."""

    from tensorfold.kernels.qwen.dense.v1 import row_matmul

    compte["rg_ecriture"] = compte.get("rg_ecriture", 0) + 1
    y = row_matmul.BACKEND(x, self["weight"], self["scales"], self["biases"], self.group_size, self.bits)
    return y + self["bias"] if "bias" in self else y


def _call(self: Any, x: Any) -> Any:
    import mlx.core as mx

    sorte = getattr(self, _ATTR, None)
    if getattr(self, "_pli_range", False) and (sorte is None or x.ndim < 2 or x.dtype != mx.bfloat16
                                               or x.size // int(x.shape[-1]) < LIGNES):
        return _rg_backend(self, x)
    if sorte is None or x.ndim < 2:
        return _orig(self, x)
    rows = x.size // int(x.shape[-1])
    if rows < LIGNES or x.dtype != mx.bfloat16:
        return _orig(self, x)
    reserve = getattr(self, "_pli_reserve", None)
    if reserve is not None:
        object.__setattr__(self, "_pli_reserve", None)
        if reserve[0] is x:                       # part déjà calculée par le premier membre de la pile, même entrée
            return reserve[1]
    if sorte in ("bf16n", "bf16b"):               # brouillon (PLI_LECTURE_DEPLIE_NN) : hors split-K seulement
        n, k = int(self["weight"].shape[0]), int(x.shape[-1])
        if _splitk(rows, n, k):
            return _orig(self, x)
        compte["appels"] += 1
        compte["brouillon"] += 1
        w = _deplier(self, mx.bfloat16)
        x2 = x.reshape(rows, k)
        y = (_nn(x2, w) if sorte == "bf16n" else x2 @ w.T).reshape(*x.shape[:-1], n)
        return y + self["bias"] if "bias" in self else y
    compte["appels"] += 1
    pile = getattr(self, "_pli_pile", None) if PILE else None
    if pile is not None and sorte == "f16" and not CONTROLE and not any(
            _splitk(rows, int(m["weight"].shape[0]), int(x.shape[-1])) for m in pile[1]):
        # (pas de pile quand un membre seul passerait par le split-K de MLX : k, v à 512 lignes, sonde pli-lotlec1 : bits différents)
        (pw, ps, pb), membres = pile
        k = int(x.shape[-1])
        w = _deplier_wsb(pw, ps, pb, self.bits, self.group_size, getattr(self, "mode", "affine"), mx.float16)
        y = _nn(x.reshape(rows, k).astype(mx.float16), w).astype(mx.bfloat16)
        y = y.reshape(*x.shape[:-1], int(y.shape[-1]))
        o, parts = 0, []
        for m in membres:
            t = int(m["weight"].shape[0])
            parts.append(y[..., o:o + t])
            o += t
        for m, part in zip(membres[1:], parts[1:]):
            object.__setattr__(m, "_pli_reserve", (x, part))
        compte["pile"] += 1
        compte["f16"] += 1
        return parts[0]
    if sorte == "f16":
        compte["f16"] += 1
        w = _deplier(self, mx.float16)
        y16 = x.astype(mx.float16) @ w.T
        if CONTROLE:
            n = int(mx.sum(mx.logical_not(mx.isfinite(y16))).item())
            if n:
                compte["hors_plage"] += n
                print(f"PLI deplie : {n} valeurs hors plage en f16 ({self._pli_nom})", file=sys.stderr, flush=True)
        y = y16.astype(mx.bfloat16)
    else:
        compte["bf16"] += 1
        w = _deplier(self, mx.bfloat16)
        if NN:
            compte["nn"] += 1
            k = int(x.shape[-1])
            y = _nn(x.reshape(rows, k), w).reshape(*x.shape[:-1], int(w.shape[0]))
        else:
            y = x @ w.T
    if "bias" in self:
        y = y + self["bias"]
    return y


def _mlp_t(self: Any, x: Any) -> Any:
    """PLI_LECTURE_DEPLIE_MLP=t : down(swiglu(gate(x), up(x))) calculé transposé ; repli sur l'appel d'origine hors lecture."""

    import mlx.core as mx

    global _swc
    g, u, d = self.gate_proj, self.up_proj, self.down_proj
    rows = x.size // int(x.shape[-1]) if x.ndim >= 2 else 0
    if (rows < LIGNES or x.dtype != mx.bfloat16 or CONTROLE or "bias" in g or "bias" in u or "bias" in d
            or getattr(g, _ATTR, None) != "f16" or getattr(u, _ATTR, None) != "f16" or getattr(d, _ATTR, None) != "bf16"):
        return type(self)._pli_base.__call__(self, x)
    if _swc is None:
        from mlx_lm.models.activations import swiglu

        _swc = mx.compile(lambda a, b: swiglu(a.astype(mx.bfloat16), b.astype(mx.bfloat16)), shapeless=True)
    compte["mlp_t"] += 1
    compte["appels"] += 3
    compte["f16"] += 2
    compte["bf16"] += 1
    q = _deplier
    xt = x.reshape(rows, int(x.shape[-1])).astype(mx.float16).T
    ht = _swc(q(g, mx.float16) @ xt, q(u, mx.float16) @ xt)
    y = mx.contiguous((q(d, mx.bfloat16) @ ht).T)
    return y.reshape(*x.shape[:-1], int(y.shape[-1]))


def _installer_mlp(couches: Any) -> int:
    """Pose _mlp_t sur les MLP denses dont gate, up (f16) et down (bf16) sont marqués ; pas sur un MLP déjà pris (PLI_MLP_ENT)."""

    try:
        from mlx_lm.models.qwen3_next import Qwen3NextMLP
    except Exception:
        return 0
    n = 0
    for couche in couches:
        m = getattr(couche, "mlp", None)
        if not isinstance(m, Qwen3NextMLP) or getattr(type(m), "_pli_ent", False) or getattr(type(m), "_pli_mlp_t", False):
            continue
        if (getattr(m.gate_proj, _ATTR, None) != "f16" or getattr(m.up_proj, _ATTR, None) != "f16"
                or getattr(m.down_proj, _ATTR, None) != "bf16"):
            continue
        base = type(m)
        if base not in _classes:
            _classes[base] = type(base.__name__ + "T", (base,), {"__call__": _mlp_t, "_pli_mlp_t": True, "_pli_base": base})
        m.__class__ = _classes[base]
        n += 1
    return n


def _installer_piles(couches: Any) -> int:
    """PLI_LECTURE_DEPLIE_PILE : sur le premier membre (in_proj_qkv, q_proj), les vues de la pile de row_matmul couvrant les membres f16."""

    from tensorfold.kernels.qwen.dense.v1 import row_matmul

    n = 0
    for couche in couches:
        for parent, kind, noms in ((getattr(couche, "linear_attn", None), "in", ("in_proj_qkv", "in_proj_z")),
                                   (getattr(couche, "self_attn", None), "qkv", ("q_proj", "k_proj", "v_proj"))):
            if parent is None:
                continue
            stack = row_matmul.stack_of(parent, kind)
            membres = [getattr(parent, nom, None) for nom in noms]
            if stack is None or any(m is None or getattr(m, _ATTR, None) != "f16" for m in membres):
                continue
            if tuple(stack.members[:len(membres)]) != tuple(membres):
                continue
            t = sum(int(m["weight"].shape[0]) for m in membres)
            vues = (stack.weight[:t], stack.scales[:t], stack.biases[:t])
            object.__setattr__(membres[0], "_pli_pile", (vues, tuple(membres)))
            n += 1
    return n


def installer_brouillon(brouillon: Any) -> int:
    """PLI_LECTURE_DEPLIE_NN : marque fc (« bf16n ») et k_proj, v_proj (« bf16b ») du brouillon ; rend le nombre de marquées."""

    global _orig
    model = getattr(brouillon, "model", None)
    if not (actif() and NN) or model is None:
        return 0
    import mlx.core as mx
    import mlx.nn as nn

    n = 0
    for nom, m in model.named_modules():
        if not isinstance(m, nn.QuantizedLinear) or getattr(m, _ATTR, None) is not None:
            continue
        if int(m.bits) != 4 or getattr(m, "mode", "affine") != "affine" or m["scales"].dtype != mx.bfloat16:
            continue
        court = nom.rsplit(".", 1)[-1]
        sorte = "bf16n" if nom == "fc" else ("bf16b" if court in ("k_proj", "v_proj") else None)
        if sorte is None:
            continue
        object.__setattr__(m, _ATTR, sorte)
        object.__setattr__(m, "_pli_nom", "brouillon." + nom)
        n += 1
    if _orig is None:
        _orig = nn.QuantizedLinear.__call__
        nn.QuantizedLinear.__call__ = _call
    print(f"PLI deplie : brouillon, {n} projections marquées (fc en nn, k et v), hors split-K", file=sys.stderr, flush=True)
    return n


def installer(couches: Any) -> int:
    """Marque les projections 4 bits des couches du modèle et pose l'appel ; idempotent ; rend le nombre de marquées."""

    global _orig
    if not actif() or couches is None:
        return 0
    import mlx.core as mx
    import mlx.nn as nn

    n = 0
    for i, couche in enumerate(couches):
        for nom, m in couche.named_modules():
            if not isinstance(m, nn.QuantizedLinear) or getattr(m, _ATTR, None) is not None:
                continue
            rg = getattr(m, "_pli_range", False) and m["scales"].dtype == mx.uint32   # PLI (P2 b+c) : matrice rangée
            if (int(m.bits) != 4 or getattr(m, "mode", "affine") != "affine" or (m["scales"].dtype != mx.bfloat16 and not rg)
                    or m["weight"].size * 8 < MIN_POIDS):
                continue
            court = nom.rsplit(".", 1)[-1]
            object.__setattr__(m, _ATTR, "f16" if (MODE == "f16" and court in F16) else "bf16")
            object.__setattr__(m, "_pli_nom", f"{i}.{nom}")
            n += 1
    t = _installer_mlp(couches) if (MLP_T and MODE == "f16") else 0
    p = _installer_piles(couches) if (PILE and MODE == "f16") else 0
    if _orig is None:
        _orig = nn.QuantizedLinear.__call__
        nn.QuantizedLinear.__call__ = _call
        print(f"PLI deplie : mode {MODE}, {n} projections marquées, à partir de {LIGNES} lignes"
              + (f", {t} MLP calculés transposés" if MLP_T else "") + (", bf16 en nn" if NN else "")
              + (f", {p} piles" if PILE else ""), file=sys.stderr, flush=True)
    return n
