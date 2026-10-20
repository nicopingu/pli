"""PLI (13/10/2026, pli-attml2 a 6) : attention de LECTURE du 27B (parts de plus de 16 lignes, D 256, bf16) avec les produits en f16, K, V et Q convertis par groupes de
HC tetes de cles. CLASSE V3 : les bits changent (mesure : ecart de calcul contre une reference f32 divise par 2 a 3 par rapport a la 2.7.0).
Branche dans prompt_attention.py, active par PLI_ATTENTION_F16=1 (0 = origine).

Meme enchainement que prompt_attention_gemm.py (deux produits du noyau steel_gemm_fused de MLX, licence MIT Apple, masque causal pose dans le rangement de Q.Kt, softmax de
MLX en precis, parts de 128 lignes, deux parts en vol) avec trois differences :
  - les entrees des deux produits sont en f16 (K et V du groupe de tetes de cles convertis avant ses parts, Q mis a l'echelle en f16) : le f16 passe plus vite que le bf16 sur
    M1 (bf16 emule). La conversion bf16 -> f16 n'est exacte qu'entre 6,1e-5 et 65504 en valeur absolue : au-dessus, le f16 donne l'infini (puis NaN dans la softmax) ;
    en dessous, les valeurs perdent des chiffres (sous-normales) puis deviennent 0. PLI_F16_ECRETE=1 borne Q, K et V a +-65280 (plus grand bf16 que le f16 represente)
    avant la conversion : memes bits tant que rien ne deborde. Scores et probabilites en f16 (probabilites sous 6,1e-5 moins precises a tres long contexte) ;
  - les tetes de cles sont traitees par groupes de HC (defaut 2) : la copie f16 de K et V ne porte que sur HC tetes sur 4, et les scores sur 6 x HC tetes (pic de memoire plus bas) ;
  - la sortie est rendue en bf16.
Conditions (sinon l'appelant garde le chemin d'origine) : bf16, b = 1, D = 256, plus de 16 lignes, nombre de cles visibles multiple de 16 et d'au moins MIN_KEYS (une part qui
ne les remplit pas prend le SDPA de MLX en f16), tampons de pas interieur 1 et de pas de ligne 256.
"""

from __future__ import annotations

import os

import mlx.core as mx

from tensorfold.kernels.qwen.dense.v1 import prompt_attention_gemm as _g

ROWS = 128
KEY_BLOCK = 16          # MLX 0.32.2's key block before M5: a part ending inside one rounds unlike one whole call
HC = int(os.environ.get("PLI_ATTENTION_F16_HC", "2"))
QK_CFG = tuple(int(x) for x in os.environ.get("PLI_ATTENTION_F16_QK", "64,64,16,2,2").split(","))
PV_CFG = tuple(int(x) for x in os.environ.get("PLI_ATTENTION_F16_PV", "32,64,32,2,2").split(","))
MIN_KEYS = int(os.environ.get("PLI_ATTENTION_F16_MIN", "16384"))
ECRETE = os.environ.get("PLI_F16_ECRETE", "0") == "1"
BF16_MAX_F16 = 65280.0   # plus grand bfloat16 que le float16 represente (le suivant, 65536, deborde)

_SOURCE = _g._SOURCE.replace("typedef bfloat16_t T;", "typedef half T;")
assert _SOURCE != _g._SOURCE
_HEADER = None
_KER: dict = {}
_EX = None


def _noyau(cfg, tb, mask):
    global _HEADER
    if _HEADER is None:
        _HEADER = _g._deplier("mlx/backend/metal/kernels/steel/gemm/gemm.h", {"mlx/backend/metal/kernels/steel/gemm/gemm.h"}) + "\n"
    key = (cfg, tb, mask)
    if key not in _KER:
        bm, bn, bk, wm, wn = cfg
        src = (_SOURCE.replace("@BM@", str(bm)).replace("@BN@", str(bn)).replace("@BK@", str(bk)).replace("@WM@", str(wm)).replace("@WN@", str(wn))
               .replace("@TB@", "true" if tb else "false").replace("@MASK@", "true" if mask else "false"))
        _KER[key] = mx.fast.metal_kernel(name="pli_attf16_%d_%d_%d_%d_%d_%d_%d" % (bm, bn, bk, wm, wn, int(tb), int(mask)),
                                         input_names=["A", "B", "prm", "ex"], output_names=["D"], source=src, header=_HEADER,
                                         ensure_row_contiguous=False)
    return _KER[key]


def _gemm(a, b, batch, m, n, k, off, grp, cfg, tb, mask):
    global _EX
    if _EX is None:
        _EX = mx.array([float(mx.finfo(mx.float16).min)], dtype=mx.float16)
    bm, bn, _bk, wm, wn = cfg
    prm = mx.array([m, n, k, off, grp], dtype=mx.int32)
    tn, tm = -(-n // bn), -(-m // bm)
    return _noyau(cfg, tb, mask)(inputs=[a, b, prm, _EX], output_shapes=[(batch, m, n)], output_dtypes=[mx.float16],
                                 grid=(tn * 32, tm * wn, batch * wm), threadgroup=(32, wn, wm))[0]


def _applicable(queries: mx.array, keys: mx.array, values: mx.array) -> bool:
    b, hq, r, d = (int(v) for v in queries.shape)
    hk, vis = int(keys.shape[1]), int(keys.shape[2])
    return (b == 1 and d == 256 and r > 16 and hq % hk == 0 and vis % 16 == 0 and vis >= MIN_KEYS
            and queries.dtype == mx.float16 and keys.dtype == mx.float16 and values.dtype == mx.float16
            and _g._layout_ok(keys) and _g._layout_ok(values))


def _part(queries: mx.array, keys: mx.array, values: mx.array, scale: float) -> mx.array:
    """Attention causale de ``queries`` (les dernieres lignes des ``keys.shape[2]`` positions) ; entrees f16, sortie f16."""
    b, hq, r, d = (int(v) for v in queries.shape)
    hk, vis = int(keys.shape[1]), int(keys.shape[2])
    g = hq // hk
    qs = (mx.array(scale, queries.dtype) * queries).reshape(hq, r, d)
    s = _gemm(qs, keys, hq, r, vis, d, vis - r, g, QK_CFG, True, True)
    p = mx.softmax(s, axis=-1, precise=True)
    o = _gemm(p, values, hq, r, d, vis, 0, g, PV_CFG, False, False)
    return o.reshape(1, hq, r, d)


def usable(queries: mx.array, keys: mx.array, values: mx.array) -> bool:
    """Le chemin f16 convient-il ? (bf16, b 1, D 256, tetes de requete multiples des tetes de cles ; pas de matrices-tuiles de l'appareil)."""
    from tensorfold.families.qwen3_5 import tensor_units

    b, hq, _r, d = (int(v) for v in queries.shape)
    hk = int(keys.shape[1])
    return (b == 1 and d == 256 and hq % hk == 0 and not tensor_units()
            and queries.dtype == mx.bfloat16 and keys.dtype == mx.bfloat16 and values.dtype == mx.bfloat16)


def attend(queries: mx.array, keys: mx.array, values: mx.array, scale: float) -> mx.array:
    """Attention causale des ``queries.shape[2]`` dernieres des ``keys.shape[2]`` positions (parts de 128 lignes comme prompt_attention.attend), sortie bf16."""
    rows, total = int(queries.shape[2]), int(keys.shape[2])
    hq, hk = int(queries.shape[1]), int(keys.shape[1])
    g = hq // hk
    bounds, begin = [], 0
    while begin < rows:
        end = min(rows, begin + ROWS)
        if end < rows:
            end -= (total - rows + end) % KEY_BLOCK                 # its keys end on a key block
        # A short tail would select vector attention and change the full prompt's bits.
        if 0 < rows - end <= 16:
            end = rows
        bounds.append((begin, end, total - rows + end))
        begin = end
    chunks: list[mx.array] = []
    prev = None
    for h0 in range(0, hk, HC):
        h1 = min(hk, h0 + HC)
        kh, vh, qh = keys[:, h0:h1], values[:, h0:h1], queries[:, h0 * g:h1 * g]
        if ECRETE:
            kh, vh, qh = (mx.clip(x, -BF16_MAX_F16, BF16_MAX_F16) for x in (kh, vh, qh))
        kh, vh, qh = kh.astype(mx.float16), vh.astype(mx.float16), qh.astype(mx.float16)
        outs: list[mx.array] = []
        for begin, end, visible in bounds:
            qq, kk, vv = qh[:, :, begin:end], kh[:, :, :visible], vh[:, :, :visible]
            if _applicable(qq, kk, vv):
                part = _part(qq, kk, vv, scale)
            else:
                part = mx.fast.scaled_dot_product_attention(qq, kk, vv, scale=scale, mask="causal")
            mx.async_eval(part)
            if prev is not None:
                mx.eval(prev)
            prev = part
            outs.append(part)
        chunks.append(mx.concatenate(outs, axis=2))
    return mx.concatenate(chunks, axis=1).astype(mx.bfloat16)


__all__ = ["attend", "usable"]
