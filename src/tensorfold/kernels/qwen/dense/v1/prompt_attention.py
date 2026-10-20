"""Bound fallback prompt attention to two causal query parts in flight."""

from __future__ import annotations

import os

import mlx.core as mx

ROWS = 128
# PLI (12/10) : PLI_ATTENTION_GROUPEE=1 passe les têtes de requête qui partagent une tête de clés en un seul appel (lignes rangées
# tête par tête) avec un masque causal explicite ; mêmes bits que l'appel causal tête par tête (essai M1 Pro ; sonde M1 Ultra à suivre).
# Seulement au-delà de 16 lignes (en dessous, MLX prend un autre noyau). 0 = origine.
GROUPEE = os.environ.get("PLI_ATTENTION_GROUPEE", "0") == "1"


def _causal(queries: mx.array, keys: mx.array, values: mx.array, scale: float) -> mx.array:
    rows = int(queries.shape[2])
    b, hq, _, d = (int(v) for v in queries.shape)
    hk = int(keys.shape[1])
    if not GROUPEE or rows <= 16 or hq == hk:
        return mx.fast.scaled_dot_product_attention(queries, keys, values, scale=scale, mask="causal")
    g, total = hq // hk, int(keys.shape[2])
    pos = mx.arange(total - rows, total)
    mask = mx.arange(total)[None, :] <= pos[:, None]
    mask = mx.broadcast_to(mask[None], (g, rows, total)).reshape(1, 1, g * rows, total)
    out = mx.fast.scaled_dot_product_attention(queries.reshape(b, hk, g * rows, d), keys, values, scale=scale, mask=mask)
    return out.reshape(b, hq, rows, d)
# PLI (12/10, pli-attlec7) : PLI_ATTENTION_GEMM=1 fait les deux produits (Q.Kt avec le masque causal dans le rangement, P.V) d'une part avec le noyau steel de MLX
# instancié avec d'autres pavés (prompt_attention_gemm.py) ; mêmes bits que le chemin non fusionné de MLX ; repli sur le SDPA de MLX quand une condition manque. 0 = origine.
GEMM = os.environ.get("PLI_ATTENTION_GEMM", "0") == "1"
# PLI (13/10, pli-attml2 a 6) : PLI_ATTENTION_F16=1 prend l'attention de lecture (au-dela de FROM_KEYS clefs, parts de plus de ROWS lignes) en f16 par groupes de tetes de cles
# (prompt_attention_f16.py) : CLASSE V3, les bits changent ; repli sur le chemin d'origine quand une condition manque. 0 = origine.
F16 = os.environ.get("PLI_ATTENTION_F16", "0") == "1"
FROM_KEYS = 4096
KEY_BLOCK = 16          # MLX 0.32.2's key block before M5: a part ending inside one rounds unlike one whole call


def _part(queries: mx.array, keys: mx.array, values: mx.array, scale: float) -> mx.array:
    if GEMM:
        from tensorfold.kernels.qwen.dense.v1 import prompt_attention_gemm as _g

        if _g.applicable(queries, keys, values):
            return _g.part(queries, keys, values, scale)
    return _causal(queries, keys, values, scale)


def attend(queries: mx.array, keys: mx.array, values: mx.array, scale: float) -> mx.array:
    """Causal attention of the last ``queries.shape[2]`` of ``keys.shape[2]`` positions."""

    rows, total = int(queries.shape[2]), int(keys.shape[2])
    if total <= FROM_KEYS or rows <= ROWS:
        return _causal(queries, keys, values, scale)
    if F16:
        from tensorfold.kernels.qwen.dense.v1 import prompt_attention_f16 as _f

        if _f.usable(queries, keys, values):
            return _f.attend(queries, keys, values, scale)
    from tensorfold.families.qwen3_5 import tensor_units

    block = 1 if tensor_units() else KEY_BLOCK
    outs: list[mx.array] = []
    begin = 0
    while begin < rows:
        end = min(rows, begin + ROWS)
        if end < rows:
            end -= (total - rows + end) % block                   # its keys end on a key block
        # A short tail would select vector attention and change the full prompt's bits.
        if 0 < rows - end <= 16:
            end = rows
        visible = total - rows + end
        part = _part(queries[:, :, begin:end], keys[:, :, :visible], values[:, :, :visible], scale)
        mx.async_eval(part)
        if outs:
            mx.eval(outs[-1])
        outs.append(part)
        begin = end
    return mx.concatenate(outs, axis=2)


__all__ = ["FROM_KEYS", "KEY_BLOCK", "ROWS", "attend"]
