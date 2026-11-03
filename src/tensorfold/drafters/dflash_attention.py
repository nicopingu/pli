"""DFlash2 attention arithmetic and sliding-window cache updates."""

from __future__ import annotations

from typing import Any

import mlx.core as mx


def _dflash_attend(attn: Any, x: Any, x_ctx: Any, rope: Any, cache: Any, masks: dict) -> Any:
    """Preserve vendor attention arithmetic with bit-exact stacked KV projections and masks shared across layers."""

    from mlx_lm.models.base import create_causal_mask

    from tensorfold.kernels.qwen.dense.v1 import lane_fuse

    B, L, _ = x.shape
    S = x_ctx.shape[1]
    if attn.is_sliding:
        keep_ctx = attn.sliding_window - 1
        if S > keep_ctx:
            skip = S - keep_ctx
            x_ctx = x_ctx[:, skip:]
            S = x_ctx.shape[1]
            cache.offset += skip
    nh, nkv = attn.n_heads, attn.n_kv_heads
    fused = _fused_qkr(attn, x, x_ctx, rope, cache, S, nh, nkv)
    if fused is not None:
        queries, prop_keys, prop_values, ctx_keys, ctx_values = fused
    else:
        # PLI (01/11/2026, PLI_DRAFT_STACK=1, off): the block's [q | k | v] and the context's [k | v] in one call each, same bits
        from tensorfold.pli import draft_stack as _bp
        qkv = _bp.project(attn, "_pli_qkv", x) if _bp.ENABLED else None
        if qkv is not None:
            queries, prop_keys, prop_values = qkv
            kvc = _bp.project(attn, "_pli_kv", x_ctx)
            ctx_keys, ctx_values = kvc if kvc is not None else (attn.k_proj(x_ctx), attn.v_proj(x_ctx))
        else:
            queries = attn.q_proj(x)
            kv_ctx = lane_fuse.attn_kv(attn, x_ctx)
            kv_x = lane_fuse.attn_kv(attn, x)
            if kv_ctx is None or kv_x is None:
                ctx_keys, ctx_values = attn.k_proj(x_ctx), attn.v_proj(x_ctx)
                prop_keys, prop_values = attn.k_proj(x), attn.v_proj(x)
            else:
                half = kv_x.shape[-1] // 2
                ctx_keys, ctx_values = kv_ctx[..., :half], kv_ctx[..., half:]
                prop_keys, prop_values = kv_x[..., :half], kv_x[..., half:]
        queries = attn.q_norm(queries.reshape(B, L, nh, -1)).transpose(0, 2, 1, 3)
        ctx_keys = attn.k_norm(ctx_keys.reshape(B, S, nkv, -1)).transpose(0, 2, 1, 3)
        ctx_values = ctx_values.reshape(B, S, nkv, -1).transpose(0, 2, 1, 3)
        prop_keys = attn.k_norm(prop_keys.reshape(B, L, nkv, -1)).transpose(0, 2, 1, 3)
        prop_values = prop_values.reshape(B, L, nkv, -1).transpose(0, 2, 1, 3)
        queries = rope(queries, offset=cache.offset + S)
        ctx_keys = rope(ctx_keys, offset=cache.offset)
        prop_keys = rope(prop_keys, offset=cache.offset + S)
    keys, values = cache.update_and_fetch(ctx_keys, ctx_values)
    ctx_len = keys.shape[2]
    # PLI (31/10/2026, PLI_DRAFT_ATTN=1, off): attention kernel of the draft (sliding window, non-causal block), bits other
    # than MLX's (V3): the draft's tokens are verified by the 27B. Outside its shape: MLX as before. (01/11: cache and block read
    # separately, without the two concatenations; same bits as the concatenated read, bstest)
    from tensorfold.pli import attention_draft as _ab
    if _ab.ENABLED and attn.is_sliding and not attn.is_causal:
        out = _ab.sdpa(queries, keys, values, ctx_len, attn.sliding_window, attn.scale, prop_keys, prop_values)
        if out is not None:
            return attn.o_proj(out)
    keys = mx.concatenate([keys, prop_keys], axis=2)
    values = mx.concatenate([values, prop_values], axis=2)
    key = (attn.is_sliding, attn.is_causal, attn.sliding_window, L, ctx_len)
    mask = masks.get(key, False)
    if mask is False:
        mask = create_causal_mask(L, offset=ctx_len) if attn.is_causal else None
        if attn.is_sliding:
            query = ctx_len + mx.arange(L)[:, None]
            k_pos = mx.arange(ctx_len + L)[None]
            context = (k_pos < ctx_len) & (query - k_pos < attn.sliding_window)
            block = k_pos >= ctx_len
            if attn.is_causal:
                block = block & (k_pos <= query)
            mask = context | block
        masks[key] = mask
    output = mx.fast.scaled_dot_product_attention(queries, keys, values, scale=attn.scale, mask=mask)
    return attn.o_proj(output.transpose(0, 2, 1, 3).reshape(B, L, -1))



def _fused_qkr(attn: Any, x: Any, x_ctx: Any, rope: Any, cache: Any, S: int, nh: int, nkv: int) -> Any:
    """PLI (01/11/2026, PLI_DRAFT_QKR=1 with PLI_DRAFT_STACK=1): stacks [q | k | v] and [k | v], then norms, RoPE and
    reordering in one kernel (same bits as MLX); None outside the served shape (the usual path takes over)."""

    from tensorfold.pli import draft_stack as _bp, draft_qkr as _qk

    if not (_bp.ENABLED and _qk.ENABLED):
        return None
    import mlx.nn as nn
    if (type(rope) is not nn.RoPE or rope.traditional or int(rope.dims) != _qk.D or type(attn.q_norm) is not nn.RMSNorm
            or type(attn.k_norm) is not nn.RMSNorm):
        return None
    yb = _bp.raw(attn, "_pli_qkv", x)
    yc = _bp.raw(attn, "_pli_kv", x_ctx) if yb is not None else None
    if yc is None:
        return None
    return _qk.qkr(yb, yc, nh, nkv, attn.q_norm.weight, attn.q_norm.eps, attn.k_norm.weight, attn.k_norm.eps,
                   cache.offset + S, cache.offset, rope.base, rope.scale)

def concat_updates(cache: list[Any]) -> list[Any]:
    """Use concatenating cache updates because one-row in-place updates can allocate negative sizes past the window."""

    for item in cache:
        if hasattr(item, "_update_concat"):
            item.update_and_fetch = item._update_concat
    return cache


__all__ = ["concat_updates"]
