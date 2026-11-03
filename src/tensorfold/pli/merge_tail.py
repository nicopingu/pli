"""PLI (S5 of the prefill plan, PLI_MERGE_TAIL): a request's final chunk merged into the chunk before it.

A request puts a checkpoint at the end of its history, a few tokens before the end of the prompt (the reply
prompt). Without this item, prefill stops at the checkpoint, keeps a copy of the cache, then reads the last tokens
(7 usually) with a full pass of the model (86 ms median on the 22/10 trace, up to 375 ms at long context).

With PLI_MERGE_TAIL=N (N > 0), when the last checkpoint is N tokens from the end or fewer, these tokens are read
in the same pass as the previous chunk. The copy of the checkpoint is made after this pass:
- attention layers: key and value cache shortened by the tail tokens (trim);
- DeltaNet layers: the recurrence state and the convolution window captured at the checkpoint, by splitting
  only the core of the recurrence (gated_delta_update) into two calls; projections, convolution and norms stay
  a single computation over all rows;
- draft context (DFlash2) shortened by the tail tokens.

Class V3: the tail tokens and the chunk that carries them are computed by other kernels (the number of rows of the
pass changes), so the first token may change; a request read again from its checkpoint no longer goes
through the same computation as a new request. Off by default (0).
"""

from __future__ import annotations

import os
from typing import Any

TAIL = max(0, int(os.environ.get("PLI_MERGE_TAIL", "0") or 0))
_installed: set = set()


def active() -> bool:
    return TAIL > 0


def _split_call(original: Any) -> Any:
    """``GatedDeltaNet.__call__`` that captures the state at row ``cache.pli_split`` when it is set (else the original)."""

    def call(self: Any, inputs: Any, mask: Any = None, cache: Any = None) -> Any:
        r = getattr(cache, "pli_split", None) if cache is not None else None
        B, S = int(inputs.shape[0]), int(inputs.shape[1])
        if r is None or not 0 < int(r) < S or B != 1 or getattr(self, "sharding_group", None) is not None:
            return original(self, inputs, mask, cache)
        import mlx.core as mx
        import mlx.nn as nn

        mod = __import__(type(self).__module__, fromlist=["gated_delta_update", "normalize_qk"])
        gated_delta_update, normalize_qk = mod.gated_delta_update, mod.normalize_qk
        r = int(r)
        cache.pli_split = None
        qkv = self.in_proj_qkv(inputs)
        z = self.in_proj_z(inputs).reshape(B, S, self.num_v_heads, self.head_v_dim)
        b = self.in_proj_b(inputs)
        a = self.in_proj_a(inputs)
        conv_state = cache[0] if cache[0] is not None else mx.zeros((B, self.conv_kernel_size - 1, self.conv_dim),
                                                                    dtype=inputs.dtype)
        if mask is not None:
            qkv = mx.where(mask[..., None], qkv, 0)
        conv_input = mx.concatenate([conv_state, qkv], axis=1)
        n_keep = self.conv_kernel_size - 1
        if cache.lengths is not None:          # several streams in one cache: outside the case handled (B = 1, null lengths)
            raise RuntimeError("PLI_MERGE_TAIL: cache with variable lengths")
        conv_r = mx.contiguous(conv_input[:, r:r + n_keep, :])      # the convolution window at row r
        cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
        conv_out = nn.silu(self.conv1d(conv_input))
        q, k, v = [
            t.reshape(B, S, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], -1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.head_k_dim, self.head_k_dim, self.head_v_dim],
            )
        ]
        state = cache[1]
        q, k = normalize_qk(q, k, inv_scale=self.head_k_dim**-0.5, eps=1e-6)
        m1 = None if mask is None else mask[:, :r]
        m2 = None if mask is None else mask[:, r:]
        out1, state_r = gated_delta_update(q[:, :r], k[:, :r], v[:, :r], a[:, :r], b[:, :r], self.A_log, self.dt_bias,
                                           state, m1, use_kernel=not self.training)
        out2, state = gated_delta_update(q[:, r:], k[:, r:], v[:, r:], a[:, r:], b[:, r:], self.A_log, self.dt_bias,
                                         state_r, m2, use_kernel=not self.training)
        cache.pli_capture = (conv_r, state_r)
        cache[1] = state
        cache.advance(S)
        out = self.norm(mx.concatenate([out1, out2], axis=1), z)
        return self.out_proj(out.reshape(B, S, -1))

    call.pli_merge_tail = True
    return call


def install(layers: Any) -> int:
    """Put the split call on the class of the model's DeltaNet layers; return the number of DeltaNet layers."""

    if not active() or not layers:
        return 0
    n = 0
    for layer in layers:
        mixer = getattr(layer, "linear_attn", None)
        if mixer is None:
            continue
        n += 1
        cls = type(mixer)
        if cls in _installed or getattr(cls.__call__, "pli_merge_tail", False):
            continue
        cls.__call__ = _split_call(cls.__call__)
        _installed.add(cls)
    if n:
        print(f"PLI merge_tail: final chunk of at most {TAIL} tokens read with the previous chunk ({n} DeltaNet layers)",
              flush=True)
    return n


def arm(cache: list[Any], row: int) -> None:
    """Before the merged pass: each layer cache receives the checkpoint's row in the pass."""

    for item in cache:
        if hasattr(item, "cache") and hasattr(item, "lengths"):      # ArraysCache (DeltaNet)
            item.pli_split = int(row)
            item.pli_capture = None


def captured(cache: list[Any]) -> dict[int, Any]:
    """After the pass: the captured states (layer index -> (convolution, recurrence)), attributes removed."""

    out: dict[int, Any] = {}
    for i, item in enumerate(cache):
        if hasattr(item, "pli_split"):
            cap = getattr(item, "pli_capture", None)
            try:
                del item.pli_split
            except AttributeError:
                pass
            try:
                del item.pli_capture
            except AttributeError:
                pass
            if cap is not None:
                out[i] = cap
    return out


def snapshot(copy: list[Any], captures: dict[int, Any], end: int, at: int) -> bool:
    """The copy of the cache (made after the merged pass, up to ``end``) brought back to the checkpoint ``at``.

    Return False if a layer cannot be brought back (the copy is then not kept)."""

    tail = int(end) - int(at)
    if tail <= 0:
        return False
    for i, item in enumerate(copy):
        if hasattr(item, "cache") and hasattr(item, "lengths"):
            cap = captures.get(i)
            if cap is None:
                return False
            item.cache = [cap[0], cap[1]] + list(item.cache[2:])
        elif type(item).__name__ == "MTPCache" and hasattr(item, "trim"):
            # PLI (24/10/2026, PLI_MTP_HEAD): the MTP head reads row t with token t + 1, so it has one row
            # fewer than the prompt at the end of prefill (end - 1); brought back to ``at`` rows, as at a checkpoint kept
            # in the middle of the prompt. Without this the checkpoint was never kept and the next request read again.
            # with PLI_MTP_WINDOW the head has only the end of the prompt: the same tail - 1 rows are removed
            if int(item.offset) > int(end) - 1 or int(item.offset) < int(end) - int(at) - 1:
                return False
            item.trim(int(end) - int(at) - 1)
        elif hasattr(item, "keys") and hasattr(item, "offset"):
            if int(item.offset) != int(end) or not hasattr(item, "trim"):
                return False
            item.trim(tail)
        else:
            proposer = getattr(item, "proposer", None)          # draft DraftSlot: context shortened
            ctx = getattr(proposer, "context", None) if proposer is not None else None
            if ctx is not None and getattr(proposer, "cache", None):
                start = int(proposer.cache[0].offset)
                if start + int(ctx.shape[1]) == int(end):
                    keep = int(ctx.shape[1]) - tail
                    if keep <= 0:
                        item.proposer = None
                    else:
                        proposer.context = ctx[:, :keep]
    return True


__all__ = ["TAIL", "active", "arm", "install", "snapshot", "captured"]
