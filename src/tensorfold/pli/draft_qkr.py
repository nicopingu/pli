"""PLI (01/11/2026, PLI_DRAFT_QKR=1, off; with PLI_DRAFT_STACK): q and k norms, RoPE and reordering (heads, rows) of the
DFlash2 draft in one kernel per layer, instead of about eight MLX operations (q_norm, two k_norm, three rope, transpositions).

Same computation as MLX 0.32.3 (read in mlx/backend/metal/kernels/rms_norm.metal, rope.metal and rope.cpp): RMSNorm of a 128 head:
one simdgroup, 4 elements per lane, sum of squares in order, simd_sum, precise::rsqrt(sum / 128 + eps), output
w * bf16(x * inv); non-traditional RoPE: d = j / 64, inv_freq = exp2(-d * log2(base)), theta = (scale * position) * inv_freq,
fast::cos and fast::sin, x1 cos - x2 sin and x1 sin + x2 cos in f32, rounded to bf16. Bits compared with MLX by qkrtest (M1 and
M1 Ultra): the setting is only used if the bits are the same.

Inputs: y_block (1, L, Wb) = the block's [q | k | v] (stack), y_ctx (1, S, Wc) = [k | v] of the new context rows (stack).
Outputs: q (1, H, L, 128), k and v of the block (1, HKV, L, 128), k and v of the context (1, HKV, S, 128), bf16.
"""

from __future__ import annotations

import math
import os
from typing import Any

import mlx.core as mx

ENABLED = os.environ.get("PLI_DRAFT_QKR", "0") == "1"
D = 128

_SRC = r"""
  // one simdgroup per (row, head) of one of the five outputs
  constexpr int D = 128;
  const int lane = int(thread_index_in_simdgroup);
  const int u = int(threadgroup_position_in_grid.x) * 8 + int(simdgroup_index_in_threadgroup);
  const int L = dims[0], S = dims[1], H = dims[2], HKV = dims[3];
  const int WB = dims[4], WC = dims[5], PB = dims[6], PC = dims[7];
  const int nq = L * H, nkb = L * HKV, nkc = S * HKV;
  int kind, r, h;
  if (u < nq) { kind = 0; r = u / H; h = u % H; }
  else if (u < nq + nkb) { kind = 1; r = (u - nq) / HKV; h = (u - nq) % HKV; }
  else if (u < nq + 2 * nkb) { kind = 2; r = (u - nq - nkb) / HKV; h = (u - nq - nkb) % HKV; }
  else if (u < nq + 2 * nkb + nkc) { kind = 3; r = (u - nq - 2 * nkb) / HKV; h = (u - nq - 2 * nkb) % HKV; }
  else if (u < nq + 2 * nkb + 2 * nkc) { kind = 4; r = (u - nq - 2 * nkb - nkc) / HKV; h = (u - nq - 2 * nkb - nkc) % HKV; }
  else return;                                    // whole simdgroup: same u
  const device bfloat* src;
  if (kind == 0) src = YB + size_t(r) * WB + h * D;
  else if (kind == 1) src = YB + size_t(r) * WB + H * D + h * D;
  else if (kind == 2) src = YB + size_t(r) * WB + H * D + HKV * D + h * D;
  else if (kind == 3) src = YC + size_t(r) * WC + h * D;
  else src = YC + size_t(r) * WC + HKV * D + h * D;
  device bfloat* dst;
  if (kind == 0) dst = OQ + (size_t(h) * L + r) * D;
  else if (kind == 1) dst = OKB + (size_t(h) * L + r) * D;
  else if (kind == 2) dst = OVB + (size_t(h) * L + r) * D;
  else if (kind == 3) dst = OKC + (size_t(h) * S + r) * D;
  else dst = OVC + (size_t(h) * S + r) * D;
  const int e0 = lane * 4;
  if (kind == 2 || kind == 4) {                   // values: reordering only
    for (int i = 0; i < 4; i++) dst[e0 + i] = src[e0 + i];
    return;
  }
  // RMSNorm (MLX's rms_single_row, N_READS = 4, one simdgroup)
  const device bfloat* w = (kind == 0) ? WQ : WK;
  const float eps = (kind == 0) ? fl[0] : fl[1];
  float tx[4];
  float acc = 0;
  for (int i = 0; i < 4; i++) { tx[i] = src[e0 + i]; acc += tx[i] * tx[i]; }
  acc = simd_sum(acc);
  const float inv = metal::precise::rsqrt(acc / 128 + eps);
  bfloat nv[4];
  for (int i = 0; i < 4; i++) nv[i] = w[e0 + i] * static_cast<bfloat>(tx[i] * inv);
  // RoPE (MLX's rope_impl, not traditional): pairs (j, j + 64) between lanes l and l ^ 16
  const int pos = ((kind == 3) ? PC : PB) + r;
  const float Lp = fl[3] * static_cast<float>(pos);
  const bool low = lane < 16;
  for (int i = 0; i < 4; i++) {
    const float other = simd_shuffle_xor(static_cast<float>(nv[i]), ushort(16));
    const int j = (e0 + i) & 63;
    const float dd = static_cast<float>(j) / static_cast<float>(64);
    const float inv_freq = metal::exp2(-dd * fl[2]);
    const float theta = Lp * inv_freq;
    const float costheta = metal::fast::cos(theta);
    const float sintheta = metal::fast::sin(theta);
    const float mine = static_cast<float>(nv[i]);
    float rx;
    if (low) { const float x1 = mine, x2 = other; rx = x1 * costheta - x2 * sintheta; }
    else { const float x1 = other, x2 = mine; rx = x1 * sintheta + x2 * costheta; }
    dst[e0 + i] = static_cast<bfloat>(rx);
  }
"""

_k: dict = {}


def _kernel() -> Any:
    k = _k.get("k")
    if k is None:
        from tensorfold.kernels import threads
        k = _k["k"] = mx.fast.metal_kernel(name="pli_draft_qkr", input_names=["YB", "YC", "WQ", "WK", "dims", "fl"],
                                           output_names=["OQ", "OKB", "OVB", "OKC", "OVC"], source=_SRC,
                                           header=threads.reserve(256))
    return k


def qkr(y_block: mx.array, y_ctx: mx.array, H: int, HKV: int, wq: mx.array, eps_q: float, wk: mx.array, eps_k: float,
        pos_block: int, pos_ctx: int, base: float, scale: float) -> tuple[mx.array, ...] | None:
    L, S = int(y_block.shape[-2]), int(y_ctx.shape[-2])
    WB, WC = int(y_block.shape[-1]), int(y_ctx.shape[-1])
    if (WB != (H + 2 * HKV) * D or WC != 2 * HKV * D or y_block.dtype != mx.bfloat16 or y_ctx.dtype != mx.bfloat16
            or wq.dtype != mx.bfloat16 or wk.dtype != mx.bfloat16 or S < 1 or L < 1):
        return None
    U = L * H + 2 * L * HKV + 2 * S * HKV
    dims = mx.array([L, S, H, HKV, WB, WC, int(pos_block), int(pos_ctx)], dtype=mx.int32)
    fl = mx.array([float(eps_q), float(eps_k), math.log2(float(base)), float(scale)], dtype=mx.float32)
    return tuple(_kernel()(inputs=[y_block, y_ctx, wq, wk, dims, fl], grid=(-(-U // 8) * 256, 1, 1), threadgroup=(256, 1, 1),
                           output_shapes=[(1, H, L, D), (1, HKV, L, D), (1, HKV, L, D), (1, HKV, S, D), (1, HKV, S, D)],
                           output_dtypes=[mx.bfloat16] * 5))
