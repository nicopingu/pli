"""PLI (prefill): 4-bit weights unpacked before the product on the large prefill calls (PLI_PREFILL_UNPACK).

Measured on M1 Ultra (12/10/2026): at 2,048 rows, "unpack then product" costs 37 ms against 43 for quantized_matmul
(the 27B's gate) in bf16, 26 ms in f16. Only prefill is touched (calls of at least PLI_PREFILL_UNPACK_ROWS rows);
decode goes through the exact-rows kernel, never through here. The bits change: reference switch, to generate without
and with before any rollout.

  PLI_PREFILL_UNPACK = 0 (default, off) | bf16 (unpack in bf16 everywhere) | f16 (f16 for the projections whose input is
                       the normed state: gate, up, in_proj_qkv, in_proj_z, q, k, v; bf16 for the others)
  PLI_PREFILL_UNPACK_ROWS = 512: number of rows from which weights are unpacked
  PLI_PREFILL_UNPACK_CHECK = 1: counts infinite or non-numeric f16 outputs (forces an evaluation: trials only)
  PLI_PREFILL_UNPACK_MLP = 0 (default) | t: with PLI_PREFILL_UNPACK=f16, the dense MLP (Qwen3NextMLP) computed transposed: gT = wg @ x.T,
                       hT = contiguous swiglu, yT = wd @ hT. Same operations as the f16 path above, same bits (checked);
                       only the tile of down's product changes (mlx 0.32.3, matmul.cpp: down goes from the tile
                       "nt large K" 64 x 32 x 32 to the nn tile 32 x 64 x 16, the sum over K is done in the same order); the
                       f16 -> bf16 conversion of the gate and up outputs is fused into the swiglu.
  PLI_PREFILL_UNPACK_KERNEL = 0 (default, off) | 1: the unpacking itself (mx.dequantize) is replaced by a custom one-pass Metal
                       kernel, SAME VALUES (checked with mx.array_equal): in f16 as in bf16 MLX computes
                       in bf16 (scale and bias rounded to bf16, scale * d + bias in bf16) then converts; the kernel does the same.
                       About 3 times faster (f16 gate at 2,048 rows: 1.45 against 0.50 ms). Falls back to mx.dequantize for
                       any shape not checked (4 bits, group_size 64, affine, 2-dimensional uint32 weights, bf16 scales and biases of
                       shape (N, K/64), K multiple of 64, f16 or bf16 output). Inputs are made row-contiguous (guaranteed by
                       mx.fast.metal_kernel, ensure_row_contiguous): a row view of a row_matmul stack is thus read at its offset.
  PLI_PREFILL_UNPACK_NN = 0 (default) | 1: (a) the projections unpacked in bf16 (out_proj, o_proj; down if PLI_PREFILL_UNPACK_MLP is
                       off) computed in nn: y = (w @ contiguous(x.T)).T copied back into rows (measured: out_proj 13.17 ->
                       11.77 ms at 2,048 rows, equal bits from 512 to 4,096 rows; mlx 0.32.3: same tile 64 x 64 x 16, same order over K);
                       (b) the draft: fc (nn form) and k_proj, v_proj (x @ w.T) unpacked in bf16 instead of quantized_matmul, outside
                       split-K (measured: bits equal to quantized_matmul as soon as ceil(M/16) x ceil(N/16) > 2,048;
                       at 2,047 rows fc 63.33 -> 48.06 ms, k, v of the 5 layers 25.84 -> 20.28).
  PLI_PREFILL_UNPACK_STACK = 0 (default) | 1: with PLI_PREFILL_UNPACK=f16, in_proj_qkv + in_proj_z (GDN) and q_proj + k_proj + v_proj
                       (attention) in a single nn product over the row_matmul stack (views, no extra memory), outputs split after
                       (measured: 25.40 -> 24.94 and 22.36 -> 21.94 ms at 2,048 rows, equal bits).
                       No stack when a member alone would go through MLX's split-K (_splitk: k_proj, v_proj at 512 rows; measured:
                       bits different from the stack): that member stays a separate product.
  With KERNEL on at the same time, the unpacking of these paths (nn, stack, draft) goes through the custom kernel (same values), then the product
  through the nn form (item pli-lotlec, 13/10/2026: mat-nn.diff carried over by hand onto the custom unpacking).
"""

from __future__ import annotations

import os
import sys
from typing import Any

MODE = os.environ.get("PLI_PREFILL_UNPACK", "0")
ROWS = int(os.environ.get("PLI_PREFILL_UNPACK_ROWS", "512"))
CHECK = os.environ.get("PLI_PREFILL_UNPACK_CHECK", "0") == "1"
MLP_T = os.environ.get("PLI_PREFILL_UNPACK_MLP", "0") == "t"
KERNEL = os.environ.get("PLI_PREFILL_UNPACK_KERNEL", "0") == "1"
NN = os.environ.get("PLI_PREFILL_UNPACK_NN", "0") == "1"
STACK = os.environ.get("PLI_PREFILL_UNPACK_STACK", "0") == "1"
F16 = ("gate_proj", "up_proj", "in_proj_qkv", "in_proj_z", "q_proj", "k_proj", "v_proj")
MIN_WEIGHTS = 1 << 22        # not the small projections (in_proj_b, in_proj_a: 48 outputs)
_ATTR = "_pli_unpack"
_orig: Any = None
count = {"calls": 0, "f16": 0, "bf16": 0, "out_of_range": 0, "mlp_t": 0, "kernel": 0, "kernel_fallback": 0,
         "nn": 0, "draft": 0, "stack": 0}
_classes: dict = {}
_swc: Any = None


def active() -> bool:
    return MODE in ("bf16", "f16")


_kernels: dict = {}


def _kernel(T: str, K: int, rg: bool = False) -> Any:
    """4-bit unpacking Metal kernel, one thread per (row, group of 64, quarter of group): 8 bytes of weights, 16 values.
    Same expression as MLX: scale and bias rounded to bf16, scale * d + bias in bf16, then conversion to T."""

    import hashlib

    import mlx.core as mx

    key = (T, K, rg)
    if key not in _kernels:
        G = K // 64
        if rg:      # PLI (P2 b+c): rg layout (pli/relayout.py), same bytes at other addresses, same values
            load = ("  const size_t p = row / 32, t = (row / 8) % 4, fm = row % 8;\n"
                       "  const uint2 wv = W2[((p * G + g) * 32 + fm * 4 + fnh) * 4 + t];\n"
                       "  const uint sbw = SC[((p * G + g) * 8 + fm) * 4 + t];\n"
                       "  const float scf = as_type<float>(sbw << 16), bif = as_type<float>(sbw & 0xFFFF0000u);\n")
        else:
            load = ("  const uint2 wv = W2[row * (K / 16) + 4 * g + fnh];\n"
                       "  const size_t sidx = row * G + g;\n  const float scf = float(SC[sidx]), bif = float(BI[sidx]);\n")
        src = (f"  constexpr int K = {K}, G = {G};\n  typedef {T} T;\n  const size_t i = thread_position_in_grid.x;\n"
               "  const size_t fnh = i & 3, g = (i >> 2) % G, row = i / (4 * G);\n"
               "  const device uint2* W2 = (const device uint2*)W;\n" + load +
               "  device T* o = OUT + row * K + g * 64 + fnh * 16;\n"
               "  const bfloat sb_ = bfloat(scf), bb_ = bfloat(bif);\n"
               "  for (int k = 0; k < 16; k++) {\n"
               "    const uchar d = uchar(((k < 8 ? wv.x : wv.y) >> (4 * (k & 7))) & 0xFu);\n"
               "    const bfloat v_ = sb_ * d + bb_;\n    o[k] = T(float(v_));\n  }\n")
        _kernels[key] = mx.fast.metal_kernel(name="pli_dqm_" + hashlib.sha256(src.encode()).hexdigest()[:12],
                                            input_names=["W", "SC", "BI"], output_names=["OUT"], source=src, header="")
    return _kernels[key]


def _unpack(m: Any, dt: Any) -> Any:
    """Weights of projection m unpacked to dt: custom kernel if PLI_PREFILL_UNPACK_KERNEL=1 and shape checked, else mx.dequantize."""

    return _unpack_wsb(m["weight"], m["scales"], m["biases"], m.bits, m.group_size, getattr(m, "mode", "affine"), dt)


def _unpack_wsb(w: Any, s: Any, b: Any, bits: Any, group_size: Any, mode: str, dt: Any) -> Any:
    """The same unpacking on arrays (a row_matmul stack is not a module): same conditions, same fallback."""

    import mlx.core as mx

    if s.dtype == mx.uint32:                       # PLI (P2 b+c): relaid-out matrix, only the custom kernel can read it
        N, K = int(w.shape[0]), int(w.shape[1]) * 8
        if not (KERNEL and dt in (mx.float16, mx.bfloat16) and int(bits) == 4 and int(group_size) == 64 and N % 32 == 0
                and K % 64 == 0 and N * (K // 64) * 4 < (1 << 31)):
            raise RuntimeError("PLI unpack: relaid-out matrix outside the custom kernel's shapes")
        count["kernel"] += 1
        return _kernel("half" if dt == mx.float16 else "bfloat", K, True)(
            inputs=[w, s, b], grid=(N * (K // 64) * 4, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(N, K)], output_dtypes=[dt])[0]
    if KERNEL:
        ok = (dt in (mx.float16, mx.bfloat16) and int(bits) == 4 and int(group_size) == 64
              and mode == "affine" and w.dtype == mx.uint32 and w.ndim == 2
              and s.dtype == mx.bfloat16 and b.dtype == mx.bfloat16)
        if ok:
            N, K = int(w.shape[0]), int(w.shape[1]) * 8
            ok = K % 64 == 0 and tuple(s.shape) == (N, K // 64) and tuple(b.shape) == (N, K // 64) and N * (K // 64) * 4 < (1 << 31)
        if ok:
            count["kernel"] += 1
            return _kernel("half" if dt == mx.float16 else "bfloat", K)(
                inputs=[w, s, b], grid=(N * (K // 64) * 4, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(N, K)], output_dtypes=[dt])[0]
        count["kernel_fallback"] += 1
    return mx.dequantize(w, s, b, group_size=group_size, bits=bits, dtype=dt)


def _splitk(rows: int, n: int, k: int) -> bool:
    """The product (rows x k) @ (k x n) would go through MLX's split-K (mlx 0.32.3, matmul.cpp, device "d"): there, quantized_matmul and
    "unpack then product" do not give the same bits (measured: threshold observed exactly at this rule, deduced)."""

    return ((rows + 15) // 16) * ((n + 15) // 16) <= 2048 and k // 16 >= 8 and k >= max(rows, n)


def _nn(x2: Any, w: Any) -> Any:
    """(w @ contiguous(x2.T)).T copied back into rows: same bits as x2 @ w.T (same tile, same order over K), another traversal."""

    import mlx.core as mx

    return mx.contiguous((w @ mx.contiguous(x2.T)).T)


def _rg_backend(self: Any, x: Any) -> Any:
    """PLI (P2 b+c): a relaid-out projection outside unpacking (fewer than ROWS rows, or not marked) goes through the decode
    exact-rows kernel, which can read the layout. MLX's product cannot. The bits differ from those of MLX's
    product (switch accepted on 15/10/2026, to measure without and with)."""

    from tensorfold.kernels.qwen.dense.v1 import row_matmul

    count["rg_decode"] = count.get("rg_decode", 0) + 1
    y = row_matmul.BACKEND(x, self["weight"], self["scales"], self["biases"], self.group_size, self.bits)
    return y + self["bias"] if "bias" in self else y


def _call(self: Any, x: Any) -> Any:
    import mlx.core as mx

    kind = getattr(self, _ATTR, None)
    if getattr(self, "_pli_relayout", False) and (kind is None or x.ndim < 2 or x.dtype != mx.bfloat16
                                                  or x.size // int(x.shape[-1]) < ROWS):
        return _rg_backend(self, x)
    if kind is None or x.ndim < 2:
        return _orig(self, x)
    rows = x.size // int(x.shape[-1])
    if rows < ROWS or x.dtype != mx.bfloat16:
        return _orig(self, x)
    reserve = getattr(self, "_pli_reserve", None)
    if reserve is not None:
        object.__setattr__(self, "_pli_reserve", None)
        if reserve[0] is x:                       # part already computed by the stack's first member, same input
            return reserve[1]
    if kind in ("bf16n", "bf16b"):                # draft (PLI_PREFILL_UNPACK_NN): outside split-K only
        n, k = int(self["weight"].shape[0]), int(x.shape[-1])
        if _splitk(rows, n, k):
            return _orig(self, x)
        count["calls"] += 1
        count["draft"] += 1
        w = _unpack(self, mx.bfloat16)
        x2 = x.reshape(rows, k)
        y = (_nn(x2, w) if kind == "bf16n" else x2 @ w.T).reshape(*x.shape[:-1], n)
        return y + self["bias"] if "bias" in self else y
    count["calls"] += 1
    stack = getattr(self, "_pli_stack", None) if STACK else None
    if stack is not None and kind == "f16" and not CHECK and not any(
            _splitk(rows, int(m["weight"].shape[0]), int(x.shape[-1])) for m in stack[1]):
        # (no stack when a member alone would go through MLX's split-K: k, v at 512 rows: measured, different bits)
        (pw, ps, pb), members = stack
        k = int(x.shape[-1])
        w = _unpack_wsb(pw, ps, pb, self.bits, self.group_size, getattr(self, "mode", "affine"), mx.float16)
        y = _nn(x.reshape(rows, k).astype(mx.float16), w).astype(mx.bfloat16)
        y = y.reshape(*x.shape[:-1], int(y.shape[-1]))
        o, parts = 0, []
        for m in members:
            t = int(m["weight"].shape[0])
            parts.append(y[..., o:o + t])
            o += t
        for m, part in zip(members[1:], parts[1:]):
            object.__setattr__(m, "_pli_reserve", (x, part))
        count["stack"] += 1
        count["f16"] += 1
        return parts[0]
    if kind == "f16":
        count["f16"] += 1
        w = _unpack(self, mx.float16)
        y16 = x.astype(mx.float16) @ w.T
        if CHECK:
            n = int(mx.sum(mx.logical_not(mx.isfinite(y16))).item())
            if n:
                count["out_of_range"] += n
                print(f"PLI unpack: {n} values out of range in f16 ({self._pli_name})", file=sys.stderr, flush=True)
        y = y16.astype(mx.bfloat16)
    else:
        count["bf16"] += 1
        w = _unpack(self, mx.bfloat16)
        if NN:
            count["nn"] += 1
            k = int(x.shape[-1])
            y = _nn(x.reshape(rows, k), w).reshape(*x.shape[:-1], int(w.shape[0]))
        else:
            y = x @ w.T
    if "bias" in self:
        y = y + self["bias"]
    return y


def _mlp_t(self: Any, x: Any) -> Any:
    """PLI_PREFILL_UNPACK_MLP=t: down(swiglu(gate(x), up(x))) computed transposed; falls back to the original call outside prefill."""

    import mlx.core as mx

    global _swc
    g, u, d = self.gate_proj, self.up_proj, self.down_proj
    rows = x.size // int(x.shape[-1]) if x.ndim >= 2 else 0
    if (rows < ROWS or x.dtype != mx.bfloat16 or CHECK or "bias" in g or "bias" in u or "bias" in d
            or getattr(g, _ATTR, None) != "f16" or getattr(u, _ATTR, None) != "f16" or getattr(d, _ATTR, None) != "bf16"):
        return type(self)._pli_base.__call__(self, x)
    if _swc is None:
        from mlx_lm.models.activations import swiglu

        _swc = mx.compile(lambda a, b: swiglu(a.astype(mx.bfloat16), b.astype(mx.bfloat16)), shapeless=True)
    count["mlp_t"] += 1
    count["calls"] += 3
    count["f16"] += 2
    count["bf16"] += 1
    q = _unpack
    xt = x.reshape(rows, int(x.shape[-1])).astype(mx.float16).T
    ht = _swc(q(g, mx.float16) @ xt, q(u, mx.float16) @ xt)
    y = mx.contiguous((q(d, mx.bfloat16) @ ht).T)
    return y.reshape(*x.shape[:-1], int(y.shape[-1]))


def _install_mlp(layers: Any) -> int:
    """Put _mlp_t on the dense MLPs whose gate, up (f16) and down (bf16) are marked; not on an MLP already taken (PLI_MLP_INT)."""

    try:
        from mlx_lm.models.qwen3_next import Qwen3NextMLP
    except Exception:
        return 0
    n = 0
    for layer in layers:
        m = getattr(layer, "mlp", None)
        if not isinstance(m, Qwen3NextMLP) or getattr(type(m), "_pli_int", False) or getattr(type(m), "_pli_mlp_t", False):
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


def _install_stacks(layers: Any) -> int:
    """PLI_PREFILL_UNPACK_STACK: on the first member (in_proj_qkv, q_proj), the views of the row_matmul stack covering the f16 members."""

    from tensorfold.kernels.qwen.dense.v1 import row_matmul

    n = 0
    for layer in layers:
        for parent, kind, names in ((getattr(layer, "linear_attn", None), "in", ("in_proj_qkv", "in_proj_z")),
                                    (getattr(layer, "self_attn", None), "qkv", ("q_proj", "k_proj", "v_proj"))):
            if parent is None:
                continue
            stack = row_matmul.stack_of(parent, kind)
            members = [getattr(parent, name, None) for name in names]
            if stack is None or any(m is None or getattr(m, _ATTR, None) != "f16" for m in members):
                continue
            if tuple(stack.members[:len(members)]) != tuple(members):
                continue
            t = sum(int(m["weight"].shape[0]) for m in members)
            views = (stack.weight[:t], stack.scales[:t], stack.biases[:t])
            object.__setattr__(members[0], "_pli_stack", (views, tuple(members)))
            n += 1
    return n


def install_draft(draft: Any) -> int:
    """PLI_PREFILL_UNPACK_NN: marks the draft's fc ("bf16n") and k_proj, v_proj ("bf16b"); returns the number marked."""

    global _orig
    model = getattr(draft, "model", None)
    if not (active() and NN) or model is None:
        return 0
    import mlx.core as mx
    import mlx.nn as nn

    n = 0
    for name, m in model.named_modules():
        if not isinstance(m, nn.QuantizedLinear) or getattr(m, _ATTR, None) is not None:
            continue
        if int(m.bits) != 4 or getattr(m, "mode", "affine") != "affine" or m["scales"].dtype != mx.bfloat16:
            continue
        short = name.rsplit(".", 1)[-1]
        kind = "bf16n" if name == "fc" else ("bf16b" if short in ("k_proj", "v_proj") else None)
        if kind is None:
            continue
        object.__setattr__(m, _ATTR, kind)
        object.__setattr__(m, "_pli_name", "draft." + name)
        n += 1
    if _orig is None:
        _orig = nn.QuantizedLinear.__call__
        nn.QuantizedLinear.__call__ = _call
    print(f"PLI unpack: draft, {n} projections marked (fc in nn, k and v), outside split-K", file=sys.stderr, flush=True)
    return n


def install(layers: Any) -> int:
    """Mark the 4-bit projections of the model's layers and put the call; idempotent; returns the number marked."""

    global _orig
    if not active() or layers is None:
        return 0
    import mlx.core as mx
    import mlx.nn as nn

    n = 0
    for i, layer in enumerate(layers):
        for name, m in layer.named_modules():
            if not isinstance(m, nn.QuantizedLinear) or getattr(m, _ATTR, None) is not None:
                continue
            rg = getattr(m, "_pli_relayout", False) and m["scales"].dtype == mx.uint32   # PLI (P2 b+c): relaid-out matrix
            if (int(m.bits) != 4 or getattr(m, "mode", "affine") != "affine" or (m["scales"].dtype != mx.bfloat16 and not rg)
                    or m["weight"].size * 8 < MIN_WEIGHTS):
                continue
            short = name.rsplit(".", 1)[-1]
            object.__setattr__(m, _ATTR, "f16" if (MODE == "f16" and short in F16) else "bf16")
            object.__setattr__(m, "_pli_name", f"{i}.{name}")
            n += 1
    t = _install_mlp(layers) if (MLP_T and MODE == "f16") else 0
    p = _install_stacks(layers) if (STACK and MODE == "f16") else 0
    if _orig is None:
        _orig = nn.QuantizedLinear.__call__
        nn.QuantizedLinear.__call__ = _call
        print(f"PLI unpack: mode {MODE}, {n} projections marked, from {ROWS} rows"
              + (f", {t} MLPs computed transposed" if MLP_T else "") + (", bf16 in nn" if NN else "")
              + (f", {p} stacks" if STACK else ""), file=sys.stderr, flush=True)
    return n
