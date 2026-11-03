"""PLI: weights and scales of the 27B relaid out for decode (PLI_RELAYOUT=1, off by default).

Measured on M1 Ultra (12/10/2026): the 8-row decode kernel reads the weights of 4 output rows at a time
(4 reads of 64 bits) and the scale and bias in two separate arrays. Laid out in packs of 32 outputs, with
the scale and the bias joined in one 32-bit word, a thread reads all it needs in 3 reads of 128 bits:
- 16.6 % (gate + up), - 18 % (down), - 11 to - 12 % (in_proj, qkv) per 8-row product, same bits. Computed: - 9 ms per round.

"rg" layout of a matrix of N outputs (N multiple of 32), K inputs, groups of 64 (G = K / 64), n = 32 p + 8 t + fm:
  weights : uint2 [p][g][fm * 4 + q][t] = uint2 no. (4 g + q) of row n (q = quarter of the group, 0 to 3)
  scales  : uint32 [p][g][fm][t] = bf16 of the scale (low 16 bits) | bf16 of the bias (high 16 bits)
The arrays keep their shape ([N, K / 8] for the weights, [N, G] for the words) and a pack takes 32 contiguous rows:
a row view aligned on 32 (member of a row_matmul stack) is still a relaid-out matrix. The array of words takes
the place of the scales; the module keeps this same array as its biases (nothing reads them). The kernels recognize the
layout by the type of the scales (uint32 instead of bf16: simd_qmm.relaid_out).

What reads these weights, and how:
- decode (row_matmul, simd_qmm): scalar, MMA and fragment kernels, "_rg" variants, same computations;
- prompt prefill, 512 rows and more (pli/unpack.py): custom unpacking kernel, rg variant, same values;
- prompt prefill, fewer than 512 rows, and unmarked projections: go through the decode kernel instead of MLX's
  product. THE BITS CHANGE THERE (choice of 15/10/2026: text switch, measured without and with before any rollout);
- not relaid out: the head (lm_head, also read by the DFlash2 draft), the embeddings, the draft, the members of a stack
  not aligned on 32 (in_proj_b, in_proj_a: 48 outputs), which get their own copy in the original layout.

Conditions (otherwise nothing is relaid out, and a message says why): PLI_PREFILL_UNPACK at f16 or bf16 with
PLI_PREFILL_UNPACK_KERNEL=1; neither PLI_PREFILL_CPU, nor PLI_MLP_INT (they read the original weights).

Each matrix is checked before being replaced: 1, 2 and 8 rows, original kernel against relaid-out kernel, same bits required.
A single difference and that matrix stays in the original layout (the rest of the model does not depend on it).
"""

from __future__ import annotations

import hashlib
import os
import sys
import time
from typing import Any

ACTIVE = os.environ.get("PLI_RELAYOUT", "0") == "1"
_conv: dict = {}
state: dict[str, Any] = {"relaid": 0, "refused": 0, "copies": 0, "bytes": 0, "reason": "", "seconds": 0.0}


def _conversion_kernel(K: int) -> Any:
    import mlx.core as mx

    if K not in _conv:
        G = K // 64
        src = (f"  constexpr int K = {K}, G = {G};\n"
               "  const uint i = thread_position_in_grid.x;          // one thread per (row n, group g)\n"
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


def convert(w: Any, s: Any, b: Any) -> tuple[Any, Any]:
    """Original (weights, scales, biases) -> (relaid-out weights, scale + bias words), same shapes, same bytes."""

    import mlx.core as mx

    N, K = int(w.shape[0]), int(w.shape[1]) * 8
    G = K // 64
    wr, sb = _conversion_kernel(K)(inputs=[w, s, b], grid=(N * G, 1, 1), threadgroup=(256, 1, 1),
                                   output_shapes=[(N, K // 8), (N, G)], output_dtypes=[mx.uint32, mx.uint32])
    return wr, sb


def suitable(w: Any, s: Any, b: Any) -> bool:
    import mlx.core as mx

    return (w.dtype == mx.uint32 and w.ndim == 2 and s.dtype == mx.bfloat16 and b.dtype == mx.bfloat16
            and int(w.shape[0]) % 32 == 0 and (int(w.shape[1]) * 8) % 64 == 0
            and tuple(s.shape) == (int(w.shape[0]), int(w.shape[1]) * 8 // 64) and tuple(b.shape) == tuple(s.shape))


def check(w: Any, s: Any, b: Any, wr: Any, sb: Any) -> bool:
    """Same bits, original kernel against relaid-out kernel: 1 row (the kernel the call would pick), 2 rows, 8 rows."""

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


def _refusal() -> str:
    from tensorfold.pli import unpack

    if not unpack.active():
        return "PLI_PREFILL_UNPACK off: prompt prefill would go through MLX's product"
    if not unpack.KERNEL:
        return "PLI_PREFILL_UNPACK_KERNEL off: mx.dequantize does not read the layout"
    try:
        if float(os.environ.get("PLI_PREFILL_CPU", "0") or 0) > 0:
            return "PLI_PREFILL_CPU on: the split prefill reads the original weights"
    except ValueError:
        return "PLI_PREFILL_CPU unreadable"
    if os.environ.get("PLI_MLP_INT", "0") == "1":
        return "PLI_MLP_INT on: the integer MLP reads the original weights"
    return ""


def _copy(a: Any) -> Any:
    """A copy that does not keep the source array alive (a row view would keep it)."""

    import mlx.core as mx

    c = mx.array(a) + mx.zeros((), dtype=a.dtype)
    mx.eval(c)
    return c


def install(model: Any) -> dict[str, Any]:
    """Relay out the layers' projections (stacks and lone projections); return ``state``. To call after row_matmul.install."""

    if not ACTIVE:
        return state
    import mlx.core as mx
    import mlx.nn as nn

    from tensorfold.kernels.qwen.dense.v1 import row_matmul

    reason = _refusal()
    if reason:
        state["reason"] = reason
        print(f"PLI relayout: nothing relaid out ({reason})", file=sys.stderr, flush=True)
        return state
    start = time.perf_counter()
    lm = getattr(model, "language_model", model)
    layers = lm.model.layers
    for layer in layers:
        in_stacks: set[int] = set()
        for _, module in layer.named_modules():
            for stack in module.__dict__.get(row_matmul._ATTR, {}).values():
                if not stack.valid():
                    continue
                in_stacks.update(id(m) for m in stack.members)
                if not suitable(stack.weight, stack.scales, stack.biases):
                    state["refused"] += 1
                    continue
                wr, sb = convert(stack.weight, stack.scales, stack.biases)
                mx.eval(wr, sb)
                if not check(stack.weight, stack.scales, stack.biases, wr, sb):
                    state["refused"] += 1
                    continue
                # members: relaid-out view if aligned on 32, else a copy in the original layout (small members only)
                o, views = 0, []
                for m, n in zip(stack.members, stack.sizes):
                    if o % 32 == 0 and n % 32 == 0:
                        views.append((m, wr[o:o + n], sb[o:o + n], sb[o:o + n], True))
                    else:
                        views.append((m, _copy(stack.weight[o:o + n]), _copy(stack.scales[o:o + n]),
                                      _copy(stack.biases[o:o + n]), False))
                        state["copies"] += 1
                    o += n
                stack.weight, stack.scales, stack.biases = wr, sb, sb
                for m, w2, s2, b2, rg in views:
                    m.weight, m.scales, m.biases = w2, s2, b2
                    object.__setattr__(m, "_pli_relayout", rg)
                mx.eval([a for m in stack.members for a in (m["weight"], m["scales"], m["biases"])])
                stack.held = tuple(m["weight"] for m in stack.members)
                state["relaid"] += 1
                state["bytes"] += int(wr.nbytes + sb.nbytes)
        for _, m in layer.named_modules():
            if not isinstance(m, nn.QuantizedLinear) or id(m) in in_stacks or getattr(m, "_pli_relayout", None) is not None:
                continue
            if int(m.bits) != 4 or int(m.group_size) != 64 or getattr(m, "mode", "affine") != "affine" or "bias" in m:
                continue
            w, s, b = m["weight"], m["scales"], m["biases"]
            if not suitable(w, s, b):
                continue
            wr, sb = convert(w, s, b)
            mx.eval(wr, sb)
            if not check(w, s, b, wr, sb):
                state["refused"] += 1
                continue
            m.weight, m.scales, m.biases = wr, sb, sb
            object.__setattr__(m, "_pli_relayout", True)
            state["relaid"] += 1
            state["bytes"] += int(wr.nbytes + sb.nbytes)
        mx.clear_cache()
    state["seconds"] = round(time.perf_counter() - start, 1)
    # relaid-out projections go through the unpacking hook right now (it routes to the relaid-out kernels)
    from tensorfold.pli import unpack

    unpack.install(layers)
    print(f"PLI relayout: {state['relaid']} matrices relaid out ({state['bytes'] / 2**30:.2f} GiB), {state['refused']} refused, "
          f"{state['copies']} small members copied, {state['seconds']} s", file=sys.stderr, flush=True)
    return state


__all__ = ["ACTIVE", "suitable", "state", "install", "convert", "check"]


# PLI (30/10/2026, draft profile: 5 DFlash2 layers at about 2.2 ms): PLI_RELAYOUT_DRAFT=1 (off) also relays out the draft's projections
# that read only the 8 rows of the block (q_proj, o_proj, gate, up, down, kernel_projection of the two convolutions); k_proj,
# v_proj and fc read the context rows (up to 2,047 at the first lattice) and keep the original layout (MLX's product reads them).
# Each matrix checked (same bits, check); draft_matmul sends any relaid-out matrix to the decode kernel. Same drafts, same text.
DRAFT = os.environ.get("PLI_RELAYOUT_DRAFT", "0") == "1"
_BLOCK = ("q_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "kernel_projection")


def install_draft(model: Any) -> dict[str, int]:
    """Relay out the draft's block projections (after row_matmul.route_drafter); return {relaid, refused}."""

    count = {"relaid": 0, "refused": 0}
    if not (ACTIVE and DRAFT):
        return count
    import mlx.core as mx
    import mlx.nn as nn

    reason = _refusal()
    if reason:
        print(f"PLI relayout of the draft: nothing relaid out ({reason})", file=sys.stderr, flush=True)
        return count
    for name, m in model.named_modules():
        if not isinstance(m, nn.QuantizedLinear) or name.rsplit(".", 1)[-1] not in _BLOCK:
            continue
        if not getattr(m, "_row_forward_draft", False) or getattr(m, "_pli_relayout", None) is not None:
            continue
        if int(m.bits) != 4 or int(m.group_size) != 64 or getattr(m, "mode", "affine") != "affine" or "bias" in m:
            continue
        w, s, b = m["weight"], m["scales"], m["biases"]
        if not suitable(w, s, b):
            continue
        wr, sb = convert(w, s, b)
        mx.eval(wr, sb)
        if not check(w, s, b, wr, sb):
            count["refused"] += 1
            continue
        m.weight, m.scales, m.biases = wr, sb, sb
        object.__setattr__(m, "_pli_relayout", True)
        count["relaid"] += 1
    mx.clear_cache()
    print(f"PLI relayout of the draft: {count['relaid']} matrices relaid out, {count['refused']} refused", file=sys.stderr, flush=True)
    return count
