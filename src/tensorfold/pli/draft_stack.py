"""PLI (01/11/2026, PLI_DRAFT_STACK=1, off): the DFlash2 draft's projections in fewer calls, same bits.

Measured (dbench2, M1 Ultra): per layer, q_proj, k_proj and v_proj on the 8 rows of the block and k_proj, v_proj on the 3 new rows of
the context make five calls of the 8-row product (V8); fc (25,600 -> 5,120) on the context rows goes through the original layout.
Here: [q | k | v] and [k | v] stacked and relaid out (pli/relayout.convert), one call each; fc relaid out. The original modules stay for
the calls outside 2 to 16 rows (one row, and the first lattice after a prefill: up to 2,047 context rows), where
draft_matmul takes MLX's product.

Same bits: the V8 product computes each column on its own, with the same plan (nf, cs, ks) for K = 5,120 whatever N is here, and the
relaid-out and original kernels do the same operations (checked at install: each stack against the separate calls, 1 to 16 rows).
"""

from __future__ import annotations

import os
import sys
from typing import Any

ENABLED = os.environ.get("PLI_DRAFT_STACK", "0") == "1"
_QKV, _KV, _SMALL = "_pli_qkv", "_pli_kv", "_pli_small"


def _backend_qmm(x: Any, wr: Any, sb: Any) -> Any:
    from tensorfold.kernels.qwen.dense.v1 import row_matmul
    return row_matmul.BACKEND.qmm(x, wr, sb, sb, 64)


def _separate(x: Any, mods: list[Any]) -> Any:
    import mlx.core as mx
    from tensorfold.kernels.qwen.dense.v1.row_matmul import draft_matmul
    return mx.concatenate([draft_matmul(x, m["weight"], m["scales"], m["biases"], m.group_size, m.bits) for m in mods], axis=-1)


def _stack(mods: list[Any]) -> tuple[Any, Any] | None:
    """The modules (original, not relaid out, 4 bits, groups of 64) stacked by rows then relaid out; None if a shape does not fit."""

    import mlx.core as mx
    from tensorfold.pli import relayout

    for m in mods:
        if (int(m.bits) != 4 or int(m.group_size) != 64 or m["scales"].dtype == mx.uint32 or "bias" in m
                or getattr(m, "mode", "affine") != "affine"):
            return None
    w = mx.concatenate([m["weight"] for m in mods], axis=0)
    s = mx.concatenate([m["scales"] for m in mods], axis=0)
    b = mx.concatenate([m["biases"] for m in mods], axis=0)
    if not relayout.suitable(w, s, b):
        return None
    wr, sb = relayout.convert(w, s, b)
    mx.eval(wr, sb)
    return wr, sb


def _same(mods: list[Any], wr: Any, sb: Any) -> bool:
    import mlx.core as mx

    K = int(mods[0]["weight"].shape[1]) * 8
    for rows in (2, 3, 7, 8, 9, 16):
        x = (mx.random.normal((1, rows, K), key=mx.random.key(rows * 977 + K)) * 0.5).astype(mx.bfloat16)
        if not bool(mx.array_equal(_separate(x, mods), _backend_qmm(x, wr, sb)).item()):
            return False
    return True


def install(model: Any) -> dict[str, int]:
    """Before relayout.install_draft (the modules are still in their original layout); return {stacks, refused}."""

    count = {"stacks": 0, "refused": 0}
    if not ENABLED:
        return count
    import mlx.core as mx
    from tensorfold.kernels.qwen.dense.v1 import qmm8, row_matmul

    if row_matmul.BACKEND is None or row_matmul.BACKEND.name != "simd_qmm" or not qmm8.ENABLED:
        print("PLI draft stacks: nothing (needs the 8-row product, PLI_V8=1)", file=sys.stderr, flush=True)
        return count
    for layer in model.layers:
        attn = layer.self_attn
        for attr, mods in ((_QKV, [attn.q_proj, attn.k_proj, attn.v_proj]), (_KV, [attn.k_proj, attn.v_proj])):
            hit = _stack(mods)
            if hit is None or not _same(mods, *hit):
                count["refused"] += 1
                continue
            object.__setattr__(attn, attr, (hit, [int(m["weight"].shape[0]) for m in mods]))
            count["stacks"] += 1
    fc = getattr(model, "fc", None)
    if fc is not None:
        hit = _stack([fc])
        if hit is not None and _same([fc], *hit):
            object.__setattr__(fc, _SMALL, hit)
            count["stacks"] += 1
        else:
            count["refused"] += 1
    mx.clear_cache()
    print(f"PLI draft stacks: {count['stacks']} stacks relaid out, {count['refused']} refused", file=sys.stderr, flush=True)
    return count


def _rows(x: Any) -> int:
    return x.size // int(x.shape[-1])


def _in_range(x: Any) -> bool:
    """The calls that draft_matmul already sends to the 8-row product (2 to WINDOW_ROWS rows): same bits; elsewhere, the original."""

    from tensorfold.kernels.qwen.dense.v1.row_matmul import WINDOW_ROWS
    return 2 <= _rows(x) <= WINDOW_ROWS


def project(attn: Any, attr: str, x: Any) -> list[Any] | None:
    """The outputs of ``attn``'s stack ``attr`` on ``x`` (one per module), or None (no stack, out of range)."""

    hit = getattr(attn, attr, None)
    if hit is None or not _in_range(x):
        return None
    (wr, sb), widths = hit
    y = _backend_qmm(x, wr, sb)
    out, a = [], 0
    for n in widths:
        out.append(y[..., a:a + n])
        a += n
    return out


def raw(attn: Any, attr: str, x: Any) -> Any | None:
    """The whole output of the stack ``attr`` (columns of all the modules), or None."""

    hit = getattr(attn, attr, None)
    if hit is None or not _in_range(x):
        return None
    return _backend_qmm(x, *hit[0])


def small(module: Any, x: Any) -> Any | None:
    """The relaid-out product of a module with two layouts (fc) within the range of the 8-row product, or None."""

    hit = getattr(module, _SMALL, None)
    if hit is None or not _in_range(x):
        return None
    return _backend_qmm(x, *hit)


# PLI (01/11/2026, PLI_DRAFT_STACK): the draft head (rows of the draft vocabulary: 98,304, 288, extended list) relaid out, in two
# calls instead of three: the first part alone (same product plan, ks = 2), the two small ones stacked (same plan, ks = 4);
# same bits as the three original calls (checked at install). Relaid-out copy of about 260 MB.
_HEAD = "_pli_head"


def install_head(drafter: Any) -> bool:
    if not ENABLED:
        return False
    import mlx.core as mx
    from tensorfold.kernels.qwen.dense.v1 import qmm8, row_matmul
    from tensorfold.kernels.qwen.dense.v1.row_matmul import draft_matmul
    from tensorfold.pli import relayout

    plain = drafter._plain_sub_head() if hasattr(drafter, "_plain_sub_head") else None
    if plain is None or row_matmul.BACKEND is None or not qmm8.ENABLED:
        return False
    parts, _, gs, bits = plain
    if int(gs) != 64 or int(bits) != 4 or len(parts) < 2:
        return False
    groups = [[parts[0]], list(parts[1:])]
    stacks = []
    for g in groups:
        w = mx.concatenate([p[0] for p in g], axis=0)
        s = mx.concatenate([p[1] for p in g], axis=0)
        b = mx.concatenate([p[2] for p in g], axis=0)
        if s.dtype == mx.uint32 or not relayout.suitable(w, s, b):
            return False
        wr, sb = relayout.convert(w, s, b)
        mx.eval(wr, sb)
        stacks.append((wr, sb))
    K = int(parts[0][0].shape[1]) * 8
    for rows in (2, 7, 8, 16):
        x = (mx.random.normal((1, rows, K), key=mx.random.key(rows * 31 + 7)) * 0.5).astype(mx.bfloat16)
        ref = mx.concatenate([draft_matmul(x, w, s, b, 64, 4) for w, s, b in parts], axis=-1)
        if not bool(mx.array_equal(ref, head(x, stacks)).item()):
            print("PLI draft stacks: head not relaid out (other bits)", file=sys.stderr, flush=True)
            return False
    object.__setattr__(drafter, _HEAD, stacks)
    mx.clear_cache()
    print(f"PLI draft stacks: head relaid out ({sum(int(p[0].shape[0]) for p in parts)} rows, 2 calls)", file=sys.stderr, flush=True)
    return True


def head(x: Any, stacks: list[tuple[Any, Any]]) -> Any:
    import mlx.core as mx
    return mx.concatenate([_backend_qmm(x, wr, sb) for wr, sb in stacks], axis=-1)


def head_of(drafter: Any, x: Any) -> Any | None:
    stacks = getattr(drafter, _HEAD, None)
    if stacks is None or not _in_range(x):
        return None
    return head(x, stacks)
