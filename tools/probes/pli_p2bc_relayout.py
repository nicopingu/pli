"""PLI P2 steps b and c: the rg layout gives the same bits as the original layout (needs MLX: Apple Silicon).

Usage: PYTHONPATH=src PLI_PREFILL_UNPACK=f16 PLI_PREFILL_UNPACK_KERNEL=1 python tools/probes/pli_p2bc_relayout.py
"""
import sys
import time

import mlx.core as mx

from tensorfold.kernels.qwen.dense.v1 import simd_qmm as Q
from tensorfold.pli import unpack, relayout as R

SHAPES = [(34816, 5120), (5120, 17408), (16480, 5120), (14336, 5120), (5120, 6144), (64, 5120)]


def weights(n, k, seed):
    w = mx.random.normal((n, k), key=mx.random.key(seed)).astype(mx.bfloat16) * 0.02
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    return wq, s.astype(mx.bfloat16), b.astype(mx.bfloat16)


def mismatches(a, b):
    mx.eval(a, b)
    return int(mx.sum(a.view(mx.uint16) != b.view(mx.uint16)).item()) if a.dtype == mx.bfloat16 else int(mx.sum(a != b).item())


gaps, cases = 0, 0
for n, k in SHAPES:
    w, s, b = weights(n, k, n + k)
    wr, sb = R.convert(w, s, b)
    mx.eval(wr, sb)
    assert wr.shape == w.shape and sb.shape == s.shape and sb.dtype == mx.uint32
    check_ok = R.check(w, s, b, wr, sb)
    cases += 1
    gaps += 0 if check_ok else 1
    for rows in (1, 2, 3, 4, 5, 8, 9, 16, 100, 300):
        x = (mx.random.normal((rows, k), key=mx.random.key(rows)) * 0.7).astype(mx.bfloat16)
        e = mismatches(Q.qmm(x, w, s, b, 64), Q.qmm(x, wr, sb, sb, 64))
        cases += 1
        gaps += e
        if e:
            print(f"MISMATCH qmm n={n} k={k} rows={rows}: {e}")
        if 2 <= rows <= 8:
            for flag in (False, True):
                Q.XF_BF16 = flag
                e = mismatches(Q.qmm_fragments(Q.fragments(x), w, s, b), Q.qmm_fragments(Q.fragments(x), wr, sb, sb))
                Q.XF_BF16 = False
                cases += 1
                gaps += e
                if e:
                    print(f"MISMATCH fragments n={n} k={k} rows={rows} bf16={flag}: {e}")
    # the engine's startup check (scalar against MMA) on the relaid-out matrix
    if not Q.check(wr, sb, sb):
        print(f"scalar/MMA check failed on the relaid-out matrix n={n} k={k} (and on the original: {Q.check(w, s, b)})")
    # prefill unpacking: custom relaid-out kernel against original custom kernel and mx.dequantize
    for dt in (mx.float16, mx.bfloat16):
        a = unpack._unpack_wsb(w, s, b, 4, 64, "affine", dt)
        c = unpack._unpack_wsb(wr, sb, sb, 4, 64, "affine", dt)
        m = mx.dequantize(w, s, b, group_size=64, bits=4, dtype=dt)
        e = mismatches(a.view(mx.uint16), c.view(mx.uint16)) + mismatches(m.view(mx.uint16), c.view(mx.uint16))
        cases += 1
        gaps += e
        if e:
            print(f"MISMATCH unpacking n={n} k={k} {dt}: {e}")
    # row view aligned on 32 = relaid-out matrix
    if n >= 64:
        o, t = 32, n - 64 if n > 96 else 32
        x = (mx.random.normal((8, k), key=mx.random.key(5))).astype(mx.bfloat16)
        e = mismatches(Q.qmm(x, w[o:o + t], s[o:o + t], b[o:o + t], 64), Q.qmm(x, wr[o:o + t], sb[o:o + t], sb[o:o + t], 64))
        cases += 1
        gaps += e
        if e:
            print(f"MISMATCH view n={n} k={k}: {e}")
print(f"{cases} cases, differing values: {gaps}")

# indicative speed at 8 rows (outside the M1 Ultra: no decision value)
for n, k in SHAPES[:2]:
    w, s, b = weights(n, k, 1)
    wr, sb = R.convert(w, s, b)
    x = mx.random.normal((8, k), key=mx.random.key(3)).astype(mx.bfloat16)
    f = Q.fragments(x)
    for name, args in (("original", (w, s, b)), ("relaid out", (wr, sb, sb))):
        mx.eval(Q.qmm_fragments(f, *args))
        t = time.perf_counter()
        mx.eval([Q.qmm_fragments(f, *args) for _ in range(200)])
        print(f"n={n} k={k} 8 rows {name}: {(time.perf_counter() - t) / 200 * 1e3:.3f} ms (indicative)")
sys.exit(1 if gaps else 0)
