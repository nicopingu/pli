"""PLI P2 step a: fragments as pairs of bf16 = same bits as the pairs of fp32 and as qmm (MMA); needs MLX (Apple Silicon).

Usage: PYTHONPATH=src python tools/probes/pli_p2a_xf_bf16.py [quick]
"""
import sys
import time

import mlx.core as mx

from tensorfold.kernels.qwen.dense.v1 import simd_qmm as Q

SHAPES = [(17408, 5120), (5120, 17408), (6144, 5120), (1024, 5120), (64, 5120)]
if "quick" in sys.argv or "rapide" in sys.argv:
    SHAPES = SHAPES[:1] + SHAPES[-1:]


def weights(n, k, seed):
    w = mx.random.normal((n, k), key=mx.random.key(seed)).astype(mx.bfloat16) * 0.02
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    return wq, s.astype(mx.bfloat16), b.astype(mx.bfloat16)


def compute(flag, x, w, s, b):
    Q.XF_BF16 = flag
    out = Q.qmm_fragments(Q.fragments(x), w, s, b)
    mx.eval(out)
    return out


gaps = cases = 0
for n, k in SHAPES:
    w, s, b = weights(n, k, n + k)
    for rows in (2, 3, 5, 8):
        for scale in (1.0, 2.0 ** -20, 2.0 ** 20, 2.0 ** -60):
            x = (mx.random.normal((rows, k), key=mx.random.key(rows * 7 + int(scale > 1))) * scale).astype(mx.bfloat16)
            ref = Q.qmm(x, w, s, b, 64, kind="mma")
            a = compute(False, x, w, s, b)
            c = compute(True, x, w, s, b)
            mx.eval(ref)
            e1 = int(mx.sum(a.view(mx.uint16) != ref.view(mx.uint16)).item())
            e2 = int(mx.sum(c.view(mx.uint16) != ref.view(mx.uint16)).item())
            cases += 1
            gaps += e2
            if e1 or e2:
                print(f"MISMATCH n={n} k={k} rows={rows} scale={scale}: fp32 {e1}, bf16 {e2}")
print(f"{cases} cases, values differing with the bf16 fragments: {gaps}")

# indicative speed (outside the M1 Ultra: no decision value)
for n, k in SHAPES[:2]:
    w, s, b = weights(n, k, 1)
    x = mx.random.normal((8, k), key=mx.random.key(3)).astype(mx.bfloat16)
    for flag in (False, True):
        Q.XF_BF16 = flag
        f = Q.fragments(x)
        mx.eval(f, compute(flag, x, w, s, b))
        t = time.perf_counter()
        outs = [Q.qmm_fragments(f, w, s, b) for _ in range(200)]
        mx.eval(outs)
        print(f"n={n} k={k} 8 rows, fragments {'bf16' if flag else 'fp32'}: {(time.perf_counter() - t) / 200 * 1e3:.3f} ms (indicative)")
sys.exit(1 if gaps else 0)
