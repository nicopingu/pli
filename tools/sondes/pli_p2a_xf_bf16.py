"""PLI P2 palier a : fragments en paires de bf16 = mêmes bits que les paires de fp32 et que qmm (MMA), MLX requis (Apple Silicon).

Usage : PYTHONPATH=src python tests/test_pli_p2a_xf_bf16.py [rapide]
"""
import sys
import time

import mlx.core as mx

from tensorfold.kernels.qwen.dense.v1 import simd_qmm as Q

FORMES = [(17408, 5120), (5120, 17408), (6144, 5120), (1024, 5120), (64, 5120)]
if "rapide" in sys.argv:
    FORMES = FORMES[:1] + FORMES[-1:]


def poids(n, k, graine):
    w = mx.random.normal((n, k), key=mx.random.key(graine)).astype(mx.bfloat16) * 0.02
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    return wq, s.astype(mx.bfloat16), b.astype(mx.bfloat16)


def calcule(drapeau, x, w, s, b):
    Q.XF_BF16 = drapeau
    out = Q.qmm_fragments(Q.fragments(x), w, s, b)
    mx.eval(out)
    return out


ecarts = cas = 0
for n, k in FORMES:
    w, s, b = poids(n, k, n + k)
    for rows in (2, 3, 5, 8):
        for echelle in (1.0, 2.0 ** -20, 2.0 ** 20, 2.0 ** -60):
            x = (mx.random.normal((rows, k), key=mx.random.key(rows * 7 + int(echelle > 1))) * echelle).astype(mx.bfloat16)
            ref = Q.qmm(x, w, s, b, 64, kind="mma")
            a = calcule(False, x, w, s, b)
            c = calcule(True, x, w, s, b)
            mx.eval(ref)
            e1 = int(mx.sum(a.view(mx.uint16) != ref.view(mx.uint16)).item())
            e2 = int(mx.sum(c.view(mx.uint16) != ref.view(mx.uint16)).item())
            cas += 1
            ecarts += e2
            if e1 or e2:
                print(f"ÉCART n={n} k={k} lignes={rows} échelle={echelle}: fp32 {e1}, bf16 {e2}")
print(f"{cas} cas, valeurs différentes avec les fragments bf16 : {ecarts}")

# vitesse indicative (hors M1 Ultra : sans valeur de décision)
for n, k in FORMES[:2]:
    w, s, b = poids(n, k, 1)
    x = mx.random.normal((8, k), key=mx.random.key(3)).astype(mx.bfloat16)
    for drapeau in (False, True):
        Q.XF_BF16 = drapeau
        f = Q.fragments(x)
        mx.eval(f, calcule(drapeau, x, w, s, b))
        t = time.perf_counter()
        outs = [Q.qmm_fragments(f, w, s, b) for _ in range(200)]
        mx.eval(outs)
        print(f"n={n} k={k} 8 lignes, fragments {'bf16' if drapeau else 'fp32'} : {(time.perf_counter() - t) / 200 * 1e3:.3f} ms (indicatif)")
sys.exit(1 if ecarts else 0)
