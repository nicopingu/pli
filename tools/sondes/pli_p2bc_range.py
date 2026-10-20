"""PLI P2 paliers b et c : le rangement rg rend les mêmes bits que le rangement d'origine (MLX requis : Apple Silicon).

Usage : PYTHONPATH=src PLI_LECTURE_DEPLIE=f16 PLI_LECTURE_DEPLIE_NOYAU=1 python tests/test_pli_p2bc_range.py
"""
import sys
import time

import mlx.core as mx

from tensorfold.kernels.qwen.dense.v1 import simd_qmm as Q
from tensorfold.pli import deplie, range_poids as R

FORMES = [(34816, 5120), (5120, 17408), (16480, 5120), (14336, 5120), (5120, 6144), (64, 5120)]


def poids(n, k, graine):
    w = mx.random.normal((n, k), key=mx.random.key(graine)).astype(mx.bfloat16) * 0.02
    wq, s, b = mx.quantize(w, group_size=64, bits=4)
    return wq, s.astype(mx.bfloat16), b.astype(mx.bfloat16)


def egal(a, b):
    mx.eval(a, b)
    return int(mx.sum(a.view(mx.uint16) != b.view(mx.uint16)).item()) if a.dtype == mx.bfloat16 else int(mx.sum(a != b).item())


ecarts, cas = 0, 0
for n, k in FORMES:
    w, s, b = poids(n, k, n + k)
    wr, sb = R.ranger(w, s, b)
    mx.eval(wr, sb)
    assert wr.shape == w.shape and sb.shape == s.shape and sb.dtype == mx.uint32
    ok_verif = R.verifie(w, s, b, wr, sb)
    cas += 1
    ecarts += 0 if ok_verif else 1
    for rows in (1, 2, 3, 4, 5, 8, 9, 16, 100, 300):
        x = (mx.random.normal((rows, k), key=mx.random.key(rows)) * 0.7).astype(mx.bfloat16)
        e = egal(Q.qmm(x, w, s, b, 64), Q.qmm(x, wr, sb, sb, 64))
        cas += 1
        ecarts += e
        if e:
            print(f"ÉCART qmm n={n} k={k} lignes={rows} : {e}")
        if 2 <= rows <= 8:
            for drapeau in (False, True):
                Q.XF_BF16 = drapeau
                e = egal(Q.qmm_fragments(Q.fragments(x), w, s, b), Q.qmm_fragments(Q.fragments(x), wr, sb, sb))
                Q.XF_BF16 = False
                cas += 1
                ecarts += e
                if e:
                    print(f"ÉCART fragments n={n} k={k} lignes={rows} bf16={drapeau} : {e}")
    # contrôle du moteur au démarrage (scalaire contre MMA) sur la matrice rangée
    if not Q.check(wr, sb, sb):
        print(f"check scalaire/MMA raté sur la matrice rangée n={n} k={k} (et sur l'originale : {Q.check(w, s, b)})")
    # dépliage de la lecture : noyau maison rangé contre noyau maison d'origine et mx.dequantize
    for dt in (mx.float16, mx.bfloat16):
        a = deplie._deplier_wsb(w, s, b, 4, 64, "affine", dt)
        c = deplie._deplier_wsb(wr, sb, sb, 4, 64, "affine", dt)
        m = mx.dequantize(w, s, b, group_size=64, bits=4, dtype=dt)
        e = egal(a.view(mx.uint16), c.view(mx.uint16)) + egal(m.view(mx.uint16), c.view(mx.uint16))
        cas += 1
        ecarts += e
        if e:
            print(f"ÉCART dépliage n={n} k={k} {dt} : {e}")
    # vue de lignes alignée sur 32 = matrice rangée
    if n >= 64:
        o, t = 32, n - 64 if n > 96 else 32
        x = (mx.random.normal((8, k), key=mx.random.key(5))).astype(mx.bfloat16)
        e = egal(Q.qmm(x, w[o:o + t], s[o:o + t], b[o:o + t], 64), Q.qmm(x, wr[o:o + t], sb[o:o + t], sb[o:o + t], 64))
        cas += 1
        ecarts += e
        if e:
            print(f"ÉCART vue n={n} k={k} : {e}")
print(f"{cas} cas, valeurs différentes : {ecarts}")

# vitesse indicative à 8 lignes (hors M1 Ultra : sans valeur de décision)
for n, k in FORMES[:2]:
    w, s, b = poids(n, k, 1)
    wr, sb = R.ranger(w, s, b)
    x = mx.random.normal((8, k), key=mx.random.key(3)).astype(mx.bfloat16)
    f = Q.fragments(x)
    for nom, args in (("origine", (w, s, b)), ("rangée", (wr, sb, sb))):
        mx.eval(Q.qmm_fragments(f, *args))
        t = time.perf_counter()
        mx.eval([Q.qmm_fragments(f, *args) for _ in range(200)])
        print(f"n={n} k={k} 8 lignes {nom} : {(time.perf_counter() - t) / 200 * 1e3:.3f} ms (indicatif)")
sys.exit(1 if ecarts else 0)
