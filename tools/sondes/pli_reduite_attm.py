"""PLI (16/10/2026) : lecture réduite portée dans attm (PLI_ATTENTION_REDUITE + PLI_ATTENTION_ATTM). Sans modèle.
1. Une couche réduite passe par attm (compteur « attm ») et non plus par l'ancien chemin.
2. Sur les clés gardées, attm est aussi juste qu'attm sur toutes les clés (écart à une référence f32 du même ordre).
3. Couche laissée pleine : même sortie qu'attm sans réduction (mêmes bits).
4. reduite=0 à chaud : mêmes bits qu'attm seul."""
import os, sys, tempfile

os.environ.setdefault("PLI_ATTENTION_REDUITE", "10")
os.environ["PLI_ATTENTION_ATTM"] = "1"
REG = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False).name
os.environ["PLI_ATTENTION_REGLAGE"] = REG
import mlx.core as mx  # noqa: E402
from tensorfold.kernels.qwen.dense.v1 import exact_attention as ea  # noqa: E402
from tensorfold.pli import attention_attm  # noqa: E402


class Cache:
    def __init__(self, bk, bv, L, couche):
        self.keys, self.values, self.offset, self.pli_couche = bk, bv, L, couche


def ref32(q, k, v, L, scale):
    T = q.shape[2]
    q32, k32, v32 = q.astype(mx.float32), k[:, :, :L].astype(mx.float32), v[:, :, :L].astype(mx.float32)
    out = []
    for t in range(T):
        n = L - T + t + 1
        kk = mx.repeat(k32[:, :, :n], 6, axis=1); vv = mx.repeat(v32[:, :, :n], 6, axis=1)
        s = (q32[:, :, t:t + 1] @ mx.swapaxes(kk, -1, -2)) * scale
        out.append(mx.softmax(s, axis=-1) @ vv)
    return mx.concatenate(out, axis=2)


def ecart(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


mx.random.seed(7)
scale = 256 ** -0.5
ok = True
for L, T in ((24000, 1), (24000, 2), (24000, 8), (40000, 4), (40000, 16)):
    cap = L + 2048
    bk = (mx.random.normal((1, 4, cap, 256)) * 1.0).astype(mx.bfloat16)
    bv = mx.random.normal((1, 4, cap, 256)).astype(mx.bfloat16)
    # quelques clés fortes dans le vieux contexte, pour que la carte ait quelque chose à choisir
    q = mx.random.normal((1, 24, T, 256)).astype(mx.bfloat16)
    keys, values = bk[:, :, :L], bv[:, :, :L]
    open(REG, "w").write("reduite=10\n"); os.utime(REG, None); ea._REGLAGE_VU[0] = 0.0
    avant = dict(ea.COMPTE)
    out_r = ea.exact_sdpa(q, keys, values, Cache(bk, bv, L, 10), scale, "causal" if T > 1 else None)
    passe_attm = ea.COMPTE["attm"] == avant["attm"] + 1
    ks, vs = ea._reduire(q, keys, values)
    Lr = int(ks.shape[2])
    e_r = ecart(out_r, ref32(q, ks, vs, Lr, scale))
    out_plein_attm = attention_attm.sdpa(q, bk, bv, L, scale)
    e_p = ecart(out_plein_attm, ref32(q, bk, bv, L, scale))
    out_pleine = ea.exact_sdpa(q, keys, values, Cache(bk, bv, L, 3), scale, "causal" if T > 1 else None)   # couche 3 : laissée pleine
    meme_pleine = bool(mx.all(out_pleine == out_plein_attm))
    open(REG, "w").write("reduite=0\n"); ea._REGLAGE_VU[0] = 0.0
    out_0 = ea.exact_sdpa(q, keys, values, Cache(bk, bv, L, 10), scale, "causal" if T > 1 else None)
    meme_0 = bool(mx.all(out_0 == out_plein_attm))
    if T < ea.REDUITE_MIN:                               # tour trop court : pas de réduction, attm sur toutes les clés
        passe_attm = (not passe_attm) and bool(mx.all(out_r == out_plein_attm)); e_r = e_p
    bon = passe_attm and meme_pleine and meme_0 and e_r <= 2 * e_p + 1e-3
    ok &= bon
    print("L=%d T=%d : gardées %d ; par attm %s ; écart réduite/f32 %.2e, attm entier/f32 %.2e ; couche pleine mêmes bits %s ; reduite=0 mêmes bits %s -> %s"
          % (L, T, Lr, passe_attm, e_r, e_p, meme_pleine, meme_0, "OK" if bon else "ECHEC"))
os.unlink(REG)
print("RESULTAT", "OK" if ok else "ECHEC")
sys.exit(0 if ok else 1)
