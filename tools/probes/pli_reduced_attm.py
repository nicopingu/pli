"""PLI (16/10/2026): reduced prefill carried into attm (PLI_ATTENTION_REDUCED + PLI_ATTENTION_ATTM). No model needed.
1. A reduced layer goes through attm (counter "attm") and no longer through the old path.
2. On the kept keys, attm is as accurate as attm on all keys (gap to an f32 reference of the same order).
3. Layer left full: same output as attm without reduction (same bits).
4. reduced=0 on the fly: same bits as attm alone."""
import os, sys, tempfile

os.environ.setdefault("PLI_ATTENTION_REDUCED", "10")
os.environ["PLI_ATTENTION_ATTM"] = "1"
REG = tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False).name
os.environ["PLI_ATTENTION_TUNING_FILE"] = REG
import mlx.core as mx  # noqa: E402
from tensorfold.kernels.qwen.dense.v1 import exact_attention as ea  # noqa: E402
from tensorfold.pli import attention_attm  # noqa: E402


class Cache:
    def __init__(self, bk, bv, L, layer):
        self.keys, self.values, self.offset, self.pli_layer = bk, bv, L, layer


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


def gap(a, b):
    return float(mx.max(mx.abs(a.astype(mx.float32) - b.astype(mx.float32))))


mx.random.seed(7)
scale = 256 ** -0.5
ok = True
for L, T in ((24000, 1), (24000, 2), (24000, 8), (40000, 4), (40000, 16)):
    cap = L + 2048
    bk = (mx.random.normal((1, 4, cap, 256)) * 1.0).astype(mx.bfloat16)
    bv = mx.random.normal((1, 4, cap, 256)).astype(mx.bfloat16)
    # a few strong keys in the old context, so that the map has something to choose
    q = mx.random.normal((1, 24, T, 256)).astype(mx.bfloat16)
    keys, values = bk[:, :, :L], bv[:, :, :L]
    open(REG, "w").write("reduced=10\n"); os.utime(REG, None); ea._TUNING_SEEN[0] = 0.0
    before = dict(ea.COUNT)
    out_r = ea.exact_sdpa(q, keys, values, Cache(bk, bv, L, 10), scale, "causal" if T > 1 else None)
    via_attm = ea.COUNT["attm"] == before["attm"] + 1
    ks, vs = ea._reduce(q, keys, values)
    Lr = int(ks.shape[2])
    e_r = gap(out_r, ref32(q, ks, vs, Lr, scale))
    out_full_attm = attention_attm.sdpa(q, bk, bv, L, scale)
    e_p = gap(out_full_attm, ref32(q, bk, bv, L, scale))
    out_full = ea.exact_sdpa(q, keys, values, Cache(bk, bv, L, 3), scale, "causal" if T > 1 else None)   # layer 3: left full
    same_full = bool(mx.all(out_full == out_full_attm))
    open(REG, "w").write("reduced=0\n"); ea._TUNING_SEEN[0] = 0.0
    out_0 = ea.exact_sdpa(q, keys, values, Cache(bk, bv, L, 10), scale, "causal" if T > 1 else None)
    same_0 = bool(mx.all(out_0 == out_full_attm))
    if T < ea.REDUCED_MIN_ROWS:                          # round too short: no reduction, attm on all the keys
        via_attm = (not via_attm) and bool(mx.all(out_r == out_full_attm)); e_r = e_p
    good = via_attm and same_full and same_0 and e_r <= 2 * e_p + 1e-3
    ok &= good
    print("L=%d T=%d: kept %d; through attm %s; gap reduced/f32 %.2e, full attm/f32 %.2e; full layer same bits %s; reduced=0 same bits %s -> %s"
          % (L, T, Lr, via_attm, e_r, e_p, same_full, same_0, "OK" if good else "FAIL"))
os.unlink(REG)
print("RESULT", "OK" if ok else "FAIL")
sys.exit(0 if ok else 1)
