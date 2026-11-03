"""Trial D2 (PLI, 16/10/2026): attm attention with an ancestor mask (pli/attention_attm.sdpa_tree), outside the engine.
A1 a chain passed through sdpa_tree gives the SAME BITS as sdpa (original kernel), for T = 1, 2, 5, 8, 16 and L = 40, 3,000, 33,000.
A2 a tree: gap to an f32 reference (numpy, context keys + ancestors) of the same order as that of the chain to its reference (ratio <= 2).
A3 a tree whose row has only the root as ancestor does not depend on the other branches (changing the values of another branch does not change its output, bits).
Usage: tools/trials/attm_tree_trial.py OUTPUT.json"""
import json, sys
import mlx.core as mx
import numpy as np
from tensorfold.pli import attention_attm as A
rng = np.random.RandomState(3)
H, HKV, D = 24, 4, 256
res = {"A1": [], "A2": [], "A3": None}
def buffers(cap):
    k = rng.standard_normal((1, HKV, cap, D)).astype(np.float32); v = rng.standard_normal((1, HKV, cap, D)).astype(np.float32)
    return mx.array(k).astype(mx.bfloat16), mx.array(v).astype(mx.bfloat16)
def reference(q, bk, bv, L, T, an, scale):
    qf = np.array(q.astype(mx.float32))[0]; kf = np.array(bk.astype(mx.float32))[0]; vf = np.array(bv.astype(mx.float32))[0]
    out = np.zeros((H, T, D), np.float32)
    for t in range(T):
        vis = list(range(L - T)) + [L - T + j for j in range(T) if (an[t] >> j) & 1]
        for h in range(H):
            g = h // (H // HKV)
            s = kf[g, vis] @ qf[h, t] * scale
            p = np.exp(s - s.max()); p /= p.sum()
            out[h, t] = p @ vf[g, vis]
    return out
scale = 256 ** -0.5
for L in (40, 3000, 33000):
    bk, bv = buffers(L + 64)
    for T in (1, 2, 5, 8, 16):
        if T > L: continue
        q = mx.array(rng.standard_normal((1, H, T, D)).astype(np.float32)).astype(mx.bfloat16)
        a = A.sdpa(q, bk, bv, L, scale); b = A.sdpa_tree(q, bk, bv, L, scale, [-1] + list(range(T - 1)))
        mx.eval(a, b)
        equal = bool(mx.array_equal(a, b).item())
        res["A1"].append({"L": L, "T": T, "same_bits": equal}); print("A1", L, T, equal, flush=True)
        if T >= 5:
            par = [-1, 0, 0, 1, 2] + [3 + (i % 2) for i in range(T - 5)]
            an = A.ancestors(par)
            c = A.sdpa_tree(q, bk, bv, L, scale, par); mx.eval(c)
            ra = reference(q, bk, bv, L, T, A.ancestors([-1] + list(range(T - 1))), scale)
            rc = reference(q, bk, bv, L, T, an, scale)
            ea = float(np.abs(np.array(a.astype(mx.float32))[0] - ra).max()); ec = float(np.abs(np.array(c.astype(mx.float32))[0] - rc).max())
            res["A2"].append({"L": L, "T": T, "chain_gap": ea, "tree_gap": ec}); print("A2", L, T, ea, ec, flush=True)
# A3: row 2 (parent 0) does not see row 1; changing the key and the value of row 1 does not change the output of row 2
L, T = 3000, 5; bk, bv = buffers(L + 64)
q = mx.array(rng.standard_normal((1, H, T, D)).astype(np.float32)).astype(mx.bfloat16)
par = [-1, 0, 0, 1, 2]
c1 = A.sdpa_tree(q, bk, bv, L, scale, par)
bk2 = mx.array(np.array(bk.astype(mx.float32))).astype(mx.bfloat16); bv2 = mx.array(np.array(bv.astype(mx.float32))).astype(mx.bfloat16)
kk = np.array(bk2.astype(mx.float32)); vv = np.array(bv2.astype(mx.float32)); kk[:, :, L - T + 1] += 5.0; vv[:, :, L - T + 1] -= 5.0
c2 = A.sdpa_tree(q, mx.array(kk).astype(mx.bfloat16), mx.array(vv).astype(mx.bfloat16), L, scale, par); mx.eval(c1, c2)
res["A3"] = {"row2_unchanged": bool(mx.array_equal(c1[:, :, 2], c2[:, :, 2]).item()), "row3_changed": not bool(mx.array_equal(c1[:, :, 3], c2[:, :, 3]).item())}
print("A3", res["A3"], flush=True)
crit = {"A1": all(x["same_bits"] for x in res["A1"]), "A2": all(x["tree_gap"] <= 2 * max(x["chain_gap"], 1e-3) for x in res["A2"]),
        "A3": res["A3"]["row2_unchanged"] and res["A3"]["row3_changed"]}
res["criteria"] = crit; print("CRITERIA", crit, flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1); sys.exit(0 if all(crit.values()) else 4)
