"""tree_anchor_trial.py (PLI, item D, 24/10/2026): anchored reduced prefill of tree windows (exact_attention._anchored_tree), outside the engine.
B1 a chain passed as a tree (parents in a chain) gives the SAME BITS as the anchored path of the chains (_anchored), new choice and kept choice.
B2 a tree: each node is close (gap of the same order as bf16) to its chain of ancestors passed through _anchored.
B3 only the choice of the accepted node enters the kept choice (commit_anchors), that of a refused sibling does not.
Usage: PLI_ATTENTION_REDUCED=10 PLI_REDUCED_ANCHOR=32 PLI_ATTENTION_ATTM=1 python tools/trials/tree_anchor_trial.py"""
import os
for k, v in (("PLI_ATTENTION_REDUCED", "10"), ("PLI_REDUCED_ANCHOR", "32"), ("PLI_ATTENTION_ATTM", "1"), ("PLI_ATTM_SHORT", "12288")):
    os.environ.setdefault(k, v)
import mlx.core as mx
from tensorfold.kernels.qwen.dense.v1 import exact_attention as E
H, HKV, D = 24, 4, 256
mx.random.seed(7)
class C:  # a layer's cache
    pass
def cache(start):
    c = C(); c.pli_request_start = start; c.pli_anchors = {}; c.pli_layer = 10; return c
res = {}
K = (mx.random.normal((1, HKV, 45100, D)) * 0.5).astype(mx.bfloat16)
V = (mx.random.normal((1, HKV, 45100, D)) * 0.5).astype(mx.bfloat16)
mx.eval(K, V)
for base in (30016, 45024):
    ca, cb = cache(21000), cache(21000)
    for p0, T in ((base, 8), (base + 28, 8), (base + 36, 3)):     # new choice; straddling an anchor (kept choice + new); kept
        L = p0 + T
        q = (mx.random.normal((1, H, T, D)) * 0.5).astype(mx.bfloat16)
        o1 = E._anchored(q, K[:, :, :L], V[:, :, :L], ca, 0.0625)
        o2 = E._anchored_tree(q, K[:, :, :L], V[:, :, :L], cb, 0.0625, tuple([-1] + list(range(T - 1))))
        E.commit_anchors([cb], list(range(T)))
        res["B1 %d x %d" % (p0, T)] = (o1 is not None and o2 is not None, o1 is not None and o2 is not None and bool(mx.array_equal(o1, o2).item()))
# B2 and B3: tree [-1, 0, 1, 2, 2, 0] whose nodes 3 and 4 (siblings, depth 3) are at position 30,016, an anchor
T = 6; p0 = 30013; L0 = p0 + T
par = (-1, 0, 1, 2, 2, 0)
k = (mx.random.normal((1, HKV, L0, D)) * 0.5).astype(mx.bfloat16)
v = (mx.random.normal((1, HKV, L0, D)) * 0.5).astype(mx.bfloat16)
q = (mx.random.normal((1, H, T, D)) * 0.5).astype(mx.bfloat16)
q0 = (mx.random.normal((1, H, 1, D)) * 0.5).astype(mx.bfloat16)
mx.eval(k, v, q, q0)
def with_memo():
    c = cache(21000)
    E._anchored(q0, k[:, :, :29985], v[:, :, :29985], c, 0.0625)      # kept choice of anchor 29,984 (row 29,984 written)
    return c
print("B1", {kk: vv for kk, vv in res.items()}, flush=True)
cb = with_memo()
print("memory after row 29,984:", sorted(cb.pli_anchors), dict(E._REASONS), flush=True)
o = E._anchored_tree(q, k, v, cb, 0.0625, par)
print("tree:", o is not None, dict(E._REASONS), flush=True)
att = dict(cb.pli_anchors_pending)
gaps = []
for i, path in ((2, [0, 1, 2]), (3, [0, 1, 2, 3]), (4, [0, 1, 2, 4]), (5, [0, 5])):
    kc = mx.concatenate([k[:, :, :p0], k[:, :, [p0 + j for j in path]]], axis=2)
    vc = mx.concatenate([v[:, :, :p0], v[:, :, [p0 + j for j in path]]], axis=2)
    ref = E._anchored(q[:, :, path], kc, vc, with_memo(), 0.0625)
    gaps.append(float(mx.max(mx.abs(o[:, :, i].astype(mx.float32) - ref[:, :, -1].astype(mx.float32))).item()))
res["B2 gaps of nodes 2, 3, 4, 5"] = gaps
E.commit_anchors([cb], [0, 1, 2, 4])
res["B3 pending choices"] = sorted(att)
kept = {a: m for a, m in cb.pli_anchors.items()}
res["B3 kept after validating path 0-1-2-4"] = sorted(kept)
b3 = sorted(att) == [(30016, 3), (30016, 4)] and 30016 in kept and all(bool(mx.array_equal(x, y).item()) for x, y in zip(kept[30016][1], att[(30016, 4)][1]))
print(res)
print("CRITERIA", {"B1": all(vv[0] and vv[1] for kk, vv in res.items() if kk.startswith("B1 ")), "B2": max(gaps) < 0.02, "B3": b3})
