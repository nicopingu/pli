"""tree2_trial.py (21/10/2026, item D; tree3 added at 11:15): the DeltaNet tree kernel "tree2" (PLI_TREE_DELTANET) gives the SAME BITS as the original "tree" kernel
and, for each node, as the chain of its ancestors passed to the chain kernel; shapes of the 27B (Hk 16, Hv 48, Dk = Dv = 128), 6 trees and the chain.
Time (indicative) of 48 calls that depend on each other (the output y of a call enters q of the next one)."""
import os, sys, time, statistics
os.environ["PLI_TREE_DELTANET"] = "1"
import mlx.core as mx
from tensorfold.kernels.qwen.dense.v1 import row_glue as RG
from tensorfold.kernels.qwen.dense.v1 import lane_tree as LT
Hk, Hv, D = 16, 48, 128
mx.random.seed(5)
TREES = [[-1, 0, 1, 2, 3, 4, 1, 2], [-1, 0, 0, 0, 1, 2, 3, 4], [-1, 0, 1, 1, 2, 3, 4, 5], [-1, 0, 1, 2, 3], [-1, 0, 0, 1, 1, 2, 2], [-1, 0, 1, 0, 3, 4, 1, 6]]
def inputs(W):
    q = (mx.random.normal((1, W, Hk, D)) * 0.1).astype(mx.bfloat16); k = (mx.random.normal((1, W, Hk, D)) * 0.1).astype(mx.bfloat16)
    v = mx.random.normal((1, W, Hv, D)).astype(mx.bfloat16)
    g = (mx.random.uniform(shape=(1, W, Hv)) * 0.5 + 0.5).astype(mx.float32); b = mx.random.uniform(shape=(1, W, Hv)).astype(mx.bfloat16)
    st = (mx.random.normal((1, Hv, D, D)) * 0.05).astype(mx.float32)
    mx.eval(q, k, v, g, b, st); return q, k, v, g, b, st
def ancestors(par, t):
    ch = []
    while t >= 0: ch.append(t); t = par[t]
    return ch[::-1]
ok = True
for par in TREES:
    W = len(par); q, k, v, g, b, st = inputs(W)
    RG.TREE_DELTANET = False; y0, _ = RG.gated_delta(q, k, v, g, b, st, par, chain=False)
    RG.TREE_DELTANET = True; RG.TREE_DELTANET3 = False; y1, _ = RG.gated_delta(q, k, v, g, b, st, par, chain=False)
    RG.TREE_DELTANET3 = True; y3, _ = RG.gated_delta(q, k, v, g, b, st, par, chain=False); RG.TREE_DELTANET3 = False
    eg = bool(mx.array_equal(y0, y1).item()) and bool(mx.array_equal(y0, y3).item())
    paths = True
    for t in range(W):
        idx = mx.array(ancestors(par, t)); n = len(ancestors(par, t))
        yc, _ = RG.gated_delta(q[:, idx], k[:, idx], v[:, idx], g[:, idx], b[:, idx], st, list(range(-1, n - 1)), chain=True)
        paths &= bool(mx.array_equal(yc[:, -1], y1[:, t]).item())
    print(par, "tree2 = tree3 = tree:", eg, "; = chain of the ancestors:", paths, flush=True)
    ok &= eg and paths
q, k, v, g, b, st = inputs(8)
def run(par, dn, dn3=False):
    RG.TREE_DELTANET = dn; RG.TREE_DELTANET3 = dn3
    def f():
        x = q
        for _ in range(48):
            y, _ = RG.gated_delta(x, k, v, g, b, st, par, chain=par == list(range(-1, 7)))
            x = (y[:, :, :Hk] * 0.1).astype(mx.bfloat16)
        mx.eval(x)
    ts = []
    for i in range(25):
        mx.synchronize(); a = time.perf_counter(); f(); mx.synchronize()
        if i >= 5: ts.append((time.perf_counter() - a) * 1e3)
    return statistics.median(ts)
print(f"48 chained calls: chain {run(list(range(-1, 7)), False):.2f} ms; tree tree {run(TREES[0], False):.2f} ms; tree tree2 {run(TREES[0], True):.2f} ms; tree tree3 {run(TREES[0], True, True):.2f} ms", flush=True)
print("CRITERION same bits:", ok)
