"""PLI bench: CPU requantization (requant_cpu.c) against the production GPU kernel (ffn.requant).

Real weights of a Qwen3.8-27B MLX 4-bit checkpoint (RQ_MODEL, its directory; layers RQ_LAYERS, read directly from the
safetensors: no model loaded), weights row by row and relaid out (relayout.convert), production shape: 18 units on two blocks (starts 8,192 and 12,800,
4,608 channels per part). Records, for each rounding mode: codes and scales equal to the GPU's; then the time of a whole
layer (2 parts x 3 matrices) on the CPU by number of threads, and that of the GPU kernel. Does not touch production.
Usage: PYTHONPATH=src python bench/rq_check.py RESULTS_DIR   (RQ_MODEL, RQ_LAYERS, RQ_THREADS, RQ_TIMING_MODE, RQ_PREFIX;
the names used before 3.7.0, RQ_MODELE, RQ_COUCHES, RQ_FILS, RQ_MODE_TEMPS, RQ_PREFIXE, still work)
"""

import json
import os
import statistics
import struct
import sys
import time
from types import SimpleNamespace

import mlx.core as mx
import numpy as np

from tensorfold.pli import relayout
from tensorfold.pli.ane import client, ffn
from tensorfold.pli.ane import requant_cpu as rq

OUTPUT = sys.argv[1]
os.makedirs(OUTPUT, exist_ok=True)


def _env(name: str, old: str, default: str) -> str:
    return os.environ.get(name, os.environ.get(old, default))


MODEL = os.path.expanduser(_env("RQ_MODEL", "RQ_MODELE", ""))
if not MODEL:
    sys.exit("set RQ_MODEL to the directory of a Qwen3.8-27B MLX 4-bit checkpoint")
LAYERS = [int(c) for c in _env("RQ_LAYERS", "RQ_COUCHES", "0,31,63").split(",")]
THREADS = [int(c) for c in _env("RQ_THREADS", "RQ_FILS", "2,4,6,8,12,16").split(",")]
TIMING_MODE = int(_env("RQ_TIMING_MODE", "RQ_MODE_TEMPS", "1"))
PREFIX = _env("RQ_PREFIX", "RQ_PREFIXE", "language_model.model.layers")
H, N, UNITS, BLOCKS = 5120, 17408, 18, 2
ANE = UNITS // BLOCKS * 512
STARTS = [N - BLOCKS * ANE + b * ANE for b in range(BLOCKS)]
log = open(os.path.join(OUTPUT, "rq-verif.jsonl"), "a")


def say(**kv):
    kv["time"] = time.strftime("%H:%M:%S")
    log.write(json.dumps(kv, ensure_ascii=False) + "\n")
    log.flush()
    print(json.dumps(kv, ensure_ascii=False), flush=True)


# ---- direct read of the safetensors (no loading of the whole model) ----
index = json.load(open(os.path.join(MODEL, "model.safetensors.index.json")))["weight_map"]
_headers: dict = {}


def tensor(name):
    path = os.path.join(MODEL, index[name])
    if path not in _headers:
        with open(path, "rb") as f:
            n = struct.unpack("<Q", f.read(8))[0]
            _headers[path] = (8 + n, json.loads(f.read(n)))
    base, header = _headers[path]
    t = header[name]
    start, end = t["data_offsets"]
    np_type = {"U32": np.uint32, "BF16": np.uint16, "F16": np.float16, "F32": np.float32}[t["dtype"]]
    a = np.fromfile(path, dtype=np_type, count=(end - start) // np.dtype(np_type).itemsize, offset=base + start)
    a = mx.array(a.reshape(t["shape"]))
    return mx.view(a, mx.bfloat16) if t["dtype"] == "BF16" else a


def mlp(layer, relaid):
    m = {}
    for p in ("gate_proj", "up_proj", "down_proj"):
        w, s, b = (tensor(f"{PREFIX}.{layer}.mlp.{p}.{k}") for k in ("weight", "scales", "biases"))
        if relaid:
            w, s = relayout.convert(w, s, b)
        mx.eval(w, s, b)
        m[p] = {"weight": w, "scales": s, "biases": b}
    return SimpleNamespace(**m)


sg = client.signs()
ffn._state = SimpleNamespace(sg128=mx.array(sg[:128]), sg512=mx.array(sg[:512]))
sg_np = np.ascontiguousarray(sg, dtype=np.float32)


def cpu(m, mode, threads, buffers=None):
    """All parts on the CPU, in numpy buffers with a widened stride (like the surfaces). Returns the 6 arrays per part."""
    g, u, d = rq.matrices(m)
    if buffers is None:
        buffers = []
        for _ in STARTS:
            buffers.append([np.zeros((ANE, H + 64), np.int8), np.zeros((ANE, 32), np.float16),
                            np.zeros((ANE, H + 64), np.int8), np.zeros((ANE, 32), np.float16),
                            np.zeros((H, ANE + 64), np.int8), np.zeros((H, 32), np.float16)])
    for (qg, eg, qu, eu, qd, ed), start in zip(buffers, STARTS):
        for M, q, e in ((g, qg, eg), (u, qu, eu)):
            rq.call(M, start, ANE, 0, H, 128, sg_np[:128], [(q.ctypes.data, q.strides[0], H)], (e.ctypes.data, 32),
                    mode, threads)
        rq.call(d, 0, H, start, ANE, 512, sg_np[:512], [(qd.ctypes.data, qd.strides[0], ANE)], (ed.ctypes.data, 32),
                mode, threads)
    return buffers


def compare(gpu, buffers):
    equal = total = e_equal = e_total = 0
    max_gap = 0
    for parts_gpu, t in zip(gpu, buffers):
        qg, eg, qu, eu, qd, ed = (np.asarray(a) for a in parts_gpu)
        for ref, ref_e, q, e, width in ((qg, eg, t[0], t[1], H), (qu, eu, t[2], t[3], H), (qd, ed, t[4], t[5], ANE)):
            c = q[:, :width]
            equal += int((c == ref).sum())
            total += ref.size
            max_gap = max(max_gap, int(np.abs(c.astype(np.int16) - ref.astype(np.int16)).max()))
            e_equal += int((e[:, 0].view(np.uint16) == ref_e.view(np.uint16)).sum())
            e_total += ref_e.size
    return equal, total, e_equal, e_total, max_gap


say(step="start", layers=LAYERS, threads=THREADS, starts=STARTS, ane=ANE, c_version=rq.lib().pli_rq_version())
for relaid in (True, False):
    for layer in LAYERS:
        m = mlp(layer, relaid)
        g, u, d = rq.matrices(m)
        no_copy = all(not M.w.flags.owndata for M in (g, u, d))
        gpu = [ffn.requant(m, start, ANE) for start in STARTS]
        mx.eval([a for p in gpu for a in p])
        for mode in (1, 3, 0):                    # 1: vector FMA (production); 3: scalar FMA; 0: two roundings
            t = cpu(m, mode, 8)
            eg_, tot, ee, et, emax = compare(gpu, t)
            say(step="accuracy", relaid=relaid, layer=layer, mode=mode, codes_equal=eg_, codes=tot,
                share_equal=round(eg_ / tot, 9), scales_equal=ee, scales=et, max_gap=emax, no_copy=no_copy,
                 rg=g.rg, kind=g.kind)
        if relaid and layer == LAYERS[0]:
            # time of a whole layer: GPU (production kernel) then CPU by number of threads
            for _ in range(3):
                mx.eval([a for start in STARTS for a in ffn.requant(m, start, ANE)])
            tg = []
            for _ in range(10):
                t0 = time.perf_counter()
                mx.eval([a for start in STARTS for a in ffn.requant(m, start, ANE)])
                tg.append((time.perf_counter() - t0) * 1e3)
            say(step="gpu_time", ms_median=round(statistics.median(tg), 3), ms_min=round(min(tg), 3))
            buffers = cpu(m, 1, 8)
            for threads in THREADS:
                tc = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    cpu(m, TIMING_MODE, threads, buffers)
                    tc.append((time.perf_counter() - t0) * 1e3)
                say(step="cpu_time", mode=TIMING_MODE, threads=threads, ms_median=round(statistics.median(tc), 3), ms_min=round(min(tc), 3))
        del m, gpu
        rq._cache.clear()
say(step="end")
