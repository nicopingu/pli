"""tensorfold.pli.ane.requant_cpu: requantization on the CPU of the weights of the FFN's Neural Engine part (21/10/2026).

The GPU kernel ffn.requant_kernel (5.59 ms per layer and per prefill chunk) and the numpy copy of
_State.write_weights are replaced by native/requant_cpu.c: the int8 codes and the scales are written directly into
the Neural Engine's surfaces, for the next layer, on a separate thread, while the GPU and the Neural Engine
compute the current layer. The GPU only carries the inputs, its part of the FFN and the join. No extra memory:
the CPU reads the model's 4-bit weights without copy (unified memory).

  PLI_ANE_REQUANT_CPU = 0 (default, off) | 1
  PLI_ANE_REQUANT_MODE = 1 (default) | 0: rounding of a weight's value, a single one after FMA (1, the Metal kernel's:
    bench/rq_check.py, 21/10: 100 % of the codes and scales) or two bf16 roundings (0); + 2: scalar path forced
  PLI_ANE_REQUANT_THREADS = 8: row parts spread by GCD
"""

from __future__ import annotations

import ctypes
import os
import subprocess
from typing import Any

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIBRARY = os.path.join(HERE, "native", "libpli_rq.dylib")
MODE = int(os.environ.get("PLI_ANE_REQUANT_MODE", "1"))
THREADS = int(os.environ.get("PLI_ANE_REQUANT_THREADS", "8"))
VERSION = 2

_lib: Any = None
_cache: dict = {}


def lib() -> Any:
    global _lib
    if _lib is None:
        if not os.path.exists(LIBRARY):
            done = subprocess.run(["bash", os.path.join(HERE, "build.sh")], capture_output=True, text=True)
            if done.returncode != 0 or not os.path.exists(LIBRARY):
                raise OSError(f"libpli_rq.dylib missing and not compiled: {(done.stderr or done.stdout)[-300:]}")
        L = ctypes.CDLL(LIBRARY)
        V, I, Z = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
        L.pli_rq_version.restype = I
        L.pli_rq_version.argtypes = []
        L.pli_rq_cpu.restype = I
        L.pli_rq_cpu.argtypes = [V, V, V, I, I, Z, I, I, I, I, I, V, I, V, V, V, V, Z, I, I]
        if L.pli_rq_version() != VERSION:
            raise OSError(f"libpli_rq.dylib version {L.pli_rq_version()}, expected {VERSION}: run build.sh again")
        _lib = L
    return _lib


def view(a: Any) -> np.ndarray:
    """numpy view without copy of an MLX array (bf16 read as uint16). Called on the main thread (MLX is not shared there)."""
    import mlx.core as mx

    if a.dtype == mx.bfloat16:
        a = mx.view(a, mx.uint16)
    mx.eval(a)
    v = np.asarray(a)
    if not v.flags.c_contiguous:
        raise ValueError("weights not contiguous: reading by the CPU impossible without copy")
    return v


class Matrix:
    """Views without copy of a 4-bit matrix (QuantizedLinear, relaid out P2 or row by row)."""

    def __init__(self, p: Any):
        import mlx.core as mx

        w, s = p["weight"], p["scales"]
        if w.dtype != mx.uint32 or w.ndim != 2:
            raise ValueError(f"weights {w.dtype} {w.shape}: 4 bits in 2-dimensional uint32 expected")
        self.K = int(w.shape[1]) * 8
        self.rg = 1 if s.dtype == mx.uint32 else 0
        self.kind = 2 if self.rg else (0 if s.dtype == mx.bfloat16 else 1)
        self.w, self.s = view(w), view(s)
        try:
            b = p["biases"]
        except (KeyError, AttributeError):
            b = None
        self.b = self.s if (self.rg or b is None) else view(b)


def matrices(m: Any) -> tuple:
    """(gate, up, down) of MLP m, kept in cache; to call on the main thread."""
    key = id(m)
    if key not in _cache:
        _cache[key] = (m, Matrix(m.gate_proj), Matrix(m.up_proj), Matrix(m.down_proj))
    return _cache[key][1:]


def call(M: Matrix, row0: int, nrows: int, C0: int, NC: int, BLOCK: int, sg: np.ndarray,
         outputs: list, scales: tuple, mode: int = MODE, threads: int = THREADS) -> None:
    """outputs: [(address, stride in bytes, width)]; scales: (address, stride in fp16 elements)."""
    n = len(outputs)
    seg_w = (ctypes.c_int * n)(*[int(w) for _, _, w in outputs])
    seg_p = (ctypes.c_void_p * n)(*[int(a) for a, _, _ in outputs])
    seg_s = (ctypes.c_size_t * n)(*[int(st) for _, st, _ in outputs])
    rc = lib().pli_rq_cpu(M.w.ctypes.data, M.s.ctypes.data, M.b.ctypes.data, M.kind, M.rg, M.K, row0, nrows, C0, NC,
                          BLOCK, sg.ctypes.data, n, seg_w, seg_p, seg_s, int(scales[0]), int(scales[1]), mode, threads)
    if rc:
        raise RuntimeError(f"pli_rq_cpu: code {rc} (row0={row0}, nrows={nrows}, C0={C0}, NC={NC}, BLOCK={BLOCK})")


def write(E: Any, m: Any, s: int, mode: int = MODE, threads: int = THREADS) -> None:
    """Weights of the Neural Engine part of MLP m into surface set s of state E (ffn._State), all parts.
    Same values as E.write_weights(s, E.requant(m)) when the mode reproduces the Metal kernel."""
    g, u, d = matrices(m)
    S = E.c.SEGMENT
    sg128, sg512 = E.sg_np[:128], E.sg_np[:512]
    for b, start in enumerate(E.starts):
        j = E.sets[s][b]
        for M, ws, es in ((g, j.wg, j.sg), (u, j.wu, j.su)):
            call(M, start, E.ane, 0, M.K, 128, sg128, [(x.base, x.stride, S) for x in ws], (es.base, es.stride // 2),
                 mode, threads)
        call(d, 0, E.H, start, E.ane, 512, sg512, [(x.base, x.stride, w) for x, w in zip(j.wd, E.f.seg_down)],
             (j.sd.base, j.sd.stride // 2), mode, threads)
