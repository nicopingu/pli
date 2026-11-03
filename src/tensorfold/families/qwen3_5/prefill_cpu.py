"""PLI (split prefill): during the prefill of a chunk, the CPU computes the MLP of the last rows.

Split by rows: the GPU keeps the first rows of the chunk, the CPU does gate, up, SiLU and down of the last ones with the
``qmm_cpu`` kernel (4-bit weights read in the GPU's buffer, without copy). No sum is stitched between devices: the rows
are put end to end. The CPU rows are computed in 32 bits: their values differ a little from the GPU's, so the
text may change (class V3 as long as an exact path does not replace it).

Off by default. ``PLI_PREFILL_CPU``: share of the rows given to the CPU (0 to 0.9). ``PLI_PREFILL_CPU_MIN``: number of rows
of the chunk below which nothing is shared (256). ``PLI_PREFILL_CPU_PRIORITY``: 0 high, 1 medium (default), 2 background.
``PLI_PREFILL_CPU_MODE``: "shifted" (default) or "layer" (old values "decale" and "couche" accepted).

"layer" mode: in each layer, the CPU does its rows while the GPU does the MLP of its own, then it waits.
"shifted" mode: the chunk is handled as two runs of rows (the GPU's, then the CPU's). The CPU does the MLP of
its rows for layer k while the GPU does the attention and the MLP of its own rows for layer k + 1 (the model
is causal: the GPU's rows do not depend on the CPU's). The GPU waits for the CPU only when it does the attention
of the CPU's rows.
"""
from __future__ import annotations

import os
import threading
import time
from typing import Any

from tensorfold.pli import env_names

STATS = {"layers": 0, "attention_ms": 0.0, "gpu_ms": 0.0, "cpu_ms": 0.0, "wait_ms": 0.0, "handoff_ms": 0.0,
         "gpu_rows": 0, "cpu_rows": 0}


def part() -> float:
    try:
        p = float(os.environ.get("PLI_PREFILL_CPU", "0") or 0)
    except ValueError:
        return 0.0
    return p if 0.0 < p <= 0.9 else 0.0


def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except ValueError:
        return default


def trace(rows: int, lc: int, t0: float) -> None:
    """PLI_PREFILL_CPU_TRACE=1: one line per chunk in the log (times accumulated since load, in ms)."""

    if os.environ.get("PLI_PREFILL_CPU_TRACE", "0") != "1":
        return
    import sys

    m = STATS
    print("[pli] split prefill: chunk of %d rows of which %d on the CPU, %.0f ms; total layers %d, GPU %.0f, attention of the "
          "CPU rows %.0f, CPU %.0f, wait for the CPU %.0f, handoff %.0f" % (rows, lc, (time.perf_counter() - t0) * 1000,
          m["layers"], m["gpu_ms"], m["attention_ms"], m["cpu_ms"], m["wait_ms"], m["handoff_ms"]), file=sys.stderr, flush=True)


def mode() -> str:
    value = env_names.value("PLI_PREFILL_CPU_MODE", os.environ.get("PLI_PREFILL_CPU_MODE", "shifted"))
    return "layer" if value == "layer" else "shifted"


class PrefillCPU:
    """The three matrices of each layer's MLP, as seen by the CPU kernel; one instance per model."""

    def __init__(self) -> None:
        self._weights: dict[int, Any] = {}
        self._kernel: Any = None
        self.minimum = max(16, _int_env("PLI_PREFILL_CPU_MIN", 256))

    def _load(self) -> Any:
        if self._kernel is None:
            from tensorfold.kernels.pli_cpu import qmm_cpu
            qmm_cpu.load()
            qmm_cpu.priority(_int_env("PLI_PREFILL_CPU_PRIORITY", 1))
            self._kernel = qmm_cpu
        return self._kernel

    def weights(self, layer: Any) -> Any:
        """(gate, up, down) readable without copy, or None if the layer does not have a dense 4-bit MLP in groups of 64."""

        key = id(layer)
        if key in self._weights:
            return self._weights[key]
        import mlx.core as mx

        trio: Any = None
        mlp = getattr(layer, "mlp", None)
        names = ("gate_proj", "up_proj", "down_proj")
        if mlp is not None and all(hasattr(mlp, n) for n in names):
            try:
                q = self._load()
                done = []
                for n in names:
                    m = getattr(mlp, n)
                    if (getattr(m, "bits", 0) != 4 or getattr(m, "group_size", 0) != 64 or getattr(m, "biases", None) is None
                            or "bias" in m or getattr(m, "mode", "affine") != "affine"):
                        raise ValueError("format not handled")
                    sb = {mx.bfloat16: 0, mx.float16: 1, mx.float32: 2}[m.scales.dtype]
                    if m.biases.dtype != m.scales.dtype:
                        raise ValueError("scale types")
                    mx.eval(m.weight, m.scales, m.biases)
                    done.append(q.Weights(m.weight, m.scales, m.biases, sb=sb))
                trio = tuple(done)
            except Exception:
                trio = None
        self._weights[key] = trio
        return trio

    def layer(self, layer: Any, x: Any, mask: Any, cache: Any, p: float) -> Any:
        """One layer of MLX's pass, the MLP of the last rows done by the CPU while the GPU does the others."""

        import mlx.core as mx
        import numpy as np

        from tensorfold.engine.family_common import cache_arrays

        trio = self.weights(layer)
        rows = int(x.shape[1])
        lc = (int(rows * p) // 8) * 8
        if trio is None or x.shape[0] != 1 or rows < self.minimum or lc < 8 or lc >= rows:
            return layer(x, mask=mask, cache=cache)
        q = self._kernel
        t0 = time.perf_counter()
        if layer.is_linear:
            r = layer.linear_attn(layer.input_layernorm(x), mask, cache)
        else:
            r = layer.self_attn(layer.input_layernorm(x), mask, cache)
        h = x + r
        xn = layer.post_attention_layernorm(h)
        mx.eval(xn, h, *cache_arrays([cache]))
        t1 = time.perf_counter()
        xc = np.array(xn[0, rows - lc:].astype(mx.float32))
        box: dict[str, Any] = {}

        def work() -> None:
            a = time.perf_counter()
            try:
                g = q.product("blas", trio[0], xc)
                u = q.product("blas", trio[1], xc)
                q.swiglu(g, u)
                box["y"] = q.product("blas", trio[2], g)
            except Exception as err:                       # the thread must never take the prefill down
                box["error"] = err
            box["ms"] = (time.perf_counter() - a) * 1000

        thread = threading.Thread(target=work, name="pli-prefill-cpu")
        t2 = time.perf_counter()
        thread.start()
        yg = layer.mlp(xn[:, :rows - lc])
        mx.eval(yg)
        t3 = time.perf_counter()
        thread.join()
        t4 = time.perf_counter()
        if "y" not in box:                                 # fallback: the GPU also does the CPU's rows
            yc = layer.mlp(xn[:, rows - lc:])
        else:
            yc = mx.array(box["y"]).astype(yg.dtype)[None]
        out = h + mx.concatenate([yg, yc], axis=1)
        M = STATS
        M["layers"] += 1; M["attention_ms"] += (t1 - t0) * 1000; M["handoff_ms"] += (t2 - t1) * 1000
        M["gpu_ms"] += (t3 - t2) * 1000; M["cpu_ms"] += box.get("ms", 0.0); M["wait_ms"] += (t4 - t3) * 1000
        M["gpu_rows"] += rows - lc; M["cpu_rows"] += lc
        return out

    def cpu_rows(self, layers: Any, h: Any, p: float) -> int:
        """Number of the chunk's rows given to the CPU in shifted mode; 0 = do not share this chunk."""

        rows = int(h.shape[1])
        lc = (int(rows * p) // 8) * 8
        if h.shape[0] != 1 or rows < self.minimum or lc < 8 or lc >= rows:
            return 0
        return lc if all(self.weights(layer) is not None for layer in layers) else 0

    def shifted_pass(self, layers: Any, caches: Any, h: Any, fa_mask: Any, ssm_mask: Any, lc: int, step: int,
                     captured: Any = None) -> Any:
        """Generator: all layers in shifted mode; yields every ``step`` layers (0 = never).

        Final value: the output states of the whole chunk, before the final norm."""

        import mlx.core as mx
        import numpy as np

        from tensorfold.engine.family_common import cache_arrays

        q = self._kernel
        start = time.perf_counter()
        g = int(h.shape[1]) - lc
        hg, hc = h[:, :g], h[:, g:]
        pending: Any = None                                 # (thread, box, h after attention, MLP input, layer)
        M = STATS

        def finish(a: Any) -> Any:
            thread, box, h1, xn, layer = a
            t = time.perf_counter()
            thread.join()
            M["wait_ms"] += (time.perf_counter() - t) * 1000
            M["cpu_ms"] += box.get("ms", 0.0)
            y = box.get("y")
            yc = layer.mlp(xn) if y is None else mx.array(y).astype(h1.dtype)[None]     # fallback: the GPU redoes these rows
            return h1 + yc

        def attention(layer: Any, x: Any, c: Any) -> Any:
            if layer.is_linear:
                return x + layer.linear_attn(layer.input_layernorm(x), ssm_mask(x, c), c)
            return x + layer.self_attn(layer.input_layernorm(x), fa_mask(x, c), c)

        total = len(layers)
        for k, (layer, c) in enumerate(zip(layers, caches)):
            trio = self.weights(layer)
            t0 = time.perf_counter()
            h1g = attention(layer, hg, c)
            hg = h1g + layer.mlp(layer.post_attention_layernorm(h1g))
            mx.eval(hg, *cache_arrays([c]))
            t1 = time.perf_counter()
            if pending is not None:
                hc = finish(pending)
            t2 = time.perf_counter()
            h1c = attention(layer, hc, c)
            xn = layer.post_attention_layernorm(h1c)
            mx.eval(xn, h1c, *cache_arrays([c]))
            xc = np.array(xn[0].astype(mx.float32))
            t3 = time.perf_counter()
            box: dict[str, Any] = {}

            def work(trio: Any = trio, xc: Any = xc, box: dict[str, Any] = box) -> None:
                a = time.perf_counter()
                try:
                    gg = q.product("blas", trio[0], xc)
                    uu = q.product("blas", trio[1], xc)
                    q.swiglu(gg, uu)
                    box["y"] = q.product("blas", trio[2], gg)
                except Exception as err:
                    box["error"] = err
                box["ms"] = (time.perf_counter() - a) * 1000

            thread = threading.Thread(target=work, name="pli-prefill-cpu")
            thread.start()
            pending = (thread, box, h1c, xn, layer)
            M["layers"] += 1; M["gpu_ms"] += (t1 - t0) * 1000; M["attention_ms"] += (t3 - t2) * 1000
            M["gpu_rows"] += g; M["cpu_rows"] += lc
            done = k + 1
            if step > 0 and done % step == 0 and done < total:
                kept = list(captured) if captured else None      # the draft's captured states: see prefill_steps
                yield
                if kept is not None:
                    captured[:] = kept
        hc = finish(pending)
        output = mx.concatenate([hg, hc], axis=1)
        mx.eval(output)
        trace(g + lc, lc, start)
        return output
