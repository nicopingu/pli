"""PLI SP1: dense FFN of the 27B split by channels between the GPU and the Neural Engine during prompt prefill (V3).

Taken from Splash 1.3.2-m1 (runtime/ops/AneFfn.cpp, Apache-2.0, see native/LICENSE-splash). The GPU computes the head
intermediate channels (the PLI_PREFILL_UNPACK_MLP=t path of unpack.py, restricted to these channels), the Neural Engine the tail
ones in W8A8 (program of client.py). The two parts of down are added in fp32. The text changes (V3).

Path a: one synchronisation by the CPU at each layer. Order of a layer i (surface set s = i % 2):
  1. GPU: inputs rotated and quantized to int8 (and the weights of layer i if set s does not hold them), evaluated;
  2. CPU: copy into the surfaces of set s, Neural Engine evaluation started on a thread (ctypes releases Python's lock);
  3. GPU: weights of the next layer requantized (first) then the GPU part of layer i; the CPU copies the next
     weights into the other set while the GPU and the Neural Engine compute;
  4. wait for the Neural Engine, copy of its output, join on the GPU.

  PLI_ANE_FFN = 0 (default, off) | 1
  PLI_ANE_FFN_UNITS = 12: units of 512 channels on the Neural Engine (measured: 12 -> T = 0.66 L at 2,048 rows)
  PLI_ANE_FFN_MIN_ROWS = 512: chunks of at least this number of rows (at most 2,048); below, GPU path unchanged
    (22/10/2026: floor lowered from 512 to 128; measured: shared layer 0.84 of the exact kernel at 256 rows, 0.68 at 384)
  PLI_ANE_BLOCKS = 1 (default, 3.3.0) | 2: units cut into two programs pinned to two blocks (PLI_ANE_BLOCK_IDS = 1,3);
    measured on 16/10/2026: 18 units on two blocks -> T = 0.76 of 3.3.0 at 2,048 rows
  PLI_ANE_REQUANT_CPU = 0 (default) | 1: weights of the Neural Engine part requantized on the CPU (requant_cpu.py),
    written directly into the surfaces during the previous layer; the GPU no longer has the pli_anerq kernel
Takes only the MLPs already set up by unpack._install_mlp (PLI_PREFILL_UNPACK=f16 and PLI_PREFILL_UNPACK_MLP=t), weights relaid out
(P2) or not. Decode and small chunks: nothing changes. Circuit breaker: at the first error, the layer is redone on the GPU
alone and the split stops for the process.
"""

from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

ACTIVE = os.environ.get("PLI_ANE_FFN", "0") == "1"
UNITS = int(os.environ.get("PLI_ANE_FFN_UNITS", "12"))
MIN_ROWS = max(128, int(os.environ.get("PLI_ANE_FFN_MIN_ROWS", "512")))   # floor 128 (22/10/2026)
MAX_ROWS = 2048
# Steps (23/10/2026, measured with keep-awake): Neural Engine units by chunk size, to balance
# the Neural Engine and the GPU part. Shared layer (ms) at 18 / 16 / 14 units: 256 rows 13.0 / 12.1 / 11.4; 384 rows
# 16.2 / 13.5 / 13.4; 511 rows 18.6 / 16.2 / 16.5; 768 rows 24.0 / 20.2 / 21.6; 1,024 rows 28.6 / 25.8 / 27.7
# (exact GPU kernel 16.3 / 24.1 / 32.1 at 256 / 384 / 511). PLI_ANE_STEPS = "rows:units,..." (e.g. 256:14,512:16,1536:18):
# one state (programs, surfaces) per step, each for the chunks of at least this number of rows; empty = a single state
# (PLI_ANE_FFN_UNITS, PLI_ANE_FFN_MIN_ROWS), as before.
STEPS = sorted((int(a), int(b)) for a, b in (p.split(":") for p in os.environ.get("PLI_ANE_STEPS", "").split(",") if p.strip()))
if STEPS:
    MIN_ROWS = max(128, STEPS[0][0])
# Requantization after the call (23/10/2026): the CPU requantization of the next layer during
# the Neural Engine call slows it down (256 rows: 10.1 ms alone, 14.0 with; 1,024 rows: 22.8 against 25.0), the GPU part
# does not (10.2). PLI_ANE_REQUANT_AFTER = 1: it starts after the call, during the join and the next layer's attention.
RQ_AFTER = os.environ.get("PLI_ANE_REQUANT_AFTER", "0") == "1"
BLOCKS = int(os.environ.get("PLI_ANE_BLOCKS", "1"))
if BLOCKS > 1:
    # two programs pinned to two blocks: only the Ultra chips have two Neural Engines
    import subprocess as _sp

    _chip = _sp.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True).stdout
    if "Ultra" not in _chip:
        print(f"[tensorfold] PLI_ANE_BLOCKS={BLOCKS} needs an Ultra chip ({_chip.strip() or 'unknown chip'}): 1 block", flush=True)
        BLOCKS = 1
BLOCK_IDS = [int(v) for v in os.environ.get("PLI_ANE_BLOCK_IDS", "1,3").split(",")]
REQUANT_CPU = os.environ.get("PLI_ANE_REQUANT_CPU", "0") == "1"
# Keep-awake (22/10/2026): the Neural Engine slows down after 2 ms of rest (call of
# 256 rows: 10.1 ms without rest, 19.9 after 20 ms; 2,048 rows: 40.1 against 53.0). In prefill, attention and
# DeltaNet run on the GPU between two FFNs: it rests at each layer. A thread chains tiny calls (FFN
# 2,560 -> 512 channels, 128 rows, on each block) between two shared layers and stops just before each FFN call
# (measured delay 0.3 to 0.5 ms); it goes quiet PLI_ANE_KEEPAWAKE_MS after the last shared layer.
KEEPAWAKE = os.environ.get("PLI_ANE_KEEPAWAKE", "0") == "1"
KEEPAWAKE_S = float(os.environ.get("PLI_ANE_KEEPAWAKE_MS", "500")) / 1e3
STEP = 128
count = {"layers": 0, "ane": 0, "gpu_only": 0, "requant": 0, "ane_ms": 0.0, "wait_ms": 0.0, "tripped": ""}
# PLI (24/10/2026, load measurement): PLI_ANE_STATS=<file> writes every 0.5 s "time ane_ms rq_cpu_ms calls"
# (totals since start) to follow the Neural Engine's occupancy; powermetrics reads 0 mW there. Off by default.
_STATS = os.environ.get("PLI_ANE_STATS", "").strip()
if _STATS:
    import threading as _th

    def _record_stats() -> None:
        try:
            with open(_STATS, "a") as f:
                while True:
                    f.write(f"{time.time():.3f} {count['ane_ms']:.2f} {count.get('rq_cpu_ms', 0.0):.2f} {count['ane']}\n")
                    f.flush()
                    time.sleep(0.5)
        except Exception:
            pass

    _th.Thread(target=_record_stats, daemon=True, name="pli-ane-stats").start()
_state: Any = None
_states: list = []         # (rows at least, state) per step (PLI_ANE_STEPS), increasing
_classes: dict = {}
_kernels: dict = {}


def _say(*a: Any) -> None:
    print("PLI ane_ffn:", *a, file=sys.stderr, flush=True)


# ---- Metal kernels ----

_LOAD_RG = ("  const size_t p = row / 32, t = (row / 8) % 4, fm = row % 8;\n"
            "  const uint w32 = W[2 * (((p * G + g) * 32 + fm * 4 + fnh) * 4 + t) + hi];\n"
            "  const uint sbw = SC[((p * G + g) * 8 + fm) * 4 + t];\n"
            "  const float scf = as_type<float>(sbw << 16), bif = as_type<float>(sbw & 0xFFFF0000u);\n")
_LOAD = ("  const uint w32 = W[row * (K / 8) + 8 * g + 2 * fnh + hi];\n"
         "  const float scf = float(SC[row * G + g]), bif = float(BI[row * G + g]);\n")
# value of a weight: same expression as unpack._kernel (scale and bias rounded to bf16, scale * d + bias in bf16)
_VALUE = ("  const bfloat sb_ = bfloat(scf), bb_ = bfloat(bif);\n"
          "  const uchar d = uchar((w32 >> (4 * (k & 7))) & 0xFu);\n"
          "  const bfloat v_ = sb_ * d + bb_;\n")


def _weight(col: str) -> str:
    """Code that reads the weight (row, col) into `v` (float); K, G, rg set by the caller."""
    return (f"  {{ const size_t c_ = {col}; const size_t g = c_ / 64, k = c_ % 64, fnh = k / 16, hi = (k & 15) >= 8 ? 1 : 0;\n")


def requant_kernel(K: int, C0: int, NC: int, BLOCK: int, rg: bool) -> Any:
    """Requantization of a 4-bit matrix for the Neural Engine, one row per group of 256 threads: columns [C0, C0+NC)
    unpacked, Hadamard rotation in blocks of BLOCK (o = sg[o] / sqrt(BLOCK) * sum H[o][i] v[i], fast transform over a
    SIMD group), then int8 per row (peak / 127, round half to even, bounded to 127) like client.int8_rows. Outputs: Q int8
    (rows, NC) and E fp16 (rows) = scale * 128."""

    import mlx.core as mx

    key = ("rq", K, C0, NC, BLOCK, rg)
    if key in _kernels:
        return _kernels[key]
    J = BLOCK // 32
    reg_stages = {128: 2, 512: 4}[BLOCK]
    src = (f"  constexpr int K = {K}, G = {K // 64}, C0 = {C0}, NC = {NC}, BLOCK = {BLOCK}, J = {J};\n"
           "  threadgroup float vals[NC];\n  threadgroup float peaks[8];\n"
           "  const uint row = threadgroup_position_in_grid.x;\n"
           "  const uint l = thread_index_in_simdgroup, sgi = simdgroup_index_in_threadgroup;\n"
           "  const uint tid = thread_position_in_threadgroup.x;\n"
           f"  const float norm = {1.0 / (BLOCK ** 0.5)!r}f;\n"
           "  float local_peak = 0.0f;\n"
           "  for (int b = int(sgi); b < NC / BLOCK; b += 8) {\n"
           "    float r[J];\n"
           "    for (int j = 0; j < J; j++) {\n"
           + _weight("C0 + b * BLOCK + j * 32 + l")
           + (_LOAD_RG if rg else _LOAD).replace("\n", "\n  ")
           + _VALUE.replace("\n", "\n  ") +
           "    r[j] = float(v_); }\n    }\n"
           # stages in the register: steps 32, 64, (128, 256)
           f"    for (int m = 0; m < {reg_stages}; m++) {{\n"
           "      const int s = 1 << m;\n"
           "      for (int j = 0; j < J; j++) if ((j & s) == 0) { const float a = r[j], c = r[j + s]; r[j] = a + c; r[j + s] = a - c; }\n"
           "    }\n"
           # stages across threads: steps 1 to 16
           "    for (uint h = 1; h < 32; h <<= 1) {\n"
           "      for (int j = 0; j < J; j++) { const float a = r[j], c = simd_shuffle_xor(a, h); r[j] = (l & h) ? (c - a) : (a + c); }\n"
           "    }\n"
           "    for (int j = 0; j < J; j++) {\n"
           "      const int o = j * 32 + int(l);\n"
           "      const float y = r[j] * (SG[o] * norm);\n"
           "      vals[b * BLOCK + o] = y; local_peak = max(local_peak, fabs(y));\n"
           "    }\n  }\n"
           "  local_peak = simd_max(local_peak);\n"
           "  if (l == 0) peaks[sgi] = local_peak;\n"
           "  threadgroup_barrier(mem_flags::mem_threadgroup);\n"
           "  float peak = peaks[0];\n  for (int i = 1; i < 8; i++) peak = max(peak, peaks[i]);\n"
           "  peak = max(peak, 6.103515625e-05f);\n"
           "  const float e = peak / 127.0f;\n"
           "  for (int c = int(tid); c < NC; c += 256) {\n"
           "    const float q = clamp(rint(vals[c] / e), -127.0f, 127.0f);\n"
           "    Q[size_t(row) * NC + c] = char(q);\n  }\n"
           "  if (tid == 0) E[row] = half(e * 128.0f);\n")
    _kernels[key] = mx.fast.metal_kernel(name="pli_anerq_" + hashlib.sha256(src.encode()).hexdigest()[:12],
                                         input_names=["W", "SC", "BI", "SG"], output_names=["Q", "E"], source=src, header="")
    return _kernels[key]


def columns_kernel(T: str, K: int, KO: int, rg: bool) -> Any:
    """Unpacking of the columns [0, KO) of each row of a 4-bit matrix of K columns (relaid out or not), same values as
    unpack._kernel: one thread per (row, group of 64, quarter of group)."""

    import mlx.core as mx

    key = ("col", T, K, KO, rg)
    if key in _kernels:
        return _kernels[key]
    src = (f"  constexpr int K = {K}, G = {K // 64}, GO = {KO // 64}, KO = {KO};\n  typedef {T} T;\n"
           "  const size_t i = thread_position_in_grid.x;\n"
           "  const size_t fnh = i & 3, g = (i >> 2) % GO, row = i / (4 * GO);\n"
           "  device T* o = OUT + row * KO + g * 64 + fnh * 16;\n"
           "  for (int hi = 0; hi < 2; hi++) {\n"
           + (_LOAD_RG if rg else _LOAD).replace("\n", "\n  ") +
           "  const bfloat sb_ = bfloat(scf), bb_ = bfloat(bif);\n"
           "    for (int k = 0; k < 8; k++) {\n"
           "      const uchar d = uchar((w32 >> (4 * k)) & 0xFu);\n"
           "      const bfloat v_ = sb_ * d + bb_;\n      o[hi * 8 + k] = T(float(v_));\n    }\n  }\n")
    _kernels[key] = mx.fast.metal_kernel(name="pli_anecol_" + hashlib.sha256(src.encode()).hexdigest()[:12],
                                         input_names=["W", "SC", "BI"], output_names=["OUT"], source=src, header="")
    return _kernels[key]


def _rg(m: Any) -> bool:
    import mlx.core as mx

    return m["scales"].dtype == mx.uint32


def requant(m: Any, n0: int, ane: int) -> Any:
    """Weights of the Neural Engine part of MLP m: (qg, eg, qu, eu, qd, ed), int8 and fp16 (scale * 128)."""

    import mlx.core as mx

    E = _state
    out = []
    for p in (m.gate_proj, m.up_proj):
        w, s, b = p["weight"], p["scales"], p["biases"]
        K = int(w.shape[1]) * 8
        k = requant_kernel(K, 0, K, 128, _rg(p))
        out += k(inputs=[w[n0:n0 + ane], s[n0:n0 + ane], b[n0:n0 + ane], E.sg128], grid=(ane * 256, 1, 1), threadgroup=(256, 1, 1),
                 output_shapes=[(ane, K), (ane,)], output_dtypes=[mx.int8, mx.float16])
    d = m.down_proj
    w, s, b = d["weight"], d["scales"], d["biases"]
    N, K = int(w.shape[0]), int(w.shape[1]) * 8
    k = requant_kernel(K, n0, ane, 512, _rg(d))
    out += k(inputs=[w, s, b, E.sg512], grid=(N * 256, 1, 1), threadgroup=(256, 1, 1),
             output_shapes=[(N, ane), (N,)], output_dtypes=[mx.int8, mx.float16])
    return out


def inputs(x2: Any) -> tuple[Any, Any]:
    """Neural Engine inputs: x rotated in blocks of 128, int8 per token, laid out per channel (H, rows); tx = scale * 128."""

    import mlx.core as mx

    E = _state
    rows, H = int(x2.shape[0]), int(x2.shape[1])
    xr = (x2.astype(mx.float32).reshape(rows, H // 128, 128) @ E.r128t).reshape(rows, H)
    peak = mx.maximum(mx.abs(xr).max(axis=1), 2.0 ** -14)
    sx = peak / 127.0
    q = mx.clip(mx.round(xr / sx[:, None]), -127, 127).astype(mx.int8)
    return mx.contiguous(q.T), (sx * 128.0).astype(mx.float16)


def gpu_part(m: Any, x2: Any, n0: int) -> Any:
    """GPU part, transposed: yT (H, rows) in bf16 of the sum over channels [0, n0). Same operations as unpack._mlp_t."""

    import mlx.core as mx

    from tensorfold.pli import unpack

    if unpack._swc is None:
        from mlx_lm.models.activations import swiglu

        unpack._swc = mx.compile(lambda a, b: swiglu(a.astype(mx.bfloat16), b.astype(mx.bfloat16)), shapeless=True)
    g, u, d = m.gate_proj, m.up_proj, m.down_proj

    def head(p: Any) -> Any:
        return unpack._unpack_wsb(p["weight"][:n0], p["scales"][:n0], p["biases"][:n0], p.bits, p.group_size,
                                  getattr(p, "mode", "affine"), mx.float16)

    xt = x2.astype(mx.float16).T
    ht = unpack._swc(head(g) @ xt, head(u) @ xt)
    w, s, b = d["weight"], d["scales"], d["biases"]
    N, K = int(w.shape[0]), int(w.shape[1]) * 8
    wd = columns_kernel("bfloat", K, n0, _rg(d))(inputs=[w, s, b], grid=(N * (n0 // 64) * 4, 1, 1), threadgroup=(256, 1, 1),
                                                 output_shapes=[(N, n0)], output_dtypes=[mx.bfloat16])[0]
    return wd @ ht


class _KeepAwake:
    """Thread that keeps the Neural Engine awake between two shared layers (PLI_ANE_KEEPAWAKE)."""

    def __init__(self, client: Any, cache: str, blocks: list[int]):
        import threading

        f = client.Shape(2560, 1024, 1)
        self.m = client.Memory(f, 128)
        mil = client.mil_program(f, self.m, [128])
        sg = client.signs()
        surf = {name: s for name, s, _, _ in self.m.inputs(f)}
        self.progs = []
        for b in blocks:
            client.next_block(b)
            try:
                pr = client.Program(mil, client.blob_rotation(f.ane, sg), cache)
            finally:
                client.next_block(0)
            p = pr.procedure("ffn128")
            self.progs.append((pr, pr.bind(p, [surf[n] for n in pr.inputs(p)], self.m.output)))
        self.ev = threading.Event()
        self.lock = threading.Lock()
        self.deadline = 0.0
        self.calls = 0
        threading.Thread(target=self._loop, name="pli-ane-keepawake", daemon=True).start()

    def _loop(self) -> None:
        k = 0
        while True:
            self.ev.wait()
            if time.monotonic() > self.deadline:
                self.ev.clear()
                continue
            pr, li = self.progs[k % len(self.progs)]
            k += 1
            with self.lock:
                if not self.ev.is_set():
                    continue
                try:
                    pr.evaluate(li)
                    self.calls += 1
                except Exception:  # noqa: BLE001 - keep-awake never makes a prefill fail
                    self.ev.clear()
                    time.sleep(0.05)

    def pause(self) -> None:
        """Before an FFN call: no more keep-awake calls, the current one finished."""
        self.ev.clear()
        with self.lock:
            pass

    def resume(self) -> None:
        self.deadline = time.monotonic() + KEEPAWAKE_S
        self.ev.set()


class _State:
    """Programs, surfaces and bindings of the split. PLI_ANE_BLOCKS = 1: one program, block chosen by the service (3.3.0).
    PLI_ANE_BLOCKS = 2: the units are cut into two equal parts, each a program pinned to its block
    (PLI_ANE_BLOCK_IDS, default 1,3; measured: 18 units, T = 0.76 of 3.3.0), evaluated together on two threads.
    The parts read the same input surfaces; their outputs are added to the GPU part in fp32."""

    def __init__(self, client: Any, H: int, N: int, units: int | None = None, keepawake: Any = None):
        import mlx.core as mx
        import numpy as np

        self.c = client
        self.nparts = BLOCKS
        u = UNITS if units is None else units
        self.units = u
        if u % BLOCKS:
            raise ValueError(f"{u} units: not divisible by PLI_ANE_BLOCKS={BLOCKS}")
        self.f = client.Shape(H, N, u // BLOCKS)     # shape of one part (the channel offset does not enter it)
        self.H, self.N, self.ane = H, N, self.f.ane       # ane: channels of one part
        self.n0 = N - BLOCKS * self.ane                   # channels [0, n0) on the GPU
        self.starts = [self.n0 + b * self.ane for b in range(BLOCKS)]
        sg = client.signs()
        self.sg128 = mx.array(sg[:128])
        self.sg512 = mx.array(sg[:512])
        self.r128t = mx.array(np.ascontiguousarray(client.rotation(128, sg).T))
        # sets[s][b]: surface set s (even or odd layer), part b; the parts of a set share x and tx
        self.sets = []
        for _ in range(2):
            parts = [client.Memory(self.f, MAX_ROWS) for _ in range(BLOCKS)]
            for m in parts[1:]:
                m.x, m.tx = parts[0].x, parts[0].tx
            self.sets.append(parts)
        self.tag = [-1, -1]
        self.functions = list(range(-(-MIN_ROWS // STEP) * STEP, MAX_ROWS + 1, STEP))
        mil = client.mil_program(self.f, self.sets[0][0], self.functions)
        cache = subprocess.run(["getconf", "DARWIN_USER_CACHE_DIR"], capture_output=True, text=True).stdout.strip() + "pli-ane"
        os.makedirs(cache, exist_ok=True)
        t0 = time.perf_counter()
        self.progs = []
        for b in range(BLOCKS):
            client.next_block(BLOCK_IDS[b] if BLOCKS > 1 else 0)
            try:
                self.progs.append(client.Program(mil, client.blob_rotation(self.ane, sg), cache))
            finally:
                client.next_block(0)
        self.load_s = time.perf_counter() - t0
        self.bindings: list[list[dict]] = []
        for parts in self.sets:
            per_part = []
            for b, j in enumerate(parts):
                prog = self.progs[b]
                surf = {name: s for name, s, _, _ in j.inputs(self.f)}
                d = {}
                for r in self.functions:
                    p = prog.procedure(f"ffn{r}")
                    d[r] = prog.bind(p, [surf[n] for n in prog.inputs(p)], j.output)
                per_part.append(d)
            self.bindings.append(per_part)
        self.pool = ThreadPoolExecutor(max_workers=BLOCKS, thread_name_prefix="pli-ane")
        self.active = True
        self.sg_np = np.ascontiguousarray(sg, dtype=np.float32)
        self.rq_fut: list = [None, None]
        self.rq_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="pli-rq") if REQUANT_CPU else None
        self.keepawake = keepawake
        if KEEPAWAKE and keepawake is None:
            try:
                self.keepawake = _KeepAwake(client, cache, BLOCK_IDS[:BLOCKS] if BLOCKS > 1 else [0])
            except Exception as e:  # without keep-awake: as before
                _say(f"keep-awake impossible ({type(e).__name__}: {e}): without keep-awake")

    def nbytes(self) -> int:
        """Bytes of the surfaces (x and tx counted once per set)."""
        total = 0
        for parts in self.sets:
            total += parts[0].nbytes()
            for m in parts[1:]:
                total += m.nbytes() - sum(x.size for x in m.x) - m.tx.size
        return total

    def requant(self, m: Any) -> list:
        return [requant(m, d, self.ane) for d in self.starts]

    def write_weights(self, s: int, weights: list) -> None:
        import numpy as np

        S = self.c.SEGMENT
        for b, pb in enumerate(weights):
            qg, eg, qu, eu, qd, ed = (np.asarray(a) for a in pb)
            j = self.sets[s][b]
            for k in range(self.f.seg_input):
                j.wg[k].write(qg[:, k * S:(k + 1) * S])
                j.wu[k].write(qu[:, k * S:(k + 1) * S])
            j.sg.write(eg[:, None])
            j.su.write(eu[:, None])
            start = 0
            for i, w in enumerate(self.f.seg_down):
                j.wd[i].write(qd[:, start:start + w])
                start += w
            j.sd.write(ed[:, None])

    def layer(self, m: Any, x: Any, rows: int) -> Any:
        if REQUANT_CPU:
            return self.layer_cpu(m, x, rows)
        import mlx.core as mx
        import numpy as np

        i, H, n0, S = m._pli_ane_i, self.H, self.n0, self.c.SEGMENT
        s = i % 2
        fn = -(-rows // STEP) * STEP
        x2 = x.reshape(rows, H)
        qT, tx = inputs(x2)
        to_eval = [qT, tx]
        weights = None
        if self.tag[s] != i:
            weights = self.requant(m)
            to_eval += [a for pb in weights for a in pb]
        mx.eval(to_eval)
        parts = self.sets[s]
        if weights is not None:
            self.tag[s] = -1
            self.write_weights(s, weights)
            self.tag[s] = i
            count["requant"] += 1
        qn = np.asarray(qT)
        j0 = parts[0]
        for k in range(self.f.seg_input):
            j0.x[k].array()[:, :rows] = qn[k * S:(k + 1) * S]
        j0.tx.array()[0, :rows] = np.asarray(tx)
        futs = [self.pool.submit(self.progs[b].evaluate, self.bindings[s][b][fn]) for b in range(self.nparts)]
        # weights of the next layer first on the GPU, then the GPU part of this layer
        nxt = m._pli_ane_next
        s2 = nxt._pli_ane_i % 2
        p2 = self.requant(nxt) if (s2 != s and self.tag[s2] != nxt._pli_ane_i) else None
        if p2 is not None:
            mx.async_eval([a for pb in p2 for a in pb])
        yg = gpu_part(m, x2, n0)
        mx.async_eval(yg)
        if p2 is not None:
            mx.eval([a for pb in p2 for a in pb])
            self.tag[s2] = -1
            self.write_weights(s2, p2)
            self.tag[s2] = nxt._pli_ane_i
            count["requant"] += 1
        t0 = time.perf_counter()
        count["ane_ms"] += max(f.result(timeout=10.0) for f in futs)
        count["wait_ms"] += (time.perf_counter() - t0) * 1e3
        f32 = mx.float32
        txf = tx.astype(f32)
        yT = yg.astype(f32)
        for j in parts:   # in the order of the parts: fixed sum order
            ys = mx.array(np.ascontiguousarray(j.output.array()[:H + 1, :rows]))
            yT = yT + ys[:H].astype(f32) * (ys[H].astype(f32) * txf)[None, :]
        count["ane"] += 1
        return mx.contiguous(yT.astype(mx.bfloat16).T).reshape(*x.shape[:-1], H)

    def _rq_next(self, m: Any, s: int) -> None:
        """pli-rq thread: weights of layer m into set s, on the CPU."""
        from tensorfold.pli.ane import requant_cpu as rq

        t0 = time.perf_counter()
        rq.write(self, m, s)
        self.tag[s] = m._pli_ane_i
        count["rq_cpu"] = count.get("rq_cpu", 0) + 1
        count["rq_cpu_ms"] = count.get("rq_cpu_ms", 0.0) + (time.perf_counter() - t0) * 1e3
        from tensorfold.pli import trace_layers as tc

        if tc.measuring():
            tc.add("rq_cpu", (time.perf_counter() - t0) * 1e3)

    def layer_cpu(self, m: Any, x: Any, rows: int) -> Any:
        """layer() with PLI_ANE_REQUANT_CPU=1: same steps, but the Neural Engine weights of the next layer are
        requantized on the CPU (pli-rq thread) and written into the other surface set while this one is computed."""
        import mlx.core as mx
        import numpy as np

        from tensorfold.pli import trace_layers as tc
        from tensorfold.pli.ane import requant_cpu as rq

        measure = tc.in_prefill()   # PLI_TRACE_LAYERS (evolution mode): measurement points, same operations, same order
        t_0 = time.perf_counter()
        i, H, n0, S = m._pli_ane_i, self.H, self.n0, self.c.SEGMENT
        s = i % 2
        fn = -(-rows // STEP) * STEP
        x2 = x.reshape(rows, H)
        qT, tx = inputs(x2)
        mx.async_eval([qT, tx])
        fut = self.rq_fut[s]
        if fut is not None:
            t0 = time.perf_counter()
            self.rq_fut[s] = None
            fut.result(timeout=30.0)
            count["rq_wait_ms"] = count.get("rq_wait_ms", 0.0) + (time.perf_counter() - t0) * 1e3
            if measure:
                tc.add("ffn_rq_wait", (time.perf_counter() - t0) * 1e3)
        if self.tag[s] != i:                       # first layer of a chunk, or a layer out of the planned order
            t0 = time.perf_counter()
            self.tag[s] = -1
            rq.write(self, m, s)
            self.tag[s] = i
            count["rq_direct"] = count.get("rq_direct", 0) + 1
            count["rq_direct_ms"] = count.get("rq_direct_ms", 0.0) + (time.perf_counter() - t0) * 1e3
            if measure:
                tc.add("ffn_rq_direct", (time.perf_counter() - t0) * 1e3)
        mx.eval([qT, tx])
        t_1 = time.perf_counter()
        parts = self.sets[s]
        qn = np.asarray(qT)
        j0 = parts[0]
        for k in range(self.f.seg_input):
            j0.x[k].array()[:, :rows] = qn[k * S:(k + 1) * S]
        j0.tx.array()[0, :rows] = np.asarray(tx)
        if self.keepawake is not None:
            self.keepawake.pause()
        futs = [self.pool.submit(self.progs[b].evaluate, self.bindings[s][b][fn]) for b in range(self.nparts)]
        t_2 = time.perf_counter()
        nxt = m._pli_ane_next
        s2 = nxt._pli_ane_i % 2

        def start_rq() -> None:
            if s2 != s and self.tag[s2] != nxt._pli_ane_i and self.rq_fut[s2] is None:
                rq.matrices(nxt)                       # views of the weights taken on this thread (MLX)
                self.tag[s2] = -1
                self.rq_fut[s2] = self.rq_pool.submit(self._rq_next, nxt, s2)

        if not RQ_AFTER:
            start_rq()
        yg = gpu_part(m, x2, n0)
        mx.async_eval(yg)
        t_3 = time.perf_counter()
        if measure:                                # end of the GPU part (the Neural Engine computes alongside)
            mx.eval(yg)
        t_4 = time.perf_counter()
        t0 = time.perf_counter()
        ane = max(f.result(timeout=10.0) for f in futs)
        if RQ_AFTER:
            start_rq()
        if self.keepawake is not None:
            self.keepawake.resume()
        count["ane_ms"] += ane
        count["wait_ms"] += (time.perf_counter() - t0) * 1e3
        t_5 = time.perf_counter()
        f32 = mx.float32
        txf = tx.astype(f32)
        yT = yg.astype(f32)
        for j in parts:   # in the order of the parts: fixed sum order
            ys = mx.array(np.ascontiguousarray(j.output.array()[:H + 1, :rows]))
            yT = yT + ys[:H].astype(f32) * (ys[H].astype(f32) * txf)[None, :]
        count["ane"] += 1
        y = mx.contiguous(yT.astype(mx.bfloat16).T).reshape(*x.shape[:-1], H)
        if measure:
            mx.eval(y)
            t_6 = time.perf_counter()
            for key, ms in (("inputs", t_1 - t_0), ("copy", t_2 - t_1), ("launch", t_3 - t_2), ("gpu", t_4 - t_3),
                            ("wait", t_5 - t_4), ("outputs", t_6 - t_5)):
                tc.add("ffn_ane_" + key, ms * 1e3)
            tc.add("ffn_ane_ne", float(ane))
        return y

    def trip(self, e: BaseException) -> None:
        self.active = False
        count["tripped"] = f"{type(e).__name__}: {e}"
        _say(f"CIRCUIT BREAKER: {count['tripped']}; GPU only for the rest of the process")


def _mlp_ane(self: Any, x: Any) -> Any:
    import mlx.core as mx

    base = type(self)._pli_ane_base
    rows = x.size // int(x.shape[-1]) if x.ndim >= 2 else 0
    E = _state
    for lo, e in _states:           # the highest step the chunk reaches
        if rows >= lo:
            E = e
    if E is None or not E.active or rows < MIN_ROWS or rows > MAX_ROWS or x.dtype != mx.bfloat16:
        if E is not None and rows >= MIN_ROWS:
            count["gpu_only"] += 1
        return base.__call__(self, x)
    try:
        return E.layer(self, x, rows)
    except Exception as e:  # circuit breaker: layer redone on the GPU alone
        E.trip(e)
        return base.__call__(self, x)


def heartbeat() -> bool:
    """PLI_HEARTBEAT_ANE (idea from Kapitsa #243): a keep-awake call between two requests, so that the Neural Engine
    does not fall asleep (measured: 87 ms at the first call after 10 s of rest, 9.5 with one call every 2 s).
    Without keep-awake installed: nothing. Never makes the engine fail."""
    g = _state.keepawake if _state is not None else None
    if g is None or not g.progs:
        return False
    pr, li = g.progs[g.calls % len(g.progs)]
    with g.lock:
        try:
            pr.evaluate(li)
            g.calls += 1
            return True
        except Exception:  # noqa: BLE001
            return False


def install(layers: Any) -> int:
    """Put the split on the transposed MLPs (unpack._mlp_t) of the layers; return the number of layers taken (0 = GPU only)."""

    global _state
    if not ACTIVE or layers is None or _state is not None:
        return 0
    mlps = [getattr(c, "mlp", None) for c in layers]
    mlps = [m for m in mlps if m is not None and getattr(type(m), "_pli_mlp_t", False)]
    if not mlps:
        _say("no transposed MLP (PLI_PREFILL_UNPACK=f16 and PLI_PREFILL_UNPACK_MLP=t required): GPU only")
        return 0
    try:
        from tensorfold.pli.ane import client

        reason = client.unavailable()
        if reason:
            _say(f"Neural Engine unavailable ({reason}): GPU only")
            return 0
        w = mlps[0].gate_proj["weight"]
        N, H = int(w.shape[0]), int(w.shape[1]) * 8
        if any(int(m.gate_proj["weight"].shape[0]) != N or int(m.down_proj["weight"].shape[0]) != H for m in mlps):
            _say("MLP shapes differ from one layer to the next: GPU only")
            return 0
        if STEPS:
            g = None
            for lo, u in STEPS:
                e_ = _State(client, H, N, units=u, keepawake=g)
                g = g or e_.keepawake
                _states.append((lo, e_))
            _state = _states[-1][1]
        else:
            _state = _State(client, H, N)
    except Exception as e:
        _say(f"setup impossible ({type(e).__name__}: {e}): GPU only")
        _state = None
        return 0
    for i, m in enumerate(mlps):
        base = type(m)
        if base not in _classes:
            _classes[base] = type(base.__name__ + "A", (base,), {"__call__": _mlp_ane, "_pli_ane_base": base})
        object.__setattr__(m, "_pli_ane_i", i)
        m.__class__ = _classes[base]
    for i, m in enumerate(mlps):
        object.__setattr__(m, "_pli_ane_next", mlps[(i + 1) % len(mlps)])
    count["layers"] = len(mlps)
    if _state.keepawake is not None:
        _say(f"keep-awake: {len(_state.keepawake.progs)} blocks, goes quiet {KEEPAWAKE_S * 1e3:.0f} ms after the last shared layer")
    if _states:
        _say("steps: " + ", ".join(f"{e_.units} units from {lo} rows ({e_.nbytes() / 2 ** 20:.0f} MiB, {e_.load_s:.1f} s)"
                                   for lo, e_ in _states) + ("; requantization after the call" if RQ_AFTER else ""))
    E = _state
    blocks = f", {E.nparts} blocks ({','.join(map(str, BLOCK_IDS[:E.nparts]))})" if E.nparts > 1 else ""
    _say(f"{len(mlps)} MLPs shared, {UNITS} units ({E.nparts * E.ane} channels of {E.N}, share {E.nparts * E.ane / E.N:.3f}{blocks}), "
         f"chunks of {MIN_ROWS} to {MAX_ROWS} rows, {len(E.functions)} functions loaded in {E.load_s:.1f} s, "
         f"surfaces {E.nbytes() / 2 ** 20:.0f} MiB")
    return len(mlps)
