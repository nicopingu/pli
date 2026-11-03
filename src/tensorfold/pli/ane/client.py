"""tensorfold.pli.ane.client: Neural Engine client for PLI (15/10/2026).

ctypes wrapper of libpli_ane.dylib (private client taken from Splash 1.3.2-m1, Apache-2.0) and generator of the MIL program
of the FFN split by channels, taken from Splash runtime/ops/AneFfn.cpp (AneFfn::function), rewritten in Python.

The Neural Engine computes the tail intermediate channels of a SwiGLU FFN layer in W8A8:
  inputs x (int8, laid out per channel, Hadamard rotation in blocks of 128) and their per-token scales tx;
  weights wg, wu (int8 per row, same rotations) and scales sg, su; wd (int8, rotation of the intermediate inputs in
  blocks of 512) and sd. Output: ds (down rows before the token scale) then one hscale row.
  The result is y[t][c] = ds[c][t] * hscale[t] * tx[t] (join in fp32).
"""

from __future__ import annotations

import ctypes
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIBRARY = os.path.join(HERE, "native", "libpli_ane.dylib")
if not os.path.exists(LIBRARY):
    # first use on this machine: compiled by build.sh (clang++ of the Command Line Tools)
    import subprocess

    done = subprocess.run(["bash", os.path.join(HERE, "build.sh")], capture_output=True, text=True)
    if done.returncode != 0 or not os.path.exists(LIBRARY):
        raise OSError("libpli_ane.dylib missing and not compiled (xcode-select --install, then "
                      f"bash {os.path.join(HERE, 'build.sh')}): {(done.stderr or done.stdout)[-300:]}")
_lib = ctypes.CDLL(LIBRARY)

V, P, U32, U64, D, S, I = (ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint32, ctypes.c_uint64,
                           ctypes.c_double, ctypes.c_char_p, ctypes.c_int)
for name, res, args in [
    ("pli_ane_error", S, []),
    ("pli_ane_unavailable", I, [ctypes.c_char_p, ctypes.c_size_t]),
    ("pli_ane_surface", V, [U32, U32, I]),
    ("pli_ane_surface_free", None, [V]),
    ("pli_ane_surface_base", V, [V]),
    ("pli_ane_surface_stride", U32, [V]),
    ("pli_ane_surface_size", U64, [V]),
    ("pli_ane_surface_type", I, [V, U32, U32, ctypes.c_char_p, ctypes.c_size_t]),
    ("pli_ane_surface_strides", I, [V, U32, ctypes.c_char_p, ctypes.c_size_t]),
    ("pli_ane_blob", ctypes.c_size_t, [V, ctypes.c_size_t, V, ctypes.c_size_t]),
    ("pli_ane_program", V, [S, V, ctypes.c_size_t, S, D]),
    ("pli_ane_program_free", None, [V]),
    ("pli_ane_procedure", I, [V, S]),
    ("pli_ane_inputs", I, [V, I, ctypes.c_char_p, ctypes.c_size_t]),
    ("pli_ane_bind", V, [V, I, P, I, V]),
    ("pli_ane_bind_free", None, [V]),
    ("pli_ane_eval", D, [V, V, D]),
    ("pli_ane_next_block", None, [I]),
]:
    f = getattr(_lib, name)
    f.restype, f.argtypes = res, args


def next_block(block: int) -> None:
    """PLI (16/10/2026): Neural Engine block (1 to 3; 0: the service chooses) imposed on the programs built afterwards."""
    _lib.pli_ane_next_block(int(block))


class AneError(RuntimeError):
    pass


def _error(what: str) -> AneError:
    return AneError(f"{what}: {_lib.pli_ane_error().decode(errors='replace')}")


def _text(f, *args) -> str:
    n = f(*args, None, 0)
    buf = ctypes.create_string_buffer(max(n, 0) + 1)
    f(*args, buf, len(buf))
    return buf.value.decode()


def unavailable() -> str:
    """Empty string if the Neural Engine is reachable, else the reason."""
    return _text(_lib.pli_ane_unavailable)


class Surface:
    """IOSurface of `rows` rows of `width` elements (int8 or fp16), each row aligned on 64 bytes."""

    def __init__(self, rows: int, width: int, int8: bool):
        self.h = _lib.pli_ane_surface(rows, width, int(int8))
        if not self.h:
            raise _error("surface")
        self.rows, self.width, self.int8 = rows, width, int8
        self.stride = _lib.pli_ane_surface_stride(self.h)
        self.size = _lib.pli_ane_surface_size(self.h)
        self.base = _lib.pli_ane_surface_base(self.h)

    def array(self) -> np.ndarray:
        """numpy view [rows, stride / size] without copy (unified memory)."""
        raw = np.ctypeslib.as_array((ctypes.c_uint8 * self.size).from_address(self.base))
        view = raw[: self.rows * self.stride].reshape(self.rows, self.stride)
        return view.view(np.int8) if self.int8 else view.view(np.float16)

    def write(self, values: np.ndarray) -> None:
        t = self.array()
        t[: values.shape[0], : values.shape[1]] = values

    def type(self, rows: int, width: int) -> str:
        return _text(_lib.pli_ane_surface_type, self.h, rows, width)

    def strides(self, rows: int) -> str:
        return _text(_lib.pli_ane_surface_strides, self.h, rows)

    def mil(self) -> str:
        return "int8" if self.int8 else "fp16"

    def __del__(self):
        if getattr(self, "h", None):
            _lib.pli_ane_surface_free(self.h)
            self.h = None


def blob(values_f16: np.ndarray) -> bytes:
    v = np.ascontiguousarray(values_f16, dtype=np.float16)
    n = _lib.pli_ane_blob(v.ctypes.data, v.size, None, 0)
    out = (ctypes.c_uint8 * n)()
    _lib.pli_ane_blob(v.ctypes.data, v.size, ctypes.addressof(out), n)
    return bytes(out)


class Program:
    def __init__(self, mil: str, weights: bytes, cache: str, timeout_s: float = 300.0):
        self._weights = (ctypes.c_uint8 * len(weights)).from_buffer_copy(weights)
        self.h = _lib.pli_ane_program(mil.encode(), ctypes.addressof(self._weights), len(weights), cache.encode(), timeout_s)
        if not self.h:
            raise _error("program")

    def procedure(self, function: str) -> int:
        p = _lib.pli_ane_procedure(self.h, function.encode())
        if p < 0:
            raise _error("procedure " + function)
        return p

    def inputs(self, procedure: int) -> list[str]:
        return _text(_lib.pli_ane_inputs, self.h, procedure).split("\n")

    def bind(self, procedure: int, inputs: list[Surface], output: Surface):
        tab = (ctypes.c_void_p * len(inputs))(*[s.h for s in inputs])
        b = _lib.pli_ane_bind(self.h, procedure, tab, len(inputs), output.h)
        if not b:
            raise _error("binding")
        return b

    def evaluate(self, binding, timeout_s: float = 10.0) -> float:
        ms = _lib.pli_ane_eval(self.h, binding, timeout_s)
        if ms < 0:
            raise _error("evaluation")
        return ms


# ---- MIL program of the split FFN (taken from AneFfn::function) ----

SEGMENT = 2560          # kSegment
INPUT_BLOCK = 128       # ANE_FFN_INPUT_BLOCK
INTERMEDIATE_BLOCK = 512  # ANE_FFN_INTERMEDIATE_BLOCK
CHANNEL_UNIT = 512      # kChannelUnit
INT8_UNIT = 128.0       # ANE_FFN_INT8_UNIT
INT8_PEAK = 127.0       # ANE_FFN_INT8_PEAK
INTERMEDIATE_FLOOR = 2.0 ** -9  # kIntermediateFloor
PEAK_FLOOR = 2.0 ** -14   # ANE_FFN_PEAK_FLOOR


def fp16(v: float) -> str:
    """fp16 constant written exactly, like Splash's C %a (0x1p-7, 0x1.fcp+6). No snprintf through ctypes:
    on Apple arm64, ctypes passes a double badly to a variadic function (everything came out 0x0p+0)."""
    h = float(v).hex()                      # '0x1.fc00000000000p+6'
    mant, exp = h.split("p")
    if "." in mant:
        mant = mant.rstrip("0").rstrip(".")
    return f"fp16({mant}p{exp})"


def dims(rows: int, width: int) -> str:
    return f"[1, 1, {rows}, {width}]"


def tensor(t: str, rows: int, width: int) -> str:
    return f"tensor<{t}, {dims(rows, width)}>"


def segments(n: int) -> list[int]:
    return [min(SEGMENT, n - b) for b in range(0, n, SEGMENT)]


class Shape:
    def __init__(self, hidden: int, intermediate: int, units: int):
        if hidden % SEGMENT:
            raise ValueError("hidden must be a multiple of 2560")
        self.hidden, self.intermediate = hidden, intermediate
        self.ane = units * CHANNEL_UNIT
        self.gpu = intermediate - self.ane
        if self.ane <= 0 or self.gpu <= 0:
            raise ValueError("split out of bounds")
        self.seg_input = hidden // SEGMENT
        self.seg_down = segments(self.ane)

    @property
    def part(self) -> float:
        return self.ane / self.intermediate


class Memory:
    """The surfaces of a set (one layer) for chunks of at most `max_rows` rows."""

    def __init__(self, f: Shape, max_rows: int):
        self.x = [Surface(SEGMENT, max_rows, True) for _ in range(f.seg_input)]
        self.tx = Surface(1, max_rows, False)
        self.wg = [Surface(f.ane, SEGMENT, True) for _ in range(f.seg_input)]
        self.wu = [Surface(f.ane, SEGMENT, True) for _ in range(f.seg_input)]
        self.sg = Surface(f.ane, 1, False)
        self.su = Surface(f.ane, 1, False)
        self.wd = [Surface(f.hidden, w, True) for w in f.seg_down]
        self.sd = Surface(f.hidden, 1, False)
        self.output = Surface(f.hidden + 1, max_rows, False)

    def inputs(self, f: Shape):
        """(name, surface, rows, width); width 0 = the chunk's rows. Order of programInputs."""
        e = [(f"x{k}", s, SEGMENT, 0) for k, s in enumerate(self.x)]
        e.append(("tx", self.tx, 1, 0))
        for k in range(f.seg_input):
            e.append((f"wg{k}", self.wg[k], f.ane, SEGMENT))
            e.append((f"wu{k}", self.wu[k], f.ane, SEGMENT))
        e.append(("sg", self.sg, f.ane, 1))
        e.append(("su", self.su, f.ane, 1))
        for i, w in enumerate(f.seg_down):
            e.append((f"wd{i}", self.wd[i], f.hidden, w))
        e.append(("sd", self.sd, f.hidden, 1))
        return e

    def nbytes(self) -> int:
        everything = self.x + [self.tx] + self.wg + self.wu + [self.sg, self.su] + self.wd + [self.sd, self.output]
        return sum(s.size for s in everything)


def function(f: Shape, inputs, output: Surface, rows: int) -> str:
    unit = fp16(1.0 / INT8_UNIT)
    params: list[str] = []
    body: list[str] = []
    line = body.append

    def f16(name, h, w, expr):
        line(f"{tensor('fp16', h, w)} {name} = {expr}")

    for name, s, r, w in inputs:
        width = w if w else rows
        params.append(f"{s.type(r, width)} {name}")
        line(f"{tensor(s.mil(), r, width)} {name}_t = tensor_buffer_to_tensor<ios17>(input = {name})")

    def matmul(name, weights, values, h, w):
        f16(weights + "_d", h, w, f"dequantize(input = {weights}_t, scale = {unit})")
        line(f"{tensor('fp16', h, rows)} {name} = matmul(transpose_x = bool(false), transpose_y = bool(false), "
             f"x = {weights}_d, y = {values})")

    def total_of(prefix, terms, h):
        total = prefix + "0"
        for t in range(1, terms):
            nxt = f"{prefix}_sum{t}"
            f16(nxt, h, rows, f"add(x = {total}, y = {prefix}{t})")
            total = nxt
        return total

    c, ns = f.ane, f.seg_input
    for k in range(ns):
        f16(f"x{k}_d", SEGMENT, rows, f"dequantize(input = x{k}_t, scale = {unit})")
    for p in ("g", "u"):
        for k in range(ns):
            matmul(f"{p}m{k}", f"w{p}{k}", f"x{k}_d", c, SEGMENT)
        f16(f"{p}s", c, rows, f"mul(x = {total_of(p + 'm', ns, c)}, y = s{p}_t)")
    f16("gt", c, rows, "mul(x = gs, y = tx_t)")
    f16("gh", c, rows, f"mul(x = gt, y = {fp16(0.5)})")
    f16("th", c, rows, "tanh(x = gh)")
    f16("tp", c, rows, f"add(x = th, y = {fp16(1.0)})")
    f16("silu", c, rows, "mul(x = gh, y = tp)")
    f16("h", c, rows, "mul(x = silu, y = us)")
    line(f"tensor<fp16, [1, {c}, 1, {rows}]> h4 = reshape(x = h, shape = tensor<int32, [4]>([1, {c}, 1, {rows}]))")
    line(f"tensor<fp16, [{c}, {INTERMEDIATE_BLOCK}, 1, 1]> rotation = const()[name = string(\"rotation\"), "
         f"val = tensor<fp16, [{c}, {INTERMEDIATE_BLOCK}, 1, 1]>(BLOBFILE(path = string(\"@model_path/weights.bin\"), "
         f"offset = uint64(64)))]")
    line(f"tensor<fp16, [1, {c}, 1, {rows}]> hr4 = conv(dilations = tensor<int32, [2]>([1, 1]), "
         f"groups = int32({c // INTERMEDIATE_BLOCK}), pad = tensor<int32, [4]>([0, 0, 0, 0]), pad_type = string(\"valid\"), "
         f"strides = tensor<int32, [2]>([1, 1]), weight = rotation, x = h4)")
    f16("hr", c, rows, f"reshape(x = hr4, shape = tensor<int32, [4]>({dims(c, rows)}))")
    f16("habs", c, rows, "abs(x = hr)")
    f16("peak", 1, rows, "reduce_max(x = habs, axes = tensor<int32, [1]>([2]), keep_dims = bool(true))")
    f16("floor", 1, rows, f"maximum(x = peak, y = {fp16(INTERMEDIATE_FLOOR)})")
    f16("inverse", 1, rows, f"real_div(x = {fp16(INT8_PEAK)}, y = floor)")
    f16("hs", c, rows, "mul(x = hr, y = inverse)")
    line(f"{tensor('int8', c, rows)} hq = quantize(input = hs, scale = fp16(1), output_dtype = string(\"int8\"))")
    f16("hd", c, rows, f"dequantize(input = hq, scale = {unit})")
    f16("hscale", 1, rows, f"mul(x = floor, y = {fp16(INT8_UNIT / INT8_PEAK)})")
    start = 0
    for i, w in enumerate(f.seg_down):
        f16(f"hd{i}", w, rows, f"slice_by_size(x = hd, begin = tensor<int32, [4]>([0, 0, {start}, 0]), "
                              f"size = tensor<int32, [4]>({dims(w, rows)}))")
        matmul(f"dm{i}", f"wd{i}", f"hd{i}", f.hidden, w)
        start += w
    f16("ds", f.hidden, rows, f"mul(x = {total_of('dm', len(f.seg_down), f.hidden)}, y = sd_t)")
    f16("yt", f.hidden + 1, rows, "concat(axis = int32(2), interleave = bool(false), values = (ds, hscale))")
    line(f"{output.type(f.hidden + 1, rows)} y = tensor_to_tensor_buffer<ios17>(input = yt, "
         f"interleave_factors = tensor<uint8, [4]>([1, 1, 1, 1]), strides = tensor<int64, [4]>({output.strides(f.hidden + 1)}))")
    text = "".join(f"        {t};\n" for t in body)
    return f"    func ffn{rows}<ios18>({', '.join(params)}) {{\n{text}    }} -> (y);\n"


def mil_program(f: Shape, m: Memory, rows: list[int]) -> str:
    e = m.inputs(f)
    return "program(1.3)\n{\n" + "".join(function(f, e, m.output, r) for r in rows) + "}\n"


# ---- rotations and quantization (CPU side for the probe; GPU side in the engine) ----

def signs(n: int = INTERMEDIATE_BLOCK, seed: int = 20261015) -> np.ndarray:
    return np.where(np.random.RandomState(seed).randint(0, 2, n) == 1, -1.0, 1.0).astype(np.float32)


def hadamard(n: int) -> np.ndarray:
    i = np.arange(n)
    pc = np.vectorize(lambda v: bin(v).count("1") & 1)(i[:, None] & i[None, :])
    return np.where(pc == 1, -1.0, 1.0).astype(np.float32)


def rotation(n: int, sg: np.ndarray) -> np.ndarray:
    """R = D H / sqrt(n), D = diag(first n signs): output o = sum_i R[o][i] v[i]."""
    return (sg[:n, None] * hadamard(n)) / np.sqrt(n)


def blob_rotation(channels: int, sg: np.ndarray) -> bytes:
    r = rotation(INTERMEDIATE_BLOCK, sg)  # [o][i]
    v = np.tile(r, (channels // INTERMEDIATE_BLOCK, 1)).reshape(channels, INTERMEDIATE_BLOCK, 1, 1)
    return blob(v.astype(np.float16))


def rotate_inputs(m: np.ndarray, block: int, r: np.ndarray) -> np.ndarray:
    """Apply R in blocks of `block` on the last axis: v' = R v (rows of m = vectors)."""
    s = m.shape
    return (m.reshape(-1, s[-1] // block, block) @ r.T).reshape(s)


def int8_rows(w: np.ndarray):
    """int8 codes per row and the row's scale (peak / 127)."""
    peak = np.maximum(np.abs(w).max(axis=1), PEAK_FLOOR)
    e = peak / INT8_PEAK
    q = np.clip(np.rint(w / e[:, None]), -127, 127).astype(np.int8)
    return q, e.astype(np.float32)
