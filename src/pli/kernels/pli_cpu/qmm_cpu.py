"""PLI: loading and calling the CPU kernel qmm_cpu.c (product y = x . W^T on MLX's 4-bit weights, without copy).

Nothing is wired into the engine: this module serves the benches. The library is compiled next to the C file at the first call.
"""
import ctypes, os, subprocess
import numpy as np

_HERE = os.path.dirname(os.path.abspath(__file__))
_LIB = None


def load(folder=None):
    """Compile qmm_cpu.c if needed and return the library."""
    global _LIB
    if _LIB is not None:
        return _LIB
    d = folder or _HERE
    import hashlib
    src = os.path.join(d, "qmm_cpu.c")
    # the library's name carries the source's fingerprint: a changed source is always recompiled (dates are not enough,
    # a copy keeps the original date, which once stopped a bench)
    lib = os.path.join(d, "libpliqmm-%s.dylib" % hashlib.sha256(open(src, "rb").read()).hexdigest()[:12])
    if not os.path.exists(lib):
        subprocess.run(["clang", "-O3", "-mcpu=apple-m1", "-shared", "-fblocks", "-ffp-contract=off", "-Wno-deprecated-declarations", src, "-framework", "Accelerate", "-o", lib],
                       check=True)
    L = ctypes.CDLL(lib)
    p, n = ctypes.c_void_p, ctypes.c_long
    for name in ("pli_qmm_f32", "pli_qmm_f16"):
        getattr(L, name).argtypes = [p, p, p, ctypes.c_int, n, n, p, n, p, n, n, ctypes.c_int]
        getattr(L, name).restype = ctypes.c_int
    L.pli_qmm_blas.argtypes = [p, p, p, ctypes.c_int, n, n, p, n, p, n, n, n]
    L.pli_qmm_blas.restype = ctypes.c_int
    L.pli_qmm_priorite.argtypes = [ctypes.c_int]
    L.pli_qmm_ent.argtypes = [p, p, p, ctypes.c_int, n, n, p, p, p, n, p, n, n, ctypes.c_int]
    L.pli_qmm_ent.restype = ctypes.c_int
    L.pli_qmm_priorite.restype = None
    L.pli_qmm_e15.argtypes = [p, p, p, ctypes.c_int, n, n, p, n, p, n, n, ctypes.c_int, ctypes.c_int]
    L.pli_qmm_e15.restype = ctypes.c_int
    L.pli_swiglu.argtypes = [p, p, n]
    L.pli_swiglu.restype = ctypes.c_int
    _LIB = L
    return L


def byte_view(arr):
    """numpy byte view on an array's buffer (MLX or numpy), without copy; None if the buffer cannot be read."""
    if isinstance(arr, np.ndarray):
        return np.ascontiguousarray(arr).view(np.uint8).reshape(-1)
    try:
        return np.frombuffer(memoryview(arr), dtype=np.uint8)
    except Exception:
        try:
            return np.asarray(memoryview(arr).cast("B"))
        except Exception:
            return None


class Weights:
    """An MLX 4-bit matrix: weight (uint32 [N, K/8]), scales and biases ([N, K/64], BF16, float16 or float32)."""

    def __init__(self, w, s, b, sb=0):
        self.N, self.K = int(w.shape[0]), int(w.shape[1]) * 8
        self.sb = sb
        self._keep = [w, s, b]
        self.vw, self.vs, self.vb = byte_view(w), byte_view(s), byte_view(b)
        if self.vw is None or self.vs is None or self.vb is None:
            raise ValueError("buffer unreadable without copy")
        nbytes = {0: 2, 1: 2, 2: 4}[sb]
        if self.vw.size != self.N * self.K // 2 or self.vs.size != self.N * (self.K // 64) * nbytes:
            raise ValueError("unexpected sizes: %d %d" % (self.vw.size, self.vs.size))

    def addresses(self):
        return self.vw.ctypes.data, self.vs.ctypes.data, self.vb.ctypes.data


def product(variant, weights, x, n0=0, n1=None, threads=0, block=4096):
    """y [L, n1 - n0] in float32. variant: "f32", "f16" (x in float16) or "blas" (x in float32)."""
    L = load()
    n1 = weights.N if n1 is None else n1
    x = np.ascontiguousarray(x, dtype=np.float16 if variant == "f16" else np.float32)
    rows = x.shape[0]
    y = np.empty((rows, n1 - n0), dtype=np.float32)
    aw, a_s, ab = weights.addresses()
    if variant == "blas":
        r = L.pli_qmm_blas(aw, a_s, ab, weights.sb, weights.N, weights.K, x.ctypes.data, rows, y.ctypes.data, n0, n1, block)
    else:
        f = L.pli_qmm_f16 if variant == "f16" else L.pli_qmm_f32
        r = f(aw, a_s, ab, weights.sb, weights.N, weights.K, x.ctypes.data, rows, y.ctypes.data, n0, n1, threads)
    if r:
        raise RuntimeError("CPU kernel: code %d" % r)
    return y


def priority(level):
    """Priority of the kernel's threads: 0 = high (default), 1 = medium, 2 = background."""
    load().pli_qmm_priorite(int(level))


def swiglu(g, u):
    """In place in g (float32): silu(g) * u."""
    load().pli_swiglu(g.ctypes.data, u.ctypes.data, g.size)
    return g


def to_ints(x, bits=16):
    """Activations x [L, K] as integers, one scale per row: (xh, xl, scale). 16 bits: value = 256 * xh + xl, centred halves
    (-128..127). 8 bits: value = xh, xl = None."""
    x = np.asarray(x, dtype=np.float64); lim = 127 if bits == 8 else 32639   # 127 * 256 + 127: the high half fits in -128..127
    scale = np.abs(x).max(axis=1, keepdims=True) / lim; scale[scale == 0] = 1.0; v = np.round(x / scale).astype(np.int64)
    if bits == 8:
        return np.ascontiguousarray(v, dtype=np.int8), None, np.ascontiguousarray(scale[:, 0], dtype=np.float32)
    xl = ((v + 128) & 255) - 128; xh = (v - xl) >> 8
    return np.ascontiguousarray(xh, dtype=np.int8), np.ascontiguousarray(xl, dtype=np.int8), np.ascontiguousarray(scale[:, 0], dtype=np.float32)


def int_product(weights, xh, xl, scale, n0=0, n1=None, threads=0):
    """"Integers first" product: y [L, n1 - n0] in float32 (see pli_qmm_ent in qmm_cpu.c for the exact order of the computation)."""
    L = load(); n1 = weights.N if n1 is None else n1; rows = xh.shape[0]
    y = np.empty((rows, n1 - n0), dtype=np.float32); aw, a_s, ab = weights.addresses()
    r = L.pli_qmm_ent(aw, a_s, ab, weights.sb, weights.N, weights.K, xh.ctypes.data, None if xl is None else xl.ctypes.data, scale.ctypes.data,
                      rows, y.ctypes.data, n0, n1, threads)
    if r:
        raise RuntimeError("CPU integer kernel: code %d" % r)
    return y


def int_product_numpy(q, s, b, xh, xl, scale):
    """The same computation in numpy, same order, to check the bits: q [N, K] integers 0..15, s and b [N, K/64] float32."""
    N, K = q.shape; G = K // 64; Lg = xh.shape[0]; acc = np.zeros((Lg, N), dtype=np.float32); f = np.float32
    for g in range(G):
        qg = q[:, 64 * g:64 * g + 64].astype(np.int64); h = xh[:, 64 * g:64 * g + 64].astype(np.int64)
        ph = (h @ qg.T).astype(f); sxh = h.sum(1).astype(f)[:, None]
        if xl is None:
            pl = np.zeros_like(ph); sxl = np.zeros_like(sxh)
        else:
            lo = xl[:, 64 * g:64 * g + 64].astype(np.int64); pl = (lo @ qg.T).astype(f); sxl = lo.sum(1).astype(f)[:, None]
        sg = s[:, g].astype(f)[None, :]; bg = b[:, g].astype(f)[None, :]; kh = f(1 if xl is None else 256)
        u = (sg * kh) * ph + sg * pl; v = (bg * kh) * sxh + bg * sxl
        acc = acc + (u + v)
    return acc * scale.astype(f)[:, None]


INT_INPUT_MAX = 16383   # like simd_qmm.INT_INPUT_MAX: the sum of a group of 64 products stays exact in 32-bit float


def chunk_count(n, k):
    """Number of chunks of TensorFold's kernel (simd_qmm.splits): sets the order of the additions."""
    return 32 if n <= 64 else (16 if n <= 6144 else 8)


def to_int15(x):
    """x [L, K] float -> (xi int16, exponents int32 [L]): xi = round(x * 2^-e), e the smallest exponent such that |xi| <= INT_INPUT_MAX.
    Power-of-2 scale: the multiplication is exact, only the rounding (to nearest, even on ties) matters."""
    x = np.asarray(x, dtype=np.float32); m = np.abs(x).max(axis=1)
    e = np.where(m > 0, np.ceil(np.log2(np.maximum(m, 1e-38) / INT_INPUT_MAX)), 0).astype(np.int32)
    xi = np.rint(np.ldexp(x, -e[:, None]))
    while True:                                   # the rounding may exceed by one unit: the exponent of these rows goes up
        over = np.abs(xi).max(axis=1) > INT_INPUT_MAX
        if not over.any():
            break
        e[over] += 1; xi[over] = np.rint(np.ldexp(x[over], -e[over][:, None]))
    return np.ascontiguousarray(xi, dtype=np.int16), e


def product_e15(weights, xi, n0=0, n1=None, threads=0):
    """Integers-first product, order of TensorFold's GPU kernel: y [L, n1 - n0] in float32, without the scale of x."""
    L = load(); n1 = weights.N if n1 is None else n1; xi = np.ascontiguousarray(xi, dtype=np.int16); rows = xi.shape[0]
    y = np.empty((rows, n1 - n0), dtype=np.float32); aw, a_s, ab = weights.addresses()
    r = L.pli_qmm_e15(aw, a_s, ab, weights.sb, weights.N, weights.K, xi.ctypes.data, rows, y.ctypes.data, n0, n1, chunk_count(weights.N, weights.K), threads)
    if r:
        raise RuntimeError("CPU e15 kernel: code %d" % r)
    return y
