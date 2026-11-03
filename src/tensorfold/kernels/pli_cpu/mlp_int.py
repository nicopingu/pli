"""PLI (item E4): the MLP in integers first, from start to end, with the same bits on the GPU and on the CPU.

Chain, for each row (the rows are independent: the result does not depend on the number of rows of the call):
  1. conversion to integers: xi = round(x * 2^-e), e taken from the bits of the row's largest |x| (xi fits in 15 signed bits);
  2. gate and up: integer products (simd_qmm.qmm_int on the GPU, pli_qmm_e15 on the CPU), output in 32 bits;
  3. activation: gate converted back to integers (gi, exponent k), then a read of a shared table T[k][gi] = silu(gi * 2^k) * 2^-k;
     a = T * up (one multiplication);
  4. a converted back to integers, integer down product, then rescaling by a power of 2 (exact).
Only floating-point operations outside the kernels: maximum, multiplication by a power of 2 (exact), x + 0.5 then floor
(the rounding), one multiplication, and the table read. Each gives the same result on any IEEE 754 compliant device.
Nothing is wired into the engine: this module serves accuracy trials."""
import numpy as np

INT_INPUT_MAX = 16383
K_MIN, K_MAX = -30, 4            # tabulated exponents; below: silu(v) = v / 2; above: silu(v) = max(v, 0)
E_LIM = 100
_table = None


def table():
    """Shared table [K_MAX - K_MIN + 3, 2 * INT_INPUT_MAX + 1] in 32 bits: row 0 = half, rows 1.. = k from K_MIN to K_MAX, last = max(gi, 0)."""
    global _table
    if _table is None:
        gi = np.arange(-INT_INPUT_MAX, INT_INPUT_MAX + 1, dtype=np.float64)
        rows = [gi / 2]
        for k in range(K_MIN, K_MAX + 1):
            v = np.ldexp(gi, k)
            with np.errstate(over="ignore"):
                rows.append(v / (1.0 + np.exp(-v)) / (2.0 ** k))
        rows.append(np.maximum(gi, 0.0))
        _table = np.ascontiguousarray(np.stack(rows), dtype=np.float32)
    return _table


# ---------- CPU (numpy + pli_qmm_e15 kernel) ----------
def to_int_np(x):
    """x [L, K] float32 -> (xi int16, e int32 [L]): xi = floor(x * 2^-e + 0.5), bounded to +- INT_INPUT_MAX."""
    x = np.ascontiguousarray(x, dtype=np.float32)
    m = np.abs(x).max(axis=1)
    e = (((m.view(np.uint32) >> 23) & 0xFF).astype(np.int32) - 127) - 13
    e = np.clip(e, -E_LIM, E_LIM)
    scale = (((127 - e).astype(np.uint32)) << 23).view(np.float32)
    v = np.floor(x * scale[:, None] + np.float32(0.5))
    return np.clip(v, -INT_INPUT_MAX, INT_INPUT_MAX).astype(np.int16), e


def pow2_np(e):
    return ((127 + np.clip(e, -126, 127)).astype(np.uint32) << 23).view(np.float32)


def mlp_np(x, gate_weights, up_weights, down_weights):
    from tensorfold.kernels.pli_cpu import qmm_cpu
    T = table()
    xi, e = to_int_np(x)
    g = qmm_cpu.product_e15(gate_weights, xi)
    u = qmm_cpu.product_e15(up_weights, xi)
    gi, eg = to_int_np(g)
    k = eg + e
    row = np.clip(k - K_MIN + 1, 0, K_MAX - K_MIN + 2)
    a = T[row[:, None], gi.astype(np.int32) + INT_INPUT_MAX] * u
    ai, ea = to_int_np(a)
    y = qmm_cpu.product_e15(down_weights, ai)
    return y * pow2_np(ea + k + e)[:, None]


# ---------- GPU (MLX + TensorFold kernel with integer input) ----------
def to_int_mx(x):
    import mlx.core as mx
    x = x.astype(mx.float32)
    m = mx.max(mx.abs(x), axis=1)
    e = ((m.view(mx.uint32) >> 23) & 0xFF).astype(mx.int32) - 127 - 13
    e = mx.clip(e, -E_LIM, E_LIM)
    scale = ((127 - e).astype(mx.uint32) << 23).view(mx.float32)
    v = mx.floor(x * scale[:, None] + mx.array(0.5, dtype=mx.float32))
    return mx.clip(v, -INT_INPUT_MAX, INT_INPUT_MAX).astype(mx.int16), e


def mlp_mx(x, gate, up, down, T=None):
    """gate, up, down: triples (weight, scales, biases) of MLX's 4-bit format. Returns y [L, outputs] in 32 bits."""
    import mlx.core as mx
    from tensorfold.kernels.qwen.dense.v1 import simd_qmm
    T = mx.array(table()) if T is None else T
    xi, e = to_int_mx(x)
    g = simd_qmm.qmm_int(xi, *gate)
    u = simd_qmm.qmm_int(xi, *up)
    gi, eg = to_int_mx(g)
    k = eg + e
    row = mx.clip(k - K_MIN + 1, 0, K_MAX - K_MIN + 2)
    flat = row[:, None] * (2 * INT_INPUT_MAX + 1) + gi.astype(mx.int32) + INT_INPUT_MAX
    a = mx.take(T.reshape(-1), flat) * u
    ai, ea = to_int_mx(a)
    y = simd_qmm.qmm_int(ai, *down)
    ef = mx.clip(ea + k + e, -126, 127)
    return y * ((127 + ef).astype(mx.uint32) << 23).view(mx.float32)[:, None]


# ---------- GPU, kernel version (measured: MLX's small operations cost 0.4 ms per conversion to integers) ----------
# Two Metal kernels do the same computation as to_int_mx and as the table activation, in one launch each:
#   pli_int_q   : conversion to integers of an array [L, K] (one row per threadgroup);
#   pli_int_act : gate converted back to integers, table, multiplication by up, result converted back to integers.
# Same operations, in the same order, as the numpy version (mlp_np): same bits.
TG = 256
_HEADER = r"""
inline int pli_exponent(float m) { return clamp(int((as_type<uint>(m) >> 23) & 0xFFu) - 140, -100, 100); }
inline float pli_scale(int e) { return as_type<float>(uint(127 - e) << 23); }
inline short pli_round(float v) { return short(clamp(floor(v + 0.5f), -16383.0f, 16383.0f)); }
"""
_SRC_Q = r"""
  constexpr uint TG = 256;
  const uint r = threadgroup_position_in_grid.x;
  const uint t = uint(simdgroup_index_in_threadgroup) * 32u + uint(thread_index_in_simdgroup);
  const uint K = uint(X_shape[1]);
  threadgroup float peaks[TG];
  float m = 0.0f;
  for (uint j = t; j < K; j += TG) m = max(m, fabs(float(X[size_t(r) * K + j])));
  peaks[t] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float mm = 0.0f;
  for (uint i = 0; i < TG; i++) mm = max(mm, peaks[i]);
  const int e = pli_exponent(mm);
  const float sc = pli_scale(e);
  for (uint j = t; j < K; j += TG) XI[size_t(r) * K + j] = pli_round(float(X[size_t(r) * K + j]) * sc);
  if (t == 0) E[r] = e;
"""
_SRC_ACT = r"""
  constexpr uint TG = 256;
  const uint r = threadgroup_position_in_grid.x;
  const uint t = uint(simdgroup_index_in_threadgroup) * 32u + uint(thread_index_in_simdgroup);
  const uint N = uint(G_shape[1]);
  const int kmin = int(P[0]), nrows = int(P[1]), width = int(P[2]);
  threadgroup float peaks[TG];
  float m = 0.0f;
  for (uint j = t; j < N; j += TG) m = max(m, fabs(G[size_t(r) * N + j]));
  peaks[t] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float mm = 0.0f;
  for (uint i = 0; i < TG; i++) mm = max(mm, peaks[i]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const int eg = pli_exponent(mm);
  const float scg = pli_scale(eg);
  const int k = eg + EX[r];
  const size_t base = size_t(clamp(k - kmin + 1, 0, nrows - 1)) * size_t(width) + size_t(width / 2);
  m = 0.0f;
  for (uint j = t; j < N; j += TG) {
    const float a = T[base + int(pli_round(G[size_t(r) * N + j] * scg))] * U[size_t(r) * N + j];
    m = max(m, fabs(a));
  }
  peaks[t] = m;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  mm = 0.0f;
  for (uint i = 0; i < TG; i++) mm = max(mm, peaks[i]);
  const int ea = pli_exponent(mm);
  const float sca = pli_scale(ea);
  for (uint j = t; j < N; j += TG) {
    const float a = T[base + int(pli_round(G[size_t(r) * N + j] * scg))] * U[size_t(r) * N + j];
    AI[size_t(r) * N + j] = pli_round(a * sca);
  }
  if (t == 0) EF[r] = ea + k + EX[r];
"""
_kernels: dict = {}


def _kernel(name):
    import mlx.core as mx
    if name not in _kernels:
        if name == "q":
            _kernels[name] = mx.fast.metal_kernel(name="pli_int_q", input_names=["X"], output_names=["XI", "E"], source=_SRC_Q, header=_HEADER)
        else:
            _kernels[name] = mx.fast.metal_kernel(name="pli_int_act", input_names=["G", "U", "EX", "T", "P"], output_names=["AI", "EF"],
                                                  source=_SRC_ACT, header=_HEADER)
    return _kernels[name]


def to_int_kernel(x):
    """Like to_int_mx, in one launch. x [L, K] in bf16 or in 32 bits."""
    import mlx.core as mx
    L, K = int(x.shape[0]), int(x.shape[1])
    xi, e = _kernel("q")(inputs=[x], grid=(TG * L, 1, 1), threadgroup=(TG, 1, 1), output_shapes=[(L, K), (L,)], output_dtypes=[mx.int16, mx.int32])
    return xi, e


_params = None


def mlp_kernels(x, gate, up, down, T=None):
    """Like mlp_mx, with the two kernels: 5 launches (integers, gate, up, activation, down) and one rescaling."""
    import mlx.core as mx
    from tensorfold.kernels.qwen.dense.v1 import simd_qmm
    global _params
    T = mx.array(table()) if T is None else T
    if _params is None:
        _params = mx.array([K_MIN, K_MAX - K_MIN + 3, 2 * INT_INPUT_MAX + 1], dtype=mx.int32)
    L = int(x.shape[0])
    xi, e = to_int_kernel(x)
    g = simd_qmm.qmm_int(xi, *gate)
    u = simd_qmm.qmm_int(xi, *up)
    N = int(g.shape[1])
    ai, ef = _kernel("act")(inputs=[g, u, e, T.reshape(-1), _params], grid=(TG * L, 1, 1), threadgroup=(TG, 1, 1),
                           output_shapes=[(L, N), (L,)], output_dtypes=[mx.int16, mx.int32])
    y = simd_qmm.qmm_int(ai, *down)
    return y * ((127 + mx.clip(ef, -126, 127)).astype(mx.uint32) << 23).view(mx.float32)[:, None]
