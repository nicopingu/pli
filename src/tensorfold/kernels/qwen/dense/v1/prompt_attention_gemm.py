"""PLI (12/10/2026): Q.Kt and P.V products of the PREFILL (parts of more than 16 rows, D 256, bf16) with MLX's steel_gemm_fused kernel
(MIT licence, Apple) instantiated with other tiles than those MLX chooses; same bits (mx.array_equal) as MLX's unfused path; the causal mask is
applied in the store of Q.Kt. To copy under src/tensorfold/kernels/qwen/dense/v1/ (NOT applied: proposal). Enabled by PLI_ATTENTION_GEMM=1 in prompt_attention.py.

Conditions (otherwise the caller keeps the original path): bf16, b = 1, D = 256, number of visible keys a multiple of 16 (otherwise MLX handles the tail of K first),
more than 16 rows, key and value buffers with inner stride 1 and row stride 256 (checked once by a small kernel that returns the strides).
"""

from __future__ import annotations

import os
import re

import mlx.core as mx

# tiles (BM, BN, BK, WM, WN) measured at equal bits on M1 Ultra; P.V: short tile up to PV_THRESHOLD visible keys, long tile beyond
PV_SHORT = tuple(int(x) for x in os.environ.get("PLI_ATTENTION_GEMM_PV", "32,32,32,4,1").split(","))
PV_LONG = tuple(int(x) for x in os.environ.get("PLI_ATTENTION_GEMM_PV_LONG", "16,32,64,2,2").split(","))
PV_THRESHOLD = int(os.environ.get("PLI_ATTENTION_GEMM_PV_THRESHOLD", "65536"))
QK_CFG = tuple(int(x) for x in os.environ.get("PLI_ATTENTION_GEMM_QK", "32,64,16,1,2").split(","))
MIN_KEYS = int(os.environ.get("PLI_ATTENTION_GEMM_MIN", "16384"))

_ROOT = os.path.join(os.path.dirname(mx.__file__), "include")
_ALREADY = {"mlx/backend/metal/kernels/utils.h", "mlx/backend/metal/kernels/bf16.h", "mlx/backend/metal/kernels/bf16_math.h",
         "mlx/backend/metal/kernels/complex.h", "mlx/backend/metal/kernels/defines.h", "mlx/backend/metal/kernels/logging.h"}
_INC = re.compile(r'^\s*#include\s+"(mlx/[^"]+)"')


def _expand_includes(path, seen):
    out = []
    with open(os.path.join(_ROOT, path)) as f:
        for line in f:
            m = _INC.match(line)
            if m:
                p = m.group(1)
                if p not in _ALREADY and p not in seen:
                    seen.add(p)
                    out.append(_expand_includes(p, seen))
                continue
            if line.strip() == "#pragma once":
                continue
            out.append(line)
    return "".join(out)


_HEADER = None
_KER: dict = {}
_EX = None
_OK: dict = {}

# inputs: A (24, M, K) contiguous; B buffer (1, HK, capacity, 256) read by its strides; prm = [M, N, K, offset, group]; ex = [finfo.min] (bf16)
_SOURCE = r"""
  typedef bfloat16_t T;
  constexpr int BM = @BM@, BN = @BN@, BK = @BK@, WM = @WM@, WN = @WN@;
  constexpr bool TB = @TB@;
  constexpr bool MASK = @MASK@;
  using gk = mlx::steel::GEMMKernel<T, T, BM, BN, BK, WM, WN, false, TB, true, true, float>;
  using la_t = typename gk::loader_a_t;
  using lb_t = typename gk::loader_b_t;
  using mma_t = typename gk::mma_t;
  const int M = prm[0], N = prm[1], K = prm[2], off = prm[3], grp = prm[4];
  const int lda = K, ldd = N;
  const int ldb = int(B_strides[2]);
  const long hbs = long(B_strides[1]);
  uint3 tid = threadgroup_position_in_grid;
  const int c_row = int(tid.y) * BM, c_col = int(tid.x) * BN;
  const int z = int(tid.z);
  const int tgp_bm = min(BM, M - c_row), tgp_bn = min(BN, N - c_col);
  const bool full = (tgp_bm == BM) && (tgp_bn == BN);
  const device T* Ap = (const device T*)A + long(z) * long(M) * long(K) + long(c_row) * lda;
  const device T* Bp = (const device T*)B + long(z / grp) * hbs + (TB ? long(c_col) * ldb : long(c_col));
  device T* Dq = (device T*)D + long(z) * long(M) * long(N);
  threadgroup T As[gk::tgp_mem_size_a];
  threadgroup T Bs[gk::tgp_mem_size_b];
  threadgroup_barrier(mem_flags::mem_none);
  thread mma_t mma_op(simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
  thread la_t la(Ap, lda, As, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
  thread lb_t lb(Bp, ldb, Bs, simdgroup_index_in_threadgroup, thread_index_in_simdgroup);
  const int iters = K / BK;
  for (int k = 0; k < iters; k++) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (full) {
      la.load_unsafe();
      lb.load_unsafe();
    } else {
      la.load_safe(short2(BK, tgp_bm));
      lb.load_safe(TB ? short2(BK, tgp_bn) : short2(tgp_bn, BK));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
    la.next();
    lb.next();
  }
  const int rem = K - iters * BK;
  if (rem > 0) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    la.load_safe(short2(rem, tgp_bm));
    lb.load_safe(TB ? short2(rem, tgp_bn) : short2(tgp_bn, rem));
    threadgroup_barrier(mem_flags::mem_threadgroup);
    mma_op.mma(As, Bs);
  }
  threadgroup_barrier(mem_flags::mem_none);
  if (MASK) {
    const T mn = ex[0];
    device T* Dt = Dq;
    for (short i = 0; i < mma_t::TM; i++) {
      for (short j = 0; j < mma_t::TN; j++) {
        thread const auto& acc = mma_op.Ctile.frag_at(i, j);
        const int r = c_row + int(mma_op.sm) + i * int(mma_t::TM_stride);
        for (short e = 0; e < 2; e++) {
          const int c = c_col + int(mma_op.sn) + j * int(mma_t::TN_stride) + e;
          if (r < M && c < N) {
            T val = static_cast<T>(acc[e]);
            Dt[long(r) * ldd + c] = (c > off + r) ? mn : val;
          }
        }
      }
    }
  } else {
    device T* Dp = Dq + long(c_row) * ldd + c_col;
    if (full) {
      mma_op.store_result(Dp, ldd);
    } else {
      mma_op.store_result_safe(Dp, ldd, short2(tgp_bn, tgp_bm));
    }
  }
"""



def _kernel(cfg, tb, mask):
    global _HEADER
    if _HEADER is None:
        _HEADER = _expand_includes("mlx/backend/metal/kernels/steel/gemm/gemm.h", {"mlx/backend/metal/kernels/steel/gemm/gemm.h"}) + "\n"
    key = (cfg, tb, mask)
    if key not in _KER:
        bm, bn, bk, wm, wn = cfg
        src = (_SOURCE.replace("@BM@", str(bm)).replace("@BN@", str(bn)).replace("@BK@", str(bk)).replace("@WM@", str(wm)).replace("@WN@", str(wn))
               .replace("@TB@", "true" if tb else "false").replace("@MASK@", "true" if mask else "false"))
        _KER[key] = mx.fast.metal_kernel(name="pli_attgemm_%d_%d_%d_%d_%d_%d_%d" % (bm, bn, bk, wm, wn, int(tb), int(mask)),
                                         input_names=["A", "B", "prm", "ex"], output_names=["D"], source=src, header=_HEADER,
                                         ensure_row_contiguous=False)
    return _KER[key]


def _gemm(a, b, batch, m, n, k, off, grp, cfg, tb, mask):
    global _EX
    if _EX is None:
        _EX = mx.array([float(mx.finfo(mx.bfloat16).min)], dtype=mx.bfloat16)
    bm, bn, _bk, wm, wn = cfg
    prm = mx.array([m, n, k, off, grp], dtype=mx.int32)
    tn, tm = -(-n // bn), -(-m // bm)
    return _kernel(cfg, tb, mask)(inputs=[a, b, prm, _EX], output_shapes=[(batch, m, n)], output_dtypes=[mx.bfloat16],
                                 grid=(tn * 32, tm * wn, batch * wm), threadgroup=(32, wn, wm))[0]


_STRIDE_PROBE = """
  out[0] = int(A_strides[0]); out[1] = int(A_strides[1]); out[2] = int(A_strides[2]); out[3] = int(A_strides[3]);
"""


def _layout_ok(x: mx.array) -> bool:
    """Inner stride 1 and row stride = D (buffer (1, hk, capacity, D)): read once per shape (hk, D)."""
    key = (int(x.shape[1]), int(x.shape[3]))
    if key not in _OK:
        ker = mx.fast.metal_kernel(name="pli_attgemm_strides", input_names=["A"], output_names=["out"], source=_STRIDE_PROBE, ensure_row_contiguous=False)
        s = ker(inputs=[x], output_shapes=[(4,)], output_dtypes=[mx.int32], grid=(1, 1, 1), threadgroup=(1, 1, 1))[0].tolist()
        _OK[key] = (s[3] == 1 and s[2] == int(x.shape[3]))
    return _OK[key]


def applicable(queries: mx.array, keys: mx.array, values: mx.array) -> bool:
    b, hq, r, d = (int(v) for v in queries.shape)
    hk, vis = int(keys.shape[1]), int(keys.shape[2])
    return (b == 1 and d == 256 and r > 16 and hq % hk == 0 and vis % 16 == 0 and vis >= MIN_KEYS
            and queries.dtype == mx.bfloat16 and keys.dtype == mx.bfloat16 and values.dtype == mx.bfloat16
            and _layout_ok(keys) and _layout_ok(values))


def part(queries: mx.array, keys: mx.array, values: mx.array, scale: float) -> mx.array:
    """Causal attention of ``queries`` (the last rows of the ``keys.shape[2]`` positions); same bits as MLX's unfused SDPA."""
    b, hq, r, d = (int(v) for v in queries.shape)
    hk, vis = int(keys.shape[1]), int(keys.shape[2])
    g = hq // hk
    qs = (mx.array(scale, queries.dtype) * queries).reshape(hq, r, d)
    s = _gemm(qs, keys, hq, r, vis, d, vis - r, g, QK_CFG, True, True)
    p = mx.softmax(s, axis=-1, precise=True)
    o = _gemm(p, values, hq, r, d, vis, 0, g, PV_LONG if vis >= PV_THRESHOLD else PV_SHORT, False, False)
    return o.reshape(1, hq, r, d)
