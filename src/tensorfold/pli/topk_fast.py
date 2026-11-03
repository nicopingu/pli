"""PLI (01/11/2026, PLI_DRAFT_TOPK=1, off): the k largest of a row (engine/topk.py) in two passes spread over threadgroups, same outputs.

engine/topk.topk_rows takes one threadgroup per row: 7 threadgroups for the draft's lattice (7 rows of 100,640 columns),
about 0.18 ms (dbench2). Here: pass 1, the same base-256 selection over C slices of each row (one threadgroup per slice),
which keeps the k best of the slice (key decreasing, id increasing); pass 2, one simdgroup per row picks the k best of the
C x k candidates in the same total order. The row's k best are among those of the slices (total order, distinct ids): same
indices and same values as topk_rows, in the same order (checked)."""

from __future__ import annotations

import hashlib
import os
from typing import Any

import mlx.core as mx

from tensorfold.engine import topk as _orig
from tensorfold.kernels import threads

ENABLED = os.environ.get("PLI_DRAFT_TOPK", "0") == "1"
C = 16            # slices per row
TPG = 256

_P1 = (_orig._SOURCE
       .replace("""  const uint row = threadgroup_position_in_grid.x;
  const uint tid = thread_position_in_threadgroup.x;
  const int Vn = dims[0], K = dims[1];
  const device ushort* x = (const device ushort*)X + (int64_t)row * Vn;""",
                """  const uint row = threadgroup_position_in_grid.x / C, ch = threadgroup_position_in_grid.x % C;
  const uint tid = thread_position_in_threadgroup.x;
  const int VN = dims[0], K = dims[1], CL = dims[2];
  const int c0 = int(ch) * CL;
  const int Vn = min(CL, VN - c0);
  const device ushort* x = (const device ushort*)X + (int64_t)row * VN + c0;""")
       .replace("""  for (int j = tid; j < K; j += TPG) {
    const uint idx = sel_idx[j];
    IDX[row * K + j] = int(idx);
    VAL[row * K + j] = float(X[(int64_t)row * Vn + idx]);
  }""", """  for (int j = tid; j < K; j += TPG) {
    CID[(row * C + ch) * K + j] = int(sel_idx[j]) + c0;
    CKEY[(row * C + ch) * K + j] = sel_key[j];
  }"""))
assert "CID[" in _P1 and "ch = threadgroup_position_in_grid.x % C" in _P1

_P2 = r"""
  // one simdgroup per row: the K best of the C * K candidates, key decreasing then id increasing
  const uint row = threadgroup_position_in_grid.x;
  const uint lane = thread_index_in_simdgroup;
  const int VN = dims[0], K = dims[1];
  constexpr int NC = C * MAXK;
  const int n = C * K;
  uint key[(NC + 31) / 32], id[(NC + 31) / 32];
  const int per = (n + 31) / 32;
  for (int t = 0; t < per; t++) {
    const int i = t * 32 + int(lane);
    key[t] = i < n ? CKEY[row * n + i] : 0u;
    id[t] = i < n ? uint(CID[row * n + i]) : 0xFFFFFFFFu;
  }
  for (int j = 0; j < K; j++) {
    uint bk = 0u, bi = 0xFFFFFFFFu; int bt = -1;
    for (int t = 0; t < per; t++) {
      if (id[t] != 0xFFFFFFFFu && (key[t] > bk || (key[t] == bk && id[t] < bi) || bt < 0)) { bk = key[t]; bi = id[t]; bt = t; }
    }
    const uint mk = simd_max(bt >= 0 ? bk : 0u);
    const uint mi = simd_min((bt >= 0 && bk == mk) ? bi : 0xFFFFFFFFu);
    if (bt >= 0 && bk == mk && bi == mi) id[bt] = 0xFFFFFFFFu;
    if (lane == 0) {
      IDX[row * K + j] = int(mi);
      VAL[row * K + j] = float(X[(int64_t)row * VN + mi]);
    }
  }
"""

_k: dict = {}


def _kernels() -> tuple[Any, Any]:
    if "p1" not in _k:
        consts = f"  constexpr int TPG = {TPG};\n  constexpr int MAXK = {_orig.MAX_K};\n  constexpr int MAXT = 2048;\n  constexpr int C = {C};\n"
        h1 = threads.reserve(TPG)
        d1 = hashlib.sha256((h1 + consts + _P1).encode()).hexdigest()[:16]
        _k["p1"] = mx.fast.metal_kernel(name=f"pli_topk_p1_{d1}", input_names=["X", "dims"], output_names=["CID", "CKEY"],
                                        source=consts + _P1, header=h1)
        c2 = f"  constexpr int MAXK = {_orig.MAX_K};\n  constexpr int C = {C};\n"
        h2 = threads.reserve(32)
        d2 = hashlib.sha256((h2 + c2 + _P2).encode()).hexdigest()[:16]
        _k["p2"] = mx.fast.metal_kernel(name=f"pli_topk_p2_{d2}", input_names=["X", "CID", "CKEY", "dims"],
                                        output_names=["IDX", "VAL"], source=c2 + _P2, header=h2)
    return _k["p1"], _k["p2"]


def topk_rows(x: mx.array, k: int) -> tuple[mx.array, mx.array]:
    """Like engine.topk.topk_rows (same outputs); the original if a slice would have fewer than k columns."""

    if x.dtype != mx.bfloat16:
        x = x.astype(mx.bfloat16)
    x2 = mx.contiguous(x.reshape(-1, int(x.shape[-1])))
    rows, vocab = int(x2.shape[0]), int(x2.shape[1])
    k = int(k)
    CL = -(-vocab // C)
    if not 1 <= k <= _orig.MAX_K or k > vocab - (C - 1) * CL or C * k > 32 * 32:
        return _orig.topk_rows(x, k)
    dims = mx.array([vocab, k, CL], dtype=mx.int32)
    p1, p2 = _kernels()
    cid, ckey = p1(inputs=[x2, dims], grid=(rows * C * TPG, 1, 1), threadgroup=(TPG, 1, 1),
                   output_shapes=[(rows * C * k,), (rows * C * k,)], output_dtypes=[mx.int32, mx.uint32])
    idx, val = p2(inputs=[x2, cid, ckey, dims], grid=(rows * 32, 1, 1), threadgroup=(32, 1, 1),
                  output_shapes=[(rows, k), (rows, k)], output_dtypes=[mx.int32, mx.float32])
    return idx, val
