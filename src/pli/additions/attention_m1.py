"""PLI, rank 2: single-read causal attention for chips without "tensor units" (M1 to M4).

Takes the algorithm of ``kernels/qwen/dense/v1/lane_attention.py`` (written for M5) with the M1's 8 x 8 matrices per
simdgroup. All rows of a window go together: the queries of a key/value head (G heads x T rows) are
laid out in tiles of 8, and each tile walks the keys only once.
A tile takes SPLIT simdgroups that share the dimensions: few registers per thread (the M1's limit).

Arithmetic fixed by the absolute position of the keys (this is what makes a row identical alone or in a window):
- key slices with fixed bounds (table SLICES: small at the start, larger after), cut into tiles of TK keys,
  themselves into blocks of 8 keys;
- score of a query for a key: one element of a chain of 8 x 8 matrix products over the D / 8 dimension blocks;
- online softmax per tile of TK keys (maximum, weights, sum in a fixed order), the masked keys weigh zero;
- output: P x V in 8 x 8 blocks, accumulated in fp32; the slices are merged in order by a second kernel.
A row depends neither on the other rows of its tile (each element of an 8 x 8 product has its own chain), nor on the
number of rows, nor on the context length at the time of the computation.

Chains only (no draft tree): row t of a window of T sees keys 0 .. L - T + t.
"""

from __future__ import annotations

import bisect
import hashlib
import os
from typing import Any

import mlx.core as mx

# Key slices: (size, count) one after the other, the last size without end. Bounds fixed by the absolute position
# (arithmetic). Measured on 09/10 (kernel alone): 512 everywhere equals or beats the tables of increasing sizes tried.
SLICES = tuple(tuple(int(x) for x in m.split("x")) for m in os.environ.get("PLI_ATTN_SLICES", "512x0").split(","))
MAX_KEYS = 1 << 20
TILE = int(os.environ.get("PLI_ATTN_TK", "64"))        # keys per softmax step (fixed once chosen: arithmetic)
MAX_QUERIES = 128
# simdgroups per tile of 8 queries: they share the D / 8 steps of the products, and their score parts are summed
# in order (fixed once chosen: arithmetic). 4: each simdgroup keeps 8 output matrices in registers.
SPLIT = int(os.environ.get("PLI_ATTN_NS", "4"))
# tiles of 8 queries per simdgroup (they share the reads of keys and values); no effect on the bits
ROW_TILES = int(os.environ.get("PLI_ATTN_RT", "1"))

_HEADER = r"""
// the bf16 of index e (0..7) among 8 bf16 packed in a uint4, in fp32
inline float bf8(uint4 v, int e) {
  const uint w = v[e / 2];
  return as_type<float>((e % 2) ? (w & 0xFFFF0000u) : (w << 16));
}
"""

_PARTIAL = r"""
  // One threadgroup per (key/value head, key slice, RT tiles of 8 queries). Its NS simdgroups share
  // the D / 8 steps of the products (JS steps each); each simdgroup handles the RT tiles over the same keys and values,
  // read and converted only once. RT does not change the bits; NS and TK are part of the arithmetic.
  const ushort lane = thread_index_in_simdgroup;
  const int sg = int(simdgroup_index_in_threadgroup);
  const int tile0 = int(threadgroup_position_in_grid.z) * RT;            // first tile of 8 queries
  const uint hk = threadgroup_position_in_grid.x;                        // key/value head
  const int c = int(threadgroup_position_in_grid.y);                     // key slice (bounds in CS)
  const int L = dims[0], NCH = dims[1], NQ = dims[2], T8 = dims[3];      // T8: tiles, multiple of RT
  const int RP = 8 * T8, RQ = G * NQ;
  constexpr int DJ = D / 8, JS = DJ / NS, JU = JS / 8;   // steps of a product; steps per simdgroup; uint4 (8 bf16) per range
  // this thread holds the elements [fm][fn] and [fm][fn + 1] of each 8 x 8 matrix of the simdgroup.
  // Dimension of an element: DJ * (inner index) + (step j); each thread thus reads contiguous ranges.
  const int qid = int(lane) >> 2;
  const int fm = (qid & 4) | ((int(lane) >> 1) & 3);
  const int fn = ((qid & 2) << 1) | ((int(lane) & 1) << 1);
  const int j0 = sg * JS;                                                // first step of this simdgroup
  const device bfloat* Kh = K + int64_t(hk) * int64_t(K_strides[1]) + j0;
  const device bfloat* Vh = V + int64_t(hk) * int64_t(V_strides[1]) + j0;
  const int64_t ks = int64_t(K_strides[2]), vs = int64_t(V_strides[2]);
  threadgroup float SP[RT * NS * 8 * TK];
  int nq[RT];                                                            // keys seen by this thread's query, per tile
  float qf[RT][JS][2];
  simdgroup_matrix<float, 8, 8> O[RT][JS];
  float m[RT], l[RT];
  for (int t = 0; t < RT; t++) {
    const int r = (tile0 + t) * 8 + fm;                                  // query t * G + g
    nq[t] = (r < RQ) ? L - NQ + r / G + 1 : 0;
    const device bfloat* qr = Qp + (int64_t(hk) * RQ + min(r, RQ - 1)) * D + j0;   // incomplete tile: query read again, masked
    for (int u = 0; u < JU; u++) {
      const uint4 ya = ((const device uint4*)(qr + DJ * fn))[u], yb = ((const device uint4*)(qr + DJ * (fn + 1)))[u];
      for (int e = 0; e < 8; e++) { qf[t][8 * u + e][0] = bf8(ya, e); qf[t][8 * u + e][1] = bf8(yb, e); }
    }
    for (int j = 0; j < JS; j++) O[t][j] = simdgroup_matrix<float, 8, 8>(0.0f);
    m[t] = -INFINITY; l[t] = 0.0f;
  }
  const int kbeg = CS[c];
  const int kend = min(int(CS[c + 1]), L);
  for (int kt = kbeg; kt < kend; kt += TK) {
    for (int b = 0; b < TK / 8; b++) {
      const int k0 = kt + 8 * b;
      const device uint4* ka = (const device uint4*)(Kh + int64_t(min(k0 + fn, L - 1)) * ks + DJ * fm);
      const device uint4* kb = (const device uint4*)(Kh + int64_t(min(k0 + fn + 1, L - 1)) * ks + DJ * fm);
      simdgroup_matrix<float, 8, 8> S[RT];
      for (int t = 0; t < RT; t++) S[t] = simdgroup_matrix<float, 8, 8>(0.0f);
      for (int u = 0; u < JU; u++) {
        const uint4 xa = ka[u], xb = kb[u];
        for (int e = 0; e < 8; e++) {
          simdgroup_matrix<float, 8, 8> bm;
          bm.thread_elements()[0] = bf8(xa, e);
          bm.thread_elements()[1] = bf8(xb, e);
          for (int t = 0; t < RT; t++) {
            simdgroup_matrix<float, 8, 8> am;
            am.thread_elements()[0] = qf[t][8 * u + e][0];
            am.thread_elements()[1] = qf[t][8 * u + e][1];
            simdgroup_multiply_accumulate(S[t], am, bm, S[t]);
          }
        }
      }
      for (int t = 0; t < RT; t++) {
        SP[((t * NS + sg) * 8 + fm) * TK + 8 * b + fn] = S[t].thread_elements()[0];
        SP[((t * NS + sg) * 8 + fm) * TK + 8 * b + fn + 1] = S[t].thread_elements()[1];
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float p[RT][TK / 4];
    for (int t = 0; t < RT; t++) {
      float s[TK / 4];
      for (int i = 0; i < TK / 4; i++) {                   // key kt + 8 (i / 2) + fn + (i % 2): the NS parts in order
        const int key = 8 * (i / 2) + fn + (i % 2);
        float v = SP[(t * NS * 8 + fm) * TK + key];
        for (int w = 1; w < NS; w++) v += SP[((t * NS + w) * 8 + fm) * TK + key];
        s[i] = (kt + key < nq[t]) ? v * scale[0] : -INFINITY;
      }
      float x = -INFINITY;
      for (int i = 0; i < TK / 4; i++) x = max(x, s[i]);
      x = max(x, simd_shuffle_xor(x, ushort(1)));
      x = max(x, simd_shuffle_xor(x, ushort(8)));
      const float nm = max(m[t], x);
      const float f = (x == -INFINITY) ? 1.0f : precise::exp(m[t] - nm);
      for (int i = 0; i < TK / 4; i++) p[t][i] = (s[i] == -INFINITY) ? 0.0f : precise::exp(s[i] - nm);
      float y = 0.0f;
      for (int b = 0; b < TK / 8; b++) y += p[t][2 * b] + p[t][2 * b + 1];
      y += simd_shuffle_xor(y, ushort(1));
      y += simd_shuffle_xor(y, ushort(8));
      if (x != -INFINITY) { l[t] = l[t] * f + y; m[t] = nm; }
      for (int j = 0; j < JS; j++) { O[t][j].thread_elements()[0] *= f; O[t][j].thread_elements()[1] *= f; }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int b = 0; b < TK / 8; b++) {
      const device bfloat* pv = Vh + int64_t(min(kt + 8 * b + fm, L - 1)) * vs;
      const device uint4* va = (const device uint4*)(pv + DJ * fn);
      const device uint4* vb = (const device uint4*)(pv + DJ * (fn + 1));
      for (int u = 0; u < JU; u++) {
        const uint4 xa = va[u], xb = vb[u];
        for (int e = 0; e < 8; e++) {
          simdgroup_matrix<float, 8, 8> vm;
          vm.thread_elements()[0] = bf8(xa, e);
          vm.thread_elements()[1] = bf8(xb, e);
          for (int t = 0; t < RT; t++) {
            simdgroup_matrix<float, 8, 8> pm;
            pm.thread_elements()[0] = p[t][2 * b];
            pm.thread_elements()[1] = p[t][2 * b + 1];
            simdgroup_multiply_accumulate(O[t][8 * u + e], pm, vm, O[t][8 * u + e]);
          }
        }
      }
    }
  }
  for (int t = 0; t < RT; t++) {
    const int64_t row = (int64_t(hk) * NCH + c) * RP + (tile0 + t) * 8 + fm;
    for (int j = 0; j < JS; j++) {
      PO[row * D + DJ * fn + j0 + j] = O[t][j].thread_elements()[0];
      PO[row * D + DJ * (fn + 1) + j0 + j] = O[t][j].thread_elements()[1];
    }
    if (sg == 0 && (int(lane) & 9) == 0) { PM[row] = m[t]; PL[row] = l[t]; }
  }
"""

_MERGE = r"""
  const uint lane = thread_index_in_simdgroup;
  const uint hk = threadgroup_position_in_grid.x;
  const uint r = threadgroup_position_in_grid.y;              // query t * G + g
  const int NCH = dims[1], NQ = dims[2], T8 = dims[3];
  const int RP = 8 * T8;
  constexpr int DP = D / 32;
  float m = -INFINITY, l = 0.0f, o[DP];
  for (int i = 0; i < DP; i++) o[i] = 0.0f;
  for (int c = 0; c < NCH; c++) {
    const int64_t row = (int64_t(hk) * NCH + c) * RP + r;
    const float mc = PM[row];
    if (mc == -INFINITY) continue;
    const float lc = PL[row];
    const float nm = max(m, mc);
    const float f1 = precise::exp(m - nm), f2 = precise::exp(mc - nm);
    l = l * f1 + lc * f2;
    for (int i = 0; i < DP; i++) o[i] = o[i] * f1 + PO[row * D + lane * DP + i] * f2;
    m = nm;
  }
  const int t = int(r) / G, g = int(r) % G, h = int(hk) * G + g;
  for (int i = 0; i < DP; i++) OUT[(int64_t(h) * NQ + t) * D + lane * DP + i] = static_cast<bfloat>(o[i] / l);
"""

_kernels: dict[str, Any] = {}
_table: list[Any] = []


def _slices() -> tuple[list[int], Any]:
    """Bounds of the slices (increasing list) and the same array for the kernel."""

    if not _table:
        bounds = [0]
        for i, (size, count) in enumerate(SLICES):
            if size % TILE or size < TILE:
                raise ValueError(f"pli attention: slice of {size} keys, not a multiple of {TILE}")
            while bounds[-1] < MAX_KEYS and (count <= 0 or i == len(SLICES) - 1 or count > 0):
                bounds.append(bounds[-1] + size)
                count -= 1
                if count == 0 and i < len(SLICES) - 1:
                    break
        _table.extend([bounds, mx.array(bounds, dtype=mx.int32)])
    return _table[0], _table[1]
_sg_cap: dict[int, int] = {}


def sources() -> dict[str, str]:
    """The text of the kernels and their arithmetic constants (for the snapshot version)."""

    return {"header": _HEADER, "partial": _PARTIAL, "merge": _MERGE, "chunk": f"{SLICES}/{TILE}/{SPLIT}"}


def _kernel(name: str, consts: tuple) -> Any:
    """One kernel per set of constants, written into the text (MLX's "template" arguments cost a regex per call)."""

    key = (name, consts)
    if key not in _kernels:
        source = "".join(f"  constexpr int {k} = {v};\n" for k, v in consts) + (_PARTIAL if name == "partial" else _MERGE)
        digest = hashlib.sha256((_HEADER + source).encode()).hexdigest()[:16]
        if name == "partial":
            _kernels[key] = mx.fast.metal_kernel(
                name=f"pli_attention_partial_{digest}", input_names=["Qp", "K", "V", "scale", "dims", "CS"],
                output_names=["PO", "PM", "PL"], source=source, header=_HEADER, ensure_row_contiguous=False)
        else:
            _kernels[key] = mx.fast.metal_kernel(
                name=f"pli_attention_merge_{digest}", input_names=["PO", "PM", "PL", "dims"], output_names=["OUT"],
                source=source)
    return _kernels[key]


def sdpa(queries: mx.array, keys: mx.array, values: mx.array, scale: float, split: int = 0, row_tiles: int = 0) -> mx.array:
    """Causal attention: queries [1, H, T, D], keys and values [1, Hkv, L, D] (views of a cache buffer allowed)."""

    _, H, T, D = (int(s) for s in queries.shape)
    HKV, L = int(keys.shape[1]), int(keys.shape[2])
    if D % 64 or H % HKV or T > MAX_QUERIES or T > L or TILE % 8 or L > MAX_KEYS:
        raise ValueError(f"pli attention: shape not handled q={queries.shape} k={keys.shape}")
    if queries.dtype != mx.bfloat16 or keys.dtype != mx.bfloat16 or values.dtype != mx.bfloat16:
        raise ValueError("pli attention: bf16 only")
    G = H // HKV
    R = G * T
    RT = max(1, min(int(row_tiles) or ROW_TILES, (R + 7) // 8))
    T8 = RT * -(-R // (8 * RT))
    RP = 8 * T8
    NS = int(split) or SPLIT
    if (D // 8) % (8 * NS):
        raise ValueError(f"pli attention: {NS} simdgroups do not divide the {D // 8} steps into ranges of 8")
    # queries laid out [key/value head, row, head of the group]; at one row it is the original order, nothing to copy
    qp = mx.contiguous(queries if T == 1 else queries.reshape(HKV, G, T, D).transpose(0, 2, 1, 3))
    bounds, cs = _slices()
    nch = bisect.bisect_left(bounds, L)
    dims = mx.array([L, nch, T, T8], dtype=mx.int32)
    po, pm, pl = _kernel("partial", (("G", G), ("D", D), ("NS", NS), ("RT", RT), ("TK", TILE)))(
        inputs=[qp, keys, values, mx.array([float(scale)], dtype=mx.float32), dims, cs],
        grid=(HKV * 32 * NS, nch, T8 // RT), threadgroup=(32 * NS, 1, 1),
        output_shapes=[(HKV * nch * RP * D,), (HKV * nch * RP,), (HKV * nch * RP,)],
        output_dtypes=[mx.float32, mx.float32, mx.float32])
    return _kernel("merge", (("G", G), ("D", D)))(
        inputs=[po, pm, pl, dims],
        grid=(HKV * 32, R, 1), threadgroup=(32, 1, 1),
        output_shapes=[(1, H, T, D)], output_dtypes=[mx.bfloat16])[0]


__all__ = ["MAX_QUERIES", "SLICES", "TILE", "sdpa", "sources"]
