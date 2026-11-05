"""PLI (03/11/2026, A1, V3): reduced attention "anchor 1", scored by the index of Latent Sparse Attention (PLI_A1_LSA_MAP=<map.npz>).

Rule (offline bench of 03/11/2026: NLL gap to full attention 0.0095 against 0.044 for the anchor of 32, first-choice agreement
96.9 % against 94.6 %): the row at position q chooses ITS OWN tiles of 32 keys of the old context [0, start),
start = max(32, ((q - 1024) // 32) * 32), on its own query, scored by the distilled map of LSA (Prefrontal-AI, Apache-2.0):
  index key of position p    = normalize(K[0..3, p, 64:256] side by side (768) @ W_K_idx) (64 values, kept in f16 on the cache);
  index query of row q       = normalize(sum of the 6 heads of the group of Q[h, q, 64:256], side by side (768) @ W_Q_idx) (f32);
  score of a tile = max over its 32 keys of q_idx . k_idx; ceil(REDUCED % of the tiles) kept, same choice for the 4 KV heads;
then the whole tail [start, q]. Full layers (PLI_ATTENTION_REDUCED_FULL_LAYERS) and rows below PLI_ATTENTION_REDUCED_THRESHOLD:
complete. A row's choice depends only on its position and the written tokens: no choice kept per anchor, nothing to redo after
a refusal.

PLI_A1_PREFILL_ONLY=1 (05/11/2026): A1 only for the prompt chunks (rows before the start of the request, the S3b path); decode
and trees keep the anchored rule. In production on an M1 Ultra since 2026-11-05 with PLI_S3B_THRESHOLD=16384 and PLI_S3B_ROWS=4096: first
token at 146K tokens 844.6 -> 712.5 s (-15.6 %); offline NLL equal to the full prefill at 48K and 97K tokens.
PLI_A1_MAX_TILES=N (0 = no cap): at most N tiles kept per row (faster at long context, but the NLL rises at 97K: not used).

Exactness (a row has the same bits alone or in a round, read in one go or resumed):
 - index keys and queries: sums in a fixed order (fma) of their own for each position (row), norm by two simd_sum;
 - tile scores: one simdgroup per tile, a fixed-order product per key, simdgroup max; sort per row (mx.argsort);
 - read: attm kernel (same blocks of 16 keys, same arithmetic) on the keys of the CACHE, no copy. Slices with ABSOLUTE bounds:
   slices of SLICE positions below S = (start // SLICE) * SLICE, then of SMALL_SLICE positions to the end. A threadgroup reads
   the blocks of the union of the tiles of its 8 rows and the whole area past the smallest start; each row masks the blocks
   that are not its own (a masked block changes neither m, nor l, nor O); a slice without a key of a row writes nothing and is
   skipped at the merge. The rows of a round that do not share S (once every SLICE positions) go in two calls.
 - K and V go to f16 clamped to +-65504: a masked block of a row cannot carry an infinity there.

Index keys: they live on the layer's cache (attribute pli_kidx = [buffer, n, id of the cache]) and serve only the old context (at
least 992 positions before the row). They are computed from K when a position enters it (a few positions per round), or at once for
a resumed prefix (first reduced read of a request). At the start of each request (family_prefill) they are cut to the number of
resumed tokens (resume()); a copy of the cache (copied checkpoint, merge) has another id: they are rebuilt there. A cache shorter
than the index keys (rollback) cuts them to its length. Disk snapshots do not write them (pli_* attributes): rebuilt on reload.
"""

from __future__ import annotations

import math
import os
from typing import Any, Sequence

import mlx.core as mx

from tensorfold.kernels import inputs as _in

MAP_PATH = os.environ.get("PLI_A1_LSA_MAP", "").strip()
ACTIVE = bool(MAP_PATH)
# PLI (05/11/2026): A1 only for the prompt chunks (rows before the start of the request); decode and trees keep the anchored rule.
# The index keys extend by themselves at the next prefill (index_keys).
PREFILL_ONLY = os.environ.get("PLI_A1_PREFILL_ONLY", "0") == "1"


def _multiple(name: str, default: int) -> int:
    try:
        v = int(os.environ.get(name, str(default)) or default)
    except ValueError:
        v = default
    return max(32, (v // 32) * 32)


# Slices of the read (part of the arithmetic, V3): SLICE positions in the far old context, SMALL_SLICE after.
SLICE = _multiple("PLI_A1_SLICE", 2048)
SMALL_SLICE = _multiple("PLI_A1_SMALL_SLICE", 128)
ROT, DH, HKV, G, H, DI = 64, 256, 4, 6, 24, 64
TILE, TAIL = 32, 1024
KIDX_CHUNK = 16384                       # positions per call of the index-key kernel (bits per position: the split changes nothing)
COUNT = {"calls": 0, "rows": 0, "index_keys": 0, "no_map": 0}
FMIN = -3.4028234663852886e+38
_MAP: dict | None = None


def set_map(lsa_map: dict | None) -> None:
    """Tests: map {layer: (W_Q_idx, W_K_idx)} (mx f32 [768, 64]) instead of the file."""

    global _MAP
    _MAP = lsa_map


def lsa_map() -> dict:
    """The distilled map: {layer: (W_Q_idx, W_K_idx)}, f32 [768, 64], read once (keys "<layer>.W_Q_idx" / ".W_K_idx")."""

    global _MAP
    if _MAP is None:
        import numpy as np

        z = np.load(MAP_PATH)
        read: dict = {}
        for name in z.files:
            layer, _, kind = name.partition(".")
            if not layer.isdigit() or kind not in ("W_Q_idx", "W_K_idx"):
                raise ValueError(f"PLI_A1_LSA_MAP: unexpected key {name} in {MAP_PATH}")
            a = np.ascontiguousarray(z[name], dtype=np.float32)
            if a.shape != (HKV * (DH - ROT), DI):
                raise ValueError(f"PLI_A1_LSA_MAP: {name} has shape {a.shape}, expected {(HKV * (DH - ROT), DI)}")
            read.setdefault(int(layer), {})[kind] = mx.array(a)
        _MAP = {c: (v["W_Q_idx"], v["W_K_idx"]) for c, v in read.items() if len(v) == 2}
        mx.eval([w for v in _MAP.values() for w in v])
    return _MAP


def has_layer(layer: Any) -> bool:
    """Does the map have this layer? Otherwise the layer keeps the original anchored read (counted in COUNT["no_map"])."""

    ok = layer is not None and int(layer) in lsa_map()
    if not ok:
        COUNT["no_map"] += 1
    return ok


def tail_start(q: int) -> int:
    """Start of the tail read in full by the row at position q (exact_attention._anchored with a = q)."""

    return max(TILE, ((q - TAIL) // TILE) * TILE)


# PLI (05/11/2026): dynamic A1, at most PLI_A1_MAX_TILES tiles kept per row (0: no cap, as before).
MAX_TILES = max(0, int(os.environ.get("PLI_A1_MAX_TILES", "0") or 0))


def kept(tiles: int, reduced: int) -> int:
    g = max(1, math.ceil(reduced / 100 * tiles))
    return min(g, MAX_TILES) if MAX_TILES else g


# ------------------------------------------------------------------------------------------------ index kernels
# 8 positions per thread (the weights read once for the 8); for each position, the same fma sequence and norm whatever the split
_KIDX = r"""
  const int j = int(thread_position_in_threadgroup.x);
  const int y0 = int(threadgroup_position_in_grid.y) * 8;
  const int n = dims[1];
  const size_t s1 = K_strides[1], s2 = K_strides[2];
  float acc[8];
  size_t kb[8];
#pragma clang loop unroll(full)
  for (int e = 0; e < 8; e++) { acc[e] = 0.0f; kb[e] = size_t(dims[0] + min(y0 + e, n - 1)) * s2 + 64; }
  for (int kv = 0; kv < 4; kv++) {
    const device bfloat* kp = K + size_t(kv) * s1;
    const device float* wp = W + size_t(kv * 192) * 64 + j;
    for (int c = 0; c < 192; c++) {
      const float w = wp[size_t(c) * 64];
#pragma clang loop unroll(full)
      for (int e = 0; e < 8; e++) acc[e] = fma(float(kp[kb[e] + c]), w, acc[e]);
    }
  }
  threadgroup float part[2][8];
#pragma clang loop unroll(full)
  for (int e = 0; e < 8; e++) {
    const float sq = simd_sum(acc[e] * acc[e]);
    if (thread_index_in_simdgroup == 0) part[simdgroup_index_in_threadgroup][e] = sq;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
#pragma clang loop unroll(full)
  for (int e = 0; e < 8; e++) {
    if (y0 + e < n) {
      const float nrm = sqrt(part[0][e] + part[1][e]);
      OUT[size_t(y0 + e) * 64 + j] = half(acc[e] / max(nrm, 1e-12f));
    }
  }
"""

_QIDX = r"""
  const int j = int(thread_position_in_threadgroup.x);
  const int t = int(threadgroup_position_in_grid.y);
  const size_t s1 = Q_strides[1], s2 = Q_strides[2];
  float acc = 0.0f;
  for (int kv = 0; kv < 4; kv++) {
    const device float* wp = W + size_t(kv * 192) * 64 + j;
    for (int c = 0; c < 192; c++) {
      float s = 0.0f;
      for (int g = 0; g < 6; g++) s += float(Q[size_t(kv * 6 + g) * s1 + size_t(t) * s2 + 64 + c]);
      acc = fma(s, wp[size_t(c) * 64], acc);
    }
  }
  threadgroup float part[2];
  const float sq = simd_sum(acc * acc);
  if (thread_index_in_simdgroup == 0) part[simdgroup_index_in_threadgroup] = sq;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const float nrm = sqrt(part[0] + part[1]);
  OUT[size_t(t) * 64 + j] = acc / max(nrm, 1e-12f);
"""

# score of tile j for each row t (j < PAV[t], else FMIN): the tile's key `lane` in registers, a fixed-order product per row
_NOTE = r"""
  const float FMIN_ = -3.4028234663852886e+38f;
  const uint lane = thread_index_in_simdgroup;
  const int j = int(threadgroup_position_in_grid.y);
  const int T = dims[0], P = dims[1];
  const device half* kp = KI + (size_t(j) * 32 + lane) * 64;
  float kx[64];
#pragma clang loop unroll(full)
  for (int e = 0; e < 64; e++) kx[e] = float(kp[e]);
  for (int t = 0; t < T; t++) {
    float s = FMIN_;
    if (j < PAV[t]) {
      const device float* qp = QI + size_t(t) * 64;
      float d = 0.0f;
#pragma clang loop unroll(full)
      for (int e = 0; e < 64; e++) d = fma(kx[e], qp[e], d);
      s = simd_max(d);
    }
    if (lane == 0) TETE[size_t(t) * P + j] = s;
  }
"""

# kept tiles: the last GA[t] of the ascending order; bits per row (LB) and union per group of 8 rows (UB)
_MARQUE = r"""
  const int i = int(thread_position_in_grid.x), t = int(thread_position_in_grid.y);
  const int P = dims[1], NW = dims[2];
  if (i < P && i >= P - GA[t]) {
    const uint j = ORD[size_t(t) * P + i];
    const uint bit = 1u << (j & 31u);
    atomic_fetch_or_explicit(&LB[size_t(t) * NW + (j >> 5)], bit, memory_order_relaxed);
    atomic_fetch_or_explicit(&UB[size_t(t / 8) * NW + (j >> 5)], bit, memory_order_relaxed);
  }
"""

_K: dict = {}


def _kernel(name: str):
    k = _K.get(name)
    if k is not None:
        return k
    if name == "kidx":
        k = mx.fast.metal_kernel(name="pli_a1_kidx", input_names=["K", "W", "dims"], output_names=["OUT"], source=_KIDX,
                                 ensure_row_contiguous=False)
    elif name == "qidx":
        k = mx.fast.metal_kernel(name="pli_a1_qidx", input_names=["Q", "W"], output_names=["OUT"], source=_QIDX,
                                 ensure_row_contiguous=False)
    elif name == "note":
        k = mx.fast.metal_kernel(name="pli_a1_note", input_names=["KI", "QI", "PAV", "dims"], output_names=["TETE"], source=_NOTE)
    elif name == "marque":
        k = mx.fast.metal_kernel(name="pli_a1_marque", input_names=["ORD", "GA", "dims"], output_names=["LB", "UB"], source=_MARQUE,
                                 atomic_outputs=True)
    else:
        raise KeyError(name)
    _K[name] = k
    return k


def compute_index_keys(keys: mx.array, wk: mx.array, a: int, b: int) -> mx.array:
    """Index keys of positions [a, b) of keys [1, 4, n, 256]: [b - a, 64] f16 (unit rows)."""

    k = _kernel("kidx")
    out = []
    for x in range(a, b, KIDX_CHUNK):
        y = min(b, x + KIDX_CHUNK)
        out.append(k(inputs=[keys, wk, _in.ints([x, y - x])], grid=(64, -(-(y - x) // 8), 1), threadgroup=(64, 1, 1),
                     output_shapes=[((y - x) * DI,)], output_dtypes=[mx.float16])[0].reshape(y - x, DI))
    return out[0] if len(out) == 1 else mx.concatenate(out)


def index_queries(queries: mx.array, wq: mx.array) -> mx.array:
    """Index queries of the T rows of queries [1, 24, T, 256]: [T, 64] f32 (unit rows)."""

    T = int(queries.shape[2])
    return _kernel("qidx")(inputs=[queries, wq], grid=(64, T, 1), threadgroup=(64, 1, 1), output_shapes=[(T * DI,)],
                          output_dtypes=[mx.float32])[0].reshape(T, DI)


def index_keys(cache: Any, keys: mx.array, wk: mx.array, need: int, length: int) -> mx.array | None:
    """Buffer of the layer's index keys, valid over [0, need); extended from K when needed. ``length``: positions of the cache
    outside the round's rows (past them, kept index keys would come from another text: they are dropped)."""

    st = getattr(cache, "pli_kidx", None)
    buf, n = None, 0
    if isinstance(st, list) and len(st) == 3 and st[2] == id(cache) and isinstance(st[0], mx.array):
        buf, n = st[0], max(0, min(int(st[1]), int(length)))
    if n < need:
        nv = compute_index_keys(keys, wk, n, need)
        if buf is None or int(buf.shape[0]) < need:
            cap = -(-(need + 4096) // 8192) * 8192
            nb = mx.zeros((cap, DI), dtype=mx.float16)
            if n:
                nb[:n] = buf[:n]
            buf = nb
        buf[n:need] = nv
        COUNT["index_keys"] += need - n
        n = need
    try:
        cache.pli_kidx = [buf, n, id(cache)]
    except Exception:
        return None
    return buf


def resume(layer: Any, start: int) -> None:
    """Start of a request (family_prefill._family_start): only the index keys of the ``start`` resumed positions remain."""

    st = getattr(layer, "pli_kidx", None)
    if isinstance(st, list) and len(st) == 3:
        try:
            layer.pli_kidx = [st[0], max(0, min(int(st[1]), int(start))), st[2]]
        except Exception:
            pass


def choose(qi: mx.array, buf: mx.array, starts: Sequence[int], reduced: int) -> tuple[mx.array, mx.array, int]:
    """Kept tiles of the rows (index queries qi [Tg, 64], tail starts ``starts``): bits per row LB [Tg, NW] and union per group of
    8 rows UB [ceil(Tg / 8), NW] (uint32, bit j = tile j), NW."""

    Tg = int(qi.shape[0])
    pav = [d // TILE for d in starts]
    P = max(pav)
    nw = -(-P // 32)
    score = _kernel("note")(inputs=[buf, qi, _in.ints(pav), _in.ints([Tg, P])], grid=(32, P, 1), threadgroup=(32, 1, 1),
                          output_shapes=[(Tg * P,)], output_dtypes=[mx.float32])[0]
    order = mx.argsort(score.reshape(Tg, P), axis=1).reshape(-1)
    lb, ub = _kernel("marque")(inputs=[order, _in.ints([kept(p, reduced) for p in pav]), _in.ints([Tg, P, nw])],
                              grid=(P, Tg, 1), threadgroup=(min(P, 256), 1, 1),
                              output_shapes=[(max(8, Tg * nw),), (max(8, -(-Tg // 8) * nw),)], output_dtypes=[mx.uint32, mx.uint32],
                              init_value=0)
    return lb, ub, nw


# ------------------------------------------------------------------------------------------------ read of each row's tiles
_NEXT = r"""
inline int a1_next(int k, int kend, int dmin, const device uint* ub) {
  while (k < kend && k < dmin && ((ub[k >> 10] >> uint((k >> 5) & 31)) & 1u) == 0u) k += 16;
  return k;
}
"""


def _replace(src: str, a: str, b: str) -> str:
    if src.count(a) != 1:
        raise RuntimeError("a1_lsa: the attm kernel changed (" + a.strip()[:60] + ")")
    return src.replace(a, b)


def _source(tree: bool) -> str:
    from tensorfold.pli import attention_attm as A

    s = A._P1
    s = _replace(s, "  const int nrow = rvalid ? NK - T + rt + 1 : 0;\n",
                   "  const int nrow = rvalid ? POS[rt] + 1 : 0;\n"
                   "  const int deb_ = rvalid ? DEB[rt] : 0;\n"
                   "  const size_t lb0_ = size_t(rvalid ? rt : 0) * size_t(dims[6]);\n"
                   + ("  const int base_ = dims[7];\n  const uint an_ = rvalid ? AN[rt] : 0u;\n" if tree else ""))
    i0 = s.index("  // the long slices")
    i1 = s.index("  const size_t kb0 = ")
    s = s[:i0] + ("  // PLI (A1): slices with absolute bounds: [0, S1) in slices of SLEN, then of SP positions up to NK\n"
                  "  const int NA_ = S1 / SLEN;\n"
                  "  const int kbeg = sl < NA_ ? sl * SLEN : S1 + (sl - NA_) * SP;\n"
                  "  const int kend = sl < NA_ ? min(kbeg + SLEN, NK) : min(kbeg + SP, NK);\n"
                  "  const int dmin_ = DM[rg];\n"
                  "  const device uint* ub_ = UB + size_t(rg) * size_t(dims[6]);\n"
                  "  const int kt0_ = a1_next(kbeg, kend, dmin_, ub_);\n") + s[i1:]
    s = _replace(s, "const int key = min((kbeg) + kk, NK - 1);", "const int key = min((kt0_) + kk, NK - 1);")
    s = _replace(s, "  for (int kt = kbeg; kt < kend; kt += BK) {\n",
                   "  int kn_ = kend;\n  for (int kt = kt0_; kt < kend; kt = kn_) {\n    kn_ = a1_next(kt + BK, kend, dmin_, ub_);\n")
    s = _replace(s, "    if (kt + BK < kend) {\n", "    if (kn_ < kend) {\n")
    s = _replace(s, "const int key = min((kt + BK) + kk, NK - 1);", "const int key = min((kn_) + kk, NK - 1);")
    s = _replace(s, "      const bool full = (kt + BK) <= (NK - T + 1);\n",
                   "      const bool okb_ = kt >= deb_ || ((LB[lb0_ + size_t(kt >> 10)] >> uint((kt >> 5) & 31)) & 1u) != 0u;\n"
                   "      const bool full = okb_ && (kt + BK) <= dims[8];\n")
    if tree:
        s = _replace(s, "          s[jt][0] = (k0 < nrow) ? a * c2 : FMIN;\n          s[jt][1] = (k0 + 1 < nrow) ? b * c2 : FMIN;\n",
                       "          s[jt][0] = (okb_ && rvalid && (k0 < base_ || (k0 < NK && ((an_ >> uint(k0 - base_)) & 1u)))) ? a * c2 : FMIN;\n"
                       "          s[jt][1] = (okb_ && rvalid && (k0 + 1 < base_ || (k0 + 1 < NK && ((an_ >> uint(k0 + 1 - base_)) & 1u)))) ? b * c2 : FMIN;\n")
    else:
        s = _replace(s, "          s[jt][0] = (k0 < nrow) ? a * c2 : FMIN;\n          s[jt][1] = (k0 + 1 < nrow) ? b * c2 : FMIN;\n",
                       "          s[jt][0] = (okb_ && k0 < nrow) ? a * c2 : FMIN;\n          s[jt][1] = (okb_ && k0 + 1 < nrow) ? b * c2 : FMIN;\n")
    po = ("#pragma clang loop unroll(full)\n    for (int cj = 0; cj < DH; cj++)\n"
          "      *(device float2*)(PO + slot * D + (d0 + cj) * 8 + fn) = float2(O[cj].thread_elements()[0], O[cj].thread_elements()[1]);\n")
    s = _replace(s, po, "    if (m != FMIN) {\n" + po + "    }\n")          # empty slice for the row: nothing written, skipped at the merge
    s = s.replace("half(bfx(", "h16(bfx(").replace("half(float(qpb", "h16(float(qpb")
    if A.LOAD:
        s = A._with_load(s)
    return s



def _p2_source() -> str:
    from tensorfold.pli import attention_attm as A

    return _replace(A._P2, "  for (int i = 0; i < NSL; i++) {\n    const float w = fast::exp2(PM[slot0 + i] - mm);\n",
                      "  for (int i = 0; i < NSL; i++) {\n    if (PM[slot0 + i] == FMIN) continue;\n"
                      "    const float w = fast::exp2(PM[slot0 + i] - mm);\n")


def _read_kernels(nsg: int, rgn: int, tree: bool):
    from tensorfold.pli import attention_attm as A

    key = ("read", nsg, rgn, tree, A.LOAD)
    if key not in _K:
        pre = "".join("  constexpr int %s = %d;\n" % kv for kv in (("BK", A.BK), ("NS", A.NS), ("NSG", nsg), ("RGN", rgn),
                                                                     ("NACC", 1), ("NBUF", 1)))
        if A.LOAD:
            pre += "  constexpr int LDK = %d;\n" % A.LDK
        names = ["Q", "K", "V", "scale", "dims", "POS", "DEB", "LB", "UB", "DM"] + (["AN"] if tree else [])
        k1 = mx.fast.metal_kernel(name="pli_a1_read%s_%d_%d%s" % ("_tree" if tree else "", nsg, rgn, "_c" if A.LOAD else ""),
                                  input_names=names, output_names=["PM", "PL", "PO"], source=pre + _source(tree),
                                  header=A._HEADER + _NEXT, ensure_row_contiguous=False)
        k2 = _K.get("p2")
        if k2 is None:
            k2 = _K["p2"] = mx.fast.metal_kernel(name="pli_a1_p2", input_names=["PM", "PL", "PO", "dims"], output_names=["OUT"],
                                                 source=_p2_source())
        _K[key] = (k1, k2)
    return _K[key]


def read(qg: mx.array, keys: mx.array, values: mx.array, scale: float, pos: Sequence[int], starts: Sequence[int], s: int,
         lb: mx.array, ub: mx.array, nw: int, an: Sequence[int] | None = None, base: int = 0) -> mx.array | None:
    """Attention of the Tg rows qg [1, 24, Tg, 256] (positions ``pos``): each its own tiles (bits LB), its tail [start, q] (chain) or
    [start, base) and its ancestors of the window ``an`` (tree, window from ``base``); long slices below ``s``."""

    from tensorfold.pli import attention_attm as A

    Tg = int(qg.shape[2])
    if qg.dtype != mx.bfloat16 or keys.dtype != mx.bfloat16 or values.dtype != mx.bfloat16 or Tg < 1 or Tg > A.MAX_ROWS \
            or int(qg.shape[1]) != H or int(qg.shape[3]) != DH or int(keys.shape[1]) != HKV or int(keys.shape[3]) != DH:
        return None
    tree = an is not None
    nk = (base + len(an)) if tree else (max(pos) + 1)
    if int(keys.shape[2]) < nk or int(values.shape[2]) < nk or s % SLICE or s > min(starts):
        return None
    nrt = -(-(G * Tg) // 8)
    tpg = max(min(nrt, A.TPG_MAX), A.TPG_LOAD.get(nrt, 0))
    nsg = tpg * A.NS
    rgn = -(-nrt // tpg)
    if rgn > 1 and tpg * 8 != 8 * G:                       # several threadgroups: 8 rows each (UB union per 8 rows)
        return None
    na = s // SLICE
    nsl = na + -(-(nk - s) // SMALL_SLICE)
    dm = [min(starts[r * 8:(r + 1) * 8]) for r in range(rgn)] if rgn > 1 else [min(starts)]
    fullb = base if tree else (min(pos) + 1)
    dims = _in.ints([nk, Tg, nsl, SLICE, s, SMALL_SLICE, nw, base, fullb])
    sa = _in.floats([float(scale)] * 8)
    k1, k2 = _read_kernels(nsg, rgn, tree)
    inputs = [qg, keys, values, sa, dims, _in.ints(pos), _in.ints(starts), lb, ub, _in.ints(dm)]
    if tree:
        inputs.append(_in.ints(an, dtype=mx.uint32))
    pm, pl, po = k1(inputs=inputs, grid=(32 * nsg, nsl, HKV * rgn), threadgroup=(32 * nsg, 1, 1),
                    output_shapes=[(H * Tg * nsl,), (H * Tg * nsl,), (H * Tg * nsl * DH,)],
                    output_dtypes=[mx.float32, mx.float32, mx.float32])
    COUNT["calls"] += 1
    COUNT["rows"] += Tg
    return k2(inputs=[pm, pl, po, dims], grid=(32, H, Tg), threadgroup=(32, 1, 1), output_shapes=[(1, H, Tg, DH)],
              output_dtypes=[qg.dtype])[0]


def _groups(red: Sequence[bool], starts: Sequence[int], order: Sequence[int]) -> list:
    """Groups of rows (indices of ``order``, 32 at most): (None, rows) complete, or (S, rows) reduced with the same S."""

    out: list = []
    for i in order:
        key = (starts[i] // SLICE) * SLICE if red[i] else None
        if out and out[-1][0] == key and len(out[-1][1]) < 32:
            out[-1][1].append(i)
        else:
            out.append((key, [i]))
    return out


def chain(queries: mx.array, keys: mx.array, values: mx.array, cache: Any, scale: float, reduced: int,
          threshold: int) -> mx.array | None:
    """The T rows of a chain round (positions L - T to L - 1); None if a kernel refuses (the caller then reads in full)."""

    from tensorfold.pli import attention_attm as A

    T, L = int(queries.shape[2]), int(keys.shape[2])
    p0 = L - T
    wq, wk = lsa_map()[int(cache.pli_layer)]
    pos = list(range(p0, L))
    starts = [tail_start(q) for q in pos]
    red = [q >= threshold for q in pos]
    if any(red):
        need = max(d for d, r in zip(starts, red) if r)
        buf = index_keys(cache, keys, wk, need, p0)
        if buf is None:
            return None
        qi = index_queries(queries, wq)
    outs = []
    for s, rows in _groups(red, starts, range(T)):
        t, t1 = rows[0], rows[-1] + 1
        qg = queries[:, :, t:t1]
        if s is None:
            end = p0 + t1
            out = A.sdpa(qg, keys[:, :, :end], values[:, :, :end], end, scale)
        else:
            lb, ub, nw = choose(qi[t:t1], buf, starts[t:t1], reduced)
            out = read(qg, keys, values, scale, pos[t:t1], starts[t:t1], s, lb, ub, nw)
        if out is None:
            return None
        outs.append(out)
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


def tree(queries: mx.array, keys: mx.array, values: mx.array, cache: Any, scale: float, parents: Sequence[int], reduced: int,
         threshold: int, complete) -> mx.array | None:
    """Tree window of T nodes (base = L - T): the node at depth d is at position base + d and chooses on its own query; it reads
    its tiles, [start, base) and its ancestors. ``complete``: complete tree attention (nodes below the threshold)."""

    from tensorfold.pli import attention_attm as A

    T, L = int(queries.shape[2]), int(keys.shape[2])
    base = L - T
    if len(parents) != T or T > A.MAX_ROWS:
        return None
    depth = [0] * T
    for i in range(1, T):
        p = int(parents[i])
        if not 0 <= p < i:
            return None
        depth[i] = depth[p] + 1
    pos = [base + d for d in depth]
    starts = [tail_start(q) for q in pos]
    red = [q >= threshold for q in pos]
    outputs: dict = {}
    if not all(red):
        out = complete(queries, keys, values, L, scale, parents)
        if out is None:
            return None
        if not any(red):
            return out
        for i in range(T):
            if not red[i]:
                outputs[i] = out[:, :, i:i + 1]
    wq, wk = lsa_map()[int(cache.pli_layer)]
    need = max(d for d, r in zip(starts, red) if r)
    buf = index_keys(cache, keys, wk, need, base)
    if buf is None:
        return None
    qi = index_queries(queries, wq)
    an = A.ancestors(parents)
    order = sorted((i for i in range(T) if red[i]), key=lambda i: (starts[i], i))
    for s, rows in _groups(red, starts, order):
        whole = rows == list(range(T))
        ix = None if whole else mx.array(rows, dtype=mx.int32)
        qg = queries if whole else mx.take(queries, ix, axis=2)
        qig = qi if whole else mx.take(qi, ix, axis=0)
        lb, ub, nw = choose(qig, buf, [starts[i] for i in rows], reduced)
        out = read(qg, keys, values, scale, [pos[i] for i in rows], [starts[i] for i in rows], s, lb, ub, nw,
                   an=[an[i] for i in rows], base=base)
        if out is None:
            return None
        if whole:
            return out
        for k, i in enumerate(rows):
            outputs[i] = out[:, :, k:k + 1]
    return mx.concatenate([outputs[i] for i in range(T)], axis=2)


__all__ = ["ACTIVE", "COUNT", "MAX_TILES", "PREFILL_ONLY", "chain", "choose", "compute_index_keys", "has_layer", "index_keys",
           "index_queries", "kept", "lsa_map", "read", "resume", "set_map", "tail_start", "tree"]
