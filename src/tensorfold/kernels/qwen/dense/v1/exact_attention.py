"""Match serial attention bits by grouping queries only when their one-query calls share MLX's kernel variant and key partition."""

from __future__ import annotations

import os
from typing import Any

import mlx.core as mx

from tensorfold.kernels.qwen.dense.v1 import prompt_attention

EXACT_MAX_QUERIES = 16
# PLI_ATTENTION_ATTEC=1 (PLI-U 2.6.0): decode attention (2 to 16 rows, 1100 keys and more) in a single launch at equal bits, see
# pli/attention_decode.py (27B: identical texts, decode + 4.5 to + 13 %). Off by default.
ATTEC = os.environ.get("PLI_ATTENTION_ATTEC", "0") == "1"
# PLI_ATTENTION_ATTM=1 (PLI-U 3.0.0, V3): custom decode attention by 8 x 8 tiles for every decode row (1 to 32 rows), see
# pli/attention_attm.py (27B: decode + 27 %, same text from one run to the next). Off by default.
ATTM = os.environ.get("PLI_ATTENTION_ATTM", "0") == "1"
# Group queries with identical one-query arithmetic; disable when MLX dispatch differs.
# PLI ("split computation" merge, 10/10): PLI_ATTENTION_PER_ROW=1 makes one call of MLX's kernel per row instead of batches.
GROUP_QUERIES = os.environ.get("PLI_ATTENTION_PER_ROW", "0") != "1"
# PLI (10/10): PLI_ATTENTION_PER_ROW=N (N > 1) switches to one call per row only from N context keys
# (measured: the per-row call gains from 25,000 to 66,000 tokens and loses a little below 7,000). Same bits in both cases.
try:
    PER_ROW_THRESHOLD = int(os.environ.get("PLI_ATTENTION_PER_ROW", "0") or 0)
except ValueError:
    PER_ROW_THRESHOLD = 0
if PER_ROW_THRESHOLD > 1:
    GROUP_QUERIES = True
_STOCK: Any = None

# MLX 0.31.2 skips masked tail keys without changing key slots; grouping is exact only within one kernel regime.
_REGIME_POINTS = (1024, 1025, 4096, 8193, 16384, 32769, 65536, 65537)

# PLI, rank 2: PLI_ATTENTION sends the windows of 1 to 128 rows (one-row steps included) through a PLI attention
# where a row has the same bits alone or in a window (the bits differ from MLX's):
#   gemm: MLX's matrix products in a single regime (tensorfold/pli/attention_gemm.py);
#   m1  : Metal kernel with 8 x 8 matrices (tensorfold/pli/attention_m1.py).
PLI_ATTENTION = os.environ.get("PLI_ATTENTION", "").strip().lower()
if PLI_ATTENTION in ("0", "no", "off", "non"):
    PLI_ATTENTION = ""
if PLI_ATTENTION and PLI_ATTENTION not in ("gemm", "m1"):
    raise ValueError(f"PLI_ATTENTION={PLI_ATTENTION}: gemm or m1")


# PLI ("split computation" merge no. 2, 10/10): reduced read of the old context with a map made by the GPU within the round.
# PLI_ATTENTION_REDUCED=20: in decode (1 to 16 rows), past PLI_ATTENTION_REDUCED_THRESHOLD keys, the GPU computes the scores
# of all the old keys, keeps the 20 % of 32-key tiles with the highest score (per key/value head, maximum over the tile and
# over the rows), then calls the usual kernel on these keys + the tail of 1,024 tokens. The FULL layers stay complete.
# The text changes (class V3): off by default (0). The layer is read on the cache (attribute pli_layer, set by the family).
def _int_env(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, str(default)) or default)
    except ValueError:
        return default


REDUCED = max(0, min(90, _int_env("PLI_ATTENTION_REDUCED", 0)))
REDUCED_THRESHOLD = max(4096, _int_env("PLI_ATTENTION_REDUCED_THRESHOLD", 20000))
REDUCED_FULL_LAYERS = tuple(int(x) for x in os.environ.get("PLI_ATTENTION_REDUCED_FULL_LAYERS", "3,7,59").split(",") if x.strip())
TAIL, TILE = 1024, 32
REDUCED_MAX_ROWS = 16        # decode rounds only; the prompt prefill stays complete
REDUCED_MIN_ROWS = max(1, _int_env("PLI_ATTENTION_REDUCED_MIN_ROWS", 2))   # reparti-red1 (10/10): at 1 row the reduction costs more than the full one
COUNT = {"reduced": 0, "full": 0, "attm": 0}
REDUCED_ATTM = os.environ.get("PLI_ATTENTION_REDUCED_ATTM", "1") == "1"   # 0: old path (without attm) on the reduced layers


# PLI (21/10/2026, measured: choosing the tiles costs 10.5 ms per round at 22K, 22.6 at 60K): PLI_REDUCED_ROUNDS=K keeps a layer's choice
# for K rounds (the old keys do not change); the keys that entered the old context since the choice stay read in full with the
# tail. The choice is redone at the K-th round, or if the old context moved back (rollback, other request). 1 = original (choice every round).
REDUCED_ROUNDS = max(1, _int_env("PLI_REDUCED_ROUNDS", 1))


def _reduce(queries: mx.array, keys: mx.array, values: mx.array, cache: Any = None) -> tuple[mx.array, mx.array]:
    """Kept keys and values: top tiles of the old context (in order), then the whole tail."""

    import math

    kvh, n, d = int(keys.shape[1]), int(keys.shape[2]), int(keys.shape[3])
    rows = int(queries.shape[2])
    lanes = (int(queries.shape[1]) // kvh) * rows
    tiles = (n - rows - TAIL) // TILE
    start = tiles * TILE
    keep = max(1, math.ceil(REDUCED / 100 * tiles))
    memo = getattr(cache, "pli_choice", None) if (cache is not None and REDUCED_ROUNDS > 1) else None
    if memo is not None and memo[2] < REDUCED_ROUNDS and memo[0] <= start and memo[3] == kvh:
        memo[2] += 1
        d0 = memo[0]
        kt = [mx.take(keys[0, h], memo[1][h], axis=0) for h in range(kvh)]
        vt = [mx.take(values[0, h], memo[1][h], axis=0) for h in range(kvh)]
        return (mx.concatenate([mx.stack(kt)[None], keys[:, :, d0:]], axis=2),
                mx.concatenate([mx.stack(vt)[None], values[:, :, d0:]], axis=2))
    # Everything is done head by head (taken from fix 55ab43ca of 10/10, bench repository): in the engine the keys are a slice of the
    # cache buffer; flattening them copied the whole array at each call (essai-reduite-pieces: 7.6 ms against 0.75 ms per head, same bits).
    qh = queries.reshape(kvh, lanes, d)
    offs = mx.arange(TILE)[None, :]
    kt, vt, ixs = [], [], []
    for h in range(kvh):
        scores = qh[h] @ mx.swapaxes(keys[0, h, :start], -1, -2)                                # [rows, old keys]
        top = mx.max(scores.reshape(lanes, tiles, TILE), axis=(0, 2)).astype(mx.float32)
        chosen = mx.sort(mx.argsort(top)[tiles - keep:])
        ix = (chosen[:, None] * TILE + offs).reshape(-1)
        ixs.append(ix)
        kt.append(mx.take(keys[0, h], ix, axis=0))
        vt.append(mx.take(values[0, h], ix, axis=0))
    if cache is not None and REDUCED_ROUNDS > 1:
        try:
            cache.pli_choice = [start, ixs, 1, kvh]
        except Exception:
            pass
    return (mx.concatenate([mx.stack(kt)[None], keys[:, :, start:]], axis=2),
            mx.concatenate([mx.stack(vt)[None], values[:, :, start:]], axis=2))


# PLI (23/10/2026): reduced prefill made the text depend on the drafts: choice of tiles on the queries of all
# the rows of the round (drafts included), reduction only at PLI_ATTENTION_REDUCED_MIN_ROWS rows and more, choice kept PLI_REDUCED_ROUNDS rounds
# whose bounds follow the acceptances. PLI_REDUCED_ANCHOR=N (V3): the row at position q reads the tiles chosen at the anchor
# a = max(N * (q // N), start of the request), on the query of row a alone; old context bounded by the anchor; reduced at any round
# size, past PLI_ATTENTION_REDUCED_THRESHOLD positions. A choice made on a refused draft row is redone: the row comes back
# at the head of the next round. The text then depends only on the written tokens (same text with or without draft, read in full or resumed).
ANCHOR = max(0, _int_env("PLI_REDUCED_ANCHOR", 0))
# PLI (02/11/2026, S3b, V3): PLI_S3B=1: prompt chunks of at most PLI_S3B_ROWS rows also go through the anchored reduced attention
# (production trace of 01/11: chunks of 256 to 511 rows at 64K and more read at 130 tokens/s, attention 31 to 47 % of it, s3b-lbench).
# Anchors bounded at the start of the request's prefill (pli_prefill_start, set by family_prefill); full layers read whole. 0 = original.
S3B = os.environ.get("PLI_S3B", "0") == "1"
S3B_ROWS = max(1, _int_env("PLI_S3B_ROWS", 2047))
S3B_THRESHOLD = max(4096, _int_env("PLI_S3B_THRESHOLD", 65536))     # S3b past this many positions (measured: +10 % at 30K and 60K, +35 % at 95K)


_REASONS: dict = {}


def _reason(name: str) -> None:
    import sys
    _REASONS[name] = _REASONS.get(name, 0) + 1
    if __import__("os").environ.get("PLI_ANCHOR_LOG") == "1" and (sum(_REASONS.values()) % 250 == 0
                                                                   or (not name.startswith("anchored") and _REASONS[name] <= 30)):
        print("[pli] anchor: %s (last: %s)" % (sorted(_REASONS.items()), name), file=sys.stderr, flush=True)


def _anchor_choice(q1: mx.array, keys: mx.array, start: int) -> list:
    """Kept tiles per key/value head, chosen on the query of a single row q1 [1, H, 1, d] and the keys [0, start)."""

    import math

    kvh, d = int(keys.shape[1]), int(keys.shape[3])
    g = int(q1.shape[1]) // kvh
    tiles = start // TILE
    keep = max(1, math.ceil(REDUCED / 100 * tiles))
    qh = q1.reshape(kvh, g, d)
    offs = mx.arange(TILE)[None, :]
    ixs = []
    for h in range(kvh):
        scores = qh[h] @ mx.swapaxes(keys[0, h, :start], -1, -2)
        top = mx.max(scores.reshape(g, tiles, TILE), axis=(0, 2)).astype(mx.float32)
        chosen = mx.sort(mx.argsort(top)[tiles - keep:])
        ixs.append((chosen[:, None] * TILE + offs).reshape(-1))
    return ixs


def _anchored(queries: mx.array, keys: mx.array, values: mx.array, cache: Any, scale: float) -> mx.array | None:
    """Attention of the round's T rows by groups of the same anchor; None if the attm kernel does not take a group (full fallback)."""

    from tensorfold.pli import attention_attm

    T, L = int(queries.shape[2]), int(keys.shape[2])
    p0 = L - T
    req_start = getattr(cache, "pli_request_start", None)       # prompt length, set at the start of prefill (family_prefill)
    if req_start is None:
        req_start = p0
        try:
            cache.pli_request_start = p0
            cache.pli_anchors = {}
        except Exception:
            return None
    bound = req_start
    if p0 < req_start:
        pstart = getattr(cache, "pli_prefill_start", None)
        if not S3B or T > S3B_ROWS or pstart is None or p0 < pstart or p0 + T > req_start \
                or getattr(cache, "pli_layer", None) in REDUCED_FULL_LAYERS:
            _reason("prompt")
            return None                                          # prompt rows: the prefill stays complete
        if p0 < S3B_THRESHOLD:
            _reason("prompt_below_threshold")
            return None
        bound = pstart                                           # S3b: anchors bounded at the start of the request's prefill
        a0 = max(ANCHOR * (p0 // ANCHOR), bound)
        if a0 < p0 and a0 not in (getattr(cache, "pli_anchors", None) or {}):
            bound = p0                                           # previous chunk read in full: anchors bounded at the start of the chunk
        if max(ANCHOR * ((p0 + T - 1) // ANCHOR), bound) < REDUCED_THRESHOLD:
            _reason("prompt_short")
            return None
    memo = getattr(cache, "pli_anchors", None)
    if memo is None:
        memo = {}
        try:
            cache.pli_anchors = memo
        except Exception:
            return None
    layer = getattr(cache, "pli_layer", None)
    outs, t = [], 0
    while t < T:
        a = max(ANCHOR * ((p0 + t) // ANCHOR), bound)
        t1 = t + 1                              # groups of the same anchor, 32 rows at most (attm kernel): copy rounds included
        while t1 < T and t1 - t < 32 and max(ANCHOR * ((p0 + t1) // ANCHOR), bound) == a:
            t1 += 1
        end = p0 + t1
        qg = queries[:, :, t:t1]
        if a < REDUCED_THRESHOLD or layer in REDUCED_FULL_LAYERS:
            out = attention_attm.sdpa(qg, keys[:, :, :end], values[:, :, :end], end, scale)
        else:
            start = max(TILE, ((a - TAIL) // TILE) * TILE)
            ixs = None
            if a < p0:
                m = memo.get(a)
                if m is not None and m[0] == start:
                    ixs = m[1]
            if ixs is None:
                ta = a - p0
                if not 0 <= ta < T:
                    _reason("anchor_missing")
                    return None
                ixs = _anchor_choice(queries[:, :, ta:ta + 1], keys, start)
                memo[a] = (start, ixs)
                for old in [x for x in memo if x < a - ANCHOR]:
                    del memo[old]
            kvh = int(keys.shape[1])
            kt = mx.stack([mx.take(keys[0, h], ixs[h], axis=0) for h in range(kvh)])[None]
            vt = mx.stack([mx.take(values[0, h], ixs[h], axis=0) for h in range(kvh)])[None]
            kr = mx.concatenate([kt, keys[:, :, start:end]], axis=2)
            vr = mx.concatenate([vt, values[:, :, start:end]], axis=2)
            out = attention_attm.sdpa(qg, kr, vr, int(kr.shape[2]), scale)
        if out is None:
            _reason("attm_refused")
            return None
        outs.append(out)
        t = t1
    COUNT["reduced"] += 1
    _reason(("s3b_%d_groups" if bound != req_start else "anchored_%d_groups") % len(outs))
    return outs[0] if len(outs) == 1 else mx.concatenate(outs, axis=2)


def _anchored_tree(queries: mx.array, keys: mx.array, values: mx.array, cache: Any, scale: float, parents) -> mx.array | None:
    """PLI (item D, 24/10/2026): anchored reduced prefill for a tree window (attm with an ancestor mask). Node i is at position
    p0 + depth(i); it reads the tiles chosen at its anchor a = max(ANCHOR x (position // ANCHOR), start of the request), on the query of
    ITS ancestor at position a (unique: a node has only one path), or on the kept choice if a is already written. The choices made on
    nodes of the round wait for validation (pli_anchors_pending): only the one of the accepted path enters the kept choice (Qwen35Family._commit),
    like a refused chain row whose choice is redone. None: the window goes through the full tree attention."""

    from tensorfold.pli import attention_attm

    T, L = int(queries.shape[2]), int(keys.shape[2])
    p0 = L - T
    if len(parents) != T:
        return None
    req_start = getattr(cache, "pli_request_start", None)
    if req_start is None or p0 < req_start:
        _reason("tree_prompt")
        return None
    memo = getattr(cache, "pli_anchors", None)
    if memo is None:
        return None
    try:
        pending = {}
        cache.pli_anchors_pending = pending
    except Exception:
        return None
    layer = getattr(cache, "pli_layer", None)
    depth = [0] * T
    for i in range(1, T):
        depth[i] = depth[int(parents[i])] + 1
    def ancestor(i: int, d: int) -> int:
        while depth[i] > d:
            i = int(parents[i])
        return i
    groups: dict = {}
    for i in range(T):
        pos = p0 + depth[i]
        a = max(ANCHOR * (pos // ANCHOR), req_start)
        if a < REDUCED_THRESHOLD or layer in REDUCED_FULL_LAYERS:
            key = ("full",)
        elif a < p0:
            key = ("kept", a)
        else:
            key = ("row", a, ancestor(i, a - p0))
        groups.setdefault(key, []).append(i)
    outputs = {}
    for key, rows in groups.items():
        tree = attention_attm.sdpa_tree_exact if os.environ.get("PLI_TREE_EXACT", "0") == "1" else attention_attm.sdpa_tree
        if key[0] == "full":
            out = tree(queries, keys, values, L, scale, parents)
        else:
            a = key[1]
            start = max(TILE, ((a - TAIL) // TILE) * TILE)
            if key[0] == "kept":
                m = memo.get(a)
                if m is None or m[0] != start:
                    _reason("tree_anchor_missing")
                    return None
                ixs = m[1]
            else:
                r = key[2]
                ixs = _anchor_choice(queries[:, :, r:r + 1], keys, start)
                pending[(a, r)] = (start, ixs)
            kvh = int(keys.shape[1])
            kt = mx.stack([mx.take(keys[0, h], ixs[h], axis=0) for h in range(kvh)])[None]
            vt = mx.stack([mx.take(values[0, h], ixs[h], axis=0) for h in range(kvh)])[None]
            kr = mx.concatenate([kt, keys[:, :, start:L]], axis=2)
            vr = mx.concatenate([vt, values[:, :, start:L]], axis=2)
            out = tree(queries, kr, vr, int(kr.shape[2]), scale, parents)
        if out is None:
            _reason("tree_attm_refused")
            return None
        for i in rows:
            outputs[i] = out
    COUNT["reduced"] += 1
    _reason("tree_anchored_%d_groups" % len(groups))
    if len(groups) == 1:
        return outputs[0]
    return mx.concatenate([outputs[i][:, :, i:i + 1] for i in range(T)], axis=2)


def commit_anchors(layers: list, path: list) -> None:
    """PLI (item D): after the validation of a tree round, the choices made on the nodes of the accepted path enter the kept choice."""

    accepted = set(int(r) for r in path)
    for c in layers:
        pending = getattr(c, "pli_anchors_pending", None)
        if not pending:
            continue
        memo = getattr(c, "pli_anchors", None)
        if memo is not None:
            for (a, r), v in pending.items():
                if r in accepted:
                    memo[a] = v
            if memo:
                top = max(memo)
                for old in [x for x in memo if x < top - 2 * ANCHOR]:
                    del memo[old]
        c.pli_anchors_pending = {}


def _one(queries: mx.array, keys: mx.array, values: mx.array, scale: float, mask: Any, t: int, T: int,
         L: int) -> mx.array:
    n = L - T + t + 1
    m = mask[..., t:t + 1, :n] if isinstance(mask, mx.array) else None
    return mx.fast.scaled_dot_product_attention(
        queries[:, :, t:t + 1], keys[:, :, :n], values[:, :, :n], scale=scale, mask=m)


# PLI (16/10/2026, benches only): PLI_ATTENTION_TUNING_FILE=/path/file changes the kept share on the fly ("reduced=N", or "reduite=N"
# before 3.7.0; 0 = full attention), without restarting the server or reading the context again; the file is read again when its date changes.
TUNING_FILE = os.environ.get("PLI_ATTENTION_TUNING_FILE", "")
_TUNING_SEEN = [0.0]


def _tuning() -> None:
    global REDUCED
    try:
        date = os.stat(TUNING_FILE).st_mtime
    except OSError:
        return
    if date == _TUNING_SEEN[0]:
        return
    _TUNING_SEEN[0] = date
    import sys

    try:
        v = dict(x.split("=", 1) for x in open(TUNING_FILE).read().split())
        REDUCED = max(0, min(90, int(v.get("reduced", v.get("reduite", REDUCED)))))
    except Exception as error:  # noqa: BLE001
        print("[pli] live tuning unreadable: %s" % error, file=sys.stderr, flush=True)
        return
    print("[pli] live tuning: reduced attention at %d %%" % REDUCED, file=sys.stderr, flush=True)


def exact_sdpa(queries: mx.array, keys: mx.array, values: mx.array, cache: Any, scale: float,
               mask: Any, sinks: Any = None) -> mx.array:
    T = int(queries.shape[2])
    if TUNING_FILE:
        _tuning()
    T = int(queries.shape[2])
    if REDUCED and ANCHOR and ATTM and T >= 1 and sinks is None and not hasattr(cache, "bits") \
            and int(queries.shape[0]) == 1 and int(keys.shape[2]) - T >= REDUCED_THRESHOLD - ANCHOR \
            and (mask is None or (isinstance(mask, str) and mask == "causal")) and queries.dtype == mx.bfloat16 \
            and keys.dtype == mx.bfloat16 and getattr(cache, "pli_layer", None) is not None:
        out = _anchored(queries, keys, values, cache, scale)
        if out is not None:
            return out
    if REDUCED and ANCHOR and 1 <= T <= 64 and int(keys.shape[2]) >= REDUCED_THRESHOLD:
        _reason("off_path_T%d" % T if int(queries.shape[0]) == 1 else "off_path_batch")
    if REDUCED and not ANCHOR and REDUCED_MIN_ROWS <= T <= REDUCED_MAX_ROWS and sinks is None and not hasattr(cache, "bits") and int(queries.shape[0]) == 1 \
            and int(keys.shape[2]) >= REDUCED_THRESHOLD and (mask is None or (isinstance(mask, str) and mask == "causal")) \
            and int(queries.shape[1]) % int(keys.shape[1]) == 0:
        layer = getattr(cache, "pli_layer", None)
        if layer is not None and layer not in REDUCED_FULL_LAYERS:
            COUNT["reduced"] += 1
            keys, values = _reduce(queries, keys, values, cache)
            # PLI (16/10/2026, V3): the reduction goes through attm when attm is on. Before, the shortened keys no longer passed attm's
            # test (cache.offset == length of the keys): the reduced layer lost attm. The kept keys form a contiguous array
            # that ends with the tail (including the round's T rows): row t sees L - T + t + 1 keys, as attm expects.
            if ATTM and REDUCED_ATTM and T <= 32 and queries.dtype == mx.bfloat16 and keys.dtype == mx.bfloat16:
                from tensorfold.pli import attention_attm

                out = attention_attm.sdpa(queries, keys, values, int(keys.shape[2]), scale)
                if out is not None:
                    COUNT["attm"] += 1
                    return out
        else:
            COUNT["full"] += 1
        if (COUNT["reduced"] + COUNT["full"]) % 20000 == 0:
            import sys
            print("[pli] reduced attention at %d %%: %d calls reduced (%d by attm), %d left full" % (REDUCED, COUNT["reduced"],
                  COUNT["attm"], COUNT["full"]), file=sys.stderr, flush=True)
    if PLI_ATTENTION and T <= 128 and sinks is None and not hasattr(cache, "bits") \
            and ((mask is None and T == 1) or (isinstance(mask, str) and mask == "causal")) and int(queries.shape[0]) == 1 \
            and int(queries.shape[-1]) % 64 == 0 and queries.dtype == mx.bfloat16 and keys.dtype == mx.bfloat16 \
            and values.dtype == mx.bfloat16:
        if PLI_ATTENTION == "gemm":
            from tensorfold.pli import attention_gemm

            return attention_gemm.sdpa(queries, keys, values, scale, cache=cache)
        from tensorfold.pli import attention_m1

        return attention_m1.sdpa(queries, keys, values, scale)
    if T > EXACT_MAX_QUERIES and int(queries.shape[-1]) == 256 \
            and isinstance(mask, str) and mask == "causal" and sinks is None \
            and not hasattr(cache, "bits"):
        return prompt_attention.attend(queries, keys, values, scale)      # prompt chunks: bounded memory
    # PLI_ATTENTION_ATTM: EVERY full decode attention row (1 to 16 rows) goes through the same arithmetic, including at 1 row (mask None)
    # and at 2 rows: the choice depends only on the keys and the shape, never on the number of rows of the round (pli-attmd, 13/10/2026).
    if ATTM and sinks is None and not hasattr(cache, "bits") and int(queries.shape[0]) == 1 and T <= EXACT_MAX_QUERIES \
            and ((isinstance(mask, str) and mask == "causal") or (T == 1 and mask is None)):
        bk, bv = getattr(cache, "keys", None), getattr(cache, "values", None)
        if bk is not None and bv is not None and getattr(cache, "offset", None) is not None and int(cache.offset) == int(keys.shape[2]) \
                and bk.ndim == 4 and bk.shape[:2] == keys.shape[:2] and bk.shape[3] == keys.shape[3]:
            from tensorfold.pli import attention_attm

            out = attention_attm.sdpa(queries, bk, bv, int(keys.shape[2]), scale)
            if out is not None:
                return out
    if T < 2 or T > EXACT_MAX_QUERIES or sinks is not None or hasattr(cache, "bits"):
        return _STOCK(queries, keys, values, cache, scale, mask, sinks)
    L = int(keys.shape[2])
    if ATTEC and isinstance(mask, str) and mask == "causal" and int(queries.shape[0]) == 1:
        bk, bv = getattr(cache, "keys", None), getattr(cache, "values", None)
        if bk is not None and bv is not None and getattr(cache, "offset", None) is not None and int(cache.offset) == L \
                and bk.ndim == 4 and bk.shape[:2] == keys.shape[:2] and bk.shape[3] == keys.shape[3]:
            from tensorfold.pli import attention_decode

            out = attention_decode.sdpa(queries, bk, bv, L, scale)
            if out is not None:
                return out
    heads, kv_heads = int(queries.shape[1]), int(keys.shape[1])
    group = 1
    if GROUP_QUERIES and not (PER_ROW_THRESHOLD > 1 and L >= PER_ROW_THRESHOLD) and not isinstance(mask, mx.array) and heads % kv_heads == 0:
        group = max(1, min(8, 32 // (heads // kv_heads)))
    outs = []
    t = 0
    while t < T:
        g = min(group, T - t)
        first, last = L - T + t + 1, L - T + t + g
        if g > 1 and not any(first < p <= last for p in _REGIME_POINTS):
            outs.append(mx.fast.scaled_dot_product_attention(
                queries[:, :, t:t + g], keys[:, :, :last], values[:, :, :last], scale=scale, mask="causal"))
        else:
            outs.extend(_one(queries, keys, values, scale, mask, j, T, L) for j in range(t, t + g))
        t += g
    return mx.concatenate(outs, axis=2)


def install() -> None:
    """Route the Qwen3-Next / Qwen3.5 full-attention call through ``exact_sdpa``. Idempotent."""

    global _STOCK
    import mlx_lm.models.qwen3_next as qn

    if _STOCK is None:
        _STOCK = qn.scaled_dot_product_attention
    qn.scaled_dot_product_attention = exact_sdpa


__all__ = ["EXACT_MAX_QUERIES", "exact_sdpa", "install"]
