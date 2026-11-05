"""PLI (A1, 03/11/2026): reduced attention "anchor 1" scored by the LSA index (pli/a1_lsa.py, PLI_A1_LSA_MAP).

- setting off: nothing goes through A1 (no index keys on the cache);
- split invariance: same bits in rounds of 8, of 1, of mixed sizes, in a prompt chunk (S3b);
- resume: same bits with index keys rebuilt at once (fresh cache, copy, rollback + start of a request);
- tree: a chain-shaped tree has the chain's bits; read close to a masked f32 attention."""

from __future__ import annotations

import os

import numpy as np
import pytest

mx = pytest.importorskip("mlx.core")

from mlx_lm.models.cache import KVCache  # noqa: E402

from tensorfold.kernels.qwen.dense.v1 import exact_attention as EA  # noqa: E402
from tensorfold.pli import a1_lsa as A1  # noqa: E402
from tensorfold.pli import attention_attm as AT  # noqa: E402

LAYER, THRESHOLD, SCALE = 11, 2048, 256 ** -0.5


def _map(seed=7):
    r = np.random.default_rng(seed)
    return {c: (mx.array((r.standard_normal((768, 64)) / 28).astype(np.float32)),
                mx.array((r.standard_normal((768, 64)) / 28).astype(np.float32))) for c in (3, 7, LAYER)}


@pytest.fixture
def a1(monkeypatch):
    monkeypatch.setattr(EA, "REDUCED", 10)
    monkeypatch.setattr(EA, "ANCHOR", 32)
    monkeypatch.setattr(EA, "ATTM", True)
    monkeypatch.setattr(EA, "REDUCED_THRESHOLD", THRESHOLD)
    monkeypatch.setattr(A1, "ACTIVE", True)
    monkeypatch.setattr(A1, "SLICE", 512)
    monkeypatch.setattr(A1, "SMALL_SLICE", 128)
    A1.set_map(_map())
    yield
    A1.set_map(None)


def _data(n, seed=0, rows=64):
    mx.random.seed(seed)
    kk = (mx.random.normal((1, 4, n, 256)) * 0.6).astype(mx.bfloat16)
    vv = mx.random.normal((1, 4, n, 256)).astype(mx.bfloat16)
    qq = mx.random.normal((1, 24, rows, 256)).astype(mx.bfloat16)
    mx.eval(kk, vv, qq)
    return kk, vv, qq


def _cache(kk, vv, layer=LAYER):
    c = KVCache()
    c.keys, c.values = kk, vv
    c.pli_layer, c.pli_request_start, c.pli_anchors = layer, 0, {}
    return c


def _write(c, kk, vv, qq, q0, p0, T):
    """Decode round of the rows at positions [p0, p0 + T) (queries qq[:, :, p - q0])."""

    c.offset = p0 + T
    out = EA.exact_sdpa(qq[:, :, p0 - q0:p0 - q0 + T], kk[:, :, :p0 + T], vv[:, :, :p0 + T], c, SCALE, "causal" if T > 1 else None)
    mx.eval(out)
    return out


def _bits(x):
    return np.array(x.view(mx.uint16))


def _run(c, kk, vv, qq, q0, sizes, base=None):
    """Rounds of sizes ``sizes`` from position q0; queries indexed from ``base`` (q0 by default)."""

    p, outs = q0, []
    for T in sizes:
        outs.append(_write(c, kk, vv, qq, q0 if base is None else base, p, T))
        p += T
    return _bits(mx.concatenate(outs, axis=2))


def test_off_nothing(monkeypatch):
    monkeypatch.setattr(EA, "REDUCED", 10)
    monkeypatch.setattr(EA, "ANCHOR", 32)
    monkeypatch.setattr(EA, "ATTM", True)
    monkeypatch.setattr(EA, "REDUCED_THRESHOLD", THRESHOLD)
    monkeypatch.setattr(A1, "ACTIVE", False)
    kk, vv, qq = _data(4200)
    c = _cache(kk, vv)
    before = dict(A1.COUNT)
    _run(c, kk, vv, qq, 4000, [8, 8])
    assert not hasattr(c, "pli_kidx") and A1.COUNT == before and c.pli_anchors


def test_index_keys_per_position(a1, monkeypatch):
    kk, _, _ = _data(3000)
    wk = A1.lsa_map()[LAYER][1]
    whole = A1.compute_index_keys(kk, wk, 0, 3000)
    monkeypatch.setattr(A1, "KIDX_CHUNK", 37)
    by_chunks = mx.concatenate([A1.compute_index_keys(kk, wk, a, min(3000, a + 413)) for a in range(0, 3000, 413)])
    assert (_bits(whole) == _bits(by_chunks)).all()
    k = mx.transpose(kk[0, :, :, 64:], (1, 0, 2)).reshape(3000, 768).astype(mx.float32) @ wk
    ref = k / mx.linalg.norm(k, axis=1, keepdims=True)
    assert float(mx.max(mx.abs(whole.astype(mx.float32) - ref))) < 2e-3


def test_read_close_to_f32(a1):
    """A row's output = f32 attention over its tiles (the engine's bits) + its tail, within the f16 error."""

    kk, vv, qq = _data(5000, seed=5)
    c = _cache(kk, vv)
    p, T = 4700, 8
    out = _write(c, kk, vv, qq, p, p, T)
    qi = A1.index_queries(qq[:, :, :T], A1.lsa_map()[LAYER][0])
    starts = [A1.tail_start(q) for q in range(p, p + T)]
    lb, _, nw = A1.choose(qi, c.pli_kidx[0], starts, 10)
    lbn = np.array(lb).reshape(-1)[:T * nw].reshape(T, nw)
    m = np.zeros((T, p + T), bool)
    for t in range(T):
        for j in range(starts[t] // 32):
            if (int(lbn[t, j >> 5]) >> (j & 31)) & 1:
                m[t, j * 32:(j + 1) * 32] = True
        m[t, starts[t]:p + t + 1] = True
    ref = mx.fast.scaled_dot_product_attention(qq[:, :, :T].astype(mx.float32), kk[:, :, :p + T].astype(mx.float32),
                                               vv[:, :, :p + T].astype(mx.float32), scale=SCALE, mask=mx.array(m)[None, None])
    assert float(mx.max(mx.abs(out.astype(mx.float32) - ref))) < 2e-2


@pytest.mark.parametrize("load", [False, True])
def test_split_same_bits(a1, monkeypatch, load):
    """Rounds of 8, of 1 and of mixed sizes (up to 19 rows: 3 threadgroups), rows below the threshold and a change of S included."""

    monkeypatch.setattr(AT, "LOAD", load)
    kk, vv, qq = _data(3200, seed=1)
    q0 = 2028                                  # 20 rows below the threshold, then S goes from 512 to 1024 near position 2048 + 1024
    ref = _run(_cache(kk, vv), kk, vv, qq, q0, [8] * 8)
    assert (ref == _run(_cache(kk, vv), kk, vv, qq, q0, [1] * 64)).all()
    assert (ref == _run(_cache(kk, vv), kk, vv, qq, q0, [3, 5, 13, 19, 2, 7, 15])).all()
    assert A1.COUNT["calls"] > 0


def test_s3b_like_decode(a1, monkeypatch):
    """A prompt chunk of 40 rows (S3b): same bits as the same positions decoded one by one."""

    monkeypatch.setattr(EA, "S3B", True)
    monkeypatch.setattr(EA, "S3B_THRESHOLD", 2600)
    kk, vv, qq = _data(3000, seed=2)
    q0 = 2700
    c = _cache(kk, vv)
    c.pli_request_start, c.pli_prefill_start = 2900, 2000
    chunk = _run(c, kk, vv, qq, q0, [40])
    assert EA._REASONS.get("a1_s3b", 0) > 0
    assert (chunk == _run(_cache(kk, vv), kk, vv, qq, q0, [1] * 40)).all()


def test_resume_same_bits(a1):
    """Index keys rebuilt at once (fresh resumed cache), cache copy (other id), rollback and start of a request."""

    from tensorfold.engine.family_prefill import FamilyPrefill
    from tensorfold.engine.lane_engine import LaneEngine

    kk, vv, qq = _data(4400, seed=4)
    q0 = 4000
    c1 = _cache(kk, vv)
    ref = _run(c1, kk, vv, qq, q0, [8] * 6)
    assert c1.pli_kidx[1] == A1.tail_start(q0 + 47)
    # fresh cache (read back from disk, copied checkpoint) from row 24
    assert (ref[:, :, 24:] == _run(_cache(kk, vv), kk, vv, qq, q0 + 24, [8] * 3, base=q0)).all()
    # copy of c1 (copy_single_cache): other id, index keys rebuilt
    c2 = _cache(kk, vv)
    _run(c2, kk, vv, qq, q0, [8] * 2)
    c3 = LaneEngine.copy_single_cache([c2])[0]
    assert c3.pli_kidx[2] != id(c3)
    assert (ref[:, :, 16:] == _run(c3, kk, vv, qq, q0 + 16, [8] * 4, base=q0)).all()
    # rollback: c1 cut to 2500 positions, K and V rewritten past it, new request (start of a request): like a fresh cache
    kk2, vv2, _ = _data(4400, seed=9)
    kn = mx.concatenate([kk[:, :, :2500], kk2[:, :, 2500:]], axis=2)
    vn = mx.concatenate([vv[:, :, :2500], vv2[:, :, 2500:]], axis=2)
    c1.trim(int(c1.offset) - 2500)
    c1.keys, c1.values = kn, vn
    F = type("F", (), {"model": None, "_family_start": FamilyPrefill._family_start})
    work, start = F()._family_start([c1], 2500, [2500])
    assert start == 2500 and work[0].pli_kidx[1] == 2500
    work[0].pli_request_start, work[0].pli_anchors, work[0].pli_layer = 0, {}, LAYER
    fresh = _run(_cache(kn, vn), kn, vn, qq, q0, [8] * 6)
    assert (fresh == _run(work[0], kn, vn, qq, q0, [8] * 6)).all()
    assert not (fresh == ref).all()


def test_rollback_without_request_start(a1):
    """Guard: a round that starts before the end of the index keys cuts them to its first position."""

    kk, vv, qq = _data(4400, seed=6)
    c = _cache(kk, vv)
    _run(c, kk, vv, qq, 4000, [8] * 4)
    kk2, vv2, _ = _data(4400, seed=8)
    kn = mx.concatenate([kk[:, :, :2900], kk2[:, :, 2900:]], axis=2)
    vn = mx.concatenate([vv[:, :, :2900], vv2[:, :, 2900:]], axis=2)
    c.keys, c.values = kn, vn
    # a round at 2900 (before the end of the index keys, 2976): they come back to 2900, then decoding goes on
    a = _run(c, kn, vn, qq, 2900, [8] * 4)
    b = _run(_cache(kn, vn), kn, vn, qq, 2900, [8] * 4)
    assert (a == b).all()


def test_tree(a1):
    """A chain-shaped tree has the chain's bits; a real tree: each node close to a masked f32 attention."""

    kk, vv, qq = _data(4300, seed=10)
    base, T = 4100, 8
    chain = _write(_cache(kk, vv), kk, vv, qq, base, base, T)
    c = _cache(kk, vv)
    c.offset = base + T
    tree = EA._anchored_tree(qq[:, :, :T], kk[:, :, :base + T], vv[:, :, :base + T], c, SCALE, list(range(-1, T - 1)))
    assert tree is not None and (_bits(tree) == _bits(chain)).all()
    parents = [-1, 0, 0, 1, 1, 2, 3, 3]
    c = _cache(kk, vv)
    c.offset = base + T
    out = EA._anchored_tree(qq[:, :, :T], kk[:, :, :base + T], vv[:, :, :base + T], c, SCALE, parents)
    an = AT.ancestors(parents)
    depth = [0] * T
    for i in range(1, T):
        depth[i] = depth[parents[i]] + 1
    starts = [A1.tail_start(base + d) for d in depth]
    qi = A1.index_queries(qq[:, :, :T], A1.lsa_map()[LAYER][0])
    lb, _, nw = A1.choose(qi, c.pli_kidx[0], starts, 10)
    lbn = np.array(lb).reshape(-1)[:T * nw].reshape(T, nw)
    m = np.zeros((T, base + T), bool)
    for t in range(T):
        for j in range(starts[t] // 32):
            if (int(lbn[t, j >> 5]) >> (j & 31)) & 1:
                m[t, j * 32:(j + 1) * 32] = True
        m[t, starts[t]:base] = True
        for j in range(T):
            m[t, base + j] = bool((an[t] >> j) & 1)
    ref = mx.fast.scaled_dot_product_attention(qq[:, :, :T].astype(mx.float32), kk[:, :, :base + T].astype(mx.float32),
                                               vv[:, :, :base + T].astype(mx.float32), scale=SCALE, mask=mx.array(m)[None, None])
    assert float(mx.max(mx.abs(out.astype(mx.float32) - ref))) < 2e-2


def test_snapshot_skips_index_keys(a1, tmp_path):
    from tensorfold.engine.prefix_snapshots import save_snapshot

    kk, vv, qq = _data(4200)
    c = _cache(kk, vv)
    _run(c, kk, vv, qq, 4000, [8])
    c.offset = 4008
    assert hasattr(c, "pli_kidx")
    assert save_snapshot(tmp_path, "m", list(range(5)), [c]) is not None
