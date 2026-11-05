"""PLI (KVD5): checkpoints reduced to 8 bits or 4 rotated bits (PLI_CHECKPOINT_BITS)."""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.models.cache import KVCache

from tensorfold.engine.prefix_snapshots import load_snapshot, save_snapshot
from tensorfold.pli import checkpoint_bits as _cb
from tensorfold.pli import kv_sharing as _kvs
from tensorfold.server.checkpoints import CheckpointStore
from tensorfold.server.memory_budget import cache_nbytes


class State:                                  # a DeltaNet-like state for the shape: never reduced
    def __init__(self) -> None:
        self.s = mx.ones((1, 4, 8, 8), dtype=mx.float32)

    @property
    def state(self):
        return [self.s]


def kv(n: int, heads: int = 4, d: int = 256, seed: int = 0) -> KVCache:
    mx.random.seed(seed)
    c = KVCache()
    cap = -(-n // 256) * 256
    k = mx.random.normal((1, heads, cap, d)).astype(mx.bfloat16)
    v = mx.random.normal((1, heads, cap, d)).astype(mx.bfloat16)
    c.keys, c.values, c.offset = k, v, n
    mx.eval(c.keys, c.values)
    return c


def copier(cache):
    from tensorfold.engine.lane_engine import LaneEngine

    return LaneEngine.copy_single_cache(cache)


@pytest.mark.parametrize("bits,bound", [(8, 0.02), (4, 0.2)])
def test_round_trip_shape_and_bounded_error(bits, bound):
    c = kv(300)
    q = _cb.reduce(c, bits)
    r = q.restore()
    assert r.offset == 300 and r.keys.shape == (1, 4, 300, 256) and r.keys.dtype == mx.bfloat16
    for a, b in ((c.keys[..., :300, :], r.keys), (c.values[..., :300, :], r.values)):
        rel = (mx.abs(a.astype(mx.float32) - b.astype(mx.float32)).mean() / mx.abs(a.astype(mx.float32)).mean()).item()
        assert rel < bound, (bits, rel)


@pytest.mark.parametrize("bits,nbytes", [(8, 64 + 4), (4, 32 + 4)])
def test_exact_byte_count(bits, nbytes):
    n = 512
    q = _cb.reduce(kv(n), bits)
    groups = 2 * 4 * n * 256 // 64               # K and V, 4 heads, n positions, 256 per head, groups of 64
    assert q.nbytes == groups * nbytes           # weights + bf16 scale and bias per group
    assert cache_nbytes([q]) == q.nbytes


def test_disk_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(_cb, "BITS", 8)
    cache = [kv(260), State()]
    disk = _cb.for_disk(cache)
    assert _cb.is_quantized(disk[0]) and disk[1] is cache[1]
    save_snapshot(tmp_path, "m|points=q8g64", list(range(260)), disk)
    tokens, back = load_snapshot(next(tmp_path.glob("*.safetensors")), "m|points=q8g64")
    assert tokens == list(range(260)) and _cb.is_quantized(back[0])
    a, b = disk[0].restore(), back[0].restore()
    assert mx.array_equal(a.keys, b.keys).item() and mx.array_equal(a.values, b.values).item()
    assert load_snapshot(next(tmp_path.glob("*.safetensors")), "m") is None      # engine with it off: other key, refused


def test_key():
    assert _cb.key_suffix(0) == ""
    assert _cb.key_suffix(8) == "|points=q8g64" and _cb.key_suffix(4) == "|points=q4hg64"


def test_off_nothing_changes(monkeypatch):
    monkeypatch.setattr(_cb, "BITS", 0)
    cache = [kv(260)]
    assert _cb.for_disk(cache)[0] is cache[0]
    store = CheckpointStore(4, copier, budget_bytes=int(cache_nbytes([kv(260)]) * 1.5), sizer=cache_nbytes)
    store.insert([1] * 260, [kv(260, seed=1)], last_prompt=[1] * 260)
    store.insert([2] * 260, [kv(260, seed=2)], last_prompt=[2] * 260)
    assert len(store) == 1 and store.reduced == 0 and store.evictions == 1


@pytest.mark.parametrize("k5", [False, True])
def test_reduce_before_evicting_and_resume(monkeypatch, k5):
    monkeypatch.setattr(_cb, "BITS", 8)
    monkeypatch.setattr(_kvs, "ACTIVE", k5)
    full = cache_nbytes([kv(512)])
    store = CheckpointStore(4, copier, budget_bytes=int(full * 1.7), sizer=cache_nbytes)
    a = kv(512, seed=1)
    ref = mx.array(a.keys[..., :512, :])
    store.insert([1] * 512, [a], last_prompt=[1] * 512)
    store.insert([2] * 512, [kv(512, seed=2)], last_prompt=[2] * 512)
    assert len(store) == 2 and store.reduced == 1 and store.evictions == 0      # 2 checkpoints kept instead of one
    old = [e for e in store._entries if e.tokens == [1] * 512][0]
    assert _cb.is_quantized(old.cache[0]) and store.nbytes <= store.budget_bytes
    n, live, _ = store.match([1] * 512 + [9])
    assert n == 512 and _kvs.is_kv(live[0]) and live[0].offset == 512
    err = mx.abs(live[0].keys[..., :512, :].astype(mx.float32) - ref.astype(mx.float32)).mean().item()
    assert err < 0.02
    if k5:                                           # the checkpoint now holds the restored layer by length
        old = [e for e in store._entries if e.tokens == [1] * 512][0]
        assert isinstance(old.cache[0], _kvs.KVLen) and old.cache[0].owner is live[0]


def test_k5_length_checkpoint_held_never_reduced(monkeypatch):
    monkeypatch.setattr(_cb, "BITS", 8)
    monkeypatch.setattr(_kvs, "ACTIVE", True)
    real = kv(512, seed=3)
    store = CheckpointStore(4, copier, budget_bytes=10**12, sizer=cache_nbytes)
    store.insert([1] * 512, [real], last_prompt=[1] * 512)
    store.insert([1] * 256, [_kvs.KVLen(real, 256)], last_prompt=[1] * 512)
    assert store._reduce_one(store._entries) is False        # the real layer is held by a length
    assert store.reduced == 0
