"""PLI (10/11/2026): evicted conversations written by a thread of their own (pli/spill_writer.py)."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
from mlx_lm.models.cache import ArraysCache, KVCache

from tensorfold.engine.prefix_snapshots import PARTIAL, load_snapshot, save_snapshot
from tensorfold.pli import spill_writer as sw

MODEL = "model|test"


@dataclass
class Entry:
    tokens: list[int]
    cache: list
    nbytes: int


def _cache(n: int, seed: int = 0) -> list:
    mx.random.seed(seed)
    kv = KVCache()
    kv.update_and_fetch(mx.random.normal((1, 2, n, 8)).astype(mx.bfloat16),
                        mx.random.normal((1, 2, n, 8)).astype(mx.bfloat16))
    state = ArraysCache(2)
    state[0] = mx.random.normal((1, 3, 4)).astype(mx.float32)
    state[1] = mx.random.normal((1, 5)).astype(mx.bfloat16)
    mx.eval(kv.keys, kv.values, state[0], state[1])
    return [kv, state]


def _same(a: list, b: list) -> bool:
    return (mx.array_equal(a[0].keys, b[0].keys).item() and mx.array_equal(a[0].values, b[0].values).item()
            and all(mx.array_equal(x, y).item() for x, y in zip(a[1].cache, b[1].cache)))


def test_header_matches_mlx_writer(tmp_path):
    arrays = {"a": mx.random.normal((3, 5)).astype(mx.bfloat16), "b": mx.arange(7, dtype=mx.uint32),
              "c": mx.zeros((0, 4)), "d": mx.contiguous(mx.ones((2, 3)).T), "e": mx.array(True),
              "f": mx.random.normal((4,)).astype(mx.float16)}
    mx.eval(list(arrays.values()))
    header, views, _ = sw.header_and_views(arrays, {"x": "1"})
    sw.write_file(tmp_path / "ours.safetensors", header, views)
    mx.save_safetensors(str(tmp_path / "ref.safetensors"), arrays, metadata={"x": "1"})
    ours, meta = mx.load(str(tmp_path / "ours.safetensors"), return_metadata=True)
    ref = mx.load(str(tmp_path / "ref.safetensors"))
    assert meta == {"x": "1"} and sorted(ours) == sorted(ref)
    for name in ref:
        assert ours[name].dtype == ref[name].dtype and ours[name].shape == ref[name].shape
        assert mx.array_equal(ours[name], ref[name]).item()


def test_background_file_reads_back_like_the_synchronous_one(tmp_path):
    cache = _cache(16)
    tokens = list(range(16))
    save_snapshot(tmp_path / "sync", MODEL, tokens, cache, keep=1 << 30)
    writer = sw.SpillWriter(tmp_path / "bg", MODEL, limit_bytes=1 << 40)
    assert writer.submit(Entry(tokens, cache, 1000))
    assert writer.close(timeout=30)
    (sync_file,) = (tmp_path / "sync").glob("*.safetensors")
    (bg_file,) = (tmp_path / "bg").glob("*.safetensors")
    assert sync_file.name == bg_file.name                    # same key
    got_sync, got_bg = load_snapshot(sync_file, MODEL), load_snapshot(bg_file, MODEL)
    assert got_sync[0] == got_bg[0] == tokens and _same(got_sync[1], got_bg[1])
    assert not list((tmp_path / "bg").glob(f"*{PARTIAL}"))
    assert writer.written == 1 and writer.failed == 0


def test_settle_waits_for_a_prefix_being_written(tmp_path, monkeypatch):
    gate = threading.Event()
    real = sw.write_file

    def slow(target, header, views):
        gate.wait(10)
        real(target, header, views)

    monkeypatch.setattr(sw, "write_file", slow)
    writer = sw.SpillWriter(tmp_path, MODEL, limit_bytes=1 << 40)
    tokens = list(range(8))
    assert writer.submit(Entry(tokens, _cache(8), 1000))
    assert not list(tmp_path.glob("*.safetensors")) or all(p.name.endswith(PARTIAL) for p in tmp_path.glob("*"))
    assert writer.settle([99, 98, 97]) == 0.0                # another conversation: no wait
    threading.Timer(0.3, gate.set).start()
    waited = writer.settle(tokens + [100, 101])              # this conversation, one turn on: waits for its file
    assert waited >= 0.2
    assert [p for p in tmp_path.glob("*.safetensors") if not p.name.endswith(PARTIAL)]
    writer.close(timeout=10)


def test_pending_bytes_are_bounded(tmp_path, monkeypatch):
    gate = threading.Event()
    real = sw.write_file
    monkeypatch.setattr(sw, "write_file", lambda t, h, v: (gate.wait(10), real(t, h, v)))
    writer = sw.SpillWriter(tmp_path, MODEL, limit_bytes=1 << 40, max_pending_bytes=1)
    assert writer.submit(Entry(list(range(8)), _cache(8, 1), 1000))
    threading.Timer(0.3, gate.set).start()
    started = time.perf_counter()
    assert writer.submit(Entry(list(range(50, 58)), _cache(8, 2), 1000))   # waits for the first to be written
    assert time.perf_counter() - started >= 0.2
    assert writer.close(timeout=10) and len(list(tmp_path.glob("*.safetensors"))) == 2


def test_pruning_waits_for_a_read(tmp_path):
    writer = sw.SpillWriter(tmp_path, MODEL, limit_bytes=1)    # every file pruned once written
    with writer.disk_lock:
        assert writer.submit(Entry(list(range(8)), _cache(8), 1))
        time.sleep(0.3)
        assert len([p for p in tmp_path.glob("*.safetensors") if not p.name.endswith(PARTIAL)]) == 1
    writer.close(timeout=10)
    assert not list(tmp_path.glob("*.safetensors"))


def test_after_close_writes_on_the_calling_thread(tmp_path):
    writer = sw.SpillWriter(tmp_path, MODEL, limit_bytes=1 << 40)
    writer.close(timeout=10)
    assert writer.submit(Entry(list(range(8)), _cache(8), 1000))
    assert len(list(tmp_path.glob("*.safetensors"))) == 1


def test_too_big_or_already_there(tmp_path):
    writer = sw.SpillWriter(tmp_path, MODEL, limit_bytes=10)
    assert not writer.submit(Entry(list(range(8)), _cache(8), 11))
    cache = _cache(8)
    save_snapshot(tmp_path, MODEL, list(range(8)), cache, keep=1 << 30)
    assert writer.submit(Entry(list(range(8)), cache, 5))     # the file is there: touched, nothing queued
    assert writer.close(timeout=10) and writer.written == 0


def test_off_by_default():
    import os
    if "PLI_SPILL_BACKGROUND" not in os.environ:
        assert sw.ENABLED is False
