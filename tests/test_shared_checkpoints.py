"""PLI (29/10/2026, lot K5): under PLI_KV_SHARING, a KV layer held by several checkpoints is counted once."""

import pytest

mx = pytest.importorskip("mlx.core")
from mlx_lm.models.cache import KVCache

from tensorfold.pli import kv_sharing as K
from tensorfold.server import checkpoints as C
from tensorfold.server.memory_budget import cache_nbytes


class State:
    """A layer that belongs to each checkpoint (like the DeltaNet state)."""

    def __init__(self, n: int) -> None:
        self.state = [mx.zeros((n,), dtype=mx.float32)]


def layer(n: int) -> KVCache:
    c = KVCache()
    c.update_and_fetch(mx.zeros((1, 2, n, 8)), mx.zeros((1, 2, n, 8)))
    mx.eval(c.keys, c.values)
    return c


@pytest.fixture
def active(monkeypatch):
    monkeypatch.setattr(K, "ACTIVE", True)


def test_prompt_and_reply_end_counted_once(active):
    kv = layer(300)
    alone = cache_nbytes([kv])
    state = cache_nbytes([State(1000)])
    store = C.CheckpointStore(4, copier=lambda c: list(c), sizer=cache_nbytes, budget_bytes=alone + 3 * state)
    prompt = list(range(250))
    store.insert(prompt, [K.KVLen(kv, 250), State(1000)], last_prompt=prompt + [1])
    assert store.nbytes == alone + state                      # single checkpoint: its layer counted
    end = list(range(300))
    store.insert(end, [kv, State(1000)], last_prompt=prompt + [1])
    # before: 2 x (alone + state) went over the budget and evicted the prompt checkpoint
    assert len(store) == 2 and store.nbytes == alone + 2 * state
    assert [e.nbytes for e in store._entries] == [alone + state, state]


def test_layer_counted_again_when_the_checkpoint_that_counted_it_leaves(active):
    kv = layer(300)
    alone, state = cache_nbytes([kv]), cache_nbytes([State(10)])
    store = C.CheckpointStore(4, copier=lambda c: list(c), sizer=cache_nbytes)
    store.insert(list(range(250)), [K.KVLen(kv, 250), State(10)], last_prompt=list(range(251)))
    store.insert(list(range(300)), [kv, State(10)], last_prompt=list(range(251)))
    assert store.evict_one()                                 # _victim: the checkpoint the reply end extends
    assert len(store) >= 1 and store.nbytes == sum(e.nbytes for e in store._entries)
    assert store._entries[0].nbytes == alone + state


def test_without_k5_as_before():
    kv = layer(300)
    alone, state = cache_nbytes([kv]), cache_nbytes([State(10)])
    store = C.CheckpointStore(4, copier=lambda c: list(c), sizer=cache_nbytes)
    store.insert(list(range(250)), [K.KVLen(kv, 250), State(10)], last_prompt=list(range(251)))
    store.insert(list(range(300)), [kv, State(10)], last_prompt=list(range(251)))
    assert store.nbytes == 2 * (alone + state)


def _store_and_three_checkpoints(monkeypatch, spill):
    monkeypatch.setattr(C, "_SPILL_PROMPT", spill)
    gone = []
    store = C.CheckpointStore(2, copier=lambda c: list(c), on_evict=lambda e: gone.append(list(e.tokens)))
    prompt = list(range(100))
    store.insert(prompt, ["a"], last_prompt=prompt + [7])                    # prompt checkpoint (prompt side)
    end = prompt + [500, 501, 502]                                          # reply end: extends the prompt
    store.insert(end, ["b"], last_prompt=prompt + [7])
    store.insert([9] * 50, ["c"], last_prompt=[9] * 51)                     # another conversation: evicts a checkpoint
    return gone, prompt


def test_evicted_prompt_goes_to_disk_even_extended_by_a_reply_end(monkeypatch):
    gone, prompt = _store_and_three_checkpoints(monkeypatch, True)
    assert gone == [prompt]          # _victim evicts the prompt (extended); it goes to disk


def test_without_spill_prompt_as_before(monkeypatch):
    gone, _ = _store_and_three_checkpoints(monkeypatch, False)
    assert gone == []


def test_prompt_extended_by_a_longer_prompt_stays_without_a_write(monkeypatch):
    monkeypatch.setattr(C, "_SPILL_PROMPT", True)
    gone = []
    store = C.CheckpointStore(2, copier=lambda c: list(c), on_evict=lambda e: gone.append(list(e.tokens)))
    a = list(range(100))
    store.insert(a, ["a"], last_prompt=a + [7])
    b = a + [7, 8, 9]
    store.insert(b, ["b"], last_prompt=b + [3])                            # next turn, prompt side
    store.insert([9] * 50, ["c"], last_prompt=[9] * 51)
    assert gone == []
