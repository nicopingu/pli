"""PLI (30/10/2026): PLI_REREAD_DISK: a checkpoint evicted between a request's open and its first chunk is read back
from disk at that chunk (here a fake disk); without the setting, the request reads its whole prompt again."""

from __future__ import annotations

from tensorfold.server import prompt_fill
from tensorfold.server.checkpoints import CheckpointStore
from tensorfold.server.scheduler import ChatJob, Scheduler
from tests.test_prompt_fill import GridEngine, _prompt, _run, _solo


def _scenario(monkeypatch, reread: bool):
    monkeypatch.setattr(prompt_fill, "_REREAD_DISK", reread)
    store = CheckpointStore(8, copier=GridEngine.copy_single_cache)
    engine = GridEngine()
    scheduler = Scheduler(engine, lanes=3, eos_ids=frozenset({-1}), checkpoints=store)
    base = _prompt(20, 16)
    _run(scheduler, [ChatJob("before", base, 2, 0.0, shared_prefix_lens=(16,))])                # stores checkpoints of [base] (chunks of 4)
    kept = [(list(e.tokens), e.cache) for e in store._entries]
    key = max((t for t, _ in kept if base[:len(t)] == t), key=len)     # the longest kept checkpoint of [base] (12 tokens)
    assert len(key) >= 8
    scheduler._stop.clear()
    disk = {"reads": 0}

    def read(prompt, usable=None):                                    # fake disk: puts the checkpoint back if it is missing
        disk["reads"] += 1
        if not any(list(e.tokens) == key for e in store._entries):
            for t, c in kept:
                if t == key:
                    store.insert(t, c, last_prompt=t)

    monkeypatch.setattr(scheduler, "_read_disk_block", read)
    other = ChatJob("other", _prompt(30, 22), 4, 0.0)
    follow = ChatJob("follow", base + _prompt(21, 6), 4, 0.0)
    jobs = [other]

    def arrive(job, events):
        if job is other and follow not in jobs:
            jobs.append(follow)
            scheduler.submit(follow)                                   # opens while "other" is being read

    open_job = scheduler._open_job

    def open_then_evict(job):                                         # evicted (spilled) right after "follow" opens
        filling = open_job(job)
        if job is follow:
            while any(list(e.tokens) == key for e in store._entries):
                store.evict_one()
        return filling

    monkeypatch.setattr(scheduler, "_open_job", open_then_evict)
    engine.prefill_calls.clear()
    _run(scheduler, jobs, on_fill=arrive)
    return engine, disk, other, follow, key


def test_without_the_setting_the_follow_up_reads_everything_again(monkeypatch):
    engine, disk, other, follow, key = _scenario(monkeypatch, False)
    assert ("follow", 0) in engine.prefill_calls and _solo(follow) and _solo(other)


def test_with_the_setting_the_follow_up_resumes_at_the_checkpoint_read_back(monkeypatch):
    engine, disk, other, follow, key = _scenario(monkeypatch, True)
    assert ("follow", len(key)) in engine.prefill_calls and _solo(follow) and _solo(other)
    assert disk["reads"] >= 2
