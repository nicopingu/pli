"""PLI (04/11/2026, ported 10/11): PLI_DISK_REPLY_END: under PLI_KEEP_REPLY the disk lookup takes the checkpoint lengths
memory takes (a reply end is not a chunk start); off, only chunk starts (previous behaviour)."""

from __future__ import annotations

import pytest

from tensorfold.server import prompt_fill
from tensorfold.server.checkpoints import CheckpointStore
from tensorfold.server.scheduler import ChatJob, Scheduler
from tests.test_prompt_fill import GridEngine, _prompt, _run, _solo

OFF_CHUNK, START = 10, 8          # prompt of 22 tokens, chunks of 4: 8 is a chunk start, 10 is not


def _seen(monkeypatch, disk: bool, keep: bool, reread: bool = False):
    monkeypatch.setattr(prompt_fill, "_DISK_REPLY_END", disk)
    monkeypatch.setattr(prompt_fill, "_KEEP_REPLY", keep)
    monkeypatch.setattr(prompt_fill, "_REREAD_DISK", reread)
    store = CheckpointStore(8, copier=GridEngine.copy_single_cache)
    scheduler = Scheduler(GridEngine(), lanes=2, eos_ids=frozenset({-1}), checkpoints=store)
    seen = []

    def read(prompt, usable=None):                                  # fake disk: records the lengths it would take
        seen.append((usable(OFF_CHUNK), usable(START)))

    monkeypatch.setattr(scheduler, "_read_disk_block", read)
    job = ChatJob("j", _prompt(3), 3, 0.0)
    _run(scheduler, [job])
    assert _solo(job)
    return seen


def test_off_the_disk_takes_chunk_starts_only(monkeypatch):
    assert set(_seen(monkeypatch, disk=False, keep=True, reread=True)) == {(False, True)}


def test_on_under_keep_reply_the_disk_takes_memorys_lengths(monkeypatch):
    seen = _seen(monkeypatch, disk=True, keep=True, reread=True)
    assert len(seen) >= 2 and set(seen) == {(True, True)}           # at the open and at the first chunk


def test_on_without_keep_reply_nothing_changes(monkeypatch):
    assert set(_seen(monkeypatch, disk=True, keep=False, reread=True)) == {(False, True)}


@pytest.mark.parametrize("disk", [False, True])
def test_memory_keeps_its_choice(monkeypatch, disk):
    monkeypatch.setattr(prompt_fill, "_DISK_REPLY_END", disk)
    monkeypatch.setattr(prompt_fill, "_KEEP_REPLY", True)
    starts = GridEngine.prefill_plan.chunks(_prompt(3))
    accepts = prompt_fill._disk_accepts(starts)
    assert accepts(START) and accepts(OFF_CHUNK) == disk
