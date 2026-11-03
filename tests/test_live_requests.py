"""PLI: /health details each engine request ("requests"), for the dashboard's per-slot view."""

import queue
import threading
from types import SimpleNamespace

from tensorfold.server import live
from tensorfold.server.live import requests


def job(job_id, prompt, *, submitted=0.0, started=0.0, cached=0, emitted=0, stream=True, background=False,
        labels=(), max_tokens=512):
    stream_ = SimpleNamespace(cached_tokens=cached, emitted=[7] * emitted) if stream else None
    return SimpleNamespace(job_id=job_id, prompt_ids=[1] * prompt, max_tokens=max_tokens, submitted_at=submitted,
                           started_at=started, cached_tokens=0, stream=stream_, background=background, label_ids=labels)


def queue_of(*jobs):
    """A queue like _JobQueue: tuples (priority, order, job) in a heap, so not necessarily sorted."""

    q = queue.PriorityQueue()
    for order, (prio, j) in enumerate(jobs):
        q.put((prio, order, j))
    return q


def test_writing_then_reading_then_waiting_with_their_numbers():
    decoding = job("req-a", 10_000, submitted=1.0, started=2.0, cached=9_000, emitted=120)
    prefilling = job("req-b", 40_000, submitted=3.0, started=4.0, cached=8_192)
    held = job("req-c", 2_000, submitted=5.0, stream=False)
    background = job("req-d", 500, submitted=6.0, stream=False, background=True)
    ahead = job("req-e", 900, submitted=7.0, stream=False)
    sched = SimpleNamespace(_jobs={"s1": decoding}, _fills=[SimpleNamespace(job=prefilling, left=12_000)], _held=held,
                            _queue=queue_of((1, background), (0, ahead)))
    seen = requests(sched, clock=lambda: 10.0)
    assert [(r["id"], r["state"]) for r in seen] == [("req-a", "decoding"), ("req-b", "prefilling"),
                                                    ("req-c", "waiting"), ("req-e", "waiting"), ("req-d", "waiting")]
    assert seen[0] == {"id": "req-a", "state": "decoding", "background": False, "decision": False,
                     "prompt_tokens": 10_000, "prefilled_tokens": 10_000, "cached_tokens": 9_000,
                     "generated_tokens": 120, "max_tokens": 512, "age_s": 9.0, "running_s": 8.0}
    assert seen[1]["prefilled_tokens"] == 28_000 and seen[1]["cached_tokens"] == 8_192 and seen[1]["generated_tokens"] == 0
    assert seen[2]["prefilled_tokens"] == 0 and seen[2]["running_s"] is None and seen[2]["age_s"] == 5.0
    assert seen[4]["background"] is True


def test_a_prompt_not_yet_chunked_reads_zero_and_a_decision_is_marked():
    j = job("decision-1", 3_000, labels=(11, 12), stream=False)
    seen = requests(SimpleNamespace(_fills=[SimpleNamespace(job=j, left=3_000)]), clock=lambda: 1.0)
    assert seen[0]["state"] == "prefilling" and seen[0]["prefilled_tokens"] == 0 and seen[0]["decision"] is True


def test_the_list_is_capped_and_never_carries_prompt_text():
    waiting = [(0, job(f"req-{n}", 100 + n, stream=False)) for n in range(live.REQUESTS_MAX + 10)]
    seen = requests(SimpleNamespace(_queue=queue_of(*waiting)), clock=lambda: 1.0)
    assert len(seen) == live.REQUESTS_MAX and seen[0]["id"] == "req-0"
    assert all(isinstance(v, (str, int, float, bool, type(None))) for r in seen for v in r.values())


def test_anything_unexpected_gives_an_empty_list_never_an_error():
    assert requests(SimpleNamespace()) == []
    assert requests(SimpleNamespace(active=1, filling=[object()], waiting=2)) == []
    broken = SimpleNamespace(_jobs={"s": SimpleNamespace(job_id="x")})          # without prompt_ids
    assert requests(broken) == []


def test_reading_while_the_engine_thread_changes_the_lists():
    sched = SimpleNamespace(_jobs={}, _fills=[], _held=None, _queue=queue_of())
    stop = threading.Event()

    def engine():
        n = 0
        while not stop.is_set():
            n += 1
            sched._jobs[f"s{n}"] = job(f"req-{n}", 10, emitted=1)
            sched._fills.append(SimpleNamespace(job=job(f"f-{n}", 10), left=5))
            if len(sched._jobs) > 8:
                sched._jobs.pop(next(iter(sched._jobs)))
                sched._fills.pop(0)

    thread = threading.Thread(target=engine)
    thread.start()
    try:
        for _ in range(2_000):
            assert isinstance(requests(sched), list)
    finally:
        stop.set()
        thread.join()
