"""PLI : /health détaille chaque requête du moteur (« requests »), pour la vue par emplacement du tableau de bord."""

import queue
import threading
from types import SimpleNamespace

from tensorfold.server import live
from tensorfold.server.live import requests


def job(job_id, prompt, *, submitted=0.0, started=0.0, cached=0, emitted=0, stream=True, background=False,
        labels=(), max_tokens=512):
    flux = SimpleNamespace(cached_tokens=cached, emitted=[7] * emitted) if stream else None
    return SimpleNamespace(job_id=job_id, prompt_ids=[1] * prompt, max_tokens=max_tokens, submitted_at=submitted,
                           started_at=started, cached_tokens=0, stream=flux, background=background, label_ids=labels)


def file_de(*jobs):
    """Une file comme _JobQueue : des tuples (priorité, ordre, job) rangés en tas, donc pas forcément triés."""

    file = queue.PriorityQueue()
    for ordre, (prio, j) in enumerate(jobs):
        file.put((prio, ordre, j))
    return file


def test_writing_then_reading_then_waiting_with_their_numbers():
    ecrit = job("req-a", 10_000, submitted=1.0, started=2.0, cached=9_000, emitted=120)
    lit = job("req-b", 40_000, submitted=3.0, started=4.0, cached=8_192)
    retenue = job("req-c", 2_000, submitted=5.0, stream=False)
    fond = job("req-d", 500, submitted=6.0, stream=False, background=True)
    devant = job("req-e", 900, submitted=7.0, stream=False)
    sched = SimpleNamespace(_jobs={"s1": ecrit}, _fills=[SimpleNamespace(job=lit, left=12_000)], _held=retenue,
                            _queue=file_de((1, fond), (0, devant)))
    vu = requests(sched, clock=lambda: 10.0)
    assert [(r["id"], r["state"]) for r in vu] == [("req-a", "decoding"), ("req-b", "prefilling"),
                                                    ("req-c", "waiting"), ("req-e", "waiting"), ("req-d", "waiting")]
    assert vu[0] == {"id": "req-a", "state": "decoding", "background": False, "decision": False,
                     "prompt_tokens": 10_000, "prefilled_tokens": 10_000, "cached_tokens": 9_000,
                     "generated_tokens": 120, "max_tokens": 512, "age_s": 9.0, "running_s": 8.0}
    assert vu[1]["prefilled_tokens"] == 28_000 and vu[1]["cached_tokens"] == 8_192 and vu[1]["generated_tokens"] == 0
    assert vu[2]["prefilled_tokens"] == 0 and vu[2]["running_s"] is None and vu[2]["age_s"] == 5.0
    assert vu[4]["background"] is True


def test_a_prompt_not_yet_chunked_reads_zero_and_a_decision_is_marked():
    j = job("decision-1", 3_000, labels=(11, 12), stream=False)
    vu = requests(SimpleNamespace(_fills=[SimpleNamespace(job=j, left=3_000)]), clock=lambda: 1.0)
    assert vu[0]["state"] == "prefilling" and vu[0]["prefilled_tokens"] == 0 and vu[0]["decision"] is True


def test_the_list_is_capped_and_never_carries_prompt_text():
    attente = [(0, job(f"req-{n}", 100 + n, stream=False)) for n in range(live.REQUESTS_MAX + 10)]
    vu = requests(SimpleNamespace(_queue=file_de(*attente)), clock=lambda: 1.0)
    assert len(vu) == live.REQUESTS_MAX and vu[0]["id"] == "req-0"
    assert all(isinstance(v, (str, int, float, bool, type(None))) for r in vu for v in r.values())


def test_anything_unexpected_gives_an_empty_list_never_an_error():
    assert requests(SimpleNamespace()) == []
    assert requests(SimpleNamespace(active=1, filling=[object()], waiting=2)) == []
    casse = SimpleNamespace(_jobs={"s": SimpleNamespace(job_id="x")})          # sans prompt_ids
    assert requests(casse) == []


def test_reading_while_the_engine_thread_changes_the_lists():
    sched = SimpleNamespace(_jobs={}, _fills=[], _held=None, _queue=file_de())
    arret = threading.Event()

    def moteur():
        n = 0
        while not arret.is_set():
            n += 1
            sched._jobs[f"s{n}"] = job(f"req-{n}", 10, emitted=1)
            sched._fills.append(SimpleNamespace(job=job(f"f-{n}", 10), left=5))
            if len(sched._jobs) > 8:
                sched._jobs.pop(next(iter(sched._jobs)))
                sched._fills.pop(0)

    fil = threading.Thread(target=moteur)
    fil.start()
    try:
        for _ in range(2_000):
            assert isinstance(requests(sched), list)
    finally:
        arret.set()
        fil.join()
