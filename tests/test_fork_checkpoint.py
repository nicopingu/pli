"""PLI (lot 20261027-084147-5ce3, TensorFold #1291): a checkpoint where the prompt forks inside its last message."""

from tensorfold.server import checkpoints as C


def test_document_sent_again_with_another_question():
    system, doc = list(range(100)), list(range(1000, 9000))
    q1, q2 = [5, 6, 7], [8, 9]
    before, now = system + doc + q1, system + doc + q2
    # resumed at the start of the message (end of the system), the next prompt forks after the document
    assert C.choose_checkpoints(len(system), len(system), before, now) == [len(system + doc)]


def test_short_fork_or_off(monkeypatch):
    system, doc = list(range(100)), list(range(1000, 2000))
    before, now = system + doc + [1], system + doc + [2]
    assert C.choose_checkpoints(len(system), len(system), before, now) == []     # fewer than 4,096 tokens read
    monkeypatch.setattr(C, "_FORK_MIN", 0)
    long_doc = list(range(1000, 9000))
    assert C.choose_checkpoints(len(system), len(system), system + long_doc + [1], system + long_doc + [2]) == []


def test_next_turn_unchanged(monkeypatch):
    # an ordinary agent turn: the previous prompt ends before the last message; same checkpoints as before the lot
    before = list(range(5000))
    now = before + list(range(9000, 9300))
    seen = C.choose_checkpoints(5100, 4990, before, now)
    monkeypatch.setattr(C, "_FORK_MIN", 0)
    assert seen == C.choose_checkpoints(5100, 4990, before, now) == [5000, 5100]


def test_fork_without_a_resume():
    # bench hq-fourche (28/10): no checkpoint prefixes the second question (nothing in common before the document); the
    # nearest kept prompt gives the fork point
    doc = list(range(1000, 9000))
    before, now = doc + [5, 6, 7], doc + [8, 9]
    store = C.CheckpointStore(3, copier=lambda c: c)
    store.insert(before, ["cache"], last_prompt=before)
    store.insert([1, 2, 3], ["other"], last_prompt=[1, 2, 3, 4])
    nearest = store.nearest_prompt(now)
    assert nearest == before and (store.hits, store.misses) == (0, 0)
    assert C.choose_checkpoints(0, 0, nearest, now) == [len(doc)]
    assert store.nearest_prompt(doc[:100] + [1]) is None                 # fewer than 4,096 tokens in common
