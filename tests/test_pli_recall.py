import gc
import os
import sys
pass
os.environ["PLI_RECALL"] = "1"
from tensorfold.pli import recall


class Base:
    last_match = 0
    def propose(self, context, max_draft):
        return []
    def observe(self, p, a):
        self.seen = (p, a)
    def telemetry(self):
        return {"ok": 1}


def request(reg, ctx, max_draft=8):
    """A request: propose at each token, then end (the proposer is released)."""
    p = recall.RecallProposer(Base(), reg)
    props = [p.propose(ctx[:i], max_draft) for i in range(recall.KEY, len(ctx) + 1)]
    del p
    gc.collect()
    return props


def test_nothing_before_the_end_then_copy():
    reg = recall.Register(10000)
    prompt, reply = list(range(100, 160)), list(range(500, 540))
    ctx = prompt + reply
    props = request(reg, ctx)
    assert all(x == [] for x in props)                     # nothing in the register during the request
    assert len(reg.ids) == len(ctx)                          # poured at the end
    b = recall.RecallProposer(Base(), reg)
    c1 = list(prompt)
    assert b.propose(c1, 8) == reply[:8] and b.source == "recall" and b.last_match >= recall.KEY
    assert b.propose(c1, 8) == reply[:8]                     # same context: same answer (depth then round)
    b.observe(8, 8)
    assert reg.accepted == 8
    c1 += reply[:10]
    assert b.propose(c1, 5) == reply[10:15]
    assert b.telemetry() == {"ok": 1}


def test_head_rule():
    reg = recall.Register(10000)
    prompt, reply = list(range(100, 160)), list(range(500, 540))
    request(reg, prompt + reply)
    b = recall.RecallProposer(Base(), reg)
    for _ in range(30):
        b.observe_head(7)                                    # the model's draft makes 7 tokens per round
    reg.expected = [3.0, 3.0, 3.0]                           # the register makes 3
    refused = [b.propose(prompt + reply[:i], 8) for i in range(0, 7)]
    assert refused[:6] == [[]] * 6                           # set aside, except a probe at the 8th refusal
    b.observe_head(0)
    assert b.head < 7


def test_silence_after_failure():
    reg = recall.Register(10000)
    prompt, reply = list(range(100, 160)), list(range(500, 540))
    request(reg, prompt + reply)
    b = recall.RecallProposer(Base(), reg)
    assert b.propose(prompt, 8)
    b.observe(8, 0)
    assert [b.propose(prompt + [1] * i, 8) for i in range(1, 5)] == [[]] * 4


def test_base_first():
    class B2(Base):
        last_match = 9
        def propose(self, context, max_draft):
            return [1, 2, 3]
    reg = recall.Register(10000)
    p = recall.RecallProposer(B2(), reg)
    assert p.propose(list(range(50)), 4) == [1, 2, 3] and p.source == "base" and p.last_match == 9
    p.observe(3, 2)
    assert p.base.seen == (3, 2)


def test_pruning_and_dedup():
    reg = recall.Register(1000)
    for k in range(30):
        request(reg, [7] * 5 + list(range(k * 100, k * 100 + 80)))
    assert len(reg.ids) <= 1000
    long_prompt = list(range(20000, 20600))
    reg.pour(long_prompt)
    n = len(reg.ids)
    reg.pour(long_prompt + list(range(30000, 30040)))
    assert len(reg.ids) - n < 200                            # the prompt already read is not poured again in full


def test_off():
    del os.environ["PLI_RECALL"]
    recall._REGISTER = None
    b = Base()
    assert recall.wrap(b) is b
