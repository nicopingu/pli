"""PLI (idea from TensorFold 1.0.12 "Recall"): drafts copied from text the server already read.

The server keeps in memory the tokens of finished requests (prompt and reply). When the ordinary copy (the request's
context, SuffixLookupProposer) finds nothing, recall looks for the request's last 16 tokens in this register and proposes
what followed them there. The model checks each copied token: the text produced is the same, recall on or off (V2).

Version 2 (17/10/2026, after a bench on the 27B):
- a request enters the register only at its end (TensorFold: "its reply when the reply ends"); v1 poured the context in at
  every round, which overwrote in the index the old occurrence with the current request's one, so that the copy
  announced while computing the draft depth (a single draft token) vanished at the next round ("head 2");
- same answer for two calls on the same context (depth computation then round);
- TensorFold's rule: a copy is proposed only if it should move further than the stream's recent drafts
  (moving average of tokens accepted per draft round, against that of the register's copies, per range of
  match length 16-31, 32-63, 64+); a copy that moves nothing silences the register for 4 rounds for this stream;
  probes are made rarer and rarer while they miss.

Off by default. PLI_RECALL=1 switches it on; PLI_RECALL_TOKENS (default 1,000,000) bounds the register, the oldest leave first.
Memory only (nothing on disk): the register starts empty at every server start.
"""

from __future__ import annotations

import os
import threading
import weakref
from array import array
from typing import Any, Sequence

import numpy as np

KEY = 16                         # tokens of the search key (TensorFold: 16 at least)
DEDUP_STEP = 64                  # step of the search for the new start of a prompt already read
_P = np.uint64(0x100000001B3)    # multiplier of the polynomial hash (modulo 2**64)
ALPHA = 0.25                     # weight of a round in the moving averages
RECALL_PRIOR = 6.0               # tokens expected from a register copy before any measurement (optimistic: it must be tried)


def _hashes(ids: np.ndarray) -> np.ndarray:
    """Hash of each window of KEY tokens of ``ids`` (len(ids) - KEY + 1 values, window i = ids[i : i + KEY])."""

    n = len(ids) - KEY + 1
    h = np.zeros(n, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for j in range(KEY):
            h = h * _P + ids[j:j + n].astype(np.uint64)
    return h


def _hash(key: Sequence[int]) -> int:
    return int(_hashes(np.asarray(key, dtype=np.int64))[0])


def _range(length: int) -> int:
    return 0 if length < 32 else (1 if length < 64 else 2)


class Register:
    """Tokens of finished requests and the index of their windows of KEY tokens (last end position)."""

    def __init__(self, max_ids: int) -> None:
        self.max_ids = max(4 * KEY, int(max_ids))
        self.ids = array("i")
        self.start = 0                  # global position of ids[0]
        self.index: dict[int, int] = {}
        self.lock = threading.Lock()
        self.expected = [RECALL_PRIOR] * 3   # tokens accepted per copy, per match range
        self.offers = self.proposed = self.accepted = 0

    @property
    def end(self) -> int:
        return self.start + len(self.ids)

    def token(self, pos: int) -> int:
        return self.ids[pos - self.start]

    def _prune(self) -> None:
        if len(self.ids) <= self.max_ids:
            return
        cut = len(self.ids) - self.max_ids // 2
        self.ids = self.ids[cut:]
        self.start += cut
        self.index = {h: p for h, p in self.index.items() if p - KEY + 1 >= self.start}

    def add(self, seq: Sequence[int]) -> None:
        """Add a finished request (a segment apart: no window straddles the previous segment)."""

        if len(seq) < KEY + 1:
            return
        with self.lock:
            at = self.end
            self.ids.extend(int(t) for t in seq)
            h = _hashes(np.asarray(seq, dtype=np.int64))
            self.index.update(zip(h.tolist(), range(at + KEY - 1, at + KEY - 1 + len(h))))
            self._prune()

    def new_start(self, seq: Sequence[int]) -> int:
        """Index of ``seq`` from which it is no longer already in the register (in steps of DEDUP_STEP)."""

        n = len(seq)
        bounds = list(range(KEY - 1, n, DEDUP_STEP))
        if not bounds:
            return 0
        tab = np.asarray(seq, dtype=np.int64)
        with self.lock:
            for i, end in enumerate(bounds):
                if _hash(tab[end - KEY + 1:end + 1]) not in self.index:
                    return 0 if i == 0 else max(0, bounds[i - 1] + 1 - (KEY - 1))
        return max(0, bounds[-1] + 1 - (KEY - 1))

    def pour(self, seq: Sequence[int]) -> None:
        """A finished request: only its new part enters the register (a prompt already read is not poured again)."""

        at = self.new_start(seq)
        self.add(seq[at:])

    def search(self, context: Sequence[int], max_draft: int) -> tuple[list[int], int]:
        """(proposed continuation, matching tokens) for the end of ``context``."""

        if len(context) < KEY or max_draft <= 0:
            return [], 0
        tail = [int(t) for t in context[-KEY:]]
        h = _hash(tail)
        with self.lock:
            pos = self.index.get(h)
            if pos is None or pos < self.start + KEY - 1 or pos + 1 >= self.end:
                return [], 0
            if [self.token(p) for p in range(pos - KEY + 1, pos + 1)] != tail:
                return [], 0            # hash collision
            length = KEY
            limit = min(64, len(context), pos - self.start + 1)
            while length < limit and self.token(pos - length) == int(context[-1 - length]):
                length += 1
            follow = [self.token(p) for p in range(pos + 1, min(self.end, pos + 1 + max_draft))]
        return follow, length


_REGISTER: Register | None = None
_LOCK = threading.Lock()


def register() -> Register | None:
    global _REGISTER
    if os.environ.get("PLI_RECALL", "") not in ("1", "yes", "on", "oui"):
        return None
    with _LOCK:
        if _REGISTER is None:
            _REGISTER = Register(int(os.environ.get("PLI_RECALL_TOKENS", "1000000") or 1000000))
            print(f"[tensorfold] PLI recall on (v2): register in memory, {_REGISTER.max_ids} tokens at most", flush=True)
        return _REGISTER


def _pour_at_end(reg: Register, holder: dict) -> None:
    ctx, n = holder.get("ctx"), int(holder.get("n", 0))
    if ctx is not None and n > KEY:
        try:
            reg.pour(ctx[:n])
        except Exception:  # noqa: BLE001 - the end of a request must never break the server
            pass


class RecallProposer:
    """Wrapper of a copy proposer: the context copy first, the register when it finds nothing."""

    SILENCE = 4                         # rounds without the register after a copy that moved nothing

    def __init__(self, base: Any, reg: Register) -> None:
        self.base = base
        self.reg = reg
        self.holder = {"ctx": None, "n": 0}   # the request's context (growing list), poured into the register at the end
        weakref.finalize(self, _pour_at_end, reg, self.holder)
        self.source = ""                # "base" or "recall": who made the last proposal
        self.last_match = 0
        self.last_confident = False
        self.head = 0.0                 # tokens accepted per draft round of the model (moving average)
        self.range = 0
        self._silence = 0
        self._refused = 0               # calls where the rule set aside a possible copy
        self._probe_every = 8           # one probe every N refusals, doubled at each missed probe
        self._probing = False
        self._last: tuple[int, int, list[int], str, int, bool] | None = None

    def __getattr__(self, name: str) -> Any:     # telemetry and attributes of the base proposer
        base = self.__dict__.get("base")
        if base is None:
            raise AttributeError(name)
        return getattr(base, name)

    def _keep(self, context: Sequence[int]) -> None:
        old, n = self.holder["ctx"], self.holder["n"]
        if old is not None and old is not context:
            follows = (len(context) >= n and n > 0 and int(context[0]) == int(old[0])
                       and int(context[n - 1]) == int(old[n - 1]))
            if not follows:                                 # context replaced: the old one is a finished request
                _pour_at_end(self.reg, dict(self.holder))
        self.holder["ctx"] = context
        self.holder["n"] = len(context)

    def propose(self, context: Sequence[int], max_draft: int) -> list[int]:
        self._keep(context)
        key = (len(context), int(max_draft))
        d = self._last
        if d is not None and (d[0], d[1]) == key:   # same context (draft depth, then round): same answer
            self.source, self.last_match, self.last_confident = d[3], d[4], d[5]
            return list(d[2])
        prop = self.base.propose(context, max_draft)
        if prop:
            res = (list(prop), "base", int(getattr(self.base, "last_match", 0) or 0),
                   bool(getattr(self.base, "last_confident", False)))
        else:
            res = ([], "", int(getattr(self.base, "last_match", 0) or 0), False)
            if self._silence > 0:
                self._silence -= 1
            else:
                follow, length = self.reg.search(context, max_draft)
                if len(follow) >= 2:
                    t = _range(length)
                    expected = min(float(len(follow)), self.reg.expected[t])
                    self._probing = False
                    if expected <= self.head:
                        self._refused += 1
                        if self._refused % self._probe_every:
                            follow = []
                        else:
                            self._probing = True
                    if follow:
                        self.range = t
                        res = (follow, "recall", length, length >= 24)
                        self.reg.offers += 1
                        self.reg.proposed += len(follow)
        self.source, self.last_match, self.last_confident = res[1], res[2], res[3]
        self._last = (key[0], key[1], list(res[0]), res[1], res[2], res[3])
        return list(res[0])

    def observe(self, proposed: int, accepted: int) -> None:
        if self.source == "recall":
            reg = self.reg
            reg.accepted += int(accepted)
            reg.expected[self.range] += ALPHA * (float(accepted) - reg.expected[self.range])
            if accepted == 0:
                self._silence = self.SILENCE
            if self._probing:
                self._probe_every = 8 if accepted > self.head else min(256, 2 * self._probe_every)
                self._probing = False
            if reg.offers % 50 == 0:
                print(f"[tensorfold] PLI recall: {reg.offers} offers, {reg.proposed} tokens proposed, {reg.accepted} "
                      f"accepted, expected {', '.join(f'{a:.1f}' for a in reg.expected)}, register {len(reg.ids)} tokens",
                      flush=True)
            return
        obs = getattr(self.base, "observe", None)
        if obs is not None:
            obs(proposed, accepted)

    def observe_head(self, accepted: int) -> None:
        """A draft round of the model (lane_family): accepted tokens."""

        self.head += ALPHA * (float(accepted) - self.head)


def wrap(base: Any) -> Any:
    """The copy proposer, wrapped with recall if PLI_RECALL=1."""

    reg = register()
    return base if reg is None or base is None else RecallProposer(base, reg)
