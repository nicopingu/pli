"""PLI (24/10/2026, PLI_HYBRID): DFlash2 and the 27B's native MTP head loaded together.

Each conversation decodes with whichever of the two drafts gives it the most tokens per millisecond, measured on
its own rounds (moving averages; the first round after a switch is not counted). Until the engine has
measured both over GLOBAL_ROUNDS rounds (12, PLI_HYBRID_GLOBAL_ROUNDS), a conversation tries 6 DFlash2 rounds then 6 MTP rounds; after that it starts with
the better of the two for the engine, without a trial. The other is probed 6 rounds in 96. The choice follows the conversation from one
request to the next (it travels with the MTP head's cache of the checkpoints). Both draft contexts follow
every round (the idle draft reads the kept rows without proposing anything), so that it can switch at any time.
The text never changes: the drafts are checked by the model.

Cache layout: the 27B's layers, then MTPCache, then DraftSlot (last, as in the DFlash2 family).
PLI_HYBRID_FORCE=dflash|mtp freezes the choice (benches). PLI_HYBRID_LOG=<file> writes each switch.
"""

from __future__ import annotations

import os
import time
from pathlib import Path
from typing import Any, Sequence

from tensorfold.families.qwen3_5.dflash_head import DFlashHead, DraftSlot
from tensorfold.families.qwen3_5.family import Qwen35Family, _calibration, _tokens
from tensorfold.families.qwen3_5_moe.family import Qwen36Family

TRIAL = 6          # rounds of each draft at the start of a conversation (the choice then follows the conversation)
RETRY = 96         # every RETRY rounds, the other draft takes TRIAL rounds again
SMOOTHING = 0.05   # per-round damping of each draft's token and ms sums (about twenty rounds)
GAP_S = 1.0        # a longer gap between two rounds (end of request, prefill of another) is not counted
MARGIN = 0.05     # the other draft must do 5 % better to be kept instead
# PLI_HYBRID_MESSAGES=N: a request whose prompt holds at most N messages (message starts <|im_start|>, the reply being written
# included) writes with DFlash2 alone: a one-turn question (engine bench: DFlash2 faster by 6 to 11 %); beyond (agent sessions),
# the hybrid chooses as before. 0 (default): off. PLI_HYBRID_IM_START: the message start token (Qwen3.8: 248045).
MESSAGES = max(0, int(os.environ.get("PLI_HYBRID_MESSAGES", "0") or 0))
IM_START = int(os.environ.get("PLI_HYBRID_IM_START", "248045") or 248045)
# rounds measured for each draft, all conversations, before skipping the trial of a new conversation
GLOBAL_ROUNDS = int(os.environ.get("PLI_HYBRID_GLOBAL_ROUNDS", "12"))
# PLI (30/10/2026): PLI_TREES_SHORT_ONLY=1: DFlash2's trees (PLI_HYBRID_TREES) only for short questions (at most
# PLI_HYBRID_MESSAGES messages); agent sessions keep the chain (benches of 30/10: trees +6 to +17 % from 10k to 26k on one
# question, but -0.8 to -2.1 % on the 18 agent requests). Off by default.
TREES_SHORT_ONLY = os.environ.get("PLI_TREES_SHORT_ONLY", "0") == "1"
# PLI (30/10/2026): PLI_TREES_TOKEN_THRESHOLD=x: on a short question, the chain stays while the moving average of the tokens kept
# per round (about ten rounds) reaches x (long code: 5 to 6 tokens per round, the tree loses 2 % there); below x, the tree (1k to 26k:
# +7 to +15 %). Empty: the tree on every short question.
TOKEN_THRESHOLD = float(os.environ.get("PLI_TREES_TOKEN_THRESHOLD", "0") or 0)


class Qwen38Hybrid(Qwen36Family):
    """The dense 27B's verification rounds; drafts from DFlash2 or from the MTP head, chosen per conversation."""

    absorb_kept = None          # round without draft: speculate then settle at 0, which read both contexts
    plain_guard = False         # like DFlash2 in production
    draft_reads_hidden = True   # the MTP head reads the rows' final state (within the PLI_MTP_WINDOW window)
    draft_prior = Qwen35Family.draft_prior

    def __init__(self, model: Any, *, drafter: Any = None, mtp_path: Path | None = None, drafts: int = 7,
                 **options: Any) -> None:
        nodes = int(os.environ.get("PLI_TREE_NODES") or options.pop("nodes", 15))
        super().__init__(model, drafter=None, mtp_path=mtp_path, drafts=drafts, **options)
        if drafter is None or self.mtp is None:
            raise SystemExit("[pli] PLI_HYBRID needs DFlash2 (--drafter) and the MTP head (PLI_MTP_HEAD)")
        self.__dict__.pop("draft_probabilities", None)    # Qwen36Family sets it to None: the method below applies
        # PLI_HYBRID_TREES=1 lets DFlash2 propose trees (verified by attm with an ancestor mask: PLI_TREES_ATTM=1;
        # PLI_TREE_EXACT=1 for the chain's text).
        from tensorfold.kernels.qwen.dense.v1 import row_forward
        trees = os.environ.get("PLI_HYBRID_TREES", "0") == "1" and (row_forward.TREES_ATTM or row_forward.ROW_ATTENTION)
        self.head_drafts = DFlashHead(drafter, nodes, _calibration(), chains=not trees)
        self.mtp_drafts = int(self.drafts)
        self.drafts = max(nodes, self.mtp_drafts)
        self.mtp_step_ms_measured = float(self.mtp_step_ms)   # one chained MTP draft (measured at load)
        self.mtp_step_ms = 0.0           # costs measured per round, like DFlash2 in production
        self.force = os.environ.get("PLI_HYBRID_FORCE", "").strip().lower()
        self.log = os.environ.get("PLI_HYBRID_LOG", "").strip()
        # FALCON R2 (24/10/2026): PLI_FALCON_HEAD=<json file> {"head": path of the weights, "version": n} ("tete" before 3.7.0); checked every
        # 30 s between two rounds, the MTP head is replaced on the fly by the version published by the trainer.
        self.falcon_head = os.environ.get("PLI_FALCON_HEAD", "").strip()
        self._falcon_seen = (0.0, None)        # (time of the last check, version in service)
        self._mtp_args = getattr(model, "language_model", model).args
        self._global: dict[str, Any] = {"dflash": None, "mtp": None}     # averages of the whole engine
        self._global_rounds = {"dflash": 0, "mtp": 0}
        print(f"[pli] hybrid: DFlash2 ({nodes} drafts at most) and MTP head ({self.mtp_drafts}), chosen per "
              f"conversation{'; frozen on ' + self.force if self.force else ''}", flush=True)

    # -- caches -----------------------------------------------------------------------------------------------------
    def make_cache(self) -> list[Any]:
        caches = list(self.inner.make_cache())
        # also called during construction (measuring the windows), before both drafts are in place
        if getattr(self.mtp, "make_cache", None) is not None:
            caches.append(self.mtp.make_cache())
        if getattr(self, "head_drafts", None) is not None:
            caches.append(self.head_drafts.slot())
        return caches

    def adopt_cache(self, cache: list[Any]) -> list[Any]:
        """A kept prefix, wherever it comes from: its draft caches put back in order, the missing ones new."""

        from tensorfold.families.qwen3_5_moe.mtp import MTPCache

        mc = next((c for c in cache[-2:] if isinstance(c, MTPCache)), None)
        slot = next((c for c in cache[-2:] if isinstance(c, DraftSlot)), None)
        while cache and isinstance(cache[-1], (MTPCache, DraftSlot)):
            cache.pop()
        if mc is not None and getattr(mc, "pli_hybrid", None) is not None:
            # the conversation resumes: its choice and its measurements follow it, in a copy of its own
            e = mc.pli_hybrid
            mc.pli_hybrid = {**e, "t": None, "prev": None, "pos": None, "avg": dict(e["avg"]),
                             "rounds": dict(e["rounds"]), "seen": dict(e.get("seen", {"dflash": 0, "mtp": 0}))}
        cache.append(mc if mc is not None else self.mtp.make_cache())
        cache.append(slot if slot is not None else self.head_drafts.slot())
        return cache

    # -- draft choice -------------------------------------------------------------------------------------------------
    def _state(self, cache: list[Any]) -> dict[str, Any]:
        mc = self._mc(cache)              # on the MTP cache: copied with the checkpoints, not to disk (pli_)
        state = getattr(mc, "pli_hybrid", None)
        if state is None:
            state = {"n": 0, "t": None, "prev": None, "mode": "dflash", "avg": {"dflash": None, "mtp": None},
                     "rounds": {"dflash": 0, "mtp": 0}}
            mc.pli_hybrid = state
        return state

    def _falcon_check(self) -> None:
        """FALCON R2: a newly published version of the MTP head replaces the one in service, between two rounds."""

        if not self.falcon_head:
            return
        now = time.time()
        if now - self._falcon_seen[0] < 30.0:
            return
        self._falcon_seen = (now, self._falcon_seen[1])
        try:
            import json

            from tensorfold.families.qwen3_5_moe import mtp as M

            pub = json.load(open(self.falcon_head))
            version = pub.get("version")
            if version is None or version == self._falcon_seen[1]:
                return
            head_path = pub["head"] if "head" in pub else pub["tete"]       # "tete": written before 3.7.0
            new_head = M.load(Path(head_path), self._mtp_args)
            self.mtp = new_head                 # the head's caches stay: only the drafts change
            self._falcon_seen = (now, version)
            print(f"[pli] falcon: MTP head version {version} in service ({head_path})", flush=True)
        except Exception as e:                  # an unreadable publication changes nothing
            print(f"[pli] falcon: publication ignored ({type(e).__name__}: {e})", flush=True)
            self._falcon_seen = (now, self._falcon_seen[1] if self._falcon_seen[1] is not None else -1)

    def _choose(self, cache: list[Any], tokens: int, end: int | None = None) -> str:
        """The round that just finished counted for its draft, then the draft of the next round.

        ``end``: position after the last kept row; the gap with the previous one gives the tokens written
        by the round (exact), ``tokens`` (kept rows) serves only as a fallback."""

        self._falcon_check()
        state = self._state(cache)
        if end is not None:
            before = state.get("pos")
            state["pos"] = int(end)
            if before is not None and 0 < int(end) - before <= 64:
                tokens = int(end) - before
        now = time.perf_counter()
        if state["prev"] == "mtp" and state.get("k"):
            # acceptance of the MTP head by depth in this conversation: depth i tried, reached if
            # at least i drafts were kept (tokens written - 1)
            tries, hits = state.setdefault("depth", ([0] * 16, [0] * 16))
            kept = max(0, int(tokens) - 1)
            for i in range(1, int(state["k"]) + 1):
                tries[i] += 1
                hits[i] += 1 if kept >= i else 0
        state["k"] = 0
        if state["t"] is not None and state["prev"] is not None and not state.get("skip"):
            ms = (now - state["t"]) * 1e3
            if 0 < ms < GAP_S * 1e3 and state["prev"] == "mtp":
                r = state.get("mtp_round_ms")
                state["mtp_round_ms"] = ms if r is None else r + 0.2 * (ms - r)
            if 0 < ms < GAP_S * 1e3:
                for avg in (state["avg"], self._global):      # the conversation, and the whole engine
                    m = avg[state["prev"]]
                    # damped sums (tokens, ms): their ratio is the speed of the last rounds, without bias
                    avg[state["prev"]] = ((float(tokens), ms) if m is None else
                                          (m[0] * (1 - SMOOTHING) + tokens, m[1] * (1 - SMOOTHING) + ms))
                self._global_rounds[state["prev"]] += 1
                state.setdefault("seen", {"dflash": 0, "mtp": 0})[state["prev"]] += 1
        state["skip"] = False
        state["t"] = now
        state["n"] += 1
        n = state["n"]

        def speed(m: Any) -> float:
            return m[0] / m[1] if m else 0.0

        d, m = state["avg"]["dflash"], state["avg"]["mtp"]
        seen = state.get("seen", {"dflash": 0, "mtp": 0})
        enough = min(seen.values()) >= TRIAL - 1                 # each draft measured over enough rounds here
        probe = n > TRIAL and (n % RETRY) < TRIAL            # now and then, a few rounds of the other
        short = MESSAGES and state.get("messages") is not None and int(state["messages"]) <= MESSAGES
        if self.force in ("dflash", "mtp"):
            mode = self.force
        elif short:                                          # a one-turn question: DFlash2 alone
            mode = "dflash"
        elif enough:                                         # the conversation has measured both: it decides
            state["choice"] = _keep(state.get("choice"), speed(d), speed(m))
            mode = _other(state["choice"]) if probe else state["choice"]
        elif min(self._global_rounds.values()) >= GLOBAL_ROUNDS:
            # the engine has already measured both: no trial, the best for it, the other probed now and then
            g = self._global
            self._gchoice = _keep(getattr(self, "_gchoice", None), speed(g["dflash"]), speed(g["mtp"]))
            mode = _other(self._gchoice) if probe else self._gchoice
        else:                                                # trial: DFlash2 then the MTP head
            mode = "dflash" if seen["dflash"] < TRIAL - 1 else "mtp"
        if mode != state["mode"] and self.log:
            try:
                with open(self.log, "a") as f:
                    d, m = state["avg"]["dflash"], state["avg"]["mtp"]
                    f.write(f"{time.time():.3f} conv {id(state) % 100000} round {n} {state['mode']} -> {mode} "
                            f"dflash {d[0] / d[1] * 1e3 if d else 0:.1f} mtp {m[0] / m[1] * 1e3 if m else 0:.1f} tokens/s\n")
            except Exception:
                pass
        if mode != state["mode"]:
            state["skip"] = True           # the first round after a switch (warm-up) is not counted
        state["mode"] = state["prev"] = mode
        state["rounds"][mode] += 1
        return mode

    def pli_messages(self, cache: list[Any], stream: Any) -> None:
        """The number of messages in the current request's prompt (counted once per request)."""

        if not MESSAGES:
            return
        state = self._state(cache)
        if state.get("stream") == id(stream):
            return
        state["stream"] = id(stream)
        state["messages"] = sum(1 for t in getattr(stream, "prompt_ids", ()) if t == IM_START)

    def _mode(self, cache: list[Any]) -> str:
        return self._state(cache)["mode"]

    # -- prompt prefill -----------------------------------------------------------------------------------------------
    def absorb_draft_taps(self, hidden: Any, next_tokens: Any, cache: list[Any], start: int = 0) -> None:
        """Outside the MTP head's window: DFlash2 alone reads the rows (its taps on the layers)."""

        Qwen35Family.absorb_draft_context(self, hidden, next_tokens, cache, start)

    def absorb_draft_context(self, hidden: Any, next_tokens: Any, cache: list[Any], start: int = 0) -> None:
        Qwen35Family.absorb_draft_context(self, hidden, next_tokens, cache, start)
        Qwen36Family.absorb_draft_context(self, hidden, next_tokens, cache, start)

    # -- tours -------------------------------------------------------------------------------------------------------
    def speculate(self, cache: list[Any], tokens: Any, position: int, sampling: Any, start: int = 0,
                  last_only: bool = False, rows: Sequence[int] | None = None) -> Any:
        import mlx.core as mx

        follow = _tokens(tokens)
        kept = list(rows) if rows is not None else list(range(start, start + len(follow)))
        if TOKEN_THRESHOLD:
            state = self._state(cache)
            state["pli_tokens"] = 0.9 * state.get("pli_tokens", TOKEN_THRESHOLD - 1.0) + 0.1 * len(follow)
        self.head_drafts.read(cache, kept, follow, sampling)
        if self._choose(cache, len(follow), int(position) + len(follow)) == "mtp":
            return Qwen36Family.speculate(self, cache, tokens, position, sampling, start=start, last_only=last_only,
                                          rows=rows)
        Qwen36Family.absorb_kept(self, cache, tokens, position, start=start, rows=rows)
        return mx.array([0], dtype=mx.uint32)

    def settle(self, cache: list[Any], keep: int, first: Any, position: int, sampling: Any, count: int) -> Any:
        if self._mode(cache) == "mtp":
            cache[-1].lazy = False                  # PLI: MTP drafts keep the row sharing
            k = self._mtp_depth(cache, min(int(count), self.mtp_drafts))
            self._state(cache)["k"] = k
            return Qwen36Family.settle(self, cache, keep, first, position, sampling, k)
        if TREES_SHORT_ONLY:
            state = self._state(cache)
            short = MESSAGES and state.get("messages") is not None and int(state["messages"]) <= MESSAGES
            above = TOKEN_THRESHOLD and state.get("pli_tokens", 0.0) >= TOKEN_THRESHOLD
            try:
                cache[-1].pli_force_chain = (not short) or bool(above)
            except Exception:
                pass
        return self.head_drafts.tree(cache, position, sampling, count) if count > 0 else []

    def draft_streams(self, caches: Sequence[list[Any]], follows: Sequence[Sequence[int]],
                      rows: Sequence[Sequence[int]], positions: Sequence[int], samplings: Sequence[Any],
                      depths: Sequence[int]) -> list[Any]:
        modes = [self._choose(c, len(f), int(p) - 1) for c, f, p in zip(caches, follows, positions)]
        out: list[Any] = [[] for _ in caches]
        dfl = [i for i, m in enumerate(modes) if m == "dflash"]
        mtp = [i for i, m in enumerate(modes) if m == "mtp"]
        if mtp:
            res = Qwen36Family.draft_streams(self, [caches[i] for i in mtp], [follows[i] for i in mtp],
                                             [rows[i] for i in mtp], [positions[i] for i in mtp],
                                             [samplings[i] for i in mtp],
                                             [min(int(depths[i]), self.mtp_drafts) for i in mtp])
            for i, r in zip(mtp, res):
                out[i] = r
            for i in mtp:                           # DFlash2 idle: it reads the kept rows
                self.head_drafts.read(caches[i], list(rows[i]), list(follows[i]), samplings[i])
        if dfl:
            for i in dfl:                           # the MTP head idle: it reads the kept rows
                n = len(follows[i])
                Qwen36Family.absorb_kept(self, caches[i], _as_array(follows[i]), int(positions[i]) - 1 - n,
                                         rows=list(rows[i]))
            res = self.head_drafts.draft_streams([caches[i] for i in dfl], [follows[i] for i in dfl],
                                                 [rows[i] for i in dfl], [positions[i] for i in dfl],
                                                 [samplings[i] for i in dfl], [depths[i] for i in dfl])
            for i, r in zip(dfl, res):
                out[i] = r
        return out

    def _mtp_depth(self, cache: list[Any], cap: int) -> int:
        """Like MTP alone: the number of drafts that gives the most tokens per ms, with the cost of one step of the
        head (mtp_step_ms_measured) and the conversation's acceptance by depth (MTP-alone prior at the start)."""

        if cap <= 0:
            return 0
        if os.environ.get("PLI_HYBRID_DEPTH", "0") != "1":     # off: no gain measured (24/10)
            return cap
        state = self._state(cache)
        tries, hits = state.get("depth", ([0] * 16, [0] * 16))
        rnd = float(state.get("mtp_round_ms") or 80.0) - self.mtp_step_ms_measured * float(state.get("k_avg", cap))
        rnd = max(20.0, rnd)
        prior = Qwen36Family.draft_prior
        best, k_best, expected, p0 = 1.0 / rnd, 0, 1.0, 1.0
        for i in range(1, cap + 1):
            p0 *= prior[min(i, len(prior)) - 1]             # conditional prior: chance of reaching depth i
            # 8 prior trials, then the conversation's measurements
            p = (hits[i] + 8 * p0) / (tries[i] + 8)
            expected += p
            v = expected / (rnd + self.mtp_step_ms_measured * i)
            if v > best:
                best, k_best = v, i
        km = state.get("k_avg")
        state["k_avg"] = k_best if km is None else km + 0.2 * (k_best - km)
        return k_best

    def draft_probabilities(self, cache: list[Any]) -> list[float] | None:
        """DFlash2's chances; the MTP head gives none (the engine takes the acceptance by depth)."""

        if self._mode(cache) == "mtp":
            return None
        return self.head_drafts.probabilities(cache)


def _other(mode: str) -> str:
    return "dflash" if mode == "mtp" else "mtp"


def _keep(choice: str | None, vd: float, vm: float) -> str:
    """The kept draft changes only if the other does better by more than MARGIN (otherwise the two alternate)."""

    if choice is None:
        return "mtp" if vm > vd else "dflash"
    current, other = (vm, vd) if choice == "mtp" else (vd, vm)
    return _other(choice) if other > current * (1.0 + MARGIN) else choice


def _as_array(tokens: Sequence[int]) -> Any:
    import mlx.core as mx

    return mx.array([int(t) for t in tokens], dtype=mx.uint32)


__all__ = ["Qwen38Hybrid"]
