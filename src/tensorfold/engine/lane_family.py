"""Verify drafts against position-keyed target samples and roll back every cache to accepted rows so shared and serial rounds agree."""

from __future__ import annotations

from tensorfold.pli.falcon import capture as _falcon
import time

from tensorfold.pli import trace as _tr
from tensorfold.pli import round_profile as _pt
from typing import Any

from tensorfold.engine.family_common import _PROFILE, _ROUND_LOG, drop_spares
from tensorfold.engine.family_depth import DraftDepth, extend_costs
from tensorfold.engine.family_prefill import FamilyPrefill
from tensorfold.engine.family_shared import SharedRounds


# PLI (30/10/2026): replay of the draft lattices logged on 29/10 (PLI_LATTICE_LOG): on the code request, the copy takes
# the round from DFlash2's chain on 32 rounds of 564 where the chain would have kept everything (8 tokens), and gives only 1 to 6
# (5.43 tokens per round replayed against 5.11 real). PLI_HEAD_BEFORE_COPY=1: never a copy when the head has a chain queued; =2: the
# copy only if its first two tokens are those of the head's chain. Off by default (0).
_HEAD_BEFORE_COPY = __import__("os").environ.get("PLI_HEAD_BEFORE_COPY", "0")
# PLI (30/10/2026): PLI_COPY_LOG=file (jsonl, off by default): each round writes its kind and the tokens it gave; a copy round also
# writes the copy, the head's queued chain (full: the head is no longer cut down to one draft) and the match length. Used to
# choose offline when the copy should go before the head. The choice itself does not change.
_COPY_LOG = __import__("os").environ.get("PLI_COPY_LOG", "")
# PLI (30/10/2026): PLI_HEAD_BEFORE_COPY=3: choice learned per stream. At each round where the copy and the head's chain are both
# queued, both are scored afterwards on the written text (tokens they would have kept); moving average (weight
# PLI_HEAD_COPY_WEIGHT, 0.2) of copy minus head; the copy goes first while this average stays above -PLI_HEAD_COPY_THRESHOLD
# (1 token). Offline (logs cj1 and cj18): long code +5.2 % tokens, 18 agent requests -0.25 %.
_HC_WEIGHT = float(__import__("os").environ.get("PLI_HEAD_COPY_WEIGHT", "0.2") or 0.2)
_HC_THRESHOLD = float(__import__("os").environ.get("PLI_HEAD_COPY_THRESHOLD", "1") or 1)


def _kept(prop: list[int], actual: list[int], cap: int) -> int:
    n = 0
    for a, b in zip(prop[:cap], actual):
        if a != b:
            break
        n += 1
    return n + 1


def _score_hc(stream: Any, state: dict) -> None:
    """Score the pending choices whose next 8 tokens are written (or whose stream is finished)."""

    ctx = stream.context
    rest = []
    for b, cp, hd in state["pending"]:
        if len(ctx) >= b + 8 or getattr(stream, "finished", False):
            actual = ctx[b:b + 8]
            d = _kept(cp, actual, len(cp)) - _kept(hd, actual, 7)
            state["avg"] = (1.0 - _HC_WEIGHT) * state["avg"] + _HC_WEIGHT * d
            state["scored"] += 1
        else:
            rest.append((b, cp, hd))
    state["pending"] = rest


def _first_tokens(queued: Any, n: int) -> list[int]:
    """The first n tokens of the head path of the queued drafts (chain, or best-first tree: first child at each step)."""

    import mlx.core as mx

    if isinstance(queued, tuple):
        tokens, parents = queued
        tokens = [int(t) for t in (tokens.tolist() if isinstance(tokens, mx.array) else tokens)]
        parents = [int(q) for q in parents]
        out, cur = [], -1
        while len(out) < n:
            nxt = next((i for i, q in enumerate(parents) if q == cur), None)
            if nxt is None:
                break
            out.append(tokens[nxt]); cur = nxt
        return out
    tokens = queued.tolist() if isinstance(queued, mx.array) else queued
    return [int(t) for t in list(tokens)[:n]]


class FamilyRounds(FamilyPrefill, SharedRounds, DraftDepth):
    """Provide lane-engine rounds for models exposing the ``lane_family`` protocol."""

    # Require enough matching context to reject coincidental copied continuations such as indentation.
    enter_match = 8

    def _family_setup(self) -> None:
        model = self.model
        self._live: list[tuple[Any, list[Any]]] = []
        self.paused: set[str] = set()                # held back from rounds for memory; their state untouched
        self._inflight: dict[str, Any] = {}          # stream id -> next token (GPU array), already queued
        self._next: dict[str, Any] = {}              # stream id -> the head's drafts for the next round
        self._mode: dict[str, str] = {}              # stream id -> "pipe" | "drain" | "verify" | "exit"
        self._depth_state: dict[str, dict[str, Any]] = {}
        self.drafted = 0
        self.accepted = 0
        self.pipelined = bool(getattr(model, "gpu_tokens", False))
        self.family_width = max(1, min(int(getattr(model, "exact_width", 1) or 1), int(self.max_rows)))
        self.max_copy = self.family_width - 1
        # a copy's first window; alone, one that lands whole doubles the next (rows) and one that breaks starts over
        self.first_copy = max(1, min(self.max_copy, int(getattr(model, "first_copy_rows", 0) or self.family_width) - 1))
        self._copy_width: dict[str, int] = {}        # stream id -> tokens its next copy may take while it runs alone
        self._alone = True                           # this step has one live stream: its copies may take their ramp
        self.base_width = min(self.family_width, self.first_copy + 1)   # forced windows and shared rounds keep it
        self.family_mtp = (getattr(model, "mtp", None) is not None and self.family_width >= 2
                           and callable(getattr(model, "speculate", None)))
        self.speculate_early = bool(getattr(model, "speculate_early", True))
        prior = getattr(model, "draft_prior", None)
        if prior:
            self.depth_prior = tuple(float(p) for p in prior)
        # a family whose drafts can cost more than they land lets the depth rule pick plain rounds
        self.plain_guard = bool(getattr(model, "plain_guard", False))
        self.most_drafts = max(0, min(int(getattr(model, "drafts", 1) or 0), self.family_width - 1,
                                      int(self.max_draft)))
        costs = getattr(model, "window_costs", None) or {}
        self.family_costs = {int(w): float(ms) for w, ms in costs.items() if 1 <= int(w) <= self.family_width}
        self.mtp_step_ms = float(getattr(model, "mtp_step_ms", 0.0) or 0.0)
        self._round_ms: dict[int, float] = {}        # measured wall time of a round with d head drafts
        # several live streams share one forward (``model.hidden_rows``) of at most this many rows and streams
        self.family_streams = (self.family_width >= 2 and getattr(model, "streams_exact", True) is True
                               and callable(getattr(model, "hidden_rows", None)))
        self.batch_rows = int(getattr(model, "batch_rows", 0) or 32)
        self.batch_streams = int(getattr(model, "max_streams", 0) or 32) if self.family_streams else 1
        self._served: dict[str, int] = {}            # stream id -> the last shared round it took part in
        self._shared_rounds = 0
        # Allocation uses forward cost by total rows and round overhead by stream count.
        timed = {int(w): float(ms) for table in (costs, getattr(model, "shared_costs", None) or {})
                 for w, ms in table.items() if 1 <= int(w) <= self.batch_rows}
        self.shared_costs = extend_costs(timed, self.batch_rows)
        if timed and self.family_streams and max(timed) < self.batch_rows:
            print(f"[lanes] forward costs timed up to {max(timed)} rows of a shared round's {self.batch_rows}: wider "
                  f"rounds are costed at {timed[max(timed)] / max(timed):.2f} ms a row", flush=True)
        self._overhead_ms: dict[int, float] = {}
        # heads that give each drafted node's chance of landing draft a budget first; rows are allocated after
        self.node_probabilities = callable(getattr(model, "draft_probabilities", None))
        self._granted: dict[str, int] = {}           # stream id -> drafts its last shared round kept
        # PLI: cost arbiter (width of copies and drafts according to the real cost of a row at the current context)
        from tensorfold.pli import arbiter

        self._arbiter = (arbiter.Arbiter(extend_costs(self.family_costs, self.family_width))
                         if arbiter.ACTIVE and self.family_costs else None)
        self._grammar_window: dict[str, Any] = {}    # stream id -> the window its grammar kept, masked at the draw

    def _draw(self, logits: Any, sampling: Any, positions: Any) -> Any:
        """Use the same model sampler or GPU sampler for every draw so streams match their own serial runs."""

        import mlx.core as mx

        from tensorfold.engine.gpu_sampling import sample as gpu_sample

        own = getattr(self.model, "sample", None)
        if callable(own):
            out = own(logits, sampling, positions)
            return out if isinstance(out, mx.array) else mx.array([int(t) for t in out], dtype=mx.uint32)
        return gpu_sample(logits, sampling, positions)

    def _family_step(self) -> dict[str, list[int]]:
        landed: dict[str, list[int]] = {}
        for stream_id, queued in list(self._next.items()):
            if callable(queued):                  # a new stream's first drafts, settled before anything reads them
                self._next[stream_id] = queued()
        live = [(s, c) for s, c in self._live if not s.finished and s.stream_id not in self.paused]
        messages = getattr(self.model, "pli_messages", None)     # PLI: number of messages in the prompt (hybrid)
        if callable(messages):
            for s, c in live:
                messages(c, s)
        if len(live) > 1 and self.family_streams:
            # a shared round is synchronous: a stream that ran a step ahead lands its queued token first
            for stream, _ in live:
                if stream.stream_id in self._inflight:
                    landed[stream.stream_id] = self._land_inflight(stream)
            live = [(s, c) for s, c in live if not s.finished]
        self._alone = len(live) == 1
        if len(live) > 1 and self.family_streams:
            live = self._take_turns(live)
            self._shared_rounds += 1
            for stream, _ in live:
                self._served[stream.stream_id] = self._shared_rounds
            for stream_id, (got, rows, keep) in self._family_round_streams(live).items():
                landed[stream_id] = landed.get(stream_id, []) + got
        else:
            for stream, cache in live:
                started = time.perf_counter()
                if (self.family_mtp and stream.drafts) or not self.pipelined or stream.constraint is not None:
                    got, rows, keep = self._family_round(stream, cache)
                else:
                    got, rows, keep = self._pipelined_round(stream, cache)
                landed[stream.stream_id] = landed.get(stream.stream_id, []) + got
                stream.min_rows = rows if not stream.min_rows else min(stream.min_rows, rows)
                ms = (time.perf_counter() - started) * 1e3
                from tensorfold.engine.lane_engine import RoundStats

                self.round_stats.append(RoundStats(
                    streams=1, width=rows, rows=rows, ragged=False, rollbacks=int(0 < keep < rows),
                    committed=len(got), forward_ms=ms, finalize_ms=0.0, rollback_ms=0.0, total_ms=ms,
                    started_at=started))
                if _tr.active():
                    prev_end = getattr(self, "_tr_end", 0.0)
                    parts = self.__dict__.pop("_tr_parts", None) or {"kind": "pipeline"}
                    _tr.pt("round", stream.stream_id, t=started, ms=ms, rows=rows, kept=keep, tokens=len(got),
                           ctx=int(stream.cache_len), gap_ms=(started - prev_end) * 1e3 if prev_end else None,
                           **parts)
                    self._tr_end = time.perf_counter()
        from tensorfold.pli import kv_sharing as _kvp
        for stream, cache in self._live:
            if stream.finished:
                stream.finished_at = time.perf_counter()
                self._release_stream_state(stream.stream_id)
                if self._keeps_decoded(stream):
                    # one-token rounds leave the last token absorbed when they ran ahead; ``cache_len`` counts it
                    # PLI: no copy, the request's KV layers become the checkpoint's
                    self.finished_caches[stream.stream_id] = (stream.context[: stream.cache_len],
                                                              drop_spares(_kvp.retain(cache, self.copy_single_cache)
                                                                          if _kvp.ACTIVE else self.copy_single_cache(cache)))
                elif _kvp.ACTIVE:
                    _kvp.release(cache)
        self._live = [(s, c) for s, c in self._live if not s.finished]
        return landed

    @staticmethod
    def _forced_next(stream: Any, drawn: Any) -> int | None:
        """The token the thinking budget or a required call's fix writes at the next position instead of ``drawn``."""

        if stream.force:
            return int(stream.force.pop(0))
        if stream.think_cut([-1]) == 0:
            return stream.start_close()
        gate = stream.call_gate
        hit = gate.cut([int(drawn.item())]) if gate is not None and gate.watching else None   # read only then
        if hit is None:
            return None
        stream.force = list(hit[1][1:])
        return int(hit[1][0])

    def _copy_proposal(self, stream: Any, min_match: int | None = None) -> list[int]:
        """Propose copied spans backed by ``enter_match`` matching tokens or a tool call's known structure."""

        if stream.proposer is None or self.max_copy <= 0 or stream.force:
            return []
        try:
            width = self._copy_width.get(stream.stream_id, self.first_copy) if self._alone else self.first_copy
            if self._arbiter is not None and self._alone:
                width = min(width, self._arbiter.copy_width(stream.stream_id, stream.cache_len, width))
                if width < 2:
                    return []
            copied = [int(t) for t in stream.proposer.propose(stream.context, min(width, stream.draft_room - 1))]
        except Exception:  # noqa: BLE001 - a proposer must never break a stream
            return []
        need = self.enter_match if min_match is None else min_match
        if len(copied) < 2 or int(getattr(stream.proposer, "last_match", 0) or 0) < need:
            return []
        return copied

    def _pipelined_round(self, stream: Any, cache: list[Any]) -> tuple[list[int], int, int]:
        """One token, the next forward queued first; a copied continuation ahead switches to verify windows."""

        import mlx.core as mx

        mode = self._mode.get(stream.stream_id, "pipe")
        if mode in ("verify", "exit"):
            proposal = self._copy_proposal(stream) if mode == "verify" and self.family_width >= 2 else []
            if proposal:
                got, rows, keep = self._family_round(stream, cache, copied=proposal)
                if keep == 1 or stream.force:
                    self._mode[stream.stream_id] = "exit"
                return got, rows, keep
            # nothing to copy: back to steps queued ahead (this one lands next round)
            self._mode[stream.stream_id] = "pipe"
            self._queue_next(stream, cache, mx.array([stream.pending[-1]], dtype=mx.uint32))
            return [], 1, 1
        current = self._inflight.pop(stream.stream_id)
        forced = self._forced_next(stream, current)
        if forced is not None:
            current = mx.array([forced], dtype=mx.uint32)   # the thinking budget's or the call's token, not the sample
        if mode == "drain":
            token = int(current.item())                     # the last queued step: no new one
            self._mode[stream.stream_id] = "verify"
        else:
            self._queue_next(stream, cache, current)        # the GPU starts the next step first
            token = int(current.item())
        stream.rounds += 1
        got = stream.commit([token])
        stream.pending = [token]
        if (self.family_width >= 2 and self._mode.get(stream.stream_id, "pipe") == "pipe" and not stream.finished
                and stream.drafts and self._copy_proposal(stream)):
            self._mode[stream.stream_id] = "drain"          # a copy window is ahead: land the queued step
        return got, 1, 1

    def _plan_window(self, stream: Any, copied: list[int] | None = None
                     ) -> tuple[str, Any, list[int], list[int] | None]:
        """Return (kind, drafts, forced, parents), prioritizing forced tokens then copies then head drafts; tree parents precede children and -1 denotes the pending row."""

        import mlx.core as mx

        queued = self._next.pop(stream.stream_id, None)
        forced = []
        if stream.force:
            width = min(self.base_width, self.batch_rows) if stream.drafts else 1
            forced = stream.force[:width - 1]
            del stream.force[:len(forced)]
            return "forced", forced, forced, None
        if copied is None:
            copied = [] if not stream.drafts else self._copy_proposal(stream)
            if copied and _COPY_LOG and queued is not None:
                stream._cj = {"copy": [int(t) for t in copied], "head": _first_tokens(queued, 8),
                              "match": int(getattr(stream.proposer, "last_match", 0) or 0)}
            if copied and _HEAD_BEFORE_COPY == "3" and queued is not None:
                hc = stream.__dict__.setdefault("_hc", {"avg": 0.0, "pending": [], "scored": 0})
                _score_hc(stream, hc)
                head = _first_tokens(queued, 7)
                if len(head) >= 4:                    # a head chain to score (not the MTP head cut down to one draft)
                    hc["pending"].append((int(stream.context_len), [int(t) for t in copied], head))
                    if hc["avg"] < -_HC_THRESHOLD:
                        copied = []                   # PLI: the head did better than the copy on this stream
            elif copied and _HEAD_BEFORE_COPY in ("1", "2") and queued is not None:
                head = _first_tokens(queued, 2)
                if len(head) >= 2 and (_HEAD_BEFORE_COPY == "1" or [int(t) for t in copied[:2]] != head):
                    copied = []                       # PLI: the head's chain goes before the copy
        if copied:
            return "copy", copied, forced, None
        if stream.drafts and queued is not None:
            parents = None
            if isinstance(queued, tuple):
                queued, tree = queued
                tree = [int(q) for q in tree]
                parents = None if tree == list(range(-1, len(tree) - 1)) else tree
                if parents is not None:
                    queued = [int(t) for t in (queued.tolist() if isinstance(queued, mx.array) else queued)]
            count = int(queued.shape[0]) if isinstance(queued, mx.array) else len(queued)
            if count:
                return "head", queued, forced, parents
        return "none", [], forced, None

    def _constrained(self, stream: Any, kind: str, drafts: Any, forced: list[int], parents: list[int] | None
                     ) -> tuple[str, Any, list[int], list[int] | None]:
        """The window a reply's grammar keeps (drafts it rejects are cut) and each row's mask, kept for the draw."""

        import mlx.core as mx

        from tensorfold.engine.grammar import GrammarError

        tokens = [int(t) for t in (drafts.tolist() if isinstance(drafts, mx.array) else drafts)]
        rows = self._row_parents(1 + len(tokens), parents)
        try:
            window = stream.constraint.window([int(stream.pending[-1]), *tokens], rows)
            if kind == "forced" and len(window.tokens) != len(rows):
                raise GrammarError("the reply's grammar rejects a forced token")
        except GrammarError as exc:                          # this stream ends; its one row lands nothing
            stream.fail(exc)
            return "none", [], [], None
        self._grammar_window[stream.stream_id] = window
        kept = [q - 1 if q > 0 else -1 for q in window.parents[1:]]
        tree = None if parents is None else kept
        return kind, window.tokens[1:], forced if kind != "forced" else window.tokens[1:], tree

    @staticmethod
    def _window_tokens(stream: Any, drafts: Any) -> Any:
        """The window's input tokens [rows] (uint32): the pending token, then the drafts."""

        import mlx.core as mx

        pending = mx.array([stream.pending[-1]], dtype=mx.uint32)
        if isinstance(drafts, mx.array):
            return mx.concatenate([pending, drafts.astype(mx.uint32)])
        return mx.array([stream.pending[-1], *[int(t) for t in drafts]], dtype=mx.uint32)

    @staticmethod
    def _row_parents(rows: int, parents: list[int] | None) -> list[int]:
        """Each window row's parent row (row 0 is the pending token): a chain when ``parents`` is None."""

        if parents is None:
            return [-1, *range(rows - 1)]
        return [-1, *[0 if q < 0 else q + 1 for q in parents]]

    def _conclude(self, stream: Any, kind: str, forced: list[int], sampled: list[int], window: list[int],
                  rows_parents: list[int]) -> tuple[list[int], list[int], int | None, list[int]]:
        """Commit the target-sampled path after the thinking-budget cut, returning landed tokens, kept rows, the cut, and each kept row's following token."""

        from tensorfold.kernels.qwen.dense.v1.lane_tree import accept_path, tree_paths

        rows = len(window)
        if kind == "forced":
            path = list(range(rows))
            bonus = stream.force.pop(0) if stream.force else sampled[-1]
            committed = [*forced, bonus]
        else:
            path = accept_path(window, rows_parents, sampled)
            committed = [*[window[r] for r in path[1:]], sampled[path[-1]]]
            if rows > 1:
                accepted = len(path) - 1
                self.drafted += rows - 1
                self.accepted += accepted
                stream.drafted += rows - 1
                stream.accepted += accepted
                if kind == "copy":
                    observe = getattr(stream.proposer, "observe", None)
                    if callable(observe):
                        observe(rows - 1, accepted)
                    if self._arbiter is not None:
                        self._arbiter.copy_seen(stream.stream_id, rows - 1, accepted)
                    if self._alone:
                        width = self._copy_width.get(stream.stream_id, self.first_copy)
                        self._copy_width[stream.stream_id] = (min(self.max_copy, 2 * width + 1)
                                                              if accepted == rows - 1 else self.first_copy)
                elif kind == "head":
                    self._observe_depth(stream, max(tree_paths(rows_parents)[0]), accepted)
                    expected = getattr(self, "_pli_expected", {}).pop(stream.stream_id, None)
                    if expected:                        # PLI (item D, PLI_TREE_CORRECT): observed acceptance against expected
                        self._pli_num = getattr(self, "_pli_num", 0.0) * 0.97 + accepted
                        self._pli_den = getattr(self, "_pli_den", 0.0) * 0.97 + float(expected)
                        self._pli_ratio = min(1.5, max(0.3, self._pli_num / self._pli_den))
                    # PLI (recall v2, PLI_RECALL): the register's copy is compared with the stream's recent drafts
                    head = getattr(stream.proposer, "observe_head", None) or getattr(
                        getattr(stream.proposer, "fallback", None), "observe_head", None)
                    if callable(head):
                        head(accepted)
        cut = stream.think_cut(committed)
        if cut is not None:
            path = path[:cut + 1]
            committed = [*committed[:cut], stream.start_close()]
        elif stream.call_gate is not None and (hit := stream.call_gate.cut(committed)) is not None:
            cut, fix = hit
            path = path[:cut + 1]
            committed = [*committed[:cut], fix[0]]
            stream.force = list(fix[1:])        # the fix's rest, forced like the thinking budget's close
        stream.rounds += 1
        got = stream.commit(committed)
        if stream.finished and len(got) < len(path):
            # Keep only rows whose tokens landed, including budget cuts, so retained caches match committed tokens.
            path = path[:len(got) + 1]
        keep = len(path)
        stream.cache_len += keep
        stream.pending = [committed[keep - 1]]
        return got, path, cut, committed[:keep]

    @staticmethod
    def _is_prefix(path: list[int]) -> bool:
        return path == list(range(len(path)))

    def _head_is_mtp(self, stream: Any) -> bool:
        """PLI (30/10/2026): true if this stream's head drafts step by step (MTP): a full chain costs one step per draft there,
        whereas a DFlash2 lattice gives it at once (tc18-3: +1 to +6 ms per round on the agent requests, served with MTP)."""

        model = getattr(self, "model", None)
        if model is None:
            return True                        # no known model: the behaviour before (a single draft)
        mode = getattr(model, "_mode", None)
        if mode is None:
            return getattr(model, "head_drafts", None) is None
        cache = next((c for s, c in self._live if s is stream), None)
        try:
            return cache is not None and mode(cache) == "mtp"
        except Exception:  # noqa: BLE001 - the copy choice must never break a stream
            return True

    def _head_depth(self, stream: Any, budget: int | None = None) -> int:
        if stream.finished:
            depth = 0
        elif budget is not None:
            depth = budget
        elif self.node_probabilities:
            depth = min(self.most_drafts, max(1, stream.draft_room - 1))
        else:
            depth = self._depth(stream)
        if stream.force:
            return 0                   # the next round verifies the thinking budget's forced tokens
        if depth and ((_HEAD_BEFORE_COPY == "0" and not _COPY_LOG) or self._head_is_mtp(stream)) and self._copy_proposal(stream):
            return 1                   # a copied continuation is ahead: one head draft, in case it is gone
        return depth

    def _draft_late(self, stream: Any, cache: list[Any], position: int, follow: list[int], rows: list[int],
                    budget: int | None = None) -> None:
        """Read kept forward rows with their following tokens and queue the next round's drafts."""

        import mlx.core as mx

        model = self.model
        depth = self._head_depth(stream, budget)
        where = ({"start": rows[0]} if rows == list(range(rows[0], rows[0] + len(rows))) else {"rows": list(rows)})
        absorb = getattr(model, "absorb_kept", None)
        if depth == 0 and callable(absorb):                  # a plain round: the head reads the kept rows, drafts none
            absorb(cache, mx.array(follow, dtype=mx.uint32), position, **where)
            self._next[stream.stream_id] = []
            return
        heads = model.speculate(cache, mx.array(follow, dtype=mx.uint32), position, stream.sampling, **where,
                                last_only=True)
        self._next[stream.stream_id] = model.settle(cache, len(follow), heads[-1], stream.cache_len + 1,
                                                    stream.sampling, depth)

    def _family_round(self, stream: Any, cache: list[Any], copied: list[int] | None = None
                      ) -> tuple[list[int], int, int]:
        """Verify pending and drafted tokens together, keeping drafts until the first sample mismatch; return committed tokens, rows, and kept rows."""

        import mlx.core as mx

        model = self.model
        started = time.perf_counter()
        position = stream.cache_len
        from tensorfold.kernels.qwen.dense.v1.lane_tree import tree_paths

        kind, drafts, forced, parents = self._plan_window(stream, copied)
        # PLI: a chain drawn on the GPU (lazy) is verified whole, with no sharing decided on the CPU
        if kind == "head" and self.node_probabilities and not getattr(cache[-1], "lazy", False):
            plan = [stream, cache, position, kind, drafts, forced, parents]
            self._allocate([plan])
            kind, drafts, parents = plan[3], plan[4], plan[6]
        if stream.constraint is not None:                    # after the rows are allocated: the window verified
            kind, drafts, forced, parents = self._constrained(stream, kind, drafts, forced, parents)
        inputs = self._window_tokens(stream, drafts)
        rows = int(inputs.shape[0])
        rows_parents = self._row_parents(rows, parents)
        depths = tree_paths(rows_parents)[0]
        tree = {} if parents is None else {"parents": rows_parents}
        sk = _pt.kind_of(kind, parents, position) if _pt.ACTIVE else kind
        if _pt.ACTIVE:
            t_pt = _pt.mark(sk, "before", started)
        hidden = model.hidden(inputs.reshape(1, rows), cache, **tree)
        if _pt.ACTIVE:
            t_pt = _pt.mark(sk, "verify", t_pt, hidden)
        logits = model.head(hidden)
        logits = logits.reshape(logits.shape[1:])            # [R, V] as a view (MLX's [0] is a gather)
        if _pt.ACTIVE:
            t_pt = _pt.mark(sk, "head", t_pt, logits)
        window = self._grammar_window.pop(stream.stream_id, None)
        if window is not None:                               # each row masked along its own path
            logits = stream.constraint.mask(logits, window)
        tokens = self._draw(logits, stream.sampling, [position + 1 + d for d in depths])
        if _pt.ACTIVE:
            t_pt = _pt.mark(sk, "draw", t_pt, tokens)
        speculate = (self.family_mtp and stream.drafts and kind != "forced" and self.speculate_early
                     and parents is None)
        parts = [tokens]
        if speculate:
            # the head's first draft for every row, queued behind the verify before anything is read
            firsts = model.speculate(cache, tokens, position, stream.sampling)
            parts.append(firsts.astype(tokens.dtype))
            prepare = getattr(model, "prepare_settle", None)
            if prepare is not None:                  # the chain's first step built while the GPU verifies
                prepare(cache, firsts, position, stream.sampling)
        if isinstance(drafts, mx.array):
            parts.append(drafts.astype(tokens.dtype))
        built = time.perf_counter()
        values = [int(t) for t in (mx.concatenate(parts) if len(parts) > 1 else tokens).tolist()]
        read = time.perf_counter()
        sampled = values[:rows]
        firsts = values[rows:2 * rows] if speculate else []
        proposed = values[-(rows - 1):] if isinstance(drafts, mx.array) and rows > 1 else [int(t) for t in drafts]
        window = [int(stream.pending[-1]), *proposed]          # the rows' tokens, the pending one first
        got, path, cut, follow = self._conclude(stream, kind, forced, sampled, window, rows_parents)
        if _COPY_LOG:
            import json as _json

            cj = getattr(stream, "_cj", None)
            stream._cj = None
            rec = {"stream": str(stream.stream_id), "pos": int(position), "kind": kind, "given": [int(t) for t in got]}
            if cj is not None and kind == "copy":
                rec.update(cj)
            with open(_COPY_LOG, "a") as handle:
                handle.write(_json.dumps(rec) + "\n")
        if kind == "head" and parents is not None and getattr(cache[-1], "kf_nodes", None):
            from tensorfold.pli import kflash as _kf          # PLI: nodes tried and accepted this round

            _kf.observe_round(cache[-1], rows_parents, path)
        _falcon.seen("alone", kind, cut, self._is_prefix(path))
        if _falcon.active() and kind != "forced" and cut is None and self._is_prefix(path):
            _falcon.capture(stream.stream_id, position, hidden[0], logits, path, follow)   # PLI: FALCON R0
        keep = len(path)
        if _pt.ACTIVE:
            t_pt = _pt.mark(sk, "read_and_conclude", t_pt)
        if keep < rows or parents is not None:      # a tree commits in keep_rows, even one kept whole
            model.keep_rows(cache, rows, keep if self._is_prefix(path) else path)
        if self.family_mtp and stream.drafts:
            if speculate and cut is None:
                self._next[stream.stream_id] = model.settle(cache, keep, firsts[path[-1]], stream.cache_len + 1,
                                                            stream.sampling, self._head_depth(stream))
            else:
                # late speculation, a forced round, or the budget replaced the kept row's next token
                if speculate:
                    model.unspeculate(cache)
                self._draft_late(stream, cache, position, follow, path)
            if kind == "head" or (kind == "none" and self.plain_guard):
                self._observe_cost(len(proposed), (time.perf_counter() - started) * 1e3,
                                   initializing=stream.rounds == 1, stream=stream)
        if _pt.ACTIVE:
            nxt = self._next.get(stream.stream_id)
            t_pt = _pt.mark(sk, "next_draft", t_pt, *([nxt] if hasattr(nxt, "shape") else []))
            _pt.end_of_round(sk, started, rows, keep)
        if _PROFILE:
            self._profile(rows, built - started, read - built, time.perf_counter() - read)
        ms = (time.perf_counter() - started) * 1e3
        if kind == "head":
            self._observe_overhead(1, rows, ms)
        if self._arbiter is not None and stream.rounds > 1 and kind != "forced":
            self._arbiter.observe(rows, position, ms, parents is not None)
        if _ROUND_LOG:
            with open(_ROUND_LOG, "a") as handle:
                handle.write(f"{position} {kind} {rows} {keep} {ms:.2f}\n")
        if _tr.active():     # PLI trace: split of the round, queued by _family_step with the gap between rounds
            self._tr_parts = {"pos": position, "kind": kind, "proposed": len(proposed),
                              "build_ms": (built - started) * 1e3, "gpu_wait_ms": (read - built) * 1e3,
                              "after_ms": (time.perf_counter() - read) * 1e3}
        return got, rows, keep

    def _profile(self, rows: int, build: float, wait: float, after: float) -> None:
        acc = self.__dict__.setdefault("_prof", [0, 0.0, 0.0, 0.0, 0.0])
        acc[0] += 1
        acc[1] += rows
        acc[2] += build * 1e3
        acc[3] += wait * 1e3
        acc[4] += after * 1e3
        if acc[0] >= 200:
            n = acc[0]
            print(f"[lanes] family rounds: {acc[1] / n:.2f} rows, build {acc[2] / n:.2f} ms, wait for the GPU "
                  f"{acc[3] / n:.2f}, after the read {acc[4] / n:.2f} (mean of {n})", flush=True)
            self._prof = [0, 0.0, 0.0, 0.0, 0.0]

    def _release_stream_state(self, stream_id: str) -> None:
        for table in (self._inflight, self._next, self._mode, self._depth_state, self._served, self._granted,
                      self._grammar_window, self._copy_width):
            table.pop(stream_id, None)
        if getattr(self, "_arbiter", None) is not None:
            self._arbiter.forget(stream_id)

    def _family_reset(self) -> None:
        self._live = []
        self._inflight = {}
        self._next = {}
        self._mode = {}
        self._depth_state = {}
        self._served = {}
        self._granted = {}
        self._grammar_window = {}
        self._copy_width = {}

    def _family_summary(self) -> dict[str, Any]:
        return {"engine": "lanes", "family": True, "rounds": len(self.round_stats), "streams": len(self.streams),
                "drafted": self.drafted, "accepted": self.accepted, "exact_width": self.family_width}
