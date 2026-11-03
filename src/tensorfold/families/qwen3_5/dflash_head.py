"""DFlash2 uses each stream's final cache entry for drafter state and committed target taps to build probability-ranked drafts."""

from __future__ import annotations

import heapq
import itertools
import json
import math
import os
from typing import Any, Sequence

import mlx.core as mx

_numbers = itertools.count(1)



# PLI (item D, 24/10/2026): draft trees only under PLI_TREE_CTX_MAX context positions (0: everywhere)
_TREE_CTX_MAX = max(0, int(__import__("os").environ.get("PLI_TREE_CTX_MAX", "0") or 0))
# PLI (28/10/2026, Kapitsa's method: no CPU round trip between the draft and the verify): PLI_GPU_CHAIN=1 draws DFlash2's
# chain (greedy) on the GPU from the lattice (candidates, scores, selector); the drafts stay a lazy array that the verify reads
# directly, the CPU reads them only with the round's result. Every draft of the chain is verified (with V8, one more row costs
# almost nothing). Off by default.
_GPU_CHAIN = __import__("os").environ.get("PLI_GPU_CHAIN", "0") == "1"

class DraftSlot:
    """A stream's DFlash2 proposer, its cache list's last entry; a copy keeps only prompt taps not read yet."""

    keys = None
    stored = False                                       # left out of snapshots on disk; adopt_cache adds a new one

    def __init__(self, drafter: Any, chains: bool = False) -> None:
        self.drafter = drafter
        self.chains = bool(chains)                       # one child a node: the tree search draws a chain
        self.proposer: Any = None
        self.anchor = 0
        self.kept: list[int] = []                        # the last round's committed tokens, the anchor last
        self.chances: list[float] | None = None          # the last drafts' chances of landing
        self.lazy = False                                # PLI: the last drafts are a lazy GPU chain
        self.number = next(_numbers)

    def _unread(self) -> Any:
        """The proposer while it holds only a prompt's taps (a prefill's state), else None."""

        proposer = self.proposer
        if self.kept or not getattr(proposer, "ready", False) or getattr(proposer, "context", None) is None:
            return None
        return proposer

    @property
    def state(self) -> list[Any]:
        unread = self._unread()
        return [] if unread is None else [unread.context]

    @property
    def nbytes(self) -> int:
        """The drafter context's arrays (its sliding window: fixed memory a stream holds; a copy starts empty)."""

        proposer = self.proposer
        if proposer is None:
            return 0
        arrays = [getattr(proposer, "context", None)]
        for item in getattr(proposer, "cache", None) or []:
            arrays += [getattr(item, "keys", None), getattr(item, "values", None)]
        return sum(int(getattr(a, "nbytes", 0) or 0) for a in arrays if a is not None)

    def __copy__(self) -> "DraftSlot":
        slot = DraftSlot(self.drafter, self.chains)
        unread = self._unread()
        if unread is not None:
            proposer = slot.get(unread.sampling)
            proposer.context, proposer.ready = unread.context, True
            for mine, theirs in zip(proposer.cache, unread.cache):
                mine.offset = theirs.offset
        return slot

    def get(self, sampling: Any) -> Any:
        if self.proposer is None:
            proposer = self.drafter.proposer(copy=None, sampling=sampling)
            proposer.ngram_weight = 0.0                  # the prior and the traces read the whole token context
            from tensorfold.pli import lattice_log as _tj

            proposer.trace_path = _tj.path(proposer)     # PLI (29/10): empty unless PLI_LATTICE_LOG
            proposer.capture_dir = ""
            _tj.codes(proposer)
            from tensorfold.pli import kflash as _kf

            if _kf.ACTIVE and not self.chains and hasattr(proposer, "tree_block") and hasattr(self.drafter, "block_size"):
                proposer.tree_block = int(self.drafter.block_size)      # PLI: DFlash2's trained block (8)
            if self.chains:
                # DFlash2's own chain: its trained block, one child a node, unary + pairwise / T + the keyed noise
                proposer.tree_children, proposer.tree_edge, proposer.tree_noise = 1, 1.0, 1.0
                proposer.tree_block = int(self.drafter.block_size)
            self.proposer = proposer
        self.proposer.sampling = sampling
        return self.proposer


class _Context:
    """What a lattice reads of a stream's tokens: their count and the last one (the anchor)."""

    __slots__ = ("length", "last")

    def __init__(self, length: int, last: int) -> None:
        self.length, self.last = int(length), int(last)

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: Any) -> Any:
        if isinstance(index, slice):
            return [self.last] if (index.start or 0) < 0 else []     # history for a prior, which this head leaves off
        if index == -1 or index == self.length - 1:
            return self.last
        raise IndexError("a draft context holds only its last token")


class DFlashHead:
    """The draft-head half of the family protocol over a ``DFlashDrafter``."""

    def __init__(self, drafter: Any, nodes: int = 15, calibration: dict[str, Any] | None = None,
                 chains: bool = False) -> None:
        self.drafter = drafter
        self.nodes = int(nodes)
        self.chains = bool(chains)
        self.calibration = dict(calibration or {})     # "greedy" / "sampled" -> drafters.calibration.Calibration
        self.log_path = os.environ.get("PLI_DRAFT_LOG", "")
        # DFlash (v1) has no candidate selector, so no lattice: chains of each position's own argmax (``block_chain``)
        model = getattr(drafter, "model", None)
        self.v1 = model is not None and not hasattr(model, "candidate_selector")

    def slot(self) -> DraftSlot:
        return DraftSlot(self.drafter, self.chains)

    def absorb(self, cache: list[Any], first: int, rows: int, row: int = 0) -> None:
        """Absorb forward rows [row, row + rows), starting at position first, into the stream's drafter context."""

        taps = self.drafter.taps()
        if taps is None:
            return
        taps = taps[:, row:row + rows]
        proposer = cache[-1].get(getattr(cache[-1].proposer, "sampling", None))
        window = self.drafter.window
        if not proposer.ready:
            if window and rows > window:
                taps, first = taps[:, -window:], first + rows - window
            for item in proposer.cache:
                item.offset = first
            proposer.context, proposer.ready = mx.contiguous(taps), True
        else:
            proposer.absorb(taps)
            held = int(proposer.context.shape[1])
            if window and held > window:
                proposer.context = proposer.context[:, held - window:]
                for item in proposer.cache:
                    item.offset += held - window
        # evaluated each chunk: a lazy context would hold every earlier chunk's taps until the first draft
        mx.async_eval(proposer.context)

    def read(self, cache: list[Any], rows: Sequence[int], follow: Sequence[int], sampling: Any) -> None:
        """Absorb the stream's kept rows and committed follow tokens, with the new pending token last."""

        proposer = cache[-1].get(sampling)
        taps = self.drafter.taps()
        if taps is not None and rows:
            proposer.absorb(mx.take(taps, mx.array([int(r) for r in rows], dtype=mx.int32), axis=1))
        cache[-1].kept = [int(t) for t in follow]
        cache[-1].anchor = cache[-1].kept[-1]

    def tree(self, cache: list[Any], position: int, sampling: Any, nodes: int) -> Any:
        """The stream's next drafts after its anchor, for positions ``position`` ..: a tree of up to ``nodes``."""

        proposer = cache[-1].get(sampling)
        context = _Context(int(position), cache[-1].anchor)
        cache[-1].lazy = False
        if _GPU_CHAIN and self.chains and sampling is None and not self.v1:
            lazy = self._gpu_chain(cache[-1], proposer, context, nodes)
            if lazy is not None:
                return lazy
        if self.v1:
            cache[-1].chances = None
            tokens = proposer.propose(context, min(int(nodes), self.nodes))
            return as_drafts((tokens, list(range(-1, len(tokens) - 1))), nodes)
        if not self.chains and ((_TREE_CTX_MAX and int(position) >= _TREE_CTX_MAX)
                                or getattr(cache[-1], "pli_force_chain", False)):
            # PLI (item D, 24/10/2026, arb5): past PLI_TREE_CTX_MAX positions, the DFlash2 chain (DraftSlot settings for chains):
            # trees gain 5 to 12 % under 25K and lose 4 to 9 % beyond (tree attention and commit more expensive at long context)
            saved = (proposer.tree_children, proposer.tree_edge, proposer.tree_noise, proposer.tree_block)
            proposer.tree_children, proposer.tree_edge, proposer.tree_noise = 1, 1.0, 1.0
            proposer.tree_block = int(self.drafter.block_size)
            try:
                tree = proposer._finish_tree(context, proposer._start_tree(context, self.nodes))
            finally:
                proposer.tree_children, proposer.tree_edge, proposer.tree_noise, proposer.tree_block = saved
            return self._drafts(cache[-1], tree, position, sampling, nodes)
        tree = proposer._finish_tree(context, proposer._start_tree(context, self.nodes))
        return self._drafts(cache[-1], tree, position, sampling, nodes)

    def _gpu_chain(self, slot: DraftSlot, proposer: Any, context: Any, nodes: int) -> Any:
        """PLI: DFlash2's greedy chain walked on the GPU over its lattice (``best_first_tree`` with one child a node:
        each depth's argmax of unary + edge x pairwise), returned lazy; None when the prelude has nothing to draft."""

        import time

        state = proposer._tree_prelude(context, self.nodes)
        if state[0] != "need":
            return None
        _, block, copy_branch, max_nodes = state
        if copy_branch:
            return None
        started = time.perf_counter()
        cands, unary, hproj = proposer._lattice(context, block)
        selector = self.drafter.model.candidate_selector
        codes = getattr(self.drafter, "_codes_mx", None)
        if codes is None:
            codes = self.drafter._codes_mx = (selector.predecessor_codebook.weight.astype(mx.float32),
                                              selector.successor_codebook.weight.astype(mx.float32))
        pred, succ = codes
        depth = min(int(cands.shape[0]), int(max_nodes), int(nodes))
        edge = float(proposer.tree_edge)
        prev = mx.array([int(context[-1])], dtype=mx.int32)
        out = []
        for d in range(depth):
            table = mx.take(succ, cands[d].astype(mx.int32), axis=0)                 # [K, R]
            edges = table @ (mx.take(pred, prev, axis=0)[0] * hproj[0, d])         # [K]
            j = mx.argmax(unary[d].astype(mx.float32) + edge * edges)
            prev = mx.take(cands[d], j.reshape(1)).astype(mx.int32)
            out.append(prev)
        tokens = mx.concatenate(out).astype(mx.uint32)
        mx.async_eval(tokens)
        anchor = len(context) - 1
        extra = int(proposer.cache[0].offset) - anchor
        if extra > 0:
            proposer.drafter._trim(proposer.cache, extra)
        proposer.context = None
        proposer.last_scores = None
        proposer.build_ms += (time.perf_counter() - started) * 1e3
        proposer.proposals += 1
        proposer.proposed_tokens += depth
        proposer.last_confident = True
        slot.chances = None
        slot.lazy = True
        return tokens

    def probabilities(self, cache: list[Any]) -> list[float] | None:
        return cache[-1].chances

    def _drafts(self, slot: DraftSlot, tree: Any, position: int, sampling: Any, nodes: int) -> Any:
        """Order drafts by chance with parents before children, storing calibrated probabilities or e^score in slot.chances."""

        tokens, parents = [int(t) for t in tree[0]], [int(q) for q in tree[1]]
        scores = slot.proposer.last_scores if slot.proposer is not None else None
        kf_nodes = getattr(slot.proposer, "last_kf", None) if slot.proposer is not None else None
        if kf_nodes is not None and scores is not None and len(scores) == len(tokens):
            # PLI: KFlash's tree is already best-first (parents before children); chance = calibrated probability of the path
            slot.chances = [math.exp(x) for x in scores][:nodes]
            slot.kf_nodes = list(kf_nodes[:nodes])
            return as_drafts((tokens, parents), nodes)
        slot.kf_nodes = None
        if scores is None or len(scores) != len(tokens):
            slot.chances = None
            return as_drafts((tokens, parents), nodes)
        if self.log_path:
            record = {"stream": slot.number, "position": int(position), "greedy": sampling is None,
                      "kept": slot.kept, "tokens": tokens, "parents": parents, "scores": [round(x, 4) for x in scores]}
            with open(self.log_path, "a") as handle:
                handle.write(json.dumps(record) + "\n")
        table = self.calibration.get("greedy" if sampling is None else "sampled")
        chances = table.probabilities(parents, scores) if table is not None else [math.exp(x) for x in scores]
        tokens, parents, chances = by_chance(tokens, parents, chances)
        slot.chances = chances[:nodes]
        return as_drafts((tokens, parents), nodes)

    def draft_streams(self, caches: Sequence[list[Any]], follows: Sequence[Sequence[int]], rows: Sequence[Sequence[int]],
                      positions: Sequence[int], samplings: Sequence[Any], depths: Sequence[int]) -> list[Any]:
        """``draft`` for every stream of a shared round, their lattices in one drafter forward."""

        from tensorfold.drafters.dflash_batch import start_trees

        if self.v1:                                  # no lattice to share: each stream's chain on its own
            for cache, follow, kept, sampling in zip(caches, follows, rows, samplings):
                self.read(cache, kept, follow, sampling)
            return [self.tree(cache, position, sampling, depth)
                    for cache, position, sampling, depth in zip(caches, positions, samplings, depths)]
        items = []
        nodes = max(1, min(self.nodes, max(depths, default=self.nodes)))     # the widest budget sets every block
        for cache, follow, kept, position, sampling in zip(caches, follows, rows, positions, samplings):
            self.read(cache, kept, follow, sampling)
            items.append((cache[-1].proposer, _Context(int(position), int(follow[-1])), nodes))
        states = start_trees(self.drafter, items)     # one block length for all: one forward
        return [self._drafts(cache[-1], proposer._finish_tree(context, state), position, sampling, depth)
                for cache, (proposer, context, _), state, position, sampling, depth
                in zip(caches, items, states, positions, samplings, depths)]


def by_chance(tokens: Sequence[int], parents: Sequence[int], chances: Sequence[float]
              ) -> tuple[list[int], list[int], list[float]]:
    """A tree re-ordered most likely first, parents before children; a node's chance is at most its parent's."""

    capped: list[float] = []
    children: dict[int, list[int]] = {}
    for i, (q, p) in enumerate(zip(parents, chances)):
        capped.append(min(float(p), capped[q]) if q >= 0 else float(p))
        children.setdefault(q, []).append(i)
    heap = [(-capped[i], i) for i in children.get(-1, [])]
    heapq.heapify(heap)
    order: list[int] = []
    while heap:
        _, i = heapq.heappop(heap)
        order.append(i)
        for c in children.get(i, []):
            heapq.heappush(heap, (-capped[c], c))
    place = {old: new for new, old in enumerate(order)}
    return ([int(tokens[i]) for i in order], [-1 if parents[i] < 0 else place[parents[i]] for i in order],
            [capped[i] for i in order])


def as_drafts(tree: tuple[list[int], list[int]], nodes: int) -> Any:
    """Take the first nodes of a best-first tree, returning a token list for a chain or (tokens, parents) otherwise."""

    tokens, parents = [int(t) for t in tree[0][:nodes]], [int(q) for q in tree[1][:nodes]]
    return tokens if parents == list(range(-1, len(tokens) - 1)) else (tokens, parents)


__all__ = ["DFlashHead", "DraftSlot", "as_drafts", "by_chance"]
