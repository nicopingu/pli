"""PLI (28/10/2026): draft tree in KFlash's manner (Kapitsa, docs/kflash.md), for greedy DFlash2.

The method comes from nace-ai/kapitsa (Apache-2.0), docs/kflash.md; this Python code for DFlash2's lattice is PLI's own.

- Probability of a node: q = the draft's conditional probability (softmax over the 16 candidates of the position of DFlash2's selector
  score, unary + pairwise, as its own chain), passed through a calibration per depth p = sigmoid(a_d + b_d * logit q).
- Best-first tree: always take the frontier node whose path has the highest probability (product of the p), add its children to
  the frontier, up to the wanted node count; a child never goes before its parent, so the first n nodes are the best tree of n nodes.
- Calibration learned online: each tree round adds, for the tried nodes (whose parent was accepted), the pair (logit q, accepted);
  every 16 rounds, a logistic regression per depth over the last 2,048 pairs (starting from p = q).
The text does not change: the verify stays exact, the tree only decides how many tokens a round keeps.
PLI_KFLASH=1 turns it on (with DFlash2's trees: PLI_TREES_ATTM=1, and PLI_HYBRID_TREES=1 in the hybrid).
"""

from __future__ import annotations

import heapq
import math
import os
from collections import deque
from typing import Any, Sequence

import numpy as np

ACTIVE = os.environ.get("PLI_KFLASH", "0") == "1"
REFIT = 16             # rounds between two fits
KEEP = 2048            # pairs kept per depth
MIN_FIT = 64           # pairs needed at least to fit a depth
DEPTHS = 16            # depths followed


class Calibration:
    def __init__(self) -> None:
        self.ab = [(0.0, 1.0)] * DEPTHS
        self.obs = [deque(maxlen=KEEP) for _ in range(DEPTHS)]
        self.rounds = 0
        self.seen = 0

    def p(self, d: int, q: float) -> float:
        q = min(max(q, 1e-6), 1.0 - 1e-6)
        a, b = self.ab[min(d, DEPTHS - 1)]
        z = a + b * math.log(q / (1.0 - q))
        return 1.0 / (1.0 + math.exp(-z)) if z > -60 else 1e-26

    def observe(self, records: Sequence[tuple[int, float, bool]]) -> None:
        for d, q, ok in records:
            q = min(max(q, 1e-6), 1.0 - 1e-6)
            self.obs[min(d, DEPTHS - 1)].append((math.log(q / (1.0 - q)), 1.0 if ok else 0.0))
        self.seen += len(records)
        self.rounds += 1
        if self.rounds % REFIT == 0:
            self.refit()

    def refit(self) -> None:
        for d in range(DEPTHS):
            o = self.obs[d]
            if len(o) < MIN_FIT:
                continue
            x = np.fromiter((u for u, _ in o), dtype=np.float64, count=len(o))
            y = np.fromiter((v for _, v in o), dtype=np.float64, count=len(o))
            a, b = self.ab[d]
            lam = 1.0                                  # weak pull towards p = q (a 0, b 1)
            for _ in range(12):                        # Newton
                z = np.clip(a + b * x, -40, 40)
                s = 1.0 / (1.0 + np.exp(-z))
                w = s * (1.0 - s)
                ga = float(np.sum(s - y)) + lam * a
                gb = float(np.sum((s - y) * x)) + lam * (b - 1.0)
                haa = float(np.sum(w)) + lam
                hab = float(np.sum(w * x))
                hbb = float(np.sum(w * x * x)) + lam
                det = haa * hbb - hab * hab
                if det <= 1e-12:
                    break
                da = (hbb * ga - hab * gb) / det
                db = (haa * gb - hab * ga) / det
                a, b = a - da, b - db
                if abs(da) + abs(db) < 1e-6:
                    break
            if math.isfinite(a) and math.isfinite(b):
                self.ab[d] = (a, b)


CALIB = Calibration()


def build(cands: Any, unary: Any, hproj: Any, anchor: int, pred: Any, succ: Any, max_nodes: int,
          calib: Calibration | None = None) -> tuple[list[int], list[int], list[float], list[tuple[int, float]]]:
    """Best-first tree of ``max_nodes`` nodes: tokens, parents (-1: the pending row), log-probability of each path, and
    (depth, q) of each node."""

    calib = calib or CALIB
    D, K = int(cands.shape[0]), int(cands.shape[1])
    tables: list[Any] = [None] * D

    def cond(token: int, d: int) -> Any:
        t = tables[d]
        if t is None:
            t = tables[d] = succ[cands[d]].astype(np.float64)
        s = unary[d] + t @ (pred[token].astype(np.float64) * hproj[d])
        s = s - s.max()
        e = np.exp(s)
        return e / e.sum()

    heap: list[tuple[float, int, int, int, float]] = []
    q0 = cond(int(anchor), 0)
    for k in range(K):
        heapq.heappush(heap, (-math.log(max(calib.p(0, float(q0[k])), 1e-30)), -1, int(cands[0][k]), 0, float(q0[k])))
    tokens: list[int] = []
    parents: list[int] = []
    scores: list[float] = []
    nodes: list[tuple[int, float]] = []
    while heap and len(tokens) < max_nodes:
        neg, parent, token, depth, q = heapq.heappop(heap)
        tokens.append(token)
        parents.append(parent)
        scores.append(-neg)
        nodes.append((depth, q))
        me = len(tokens) - 1
        if depth + 1 < D:
            qd = cond(token, depth + 1)
            for k in range(K):
                heapq.heappush(heap, (neg - math.log(max(calib.p(depth + 1, float(qd[k])), 1e-30)), me,
                                      int(cands[depth + 1][k]), depth + 1, float(qd[k])))
    return tokens, parents, scores, nodes


# PLI (30/10/2026): PLI_TREE_GAIN_MIN=x: the tree is kept only if its expected tokens (1 + sum of the calibrated probabilities of the
# paths) exceed those of DFlash2's greedy chain by at least x tokens (the cost of a tree round, about 1.3 ms, is worth 0.1 to 0.15 token);
# otherwise the chain is verified alone (no tree cost). Measured on 30/10: the tree gains at 1k (+6.6 %) and 8k (+18 %), loses on
# long code (-2.2 %) where the chain already keeps 6 tokens per round. Empty: always the tree (behaviour before).
GAIN_MIN = os.environ.get("PLI_TREE_GAIN_MIN", "")
CHOICES = {"tree": 0, "chain": 0}


def chain(cands: Any, unary: Any, hproj: Any, anchor: int, pred: Any, succ: Any, max_nodes: int,
           calib: Calibration | None = None) -> tuple[list[int], list[int], list[float], list[tuple[int, float]]]:
    """The greedy chain (argmax of the conditional probability at each depth), in ``build``'s format."""

    calib = calib or CALIB
    D = int(cands.shape[0])
    tokens: list[int] = []
    scores: list[float] = []
    nodes: list[tuple[int, float]] = []
    tok, logp = int(anchor), 0.0
    for d in range(min(D, int(max_nodes))):
        t = succ[cands[d]].astype(np.float64)
        sc = unary[d] + t @ (pred[tok].astype(np.float64) * hproj[d])
        sc = sc - sc.max()
        e = np.exp(sc)
        q = e / e.sum()
        j = int(np.argmax(q))
        tok = int(cands[d][j])
        logp += math.log(max(calib.p(d, float(q[j])), 1e-30))
        tokens.append(tok)
        scores.append(logp)
        nodes.append((d, float(q[j])))
    return tokens, list(range(-1, len(tokens) - 1)), scores, nodes


def choose(cands: Any, unary: Any, hproj: Any, anchor: int, pred: Any, succ: Any, max_nodes: int,
           calib: Calibration | None = None) -> tuple[list[int], list[int], list[float], list[tuple[int, float]]]:
    """``build``, or the chain if the tree does not bring PLI_TREE_GAIN_MIN more expected tokens."""

    tree = build(cands, unary, hproj, anchor, pred, succ, max_nodes, calib)
    if not GAIN_MIN:
        return tree
    ch = chain(cands, unary, hproj, anchor, pred, succ, max_nodes, calib)
    e_tree = sum(math.exp(x) for x in tree[2])
    e_chain = sum(math.exp(x) for x in ch[2])
    if e_tree - e_chain >= float(GAIN_MIN):
        CHOICES["tree"] += 1
        return tree
    CHOICES["chain"] += 1
    if (CHOICES["tree"] + CHOICES["chain"]) % 500 == 0:
        import sys
        print("[PLI] tree or chain: %s" % CHOICES, file=sys.stderr, flush=True)
    return ch


def observe_round(slot: Any, rows_parents: Sequence[int], path: Sequence[int], calib: Calibration | None = None) -> None:
    """After a tree round: each verified node whose parent is on the kept path was tried; accepted if it is on the path."""

    nodes = getattr(slot, "kf_nodes", None)
    if not nodes:
        return
    slot.kf_nodes = None
    kept = set(int(r) for r in path)
    records = []
    for r in range(1, len(rows_parents)):
        i = r - 1
        if i >= len(nodes):
            break
        if int(rows_parents[r]) in kept:
            d, q = nodes[i]
            if q is not None:
                records.append((d, q, r in kept))
    if records:
        (calib or CALIB).observe(records)


__all__ = ["ACTIVE", "CALIB", "CHOICES", "Calibration", "build", "chain", "choose", "observe_round"]
