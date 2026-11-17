"""PLI: round cost arbiter (PLI_ARBITER=1). Changes only the width of the windows, never the tokens.

TensorFold picks a window's width with costs measured at load time, with an empty context. Yet each row of a
window reads the context again in the full-attention layers: at long context, a row costs much more than planned,
and wide copies (16 to 128 rows) waste time. The arbiter learns online the real cost of a round:

    ms = base(rows) + A + K x rows x context / 100,000

base: load-time costs; A: fixed part of the round (draft, sampling); K: cost of a row per 100,000 context tokens.
A and K are fitted by least squares with forgetting over the observed rounds. It serves two choices:
- the width of a copy: the one that gives the most expected tokens per millisecond, with the success rate of the
  current request's copies; no copy if a one-row round does better;
- the number of drafts of a round (heads that give their probabilities, like DFlash2): same costs.

PLI_ARBITER_TABLE=1 (23/10/2026, arb_plafond measurement on the 21-22/10 trace): since attm, reduced prefill and tiles, one more row
costs almost nothing under 8 rows (past 70,000 tokens: 4 rows 98 ms, 8 rows 106 ms), while the formula counts each row
K x context and wrongly shrinks 57 to 69 % of the draft rounds past 40,000 tokens. The table of the number of drafts then takes
the cost measured in service (moving average per range of 8,192 context tokens and per number of rows, from 4 measured rounds), the
formula elsewhere, then makes the table increasing (a narrower round never costs more than a wider one). The width of copies keeps
the formula (its choices are within 1.6 % of the best on the trace).
Item D (24/10/2026): tree rounds have their own measured table (tree kernels of attention and DeltaNet, another cost per row);
without a tree measurement, the tree table starts from the chains' one.
"""

from __future__ import annotations

import os

ACTIVE = os.environ.get("PLI_ARBITER", "0") == "1"
LOG = os.environ.get("PLI_ARBITER_LOG", "")
TABLE = os.environ.get("PLI_ARBITER_TABLE", "0") == "1"
RANGE = 8192            # context tokens per range of the measured table
SEEN = 4                # rounds measured before a cell of the table is used
# PLI_ARBITER_MIN_ROWS=N (23/10/2026): a lone request keeps at least N - 1 drafts (N rows) when the head proposes that many:
# under the step of 8 rows, one more row costs almost nothing (arb_plafond measurement). 0: the arbiter's choice alone.
MIN_ROWS = max(0, int(os.environ.get("PLI_ARBITER_MIN_ROWS", "0") or 0))


class Arbiter:
    decay = 0.985            # weight of a past round at each new round (about 65 rounds of memory)
    copy_decay = 0.9         # same for the success rate of a request's copies
    copy_prior = (15.0, 1.0)     # successes and failures assumed at start: 0.94, TensorFold's rate
    margin = 1.05            # a copy or a draft must do at least 5 % better than a one-row round

    def __init__(self, base: dict[int, float]) -> None:
        self.base = {int(w): float(ms) for w, ms in base.items()}
        self.a, self.k = 8.0, 0.0
        self.n = self.sx = self.sy = self.sxx = self.sxy = 0.0
        self.copies: dict[str, list[float]] = {}
        self.rounds = 0
        self.seen: dict[tuple[int, int], list[float]] = {}     # (context range, rows) -> [moving mean ms, rounds]

    def observe(self, rows: int, context: int, ms: float, tree: bool = False) -> None:
        """A round of a lone request: ``rows`` rows at ``context`` tokens, ``ms`` in total."""

        b = self.base.get(int(rows))
        if b is None or ms <= 0:
            return
        if TABLE:
            c = self.seen.setdefault((int(context) // RANGE, int(rows), bool(tree)), [0.0, 0.0])
            c[0] = ms if c[1] == 0 else c[0] + 0.1 * (ms - c[0])
            c[1] += 1
        if tree:                                  # the formula learns on chains only
            return
        x, y, w = rows * context / 1e5, ms - b, self.decay
        self.n, self.sx, self.sy = self.n * w + 1.0, self.sx * w + x, self.sy * w + y
        self.sxx, self.sxy = self.sxx * w + x * x, self.sxy * w + x * y
        self.rounds += 1
        det = self.n * self.sxx - self.sx * self.sx
        if self.rounds >= 12 and det > 1e-3 * self.n * self.sxx:       # enough rounds, and widths varied enough
            self.k = max(0.0, (self.n * self.sxy - self.sx * self.sy) / det)
        self.a = max(0.0, (self.sy - self.k * self.sx) / self.n)
        if LOG and self.rounds % 50 == 0:
            with open(LOG, "a") as h:
                h.write(f"{self.rounds} A={self.a:.1f} K={self.k:.2f}\n")

    def cost(self, rows: int, context: int) -> float | None:
        b = self.base.get(int(rows))
        return None if b is None else b + self.a + self.k * rows * context / 1e5

    def table(self, context: float, most: int, tree: bool = False) -> dict[int, float]:
        """Costs per number of rows, without the fixed part (it is passed apart, like TensorFold's overhead)."""

        t = {w: b + self.k * w * context / 1e5 for w, b in self.base.items() if w <= most}
        if not TABLE or not t:
            return t
        rg = int(context) // RANGE
        for w in t:
            c = self.seen.get((rg, w, False))
            if c is not None and c[1] >= SEEN:
                t[w] = max(1.0, c[0] - self.a)
        if tree:                                  # trees: measured cells, elsewhere the chain at the measured tree / chain ratio
            seen = {w: max(1.0, c[0] - self.a) for w in t for c in [self.seen.get((rg, w, True))] if c is not None and c[1] >= SEEN}
            if seen:
                ratios = sorted(seen[w] / t[w] for w in seen)
                r = ratios[len(ratios) // 2]
                t = {w: seen.get(w, t[w] * r) for w in t}
        lowest = float("inf")
        for w in sorted(t, reverse=True):         # increasing: a narrower round never costs more than a wider one
            lowest = min(lowest, t[w])
            t[w] = lowest
        return t

    def copy_seen(self, stream_id: str, proposed: int, accepted: int) -> None:
        s = self.copies.setdefault(stream_id, list(self.copy_prior))
        s[0] = s[0] * self.copy_decay + accepted
        s[1] = s[1] * self.copy_decay + (1.0 if accepted < proposed else 0.0)

    def copy_width(self, stream_id: str, context: int, cap: int) -> int:
        """Tokens a copy may propose (0: no copy), at most ``cap``."""

        s = self.copies.get(stream_id, self.copy_prior)
        r = s[0] / (s[0] + s[1])
        alone = self.cost(1, context)
        if alone is None:
            return cap
        best, width = self.margin / alone, 0
        expected = left = 1.0
        for w in range(1, int(cap) + 1):
            left *= r
            expected += left
            c = self.cost(w + 1, context)
            if c is not None and expected / c > best:
                best, width = expected / c, w
        return width

    def forget(self, stream_id: str) -> None:
        self.copies.pop(stream_id, None)


__all__ = ["ACTIVE", "MIN_ROWS", "TABLE", "Arbiter"]
