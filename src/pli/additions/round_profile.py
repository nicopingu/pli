"""PLI (29/10/2026, PLI_ROUND_PROFILE=1, off by default): where the time of a decode round goes, step by step.

Each step of the round (verification, head, draw, the next round's draft, the rest) is evaluated at its end (mx.eval): times without
overlap between steps, a little longer than in service; their sum is to be compared with the round without this profile. Same
operations, same order: the text does not change. One "[PLI] round profile" line every 200 rounds, means in ms per kind of round.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Any

ACTIVE = os.environ.get("PLI_ROUND_PROFILE", "0") == "1"
N = max(1, int(os.environ.get("PLI_ROUND_PROFILE_N", "200") or 200))   # rounds between two lines, per kind
BUCKET = max(0, int(os.environ.get("PLI_ROUND_PROFILE_BUCKET", "0") or 0))   # 30/10: kinds split by context bucket


def kind_of(kind: str, parents: Any, position: int) -> str:
    """The kind of round: head or copy, tree or chain, and the context bucket if PLI_ROUND_PROFILE_BUCKET is set."""

    s = kind + ("-tree" if parents is not None else "")
    if BUCKET:
        s += "-%dk" % (int(position) // BUCKET * BUCKET // 1000)
    return s
_acc: dict[str, dict[str, float]] = {}
_t = [0.0]


def top() -> float:
    return time.perf_counter()


def mark(kind: str, step: str, since: float, *arrays: Any) -> float:
    """Evaluate ``arrays`` and add the time elapsed since ``since`` to the step; return the present instant."""

    import mlx.core as mx

    if arrays:
        mx.eval(*[a for a in arrays if a is not None])
    now = time.perf_counter()
    d = _acc.setdefault(kind, {})
    d[step] = d.get(step, 0.0) + (now - since) * 1e3
    return now


def end_of_round(kind: str, start: float, rows: int, kept: int) -> None:
    d = _acc.setdefault(kind, {})
    d["round"] = d.get("round", 0.0) + (time.perf_counter() - start) * 1e3
    d["n"] = d.get("n", 0) + 1
    d["rows"] = d.get("rows", 0) + rows
    d["kept"] = d.get("kept", 0) + kept
    if int(d["n"]) % N == 0:
        n = d["n"]
        txt = ", ".join(f"{k} {v / n:.2f}" for k, v in d.items() if k != "n")
        print(f"[PLI] round profile ({kind}, {int(n)} rounds, ms or rows per round): {txt}", file=sys.stderr, flush=True)
        _acc[kind] = {}
