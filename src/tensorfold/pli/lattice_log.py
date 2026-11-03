"""PLI (29/10/2026): log of the DFlash2 lattices served by the head (PLI_LATTICE_LOG=folder; off by default).

Each round writes, through the proposer's existing tracer (PLI_TREE_TRACE), the new context then the lattice (candidates, unary
scores, selector projection, anchor token); the selector's two tables are kept once (codes.npz, exact bf16 bits).
Used to replay offline, on the same text, the chain, PLI's tree, the KFlash tree and the candidates' ceiling. Nothing changes
in the computation or in the text.
"""

from __future__ import annotations

import os
from typing import Any

FOLDER = os.environ.get("PLI_LATTICE_LOG", "")


def path(proposer: Any) -> str:
    if not FOLDER:
        return ""
    os.makedirs(FOLDER, exist_ok=True)
    return os.path.join(FOLDER, f"lattice-{os.getpid()}-{id(proposer):x}.pkl")


def codes(proposer: Any) -> None:
    if not FOLDER or not hasattr(proposer, "_codebooks"):
        return
    f = os.path.join(FOLDER, "codes.npz")
    if os.path.exists(f):
        return
    import numpy as np

    pred, succ = proposer._codebooks()
    bits = lambda a: (np.ascontiguousarray(a, dtype=np.float32).view(np.uint32) >> 16).astype(np.uint16)
    np.savez(f, pred=bits(pred), succ=bits(succ))


__all__ = ["FOLDER", "codes", "path"]
