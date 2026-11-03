"""PLI: trace under the chunk (evolution mode: PLI_TRACE_LAYERS=1 with the trace on; off by default).

For each prefill chunk of the 27B (Qwen3.8 dense): time of the embedding, of full attention and of DeltaNet
(projections apart), of the FFN (GPU alone, or shared with the Neural Engine: see ane/ffn.py, fields ffn_ane_*), of whole
layers (the rest: norms, residuals), of the final norm, and of the chunk's end (draft, final evaluation; in
engine/family_prefill.py). Each measured module is evaluated at its input then at its output (mx.eval): these are times
without overlap between modules, a little longer than in service; compare with a replay without this trace. Same
operations, same order: only the moment of evaluation changes (like the interleaved split).

A "sub_chunk" line in the trace (trace.py), queued by prompt_fill right after "chunk", same request:
sums in ms over the chunk's layers, n_<key> = number of calls. Keys: emb, layer, fa, fa_proj, dn, dn_proj, ffn,
norm, end, ffn_ane_* and ffn_rq_* (ane/ffn.py), rq_cpu (requantization thread), rows, pos (start of the chunk).
Derived: attention core = fa - fa_proj; DeltaNet core = dn - dn_proj; rest = layer - fa - dn - ffn.
Without PLI_TRACE_LAYERS=1, nothing is installed and every call returns at once.
"""

from __future__ import annotations

import os
import time
from typing import Any

from tensorfold.pli import trace as _tr

ACTIVE = os.environ.get("PLI_TRACE_LAYERS", "0") == "1"
_prefill = False
_installed = False
_acc: dict[str, Any] = {}


def measuring() -> bool:
    """True if the trace under the chunk must measure now (setting on and trace on)."""

    return ACTIVE and _installed and _tr.active()


def in_prefill() -> bool:
    return _prefill


class prefill:
    """Prompt prefill block: the installed modules measure during this block, and only there (not decode)."""

    def __enter__(self) -> "prefill":
        global _prefill
        self.before = _prefill
        _prefill = measuring()
        return self

    def __exit__(self, *a: Any) -> None:
        global _prefill
        _prefill = self.before


def add(key: str, ms: float) -> None:
    _acc[key] = _acc.get(key, 0.0) + ms
    _acc["n_" + key] = _acc.get("n_" + key, 0) + 1


def put(key: str, value: Any) -> None:
    _acc[key] = value


def begin(rows: int, pos: int) -> None:
    """Start of a chunk: the sums restart from zero (nothing from a warm-up prefill or an abandoned chunk)."""

    _acc.clear()
    _acc["rows"], _acc["pos"] = rows, pos


def take() -> dict[str, Any]:
    """Sums of the chunk, reset to zero."""

    d = {k: (round(v, 3) if isinstance(v, float) else v) for k, v in _acc.items()}
    _acc.clear()
    return d


def _call(self: Any, *a: Any, **kw: Any) -> Any:
    base = type(self)._pli_tc_base
    if not _prefill:
        return base.__call__(self, *a, **kw)
    import mlx.core as mx

    mx.eval([v for v in (*a, *kw.values()) if isinstance(v, mx.array)])
    t0 = time.perf_counter()
    out = base.__call__(self, *a, **kw)
    mx.eval(out)
    add(self._pli_tc_key, (time.perf_counter() - t0) * 1e3)
    return out


_classes: dict[type, type] = {}


def _wrap(m: Any, key: str) -> None:
    if m is None or getattr(m, "_pli_tc_key", None) is not None:
        return
    base = type(m)
    if base not in _classes:
        _classes[base] = type(base.__name__ + "TC", (base,), {"__call__": _call, "_pli_tc_base": base})
    object.__setattr__(m, "_pli_tc_key", key)
    m.__class__ = _classes[base]


def _linears(m: Any) -> list[Any]:
    import mlx.nn as nn

    return [v for _, v in m.children().items() if isinstance(v, nn.Module) and "Linear" in type(v).__name__]


def install(core: Any) -> int:
    """Put the measurements on the model's modules (after the Neural Engine split); return the number of layers taken."""

    global _installed
    if not ACTIVE or _installed or core is None:
        return 0
    layers = getattr(core, "pipeline_layers", None) or getattr(core, "layers", None) or []
    for c in layers:
        lin = getattr(c, "linear_attn", None) if getattr(c, "is_linear", False) else None
        att = lin if lin is not None else getattr(c, "self_attn", None)
        key = "dn" if lin is not None else "fa"
        if att is not None:
            for p in _linears(att):
                _wrap(p, key + "_proj")
            _wrap(att, key)
        _wrap(getattr(c, "mlp", None), "ffn")
        _wrap(c, "layer")
    _wrap(getattr(core, "embed_tokens", None), "emb")
    _wrap(getattr(core, "norm", None), "norm")
    _installed = True
    import sys

    print(f"PLI trace_layers: {len(layers)} layers measured in prefill (evolution mode)", file=sys.stderr, flush=True)
    return len(layers)
