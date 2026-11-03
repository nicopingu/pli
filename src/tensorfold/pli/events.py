"""PLI: event log, one JSON line per action, with the conversation (PLI 3.6.0).

Switched on by ``PLI_EVENTS=<file>`` (pli.env); off, every call returns at once. Read by the
supervision dashboard (Inference view, PLI actions per session). Touches neither the computation nor the order
of requests: the text produced is the same. A write error is swallowed, never the request.

Nothing of the requests' content is written: a conversation is only a fingerprint (10 characters) of the
messages up to and including the first user message, which the following turns keep. Actions that
know only tokens (checkpoint, spill, read back from disk) are attached to the recent
conversation whose prompt starts with those tokens; several: "shared" (common system block).

Fields: t (Unix time), m (model), a (action), req (request), conv (conversation), then the action's own.
The file moves to ``<file>.1`` past PLI_EVENTS_MAX_MB (50 MB by default).
Action and field names became English in 3.7.0 (the table of old names is in TRANSLATION-REPORT.md).
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import os
import sys
import threading
import time
from typing import Any

from tensorfold.pli import trace as _trace

_PATH = os.environ.get("PLI_EVENTS", "").strip()
try:
    _MAX_BYTES = max(1, int(os.environ.get("PLI_EVENTS_MAX_MB", "50"))) * 1024 * 1024
except ValueError:
    _MAX_BYTES = 50 * 1024 * 1024
_LOCK = threading.Lock()
_RECENT: "OrderedDict[str, tuple[str, list[int]]]" = OrderedDict()   # request -> (conversation, prompt)
_MAX_RECENT = 64
_MODEL: str | None = None


def active() -> bool:
    return bool(_PATH)


def _model_name() -> str:
    """Served name (--name), else the model's folder, read once from the command line of "tensorfold serve"."""

    global _MODEL
    if _MODEL is None:
        args, name = sys.argv, ""
        try:
            if "--name" in args:
                name = args[args.index("--name") + 1]
            elif "serve" in args:
                name = os.path.basename(str(args[args.index("serve") + 1]).rstrip("/"))
        except (IndexError, ValueError):
            name = ""
        _MODEL = name
    return _MODEL


def _text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(str(p.get("text", "")) if isinstance(p, dict) else str(p) for p in content)
    return "" if content is None else str(content)


def conversation(messages: Any) -> str:
    """Fingerprint of the messages up to and including the first user message ("" if unreadable)."""

    try:
        h = hashlib.sha1()
        for m in messages or []:
            role = str(m.get("role", "")) if isinstance(m, dict) else ""
            h.update(role.encode("utf-8"))
            h.update(b"\0")
            h.update(_text(m.get("content") if isinstance(m, dict) else m).encode("utf-8", "replace"))
            h.update(b"\1")
            if role == "user":
                break
        return h.hexdigest()[:10]
    except Exception:  # noqa: BLE001 - a missed fingerprint only costs the grouping
        return ""


def bind(job: Any, conv: str) -> None:
    """Remember a request's conversation and prompt (the last 64), to attach the actions known by tokens."""

    if not _PATH:
        return
    try:
        job.pli_conv = conv
        with _LOCK:
            _RECENT[job.job_id] = (conv, list(job.prompt_ids))
            _RECENT.move_to_end(job.job_id)
            while len(_RECENT) > _MAX_RECENT:
                _RECENT.popitem(last=False)
    except Exception:  # noqa: BLE001
        pass


def conv_of_tokens(tokens: Any) -> str:
    """Conversation whose prompt starts with ``tokens`` or is their start; "shared" if several."""

    if not _PATH:
        return ""
    try:
        t = list(tokens)
        n = len(t)
        if not n:
            return ""
        with _LOCK:
            recent = list(_RECENT.values())
        found: list[str] = []
        for conv, ids in reversed(recent):
            m = min(n, len(ids))
            if conv and m and ids[:m] == t[:m] and m >= n // 2 and conv not in found:
                found.append(conv)
        if not found:
            return ""
        return found[0] if len(found) == 1 else "shared"
    except Exception:  # noqa: BLE001
        return ""


def ev(action: str, job: Any = None, conv: str | None = None, **fields: Any) -> None:
    """Write an action (one line). Fields at None are left out; no exception leaves this function."""

    if not _PATH and not _trace.active():
        return
    try:
        d: dict[str, Any] = {"t": round(time.time(), 3), "m": _model_name(), "a": action}
        if job is not None:
            d["req"] = str(getattr(job, "job_id", ""))
            if conv is None:
                conv = getattr(job, "pli_conv", "") or ""
        if conv:
            d["conv"] = conv
        for key, value in fields.items():
            if value is None:
                continue
            d[key] = round(value, 3) if isinstance(value, float) else value
        if _trace.active():
            _trace.line({**d, "e": "ev." + action})
        if not _PATH:
            return
        line = json.dumps(d, ensure_ascii=False, separators=(",", ":")) + "\n"
        with _LOCK:
            try:
                if os.path.getsize(_PATH) > _MAX_BYTES:
                    os.replace(_PATH, _PATH + ".1")
            except OSError:
                pass
            with open(_PATH, "a", encoding="utf-8") as f:
                f.write(line)
    except Exception:  # noqa: BLE001 - the log never makes a request fail
        pass
