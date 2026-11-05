"""PLI: what the server really serves, for /health and /v1/models (the "checkpoint" field).

The served name (``--name``, the "model" field) is the one clients ask for; it does not say which weights run.
This field does: folder or Hugging Face repository of the model (and snapshot revision), model_type and quantization read
from its config.json, draft (DFlash2 or other), MTP head (PLI_MTP_HEAD, hybrid or not) and PLI version.
Computed once at startup; a read that fails gives null for that detail, never an error.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ENGINE = "PLI"
BASE = "TensorFold 0.6.6"


def hub_snapshot(path: str | Path) -> tuple[str, str] | None:
    """("owner/name", revision) when ``path`` is (in) a snapshot of the Hugging Face cache, else None.

    Cache layout: ``…/models--owner--name/snapshots/<revision>/…``.
    """

    parts = Path(path).parts
    for i, part in enumerate(parts[:-2]):
        if part.startswith("models--") and parts[i + 1] == "snapshots":
            repo = part[len("models--"):].replace("--", "/", 1)
            if repo and "/" in repo:
                return repo, parts[i + 2]
    return None


def _config(directory: Path) -> dict[str, Any]:
    try:
        value = json.loads((directory / "config.json").read_text())
    except (OSError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _quantization(config: dict[str, Any]) -> dict[str, Any] | None:
    found = config.get("quantization") or config.get("quantization_config")
    if not isinstance(found, dict):
        found = (config.get("text_config") or {}).get("quantization") if isinstance(config.get("text_config"), dict) else None
    if not isinstance(found, dict):
        return None
    keep = {key: found[key] for key in ("bits", "group_size", "mode", "quant_method") if key in found}
    return keep or None


def describe_weights(path: str | Path) -> dict[str, Any]:
    """A weights folder: readable name, path, repository and revision (HF cache), model_type and quantization."""

    directory = Path(path).expanduser()
    snapshot = hub_snapshot(directory)
    config = _config(directory) if directory.is_dir() else {}
    text = config.get("text_config") if isinstance(config.get("text_config"), dict) else {}
    return {
        "name": snapshot[0] if snapshot else directory.name,
        "path": str(directory),
        "repo": snapshot[0] if snapshot else None,
        "revision": snapshot[1] if snapshot else None,
        "model_type": config.get("model_type"),
        "text_model_type": text.get("model_type"),
        "quantization": _quantization(config),
    }


def _mtp_head(model: Any) -> dict[str, Any] | None:
    """The MTP head in service: the PLI_MTP_HEAD one (recorded at load), or the checkpoint's own (35B)."""

    recorded = getattr(model, "pli_mtp_head", None)
    if isinstance(recorded, dict):
        head = Path(str(recorded.get("path", "")))
        return {"name": f"{head.parent.name}/{head.name}" if head.parent.name else head.name,
                "path": str(head), "hybrid": bool(recorded.get("hybrid")),
                "active": getattr(model, "mtp", None) is not None, "in_checkpoint": False}
    # without PLI_MTP_HEAD: the checkpoint's MTP layer (Qwen3.6 family: "mtp" without a DFlash draft)
    if hasattr(model, "head_drafts") and model.head_drafts is None and getattr(model, "mtp", None) is not None:
        return {"name": None, "path": None, "hybrid": False, "active": True, "in_checkpoint": True}
    return None


def describe(model_dir: str | Path | None, *, drafter: str = "", model: Any = None,
             version: str | None = None) -> dict[str, Any]:
    """The "checkpoint" field: model, draft, MTP head and PLI version."""

    if version is None:
        from tensorfold import __version__ as version
    info: dict[str, Any] = describe_weights(model_dir) if model_dir else {
        "name": None, "path": None, "repo": None, "revision": None, "model_type": None, "text_model_type": None,
        "quantization": None}
    head = _mtp_head(model) if model is not None else None
    # the MTP head alone (PLI_MTP_HEAD without hybrid) replaces the draft: it is not loaded
    replaced = head is not None and not head["in_checkpoint"] and not head["hybrid"]
    loaded = getattr(model, "head_drafts", True) is not None if model is not None else True
    info["draft"] = describe_weights(drafter) if drafter and loaded and not replaced else None
    info["mtp_head"] = head
    info["engine"] = {"name": ENGINE, "version": str(version), "base": BASE}
    return info
