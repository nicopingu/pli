"""PLI machine profiles: PLI_* settings per chip and per model family.

A profile is a folder of ``tensorfold/pli/profiles/``: ``common.env`` (all families) and ``<model_type>.env``
(one family, for example ``qwen3_5`` for the dense Qwen3.8). Lines ``KEY=value``; ``#`` comments.
A variable already in the environment always wins over the profile.
The profile is applied before any kernel import: several modules read their settings at import.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from tensorfold.pli import env_names

FOLDER = Path(__file__).resolve().parent / "profiles"


def available() -> list[str]:
    return sorted(p.name for p in FOLDER.iterdir() if p.is_dir()) if FOLDER.is_dir() else []


def _read(path: Path) -> dict[str, str]:
    values = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip('"')
    return values


def model_type_of(folder: Path | None) -> str | None:
    """The ``model_type`` of a model folder's config.json, without importing a family."""

    try:
        return json.loads((Path(folder) / "config.json").read_text()).get("model_type") if folder else None
    except (OSError, ValueError):
        return None


def apply(name: str, model_type: str | None = None, environ=os.environ) -> list[str]:
    """Set the settings of profile ``name`` (and of family ``model_type``); return the files read."""

    folder = FOLDER / name
    if not folder.is_dir():
        raise ValueError(f"unknown profile: {name} (available profiles: {', '.join(available()) or 'none'})")
    read = []
    common = folder / "common.env"
    if not common.is_file() and (folder / "commun.env").is_file():
        common = folder / "commun.env"                     # a profile folder written before 3.7.0
    for path in [common] + ([folder / f"{model_type}.env"] if model_type else []):
        if path.is_file():
            for key, value in _read(path).items():
                environ.setdefault(env_names.NEW_NAME.get(key, key), value)   # a profile may use the old names
            read.append(f"{name}/{path.name}")
    return read
