"""PLI (10/11/2026): the code that computes a checkpoint enters its file key.

The key of a checkpoint (cli.py) names the model, MLX, the PLI version, the kernels of the family and the PLI_* settings.
It did not name the rest of PLI's own code: ``src/tensorfold/pli`` (A1, Neural Engine, attm, KVD5 reduction...) and the
prompt path of ``src/tensorfold/engine``. A pose that changed one of these files without a new version number left
checkpoints computed by the old code to be resumed by the new one (found on 04/11, checked again on 10/11 on 4.1.2).

``suffix()`` hashes those sources: any change to them gives a new key, so old checkpoints are no longer resumed (they
are pruned by size like any other). Files that cannot change a checkpoint's bits stay out: logs, traces, banner, the
FALCON capture, the setting names.

PLI_CODE_KEY=1 (default): on, "|code=<12 hex>" ends the key. 0: off, previous key bit for bit.
Text unchanged: only which checkpoints may be resumed changes. Cost: every pose that changes these files loses the
checkpoints kept until then, memory and disk.
"""

from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

ENABLED = os.environ.get("PLI_CODE_KEY", "1") == "1"

ROOT = Path(__file__).resolve().parent.parent           # src/tensorfold
SOURCE_SUFFIXES = (".py", ".mm", ".m", ".c", ".h", ".hpp", ".metal")

# src/tensorfold/pli: everything, except what cannot change a checkpoint's bits.
PLI_EXCLUDED_DIRS = ("falcon",)                        # capture of training targets for the MTP head (reads, never computes)
PLI_EXCLUDED_FILES = (
    "banner.py", "events.py", "env_names.py", "lattice_log.py", "round_profile.py",
    "trace.py", "trace_analysis.py", "trace_layers.py",
    "rekey.py",                                         # rewrites keys of checkpoints on disk, never computes one
)
# src/tensorfold/engine: the prompt path and the cache it builds; sampling, grammar and admission stay out.
ENGINE_FILES = (
    "alternating_kv.py", "family_common.py", "family_depth.py", "family_prefill.py", "family_shared.py",
    "lane_engine.py", "lane_family.py", "prefill_plan.py", "prefill_step.py", "prefix_snapshots.py",
)


def files(root: Path = ROOT) -> list[Path]:
    """The hashed source files, sorted."""

    pli = root / "pli"
    out = []
    for path in sorted(pli.rglob("*")):
        rel = path.relative_to(pli)
        if not path.is_file() or path.suffix not in SOURCE_SUFFIXES or "__pycache__" in rel.parts:
            continue
        if rel.parts[0] in PLI_EXCLUDED_DIRS or (len(rel.parts) == 1 and rel.name in PLI_EXCLUDED_FILES):
            continue
        out.append(path)
    out.extend(p for p in (root / "engine" / name for name in ENGINE_FILES) if p.is_file())
    return out


@lru_cache(maxsize=4)
def digest(root: Path = ROOT) -> str:
    h = hashlib.sha256()
    for path in files(root):
        h.update(path.relative_to(root).as_posix().encode())
        h.update(b"\0")
        h.update(path.read_bytes())
        h.update(b"\0")
    return h.hexdigest()[:12]


# The PLI_* settings left out of the key's settings fingerprint: they never change a checkpoint's bits. KVD5's bits and
# group are carried by "|points=...", the code by "|code=..."; the others only choose where a checkpoint is written,
# how a call is checked, or what is logged. PLI 4.2.0: FALCON (captures, request tokens), traces and the event log join
# them (lot-1111b had changed the fingerprint by adding PLI_FALCON_TOKENS: 398 checkpoints were no longer resumed).
SETTINGS_EXCLUDED = ("PLI_SNAPSHOT_SETTINGS", "PLI_ATTENTION_TUNING_FILE", "PLI_CHECKPOINT_BITS", "PLI_CHECKPOINT_GROUP",
                     "PLI_CODE_KEY", "PLI_SPILL_BACKGROUND", "PLI_SPILL_BACKGROUND_MAX_GIB", "PLI_TOOL_GRAMMAR",
                     "PLI_DISK_REPLY_END")
SETTINGS_EXCLUDED_PREFIXES = ("PLI_FALCON", "PLI_TRACE", "PLI_EVENTS", "PLI_COPY_LOG")   # 12/11: the copy log


def settings_fingerprint(environ=None) -> str:
    """12 hex of the PLI_* settings that can change a checkpoint's bits (new names, sorted)."""

    from tensorfold.pli import env_names

    pairs = [(k, v) for k, v in env_names.pli_settings(environ, exclude=SETTINGS_EXCLUDED)
             if not k.startswith(SETTINGS_EXCLUDED_PREFIXES)]
    return hashlib.sha256(repr(pairs).encode()).hexdigest()[:12]


def release_part(version: str, enabled: bool | None = None) -> str:
    """PLI 4.2.0: the release number's part of the key. With the code in the key it tells nothing more (what changes a
    checkpoint's bits is the code, hashed by suffix(), with the libraries, kernels and settings already in the key) and
    is left out, so a new release number alone keeps the checkpoints. Off: ``|tensorfold=X.Y.Z`` as before."""

    on = ENABLED if enabled is None else enabled
    return "" if on else f"|tensorfold={version}"


def suffix(enabled: bool | None = None, root: Path = ROOT) -> str:
    """The file-key part; empty when off (previous key, bit for bit)."""

    on = ENABLED if enabled is None else enabled
    return f"|code={digest(root)}" if on else ""
