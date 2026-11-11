"""PLI (11/11/2026): give the checkpoints on disk a new key without recomputing them.

A checkpoint file is named after sha256(model key, tokens) and carries the key in its metadata ("model"). When the key
changes but the bits of the checkpoints do not (a release number, a new key layout), this rewrites each file's header
and name; the tensors are copied as they are, the file keeps its modification time (eviction goes by age). It does not
check the computation: the caller knows the bits are the same (same code, same settings).

  python -m tensorfold.pli.rekey plan DIR... --version X.Y.Z --pli FP[,FP] --env FILE
      the keys of release X.Y.Z with one of the settings fingerprints FP, and the key 4.2.0 gives them: release number
      left out, settings fingerprint of FILE (the engine's environment), code digest added
  python -m tensorfold.pli.rekey move --old KEY [--old KEY] --new KEY DIR [DIR...]
      every checkpoint of DIR keyed KEY gets the new key; prints how many files, bytes and seconds
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import struct
import sys
import time
from pathlib import Path

from tensorfold.engine.prefix_snapshots import partial_path, read_metadata, snapshot_key

COPY_BYTES = 64 << 20


def checkpoints(directory: Path) -> list[Path]:
    return [p for p in directory.glob("*.safetensors") if not p.name.endswith(".partial.safetensors")
            and "." not in p.name[:-len(".safetensors")]]


def new_key(old: str, version: str, code: str) -> str:
    """``old`` without its release number (``|tensorfold=X.Y.Z``) and with ``|code=<digest>`` instead of any older one."""

    key = old.replace(f"|tensorfold={version}", "")
    key = re.sub(r"\|code=[0-9a-f]+", "", key)
    return key + f"|code={code}"


def rewrite(path: Path, old: str, new: str) -> Path | None:
    """Move one checkpoint from key ``old`` to ``new``; None if it is not keyed ``old``.

    When the new header fits in the old one's room (keys of about the same length), only the header is written, in
    place, padded with spaces, and the file renamed: no tensor is copied. Otherwise the file is copied with its new
    header to a partial file, then renamed."""

    with open(path, "rb") as src:
        size = struct.unpack("<Q", src.read(8))[0]
        header = json.loads(src.read(size))
    meta = header.get("__metadata__") or {}
    if meta.get("model") != old:
        return None
    meta["model"] = new
    header["__metadata__"] = meta
    raw = json.dumps(header, separators=(",", ":")).encode()
    saved = str(meta.get("saved", ""))
    while len(raw) > size and "." in saved:           # "saved" is never read back: its decimals make room in place
        saved = saved[:-1].rstrip(".") if not saved.endswith(".") else saved[:-1]
        meta["saved"] = saved
        raw = json.dumps(header, separators=(",", ":")).encode()
    tokens = json.loads(meta["tokens"])
    target = path.with_name(snapshot_key(new, tokens) + ".safetensors")
    stat = path.stat()
    if target.exists():
        path.unlink()
        return target
    if len(raw) <= size:                               # safetensors: the header may end with spaces
        with open(path, "r+b") as handle:
            handle.seek(8)
            handle.write(raw + b" " * (size - len(raw)))
        os.utime(path, (stat.st_atime, stat.st_mtime))
        path.rename(target)
        return target
    raw += b" " * (-len(raw) % 8)
    partial = partial_path(target)
    try:
        with open(path, "rb") as src, open(partial, "wb") as dst:
            src.seek(8 + size)
            dst.write(struct.pack("<Q", len(raw)))
            dst.write(raw)
            shutil.copyfileobj(src, dst, COPY_BYTES)
        os.utime(partial, (stat.st_atime, stat.st_mtime))
        partial.rename(target)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    path.unlink()
    return target


def move(directories: list[Path], old: str, new: str) -> dict:
    t0 = time.time()
    out = {"files": 0, "bytes": 0, "others": 0}
    for directory in directories:
        for path in checkpoints(directory):
            try:
                if read_metadata(path).get("model") != old:
                    out["others"] += 1
                    continue
            except Exception:  # noqa: BLE001 - unreadable file: left alone
                out["others"] += 1
                continue
            n = path.stat().st_size
            if rewrite(path, old, new) is not None:
                out["files"] += 1
                out["bytes"] += n
    out["seconds"] = round(time.time() - t0, 1)
    return out


def keys(directories: list[Path]) -> dict[str, int]:
    """How many checkpoints each key has, newest key first."""

    found: dict[str, list] = {}
    for directory in directories:
        for path in checkpoints(directory):
            try:
                key = read_metadata(path)["model"]
            except Exception:  # noqa: BLE001
                continue
            n, t = found.get(key, (0, 0.0))
            found[key] = (n + 1, max(t, path.stat().st_mtime))
    return {k: v[0] for k, v in sorted(found.items(), key=lambda kv: -kv[1][1])}


def read_env(path: Path) -> dict[str, str]:
    """KEY=VALUE lines (the engine's environment as ``ps -E`` prints it, one per line)."""

    out = {}
    for line in path.read_text().splitlines():
        if "=" in line and not line.startswith("#"):
            k, v = line.split("=", 1)
            out[k.strip()] = v
    return out


def plan(directories: list[Path], version: str, fingerprints: list[str], environ: dict[str, str]) -> dict:
    """The key 4.2.0 gives the checkpoints of release ``version`` whose settings fingerprint is one of ``fingerprints``
    (fingerprints of settings known to give the same bits), from the base and KVD5 part of the newest such key."""

    from tensorfold.pli import code_key

    found = keys(directories)
    marker = f"|tensorfold={version}|pli="
    olds = [k for k in found if marker in k and "|code=" not in k
            and k.split(marker, 1)[1].split("|", 1)[0] in fingerprints]
    if not olds:
        raise SystemExit(f"no checkpoint of {version} with fingerprints {fingerprints}")
    base, rest = olds[0].split(marker, 1)
    points = rest[rest.index("|"):] if "|" in rest else ""
    olds = [k for k in olds if k.split(marker, 1)[0] == base and k.endswith(points)]
    new = base + "|pli=" + code_key.settings_fingerprint(environ) + points + "|code=" + code_key.digest()
    return {"new": new, "old": olds, "counts": {k: found[k] for k in olds}, "others": sum(v for k, v in found.items() if k not in olds)}


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="python -m tensorfold.pli.rekey")
    sub = p.add_subparsers(dest="action", required=True)
    a = sub.add_parser("plan"); a.add_argument("directories", type=Path, nargs="+")
    a.add_argument("--version", default="4.1.2"); a.add_argument("--pli", required=True, help="fingerprints, comma-separated")
    a.add_argument("--env", type=Path, required=True, help="the engine's settings, KEY=VALUE lines")
    b = sub.add_parser("move"); b.add_argument("--old", required=True, action="append"); b.add_argument("--new", required=True)
    b.add_argument("directories", type=Path, nargs="+")
    args = p.parse_args(argv)
    if args.action == "plan":
        print(json.dumps(plan(args.directories, args.version, args.pli.split(","), read_env(args.env))))
    else:
        out = {"files": 0, "bytes": 0, "seconds": 0.0}
        for old in args.old:
            r = move(args.directories, old, args.new)
            for k in out:
                out[k] += r[k]
        print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())
