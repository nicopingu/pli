"""PLI 5.6.3 (11/17/2026): startup measurements kept on disk (``pli serve --startup-cache on``, off by default).

A 27B start took 68 s on the validated M1 Ultra, of which about 60 s are measurements redone at every start (startup
profile of 11/16/2026): the decoder's window check (16 s), the stream memory probes (20 s),
the prompt memory probe (19 s) and the image encoder workspace (4 s). Their results only depend on the weights, the code,
the settings and the machine: they are written once to a file named by a key over all of these, and read back at the next
start. Any change of weights (size or date of a file), of a PLI source file (``code_key.suffix``), of a ``PLI_*`` or
``MLX_*`` variable, of the serve arguments, of MLX or mlx-lm, of the PLI version or of the machine gives another key, so
the measurements are redone. ``--startup-cache remeasure`` measures again and rewrites the file. ``PLI_STARTUP_CACHE_DIR``
(default ``~/.cache/pli/startup``) holds one JSON file per key.

An option rather than a ``PLI_*`` variable, and a module outside ``pli/additions``: neither enters the checkpoint key
(``code_key``), so turning it on keeps the conversations already saved. ``PLI_STARTUP_CACHE=1`` (or ``remeasure``) still
works where no option is given, at the cost of a new settings fingerprint.

What a start without the measurements does not do: the probe prefills also warmed some kernels, so the first request after
a start from the cache may pay a little compilation (to be measured). The memory budget itself (what the rest of the machine
holds) is still read at every start: only what belongs to the model is kept.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

MODES = ("off", "on", "remeasure")
_ENV = os.environ.get("PLI_STARTUP_CACHE", "0")
ACTIVE = _ENV in ("1", "on", "remeasure")
REMEASURE = _ENV == "remeasure" or os.environ.get("PLI_STARTUP_REMEASURE", "0") == "1"
DIRECTORY = Path(os.environ.get("PLI_STARTUP_CACHE_DIR", "~/.cache/pli/startup")).expanduser()
_OWN = ("PLI_STARTUP_CACHE", "PLI_STARTUP_REMEASURE", "PLI_STARTUP_CACHE_DIR")
_SKIP_ARGS = {"--port", "--host", "--startup-cache"}

_key: str | None = None
_values: dict[str, Any] = {}
_counts = {"read": 0, "measured": 0}


def _file_mark(path: Path) -> list[Any]:
    try:
        st = path.stat()
        return [path.name, st.st_size, int(st.st_mtime)]
    except OSError:
        return [path.name, None, None]


def _dir_mark(path: str | os.PathLike[str] | None) -> list[Any]:
    """Sizes and dates of a checkpoint's top files (weights, configs, tokenizer): enough to see a changed checkpoint."""

    if not path:
        return []
    p = Path(path).expanduser()
    if p.is_file():
        return [_file_mark(p)]
    if not p.is_dir():
        return [str(p)]
    return [_file_mark(f) for f in sorted(p.iterdir()) if f.is_file() and f.suffix in (".safetensors", ".json", ".jinja", ".model")]


def _serve_args(argv: list[str]) -> list[str]:
    out, skip = [], False
    for a in argv:
        if skip:
            skip = False
            continue
        if a in _SKIP_ARGS:
            skip = True
            continue
        if any(a.startswith(s + "=") for s in _SKIP_ARGS):
            continue
        out.append(a)
    return out


def _settings() -> dict[str, str]:
    """The settings that enter the key: the PLI_* of the checkpoint fingerprint (``code_key``: journals, traces, FALCON
    captures and background tasks stay out, they never change a computation), less generation (``PLI_GEN*``, children
    only) and this cache's own switches, plus the MLX_* and TENSORFOLD_* variables. A journal named by a variable grows at
    every start: kept in the key, its size and date would give a new key each time (seen at the 5.6.3 pose, 11/17)."""

    from pli.additions import code_key, env_names

    pli = {k: v for k, v in env_names.pli_settings(exclude=code_key.SETTINGS_EXCLUDED)
           if not k.startswith(code_key.SETTINGS_EXCLUDED_PREFIXES + ("PLI_GEN",)) and k not in _OWN}
    other = {k: v for k, v in os.environ.items() if k.startswith(("MLX_", "TENSORFOLD_"))}
    return dict(sorted({**pli, **other}.items()))


def configure(model_dir: str | os.PathLike[str], drafter: str | None = None, argv: list[str] | None = None,
              mode: str | None = None) -> str | None:
    """Computes the key of this start (called by ``pli serve`` before the model loads). None when the cache is off.
    ``mode`` (the ``--startup-cache`` option: off, on, remeasure) wins over the environment when it is not off."""

    global _key, _values, ACTIVE, REMEASURE
    if mode in ("on", "remeasure"):
        ACTIVE, REMEASURE = True, mode == "remeasure"
    _key = None
    if not ACTIVE:
        return None
    from importlib.metadata import PackageNotFoundError, version

    import mlx.core as mx

    from pli import __version__
    from pli.additions import code_key

    def ver(name: str) -> str:
        try:
            return version(name)
        except PackageNotFoundError:
            return ""

    env = _settings()
    files = {k: _file_mark(Path(v)) for k, v in env.items() if v.startswith("/") and Path(v).is_file()}
    try:
        ram = int(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES"))
    except (ValueError, OSError):
        ram = 0
    parts = {
        "model": str(Path(model_dir).resolve()), "weights": _dir_mark(model_dir),
        "drafter": str(drafter or ""), "drafter_weights": _dir_mark(drafter) if drafter and "/" in str(drafter) else [],
        "args": _serve_args(list(sys.argv[2:] if argv is None else argv)),
        "env": env, "env_files": files, "code": code_key.suffix(), "pli": __version__,
        "mlx": mx.__version__, "mlx_lm": ver("mlx-lm"), "python": platform.python_version(),
        "machine": [platform.machine(), platform.mac_ver()[0], ram],
    }
    _key = hashlib.sha256(json.dumps(parts, sort_keys=True, default=str).encode()).hexdigest()[:20]
    _values = {}
    if not REMEASURE:
        try:
            _values = json.loads((DIRECTORY / f"{_key}.json").read_text()).get("values", {})
        except (OSError, ValueError):
            _values = {}
    return _key


def _name(what: str, params: dict[str, Any] | None) -> str:
    return what + ("|" + json.dumps(params, sort_keys=True, default=str) if params else "")


def remember(what: str, params: dict[str, Any] | None, measure: Callable[[], Any],
             dump: Callable[[Any], Any] = lambda v: v, load: Callable[[Any], Any] = lambda v: v) -> Any:
    """The value of ``measure()`` for this start's key and these parameters: read from the file, or measured and written."""

    if _key is None:
        return measure()
    name = _name(what, params)
    if name in _values:
        try:
            value = load(_values[name])
            _counts["read"] += 1
            return value
        except Exception:  # noqa: BLE001 - an unreadable entry is measured again
            pass
    value = measure()
    _values[name] = dump(value)
    _counts["measured"] += 1
    _write()
    return value


def _write() -> None:
    try:
        DIRECTORY.mkdir(parents=True, exist_ok=True)
        fd, tmp = tempfile.mkstemp(dir=DIRECTORY, prefix=".tmp-", suffix=".json")
        with os.fdopen(fd, "w") as f:
            json.dump({"key": _key, "values": _values}, f)
        os.replace(tmp, DIRECTORY / f"{_key}.json")
    except OSError as exc:
        print(f"[pli] startup cache: not written ({exc})", file=sys.stderr, flush=True)


def summary() -> str | None:
    if _key is None:
        return None
    return (f"[pli] startup cache {_key}: {_counts['read']} measurement(s) read, {_counts['measured']} measured"
            + (" (remeasure)" if REMEASURE else ""))


__all__ = ["ACTIVE", "MODES", "REMEASURE", "configure", "remember", "summary"]
