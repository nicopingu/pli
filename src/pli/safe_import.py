"""Imports by name, limited to known packages (PLI 5.6.3).

Some names come from data: a family's module from the families table, a cache class from the metadata of a
checkpoint written to disk. ``import_module`` imports such a name only when it starts with one of the given package
prefixes, so a checkpoint file cannot make PLI import an arbitrary module.
"""

from __future__ import annotations

import importlib
from types import ModuleType

PLI = ("pli.",)


def import_module(name: str, prefixes: tuple[str, ...] = PLI) -> ModuleType:
    """``importlib.import_module(name)`` when ``name`` is in one of ``prefixes``; ``ImportError`` otherwise."""

    if not isinstance(name, str) or not name.startswith(tuple(prefixes)):
        raise ImportError(f"module {name!r} is outside {', '.join(prefixes)}")
    # the name is checked against fixed package prefixes just above
    return importlib.import_module(name)  # nosemgrep: python.lang.security.audit.non-literal-import.non-literal-import


__all__ = ["PLI", "import_module"]
