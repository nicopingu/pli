"""Former name of the PLI package, kept for compatibility (renamed to ``pli`` on 11/15/2026).

PLI 5 renames the Python package ``tensorfold`` to ``pli`` (and ``tensorfold.pli`` to ``pli.additions``). This module
keep at least one version, for launchers and scripts that still import the old name (for example the command
from an installation made before the renaming):

- ``import tensorfold.X`` gives the ``pli.X`` module itself (the same object, not a copy);
- ``import tensorfold.pli.X`` donne ``pli.additions.X`` ;
- ``python -m tensorfold.pli.rekey ...`` lance ``pli.additions.rekey`` ;
- a warning (FutureWarning) is issued once, at import time.
"""

from __future__ import annotations

import importlib
import importlib.abc
import importlib.machinery
import importlib.util
import sys
import warnings

import pli as _pli
from pli.safe_import import import_module as _pli_import

ANCIEN = "tensorfold"


def cible(nom: str) -> str:
    """The name of the PLI module that responds to the old name ``nom``."""

    if nom == "tensorfold.pli" or nom.startswith("tensorfold.pli."):
        return "pli.additions" + nom[len("tensorfold.pli"):]
    return "pli" + nom[len(ANCIEN):]


class Redirection(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Each imported ``tensorfold.X`` is ``pli.X``; ``get_code`` serves ``python -m tensorfold.X``."""

    def find_spec(self, fullname, path=None, target=None):
        if not fullname.startswith(ANCIEN + "."):
            return None
        vrai = importlib.util.find_spec(cible(fullname))
        if vrai is None:
            return None
        # origin : runpy en fait sys.argv[0] pour « python -m tensorfold.X » (argparse en tire son nom ; le fils vidéo de
        # PLI 5 died from this, deployment v55c on 11/15/2026 at 3:45 PM)
        return importlib.machinery.ModuleSpec(fullname, self, origin=vrai.origin,
                                              is_package=vrai.submodule_search_locations is not None)

    def create_module(self, spec):
        return _pli_import(cible(spec.name))        # PLI 5.6.3: pli.* only

    def exec_module(self, module):
        pass                                          # the PLI module is already loaded

    def get_code(self, fullname):
        nom = cible(fullname)
        spec = importlib.util.find_spec(nom)
        if spec.submodule_search_locations is not None:
            nom += ".__main__"
            spec = importlib.util.find_spec(nom)
        return spec.loader.get_code(nom)

    def is_package(self, fullname):
        return importlib.util.find_spec(cible(fullname)).submodule_search_locations is not None


if not any(isinstance(f, Redirection) for f in sys.meta_path):
    sys.meta_path.insert(0, Redirection())
    warnings.warn("the tensorfold package is the old name of pli (renamed in PLI 5): import pli; tensorfold "
                  "will be removed in a subsequent version", FutureWarning, stacklevel=2)


def __getattr__(nom: str):
    return getattr(_pli, nom)


def __dir__():
    return sorted(set(dir(_pli)))
