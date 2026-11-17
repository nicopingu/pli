"""The tensorfold compatibility package (PLI 5: the package is named pli): same modules, same commands."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

RACINE = Path(__file__).resolve().parent.parent
SRC = RACINE / "src"


def _py(*args: str, code: str | None = None) -> subprocess.CompletedProcess:
    env = {**os.environ, "PYTHONPATH": str(SRC), "PLI_NO_UPDATE_CHECK": "1"}
    argv = [sys.executable, *(["-c", code] if code else []), *args]
    return subprocess.run(argv, capture_output=True, text=True, env=env, cwd=str(RACINE))


def test_memes_modules_et_un_avertissement():
    r = _py(code=(
        "import pli, tensorfold\n"
        "import tensorfold.cli, pli.cli\n"
        "import tensorfold.pli.env_names as a, pli.additions.env_names as b\n"
        "from tensorfold.pli import code_key\n"
        "from tensorfold import cli\n"
        "import pli.additions.code_key as ck\n"
        "assert tensorfold.cli is pli.cli and a is b and code_key is ck and cli is pli.cli\n"
        "assert tensorfold.__version__ == pli.__version__\n"
        "import sys; assert 'mlx' not in sys.modules\n"
        "print('ok')"))
    assert r.returncode == 0, r.stderr
    assert r.stdout.strip() == "ok"
    assert r.stderr.count("FutureWarning") == 1 and "import pli" in r.stderr


def test_pli_ne_charge_pas_tensorfold():
    r = _py(code="import sys, pli, pli.cli; print('tensorfold' in sys.modules)")
    assert r.returncode == 0 and r.stdout.strip() == "False" and "FutureWarning" not in r.stderr


@pytest.mark.parametrize("argv", [["serve", "--help"], ["--version"], ["config:list", "cache"], []])
def test_l_ancienne_commande_affiche_la_meme_chose(argv):
    """bin/tensorfold from a previous installation (``from tensorfold.cli import main``) versus bin/pli."""

    ancien = _py(*argv, code="import sys; from tensorfold.cli import main; sys.argv[0] = 'tensorfold'; "
                              "sys.exit(main())")
    nouveau = _py(*argv, code="import sys; from pli.cli import main; sys.argv[0] = 'pli'; sys.exit(main())")
    assert ancien.returncode == nouveau.returncode == 0
    assert ancien.stdout == nouveau.stdout and nouveau.stdout


def test_python_m_avec_l_ancien_nom():
    for ancien, nouveau in ((["-m", "tensorfold", "serve", "--help"], ["-m", "pli", "serve", "--help"]),
                            (["-m", "tensorfold.pli.rekey", "--help"], ["-m", "pli.additions.rekey", "--help"]),
                            # le fils vidéo de PLI 5 est lancé par « -m tensorfold.gen.fils_video » : sys.argv[0] doit être posé
                            (["-m", "tensorfold.gen.fils_video", "--help"], ["-m", "pli.gen.fils_video", "--help"])):
        a, n = _py(*ancien), _py(*nouveau)
        assert a.returncode == n.returncode == 0, a.stderr
        assert a.stdout == n.stdout


def test_anciens_chemins_de_la_production():
    """Paths named by the production files from before the renaming (gen, expanded list)."""

    for nom in ("fils_audio", "fils_image"):
        texte = (SRC / "tensorfold" / "gen" / f"{nom}.py").read_text()
        assert f'"pli" / "gen" / "{nom}.py"' in texte and (SRC / "pli" / "gen" / f"{nom}.py").is_file()
    liste = SRC / "tensorfold" / "pli" / "extended-list-2048.txt"
    assert liste.resolve() == (SRC / "pli" / "additions" / "extended-list-2048.txt").resolve()
    assert liste.read_bytes() == (SRC / "pli" / "additions" / "extended-list-2048.txt").read_bytes()


def test_fils_par_l_ancien_chemin(tmp_path):
    """The old path file correctly launches the real one: --help of the audio child process via both paths."""

    vieux = subprocess.run([sys.executable, "-I", str(SRC / "tensorfold" / "gen" / "fils_audio.py"), "--help"],
                           capture_output=True, text=True)
    neuf = subprocess.run([sys.executable, "-I", str(SRC / "pli" / "gen" / "fils_audio.py"), "--help"],
                          capture_output=True, text=True)
    assert vieux.returncode == neuf.returncode
    assert vieux.stdout.replace("tensorfold/gen", "pli/gen") == neuf.stdout
