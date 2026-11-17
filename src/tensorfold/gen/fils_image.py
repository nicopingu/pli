"""Old path for the PLI 5 image child process (package renamed on 11/15/2026).

Generation configurations written before the rename launch this file by its path, with a different Python
(``python -I .../src/tensorfold/gen/fils_image.py``). It launches the real file, ``src/pli/gen/fils_image.py``, as if it
had been launched itself. To be removed when no configuration names this path anymore.
"""

import runpy
import sys
from pathlib import Path

CIBLE = Path(__file__).resolve().parents[2] / "pli" / "gen" / "fils_image.py"
sys.argv[0] = str(CIBLE)
sys.path[0] = str(CIBLE.parent)
runpy.run_path(str(CIBLE), run_name="__main__")
