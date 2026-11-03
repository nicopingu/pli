#!/usr/bin/env python3
"""Run the public bench suite pli-bench-v1 from a clone of the repository, with standard Python only.

The code and the fixed texts are part of the PLI package (src/tensorfold/bench/); an installed PLI runs the same
bench with `pli bench`. This file loads that code by its path, without importing PLI or MLX:

    python3 tools/pli_bench.py --url http://127.0.0.1:8080/v1 --salt pub --only single --sizes code,1k,8k
    python3 tools/pli_bench.py --help
"""
from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

_CODE = Path(__file__).resolve().parents[1] / "src" / "tensorfold" / "bench" / "pli_bench.py"
_spec = importlib.util.spec_from_file_location("pli_bench", _CODE)
_bench = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_bench)
# `import pli_bench` from this folder keeps giving the bench's functions (load_suite, piece, send, ...)
globals().update({k: v for k, v in vars(_bench).items() if not k.startswith("__")})

if __name__ == "__main__":
    sys.exit(_bench.main())
