#!/usr/bin/env python3
"""Write the MTP head of Qwen3.8-27B in 4 bits for PLI_MTP_HEAD (the code is src/tensorfold/install/mtp_head.py).

    python tools/quantize_mtp_head.py --model <27B model dir> SOURCE.safetensors OUTPUT.safetensors [--bits 4] [--group 64]

`pli setup` runs the same code after downloading the head. It needs MLX and PLI's packages: run it with the Python of
PLI's virtual environment.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))     # this clone's code first
from tensorfold.install.mtp_head import main  # noqa: E402

if __name__ == "__main__":
    main()
