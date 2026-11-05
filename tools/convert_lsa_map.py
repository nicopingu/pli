#!/usr/bin/env python3
"""Convert A1's distilled map (Latent Sparse Attention, release d-lsa-weights-v1, Apache-2.0) to NumPy for PLI_A1_LSA_MAP.

The code is src/tensorfold/install/lsa_map.py; `pli setup` downloads and converts the map itself. By hand (no torch
needed; the file of the release for Qwen3.8-27B has the SHA-256
24cee934f969756db228a8df7f2bc7613979b63ac3b1d5ac3d9aa42e4339bb9b):

    gh release download d-lsa-weights-v1 -R Prefrontal-AI/latent-sparse-attention -p qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt
    python tools/convert_lsa_map.py qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt qwen38_27b_d2048.npz

It needs NumPy only: run it with the Python of PLI's virtual environment, or any Python with NumPy.
"""
from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))     # this clone's code first
from tensorfold.install.lsa_map import main  # noqa: E402

if __name__ == "__main__":
    main()
