#!/usr/bin/env python3
"""Convert a distilled map of Latent Sparse Attention (Prefrontal-AI, release d-lsa-weights-v1, Apache-2.0) for PLI's A1.

The map is a PyTorch file of `<layer>.W_Q_idx` / `<layer>.W_K_idx` tensors, [768, 64] each; PLI reads a NumPy .npz of the same
tensors in float32 (PLI_A1_LSA_MAP=<file.npz>). The .pt is read with weights_only=True: no code of the file is run. Needs torch.

Usage (the release's file for Qwen3.8-27B, SHA-256 24cee934f969756db228a8df7f2bc7613979b63ac3b1d5ac3d9aa42e4339bb9b):
  gh release download d-lsa-weights-v1 -R Prefrontal-AI/latent-sparse-attention -p qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt
  python tools/convert_lsa_map.py qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt qwen38_27b_d2048.npz
"""

import sys

import numpy as np
import torch


def main() -> int:
    if len(sys.argv) != 3:
        print(__doc__)
        return 2
    src, dst = sys.argv[1], sys.argv[2]
    blob = torch.load(src, map_location="cpu", weights_only=True)
    if not isinstance(blob, dict) or not blob:
        print("not a dict of tensors", file=sys.stderr)
        return 1
    out = {}
    for name, t in sorted(blob.items(), key=lambda kv: (int(kv[0].split(".")[0]), kv[0])):
        layer, _, kind = name.partition(".")
        if not (layer.isdigit() and kind in ("W_Q_idx", "W_K_idx") and isinstance(t, torch.Tensor)):
            print(f"unexpected entry {name}", file=sys.stderr)
            return 1
        a = t.detach().float().numpy()
        out[name] = a
        print(name, tuple(a.shape), str(t.dtype))
    np.savez(dst, **out)
    print("layers", sorted({int(n.split(".")[0]) for n in out}), "->", dst)
    return 0


if __name__ == "__main__":
    sys.exit(main())
