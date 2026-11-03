#!/usr/bin/env python3
"""Write the MTP head of Qwen3.8-27B in 4 bits for PLI_MTP_HEAD.

The linear layers are quantized; the norms stay in floating point. The engine reads the format of each layer from
the file (tensorfold.families.qwen3_5_moe.mtp.load), so the 4-bit file and the bf16 file are both accepted.

    python tools/quantize_mtp_head.py --model <27B model dir> SOURCE.safetensors OUTPUT.safetensors [--bits 4] [--group 64]

SOURCE is the bf16 head, for example model.safetensors of EigenLabs/Qwen3.8-27B-MTP-bf16 (the 15 mtp.* tensors of
Qwen3.8-27B, copied unchanged). --model is the 27B the head will draft for; its config.json gives the layer shapes.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models import qwen3_5

from tensorfold.families.qwen3_5_moe import mtp


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", type=Path)
    ap.add_argument("output", type=Path)
    ap.add_argument("--model", type=Path, required=True, help="27B model directory (config.json)")
    ap.add_argument("--bits", type=int, default=4)
    ap.add_argument("--group", type=int, default=64)
    a = ap.parse_args()

    cfg = json.loads((a.model / "config.json").read_text())
    args = qwen3_5.TextModelArgs.from_dict(cfg.get("text_config", cfg))
    head = mtp.load(a.source, args)
    before = sum(v.nbytes for _, v in tree_flatten(head.parameters()))
    nn.quantize(head, group_size=a.group, bits=a.bits,
                class_predicate=lambda _p, m: isinstance(m, nn.Linear) and m.weight.shape[-1] % a.group == 0)
    mx.eval(head.parameters())
    weights = dict(tree_flatten(head.parameters()))
    mx.save_safetensors(str(a.output), weights)
    after = sum(v.nbytes for v in weights.values())
    print(f"MTP head: {before / 2**20:.0f} MiB in floating point, {after / 2**20:.0f} MiB in {a.bits} bits "
          f"(groups of {a.group}) -> {a.output}")


if __name__ == "__main__":
    main()
