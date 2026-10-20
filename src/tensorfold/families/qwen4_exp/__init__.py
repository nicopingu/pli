"""Qwen3.8 Flash Next forward pass and fused decode with exact MTP drafting on Metal."""

from __future__ import annotations

from pathlib import Path
from typing import Any

MODEL_TYPES = ("qwen4_exp", "qwen3_8_flash_next")   # the second: the name newer exports (Mia-AiLab's NVFP4) carry
TITLE = "Qwen3.8 Flash Next"
LANES = True
# with their MTP head: MLX affine (4-bit the default; oQ4e, oQ5e, 6- and 8-bit read too)
MODELS = ("TensorFold/Qwen3.8-Flash-Next-MLX-4bit-MTP",)
NVFP4_MODELS = ("local-inference-lab/Qwen3.8-Flash-Next-NVFP4", "RadixArk/Qwen3.8-Flash-Next-NVFP4")   # named in refusals only
KERNEL_PACKAGE = "tensorfold.kernels.qwen.flash_next.v1"
KERNEL_VERSION = "v1"
# The CLI sets defaults before MLX starts so expert bindings do not end each command buffer.
MLX_ENV = {"MLX_MAX_OPS_PER_BUFFER": "200", "MLX_MAX_MB_PER_BUFFER": "100000"}


def has_mtp(model_dir: Path) -> bool:
    """Whether the checkpoint kept the MTP head's weights (``mtp.*``)."""

    import json

    index = Path(model_dir) / "model.safetensors.index.json"
    if not index.is_file():
        return False
    return any(".mtp." in name or name.startswith("mtp.") for name in json.loads(index.read_text())["weight_map"])


def check_quantization(config: dict[str, Any], backend: str) -> None:
    """Refuse from config.json alone an MLX format the kernels do not read: affine 2-8 bits in groups of 32-128."""

    from tensorfold.quantization import checkpoint_specs

    specs = [spec for spec in checkpoint_specs(config).values() if spec is not None]
    if not specs:
        raise ValueError(f"{TITLE} reads MLX affine-quantized weights; this checkpoint has none. Use {MODELS[0]}.")
    formats = {(spec.bits, spec.group_size) for spec in specs}


def check(model_dir: Path) -> None:
    from tensorfold.families import EXL3_QUANT, OWN_MODEL_HELP, describe_quantization, quant_method, read_config

    config = read_config(model_dir)
    if quant_method(config) == EXL3_QUANT:
        # an EXL3 pack (refused on a Mac by require_readable): only the MTP head to report
        if (Path(model_dir) / "model.safetensors.index.json").is_file() and not has_mtp(model_dir):
            print("[tensorfold] this EXL3 checkpoint has no MTP head: decoding without MTP drafts", flush=True)
        return
    if quant_method(config) == "modelopt":
        from tensorfold.vision.qwen_checkpoint import vision_key

        # NVFP4 experts and n-gram tables; other linears are bf16, MXFP8 or block FP8
        found = config.get("quantization") or config.get("quantization_config") or {}
        algo = str(found.get("quant_algo") or "NVFP4").upper()
        # FP8 is read in the MTP drafter's experts (dequantized and re-quantized at load: they only draft) and in the
        # n-gram tables (their own FP8 reader, host_table.FP8Table); FP8 elsewhere stays refused
        def fp8_read(name: str) -> bool:
            return {"mtp", "experts"} <= set(name.split(".")) or ".ple.ple_embedding.ngram_embedding." in name + "."

        layers = {str(v.get("quant_algo", "")).upper() for k, v in (found.get("quantized_layers") or {}).items()
                  if not (str(v.get("quant_algo", "")).upper() == "FP8" and fp8_read(k))}
        algos = layers if algo == "MIXED_PRECISION" else {algo}
        weights = [g.get("weights") or {} for g in (found.get("config_groups") or {}).values()]
        fp4 = {int(w.get("group_size", 16)) for w in weights if int(w.get("num_bits", 4)) == 4}
        if not algos <= {"NVFP4", "W4A16_NVFP4", "MXFP8", "FP8_PB_WO"} or fp4 - {16}:
            raise ValueError(f"TensorFold's Flash Next kernels read NVFP4 (ModelOpt FP4) weights in blocks of 16, the "
                             f"other linears bf16, MXFP8 or 128x128-block FP8 ({', '.join(NVFP4_MODELS)}); "
                             "this checkpoint has "
                             + describe_quantization(config) + f". {OWN_MODEL_HELP}")
        # N-gram tables have their own NVFP4 reader; other non-expert layers would cast packed bytes to bf16.
        outside = sorted(name for name, layer in (found.get("quantized_layers") or {}).items()
                         if "NVFP4" in str(layer.get("quant_algo", "")).upper() and "experts" not in name.split(".")
                         and ".ple.ple_embedding.ngram_embedding." not in name + "."
                         and vision_key(name) is None)
        if outside:
            raise ValueError("TensorFold's Flash Next kernels read NVFP4 in routed experts and n-gram tables only; "
                             f"this checkpoint has it on {len(outside)} other layer(s), e.g. {outside[0]}. "
                             f"{OWN_MODEL_HELP}")
        if (Path(model_dir) / "model.safetensors.index.json").is_file() and not has_mtp(model_dir):
            print("[tensorfold] this NVFP4 checkpoint has no MTP head: decoding without MTP drafts", flush=True)
        return
    check_quantization(config, "mlx")
    # Config-only preflight cannot establish whether the MTP head is missing; wait for weights or their index.
    if ((Path(model_dir) / "model.safetensors.index.json").is_file()
            or any(Path(model_dir).glob("model*.safetensors"))) and not has_mtp(model_dir):
        print(f"[tensorfold] this checkpoint has no MTP head: decoding without MTP drafts ({MODELS[0]} has one)",
              flush=True)


# a table's tensors in the checkpoint: weights, scales and biases of each shard, the only bytes read on the host
_TABLE = r"language_model\.model\.layers\.\d+\.ple\.ple_embedding\.ngram_embedding\.shard_\d+\.(weight|scales|biases)"


def ple_bytes(model_dir: Path) -> int:
    """Bytes of the checkpoint's n-gram (PLE) tables, which stay on the host (mapped, or on SSD with --ple-on-ssd)."""

    import re

    from tensorfold.families.qwen4_exp.host_table import read_header

    return sum(entry["data_offsets"][1] - entry["data_offsets"][0]
               for path in Path(model_dir).glob("model*.safetensors")
               for name, entry in read_header(path).items() if re.fullmatch(_TABLE, name))


def expert_bytes(model_dir: Path) -> int:
    """Bytes of the decoder layers' routed expert stacks, which --ssd-experts leaves on disk."""

    from tensorfold.streaming.checkpoint import tensor_bytes

    return tensor_bytes(Path(model_dir), lambda name: name.startswith("language_model.model.layers.")
                        and ".mlp.switch_mlp." in name)


def weight_bytes(model_dir: Path, ple_on_ssd: bool = False) -> int:
    """The bytes MLX loads: the checkpoint, less its n-gram tables where the loader keeps them on the host."""

    from tensorfold.families.qwen4_exp.host_table import ngrams_on_host

    size = sum(p.stat().st_size for p in Path(model_dir).glob("*.safetensors"))
    return size - ple_bytes(model_dir) if ngrams_on_host(model_dir, ple_on_ssd) else size


def load(model_dir: Path, *, mtp_drafts: int | None = None, ple_on_ssd: bool = False,
         ssd_experts: float | None = None, **_: Any) -> tuple[Any, Any]:
    from tensorfold.families.qwen4_exp.runtime import load as load_runtime

    drafts = mtp_drafts if has_mtp(Path(model_dir)) else 0
    return load_runtime(Path(model_dir), drafts=drafts, ple_on_ssd=ple_on_ssd, ssd_experts=ssd_experts)


def engine_settings(model: Any) -> dict[str, Any]:
    """The widest exact window, prompt chunks the memory allows (8,192 where tensor units run MLX's gathers)."""

    from tensorfold.families.qwen3_5 import tensor_units

    width = int(getattr(model, "exact_width", 1) or 1)
    steps = (8192, 4096, 2048) if tensor_units() else (4096, 2048)
    return {"max_rows": width, "max_draft": max(0, width - 1), "prefill_steps": steps}


def kernel_version(model: Any) -> str:
    """Include the loaded model's prompt-attention selection modes in its kernel fingerprint."""

    from tensorfold.families import families, kernel_source_version

    modes = [str(int(bool(getattr(layer.self_attn, "kernel_select", False))))
             for layer in getattr(model, "layers", ()) if hasattr(layer, "self_attn")]
    source = kernel_source_version(families()["qwen4_exp"])
    prefill = getattr(model, "prefill_key", None)                     # the prefill path, matmul route and GPU
    return f"{source}|prompt_attention={','.join(modes)}" + (f"|{prefill}" if prefill else "")
