"""The NVFP4 / FP8 reader against an independent dequantizer (torch's float8 types and the e2m1 value list)."""

from __future__ import annotations

import numpy as np
import pytest

from tensorfold.formats import nvfp4 as fmt

E2M1_VALUES = [0, 0.5, 1, 1.5, 2, 3, 4, 6]


def _codes(rng, n, k):
    return rng.integers(0, 256, size=(n, k // 2), dtype=np.uint8)






def test_compressed_tensors_global_scale_is_a_reciprocal():
    packed = np.array([[0x21, 0x07]], dtype=np.uint8)                          # codes 1, 2, 7, 0
    scale = np.full((1, 1), 0x38, dtype=np.uint8)                              # e4m3 1.0
    a = fmt.dequant("nvfp4", np.pad(packed, ((0, 0), (0, 6))), scale, 4.0, reciprocal_global=True)
    assert list(a[0, :4]) == [0.125, 0.25, 1.5, 0.0]


def test_schemes_from_tensor_storage():
    assert fmt.scheme({"weight": ("U8", [8, 32]), "weight_scale": ("F8_E4M3", [8, 4]),
                       "weight_scale_2": ("F32", [])}) == "nvfp4"
    assert fmt.scheme({"weight_packed": ("U8", [8, 32]), "weight_scale": ("F8_E4M3", [8, 4]),
                       "weight_global_scale": ("F32", [1])}) == "nvfp4"
    assert fmt.scheme({"weight": ("F8_E4M3", [8, 64]), "weight_scale": ("F32", [])}) == "fp8"
    assert fmt.scheme({"weight": ("F8_E4M3", [8, 64]), "weight_scale": ("U8", [8, 2])}) == "mxfp8"
    assert fmt.scheme({"weight": ("BF16", [8, 64])}) == "bf16"


def test_config_gate_takes_nvfp4_and_fp8_and_refuses_integer_weights():
    kw = dict(where="CUDA", tested="x", help="")
    ok = {"quantization_config": {"quant_method": "modelopt", "config_groups": {
        "a": {"weights": {"num_bits": 8, "type": "float"}}, "b": {"weights": {"num_bits": 4, "type": "float",
                                                                               "group_size": 16}}}}}
    fmt.require_config(ok, **kw)
    assert fmt.config_block(ok)["quant_method"] == "modelopt"
    ct = {"quantization_config": {"quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized",
                                  "config_groups": {"g": {"weights": {"num_bits": 4, "type": "float"}}}}}
    fmt.require_config(ct, **kw)
    awq = {"quantization_config": {"quant_method": "compressed-tensors", "format": "pack-quantized",
                                   "config_groups": {"g": {"weights": {"num_bits": 4, "type": "int"}}}}}
    with pytest.raises(ValueError, match="4-bit int"):
        fmt.require_config(awq, **kw)




def test_block_fp8_scheme_from_tensor_storage():
    assert fmt.scheme({"weight": ("F8_E4M3", [200, 256]), "weight_scale_inv": ("F32", [2, 2])}) == "fp8block"
    assert fmt.scheme({"weight": ("F8_E4M3", [200, 256]), "weight_scale": ("F32", [])}) == "fp8"
