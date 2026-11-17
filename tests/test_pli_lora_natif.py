"""Tests for tools/pli-lora-natif on a fake tiny BF16 model: the in-memory form gives the fused weights.

    python -m pytest -q tests/test_pli_lora_natif.py
"""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

RACINE = Path(__file__).resolve().parents[1]
NATIF = RACINE / "tools" / "pli-lora-natif"


def bf16(x: np.ndarray) -> np.ndarray:
    u = np.ascontiguousarray(x, dtype=np.float32).view(np.uint32)
    return ((u + (((u >> 16) & 1) + 0x7FFF)) >> 16).astype("<u2")


def f32(u: np.ndarray) -> np.ndarray:
    return (u.astype(np.uint32) << 16).view(np.float32)


def ecrire(chemin: Path, tenseurs: dict[str, tuple[str, np.ndarray]], meta: dict | None = None) -> None:
    entete, donnees, pos = {}, b"", 0
    for nom, (dt, t) in tenseurs.items():
        b = (bf16(t) if dt == "BF16" else np.ascontiguousarray(t, dtype="<f4")).tobytes()
        entete[nom] = {"dtype": dt, "shape": list(t.shape), "data_offsets": [pos, pos + len(b)]}
        donnees += b
        pos += len(b)
    if meta:
        entete["__metadata__"] = meta
    h = json.dumps(entete).encode()
    chemin.write_bytes(struct.pack("<Q", len(h)) + h + donnees)


def lire(chemin: Path) -> dict[str, np.ndarray]:
    brut = chemin.read_bytes()
    n = struct.unpack("<Q", brut[:8])[0]
    entete = json.loads(brut[8:8 + n])
    entete.pop("__metadata__", None)
    return {k: f32(np.frombuffer(brut[8 + n + v["data_offsets"][0]:8 + n + v["data_offsets"][1]], "<u2")).reshape(v["shape"])
            for k, v in entete.items()}


@pytest.fixture
def modele(tmp_path):
    rng = np.random.default_rng(0)
    tdir = tmp_path / "modele" / "FL2VA" / "transformer"
    tdir.mkdir(parents=True)
    (tdir / "config.json").write_text("{}")
    W = {"blocks.0.attn.qkv_proj.weight": rng.standard_normal((768, 8)),     # 2 heads of 128
         "blocks.0.mlp.fc1.weight": rng.standard_normal((16, 8)),
         "audio_patch_proj.weight": rng.standard_normal((8, 4))}
    ecrire(tdir / "a.safetensors", {k: ("F32" if k.startswith("audio") else "BF16", v) for k, v in W.items()})
    return tmp_path / "modele", {k: f32(bf16(v)).reshape(v.shape) for k, v in W.items()}, rng


def lancer(modele: Path, lora: Path, sortie: Path, *plus: str):
    return subprocess.run([sys.executable, str(NATIF), str(modele), str(lora), str(sortie), *plus],
                          capture_output=True, text=True, check=False)


def test_comfy_qkv_remis_par_tete_et_alpha(modele, tmp_path):
    racine, W, rng = modele
    A, B = rng.standard_normal((4, 8)), rng.standard_normal((768, 4))
    lora = tmp_path / "comfy.safetensors"
    ecrire(lora, {"diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": ("F32", A),
                  "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": ("F32", B),
                  "diffusion_model.blocks.0.attn.qkv_proj.alpha": ("F32", np.array(2.0))})
    sortie = tmp_path / "natif.safetensors"
    r = lancer(racine, lora, sortie)
    assert r.returncode == 0, r.stderr
    t = lire(sortie)
    delta = t["blocks.0.attn.qkv_proj.lora_up"] @ t["blocks.0.attn.qkv_proj.lora_down_t"].T
    n = np.arange(768)
    h, p, d = n // 384, (n % 384) // 128, n % 128
    attendu = (B[p * 256 + h * 128 + d] * 0.5) @ A          # contiguous Comfy rows -> native rows, alpha / rank = 0.5
    assert np.allclose(delta, attendu, rtol=2e-2, atol=2e-2)


def test_peft_trois_rangs_et_fc1_inverse(modele, tmp_path):
    racine, W, rng = modele
    t = {}
    for p in ("q", "k", "v"):
        t[f"transformer_blocks.0.attn.to_{p}.lora_A.default.weight"] = ("F32", rng.standard_normal((2, 8)))
        t[f"transformer_blocks.0.attn.to_{p}.lora_B.default.weight"] = ("F32", rng.standard_normal((256, 2)))
    t["transformer_blocks.0.ff.net.0.proj.lora_A.default.weight"] = ("F32", rng.standard_normal((2, 8)))
    t["transformer_blocks.0.ff.net.0.proj.lora_B.default.weight"] = ("F32", rng.standard_normal((16, 2)))
    lora = tmp_path / "peft.safetensors"
    ecrire(lora, t, {"alpha": "1"})
    sortie = tmp_path / "natif.safetensors"
    r = lancer(racine, lora, sortie)
    assert r.returncode == 0, r.stderr
    s = lire(sortie)
    assert s["blocks.0.attn.qkv_proj.lora_up"].shape == (768, 6)
    B = t["transformer_blocks.0.ff.net.0.proj.lora_B.default.weight"][1]
    A = t["transformer_blocks.0.ff.net.0.proj.lora_A.default.weight"][1]
    delta = s["blocks.0.mlp.fc1.lora_up"] @ s["blocks.0.mlp.fc1.lora_down_t"].T
    attendu = (B[(np.arange(16) + 8) % 16] * 0.5) @ A        # [value; gate] -> [gate; value]
    assert np.allclose(delta, attendu, rtol=2e-2, atol=2e-2)


def test_refus_sans_rien_ecrire(modele, tmp_path):
    racine, W, rng = modele
    lora = tmp_path / "audio.safetensors"
    ecrire(lora, {"diffusion_model.audio_patch_proj.lora_A.weight": ("F32", rng.standard_normal((2, 4))),
                  "diffusion_model.audio_patch_proj.lora_B.weight": ("F32", rng.standard_normal((8, 2)))})
    sortie = tmp_path / "natif.safetensors"
    r = lancer(racine, lora, sortie)
    assert r.returncode == 2 and "BF16 matrix" in r.stderr and not sortie.exists()
