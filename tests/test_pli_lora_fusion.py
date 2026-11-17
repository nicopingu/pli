"""Tests for tools/pli-lora-fusion on a fake tiny model (no real model, numpy only).

    python -m pytest -q tests/test_pli_lora_fusion.py      ou      python -m unittest tests.test_pli_lora_fusion
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

OUTIL = Path(__file__).resolve().parents[1] / "tools" / "pli-lora-fusion"


def ecrire_safetensors(chemin: Path, tenseurs: dict[str, np.ndarray], meta: dict | None = None) -> None:
    entete, donnees, pos = {}, b"", 0
    for nom, t in tenseurs.items():
        b = np.ascontiguousarray(t, dtype="<f4").tobytes()
        entete[nom] = {"dtype": "F32", "shape": list(t.shape), "data_offsets": [pos, pos + len(b)]}
        donnees += b
        pos += len(b)
    if meta:
        entete["__metadata__"] = meta
    h = json.dumps(entete).encode()
    chemin.write_bytes(struct.pack("<Q", len(h)) + h + donnees)


def lire_safetensors(chemin: Path) -> dict[str, np.ndarray]:
    brut = chemin.read_bytes()
    n = struct.unpack("<Q", brut[:8])[0]
    entete = json.loads(brut[8:8 + n])
    entete.pop("__metadata__", None)
    return {k: np.frombuffer(brut[8 + n + v["data_offsets"][0]:8 + n + v["data_offsets"][1]], "<f4").reshape(v["shape"])
            for k, v in entete.items()}


class Fusion(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory()
        self.d = Path(self.tmp.name)
        rng = np.random.default_rng(0)
        self.modele = self.d / "modele"
        tdir = self.modele / "FL2VA" / "transformer"
        tdir.mkdir(parents=True)
        (tdir / "config.json").write_text("{}")
        (self.modele / "model_index.json").write_text("{}")
        self.W = {"blocks.0.attn.qkv_proj.weight": rng.standard_normal((768, 8)).astype(np.float32),   # 2 heads of 128
                  "blocks.0.attn.out_proj.weight": rng.standard_normal((8, 8)).astype(np.float32)}
        ecrire_safetensors(tdir / "a.safetensors", self.W)
        self.autre = {"blocks.1.mlp.fc2.weight": rng.standard_normal((8, 8)).astype(np.float32)}
        ecrire_safetensors(tdir / "b.safetensors", self.autre)
        self.rng = rng

    def tearDown(self) -> None:
        self.tmp.cleanup()

    def lancer(self, sortie: Path, *lora: str, essai: bool = False):
        cmd = [sys.executable, str(OUTIL), str(self.modele), str(sortie)]
        for x in lora:
            cmd += ["--lora", x]
        if essai:
            cmd.append("--essai")
        return subprocess.run(cmd, capture_output=True, text=True, check=False)

    def test_lora_natif_avec_alpha(self) -> None:
        A = self.rng.standard_normal((2, 8)).astype(np.float32)
        B = self.rng.standard_normal((8, 2)).astype(np.float32)
        lora = self.d / "natif.safetensors"
        ecrire_safetensors(lora, {"blocks.0.attn.out_proj.lora_A.weight": A, "blocks.0.attn.out_proj.lora_B.weight": B,
                                  "blocks.0.attn.out_proj.alpha": np.array(4.0, np.float32)})
        sortie = self.d / "sortie"
        r = self.lancer(sortie, f"{lora}@0.5")
        self.assertEqual(r.returncode, 0, r.stderr)
        t = sortie / "FL2VA" / "transformer"
        fusion = lire_safetensors(t / "a.safetensors")
        attendu = self.W["blocks.0.attn.out_proj.weight"] + 0.5 * (4.0 / 2) * (B @ A)
        np.testing.assert_allclose(fusion["blocks.0.attn.out_proj.weight"], attendu, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(fusion["blocks.0.attn.qkv_proj.weight"], self.W["blocks.0.attn.qkv_proj.weight"])
        self.assertTrue((t / "b.safetensors").is_symlink(), "an untouched file remains a symlink")
        self.assertTrue((sortie / "model_index.json").is_symlink())
        meta = json.loads((sortie / "fusion_lora_h3.json").read_text())
        self.assertEqual(meta["transformer"], "FL2VA/transformer")
        self.assertEqual(meta["loras"], [{"fichier": str(lora), "force": 0.5}])

    def test_qkv_comfy_permute(self) -> None:
        """Comfy LoRA stores qkv contiguously (p*256 + h*128 + d); native interleaves by head (h*384 + p*128 + d)."""

        A = self.rng.standard_normal((2, 8)).astype(np.float32)
        B = self.rng.standard_normal((768, 2)).astype(np.float32)
        lora = self.d / "comfy.safetensors"
        ecrire_safetensors(lora, {"diffusion_model.blocks.0.attn.qkv_proj.lora_A.weight": A,
                                  "diffusion_model.blocks.0.attn.qkv_proj.lora_B.weight": B})
        sortie = self.d / "sortie"
        r = self.lancer(sortie, str(lora))
        self.assertEqual(r.returncode, 0, r.stderr)
        fusion = lire_safetensors(sortie / "FL2VA" / "transformer" / "a.safetensors")["blocks.0.attn.qkv_proj.weight"]
        delta = B @ A                                          # alpha absent: alpha = rank, scale 1
        n = np.arange(768)
        h, reste = n // 384, n % 384
        p, d = reste // 128, reste % 128
        attendu = self.W["blocks.0.attn.qkv_proj.weight"] + delta[p * 256 + h * 128 + d]
        np.testing.assert_allclose(fusion, attendu, rtol=1e-5, atol=1e-5)

    def test_essai_n_ecrit_rien(self) -> None:
        lora = self.d / "natif.safetensors"
        ecrire_safetensors(lora, {"blocks.0.attn.out_proj.lora_A.weight": np.ones((1, 8), np.float32),
                                  "blocks.0.attn.out_proj.lora_B.weight": np.ones((8, 1), np.float32)})
        sortie = self.d / "sortie"
        r = self.lancer(sortie, str(lora), essai=True)
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("--essai", r.stdout)
        self.assertFalse(sortie.exists())

    def test_cle_inconnue_refusee(self) -> None:
        lora = self.d / "faux.safetensors"
        ecrire_safetensors(lora, {"blocks.0.attn.absent.lora_A.weight": np.ones((1, 8), np.float32),
                                  "blocks.0.attn.absent.lora_B.weight": np.ones((8, 1), np.float32)})
        sortie = self.d / "sortie"
        r = self.lancer(sortie, str(lora))
        self.assertEqual(r.returncode, 2)
        self.assertIn("REFUS", r.stderr)
        self.assertFalse(sortie.exists())

    def test_sortie_non_vide_refusee(self) -> None:
        lora = self.d / "natif.safetensors"
        ecrire_safetensors(lora, {"blocks.0.attn.out_proj.lora_A.weight": np.ones((1, 8), np.float32),
                                  "blocks.0.attn.out_proj.lora_B.weight": np.ones((8, 1), np.float32)})
        sortie = self.d / "sortie"
        sortie.mkdir()
        (sortie / "x").write_text("")
        r = self.lancer(sortie, str(lora))
        self.assertEqual(r.returncode, 2)
        self.assertIn("sortie non vide", r.stderr)


if __name__ == "__main__":
    unittest.main()
