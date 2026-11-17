"""Tests for tools/pli-lora-image on tiny fake LoRAs (numpy only, no model)."""

from __future__ import annotations

import json
import struct
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

OUTIL = Path(__file__).resolve().parents[1] / "tools" / "pli-lora-image"


def ecrire(chemin: Path, tenseurs: dict, meta: dict | None = None, dtype: str = "F32") -> None:
    entete, corps, pos = {}, b"", 0
    for nom, t in tenseurs.items():
        t = np.asarray(t, np.float32)
        if dtype == "BF16" and not nom.endswith(".alpha"):
            b = (t.view(np.uint32) >> 16).astype("<u2").tobytes()       # valeurs choisies exactes en bf16
            dt = "BF16"
        else:
            b, dt = t.astype("<f4").tobytes(), "F32"
        entete[nom] = {"dtype": dt, "shape": list(t.shape), "data_offsets": [pos, pos + len(b)]}
        corps += b
        pos += len(b)
    if meta:
        entete["__metadata__"] = meta
    h = json.dumps(entete).encode()
    chemin.write_bytes(struct.pack("<Q", len(h)) + h + corps)


def lire(chemin: Path) -> dict:
    brut = chemin.read_bytes()
    n = struct.unpack("<Q", brut[:8])[0]
    entete = json.loads(brut[8:8 + n])
    assert "__metadata__" not in entete
    sortie = {}
    for k, v in entete.items():
        b = brut[8 + n + v["data_offsets"][0]:8 + n + v["data_offsets"][1]]
        x = (np.frombuffer(b, "<f4") if v["dtype"] == "F32"
             else (np.frombuffer(b, "<u2").astype(np.uint32) << 16).view(np.float32))
        sortie[k] = (v["dtype"], x.reshape(v["shape"]))
    return sortie


def lancer(*args):
    return subprocess.run([sys.executable, str(OUTIL), *map(str, args)], capture_output=True, text=True, check=False)


A = np.array([[1.0, 2.0, 0.5, -1.0], [0.25, -2.0, 4.0, 1.0]], np.float32)          # rank 2, input 4
B = np.array([[1.0, -1.0], [2.0, 0.5], [0.0, 3.0]], np.float32)                   # sortie 3


def test_down_up_avec_alpha_replie_dans_lora_b(tmp_path):
    entree, sortie = tmp_path / "a.safetensors", tmp_path / "out" / "b.safetensors"
    ecrire(entree, {"transformer_blocks.0.attn.to_q.lora.down.weight": A,
                    "transformer_blocks.0.attn.to_q.lora.up.weight": B,
                    "transformer_blocks.0.attn.to_q.alpha": np.array(1.0, np.float32)})
    r = lancer(entree, sortie)
    assert r.returncode == 0, r.stderr
    t = lire(sortie)
    assert set(t) == {"transformer_blocks.0.attn.to_q.lora_A.weight", "transformer_blocks.0.attn.to_q.lora_B.weight"}
    np.testing.assert_array_equal(t["transformer_blocks.0.attn.to_q.lora_A.weight"][1], A)
    np.testing.assert_array_equal(t["transformer_blocks.0.attn.to_q.lora_B.weight"][1], B * 0.5)    # alpha 1 / rang 2


def test_peft_alpha_de_la_metadonnee_et_bf16_garde(tmp_path):
    entree, sortie = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    meta = {"lora_adapter_metadata": json.dumps({"transformer.lora_alpha": 8})}
    ecrire(entree, {"x.lora_A.weight": A, "x.lora_B.weight": B}, meta, dtype="BF16")
    r = lancer(entree, sortie)
    assert r.returncode == 0, r.stderr
    t = lire(sortie)
    assert t["x.lora_B.weight"][0] == "BF16"
    np.testing.assert_array_equal(t["x.lora_B.weight"][1], B * 4.0)                  # alpha 8 / rang 2


def test_sans_alpha_echelle_un_et_essai_n_ecrit_rien(tmp_path):
    entree, sortie = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    ecrire(entree, {"x.lora_down.weight": A, "x.lora_up.weight": B})
    r = lancer(entree, sortie, "--essai")
    assert r.returncode == 0 and "--essai" in r.stdout and not sortie.exists()
    assert lancer(entree, sortie).returncode == 0
    np.testing.assert_array_equal(lire(sortie)["x.lora_B.weight"][1], B)


@pytest.mark.parametrize("tenseurs, meta, motif", [
    ({"x.lora_A.weight": A, "x.lora_B.weight": B, "x.autre.weight": A}, None, "unexpected"),
    ({"x.lora_A.weight": A}, None, "incomplete pairs"),
    ({"x.lora_A.weight": A, "x.lora_B.weight": B},
     {"lora_adapter_metadata": json.dumps({"transformer.use_rslora": True})}, "rslora"),
])
def test_refus_sans_rien_ecrire(tmp_path, tenseurs, meta, motif):
    entree, sortie = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    ecrire(entree, tenseurs, meta)
    r = lancer(entree, sortie)
    assert r.returncode == 2 and motif in r.stderr and not sortie.exists()


def test_sortie_existante_refusee(tmp_path):
    entree, sortie = tmp_path / "a.safetensors", tmp_path / "b.safetensors"
    ecrire(entree, {"x.lora_A.weight": A, "x.lora_B.weight": B})
    sortie.write_bytes(b"x")
    r = lancer(entree, sortie)
    assert r.returncode == 2 and "already exists" in r.stderr
