"""A1's map: the PyTorch file read without torch, the conversion, the checksum, the model check.

tests/fixtures/lsa-tiny.pt was written by torch.save (torch 2.13.0) from four small tensors: a plain one, a transposed
view (strides 1, 3), a view at an offset into a larger storage, and a float16 one.
"""

from __future__ import annotations

import io
import json
from pathlib import Path
import pickle
import zipfile

import numpy as np
import pytest

from tensorfold.install import lsa_map

TINY = Path(__file__).parent / "fixtures" / "lsa-tiny.pt"


def test_the_tensors_are_read_without_torch():
    t = lsa_map.read_pt(TINY)
    assert list(t) == ["3.W_Q_idx", "3.W_K_idx", "7.W_Q_idx", "7.W_K_idx"]
    assert np.array_equal(t["3.W_Q_idx"], np.arange(12, dtype=np.float32).reshape(3, 4))
    assert np.array_equal(t["3.W_K_idx"], np.arange(6, dtype=np.float32).reshape(2, 3).T)
    assert np.array_equal(t["7.W_Q_idx"], np.arange(5, 17, dtype=np.float32).reshape(3, 4))
    assert t["7.W_K_idx"].dtype == np.float16
    assert np.array_equal(t["7.W_K_idx"], (np.arange(8, dtype=np.float16) / 4).reshape(2, 4))


def test_convert_writes_float32_arrays_by_layer(tmp_path):
    out = tmp_path / "map.npz"
    assert lsa_map.convert(TINY, out) == [3, 7]
    z = np.load(out)
    assert z.files == ["3.W_K_idx", "3.W_Q_idx", "7.W_K_idx", "7.W_Q_idx"]
    assert all(z[name].dtype == np.float32 and z[name].flags.c_contiguous for name in z.files)
    assert np.array_equal(z["7.W_K_idx"], (np.arange(8, dtype=np.float32) / 4).reshape(2, 4))


def _zip(path: Path, data_pkl: bytes, extra: dict[str, bytes] | None = None) -> Path:
    with zipfile.ZipFile(path, "w") as z:
        z.writestr("x/data.pkl", data_pkl)
        for name, data in (extra or {}).items():
            z.writestr(name, data)
    return path


def test_a_file_that_names_other_code_is_refused(tmp_path):
    evil = _zip(tmp_path / "evil.pt", b"cos\nsystem\n(S'echo pwned'\ntR.")
    with pytest.raises(pickle.UnpicklingError, match="os.system is not allowed"):
        lsa_map.read_pt(evil)


def test_a_tensor_that_reads_past_its_storage_is_refused(tmp_path):
    # a float32 storage of 4 values, and a tensor of 8 values over it (the pickle torch.save writes, by hand)
    proto = (b"\x80\x02}q\x00X\x01\x00\x00\x00aq\x01ctorch._utils\n_rebuild_tensor_v2\nq\x02((X\x07\x00\x00\x00storage"
             b"ctorch\nFloatStorage\nX\x01\x00\x00\x000X\x03\x00\x00\x00cpuK\x04tQK\x00K\x08\x85K\x01\x85\x89"
             b"ccollections\nOrderedDict\n)RtRs.")
    bad = _zip(tmp_path / "bad.pt", proto, {"x/data/0": np.arange(4, dtype=np.float32).tobytes()})
    with pytest.raises(ValueError, match="past its storage"):
        lsa_map.read_pt(bad)


def test_not_a_torch_file(tmp_path):
    with zipfile.ZipFile(tmp_path / "a.pt", "w") as z:
        z.writestr("a/other.bin", b"")
    with pytest.raises(ValueError, match="data.pkl not found"):
        lsa_map.read_pt(tmp_path / "a.pt")


def test_download_checks_the_sha256_and_keeps_nothing_when_it_differs(tmp_path):
    out = tmp_path / "models" / lsa_map.NPZ
    with pytest.raises(ValueError, match="SHA-256"):
        lsa_map.download(out, opener=lambda url: io.BytesIO(TINY.read_bytes()))
    assert not out.exists() and list(out.parent.iterdir()) == []


def test_download_converts_a_file_with_the_right_sha256(tmp_path, monkeypatch):
    monkeypatch.setattr(lsa_map, "SHA256", lsa_map.sha256(TINY))
    seen = []

    def opener(url):
        seen.append(url)
        return io.BytesIO(TINY.read_bytes())

    out = tmp_path / "models" / lsa_map.NPZ
    lsa_map.download(out, opener=opener)
    assert seen == [lsa_map.URL] and sorted(np.load(out).files)[0] == "3.W_K_idx"
    assert [p.name for p in out.parent.iterdir()] == [lsa_map.NPZ]


def test_the_map_fits_the_27b_only(tmp_path):
    def model(name, cfg):
        d = tmp_path / name
        d.mkdir()
        (d / "config.json").write_text(json.dumps(cfg))
        return d

    text = {"num_hidden_layers": 64, "num_key_value_heads": 4, "head_dim": 256}
    assert lsa_map.fits(model("vl", {"model_type": "qwen3_5", "text_config": text}))
    assert lsa_map.fits(model("flat", text))
    assert not lsa_map.fits(model("small", {**text, "num_hidden_layers": 40}))
    assert not lsa_map.fits(tmp_path / "missing")
    assert not lsa_map.fits(None)
