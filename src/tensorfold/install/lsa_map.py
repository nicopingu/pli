"""A1's distilled map: downloaded from Latent Sparse Attention's release, checked, converted to NumPy without torch.

A1 (``PLI_A1_LSA_MAP``) scores the tiles of keys a prompt chunk reads with the index of
[Latent Sparse Attention](https://github.com/Prefrontal-AI/latent-sparse-attention) (Prefrontal-AI, Apache-2.0). The
map for Qwen3.8-27B is a PyTorch file of ``<layer>.W_Q_idx`` / ``<layer>.W_K_idx`` tensors, [768, 64] float32 each;
PLI reads the same arrays from a NumPy .npz.

The .pt is a zip archive: a pickle that names the tensors, and one raw file per tensor. It is read here with a pickle
reader that accepts only the five names a tensor file uses (the tensor rebuild, three storage types and OrderedDict)
and refuses anything else, so no code from the file is run, and torch is not needed.

    python -m tensorfold.install.lsa_map SOURCE.pt OUTPUT.npz
    python tools/convert_lsa_map.py SOURCE.pt OUTPUT.npz        (the same, from a clone of the repository)

`pli setup` downloads the file (after asking), checks its SHA-256, converts it into ``<prefix>/models`` and sets
``PLI_A1_LSA_MAP`` with ``PLI_S3B_THRESHOLD`` and ``PLI_S3B_ROWS`` for the service.
"""

from __future__ import annotations

import argparse
import collections
import hashlib
import json
import os
from pathlib import Path
import pickle
import tempfile
import urllib.request
import zipfile

import numpy as np

RELEASE = "https://github.com/Prefrontal-AI/latent-sparse-attention/releases/tag/d-lsa-weights-v1"
FILE = "qwen38_27b__long-16k-1M-heavy-key-and-query-d2048.pt"
URL = f"https://github.com/Prefrontal-AI/latent-sparse-attention/releases/download/d-lsa-weights-v1/{FILE}"
SHA256 = "24cee934f969756db228a8df7f2bc7613979b63ac3b1d5ac3d9aa42e4339bb9b"
BYTES = 6_302_911
LICENSE = "Apache-2.0"
NPZ = "qwen38_27b_d2048.npz"                    # under PREFIX/models
# the S3b values that go with the map: chunks past 16 384 positions, up to 4 096 rows, read through A1
ENV = {"PLI_S3B_THRESHOLD": "16384", "PLI_S3B_ROWS": "4096"}
SHAPE = (768, 64)
# the 27B the map is distilled for (text config of Qwen3.8-27B and its fine-tunes)
TARGET = {"num_hidden_layers": 64, "num_key_value_heads": 4, "head_dim": 256}

_STORAGES = {"FloatStorage": np.float32, "HalfStorage": np.float16, "DoubleStorage": np.float64}


class _Storage:
    def __init__(self, dtype):
        self.dtype = np.dtype(dtype).newbyteorder("<")


def _rebuild(storage, offset, size, stride, requires_grad=False, hooks=None, metadata=None):
    """torch._utils._rebuild_tensor_v2, for NumPy: the view of ``storage`` the tensor is, copied."""

    size, stride, offset = tuple(int(n) for n in size), tuple(int(n) for n in stride), int(offset)
    if any(n < 0 for n in size + stride) or offset < 0 or len(size) != len(stride):
        raise ValueError("tensor with a negative size, stride or offset")
    if 0 in size:
        return np.zeros(size, dtype=storage.dtype.newbyteorder("="))
    last = offset + sum((n - 1) * s for n, s in zip(size, stride))
    if last >= storage.size:
        raise ValueError(f"tensor reads past its storage ({last} >= {storage.size})")
    view = np.lib.stride_tricks.as_strided(storage[offset:], shape=size,
                                           strides=tuple(s * storage.itemsize for s in stride), writeable=False)
    return np.array(view, dtype=storage.dtype.newbyteorder("="))


class _Reader(pickle.Unpickler):
    def __init__(self, data: bytes, archive: zipfile.ZipFile, root: str):
        import io

        super().__init__(io.BytesIO(data))
        self.archive, self.root, self.loaded = archive, root, {}

    def find_class(self, module, name):
        if (module, name) == ("torch._utils", "_rebuild_tensor_v2"):
            return _rebuild
        if module == "torch" and name in _STORAGES:
            return _Storage(_STORAGES[name])
        if (module, name) == ("collections", "OrderedDict"):
            return collections.OrderedDict
        raise pickle.UnpicklingError(f"{module}.{name} is not allowed in a map file")

    def persistent_load(self, pid):
        if not (isinstance(pid, tuple) and len(pid) == 5 and pid[0] == "storage" and isinstance(pid[1], _Storage)):
            raise pickle.UnpicklingError(f"unexpected persistent id {pid!r}")
        _, kind, key, _location, numel = pid
        if key not in self.loaded:
            raw = self.archive.read(f"{self.root}/data/{key}")
            data = np.frombuffer(raw, dtype=kind.dtype)
            if data.size < int(numel):
                raise ValueError(f"storage {key}: {data.size} values, {numel} announced")
            self.loaded[key] = data
        return self.loaded[key]


def read_pt(path: str | os.PathLike) -> dict[str, np.ndarray]:
    """The tensors of a PyTorch file saved with ``torch.save`` (zip format), as NumPy arrays, without torch."""

    with zipfile.ZipFile(path) as archive:
        pickles = [n for n in archive.namelist() if n.endswith("/data.pkl") and n.count("/") == 1]
        if len(pickles) != 1:
            raise ValueError(f"{path}: not a PyTorch zip file (data.pkl not found)")
        root = pickles[0].split("/")[0]
        order = f"{root}/byteorder"
        if order in archive.namelist() and archive.read(order).strip() != b"little":
            raise ValueError(f"{path}: big-endian tensors are not read")
        blob = _Reader(archive.read(pickles[0]), archive, root).load()
    if not isinstance(blob, dict) or not blob or not all(isinstance(v, np.ndarray) for v in blob.values()):
        raise ValueError(f"{path}: not a dict of tensors")
    return dict(blob)


def convert(source: str | os.PathLike, output: str | os.PathLike) -> list[int]:
    """Write the map of ``source`` (.pt) to ``output`` (.npz, float32); returns the layers it covers."""

    tensors = read_pt(source)
    out: dict[str, np.ndarray] = {}
    for name in sorted(tensors, key=lambda n: (int(n.split(".")[0]) if n.split(".")[0].isdigit() else -1, n)):
        layer, _, kind = name.partition(".")
        if not (layer.isdigit() and kind in ("W_Q_idx", "W_K_idx")):
            raise ValueError(f"unexpected entry {name}")
        out[name] = np.ascontiguousarray(tensors[name], dtype=np.float32)
    layers = sorted({int(n.split(".")[0]) for n in out})
    if any(f"{layer}.{kind}" not in out for layer in layers for kind in ("W_Q_idx", "W_K_idx")):
        raise ValueError("a layer lacks W_Q_idx or W_K_idx")
    np.savez(output, **out)
    return layers


def fits(model_dir: str | os.PathLike | None) -> bool:
    """Is the model in ``model_dir`` the 27B the map is distilled for (its config.json)?"""

    if model_dir is None:
        return False
    try:
        cfg = json.loads((Path(model_dir) / "config.json").read_text())
    except (OSError, ValueError):
        return False
    text = cfg.get("text_config", cfg)
    return all(text.get(key) == value for key, value in TARGET.items())


def sha256(path: str | os.PathLike) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(output: Path, opener=urllib.request.urlopen) -> None:
    """Download the release's file, check its SHA-256, convert it to ``output`` (.npz). Nothing is kept on failure."""

    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as tmp:
        pt = Path(tmp) / FILE
        with opener(URL) as answer, open(pt, "wb") as f:
            while chunk := answer.read(1 << 20):
                f.write(chunk)
        got = sha256(pt)
        if got != SHA256:
            raise ValueError(f"{FILE}: SHA-256 {got}, expected {SHA256}; nothing kept")
        npz = Path(tmp) / NPZ
        convert(pt, npz)
        os.replace(npz, output)


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("source", type=Path, help=f"the map (.pt), for Qwen3.8-27B {FILE} of {RELEASE}")
    ap.add_argument("output", type=Path, help="the .npz to write, for PLI_A1_LSA_MAP")
    a = ap.parse_args(argv)
    if a.source.name == FILE and sha256(a.source) != SHA256:
        print(f"warning: {a.source} does not have the release's SHA-256 {SHA256}")
    layers = convert(a.source, a.output)
    print(f"layers {layers} -> {a.output}")


if __name__ == "__main__":
    main()
