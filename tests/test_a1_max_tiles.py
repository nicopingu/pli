"""PLI (05/11/2026): PLI_A1_MAX_TILES, cap on the tiles kept per row (dynamic A1)."""

from __future__ import annotations

import pytest

pytest.importorskip("mlx.core")

from tensorfold.pli import a1_lsa as A1  # noqa: E402


def test_cap(monkeypatch):
    monkeypatch.setattr(A1, "MAX_TILES", 0)
    assert A1.kept(4000, 10) == 400
    monkeypatch.setattr(A1, "MAX_TILES", 160)
    assert A1.kept(4000, 10) == 160 and A1.kept(1000, 10) == 100 and A1.kept(3, 10) == 1


def test_cap_changes_the_prefill(monkeypatch):
    """Prefill of a prompt chunk past the threshold: the cap reduces the tiles read; without a cap, same bits as before."""
    import importlib.util, os, sys
    spec = importlib.util.spec_from_file_location("t_a1cap", os.path.join(os.path.dirname(__file__), "test_a1_lsa.py"))
    T = importlib.util.module_from_spec(spec); sys.modules["t_a1cap"] = T; spec.loader.exec_module(T)
    from tensorfold.kernels.qwen.dense.v1 import exact_attention as EA
    for k, v in (("REDUCED", 10), ("ANCHOR", 32), ("ATTM", True), ("REDUCED_THRESHOLD", T.THRESHOLD), ("S3B", True), ("S3B_THRESHOLD", 2600)):
        monkeypatch.setattr(EA, k, v)
    monkeypatch.setattr(A1, "ACTIVE", True); monkeypatch.setattr(A1, "SLICE", 512); monkeypatch.setattr(A1, "SMALL_SLICE", 128)
    A1.set_map(T._map())
    try:
        kk, vv, qq = T._data(6200, seed=13)

        def read(cap):
            monkeypatch.setattr(A1, "MAX_TILES", cap)
            c = T._cache(kk, vv)
            c.pli_request_start, c.pli_prefill_start = 6100, 5000
            return T._run(c, kk, vv, qq, 6000, [40], base=6000)

        ref, big, small = read(0), read(10_000), read(2)
        assert (ref == big).all()
        assert not (ref == small).all()
    finally:
        A1.set_map(None)
