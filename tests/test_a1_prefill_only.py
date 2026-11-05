"""PLI (05/11/2026): PLI_A1_PREFILL_ONLY=1, A1 only for the prompt prefill."""

from __future__ import annotations

import pytest

pytest.importorskip("mlx.core")

from tensorfold.kernels.qwen.dense.v1 import exact_attention as EA  # noqa: E402
from tensorfold.pli import a1_lsa as A1  # noqa: E402

import importlib.util, os, sys  # noqa: E402,E401

_spec = importlib.util.spec_from_file_location("t_a1po", os.path.join(os.path.dirname(__file__), "test_a1_lsa.py"))
T = importlib.util.module_from_spec(_spec)
sys.modules["t_a1po"] = T
_spec.loader.exec_module(T)
a1 = T.a1


def _writing(kk, vv, qq, only, monkeypatch):
    monkeypatch.setattr(A1, "PREFILL_ONLY", only)
    c = T._cache(kk, vv)
    c.pli_request_start = 4000
    return T._run(c, kk, vv, qq, 4000, [8] * 3)


def test_decode_by_the_anchored_path(a1, monkeypatch):
    kk, vv, qq = T._data(4200, seed=11)
    before = EA._REASONS.get("a1", 0)
    only = _writing(kk, vv, qq, True, monkeypatch)
    assert EA._REASONS.get("a1", 0) == before
    monkeypatch.setattr(A1, "ACTIVE", False)
    off = _writing(kk, vv, qq, False, monkeypatch)
    assert (only == off).all()
    monkeypatch.setattr(A1, "ACTIVE", True)
    complete = _writing(kk, vv, qq, False, monkeypatch)
    assert not (complete == off).all()


def test_prefill_like_full_a1_and_index_extended(a1, monkeypatch):
    """Prompt chunk, then decode outside A1, then a new prompt chunk: same bits as A1 everywhere."""

    monkeypatch.setattr(EA, "S3B", True)
    monkeypatch.setattr(EA, "S3B_THRESHOLD", 2600)
    kk, vv, qq = T._data(3200, seed=12)

    def unroll(only):
        monkeypatch.setattr(A1, "PREFILL_ONLY", only)
        c = T._cache(kk, vv)
        c.pli_request_start, c.pli_prefill_start = 2740, 2000
        read1 = T._run(c, kk, vv, qq, 2700, [40], base=2700)
        T._run(c, kk, vv, qq, 2740, [8] * 2, base=2700)
        c.pli_request_start, c.pli_prefill_start = 3000, 2756
        before = EA._REASONS.get("a1_s3b", 0)
        read2 = T._run(c, kk, vv, qq, 2756, [40], base=2700)
        assert EA._REASONS.get("a1_s3b", 0) > before
        return read1, read2

    s1, s2 = unroll(True)
    c1, c2 = unroll(False)
    assert (s1 == c1).all() and (s2 == c2).all()
