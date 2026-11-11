"""FALCON B (11/11/2026): tokens of each request kept on disk, purge after N days."""

import importlib
import os
import time

import numpy as np


def _module(tmp_path, monkeypatch, days="7"):
    monkeypatch.setenv("PLI_FALCON_TOKENS", str(tmp_path / "jetons"))
    monkeypatch.setenv("PLI_FALCON_TOKENS_DAYS", days)
    from tensorfold.pli.falcon import tokens
    return importlib.reload(tokens)


def test_off_by_default(monkeypatch):
    monkeypatch.delenv("PLI_FALCON_TOKENS", raising=False)
    from tensorfold.pli.falcon import tokens
    t = importlib.reload(tokens)
    assert not t.active()
    t.record([1, 2], [3])
    assert t.count["records"] == 0


def test_write_read(tmp_path, monkeypatch):
    t = _module(tmp_path, monkeypatch)
    name = t.write([5, 6, 7], [8, 9], {"conv": "abc", "vision": False})
    p, o, m = t.read(name)
    assert p.dtype == np.int32 and p.tolist() == [5, 6, 7] and o.tolist() == [8, 9]
    assert m["conv"] == "abc" and m["vision"] is False
    assert oct(os.stat(name.parent).st_mode & 0o777) == "0o700"
    assert oct(os.stat(name).st_mode & 0o777) == "0o600"
    assert not list(name.parent.glob("*.tmp.npz"))


def test_record_thread(tmp_path, monkeypatch):
    t = _module(tmp_path, monkeypatch)
    t.record(list(range(1000)), [1, 2, 3], conv="c", request="r", cached=10, reason="stop", vision=False)
    for _ in range(50):
        if t.count["files"]:
            break
        time.sleep(0.05)
    files = list((tmp_path / "jetons").glob("jetons-*.npz"))
    assert len(files) == 1
    p, o, m = t.read(files[0])
    assert len(p) == 1000 and o.tolist() == [1, 2, 3] and m["request"] == "r" and "t" in m


def test_purge(tmp_path, monkeypatch):
    t = _module(tmp_path, monkeypatch, days="2")
    old = t.write([1], [2], {})
    new = t.write([3], [4], {})
    past = time.time() - 3 * 86400
    os.utime(old, (past, past))
    assert t.purge(old.parent, t.DAYS) == 1
    assert not old.exists() and new.exists()
