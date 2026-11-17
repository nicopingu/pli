"""PLI 5.6.3: startup measurements kept on disk (pli.additions.startup_cache)."""

import json
from dataclasses import asdict

import pytest

from pli.server import startup_cache as sc
from pli.engine.memory import StreamMemory


@pytest.fixture
def cache(tmp_path, monkeypatch):
    model = tmp_path / "model"
    model.mkdir()
    (model / "model.safetensors").write_bytes(b"x" * 10)
    (model / "config.json").write_text("{}")
    monkeypatch.setattr(sc, "ACTIVE", True)
    monkeypatch.setattr(sc, "REMEASURE", False)
    monkeypatch.setattr(sc, "DIRECTORY", tmp_path / "cache")
    monkeypatch.setattr(sc, "_key", None)
    monkeypatch.setattr(sc, "_values", {})
    monkeypatch.setattr(sc, "_counts", {"read": 0, "measured": 0})
    return model


def test_off_measures_every_time(monkeypatch):
    monkeypatch.setattr(sc, "ACTIVE", False)
    monkeypatch.setattr(sc, "_key", None)
    assert sc.configure("/nowhere") is None
    calls = []
    assert sc.remember("x", None, lambda: calls.append(1) or 7) == 7
    assert sc.remember("x", None, lambda: calls.append(1) or 7) == 7
    assert len(calls) == 2 and sc.summary() is None


def test_measured_once_then_read_at_the_next_start(cache):
    key = sc.configure(cache, argv=["m", "--port", "8000"])
    calls = []
    stream = StreamMemory(1, 2, 3, 4, 0.5, 1.5, 2.5, 6, 512)
    got = sc.remember("memory.measure", {"chunk": 512}, lambda: calls.append(1) or stream, dump=asdict,
                      load=lambda v: StreamMemory(**v))
    assert got == stream and len(calls) == 1
    assert json.loads((sc.DIRECTORY / f"{key}.json").read_text())["values"]
    # next start: same key, read from the file, not measured
    assert sc.configure(cache, argv=["m", "--port", "9000"]) == key     # the port is not in the key
    got = sc.remember("memory.measure", {"chunk": 512}, lambda: calls.append(1) or None, dump=asdict,
                      load=lambda v: StreamMemory(**v))
    assert got == stream and len(calls) == 1
    assert "1 measurement(s) read" in sc.summary()


def test_other_parameters_are_measured_apart(cache):
    sc.configure(cache, argv=[])
    assert sc.remember("w", {"widest": 8}, lambda: 1) == 1
    assert sc.remember("w", {"widest": 16}, lambda: 2) == 2
    assert sc.remember("w", {"widest": 8}, lambda: 3) == 1


def test_changed_weights_settings_or_args_change_the_key(cache, monkeypatch):
    first = sc.configure(cache, argv=["m"])
    assert sc.configure(cache, argv=["m", "--ctx", "4096"]) != first
    monkeypatch.setenv("PLI_SOMETHING", "1")
    assert sc.configure(cache, argv=["m"]) != first
    monkeypatch.delenv("PLI_SOMETHING")
    monkeypatch.setenv("PLI_STARTUP_REMEASURE", "1")          # the cache's own switches are not in the key
    assert sc.configure(cache, argv=["m"]) == first
    (cache / "model.safetensors").write_bytes(b"y" * 11)
    assert sc.configure(cache, argv=["m"]) != first


def test_remeasure_ignores_the_file_and_rewrites_it(cache, monkeypatch):
    sc.configure(cache, argv=[])
    sc.remember("w", None, lambda: 1)
    monkeypatch.setattr(sc, "REMEASURE", True)
    sc.configure(cache, argv=[])
    assert sc.remember("w", None, lambda: 2) == 2
    monkeypatch.setattr(sc, "REMEASURE", False)
    sc.configure(cache, argv=[])
    assert sc.remember("w", None, lambda: 3) == 2


def test_an_unreadable_entry_is_measured_again(cache):
    sc.configure(cache, argv=[])
    sc._values["w"] = "garbage"
    assert sc.remember("w", None, lambda: (4, {1: 2.0}), load=lambda v: (int(v[0]), dict(v[1]))) == (4, {1: 2.0})


def test_the_option_turns_it_on_and_is_not_in_the_key(cache, monkeypatch):
    monkeypatch.setattr(sc, "ACTIVE", False)
    first = sc.configure(cache, argv=["m", "--startup-cache", "on"], mode="on")
    assert first is not None
    assert sc.configure(cache, argv=["m", "--startup-cache", "remeasure"], mode="remeasure") == first
    assert sc.REMEASURE


def test_outside_the_checkpoint_code_key():
    from pli.additions import code_key
    assert not any(p.name == "startup_cache.py" for p in code_key.files())


def test_journals_and_generation_settings_stay_out_of_the_key(cache, monkeypatch, tmp_path):
    journal = tmp_path / "events.jsonl"
    journal.write_text("a")
    monkeypatch.setenv("PLI_EVENTS", str(journal))
    monkeypatch.setenv("PLI_GEN_CONFIG", "/x.json")
    first = sc.configure(cache, argv=["m"])
    journal.write_text("ab")                                   # a journal grows between two starts
    monkeypatch.setenv("PLI_GEN_CONFIG", "/y.json")
    assert sc.configure(cache, argv=["m"]) == first
    weights = tmp_path / "head.safetensors"
    weights.write_bytes(b"x")
    monkeypatch.setenv("PLI_MTP27", str(weights))              # an input file named by a setting does count
    second = sc.configure(cache, argv=["m"])
    assert second != first
    weights.write_bytes(b"xy")
    assert sc.configure(cache, argv=["m"]) != second
