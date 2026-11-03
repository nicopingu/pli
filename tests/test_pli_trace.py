"""PLI trace: off, nothing is written; forced (PLI_TRACE) or on demand (PLI_TRACE_DIR + switch)."""

import importlib
import json
import sys
import time
from types import SimpleNamespace


def _reload(monkeypatch, force: str = "", folder: str = "", events_path: str = ""):
    monkeypatch.setenv("PLI_TRACE", force)
    monkeypatch.setenv("PLI_TRACE_DIR", folder)
    monkeypatch.setenv("PLI_EVENTS", events_path)
    from tensorfold.pli import events, trace

    trace = importlib.reload(trace)
    events = importlib.reload(events)
    return trace, events


def _lines(path):
    return [json.loads(x) for x in path.read_text().splitlines()] if path.exists() else []


def _wait(cond, timeout=4.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end and not cond():
        time.sleep(0.05)
    return cond()


def test_off_nothing_written(monkeypatch, tmp_path):
    trace, events = _reload(monkeypatch)
    assert not trace.active()
    trace.pt("round", "req-1", ms=1.0)
    events.ev("arrival", SimpleNamespace(job_id="req-1"), prompt=3)
    assert list(tmp_path.iterdir()) == []


def test_forced_lines_and_log(monkeypatch, tmp_path):
    path = tmp_path / "trace.jsonl"
    trace, events = _reload(monkeypatch, force=str(path))
    assert trace.active()
    trace.pt("round", "req-1", ms=12.3456, rows=8, absent=None)
    events.ev("arrival", SimpleNamespace(job_id="req-1"), prompt=3)
    trace.flush()
    lines = _lines(path)
    rnd = next(d for d in lines if d["e"] == "round")
    assert rnd["req"] == "req-1" and rnd["ms"] == 12.346 and rnd["rows"] == 8 and "absent" not in rnd
    ev = next(d for d in lines if d["e"] == "ev.arrival")
    assert ev["a"] == "arrival" and ev["prompt"] == 3
    assert sorted(p.name for p in tmp_path.iterdir()) == ["trace.jsonl"]    # event log off


def test_on_demand(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "argv", ["tensorfold", "serve", "/models/X", "--name", "trial"])
    trace, _ = _reload(monkeypatch, folder=str(tmp_path))
    out = tmp_path / "trace-trial.jsonl"
    assert not trace.active()                       # ready, off
    trace.pt("round", "before", ms=1.0)
    trace.main(["on", "--dir", str(tmp_path)])
    assert _wait(trace.active)
    trace.pt("round", "during", ms=2.0)
    trace.main(["off", "--dir", str(tmp_path)])
    assert _wait(lambda: not trace.active())
    trace.pt("round", "after", ms=3.0)
    trace.main(["on", "--dir", str(tmp_path), "--duration", "3s"])
    assert _wait(trace.active)
    assert _wait(lambda: not trace.active(), timeout=7.0)      # deadline: switches itself off
    assert not (tmp_path / "trace.on").exists()
    trace.flush()
    reqs = [d.get("req") for d in _lines(out) if d["e"] == "round"]
    assert reqs == ["during"]
    states = [d["state"] for d in _lines(out) if d["e"] == "trace"]
    assert states == ["on", "off", "on", "off"]


def test_old_switch_and_options(monkeypatch, tmp_path):
    trace, _ = _reload(monkeypatch, folder=str(tmp_path))
    (tmp_path / "trace.on").write_text(f"depuis=1 jusqu={time.time() + 3600:.0f}\n")   # written before 3.7.0
    assert trace._deadline(str(tmp_path / "trace.on")) is not None
    assert trace.main(["etat", "--dossier", str(tmp_path)]) == 0
    trace.main(["on", "--dossier", str(tmp_path), "--duree", "1h"])
    assert "until=" in (tmp_path / "trace.on").read_text()


def test_analysis_on_trace(monkeypatch, tmp_path):
    path = tmp_path / "trace.jsonl"
    trace, _ = _reload(monkeypatch, force=str(path))
    for i in range(3):
        trace.pt("chunk", "req-1", tokens=512, ms=2000.0, left=0, streams=0, prefills=1)
        trace.pt("round", "req-1", ms=100.0, rows=8, kept=3, tokens=3, ctx=1000 + i, kind="head", proposed=7,
                 build_ms=9.0, gpu_wait_ms=80.0, after_ms=11.0, gap_ms=2.0)
    trace.pt("request", "req-1", prompt=1536, cache=0, output=9, prefill_ms=6000.0, decode_ms=306.0, total_ms=6400.0)
    trace.flush()
    from tensorfold.pli import trace_analysis

    report = trace_analysis.analyze(trace_analysis.read([str(path)], None, None))
    assert "## 6. Where to gain" in report and "prefill: chunks" in report


def test_analysis_reads_a_trace_written_before_3_7_0(tmp_path):
    from tensorfold.pli import trace_analysis

    old = tmp_path / "old.jsonl"
    rows = [{"t": 1.0 + i, "e": "morceau", "req": "r", "jetons": 512, "ms": 2000.0, "flux": 0, "lectures": 1} for i in range(3)]
    rows += [{"t": 10.0, "e": "tour", "req": "r", "ms": 100.0, "lignes": 8, "gardees": 3, "jetons": 3, "ctx": 1000,
              "sorte": "head", "proposes": 7, "construit_ms": 9.0, "attente_gpu_ms": 80.0, "apres_ms": 11.0},
             {"t": 11.0, "e": "demande", "req": "r", "prompt": 1536, "cache": 0, "sortie": 9, "lecture_ms": 6000.0,
              "ecriture_ms": 306.0, "total_ms": 6400.0},
             {"t": 12.0, "e": "ev.relecture_disque", "a": "relecture_disque", "jetons": 900, "duree": 0.5}]
    old.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
    new = tmp_path / "new.jsonl"
    names = {"morceau": "chunk", "tour": "round", "demande": "request", "ev.relecture_disque": "ev.disk_read"}
    new.write_text("\n".join(json.dumps({trace_analysis.OLD_FIELDS.get(k, k): (names.get(v, v) if k == "e" else
                                         "disk_read" if k == "a" else v) for k, v in r.items()}) for r in rows) + "\n")
    report_old = trace_analysis.analyze(trace_analysis.read([str(old)], None, None))
    report_new = trace_analysis.analyze(trace_analysis.read([str(new)], None, None))
    assert report_old == report_new and "disk_read" in report_old and "## 3. Decode" in report_old


def teardown_module(module):
    import os

    os.environ.pop("PLI_TRACE", None)
    os.environ.pop("PLI_TRACE_DIR", None)
    from tensorfold.pli import events, trace

    importlib.reload(trace)
    importlib.reload(events)
