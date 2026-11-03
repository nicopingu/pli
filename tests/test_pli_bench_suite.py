"""The public bench suite (src/tensorfold/bench/data/suite-v1.json): its corpus and every request are the frozen texts."""

import hashlib
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "src" / "tensorfold" / "bench" / "data" / "suite-v1.json"


def _bench():
    spec = importlib.util.spec_from_file_location("pli_bench", ROOT / "src" / "tensorfold" / "bench" / "pli_bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_the_corpus_matches_the_suite():
    suite, corpus = _bench().load_suite(str(SUITE), None)
    assert hashlib.sha256(corpus.encode()).hexdigest() == suite["corpus_sha256"]


def test_every_single_request_is_the_frozen_text():
    bench = _bench()
    suite, corpus = bench.load_suite(str(SUITE), None)
    names = [r["name"] for r in suite["single"]]
    assert names == ["code", "1k", "8k", "32k", "64k", "96k"]
    for r in suite["single"]:
        text = r["text"] if "text" in r else bench.piece(corpus, r, "before", "after")
        assert hashlib.sha256(text.encode()).hexdigest() == r["sha256"], r["name"]


def test_the_agent_session_reads_whole_files_and_measures_eight_times():
    bench = _bench()
    suite, corpus = bench.load_suite(str(SUITE), None)
    ag = suite["agent"]
    assert len(ag["measure"]) == 8
    turns = {m["after_turn"] for m in ag["measure"]}
    assert max(turns) <= len(ag["turns"])
    for t in ag["turns"]:
        path = t["assistant"].split("`")[1]
        o, n = t["slice"]
        head = corpus.rfind("# ===== file: ", 0, o)
        assert corpus[head:o] == f"# ===== file: {path}\n"
        assert corpus[o + n] == "\n"
