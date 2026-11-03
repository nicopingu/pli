#!/usr/bin/env python3
"""Measure prompt reading and writing speed of a running server, with fixed public prompts.

Works with any server that speaks the OpenAI chat completions API with streaming (PLI, llama.cpp, mlx-lm,
Ollama, oMLX, ...), so engines can be compared on the same machine with the same requests.

    python tools/pli_bench.py --url http://127.0.0.1:8080/v1 --model qwen27
    python tools/pli_bench.py --only single --sizes code,1k,8k,32k,64k,96k --out bench.json
    python tools/pli_bench.py --only agent

By default the requests come from the suite tools/bench_data/suite-v1.json and its corpus (PLI's own source code,
checked by its SHA-256), so every run, on every machine, sends the same texts:
  - single: one message each, a short code request and prompts of about 1k, 8k, 32k, 64k and 96k tokens
    (sizes counted with the Qwen3.8 tokenizer) followed by one question;
  - agent: one session that grows from about 6k to 50k tokens (an agent reading files of a repository, with fixed
    replies); at 8 points a question is asked and its answer is measured, then the session goes on without it, so
    the prompt prefix is shared from one request to the next as in real agent work.
Only the first line of each conversation changes ("Bench run <salt>, ..."), so that no prompt cache from an earlier
run serves it; pass --salt to fix it. Decoding is greedy (temperature 0). The bench sends no thinking field:
measure with thinking off, on the server (pli serve ... --no-thinking; llama.cpp: --reasoning off when the server
starts) or with --extra ('{"chat_template_kwargs": {"enable_thinking": false}}' for PLI and mlx-lm,
'{"reasoning_effort": "none"}' for Kapitsa).

For each request:
  - reading (tokens/s)  = prompt tokens / time to the first written token (agent: the prefix may be cached)
  - writing (tokens/s)  = (written tokens - 1) / (end - first written token)
The agent summary is all written tokens over all writing time. Writing speed depends on the drafts the model
accepts, so it depends on the text: compare runs of the same suite, model and settings.
Memory is sampled every second (macOS): wired memory, available memory, swap, and the server's resident memory
when --pid is given. Only standard Python is used; nothing else needs to be installed.
--free keeps the older mode: free sizes (--contexts), filler from --text-file or PLI's installed source.

Run it on a machine doing nothing else: another program using the GPU or memory changes the numbers.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

QUESTION = "\n\nSummarize what this code does in five numbered points."
CODE_TASK = ("Write a Python module with an LRU cache (class LRUCache with get, put, a capacity and hit and miss "
             "statistics), then its unit tests with unittest. Give only the code.")


# ---------------------------------------------------------------- text source

def source_text(path: str | None) -> str:
    """Filler text for long prompts: Python source, from --text-file or from the installed PLI package."""
    if path:
        return Path(path).read_text(errors="replace")
    spec = importlib.util.find_spec("tensorfold")          # located without importing it (no MLX load)
    roots = list(spec.submodule_search_locations) if spec and spec.submodule_search_locations else []
    if not roots:
        roots = [str(Path(__file__).resolve().parents[1] / "src" / "tensorfold")]
    text = []
    for root in roots:
        for d, _, files in sorted(os.walk(root)):
            for f in sorted(files):
                if f.endswith(".py"):
                    text.append(Path(d, f).read_text(errors="replace"))
    out = "".join(text)
    if len(out) < 100_000:
        sys.exit("pli_bench: not enough source text found; pass --text-file with a large text file")
    return out


# ---------------------------------------------------------------- memory

def vm_sample() -> dict[str, float]:
    if platform.system() != "Darwin":
        return {}
    v = subprocess.run(["vm_stat"], capture_output=True, text=True).stdout
    page = int(re.search(r"page size of (\d+)", v).group(1))
    d = {m.group(1): int(m.group(2)) for m in re.finditer(r'^"?([^:"]+)"?:\s+(\d+)\.', v, re.M)}
    s = subprocess.run(["sysctl", "-n", "vm.swapusage"], capture_output=True, text=True).stdout
    sw = re.search(r"used = ([\d.,]+)M", s)
    free = ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable")
    return {"wired_gb": d.get("Pages wired down", 0) * page / 2**30,
            "available_gb": sum(d.get(k, 0) for k in free) * page / 2**30,
            "swap_gb": float(sw.group(1).replace(",", ".")) / 1024 if sw else 0.0}


def rss_tree_gb(pid: int) -> float:
    out = subprocess.run(["ps", "-A", "-o", "pid=,ppid=,rss="], capture_output=True, text=True).stdout
    children: dict[int, list[int]] = {}
    rss: dict[int, int] = {}
    for line in out.splitlines():
        c = line.split()
        if len(c) == 3:
            p, pp, r = map(int, c)
            children.setdefault(pp, []).append(p)
            rss[p] = r
    total, stack = 0, [pid]
    while stack:
        p = stack.pop()
        total += rss.get(p, 0)
        stack += children.get(p, [])
    return total / 2**20


class MemoryWatch:
    """Samples memory every second; keeps the peak over the whole run and over the current request."""

    def __init__(self, pid: int | None):
        self.pid, self.stop = pid, False
        self.total = self._empty()
        self.window: dict[str, float] | None = None
        threading.Thread(target=self._loop, daemon=True).start()

    @staticmethod
    def _empty() -> dict[str, float]:
        return {"wired_max_gb": 0.0, "available_min_gb": 1e9, "swap_max_gb": 0.0, "server_rss_max_gb": 0.0}

    def _loop(self) -> None:
        while not self.stop:
            try:
                m = vm_sample()
                if m:
                    m["rss"] = rss_tree_gb(self.pid) if self.pid else 0.0
                    for d in [self.total] + ([self.window] if self.window is not None else []):
                        d["wired_max_gb"] = max(d["wired_max_gb"], m["wired_gb"])
                        d["available_min_gb"] = min(d["available_min_gb"], m["available_gb"])
                        d["swap_max_gb"] = max(d["swap_max_gb"], m["swap_gb"])
                        d["server_rss_max_gb"] = max(d["server_rss_max_gb"], m["rss"])
            except Exception:  # noqa: BLE001  sampling must never stop the bench
                pass
            time.sleep(1.0)

    def open(self) -> None:
        self.window = self._empty()

    def close(self) -> dict[str, float]:
        w, self.window = self.window, None
        if not w or w["available_min_gb"] == 1e9:
            return {}
        return {k: round(v, 2) for k, v in w.items()}


# ---------------------------------------------------------------- requests

def send(a: argparse.Namespace, text: "str | list[dict]", max_tokens: int) -> dict:
    messages = text if isinstance(text, list) else [{"role": "user", "content": text}]
    body = {"model": a.model, "stream": True, "temperature": 0.0, "max_tokens": max_tokens,
            "stream_options": {"include_usage": True}, "messages": messages}
    if a.extra:
        body.update(json.loads(a.extra))
    headers = {"Content-Type": "application/json"}
    if a.api_key:
        headers["Authorization"] = "Bearer " + a.api_key
    rq = urllib.request.Request(a.url.rstrip("/") + "/chat/completions", json.dumps(body).encode(), headers)
    t0 = time.time()
    t1 = None
    chunks, usage, out = 0, {}, []
    try:
        with urllib.request.urlopen(rq, timeout=a.timeout) as f:
            for line in f:
                line = line.decode("utf-8", "replace").strip()
                if not line.startswith("data:") or line.endswith("[DONE]"):
                    continue
                j = json.loads(line[5:])
                if j.get("usage"):
                    usage = j["usage"]
                for c in j.get("choices", []):
                    d = c.get("delta") or {}
                    x = d.get("content") or d.get("reasoning_content") or d.get("reasoning")
                    if x:
                        if t1 is None:
                            t1 = time.time()
                        out.append(x)
                        chunks += 1
    except urllib.error.HTTPError as e:
        return {"error": "HTTP %s %s" % (e.code, e.read().decode("utf-8", "replace")[:300])}
    except Exception as e:  # noqa: BLE001
        return {"error": repr(e)[:300]}
    t2 = time.time()
    written = usage.get("completion_tokens") or chunks
    read = usage.get("prompt_tokens")
    s = "".join(out)
    return {"prompt_tokens": read, "written_tokens": written,
            "counts_estimated": not (usage.get("completion_tokens") and read),
            "cached_tokens": (usage.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "first_token_s": round((t1 or t2) - t0, 2), "total_s": round(t2 - t0, 2),
            "reading_tok_s": round(read / ((t1 or t2) - t0), 1) if read and t1 else None,
            "writing_tok_s": round((written - 1) / (t2 - t1), 1) if t1 and written > 1 and t2 > t1 else None,
            "sha": hashlib.sha256(s.encode()).hexdigest()[:12]}


def wait_ready(a: argparse.Namespace) -> None:
    deadline = time.time() + a.wait
    while True:
        try:
            rq = urllib.request.Request(a.url.rstrip("/") + "/models",
                                        headers={"Authorization": "Bearer " + a.api_key} if a.api_key else {})
            with urllib.request.urlopen(rq, timeout=10):
                return
        except Exception:  # noqa: BLE001
            if time.time() > deadline:
                sys.exit(f"pli_bench: no server answering at {a.url}")
            time.sleep(2)


# ---------------------------------------------------------------- suite (fixed public prompts)

def load_suite(path: str, corpus_path: str | None) -> tuple[dict, str]:
    p = Path(path)
    if not p.exists():
        sys.exit(f"pli_bench: suite not found: {p} (or use --free)")
    suite = json.loads(p.read_text())
    c = Path(corpus_path) if corpus_path else p.parent / suite["corpus"]
    corpus = c.read_text()
    sha = hashlib.sha256(corpus.encode()).hexdigest()
    if sha != suite["corpus_sha256"]:
        sys.exit(f"pli_bench: corpus {c} has SHA-256 {sha}, the suite expects {suite['corpus_sha256']}")
    return suite, corpus


def piece(corpus: str, r: dict, before: str, after: str) -> str:
    o, n = r["slice"]
    return r[before] + corpus[o:o + n] + r[after]


class Table:
    def __init__(self, a: argparse.Namespace, results: dict):
        self.a, self.results = a, results
        print(f"{'request':>10} {'prompt':>8} {'cached':>7} {'written':>8} {'reading':>9} {'writing':>9} "
              f"{'first s':>8} {'wired GB':>9}")

    def add(self, r: dict) -> None:
        self.results["requests"].append(r)
        if "error" in r:
            print(f"{r['name']:>10} error: {r['error']}", flush=True)
        else:
            print(f"{r['name']:>10} {r['prompt_tokens'] or '?':>8} {r['cached_tokens'] if r['cached_tokens'] is not None else '-':>7} "
                  f"{r['written_tokens']:>8} {r['reading_tok_s'] or '-':>9} {r['writing_tok_s'] or '-':>9} "
                  f"{r['first_token_s']:>8} {r['memory'].get('wired_max_gb', '-'):>9}", flush=True)
        Path(self.a.out).write_text(json.dumps(self.results, indent=1))


def run_suite(a: argparse.Namespace) -> None:
    suite, corpus = load_suite(a.suite, a.corpus)
    salt = a.salt or "%08x" % random.getrandbits(32)
    line = suite["run_line"]
    mem = MemoryWatch(a.pid)
    results = {"suite": suite["suite"], "corpus_sha256": suite["corpus_sha256"], "salt": salt, "url": a.url,
               "model": a.model, "machine": platform.machine(), "os": platform.platform(),
               "started": time.strftime("%Y-%m-%d %H:%M:%S"), "requests": []}
    wait_ready(a)
    tab = Table(a, results)
    wanted = set(a.sizes.split(",")) if a.sizes else None
    for rep in range(a.repeat):
        if a.only in ("all", "single"):
            for r in suite["single"]:
                if wanted is not None and r["name"] not in wanted:
                    continue
                text = r["text"] if "text" in r else piece(corpus, r, "before", "after")
                if hashlib.sha256(text.encode()).hexdigest() != r["sha256"]:
                    sys.exit(f"pli_bench: request {r['name']} does not match the suite")
                mem.open()
                out = send(a, line.format(salt=f"{salt}.{rep}", name=r["name"]) + text, r["max_tokens"])
                out.update({"name": r["name"], "kind": "single", "run": rep, "memory": mem.close()})
                tab.add(out)
        if a.only in ("all", "agent"):
            ag = suite["agent"]
            msgs = [{"role": "system", "content": line.format(salt=f"{salt}.{rep}", name="agent") + ag["system"]},
                    {"role": "user", "content": ag["task"]}]
            asks = {m["after_turn"]: m for m in ag["measure"]}
            for i, t in enumerate(ag["turns"], 1):
                msgs += [{"role": "assistant", "content": t["assistant"]},
                         {"role": "user", "content": piece(corpus, t, "user_before", "user_after")}]
                m = asks.get(i)
                if m is None:
                    continue
                mem.open()
                out = send(a, msgs + [{"role": "user", "content": m["ask"]}], m["max_tokens"])
                out.update({"name": f"agent-{i}", "kind": "agent", "run": rep, "memory": mem.close()})
                tab.add(out)
    mem.stop = True
    ag = [r for r in results["requests"] if r.get("kind") == "agent" and r.get("writing_tok_s")]
    if ag:
        w = sum(r["written_tokens"] - 1 for r in ag)
        s = sum((r["written_tokens"] - 1) / r["writing_tok_s"] for r in ag)
        results["agent_writing_tok_s"] = round(w / s, 1)
    results["memory_peak"] = ({k: round(v, 2) for k, v in mem.total.items()}
                              if mem.total["available_min_gb"] != 1e9 else {})
    Path(a.out).write_text(json.dumps(results, indent=1))
    print(f"\nsuite {suite['suite']}, salt {salt}; reading and writing in tokens/s (higher is better); results in {a.out}")
    if "agent_writing_tok_s" in results:
        print(f"agent session: {results['agent_writing_tok_s']} tokens/s written over {len(ag)} answers")
    if results["memory_peak"]:
        print("memory peak: " + ", ".join(f"{k} {v}" for k, v in results["memory_peak"].items()))


# ---------------------------------------------------------------- main

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--url", default="http://127.0.0.1:8080/v1", help="OpenAI-compatible base URL (default: %(default)s)")
    ap.add_argument("--model", default="default", help="model name sent in each request")
    ap.add_argument("--api-key", default=os.environ.get("PLI_API_KEY", ""), help="bearer token, if the server needs one")
    ap.add_argument("--suite", default=str(Path(__file__).resolve().parent / "bench_data" / "suite-v1.json"),
                    help="suite of fixed requests (default: %(default)s)")
    ap.add_argument("--corpus", default=None, help="corpus file of the suite (default: next to the suite)")
    ap.add_argument("--only", choices=("all", "single", "agent"), default="all", help="part of the suite to run")
    ap.add_argument("--sizes", default="", help="single requests to run, e.g. code,1k,8k (default: all; 64k and "
                    "96k take several minutes each)")
    ap.add_argument("--salt", default="", help="first-line tag of each conversation (default: random)")
    ap.add_argument("--free", action="store_true", help="older mode: free sizes (--contexts) and filler text")
    ap.add_argument("--contexts", default="1000,8000,32000", help="--free: prompt sizes in tokens, comma-separated")
    ap.add_argument("--max-tokens", type=int, default=200, help="--free: tokens written per context request")
    ap.add_argument("--no-code", action="store_true", help="--free: skip the short code-writing request")
    ap.add_argument("--code-tokens", type=int, default=400, help="tokens written by the code request")
    ap.add_argument("--repeat", type=int, default=1, help="runs of each request")
    ap.add_argument("--pid", type=int, default=None, help="server process id, to record its resident memory")
    ap.add_argument("--text-file", default=None, help="--free: filler text for long prompts (default: PLI's own source)")
    ap.add_argument("--extra", default="", help='extra JSON merged into each request, e.g. \'{"top_k": 1}\'')
    ap.add_argument("--timeout", type=float, default=1800, help="seconds allowed per request")
    ap.add_argument("--wait", type=float, default=600, help="seconds to wait for the server to answer")
    ap.add_argument("--out", default="pli-bench.json", help="JSON results file (default: %(default)s)")
    a = ap.parse_args()
    if not a.free:
        run_suite(a)
        return

    wait_ready(a)
    text = source_text(a.text_file)
    chars_per_token = 3.0                                     # corrected after each request from the real count
    mem = MemoryWatch(a.pid)
    plan = ([("code", 0)] if not a.no_code else []) + [(f"{int(c) // 1000}k", int(c)) for c in a.contexts.split(",") if c]
    results = {"url": a.url, "model": a.model, "machine": platform.machine(), "os": platform.platform(),
               "started": time.strftime("%Y-%m-%d %H:%M:%S"), "requests": []}
    offset = 0
    print(f"{'request':>8} {'prompt':>8} {'written':>8} {'reading':>9} {'writing':>9} {'first s':>8} {'wired GB':>9}")
    for rep in range(a.repeat):
        for name, ctx in plan:
            tag = "Request %s.%d, %08x." % (name, rep, random.getrandbits(32))
            if ctx <= 0:
                prompt, maxi = tag + " " + CODE_TASK, a.code_tokens
            else:
                n = int(ctx * chars_per_token)
                if n > len(text):
                    text = text * (n // len(text) + 1)
                d = offset % max(1, len(text) - n)
                offset += 7919 * 13
                prompt, maxi = tag + " Here is some Python code.\n\n" + text[d:d + n] + QUESTION, a.max_tokens
            mem.open()
            r = send(a, prompt, maxi)
            r.update({"name": name, "target_tokens": ctx, "run": rep, "memory": mem.close()})
            results["requests"].append(r)
            if ctx > 0 and r.get("prompt_tokens"):
                chars_per_token = len(prompt) / r["prompt_tokens"]
            if "error" in r:
                print(f"{name:>8} error: {r['error']}")
            else:
                print(f"{name:>8} {r['prompt_tokens'] or '?':>8} {r['written_tokens']:>8} "
                      f"{r['reading_tok_s'] or '-':>9} {r['writing_tok_s'] or '-':>9} {r['first_token_s']:>8} "
                      f"{r['memory'].get('wired_max_gb', '-'):>9}", flush=True)
            Path(a.out).write_text(json.dumps(results, indent=1))
    mem.stop = True
    results["memory_peak"] = {k: round(v, 2) for k, v in mem.total.items()} if mem.total["available_min_gb"] != 1e9 else {}
    Path(a.out).write_text(json.dumps(results, indent=1))
    print(f"\nreading and writing in tokens/s (higher is better); results in {a.out}")
    if results["memory_peak"]:
        print("memory peak: " + ", ".join(f"{k} {v}" for k, v in results["memory_peak"].items()))


if __name__ == "__main__":
    main()
