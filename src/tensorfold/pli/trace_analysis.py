"""Read a PLI trace (pli/trace.py) and tell where the time goes, and what each gain would bring.

    python -m tensorfold.pli.trace_analysis trace.jsonl [--md out.md] [--since HH:MM] [--until HH:MM]

Outputs (Markdown, tables):
  1. requests: end-to-end split of the time (template, queue, prefill, decode, after the engine, output)
  2. prompt prefill: chunks, throughput by size, chunks prefilled while others decode
  3. decode: lone and shared rounds by context range, split build / GPU wait / after,
     time outside rounds (gap between two rounds), tokens per round
  4. draft: build, GPU wait, search, nodes, share kept
  5. actions of the event log (disk reads, saves, waits, jumps ahead)
  6. ranking: each step, its total time, its share of the lived time, the gain if it drops by 10 % or disappears

Reading rule: a time is [measured] in the engine; "GPU wait" is the time spent in the read that forces the
computation (not the GPU's own time); the gains of section 6 are [computed] (Amdahl's law, without overlap:
a step that overlaps another gains less than said).
Traces written before 3.7.0 (French step and field names) are read too: see OLD_STEPS, OLD_EVENTS, OLD_FIELDS.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from typing import Any, Iterable

RANGES = [(0, 8_000, "< 8K"), (8_000, 32_000, "8K-32K"), (32_000, 64_000, "32K-64K"),
          (64_000, 128_000, "64K-128K"), (128_000, 10**9, ">= 128K")]

# Names written by PLI before 3.7.0 -> names written since.
OLD_STEPS = {"demande": "request", "ouverture": "open", "reprise_prefixe": "prefix_resume", "morceau": "chunk",
             "sous_morceau": "sub_chunk", "garde_points": "checkpoints", "tour": "round", "tour_p": "round_p",
             "brouillon": "draft"}
OLD_EVENTS = {"annulee": "cancelled", "attente": "waiting", "passe_devant": "jumps_ahead",
              "reste_sur_disque": "stays_on_disk", "relecture_disque": "disk_read",
              "sauvegarde_bloc_systeme": "system_block_saved", "lecture_debut": "prefill_start",
              "lecture_fin": "prefill_end", "point_de_reprise": "checkpoint", "deversement": "spill",
              "arrivee": "arrival", "reprise": "resume", "fin": "done"}
OLD_FIELDS = {"reste": "left", "lectures": "prefills", "ecritures": "decodes", "flux_ms": "stream_ms", "repris": "reused",
              "points": "checkpoints", "sans_copie": "no_copy", "jetons": "tokens", "large": "wide", "flux": "streams",
              "arret": "abort", "sortie": "output", "fond": "background", "gabarit_ms": "template_ms",
              "avant_file_ms": "before_queue_ms", "file_ms": "queue_ms", "lecture_ms": "prefill_ms",
              "premier_jeton_ms": "first_token_ms", "ecriture_ms": "decode_ms", "apres_moteur_ms": "after_engine_ms",
              "flux_morceaux": "stream_chunks", "sortie_ms": "output_ms", "raison": "reason", "chaine_ms": "chain_ms",
              "noeuds": "nodes", "construit_ms": "build_ms", "attente_gpu_ms": "gpu_wait_ms", "recherche_ms": "search_ms",
              "copie": "copy", "bloc": "block", "lignes": "rows", "largeur": "width", "lignes_par": "rows_per",
              "gardees_par": "kept_per", "ecart_ms": "gap_ms", "gardees": "kept", "sorte": "kind", "proposes": "proposed",
              "apres_ms": "after_ms", "etat": "state", "jusqu": "until", "devant": "ahead_of", "genre": "kind",
              "duree": "duration", "gio": "gib", "histoire": "history", "deja": "already", "attente": "wait",
              "lecture": "prefill", "couche": "layer", "n_couche": "n_layer", "norme": "norm", "n_norme": "n_norm",
              "fin": "end", "n_fin": "n_end"}


def normalize(d: dict[str, Any]) -> dict[str, Any]:
    """A trace line under the names written since 3.7.0 (a line written before keeps its values)."""

    e = d.get("e")
    if isinstance(e, str):
        if e.startswith("ev."):
            e = "ev." + OLD_EVENTS.get(e[3:], e[3:])
        d["e"] = OLD_STEPS.get(e, e)
    if isinstance(d.get("a"), str):
        d["a"] = OLD_EVENTS.get(d["a"], d["a"])
    out = {}
    for k, v in d.items():
        new = OLD_FIELDS.get(k, k)
        out[new if new not in d else k] = v
    return out


def context_range(n: int | None) -> str:
    for a, b, name in RANGES:
        if n is not None and a <= n < b:
            return name
    return "?"


def read(paths: Iterable[str], since: float | None, until: float | None) -> list[dict[str, Any]]:
    lines = []
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for raw in f:
                try:
                    d = normalize(json.loads(raw))
                except ValueError:
                    continue
                t = d.get("t", 0)
                if (since and t < since) or (until and t > until):
                    continue
                lines.append(d)
    lines.sort(key=lambda d: d.get("t", 0))
    return lines


def s(values: list[float]) -> float:
    return float(sum(v for v in values if v is not None))


def med(values: list[float]) -> float | None:
    v = [x for x in values if x is not None]
    return statistics.median(v) if v else None


def p90(values: list[float]) -> float | None:
    v = sorted(x for x in values if x is not None)
    return v[min(len(v) - 1, int(0.9 * len(v)))] if v else None


def f(x: float | None, d: int = 1) -> str:
    if x is None:
        return "-"
    return f"{x:,.{d}f}"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "---|" * len(headers)]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def analyze(lines: list[dict[str, Any]]) -> str:
    by = defaultdict(list)
    for d in lines:
        by[d.get("e", "?")].append(d)
    requests, chunks = by["request"], by["chunk"]
    rounds, rounds_p, drafts = by["round"], by["round_p"], by["draft"]
    # outside rounds: gap between the end of a round and the start of the same stream's next round (the engine's gap
    # also counts moments without any request); past 5 s, the stream waited for something else (long prefill): counted apart
    prev_end: dict[Any, float] = {}
    for t in rounds:
        r = t.get("req")
        e = (t["t"] - prev_end[r]) * 1e3 if r in prev_end else None
        t["gap_ms"] = e if e is not None and 0 <= e <= 5000 else None
        t["long_gap_ms"] = e if e is not None and e > 5000 else None
        prev_end[r] = t["t"] + (t.get("ms") or 0) / 1e3
    for t in rounds_p:
        t["gap_ms"] = None
    out: list[str] = []
    if lines:
        start, end = lines[0]["t"], lines[-1]["t"]
        out.append(f"# PLI trace: {time.strftime('%d/%m/%Y %H:%M', time.localtime(start))} to "
                   f"{time.strftime('%H:%M', time.localtime(end))} ({f((end - start) / 60, 0)} min)\n")
        out.append(f"{len(requests)} requests, {len(chunks)} prefill chunks, {len(rounds)} lone rounds, "
                   f"{len(rounds_p)} shared rounds, {len(drafts)} drafts. All times in ms.\n")

    # 1. requests --------------------------------------------------------------------------------------
    steps = [("template_ms", "template and tokens"), ("before_queue_ms", "before the queue"), ("queue_ms", "waiting in queue"),
             ("prefill_ms", "prompt prefill"), ("decode_ms", "decode"), ("after_engine_ms", "after the engine"),
             ("output_ms", "detokenize and reply")]
    if requests:
        out.append("## 1. Requests: where the end-to-end time goes\n")
        total = s([d.get("total_ms") for d in requests])
        rows = []
        for key, name in steps:
            v = [d.get(key) for d in requests]
            rows.append([name, f(s(v), 0), f(100 * s(v) / total if total else None), f(med(v)), f(p90(v))])
        rows.append(["**total**", f(total, 0), "100", f(med([d.get("total_ms") for d in requests])),
                     f(p90([d.get("total_ms") for d in requests]))])
        out.append(table(["Step", "Total", "Share %", "Median", "9 in 10 under"], rows) + "\n")
        rows = []
        for _, _, name in RANGES:
            g = [d for d in requests if context_range(d.get("prompt")) == name]
            if not g:
                continue
            new = [max(0, d.get("prompt", 0) - d.get("cache", 0)) for d in g]
            rows.append([name, len(g), f(100 * s([min(d.get("cache", 0), d.get("prompt", 0)) for d in g]) / max(1, s([d.get("prompt", 0) for d in g])), 0),
                         f(med(new), 0), f(med([d.get("queue_ms") for d in g])), f(med([d.get("prefill_ms") for d in g])),
                         f(med([d.get("first_token_ms") for d in g])), f(med([d.get("output") for d in g]), 0),
                         f(med([1e3 * d["output"] / d["decode_ms"] for d in g if d.get("decode_ms") and d.get("output")]))])
        out.append(table(["Context", "Requests", "From cache %", "Tokens prefilled (med.)", "Queue (med.)",
                          "Prefill (med.)", "First token (med.)", "Tokens written (med.)", "Decode tok/s (med.)"], rows) + "\n")

    # 2. prefill ---------------------------------------------------------------------------------------
    if chunks:
        out.append("## 2. Prompt prefill: chunks\n")
        rows = []
        for name, g in (("alone (no stream decoding)", [m for m in chunks if not m.get("streams")]),
                        ("during a decode", [m for m in chunks if m.get("streams")]),
                        ("several prefills open", [m for m in chunks if (m.get("prefills") or 0) > 1])):
            if g:
                j, ms = s([m.get("tokens", 0) for m in g]), s([m.get("ms") for m in g])
                rows.append([name, len(g), f(j, 0), f(ms, 0), f(1e3 * j / ms if ms else None, 0), f(med([m.get("ms") for m in g]))])
        out.append(table(["Case", "Chunks", "Tokens", "Time", "Tokens/s", "ms per chunk (med.)"], rows) + "\n")
        # split of a prefill: open, prefix resume, chunks, keeping checkpoints, the rest (loop, rounds
        # of the other streams slipped between two chunks, waiting for room)
        opn = {d.get("req"): d for d in by["open"]}
        res = {d.get("req"): d for d in by["prefix_resume"]}
        kpt = {d.get("req"): d for d in by["checkpoints"]}
        chk = defaultdict(float)
        for m in chunks:
            chk[m.get("req")] += m.get("ms") or 0
        rows = []
        for name, g in (("fewer than 512 tokens prefilled", [d for d in requests if d.get("prompt", 0) - d.get("cache", 0) < 512]),
                        ("512 to 8,191", [d for d in requests if 512 <= d.get("prompt", 0) - d.get("cache", 0) < 8192]),
                        ("8,192 and more", [d for d in requests if d.get("prompt", 0) - d.get("cache", 0) >= 8192])):
            g = [d for d in g if d.get("prefill_ms")]
            if not g:
                continue
            o = [opn.get(d["req"], {}).get("ms") for d in g]
            r = [res.get(d["req"], {}).get("ms") for d in g]
            k = [chk.get(d["req"]) for d in g]
            p = [kpt.get(d["req"], {}).get("ms") for d in g]
            rest = [d["prefill_ms"] - sum(x or 0 for x in (a, b, c, e)) for d, a, b, c, e in zip(g, o, r, k, p)]
            rows.append([name, len(g), f(med([d["prefill_ms"] for d in g])), f(med(o)), f(med(r)), f(med(k)), f(med(p)), f(med(rest))])
        out.append("Split of a prefill (medians, ms). \"Rest\" = prefill time outside these steps: rounds of the "
                   "other streams between two chunks, waiting for room, loop.\n")
        out.append(table(["Tokens prefilled", "Requests", "Prefill", "Open", "Prefix resume", "Chunks",
                          "Keeping checkpoints", "Rest"], rows) + "\n")
        small = [m for m in chunks if 0 < (m.get("tokens") or 0) < 512]
        if small:
            out.append(f"Chunks of fewer than 512 tokens: {len(small)}, {f(s([m['ms'] for m in small]), 0)} ms, "
                       f"{f(1e3 * s([m['tokens'] for m in small]) / max(1e-9, s([m['ms'] for m in small])), 0)} tokens/s.\n")

    # 3. decode ----------------------------------------------------------------------------------------
    if rounds or rounds_p:
        out.append("## 3. Decode: rounds\n")
        rows = []
        for _, _, name in RANGES:
            g = [t for t in rounds if context_range(t.get("ctx")) == name]
            if not g:
                continue
            ms, tok, gap = s([t.get("ms") for t in g]), s([t.get("tokens", 0) for t in g]), s([t.get("gap_ms") for t in g])
            rows.append([name, len(g), f(med([t.get("ms") for t in g])), f(med([t.get("build_ms") for t in g])),
                         f(med([t.get("gpu_wait_ms") for t in g])), f(med([t.get("after_ms") for t in g])),
                         f(med([t.get("gap_ms") for t in g])), f(med([t.get("rows") for t in g])),
                         f(tok / len(g), 2), f(100 * gap / (ms + gap) if ms + gap else None),
                         f(1e3 * tok / (ms + gap) if ms + gap else None)])
        out.append("Rounds of a lone stream. \"Outside round\" = gap between the end of a round and the same stream's next "
                   "round, 5 s at most (loop, admissions, prefill chunks slipped between two rounds, Python). \"Build\" "
                   "also counts the GPU wait when MLX blocks the submission (too much work in flight): build + GPU "
                   "wait = round time held by the GPU, to split with the synchronized measurement.\n")
        out.append(table(["Context", "Rounds", "Round (med.)", "Build", "GPU wait", "After", "Outside round (med.)",
                          "Rows", "Tokens per round", "Outside round %", "Lived tokens/s"], rows) + "\n")
        kinds = defaultdict(list)
        for t in rounds:
            kinds[t.get("kind", "?")].append(t)
        out.append(table(["Kind of round", "Rounds", "ms (med.)", "Tokens per round"],
                         [[k, len(v), f(med([t.get("ms") for t in v])), f(s([t.get("tokens", 0) for t in v]) / len(v), 2)]
                          for k, v in sorted(kinds.items(), key=lambda kv: -len(kv[1]))]) + "\n")
        if rounds_p:
            rows = []
            for n in sorted({t.get("streams") for t in rounds_p}):
                g = [t for t in rounds_p if t.get("streams") == n]
                ms, tok = s([t.get("ms") for t in g]), s([t.get("tokens", 0) for t in g])
                rows.append([n, len(g), f(med([t.get("ms") for t in g])), f(med([t.get("rows") for t in g])),
                             f(tok / len(g), 2), f(1e3 * tok / ms if ms else None)])
            out.append(table(["Streams together", "Rounds", "ms (med.)", "Rows (med.)", "Tokens per round", "Total tokens/s"], rows) + "\n")

    # 4. draft -----------------------------------------------------------------------------------------
    trees = [b for b in drafts if "search_ms" in b]
    if trees:
        out.append("## 4. Draft (DFlash, lattice)\n")
        out.append(table(["Step", "Total", "Median", "9 in 10 under"],
                         [[name, f(s([b.get(k) for b in trees]), 0), f(med([b.get(k) for b in trees]), 2), f(p90([b.get(k) for b in trees]), 2)]
                          for k, name in (("build_ms", "build (CPU)"), ("gpu_wait_ms", "GPU wait"),
                                          ("search_ms", "tree search (CPU)"))]) + "\n")
        prop = s([t.get("proposed", 0) for t in rounds])
        kept = s([max(0, (t.get("kept") or 1) - 1) for t in rounds if t.get("kind") == "head"])
        out.append(f"Nodes proposed (med.): {f(med([b.get('nodes') for b in trees]), 0)}; "
                   f"drafts kept out of proposed (\"head\" rounds): {f(100 * kept / prop if prop else None)} %.\n")

    # 5. actions ---------------------------------------------------------------------------------------
    evs = [d for d in lines if str(d.get("e", "")).startswith("ev.")]
    if evs:
        out.append("## 5. Actions of the event log\n")
        grp = defaultdict(list)
        for d in evs:
            grp[d["e"][3:]].append(d)
        out.append(table(["Action", "Count", "Total duration (s)", "Tokens (med.)"],
                         [[a, len(v), f(s([x.get("duration") for x in v]), 1) if any(x.get("duration") for x in v) else "-",
                           f(med([x.get("tokens") for x in v]), 0)] for a, v in sorted(grp.items(), key=lambda kv: -len(kv[1]))]) + "\n")

    # 6. ranking ---------------------------------------------------------------------------------------
    out.append("## 6. Where to gain: each step, its share of the lived time and the possible gain [computed]\n")
    lived = s([d.get("total_ms") for d in requests]) or (s([t.get("ms") for t in rounds]) + s([t.get("gap_ms") for t in rounds]))
    items = []
    for key, name in steps:
        if key not in ("prefill_ms", "decode_ms"):
            items.append((name, s([d.get(key) for d in requests])))
    items.append(("prefill: chunks", s([m.get("ms") for m in chunks])))
    items.append(("prefill: open (disk, room, prefix lookup)", s([d.get("ms") for d in by["open"]])))
    items.append(("prefill: resume of the kept prefix", s([d.get("ms") for d in by["prefix_resume"]])))
    items.append(("prefill: keeping checkpoints", s([d.get("ms") for d in by["checkpoints"]])))
    items.append(("prefill: rest (outside the steps above)", max(0.0, s([d.get("prefill_ms") for d in requests])
                  - s([m.get("ms") for m in chunks]) - s([d.get("ms") for d in by["open"]])
                  - s([d.get("ms") for d in by["prefix_resume"]]) - s([d.get("ms") for d in by["checkpoints"]]))))
    items.append(("decode: lone round, build + GPU wait (verification)",
                  s([t.get("build_ms") for t in rounds]) + s([t.get("gpu_wait_ms") for t in rounds])))
    items.append(("decode: after the round (acceptance, commit, next draft)", s([t.get("after_ms") for t in rounds])))
    items.append(("decode: shared rounds", s([t.get("ms") for t in rounds_p])))
    items.append(("decode: outside round (5 s at most)", s([t.get("gap_ms") for t in rounds])))
    items.append(("decode: streams waiting more than 5 s between two rounds", s([t.get("long_gap_ms") for t in rounds])))
    items.append(("of which draft: build + search (CPU)", s([b.get("build_ms") for b in trees]) + s([b.get("search_ms") for b in trees])))
    items = [(n, v) for n, v in items if v]
    items.sort(key=lambda p: -p[1])
    out.append(f"Reference lived time: {f(lived / 1e3, 1)} s (sum of the requests, or of the rounds without requests). "
               "The \"of which\" lines are already counted in another line.\n")
    out.append(table(["Step", "Total (s)", "Share of lived % (= gain if removed)", "Gain if - 10 % (s)"],
                     [[n, f(v / 1e3, 1), f(100 * v / lived if lived else None), f(v / 1e4, 1)]
                      for n, v in items]) + "\n")
    return "\n".join(out)


def time_of_day(text: str | None, ref: float) -> float | None:
    if not text:
        return None
    h, m = (int(x) for x in text.split(":"))
    lt = time.localtime(ref)
    return time.mktime((lt.tm_year, lt.tm_mon, lt.tm_mday, h, m, 0, 0, 0, -1))


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("traces", nargs="+")
    p.add_argument("--md", help="write the report to this file")
    p.add_argument("--since", dest="since", help="first time of day kept (HH:MM)")
    p.add_argument("--depuis", dest="since", default=argparse.SUPPRESS, help=argparse.SUPPRESS)   # name before 3.7.0
    p.add_argument("--until", dest="until", help="last time of day kept (HH:MM)")
    p.add_argument("--jusqu", dest="until", default=argparse.SUPPRESS, help=argparse.SUPPRESS)   # name before 3.7.0
    a = p.parse_args(argv)
    lines = read(a.traces, None, None)
    if not lines:
        print("empty trace", file=sys.stderr)
        return 1
    ref = lines[0]["t"]
    lines = [d for d in lines if (not a.since or d["t"] >= time_of_day(a.since, ref))
             and (not a.until or d["t"] <= time_of_day(a.until, ref))]
    if not lines:
        print("nothing in this time range", file=sys.stderr)
        return 1
    report = analyze(lines)
    if a.md:
        with open(a.md, "w", encoding="utf-8") as fh:
            fh.write(report + "\n")
    print(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
