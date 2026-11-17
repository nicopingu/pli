"""PLI (16/11/2026): background task ``falcon``, FALCON B's adjustment of the DFlash2 drafter done by PLI itself.

Until 15/11/2026 the adjustment ran each night from another machine (cron ``falcon-dfl`` at 1 h 50 -> bench queue, service cut ->
~/falcon/dfl/nuit.py -> ~/dev/dflash2-ajuste/ajuste.py, which loaded its own copy of the 27B: 20 to 40 minutes without
service). Decision (15/11/2026): PLI does it in the gaps of service, without cutting it.

A cycle, once every PLI_TASKS_FALCON_EVERY_H hours (24), started only within PLI_TASKS_FALCON_HOURS (``1-7``: local hours,
start included, end excluded), then carried on in any quiet moment until it ends:
1. preparation (``preparer_falcon.py``, a CPU subprocess, as nuit.py): PLI's kept tokens -> training conversations (the
   last PLI_TASKS_FALCON_SINCE_H hours, 26) and control conversations (one in five, never trained on);
2. training of the drafter's input projection (fc + hidden_norm, as ajuste.py ``--portee projection``, lr 5e-5, AdamW, cosine
   schedule with warm-up, gradient clipped at 1), from the current drafter (``courant.txt``, z-lab at first), on at most
   PLI_TASKS_FALCON_SESSIONS conversations (60); the 27B's states come from the served model (falcon/states.py), a piece
   of PLI_TASKS_FALCON_CHUNK tokens (512) a step, then one batch of 8 anchors a step;
3. control: z-lab and the new drafter, both in 4 bits as PLI serves them, on PLI_TASKS_FALCON_CONTROL conversations (12):
   accepted length against the 27B's greedy choice, by domain;
4. decision as nuit.py: the new drafter becomes the current one if its gain on z-lab is at least the last kept gain minus
   0.5 point and no domain is below -1 %; then its folder goes to PLI_TASKS_DRAFTER_FILE, where the ``drafter`` task takes it
   up (served after the served one's watch, judged in service); one line in historique.jsonl.
Tri as it goes: the prepared conversations (text of the requests) are deleted at the end of the cycle, so are the cycles
whose drafter is neither the current one nor the served one (``servi.txt``).

Shadow (16/11/2026: the cycle in PLI must not come at the expense of the improvement):
PLI_TASKS_FALCON_SHADOW=1 runs the cycle beside the night job without taking its place: its own folder
(PLI_TASKS_FALCON_DIR, e.g. ~/falcon/dfl-pli, its own historique.jsonl), the night job's current drafter as its base
(PLI_TASKS_FALCON_BASE_FILE, e.g. ~/falcon/dfl/courant.txt), and nothing proposed to the ``drafter`` task. Its gains on
z-lab are compared with the night job's, night after night, before it replaces it.

Memory: about 6 GiB held while a cycle runs (the drafter in 16 bits and its optimizer state, a conversation's states);
a step waits while that does not fit (PLI_TASKS_MEMORY_FRACTION). A restart drops a cycle in progress (it is started again
at the next window). The cycle's folder is ~/falcon/dfl/nuits/<date>-pli (PLI_TASKS_FALCON_DIR).

The training and measurement code is ajuste.py's (16/11/2026 copy, functions of the same names), without its global
changes of the z-lab module (the served drafter uses that module).
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import stat
import random
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Iterator

from pli.tasks import Task, _env_float

HOME = Path.home()
FIN = 2047                       # the drafter's sliding window, positions of context before an anchor
BLOC = 8
GAIN_KEEP_TOLERANCE, DOMAIN_FLOOR = 0.005, -0.01


def _hours(text: str) -> tuple[int, int]:
    try:
        a, b = (int(x) for x in text.split("-"))
        return a % 24, b % 24
    except ValueError:
        return 1, 7


def in_window(hour: int, window: tuple[int, int]) -> bool:
    a, b = window
    return a <= hour < b if a <= b else (hour >= a or hour < b)


# -- ajuste.py, the pieces the training and the control use ----------------------------------------------------------
def lire_sessions(fichier: Path, n: int | None = None, graine: int = 0) -> list[dict]:
    rows = [json.loads(line) for line in open(fichier, encoding="utf-8")]
    if n:
        random.Random(graine).shuffle(rows)
        rows = rows[:n]
    return rows


def coupe_reponses(ecrits, jetons, im_end):
    out = []
    for a, b, dom in ecrits:
        b = min(b, len(jetons))
        for k in range(a, b):
            if jetons[k] == im_end:
                b = k + 1
                break
        if b - a >= 2:
            out.append((a, b, dom))
    return out


def preparer_sessions(rows, coupe, im_end):
    out = []
    for r in rows:
        c = r.get("entete", 0) + coupe
        j = r["jetons"][:c]
        e = [[a, min(b, c), d] for a, b, d in r["ecrits"] if a < c]
        if e:
            out.append(dict(r, jetons=j, ecrits=e))
    out.sort(key=lambda r: r["jetons"])
    return out


def empreintes(fichier: Path, coupe: int) -> set[str]:
    vus = set()
    for line in open(fichier, encoding="utf-8"):
        r = json.loads(line)
        c = r.get("entete", 0) + coupe
        for a, b, _ in r["ecrits"]:
            if a < c:
                vus.add(hashlib.md5(json.dumps(r["jetons"][a:min(b, c)]).encode()).hexdigest())
    return vus


def hors(reponses, jetons, vus):
    return [x for x in reponses
            if hashlib.md5(json.dumps(jetons[x[0]:x[1]]).encode()).hexdigest() not in vus] if vus else reponses


def nouvelles_reponses(ecrits, jetons, im_end, n_commun):
    return [x for x in coupe_reponses(ecrits, jetons, im_end) if x[1] > n_commun]


def ancrages(reponses, n_max, rng):
    pos = [(k, b, dom) for a, b, dom in reponses for k in range(max(a, 1), b - 1)]
    rng.shuffle(pos)
    return pos[:n_max]


def ancrages_suivis(reponses, n_lots, rng, bloc):
    lots = []
    for a, b, dom in reponses:
        a = max(a, 1)
        for k in range(a, b - 1, bloc):
            lots.append([(j, b, dom) for j in range(k, min(k + bloc, b - 1))])
    rng.shuffle(lots)
    return lots[:n_lots]


def lot(mx, mask_id, H, G, J, ancres, bloc):
    L = min(FIN, min(a for a, _, _ in ancres))
    ctx = mx.stack([H[a - L:a] for a, _, _ in ancres])
    entree = mx.stack([mx.concatenate([J[a:a + 1], mx.full((bloc - 1,), mask_id, mx.int32)]) for a, _, _ in ancres])
    n = J.shape[0]
    idx = mx.array([[min(a + 1 + k, n - 1) for k in range(bloc - 1)] for a, _, _ in ancres])
    masque = mx.array([[1.0 if a + 1 + k < b else 0.0 for k in range(bloc - 1)] for a, b, _ in ancres])
    lab = J[idx]
    glou = G[idx - 1]
    pred = mx.concatenate([J[mx.array([a for a, _, _ in ancres])][:, None], lab[:, :-1]], axis=1)
    return ctx, entree, lab, pred, masque, glou


def lot_suivi(mx, br, H, J, ancres, bloc):
    L = min(FIN, min(a for a, _, _ in ancres))
    lo, hi = min(a for a, _, _ in ancres) - L, max(a for a, _, _ in ancres)
    W = br.fc.weight.astype(mx.bfloat16)
    U = H[lo:hi] @ W.T
    U = mx.fast.rms_norm(U, br.hidden_norm.weight.astype(mx.bfloat16), br.hidden_norm.eps)
    ctx = mx.stack([U[a - L - lo:a - lo] for a, _, _ in ancres])
    mask_id = br.config.mask_token_id
    entree = mx.stack([mx.concatenate([J[a:a + 1], mx.full((bloc - 1,), mask_id, mx.int32)]) for a, _, _ in ancres])
    n = J.shape[0]
    idx = mx.array([[min(a + 1 + k, n - 1) for k in range(bloc - 1)] for a, _, _ in ancres])
    masque = mx.array([[1.0 if a + 1 + k < b else 0.0 for k in range(bloc - 1)] for a, b, _ in ancres])
    lab = J[idx]
    pred = mx.concatenate([J[mx.array([a for a, _, _ in ancres])][:, None], lab[:, :-1]], axis=1)
    return ctx, entree, lab, pred, masque


def avance_projete(br, entree, hctx, cache):
    h = br.embed_tokens(entree) * br.embed_scale
    hctx = hctx.astype(h.dtype)
    for layer, c in zip(br.layers, cache):
        h = layer(h, hctx, br.rope, c)
    return br.norm(h[:, 1:]).astype(hctx.dtype)


def perte_suivie(mx, nn, br, H, J, ancres, bloc, gamma, poids_sel):
    ctx, entree, lab, pred, masque = lot_suivi(mx, br, H, J, ancres, bloc)
    h = avance_projete(br, entree, ctx, br.make_cache())
    logits = br.compute_logits(h).astype(mx.float32)
    w = mx.exp(-mx.arange(lab.shape[1], dtype=mx.float32) / gamma)[None] * masque
    ce = nn.losses.cross_entropy(logits, lab, reduction="none")
    base = mx.sum(ce * w) / mx.maximum(mx.sum(w), 1e-6)
    sel = br.candidate_selector
    cand = mx.stop_gradient(mx.argpartition(logits, -sel.top_k, axis=-1)[..., -sel.top_k:])
    unaire = mx.take_along_axis(logits, cand, axis=-1)
    hp = sel.hidden_projection(h)
    aretes = mx.sum(sel.predecessor_codebook(pred)[:, :, None, :] * hp[:, :, None, :]
                    * sel.successor_codebook(cand), axis=-1).astype(mx.float32)
    egal = cand == lab[..., None]
    ws = w * mx.any(egal, axis=-1).astype(mx.float32)
    ce_sel = nn.losses.cross_entropy(unaire + aretes, mx.argmax(egal.astype(mx.int32), axis=-1), reduction="none")
    selp = mx.sum(ce_sel * ws) / mx.maximum(mx.sum(ws), 1e-6)
    return base + poids_sel * selp, (base, selp)


def acceptation(mx, brouillon, ctx, entree, masque, refs):
    cache = brouillon.make_cache()
    h = brouillon.hidden_states(entree, ctx, cache, logits_start=1)
    logits = brouillon.compute_logits(h)
    chemin, _, _ = brouillon.candidate_selector.select(h, logits, entree[:, 0], 0.0)
    ok = masque > 0
    return [mx.sum(mx.cumprod(((chemin == r) & ok).astype(mx.int32), axis=1), axis=1) for r in refs]


def portee_projection(mx, br) -> int:
    from mlx.utils import tree_flatten

    br.freeze()
    for m in (br.fc, br.hidden_norm):
        m.unfreeze()
        m.set_dtype(mx.float32)
    return sum(v.size for _, v in tree_flatten(br.trainable_parameters()))


def sauver(mx, brouillon, sortie: Path, zlab: str, info: dict) -> None:
    from mlx.utils import tree_flatten

    sortie.mkdir(parents=True, exist_ok=True)
    orig = mx.load(str(Path(zlab) / "model.safetensors"))
    params = dict(tree_flatten(brouillon.parameters()))
    poids = {}
    for k, v in orig.items():
        src = k + ".weight" if k in ("candidate_selector.predecessor_codebook",
                                     "candidate_selector.successor_codebook") else k
        if src not in params:
            raise RuntimeError("key missing from the trained drafter: " + k)
        poids[k] = params[src].astype(v.dtype)
        if poids[k].shape != v.shape:
            raise RuntimeError("shape changed: " + k)
    mx.eval(poids)
    mx.save_safetensors(str(sortie / "model.safetensors"), poids, metadata={"format": "pt"})
    (sortie / "config.json").write_text((Path(zlab) / "config.json").read_text())
    (sortie / "ajustement.json").write_text(json.dumps(info, ensure_ascii=False, indent=1))


def gains(c: dict) -> tuple[dict, int]:
    z, a = c["zlab4"], c["ajuste4"]
    g = {k: a[k]["acceptes_glouton"] / z[k]["acceptes_glouton"] - 1 for k in z if z[k]["acceptes_glouton"] > 0}
    return g, z["tout"]["blocs"]


# -- the task ---------------------------------------------------------------------------------------------------------
class Falcon(Task):
    name = "falcon"

    def __init__(self, tasks: Any, context: dict[str, Any]) -> None:
        super().__init__(tasks, context)
        self.family = context.get("family")
        drafter = context.get("drafter")
        if self.family is None or drafter is None:
            raise ValueError("no DFlash2 drafter in this server (--drafter)")
        self.root = Path(os.environ.get("PLI_TASKS_FALCON_DIR", "").strip() or HOME / "falcon/dfl").expanduser()
        self.tokens = Path(os.environ.get("PLI_TASKS_FALCON_TOKENS", "").strip()
                           or os.environ.get("PLI_FALCON_TOKENS", "").strip() or HOME / "falcon/sabi-noema-27b/jetons")
        zlab = os.environ.get("PLI_TASKS_FALCON_ZLAB", "").strip()
        if not zlab:
            from pli.drafters.dflash_drafter import resolve_draft_path

            zlab = resolve_draft_path("z-lab/Qwen3.8-27B-DFlash2")      # the reference of every gain, never the served one
        self.zlab = str(zlab)
        tok = context.get("tokenizer")
        self.im_end = int(tok.convert_tokens_to_ids("<|im_end|>")) if tok is not None else 248046
        self.window = _hours(os.environ.get("PLI_TASKS_FALCON_HOURS", "1-7"))
        self.every_s = _env_float("PLI_TASKS_FALCON_EVERY_H", 24.0) * 3600
        self.since_h = _env_float("PLI_TASKS_FALCON_SINCE_H", 26.0)
        self.sessions = int(_env_float("PLI_TASKS_FALCON_SESSIONS", 60))
        self.controls = int(_env_float("PLI_TASKS_FALCON_CONTROL", 12))
        self.chunk = int(_env_float("PLI_TASKS_FALCON_CHUNK", 512))
        self.anchors = int(_env_float("PLI_TASKS_FALCON_ANCHORS", 256))
        self.anchors_control = int(_env_float("PLI_TASKS_FALCON_ANCHORS_CONTROL", 128))
        self.shadow = os.environ.get("PLI_TASKS_FALCON_SHADOW", "0") == "1"
        self.drafter_file = "" if self.shadow else os.environ.get("PLI_TASKS_DRAFTER_FILE", "").strip()
        base_file = os.environ.get("PLI_TASKS_FALCON_BASE_FILE", "").strip()
        self.base_file = Path(base_file).expanduser() if base_file else self.root / "courant.txt"
        self.cycle: Iterator[None] | None = None
        self.held = 0
        self.rec: dict[str, Any] = {}

    def due(self, now: float) -> bool:
        if self.cycle is not None:
            return True
        if not self.tokens.is_dir():
            return False
        if now - float(self.state.get("last_start", 0) or 0) < self.every_s * 0.9:
            return False
        return in_window(time.localtime(now).tm_hour, self.window)

    def need_bytes(self) -> int:
        return 8 * 2**30 if self.cycle is None else 2 * 2**30

    def step(self, engine: Any) -> None:
        if self.cycle is None:
            self.state["last_start"] = time.time()
            self.save()
            self.cycle = self._cycle()
        try:
            next(self.cycle)
        except StopIteration:
            self.cycle = None
        except BaseException:
            self.cycle = None
            self._abandon("error")
            raise

    # -- one cycle -----------------------------------------------------------------------------------------------
    def _cycle(self) -> Iterator[None]:
        import mlx.core as mx
        import mlx.nn as nn
        import mlx.optimizers as optim
        from mlx.utils import tree_flatten

        from pli.additions.falcon.states import read_states
        from pli.drafters.dflash_drafter import _vendor

        zl = _vendor()
        R = self.root
        name = time.strftime("%Y%m%d-%H%M") + "-pli"
        N = R / "nuits" / name
        N.mkdir(parents=True, exist_ok=True)
        os.chmod(R, stat.S_IRWXU)
        self.N = N
        rec = self.rec = {"nuit": name, "debut": time.strftime("%F %T"), "par": "pli"}
        self.log("cycle_start", folder=str(N))
        started = time.time()
        # 1. preparation, a CPU subprocess
        with open(N / "sortie.log", "a") as out:
            proc = subprocess.Popen([sys.executable, str(R / "preparer_falcon.py"), str(self.tokens), str(N / "donnees"),
                                     "--depuis", str(self.since_h)], stdout=out, stderr=subprocess.STDOUT,
                                    env=dict(os.environ, HF_HUB_OFFLINE="1"))
        self.proc = proc
        while proc.poll() is None:
            yield
        if proc.returncode:
            return self._finish(f"preparation failed ({proc.returncode})")
        prep = json.loads((N / "donnees/preparation.json").read_text())
        rec["preparation"] = prep
        if prep["train"] < 5 or prep["controle"] < 1:
            return self._finish("too few conversations (%d, control %d)" % (prep["train"], prep["controle"]))
        base = self.base_file.read_text().strip() if self.base_file.exists() else self.zlab
        rec["base"] = base
        rec["ombre"] = self.shadow
        im_end = self.im_end
        target = self.context["drafter"].target
        # 2. training
        zl.snapshot_download = lambda repo_id, **_: base
        br = zl.load_draft(base)
        br.bind(target)
        trained = portee_projection(mx, br)
        opt = optim.AdamW(learning_rate=5e-5, weight_decay=0.01)
        graine = int(time.strftime("%Y%m%d"))
        sess = preparer_sessions(lire_sessions(N / "donnees/train.jsonl", self.sessions, graine), 12288, im_end)
        total = max(1, len(sess))
        rng = random.Random(graine + 1)
        rec["poids_entraines"] = trained
        steps, seen, read_tokens, previous = 0, set(), 0, []
        journal = open(N / "journal.jsonl", "a")
        yield
        for vu, s in enumerate(sess):
            jet = s["jetons"]
            brutes = hors([tuple(x) for x in s["ecrits"]], jet, seen)
            seen |= {hashlib.md5(json.dumps(jet[a:b]).encode()).hexdigest() for a, b, _ in s["ecrits"]}
            common = 0                                   # as ajuste.Lecteur.commun: answers the previous one already showed
            while common < min(len(jet), len(previous)) and jet[common] == previous[common]:
                common += 1
            previous = jet
            lots = ancrages_suivis(nouvelles_reponses(brutes, jet, im_end, common), self.anchors // BLOC, rng, BLOC)
            if not lots:
                continue
            t_read = time.time()
            pieces = []
            for _, _, taps, _ in read_states(self.family, jet, chunk=self.chunk):
                pieces.append(taps)
                yield
            H = mx.concatenate(pieces, axis=0) if len(pieces) > 1 else pieces[0]
            del pieces
            read_s = time.time() - t_read
            read_tokens += len(jet)
            J = mx.array(jet, dtype=mx.int32)
            losses = []
            for groupe in lots:
                frac = vu / total
                warm = min(1.0, (vu + 1) / max(1.0, total * 0.04))
                opt.learning_rate = mx.array(5e-5 * warm * 0.5 * (1 + math.cos(math.pi * frac)))
                f = nn.value_and_grad(br, lambda m: perte_suivie(mx, nn, m, H, J, groupe, BLOC, 4.0, 1.0))
                (loss, (b0, s0)), g = f(br)
                g, _ = optim.clip_grad_norm(g, 1.0)
                opt.update(br, g)
                mx.eval(loss, b0, s0, br.trainable_parameters(), opt.state)
                losses.append((loss.item(), b0.item(), s0.item()))
                steps += 1
                yield
            del H, J
            m = [sum(x[i] for x in losses) / len(losses) for i in range(3)]
            journal.write(json.dumps({"pas": steps, "session": vu + 1, "jetons": len(jet), "lecture_s": round(read_s, 1),
                                      "perte": round(m[0], 4), "bloc": round(m[1], 4), "selecteur": round(m[2], 4),
                                      "t": round(time.time() - started)}) + "\n")
            journal.flush()
        journal.close()
        rec.update(pas=steps, jetons_lus=read_tokens)
        info = {"portee": "projection", "pas": steps, "sessions": total, "par": "pli", "base": base}
        sauver(mx, br, N / "ajuste/brouillon", self.zlab, info)
        del br, opt
        mx.clear_cache()
        yield
        # 3. control, both drafters in 4 bits as PLI serves them
        def four_bits(path: str) -> Any:
            zl.snapshot_download = lambda repo_id, **_: path
            d = zl.load_draft(path)
            nn.quantize(d, group_size=64, bits=4,
                        class_predicate=lambda _, mod: isinstance(mod, nn.Linear) and mod.weight.shape[-1] % 64 == 0)
            mx.eval(d.parameters())
            d.bind(target)
            return d

        brs = {"zlab4": four_bits(self.zlab), "ajuste4": four_bits(str(N / "ajuste/brouillon"))}
        mask_id = brs["zlab4"].config.mask_token_id
        sess_c = preparer_sessions(lire_sessions(N / "donnees/controle.jsonl", self.controls, graine), 12288, im_end)
        exclues = empreintes(N / "donnees/train.jsonl", 12288)
        rng_c = random.Random(graine)
        previous = []
        res: dict[str, dict[str, list[int]]] = {k: {} for k in brs}
        for s in sess_c:
            jet = s["jetons"]
            brutes = hors([(a, b, d) for a, b, d in s["ecrits"]], jet, exclues)
            common = 0
            while common < min(len(jet), len(previous)) and jet[common] == previous[common]:
                common += 1
            previous = jet
            anc = ancrages(nouvelles_reponses(brutes, jet, im_end, common), self.anchors_control, rng_c)
            if not anc:
                continue
            Hs, Gs = [], []
            for _, _, taps, greedy in read_states(self.family, jet, chunk=self.chunk):
                Hs.append(taps)
                Gs.append(greedy)
                yield
            H = mx.concatenate(Hs, axis=0) if len(Hs) > 1 else Hs[0]
            G = mx.concatenate(Gs, axis=0) if len(Gs) > 1 else Gs[0]
            del Hs, Gs
            J = mx.array(jet, dtype=mx.int32)
            anc.sort()
            for j in range(0, len(anc), 8):
                groupe = anc[j:j + 8]
                ctx, entree, lab, _, masque, glou = lot(mx, mask_id, H, G, J, groupe, BLOC)
                full = (mx.sum(masque, axis=1) == BLOC - 1).tolist()
                for nom, d in brs.items():
                    le, lg = acceptation(mx, d, ctx, entree, masque, (lab, glou))
                    mx.eval(le, lg)
                    for (_, _, dom), p, e, gl in zip(groupe, full, le.tolist(), lg.tolist()):
                        if not p:
                            continue
                        for key in (dom, "tout"):
                            v = res[nom].setdefault(key, [0, 0, 0])
                            v[0] += 1
                            v[1] += e
                            v[2] += gl
                yield
            del H, G, J
        del brs
        mx.clear_cache()
        control = {nom: {d: {"blocs": v[0], "acceptes_ecrit": round(v[1] / max(1, v[0]), 4),
                             "acceptes_glouton": round(v[2] / max(1, v[0]), 4)} for d, v in r.items()}
                   for nom, r in res.items()}
        (N / "ajuste/controle.json").write_text(json.dumps(control, ensure_ascii=False, indent=1))
        if "tout" not in control.get("zlab4", {}):
            return self._finish("empty control")
        # 4. decision, as nuit.py
        g, blocks = gains(control)
        rec.update(gains={k: round(v, 4) for k, v in g.items()}, blocs=blocks, duree_s=round(time.time() - started))
        hist = [json.loads(line) for line in open(R / "historique.jsonl")] if (R / "historique.jsonl").exists() else []
        prev = next((h["gains"]["tout"] for h in reversed(hist) if h.get("garde")), 0.0)
        worst = min((v for k, v in g.items() if k != "tout"), default=0.0)
        rec["garde"] = g["tout"] >= prev - GAIN_KEEP_TOLERANCE and worst >= DOMAIN_FLOOR
        if rec["garde"]:
            folder = str(N / "ajuste/brouillon")
            if not self.shadow:
                (R / "courant.txt").write_text(folder + "\n")
            if self.drafter_file:
                Path(self.drafter_file).write_text(folder + "\n")
        self._finish("gain %+.1f %% on %d blocks (worst domain %+.1f %%) : %s" % (
            100 * g["tout"], blocks, 100 * worst, "kept" if rec["garde"] else "discarded"))

    def _abandon(self, why: str) -> None:
        proc = getattr(self, "proc", None)
        if proc is not None and proc.poll() is None:
            proc.terminate()
        if self.rec:
            self._finish(f"cycle abandoned ({why})")

    def _finish(self, end: str) -> None:
        R, N, rec = self.root, getattr(self, "N", None), self.rec
        rec["fin"], rec["fin_t"] = end, time.strftime("%F %T")
        if N is not None:
            shutil.rmtree(N / "donnees", ignore_errors=True)       # text of the requests: not kept past the cycle
        with open(R / "historique.jsonl", "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        courant = (R / "courant.txt").read_text().strip() if (R / "courant.txt").exists() else ""
        servi = (R / "servi.txt").read_text().strip() if (R / "servi.txt").exists() else ""
        wanted = ""
        if self.drafter_file and Path(self.drafter_file).exists():
            wanted = Path(self.drafter_file).read_text().strip()
        served = str(getattr(self.context.get("drafter"), "path", "") or "")
        keep = [x for x in (courant, servi, wanted, served) if x]
        if self.shadow and rec.get("garde") and N is not None:
            keep.append(str(N))                         # shadow: the last kept one stays for the comparison

        for d in sorted((R / "nuits").iterdir()) if (R / "nuits").is_dir() else []:
            if not any(k.startswith(str(d)) for k in keep):
                shutil.rmtree(d, ignore_errors=True)
        self.log("cycle_end", end=end, gains=rec.get("gains"), garde=rec.get("garde"))
        self.rec = {}

    def summary(self) -> dict[str, Any]:
        return {"running": self.cycle is not None, "last_start": self.state.get("last_start"),
                "window": "%d-%d" % self.window}
