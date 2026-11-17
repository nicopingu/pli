"""Video generation child process: a job queue, one at a time, each launched as an h3 subprocess (MiniMax H3).

    python -m pli.gen.fils_video --port N --catalogue catalogue-video.json --sorties DOSSIER [--simuler]
    python -m pli.gen.fils_video --port N --h3 CHEMIN --modele DOSSIER --sorties DOSSIER [--simuler]
    common option: --ssd-streaming yes|no (replaces "ssd_streaming" from the catalog)

Routes (OpenAI-style video format):
    GET    /sante                  -> {"ok": true, "occupe": bool}  (occupe: a job is queued or in progress)
    POST   /v1/videos              -> 202 {"id", "status": "queued", ...}
           JSON: prompt (required), seconds or frames (not both), size "512x512", model (catalog name),
           lora [{"nom": ..., "force": ...}] (one video LoRA at a time), and PLI fields steps, seed, reuse,
           layers, first_frame, last_frame (base64, optional, "data:...;base64," prefix allowed)
    GET    /v1/videos/{id}         -> the job state (see Travail.vue)
    GET    /v1/videos/{id}/content -> the MP4 (409 until the job is finished)
    DELETE /v1/videos/{id}         -> queued or in-progress job: cancelled (200, status "cancelled");
                                      finished job: deleted with its files (200 {"deleted": true})
    POST   /annuler                -> cancels the current job (200 {"annulee": bool}), like other child processes

States ("status"): queued, in_progress, completed, failed, cancelled. "etape" specifies: waiting (queued),
fusion (LoRA fusion, counted in job time), render (h3), null once finished. "attente" gives the reason
for a queued job that does not start: queue (another video is running), general, image_ou_voix, memoire_voix. "position": rank
in the queue (1 = next). "pauses", "en_pause", "secondes_pause": preemption by the general model.

Progress: h3_cli.c:110-125 (cli_progress) writes to stderr "\r<phase 25 car.> <done>/<total>" on each change,
with a newline at the end of the phase. "progress" (0 to 1) follows only the "denoise" phase (done/total of steps);
"phase" gives the last phase read with its done/total ({"nom": "fusion"} during fusion).

Streaming mode (default): h3 receives --ssd-streaming (main.c:270 and 357 of h3.c) and keeps two DiT blocks in memory
(measurement from 14/11/2026, 512x512: 10.5 to 10.9 GB vs 39 to 43 GB without). Memory requirement per model: "memoire_go"
(streaming, default 12) or "memoire_go_resident" (no streaming, default 43), checked at creation if no video is
running and at job start (503 "memoire" / job failed). In streaming, the video is accepted alongside the
loaded general model; without streaming it is refused with 503 "general_charge" when the request carries the header
X-PLI-General: charge (set by routes.py).

Queue and preemption: one job at a time. A job does not start as long as PLI_GEN_DRAPEAU (general model at
work) or PLI_GEN_DRAPEAU_AUTRES (an image or a voice in progress, set by the supervisor) exists. During a job,
h3 (or fusion) has no callback: its group receives SIGSTOP as soon as PLI_GEN_DRAPEAU appears and SIGCONT as soon as it
disappears. Cancellation and stop first resume the process (SIGCONT) then terminate it (SIGTERM, then SIGKILL).

Room for the voice assistant during a video (14/11/2026, video_* keys of PLI_GEN_CONFIG; the supervisor sets these variables, nothing without them):
PLI_GEN_DRAPEAU_VOIX : h3 is also paused while a voice or transcription is in progress (as with the general model).
PLI_GEN_DRAPEAU_VIDEO : set while a job is running (the supervisor then reduces the general model's parallelism).
PLI_GEN_VOIX_OCTETS (audio family need + reserve) and PLI_GEN_VOIX_RESERVE_OCTETS : a job only starts if the
available memory covers its need + PLI_GEN_VOIX_OCTETS, or its need + PLI_GEN_VOIX_RESERVE_OCTETS if PLI_GEN_DRAPEAU_AUDIO
exists (voice loaded); otherwise it stays queued, wait « memoire_voix », and writes what is missing to PLI_GEN_BESOIN (the
supervisor then makes the general model release memory). Beyond PLI_GEN_ATTENTE_MEMOIRE_S (default 1800): failed "memoire".

LoRA (catalog "lora"): a LoRA is merged into the transformer by fusion_lora_h3.py (complete model folder,
transformer rewritten, the rest as links). Already merged for (base model, LoRA file, force): resumed as
is, whether saved ("fusions", "dossier_fusions") or in cache ("cache_fusions"), recognized by its
fusion_lora_h3.json. Otherwise: "sauvegardee" mode -> merge written to "dossier_fusions" and kept; mode
"a_la_demande" -> merge written to "cache_fusions", capped at "cache_go" (eviction of least recently used;
a saved merge is never in the cache, so never evicted). Disk space checked by os.statvfs before the
merge: 507 "disque" at creation, job failed at start if it ran out in the meantime.

LoRA in memory (PLI 5.6.5, 17/11/2026): mode "en_memoire" -> no merge on disk; the LoRA is converted once to the
native layout of h3.c (tools/pli-lora-natif, files in "lora_natif.dossier", named after the source size and date)
and passed to h3 by --lora FILE@FORCE (h3.c with tools/h3c-lora.patch): h3 adds force * B @ A to each block's BF16
weights as it loads them. Several LoRA at once (up to LORA_MAX) only in this mode; the defaults (steps, size,
seconds) come from the first LoRA of the list that has them. Memory: + converted file size + the F32 delta buffer.
Catalog: "lora_natif": {"python": ..., "script": ..., "dossier": ...} (default python and script: those of
"fusion", script pli-lora-natif or lora_natif_h3.py next to the fusion script).

Catalogue (JSON) : voir docs/gen/catalogue-video.exemple.json.
  {"h3": CHEMIN, "defaut": "h3", "ssd_streaming": true,
   "modeles": {"h3": {"chemin": DOSSIER, "memoire_go": 12, "memoire_go_resident": 43}},
   "lora": {"turbo": {"chemin": FICHIER, "force": 1.0, "mode": "sauvegardee|a_la_demande",
                           "base": ["h3"], "taille_go": 33, "steps": null}},
   "fusions": [DOSSIER, ...], "dossier_fusions": DOSSIER, "cache_fusions": DOSSIER, "cache_go": 150,
   "fusion": {"python": CHEMIN, "script": CHEMIN, "taille_go": 62, "marge_go": 4}}

Jobs are written to the outputs folder (<id>.json, <id>.mp4) and re-read at startup; a job in the queue or
running at the time of a stop is re-read as "failed" (interrupted). SIGTERM stops h3 then the child process.
--simuler mode: fake h3 and fake merge (Python scripts that sleep, write their progress and a small file).
"""

from __future__ import annotations

import argparse
import base64
import binascii
import hashlib
import json
import os
import shutil
import queue
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

ATTENTE_DRAPEAU_MAX_S = 6 * 3600.0     # a flag never removed does not block the queue forever
PAS_DRAPEAU_S = 0.05
PAS_MEMOIRE_S = 0.5                    # memory measurement (vm_stat) while waiting for "memoire_voix"


def _effacer(chemin: str) -> None:
    if chemin:
        try:
            os.unlink(chemin)
        except FileNotFoundError:
            pass


def _poser(chemin: str, oui: bool) -> None:
    if not chemin:
        return
    if oui:
        open(chemin, "w").close()
    else:
        _effacer(chemin)


LIMITE_CORPS = 256 * 1024**2           # first and last images in base64
RE_ID = re.compile(r"^video_[0-9a-f]{16}$")
RE_TAILLE = re.compile(r"^(\d{2,5})x(\d{2,5})$")
RE_PROGRES = re.compile(r"^(?P<phase>.+?)\s+(?P<fait>\d+)/(?P<total>\d+)\s*$")
MODELE = "minimax-h3"
GO = 1024**3
MEMOIRE_GO_STREAMING = 12.0            # 10.5 to 10.9 GB measured (14/11/2026) + margin
MEMOIRE_GO_RESIDENT = 43.0             # 39 to 43 GB measured without streaming
TAILLE_FUSION_GO = 62.0                # 33 to 62 GB per merge (14/11/2026)
MARGE_DISQUE_GO = 4.0                  # fusion_lora_h3.py itself requires 2 GiB of margin (main, "insufficient space")
CACHE_GO = 150.0
MODES_LORA = ("sauvegardee", "a_la_demande", "en_memoire")
LORA_MAX = 4                           # LoRA in memory per video (h3.c accepts 8)
DELTA_LORA_GO = 0.6                    # F32 delta buffer of h3: FFN*2*HIDDEN floats (h3_dit.c, lora_load_all)
NATIF_FACTEUR = 1.5                    # converted size / source size before conversion (turbo: 1.36, qkv 3r)
META_FUSION = "fusion_lora_h3.json"    # written by fusion_lora_h3.py at the end of a successful merge
MARQUE_USAGE = ".pli_usage"            # date of last use of a cache merge (LRU eviction)
FINIS = ("completed", "failed", "cancelled")

FAUX_H3 = r'''
import sys, time, os
a = sys.argv[1:]
sortie = a[a.index("-o") + 1]
duree = float(os.environ.get("FAUX_H3_PAS_S", "0"))
sys.stderr.write("h3: simulation\n")
for phase, n in (("text encoder", 2), ("denoise", 4), ("FFmpeg", 2)):
    for i in range(0, n + 1):
        sys.stderr.write("\r%-25s %4d/%-4d" % (phase, i, n)); sys.stderr.flush()
        time.sleep(duree)
    sys.stderr.write("\n")
if os.environ.get("FAUX_H3_ECHEC"):
    sys.stderr.write("h3: echec simule\n"); sys.exit(2)
with open(sortie, "wb") as f:
    f.write(b"MP4-SIMULE " + " ".join(a).encode())
'''

# same interface as fusion_lora_h3.py: MODELE SORTIE --lora FICHIER@FORCE, fusion_lora_h3.json written at the end
FAUSSE_FUSION = r'''
import json, os, sys, time
a = sys.argv[1:]
modele, sortie = a[0], a[1]
fichier, _, force = a[a.index("--lora") + 1].rpartition("@")
os.makedirs(sortie, exist_ok=True)
n = int(os.environ.get("FAUSSE_FUSION_PAS", "4"))
for i in range(n):
    print("  shard %d fait" % i, flush=True)
    time.sleep(float(os.environ.get("FAUSSE_FUSION_PAS_S", "0")))
if os.environ.get("FAUSSE_FUSION_ECHEC"):
    print("REFUS : echec simule", file=sys.stderr); sys.exit(2)
with open(os.path.join(sortie, "transformer.bin"), "wb") as f:
    f.write(b"0" * int(os.environ.get("FAUSSE_FUSION_OCTETS", "1024")))
json.dump({"modele": os.path.abspath(modele), "transformer": "FL2VA/transformer", "qkv": ["comfy"],
           "loras": [{"fichier": os.path.abspath(fichier), "force": float(force)}]},
          open(os.path.join(sortie, "fusion_lora_h3.json"), "w"), indent=1)
'''


# same interface as lora_natif_h3.py: MODELE LORA SORTIE
FAUX_NATIF = r'''
import os, sys
modele, lora, sortie = sys.argv[1:4]
if os.environ.get("FAUX_NATIF_ECHEC"):
    print("REFUS : echec simule", file=sys.stderr); sys.exit(2)
with open(sortie, "wb") as f:
    f.write(b"0" * int(os.environ.get("FAUX_NATIF_OCTETS", "1024")))
'''


class ErreurDemande(Exception):
    def __init__(self, code: int, message: str, type_: str = "invalid_request_error") -> None:
        super().__init__(message)
        self.code, self.message, self.type = code, message, type_


def lancement_h3(h3: str) -> dict:
    """cwd and environment for h3 (tests from 13/11/2026): h3.c opens h3_shaders.metal by a path relative to the current
    directory (h3.c:706 and following), and looks for ffmpeg via PATH (H3_FFMPEG replaces it, h3_ffmpeg.c:18); a service
    started by launchd or ssh does not have /opt/homebrew/bin in its PATH."""

    options: dict = {}
    dossier = os.path.dirname(os.path.abspath(h3))
    if os.path.exists(os.path.join(dossier, "h3_shaders.metal")):
        options["cwd"] = dossier
    if not os.environ.get("H3_FFMPEG"):
        ffmpeg = shutil.which("ffmpeg") or ("/opt/homebrew/bin/ffmpeg" if os.path.exists("/opt/homebrew/bin/ffmpeg") else "")
        if ffmpeg:
            options["env"] = dict(os.environ, H3_FFMPEG=ffmpeg)
    return options


# ---------------------------------------------------------------- catalogue

def _chemin(v: Any) -> str:
    return os.path.expanduser(str(v)) if v else ""


def catalogue_simple(h3: str, modele: str) -> dict[str, Any]:
    """Catalog of a single model, without LoRA (argv --h3 --modele form from before the catalog)."""

    return valider_catalogue({"h3": h3, "defaut": MODELE, "modeles": {MODELE: {"chemin": modele}}})


def charger_catalogue(chemin: str, environ=os.environ) -> dict[str, Any]:
    """The JSON catalog, plus the private LoRAs from PLI_GEN_CATALOGUE_PRIVE (gen/catalogue_prive.py), then validated."""

    from pli.gen import catalogue_prive

    with open(chemin, encoding="utf-8") as f:
        c = json.load(f)
    if isinstance(c, dict):
        catalogue_prive.fusionner(c, "video", environ)
    return valider_catalogue(c)


def valider_catalogue(c: dict[str, Any]) -> dict[str, Any]:
    modeles, loras = c.get("modeles"), c.get("lora", {}) or {}
    if not isinstance(modeles, dict) or not modeles:
        raise ValueError("video catalog: 'modeles' key missing or empty")
    if not isinstance(loras, dict):
        raise ValueError("video catalog: 'lora' must be an object")
    for nom, m in modeles.items():
        if not isinstance(m, dict):
            raise ValueError(f"model {nom}: object expected")
        m["chemin"] = _chemin(m.get("chemin"))
        for cle, defaut in (("memoire_go", MEMOIRE_GO_STREAMING), ("memoire_go_resident", MEMOIRE_GO_RESIDENT)):
            v = m.get(cle, defaut)
            if isinstance(v, bool) or not isinstance(v, (int, float)) or v < 0:
                raise ValueError(f"model {nom}: '{cle}' must be a positive number of GB")
            m[cle] = float(v)
    for nom, lo in loras.items():
        if not isinstance(lo, dict) or not lo.get("chemin"):
            raise ValueError(f"LoRA {nom} : « chemin » obligatoire")
        lo["chemin"] = _chemin(lo["chemin"])
        lo["force"] = float(lo.get("force", 1.0))
        lo["mode"] = lo.get("mode", "a_la_demande")
        if lo["mode"] not in MODES_LORA:
            raise ValueError(f"LoRA {nom} : « mode » attendu parmi {', '.join(MODES_LORA)}")
        base = lo.get("base")
        if base is not None and (not isinstance(base, list) or any(b not in modeles for b in base)):
            raise ValueError(f"LoRA {nom}: 'base' must name models from the catalog")
        d = lo.get("defauts", {})
        if not isinstance(d, dict) or set(d) - {"passes", "taille", "secondes"}:
            raise ValueError(f"LoRA {nom} : « defauts » : passes, taille, secondes seulement")
        if "taille" in d and not RE_TAILLE.match(str(d["taille"])):
            raise ValueError(f"LoRA {nom}: defauts.taille expected in the form 512x512")
    defaut = c.get("defaut") or next(iter(modeles))
    if defaut not in modeles:
        raise ValueError(f"video catalog: default {defaut!r} not in models")
    fusions = c.get("fusions", []) or []
    if not isinstance(fusions, list):
        raise ValueError("video catalog: 'fusions' must be a list of directories")
    fusion = c.get("fusion", {}) or {}
    natif = c.get("lora_natif", {}) or {}
    if not isinstance(natif, dict):
        raise ValueError("video catalog: 'lora_natif' must be an object")
    script_fusion = _chemin(fusion.get("script"))
    script_natif = _chemin(natif.get("script"))
    if not script_natif and script_fusion:
        voisins = [os.path.join(os.path.dirname(script_fusion), n) for n in ("pli-lora-natif", "lora_natif_h3.py")]
        script_natif = next((v for v in voisins if os.path.exists(v)), "")
    c.update(defaut=defaut, lora=loras, h3=_chemin(c.get("h3")), ssd_streaming=bool(c.get("ssd_streaming", True)),
             fusions=[_chemin(f) for f in fusions], dossier_fusions=_chemin(c.get("dossier_fusions")),
             cache_fusions=_chemin(c.get("cache_fusions")), cache_go=float(c.get("cache_go", CACHE_GO)),
             fusion={"python": _chemin(fusion.get("python")) or sys.executable, "script": _chemin(fusion.get("script")),
                     "taille_go": float(fusion.get("taille_go", TAILLE_FUSION_GO)),
                     "marge_go": float(fusion.get("marge_go", MARGE_DISQUE_GO))},
             lora_natif={"python": _chemin(natif.get("python")) or _chemin(fusion.get("python")) or sys.executable,
                         "script": script_natif, "dossier": _chemin(natif.get("dossier"))})
    return c


def lire_meta_fusion(dossier: str) -> tuple[str, list[tuple[str, float]]] | None:
    try:
        with open(os.path.join(dossier, META_FUSION), encoding="utf-8") as f:
            d = json.load(f)
        return (os.path.realpath(d["modele"]),
                [(os.path.realpath(x["fichier"]), float(x["force"])) for x in d["loras"]])
    except (OSError, ValueError, KeyError, TypeError):
        return None


def taille_dossier(dossier: str) -> int:
    """Bytes of the clean files in the directory (links to the base model do not count)."""

    total = 0
    for racine, dossiers, fichiers in os.walk(dossier):
        for nom in fichiers:
            try:
                st = os.lstat(os.path.join(racine, nom))
            except OSError:
                continue
            if not os.path.islink(os.path.join(racine, nom)):
                total += st.st_size
    return total


def place_libre_systeme(dossier: str) -> int:
    while dossier and not os.path.exists(dossier):
        dossier = os.path.dirname(dossier)
    st = os.statvfs(dossier or "/")
    return st.f_bavail * st.f_frsize


# ---------------------------------------------------------------- travaux

class Travail:
    def __init__(self, id_: str, params: dict, numero: int = 0) -> None:
        self.id = id_
        self.params = params
        self.numero = numero
        self.status = "queued"
        self.etape: str | None = "attente"
        self.attente: str | None = None
        self.progress = 0.0
        self.phase: dict | None = None
        self.error: str | None = None
        self.error_type: str | None = None
        self.created_at = int(time.time())
        self.completed_at: int | None = None
        self.pauses = 0
        self.en_pause = False
        self.secondes_pause = 0.0
        self.fusion: dict | None = None          # {"chemin", "source": sauvegardee|cache|nouvelle, "mode"}
        self.secondes_fusion: float | None = None
        self.secondes_rendu: float | None = None
        self.annule = False

    def vue(self, position: int | None = None) -> dict:
        return {"id": self.id, "object": "video", "model": self.params.get("modele", MODELE), "status": self.status,
                "etape": self.etape, "attente": self.attente, "position": position,
                "progress": round(self.progress, 4), "phase": self.phase, "error": self.error,
                "error_type": self.error_type, "created_at": self.created_at, "completed_at": self.completed_at,
                "seconds": self.params.get("seconds"), "frames": self.params.get("frames"),
                "size": self.params.get("size"), "lora": self.params.get("lora") or self.params.get("lora_memoire"),
                "ssd_streaming": self.params.get("ssd_streaming"), "fusion": self.fusion,
                "pauses": self.pauses, "en_pause": self.en_pause, "secondes_pause": round(self.secondes_pause, 2),
                "secondes_fusion": self.secondes_fusion, "secondes_rendu": self.secondes_rendu}


class Moteur:
    """The queue, the worker thread and the current child process (fusion or h3)."""

    def __init__(self, h3: str, modele: str, sorties: str, simuler: bool = False, *,
                 catalogue: dict[str, Any] | None = None, ssd_streaming: bool | None = None,
                 memoire_disponible: Callable[[], int] | None = None,
                 place_libre: Callable[[str], int] | None = None) -> None:
        self.cat = catalogue if catalogue is not None else catalogue_simple(h3, modele)
        self.h3 = self.cat["h3"]
        self.sorties, self.simuler = sorties, simuler
        self.ssd_streaming = self.cat["ssd_streaming"] if ssd_streaming is None else bool(ssd_streaming)
        if memoire_disponible is None and not simuler:
            from pli.gen.superviseur import memoire_disponible_systeme
            memoire_disponible = memoire_disponible_systeme
        self.memoire_disponible = memoire_disponible           # None: no control (simulation)
        self.place_libre = place_libre or place_libre_systeme
        os.makedirs(sorties, exist_ok=True)
        self.travaux: dict[str, Travail] = {}
        self.file: queue.Queue[str | None] = queue.Queue()
        self.verrou = threading.Lock()
        self.verrou_proc = threading.Lock()                     # SIGSTOP, SIGCONT and stopping the current process
        self.arret = threading.Event()
        self.proc: subprocess.Popen | None = None
        self.courant: Travail | None = None
        self.numero = 0
        self.drapeau = os.environ.get("PLI_GEN_DRAPEAU", "")
        self.drapeau_autres = os.environ.get("PLI_GEN_DRAPEAU_AUTRES", "")
        # room for the voice assistant (see header): variables set by the supervisor according to video_* keys; empty = before
        self.drapeau_voix = os.environ.get("PLI_GEN_DRAPEAU_VOIX", "")
        self.drapeau_video = os.environ.get("PLI_GEN_DRAPEAU_VIDEO", "")
        self.drapeau_audio = os.environ.get("PLI_GEN_DRAPEAU_AUDIO", "")
        self.fichier_besoin = os.environ.get("PLI_GEN_BESOIN", "")
        self.voix_octets = int(os.environ.get("PLI_GEN_VOIX_OCTETS", "0") or 0)
        self.voix_reserve_octets = int(os.environ.get("PLI_GEN_VOIX_RESERVE_OCTETS", "0") or 0)
        self.attente_memoire_max_s = float(os.environ.get("PLI_GEN_ATTENTE_MEMOIRE_S", "1800") or 1800)
        for chemin in (self.drapeau_video, self.fichier_besoin):     # leftovers from a child stopped without cleanup
            _effacer(chemin)
        self.faux = self.fausse_fusion = self.faux_natif = None
        if simuler:
            self.faux = os.path.join(sorties, ".faux_h3.py")
            self.fausse_fusion = os.path.join(sorties, ".fausse_fusion.py")
            self.faux_natif = os.path.join(sorties, ".faux_natif.py")
            for chemin, texte in ((self.faux, FAUX_H3), (self.fausse_fusion, FAUSSE_FUSION), (self.faux_natif, FAUX_NATIF)):
                with open(chemin, "w", encoding="utf-8") as f:
                    f.write(texte)
        self._relire()
        self.fil = threading.Thread(target=self._boucle, name="travail-video", daemon=True)
        self.fil.start()

    # disque

    def _chemin(self, id_: str, ext: str) -> str:
        return os.path.join(self.sorties, id_ + ext)

    def _ecrire(self, t: Travail) -> None:
        tmp = self._chemin(t.id, ".json.tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({**t.vue(), "params": t.params}, f)
        os.replace(tmp, self._chemin(t.id, ".json"))

    def _relire(self) -> None:
        for nom in os.listdir(self.sorties):
            if not nom.endswith(".json") or not RE_ID.match(nom[:-5]):
                continue
            try:
                with open(os.path.join(self.sorties, nom), encoding="utf-8") as f:
                    d = json.load(f)
                t = Travail(d["id"], d.get("params", {}))
                t.status, t.progress, t.phase = d["status"], d.get("progress", 0.0), d.get("phase")
                t.error, t.created_at, t.completed_at = d.get("error"), d.get("created_at", 0), d.get("completed_at")
                t.pauses, t.fusion = d.get("pauses", 0), d.get("fusion")
                t.etape = None
            except (OSError, ValueError, KeyError):
                continue
            if t.status in ("queued", "in_progress"):
                t.status, t.error = "failed", "interrupted by the stop of the video child"
                self._ecrire(t)
            self.travaux[t.id] = t

    # catalog and fusions

    def _besoin_octets(self, modele: str, streaming: bool, loras: list[dict] | None = None) -> int:
        m = self.cat["modeles"][modele]
        return int((m["memoire_go"] if streaming else m["memoire_go_resident"]) * GO) + self._besoin_lora(loras)

    def _besoin_lora(self, loras: list[dict] | None) -> int:
        """LoRA in memory: converted file (or source size x NATIF_FACTEUR before conversion) + the delta buffer."""

        if not loras:
            return 0
        total = DELTA_LORA_GO * GO
        for x in loras:
            natif = self._chemin_natif(x["nom"])
            try:
                total += os.path.getsize(natif) if natif and os.path.exists(natif) else \
                    os.path.getsize(self.cat["lora"][x["nom"]]["chemin"]) * NATIF_FACTEUR
            except OSError:
                pass
        return int(total)

    def _chemin_natif(self, nom: str) -> str:
        """Converted file of a LoRA: name, source size and date (a new source file is converted again)."""

        dossier = self.cat["lora_natif"]["dossier"]
        if not dossier:
            return ""
        try:
            st = os.stat(self.cat["lora"][nom]["chemin"])
        except OSError:
            return ""
        return os.path.join(dossier, f"{nom}-{st.st_size:x}-{int(st.st_mtime):x}.safetensors")

    def _dossiers_fusions(self, base: str) -> list[str]:
        """Candidate directories: each entry of 'fusions' (itself a fusion, or a directory of fusions), the
        subdirectories of 'dossier_fusions' (default: the parent directory of the base model) and those of the cache."""

        candidats: list[str] = []
        for d in self.cat["fusions"]:
            if os.path.exists(os.path.join(d, META_FUSION)):
                candidats.append(d)
            elif os.path.isdir(d):
                candidats += [os.path.join(d, n) for n in sorted(os.listdir(d))]
        for d in (self._dossier_sauvegarde(base), self.cat["cache_fusions"]):
            if d and os.path.isdir(d):
                candidats += [os.path.join(d, n) for n in sorted(os.listdir(d)) if not n.endswith(".partiel")]
        return candidats

    def _dossier_sauvegarde(self, base: str) -> str:
        return self.cat["dossier_fusions"] or os.path.dirname(os.path.realpath(base))

    def _dans_cache(self, dossier: str) -> bool:
        cache = self.cat["cache_fusions"]
        return bool(cache) and os.path.dirname(os.path.realpath(dossier)) == os.path.realpath(cache)

    def trouver_fusion(self, base: str, lora: str, force: float) -> str | None:
        voulu = (os.path.realpath(base), [(os.path.realpath(lora), float(force))])
        for d in self._dossiers_fusions(base):
            if lire_meta_fusion(d) == voulu:
                return d
        return None

    def _taille_fusion(self, nom_lora: str) -> int:
        lo = self.cat["lora"][nom_lora]
        return int(float(lo.get("taille_go", self.cat["fusion"]["taille_go"])) * GO)

    def fusions_du_cache(self) -> list[tuple[float, str, int]]:
        """(last use, directory, bytes) of the fusions in the cache, the oldest used first."""

        cache = self.cat["cache_fusions"]
        if not cache or not os.path.isdir(cache):
            return []
        sortie = []
        for n in os.listdir(cache):
            d = os.path.join(cache, n)
            if n.endswith(".partiel") or not os.path.isdir(d) or lire_meta_fusion(d) is None:
                continue
            try:
                usage = os.path.getmtime(os.path.join(d, MARQUE_USAGE))
            except OSError:
                usage = os.path.getmtime(d)
            sortie.append((usage, d, taille_dossier(d)))
        return sorted(sortie)

    def _destination(self, base: str, nom_lora: str, force: float) -> str:
        mode = self.cat["lora"][nom_lora]["mode"]
        if mode == "sauvegardee":
            return os.path.join(self._dossier_sauvegarde(base), f"{os.path.basename(os.path.realpath(base))}-{nom_lora}-f{force:g}")
        cache = self.cat["cache_fusions"]
        if not cache:
            raise ErreurDemande(503, f"LoRA {nom_lora} 'a_la_demande': 'cache_fusions' missing from catalog", "fusion")
        empreinte = hashlib.sha256(  # PLI 5.6.3: was SHA-1 (merges made before are made again once)
            f"{os.path.realpath(base)}|{self.cat['lora'][nom_lora]['chemin']}|{force:g}"
                                 .encode()).hexdigest()[:8]
        return os.path.join(cache, f"{nom_lora}-f{force:g}-{empreinte}")

    def _controle_disque(self, base: str, nom_lora: str, force: float, evincer: bool) -> str:
        """Destination of the fusion; ErreurDemande 507 'disk' if space is lacking. In on-demand mode, the cache is
        brought under 'cache_go' (eviction of least recently used); evincer=False only counts (creation)."""

        dest = self._destination(base, nom_lora, force)
        besoin = self._taille_fusion(nom_lora)
        marge = int(self.cat["fusion"]["marge_go"] * GO)
        liberable = 0
        if self._dans_cache(dest):
            plafond = int(self.cat["cache_go"] * GO)
            if besoin > plafond:
                raise ErreurDemande(507, f"fusion of {nom_lora}: {besoin / GO:.0f} GB estimated, more than the limit of the "
                                         f"cache ({plafond / GO:.0f} Go, « cache_go »)", "disque")
            fusions = self.fusions_du_cache()
            occupe = sum(o for _, _, o in fusions)
            while fusions and occupe + besoin > plafond:
                _, d, octets = fusions.pop(0)
                occupe -= octets
                liberable += octets
                if evincer:
                    shutil.rmtree(d, ignore_errors=True)
                    print(f"[video] fusion evicted from cache: {d}", flush=True)
            if evincer:
                liberable = 0                                # already returned to disk: the measure sees it
        try:
            libre = int(self.place_libre(os.path.dirname(dest)))
        except OSError as exc:
            raise ErreurDemande(503, f"disk space unreadable for {dest}: {exc}", "disk") from None
        if libre + liberable < besoin + marge:
            raise ErreurDemande(507, f"insufficient disk space to fuse {nom_lora} in {os.path.dirname(dest)}: "
                                     f"{libre / GO:.1f} GB free, {(besoin + marge) / GO:.1f} GB required "
                                     f"(fusion {besoin / GO:.0f} Go + marge {marge / GO:.0f} Go)", "disque")
        return dest

    # creation

    def creer(self, corps: dict, general_charge: bool = False) -> Travail:
        """Validates the request and puts it in the queue; ErreurDemande (code and message for the client) otherwise."""

        prompt = corps.get("prompt")
        if not isinstance(prompt, str) or not prompt.strip():
            raise ErreurDemande(400, "« prompt » (non-empty text) is required")
        if corps.get("seconds") is not None and corps.get("frames") is not None:
            raise ErreurDemande(400, "« seconds » and « frames » are mutually exclusive")
        nom = corps.get("model") or self.cat["defaut"]
        if nom == MODELE and nom not in self.cat["modeles"]:
            nom = self.cat["defaut"]
        if nom not in self.cat["modeles"]:
            raise ErreurDemande(400, f"model {nom!r} not in catalog (available: {', '.join(self.cat['modeles'])})")
        params: dict = {"prompt": prompt, "modele": nom, "ssd_streaming": self.ssd_streaming}
        if general_charge and not self.ssd_streaming:
            raise ErreurDemande(503, "the general model is loaded and the video runs without --ssd-streaming: it waits "
                                     "for its unloading (ssd_streaming=true in the catalog to run alongside)",
                                "general_charge")
        if corps.get("seconds") is not None:
            try:
                sec = float(corps["seconds"])
            except (TypeError, ValueError):
                raise ErreurDemande(400, "« seconds » must be a number") from None
            if not 0 < sec <= 3600:
                raise ErreurDemande(400, "« seconds » hors limites")
            params["seconds"] = sec
        if corps.get("frames") is not None:
            params["frames"] = self._entier(corps, "frames", 1, 100000)
        if corps.get("size") is not None:
            if not RE_TAILLE.match(str(corps["size"])):
                raise ErreurDemande(400, "« size » expected in the form 512x512")
            params["size"] = str(corps["size"])
        for cle, bas, haut in (("steps", 1, 1000), ("seed", 0, 2**63 - 1), ("reuse", 1, 3), ("layers", 1, 50)):
            if corps.get(cle) is not None:
                params[cle] = self._entier(corps, cle, bas, haut)
        loras = self._lora(corps, nom)
        lora = None
        if loras and self.cat["lora"][loras[0]["nom"]]["mode"] == "en_memoire":
            if not self.cat["lora_natif"]["dossier"] or not (self.cat["lora_natif"]["script"] or self.faux_natif):
                raise ErreurDemande(503, "LoRA in memory not configured (« lora_natif » missing from the catalog)", "lora")
            params["lora_memoire"] = loras
        elif loras:
            lora = params["lora"] = loras[0]
        for x in loras:                                 # defaults: the first LoRA that has them
            lo = self.cat["lora"][x["nom"]]
            defauts = lo.get("defauts") or {}
            steps = lo.get("steps") or defauts.get("passes")
            if "steps" not in params and steps:
                params["steps"] = int(steps)
            if "size" not in params and defauts.get("taille"):
                params["size"] = str(defauts["taille"])
            if "seconds" not in params and "frames" not in params and defauts.get("secondes"):
                params["seconds"] = float(defauts["secondes"])
        with self.verrou:
            libre_de_video = self.courant is None
        if self.memoire_disponible is not None and libre_de_video:
            self._controle_memoire(nom, self.ssd_streaming, params.get("lora_memoire"))
        if lora is not None:
            base = self.cat["modeles"][nom]["chemin"]
            if self.trouver_fusion(base, self.cat["lora"][lora["nom"]]["chemin"], lora["force"]) is None:
                self._controle_disque(base, lora["nom"], lora["force"], evincer=False)
        id_ = "video_" + uuid.uuid4().hex[:16]
        for cle in ("first_frame", "last_frame"):
            if corps.get(cle) is not None:
                params[cle] = self._ecrire_image(id_, cle, corps[cle])
        with self.verrou:
            self.numero += 1
            t = Travail(id_, params, self.numero)
            self.travaux[id_] = t
            self._ecrire(t)
        self.file.put(id_)
        return t

    def _lora(self, corps: dict, modele: str) -> list[dict]:
        brut = corps.get("lora")
        if brut in (None, []):
            return []
        if not isinstance(brut, list) or not all(isinstance(x, dict) for x in brut):
            raise ErreurDemande(400, "lora: list expected, e.g. [{\"nom\": \"...\", \"force\": 1.0}]")
        if len(brut) > LORA_MAX:
            raise ErreurDemande(400, f"lora: at most {LORA_MAX} video LoRA at a time")
        sortie: list[dict] = []
        for x in brut:
            nom, force = x.get("nom"), x.get("force")
            lo = self.cat["lora"].get(nom) if isinstance(nom, str) else None
            if lo is None:
                raise ErreurDemande(400, f"LoRA {nom!r} hors catalogue (disponibles : {', '.join(self.cat['lora']) or 'aucun'})")
            if lo.get("base") and modele not in lo["base"]:
                raise ErreurDemande(400, f"LoRA {nom}: reserved for {', '.join(lo['base'])}")
            if any(s["nom"] == nom for s in sortie):
                raise ErreurDemande(400, f"lora {nom}: named twice")
            if force is None:
                force = lo["force"]
            elif isinstance(force, bool) or not isinstance(force, (int, float)) or not -10 <= force <= 10:
                raise ErreurDemande(400, f"lora {nom}: « force » must be a number between -10 and 10")
            sortie.append({"nom": nom, "force": float(force)})
        modes = {self.cat["lora"][s["nom"]]["mode"] for s in sortie}
        if len(sortie) > 1 and modes != {"en_memoire"}:
            fusion = ", ".join(s["nom"] for s in sortie if self.cat["lora"][s["nom"]]["mode"] != "en_memoire")
            raise ErreurDemande(400, f"lora: several LoRA at once only in memory; merged on disk, one at a time: {fusion}")
        return sortie

    def _controle_memoire(self, modele: str, streaming: bool, loras: list[dict] | None = None) -> None:
        besoin = self._besoin_octets(modele, streaming, loras)
        try:
            libre = int(self.memoire_disponible())
        except Exception:  # noqa: BLE001 - read failed: do not reject
            return
        if libre < besoin:
            raise ErreurDemande(503, f"insufficient memory for video {modele} ({'streaming' if streaming else 'resident'}) :"
                                     f" {libre / GO:.1f} GB available, {besoin / GO:.1f} GB required", "memoire")

    @staticmethod
    def _entier(corps: dict, cle: str, bas: int, haut: int) -> int:
        v = corps[cle]
        if isinstance(v, bool) or not isinstance(v, (int, float)) or int(v) != v or not bas <= int(v) <= haut:
            raise ErreurDemande(400, f"« {cle} » must be an integer between {bas} and {haut}")
        return int(v)

    def _ecrire_image(self, id_: str, cle: str, valeur) -> str:
        if not isinstance(valeur, str):
            raise ErreurDemande(400, f"« {cle} » must be base64")
        if valeur.startswith("data:"):
            valeur = valeur.partition(",")[2]
        try:
            octets = base64.b64decode(valeur, validate=True)
        except (binascii.Error, ValueError):
            raise ErreurDemande(400, f"« {cle} » is not valid base64") from None
        if not octets:
            raise ErreurDemande(400, f"« {cle} » is empty")
        chemin = self._chemin(id_, f".{cle}")
        with open(chemin, "wb") as f:
            f.write(octets)
        return chemin

    # lecture

    def occupe(self) -> bool:
        with self.verrou:
            return any(t.status in ("queued", "in_progress") for t in self.travaux.values())

    def vue(self, t: Travail) -> dict:
        with self.verrou:
            position = None
            if t.status == "queued":
                position = 1 + sum(1 for u in self.travaux.values() if u.status == "queued" and u.numero < t.numero)
            if t.status == "queued" and t.attente is None and self.courant is not None:
                return {**t.vue(position), "attente": "file"}
            return t.vue(position)

    # annulation

    def annuler(self, id_: str | None = None) -> bool:
        """Cancels job id_ (in queue or running) or, without id_, the running one. False if nothing is cancelled."""

        with self.verrou:
            t = self.courant if id_ is None else self.travaux.get(id_)
            if t is None or t.status in FINIS:
                return False
            t.annule = True
            if t.status == "queued":
                t.status, t.etape, t.error, t.error_type = "cancelled", None, "cancelled", "annulee"
                t.completed_at = int(time.time())
                self._ecrire(t)
                return True
            courant = self.courant is t
        with self.verrou_proc:
            proc = self.proc if courant else None
        if proc is not None:
            self._terminer(proc)
        fin = time.monotonic() + 5.0                     # the worker thread writes « cancelled » just after
        while t.status not in FINIS and time.monotonic() < fin and self.fil.is_alive():
            time.sleep(0.01)
        return True

    def supprimer(self, id_: str) -> bool:
        with self.verrou:
            t = self.travaux.get(id_)
            if t is None or t.status not in FINIS:
                return False
            del self.travaux[id_]
        for ext in (".json", ".mp4", ".first_frame", ".last_frame"):
            try:
                os.unlink(self._chemin(id_, ext))
            except FileNotFoundError:
                pass
        return True

    # execution

    def _commande(self, t: Travail, dossier_modele: str) -> list[str]:
        p = t.params
        argv = [sys.executable, self.faux] if self.faux else [self.h3]
        argv += ["-d", dossier_modele, "-p", p["prompt"], "-o", self._chemin(t.id, ".mp4")]
        if "size" in p:
            largeur, hauteur = p["size"].split("x")
            argv += ["--width", largeur, "--height", hauteur]
        if "seconds" in p:
            argv += ["--seconds", f"{p['seconds']:g}"]
        if "frames" in p:
            argv += ["--frames", str(p["frames"])]
        for cle, option in (("steps", "--steps"), ("seed", "--seed"), ("reuse", "--reuse"), ("layers", "--layers"),
                            ("first_frame", "--first-frame"), ("last_frame", "--last-frame")):
            if cle in p:
                argv += [option, str(p[cle])]
        if p.get("ssd_streaming", self.ssd_streaming):
            argv.append("--ssd-streaming")                  # h3.c main.c:270, 357
        for fichier, force in getattr(t, "lora_fichiers", []) or []:
            argv += ["--lora", f"{fichier}@{force:g}"]      # h3 with tools/h3c-lora.patch (main.c OPT_LORA)
        return argv

    def _natif(self, t: Travail, base: str, nom: str) -> str:
        """Converted file of a LoRA in memory, converted now if absent (tools/pli-lora-natif, a few seconds)."""

        cible = self._chemin_natif(nom)
        if not cible:
            raise ErreurDemande(503, f"LoRA {nom}: source file unreadable or « lora_natif.dossier » missing", "lora")
        if os.path.exists(cible):
            return cible
        ln = self.cat["lora_natif"]
        if ln["script"]:
            argv = [ln["python"], ln["script"]]
        elif self.faux_natif:
            argv = [sys.executable, self.faux_natif]
        else:
            raise ErreurDemande(503, "LoRA in memory: « lora_natif.script » missing from the catalog", "lora")
        os.makedirs(os.path.dirname(cible), exist_ok=True)
        partiel = cible + ".tmp"
        _effacer(partiel)
        code, fin = self._suivre(t, argv + [base, self.cat["lora"][nom]["chemin"], partiel], {},
                                 stdout=subprocess.PIPE, progres=False)
        if code != 0 or self.arret.is_set() or t.annule or not os.path.exists(partiel):
            _effacer(partiel)
            if self.arret.is_set() or t.annule:
                raise ErreurDemande(499, "cancelled", "annulee")
            raise ErreurDemande(500, fin if code is None else f"conversion of LoRA {nom} failed (code {code})"
                                + (f" : {fin}" if fin and code is not None else ""), "lora")
        os.replace(partiel, cible)
        return cible

    def _libre(self) -> str | None:
        """Reason not to start: « general » (PLI_GEN_DRAPEAU), « image_ou_voix » (PLI_GEN_DRAPEAU_AUTRES)."""

        if self.drapeau and os.path.exists(self.drapeau):
            return "general"
        if self.drapeau_autres and os.path.exists(self.drapeau_autres):
            return "image_ou_voix"
        return None

    def _manque_voix(self, t: Travail) -> int:
        """Bytes missing to start t while keeping space for the voice; 0 if enough, or no voice reserve."""

        if not self.voix_octets or self.memoire_disponible is None:
            return 0
        p = t.params
        besoin = self._besoin_octets(p["modele"], p.get("ssd_streaming", self.ssd_streaming), p.get("lora_memoire"))
        charge = bool(self.drapeau_audio) and os.path.exists(self.drapeau_audio)
        besoin += self.voix_reserve_octets if charge else self.voix_octets
        try:
            libre = int(self.memoire_disponible())
        except Exception:  # noqa: BLE001 - read failed: do not hold (like _controle_memoire)
            return 0
        return max(0, besoin - libre)

    def _ecrire_besoin(self, t: Travail, manque: int) -> None:
        if not self.fichier_besoin:
            return
        if manque <= 0:
            return _effacer(self.fichier_besoin)
        tmp = self.fichier_besoin + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({"octets": int(manque), "id": t.id}, f)
        os.replace(tmp, self.fichier_besoin)

    def _attendre_libre(self, t: Travail) -> str | None:
        """Waits until nothing holds t (general, image or voice, voice memory). Returns the failure message if the
        voice memory does not arrive within attente_memoire_max_s, else None."""

        debut = time.monotonic()
        debut_memoire: float | None = None
        echec = None
        try:
            while not self.arret.is_set() and not t.annule and time.monotonic() - debut < ATTENTE_DRAPEAU_MAX_S:
                raison = self._libre()
                if raison is None:
                    manque = self._manque_voix(t)
                    self._ecrire_besoin(t, manque)
                    if manque > 0:
                        raison = "memoire_voix"
                        if debut_memoire is None:
                            debut_memoire = time.monotonic()
                        elif time.monotonic() - debut_memoire > self.attente_memoire_max_s:
                            echec = (f"insufficient memory for video alongside the voice after "
                                     f"{self.attente_memoire_max_s:g} s d'attente : il manque {manque / GO:.1f} Go")
                            break
                if raison is None:
                    break
                if t.attente != raison:
                    with self.verrou:
                        t.attente = raison
                        self._ecrire(t)
                time.sleep(PAS_MEMOIRE_S if raison == "memoire_voix" else PAS_DRAPEAU_S)
        finally:
            self._ecrire_besoin(t, 0)
        with self.verrou:
            t.attente = None
        return echec

    def _boucle(self) -> None:
        while not self.arret.is_set():
            id_ = self.file.get()
            if id_ is None or self.arret.is_set():
                return
            t = self.travaux.get(id_)
            if t is None or t.status != "queued":
                continue                                   # cancelled or deleted while waiting
            echec = self._attendre_libre(t)
            if self.arret.is_set():
                self._finir(t, "failed", "interrupted by the stop of the video child")
                return
            if t.annule:
                continue
            if echec is not None:
                self._finir(t, "failed", echec, "memoire")
                continue
            with self.verrou:
                if t.status != "queued":
                    continue
                self.courant = t
                t.status, t.etape = "in_progress", ("fusion" if t.params.get("lora") else
                                                    "lora" if t.params.get("lora_memoire") else "rendu")
                self._ecrire(t)
            _poser(self.drapeau_video, True)                # the supervisor reduces general parallelism
            try:
                self._executer(t)
            except Exception as exc:  # noqa: BLE001 - the worker thread does not die on an unexpected failure
                self._finir(t, "failed", f"{type(exc).__name__} : {exc}")
            finally:
                with self.verrou:
                    self.courant = None
                _poser(self.drapeau_video, False)

    def _finir(self, t: Travail, statut: str, erreur: str | None = None, type_: str | None = None) -> None:
        with self.verrou:
            t.status, t.error, t.etape, t.en_pause = statut, erreur, None, False
            t.error_type = type_ if erreur else None
            if statut == "completed":
                t.progress = 1.0
            t.completed_at = int(time.time())
            self._ecrire(t)

    def _lire_progres(self, t: Travail, texte: str) -> None:
        m = RE_PROGRES.match(texte.strip())
        if not m:
            return
        phase, fait, total = m["phase"].strip(), int(m["fait"]), int(m["total"])
        with self.verrou:
            t.phase = {"nom": phase, "fait": fait, "total": total}
            if phase == "denoise" and total > 0:
                t.progress = min(1.0, fait / total) * 0.999       # 1.0 only at the end of the job

    def _fin_si_interrompu(self, t: Travail) -> bool:
        if self.arret.is_set():
            self._finir(t, "failed", "interrupted by the stop of the video child", "arret")
            return True
        if t.annule:
            self._finir(t, "cancelled", "cancelled", "annulee")
            return True
        return False

    def _executer(self, t: Travail) -> None:
        p = t.params
        dossier = self.cat["modeles"][p["modele"]]["chemin"]
        if self.memoire_disponible is not None:
            try:
                self._controle_memoire(p["modele"], p.get("ssd_streaming", self.ssd_streaming), p.get("lora_memoire"))
            except ErreurDemande as exc:
                return self._finir(t, "failed", exc.message, exc.type)
        t.lora_fichiers = []
        if p.get("lora_memoire"):
            debut = time.monotonic()
            with self.verrou:
                t.etape, t.phase = "lora", {"nom": "lora", "fait": 0, "total": len(p["lora_memoire"])}
                self._ecrire(t)
            try:
                for i, x in enumerate(p["lora_memoire"]):
                    t.lora_fichiers.append((self._natif(t, dossier, x["nom"]), x["force"]))
                    with self.verrou:
                        t.phase = {"nom": "lora", "fait": i + 1, "total": len(p["lora_memoire"])}
            except ErreurDemande as exc:
                t.secondes_fusion = round(time.monotonic() - debut, 2)
                if not self._fin_si_interrompu(t):
                    self._finir(t, "failed", exc.message, exc.type)
                return
            t.secondes_fusion = round(time.monotonic() - debut, 2)
            t.fusion = {"mode": "en_memoire", "fichiers": [f for f, _ in t.lora_fichiers]}
            if self._fin_si_interrompu(t):
                return
            with self.verrou:
                t.etape, t.phase = "rendu", None
                self._ecrire(t)
        if p.get("lora"):
            debut = time.monotonic()
            with self.verrou:
                t.etape, t.phase = "fusion", {"nom": "fusion", "fait": 0, "total": 0}
                self._ecrire(t)
            erreur = None
            try:
                dossier = self._fusion(t, dossier)
            except ErreurDemande as exc:
                erreur = exc
            t.secondes_fusion = round(time.monotonic() - debut, 2)
            if erreur is not None:
                if not self._fin_si_interrompu(t):
                    self._finir(t, "failed", erreur.message, erreur.type)
                return
            if self._fin_si_interrompu(t):
                return
            with self.verrou:
                t.etape, t.phase = "rendu", None
                self._ecrire(t)
        debut = time.monotonic()
        options = lancement_h3(self.h3) if not self.faux else {}
        code, fin = self._suivre(t, self._commande(t, dossier), options, stdout=subprocess.DEVNULL, progres=True)
        t.secondes_rendu = round(time.monotonic() - debut, 2)
        if code is None:
            return self._finir(t, "failed", fin, "lancement")
        if self._fin_si_interrompu(t):
            return
        if code == 0 and os.path.exists(self._chemin(t.id, ".mp4")):
            self._finir(t, "completed")
        else:
            self._finir(t, "failed", f"h3 exited with code {code}" + (f" : {fin}" if fin else ""), "h3")

    def _fusion(self, t: Travail, base: str) -> str:
        """Merged model directory: existing merge (saved or cache), otherwise merge done now."""

        nom, force = t.params["lora"]["nom"], t.params["lora"]["force"]
        lo = self.cat["lora"][nom]
        existante = self.trouver_fusion(base, lo["chemin"], force)
        if existante is not None:
            source = "cache" if self._dans_cache(existante) else "sauvegardee"
            if source == "cache":
                self._marquer_usage(existante)
            t.fusion = {"chemin": existante, "source": source, "mode": lo["mode"]}
            return existante
        dest = self._controle_disque(base, nom, force, evincer=True)
        partiel = dest + ".partiel"
        shutil.rmtree(partiel, ignore_errors=True)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        if self.fausse_fusion and not self.cat["fusion"]["script"]:
            argv = [sys.executable, self.fausse_fusion]
        elif self.cat["fusion"]["script"]:
            argv = [self.cat["fusion"]["python"], self.cat["fusion"]["script"]]
        else:
            raise ErreurDemande(503, "LoRA merge not configured (« fusion.script » missing from the catalog)", "fusion")
        argv += [base, partiel, "--lora", f"{lo['chemin']}@{force:g}"]
        t.fusion = {"chemin": dest, "source": "nouvelle", "mode": lo["mode"]}
        code, fin = self._suivre(t, argv, {}, stdout=subprocess.PIPE, progres=False)
        if code != 0 or self.arret.is_set() or t.annule or lire_meta_fusion(partiel) is None:
            shutil.rmtree(partiel, ignore_errors=True)
            if self.arret.is_set() or t.annule:
                raise ErreurDemande(499, "cancelled", "annulee")
            raise ErreurDemande(500, fin if code is None else f"merge of {nom} failed (code {code})"
                                + (f" : {fin}" if fin and code is not None else ""), "fusion")
        if os.path.exists(dest):
            shutil.rmtree(dest, ignore_errors=True)
        os.rename(partiel, dest)
        if self._dans_cache(dest):
            self._marquer_usage(dest)
        return dest

    @staticmethod
    def _marquer_usage(dossier: str) -> None:
        with open(os.path.join(dossier, MARQUE_USAGE), "w") as f:
            f.write(str(time.time()))
        os.utime(os.path.join(dossier, MARQUE_USAGE))

    def _suivre(self, t: Travail, argv: list[str], options: dict, *, stdout, progres: bool) -> tuple[int | None, str]:
        """Launches argv in its own group, reads its error output (and stdout if stdout=PIPE), pauses it
        while the general flag is set. (code, last 3 lines); code None: launch impossible (message)."""

        if self.arret.is_set() or t.annule:
            return -1, ""
        try:
            proc = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stdout,
                                    stderr=subprocess.STDOUT if stdout == subprocess.PIPE else subprocess.PIPE,
                                    start_new_session=True, **options)
        except OSError as exc:
            return None, f"{os.path.basename(argv[0])} not found or not launchable: {exc}"
        with self.verrou_proc:
            self.proc = proc
            tardif = self.arret.is_set() or t.annule      # cancelled between the check and the launch
        if tardif:
            self._terminer(proc)
        fin_suivi = threading.Event()
        veille = threading.Thread(target=self._surveiller_pause, args=(t, proc, fin_suivi), daemon=True)
        veille.start()
        flux = proc.stdout if stdout == subprocess.PIPE else proc.stderr
        queue_sortie: list[str] = []
        reste = ""
        derniere_ecriture = 0.0
        try:
            while True:
                morceau = os.read(flux.fileno(), 4096)
                if not morceau:
                    break
                reste += morceau.decode("utf-8", "replace")
                *lignes, reste = re.split(r"[\r\n]", reste)
                for ligne in lignes:
                    if ligne.strip():
                        queue_sortie = (queue_sortie + [ligne.strip()])[-10:]
                        if progres:
                            self._lire_progres(t, ligne)
                if progres and reste.strip():
                    self._lire_progres(t, reste)          # h3 only writes the newline at the end of the phase
                if time.monotonic() - derniere_ecriture > 1.0:
                    derniere_ecriture = time.monotonic()
                    with self.verrou:
                        self._ecrire(t)
            code = proc.wait()
        finally:
            fin_suivi.set()
            veille.join(5)
            with self.verrou_proc:
                self.proc = None
        return code, " | ".join(queue_sortie[-3:])

    def _surveiller_pause(self, t: Travail, proc: subprocess.Popen, fin: threading.Event) -> None:
        """SIGSTOP to the process group while the general flag (or the voice one, PLI_GEN_DRAPEAU_VOIX)
        exists, SIGCONT when it disappears."""

        debut_pause = 0.0
        while not fin.wait(PAS_DRAPEAU_S):
            voulu = ((bool(self.drapeau) and os.path.exists(self.drapeau))
                     or (bool(self.drapeau_voix) and os.path.exists(self.drapeau_voix)))
            with self.verrou_proc:
                if proc.poll() is not None:
                    break
                if self.arret.is_set() or t.annule:
                    voulu = False                           # the shutdown itself resumes the process
                if voulu and not t.en_pause:
                    if self._signal(proc, signal.SIGSTOP):
                        t.en_pause, t.pauses, debut_pause = True, t.pauses + 1, time.monotonic()
                elif not voulu and t.en_pause:
                    self._signal(proc, signal.SIGCONT)
                    t.en_pause = False
                    t.secondes_pause += time.monotonic() - debut_pause
        with self.verrou_proc:
            if t.en_pause:
                self._signal(proc, signal.SIGCONT)
                t.en_pause = False
                t.secondes_pause += time.monotonic() - debut_pause

    @staticmethod
    def _signal(proc: subprocess.Popen, sig: int) -> bool:
        try:
            os.killpg(proc.pid, sig)                    # start_new_session: the process is the leader of its group
            return True
        except OSError:
            return False

    def _terminer(self, proc: subprocess.Popen) -> None:
        """Resumes the process (SIGCONT, a stopped process does not handle SIGTERM), then SIGTERM, then SIGKILL."""

        for sig, delai in ((signal.SIGTERM, 10.0), (signal.SIGKILL, 5.0)):
            with self.verrou_proc:
                if proc.poll() is not None:
                    return
                self._signal(proc, signal.SIGCONT)
                if not self._signal(proc, sig):
                    return
            try:
                proc.wait(timeout=delai)
                return
            except subprocess.TimeoutExpired:
                continue

    def arreter(self) -> None:
        """Clean SIGTERM: the current process first (resumed if it is paused), then the queue."""

        with self.verrou_proc:
            self.arret.set()
            proc = self.proc
        self.file.put(None)
        if proc is not None:
            self._terminer(proc)
        self.fil.join(15)
        _effacer(self.drapeau_video)
        _effacer(self.fichier_besoin)


def faire_serveur(port: int, moteur: Moteur) -> ThreadingHTTPServer:
    class Gestionnaire(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"

        def log_message(self, format, *args):
            pass

        def _envoyer(self, code: int, corps: bytes, type_: str = "application/json") -> None:
            self.send_response(code)
            self.send_header("Content-Type", type_)
            self.send_header("Content-Length", str(len(corps)))
            self.end_headers()
            self.wfile.write(corps)

        def _json(self, code: int, charge: dict) -> None:
            self._envoyer(code, json.dumps(charge, ensure_ascii=False).encode())

        def _erreur(self, code: int, type_: str, message: str) -> None:
            self._json(code, {"error": {"type": type_, "message": message}})

        def _travail(self, chemin: str):
            m = re.match(r"^/v1/videos/([^/]+)(/content)?$", chemin)
            if not m:
                self._erreur(404, "inconnu", f"chemin inconnu : {chemin}")
                return None, None
            t = moteur.travaux.get(m.group(1)) if RE_ID.match(m.group(1)) else None
            if t is None:
                self._erreur(404, "travail_inconnu", f"unknown video: {m.group(1)}")
                return None, None
            return t, m.group(2)

        def do_GET(self) -> None:
            chemin = self.path.split("?", 1)[0].rstrip("/")
            if chemin == "/sante":
                return self._json(200, {"ok": True, "occupe": moteur.occupe(), "ssd_streaming": moteur.ssd_streaming})
            t, contenu = self._travail(chemin)
            if t is None:
                return
            if not contenu:
                return self._json(200, moteur.vue(t))
            if t.status != "completed":
                return self._erreur(409, "pas_pret", f"the video is « {t.status} »")
            try:
                with open(moteur._chemin(t.id, ".mp4"), "rb") as f:
                    donnees = f.read()
            except OSError:
                return self._erreur(410, "fichier_absent", "the MP4 file no longer exists")
            self._envoyer(200, donnees, "video/mp4")

        def do_DELETE(self) -> None:
            longueur = int(self.headers.get("Content-Length") or 0)
            if longueur:
                self.rfile.read(min(longueur, LIMITE_CORPS))
            t, contenu = self._travail(self.path.split("?", 1)[0].rstrip("/"))
            if t is None:
                return
            if contenu:
                return self._erreur(405, "methode", "DELETE applies to /v1/videos/{id}")
            if t.status in FINIS:
                moteur.supprimer(t.id)
                return self._json(200, {"id": t.id, "object": "video.deleted", "deleted": True})
            moteur.annuler(t.id)
            self._json(200, moteur.vue(t))

        def do_POST(self) -> None:
            longueur = int(self.headers.get("Content-Length") or 0)
            if longueur > LIMITE_CORPS:
                self.close_connection = True
                return self._erreur(413, "trop_gros", "corps trop gros")
            brut = self.rfile.read(longueur) if longueur else b""
            chemin = self.path.split("?", 1)[0].rstrip("/")
            if chemin == "/annuler":
                return self._json(200, {"annulee": moteur.annuler()})
            if chemin != "/v1/videos":
                return self._erreur(404, "inconnu", f"chemin inconnu : {self.path}")
            try:
                corps = json.loads(brut or b"{}")
                if not isinstance(corps, dict):
                    raise ValueError
            except ValueError:
                return self._erreur(400, "invalid_request_error", "the body must be a JSON object")
            try:
                t = moteur.creer(corps, general_charge=self.headers.get("X-PLI-General", "") == "charge")
            except ErreurDemande as exc:
                return self._erreur(exc.code, exc.type, exc.message)
            self._json(202, moteur.vue(t))

    serveur = ThreadingHTTPServer(("127.0.0.1", port), Gestionnaire)
    serveur.daemon_threads = True
    return serveur


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--catalogue", default="")
    p.add_argument("--h3", default="")
    p.add_argument("--modele", default="")
    p.add_argument("--sorties", required=True)
    p.add_argument("--ssd-streaming", choices=("oui", "non"), default=None,
                   help="replaces « ssd_streaming » from the catalog (catalog default: yes)")
    p.add_argument("--memoire-disponible-go", type=float, default=None,
                   help="imposed available memory (tests); otherwise vm_stat outside --simuler, no check in --simuler")
    p.add_argument("--simuler", action="store_true")
    a = p.parse_args()
    if a.catalogue:
        cat = charger_catalogue(a.catalogue)
    else:
        if not a.simuler and not (a.h3 and a.modele):
            p.error("--catalogue, or --h3 and --modele, are required outside of --simuler")
        cat = catalogue_simple(a.h3, a.modele)
    if not a.simuler and not cat["h3"]:
        p.error("video catalogue: « h3 » (path to the executable) is required outside of --simuler")
    memoire = (lambda: int(a.memoire_disponible_go * GO)) if a.memoire_disponible_go is not None else None
    moteur = Moteur("", "", a.sorties, a.simuler, catalogue=cat,
                    ssd_streaming=None if a.ssd_streaming is None else a.ssd_streaming == "oui",
                    memoire_disponible=memoire)
    serveur = faire_serveur(a.port, moteur)

    def fin(signum, frame):
        threading.Thread(target=serveur.shutdown, daemon=True).start()

    signal.signal(signal.SIGTERM, fin)
    signal.signal(signal.SIGINT, fin)
    try:
        serveur.serve_forever()
    finally:
        moteur.arreter()
        serveur.server_close()


if __name__ == "__main__":
    main()
