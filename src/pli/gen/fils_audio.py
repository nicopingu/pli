"""PLI 5 audio child process: voice synthesis (OpenAI /v1/audio/speech) and transcription (/v1/audio/transcriptions).

    VENV_MLX_AUDIO/bin/python fils_audio.py --port N [--voix catalogue.json] [--simuler]

Standalone file: standard library, numpy and mlx_audio (imported only outside --simuler). It does not import
pli, because the mlx-audio venv ships a different mlx than the server's.

Voices: a JSON catalog name -> {"modele": "drama3" | "kokoro" | "qwen3tts", "repo": ..., ...}
  drama3   : "references": [[wav, txt], ...]  (cloning by references: a recording and its text)
  kokoro   : "voice", "lang_code"
  qwen3tts : "repo" required, "voice" and "lang_code", or "references" (a single pair)
Only one synthesis model is loaded at a time; the previous one is unloaded upon change. The transcription
(Qwen3-ASR 0.6B) has its own model, lightweight, kept in memory once loaded.

Memory: "memoire_go" per voice (default per model: MEMOIRE_GO_DEFAUT). Before loading a model, the child compares the
available memory (vm_stat) to this need and responds 503 {"error": {"type": "memoire"}} without loading. The buffer cache
cache is limited (--cache-limit-go, default 2) and cleared after each load and each generation.

Preemption: as long as the PLI_GEN_DRAPEAU file exists, the child waits between produced chunks (Kokoro, Qwen3-TTS) and
all FRAMES_ENTRE_ATTENTES frames of Drama 3. The number of wait steps is returned in the X-PLI-Attentes header.

A single MLX thread: loading, generation, transcription and unloading all run on a single-threaded executor
(Moteur.executer). MLX streams belong to the thread that creates them: a model loaded in one request's HTTP thread and reused
in another's raises "There is no Stream(cpu, N) in current thread". HTTP threads wait for the result;
two requests (including voice and transcription) therefore pass one after the other.

Cancellation: POST /annuler (empty body) cancels the request currently running. The child sees it at the next wait
(between two chunks or frames, during the flag wait, between two loading batches) and responds 499
{"error": {"type": "annulee"}} to that request; /annuler responds 200 {"annulee": true|false}. 499 ("client closed", usage of
nginx) rather than 409: the caller gave up, there is no state conflict. Transcription is interrupted only before
loading and before generate: a generate call is a single block of mlx_audio.
"""

from __future__ import annotations

import argparse
import gc
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import wave
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

DEPOTS = {
    "drama3": "mlx-community/fish-audio-drama-3-4bit",
    "kokoro": "mlx-community/Kokoro-82M-bf16",
}
MODELES_TTS = ("drama3", "kokoro", "qwen3tts")
ASR_DEFAUT = "mlx-community/Qwen3-ASR-0.6B-8bit"
# Default voice in code: Kokoro only (no cloned voice, no machine path). The voice catalog (--voix)
# chooses its own by its "defaut" key: {"drama3": NOM, "kokoro": NOM, "*": NOM}; "*" is used when the request
# names neither a voice nor a known synthesis model. A voice can keep old names: "alias": [ANCIEN, ...].
VOIX_DEFAUT = {"kokoro": "kokoro-siwis", "*": "kokoro-siwis"}
CATALOGUE_DEFAUT = {
    "kokoro-siwis": {"modele": "kokoro", "voice": "ff_siwis", "lang_code": "f"},
}
# Qwen3-ASR compares the language (case-insensitive) to its full names (qwen3_asr.py:935-940): ISO codes are translated here.
LANGUES = {
    "fr": "French", "en": "English", "de": "German", "es": "Spanish", "it": "Italian", "pt": "Portuguese",
    "nl": "Dutch", "ru": "Russian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean", "ar": "Arabic",
    "tr": "Turkish", "hi": "Hindi", "sv": "Swedish", "da": "Danish", "fi": "Finnish", "pl": "Polish",
    "cs": "Czech", "el": "Greek", "ro": "Romanian", "hu": "Hungarian", "id": "Indonesian", "th": "Thai",
    "vi": "Vietnamese", "ms": "Malay", "fa": "Persian",
}
LIMITE_CORPS = 256 * 1024**2
GO = 1024**3
# memory requirement (Go) of the model, measured on an M1 Ultra 64 GB; "memoire_go" in the voice replaces it.
# Drama 3 4 bits: 22.4 GiB of parameters (backbone Qwen3.5-MoE 19.3, tower 1.4, codec 1.75), MLX active 22.4 after
# loading, 23.4 to 24 with transcription, peak 27.5 during a voice, cache 0 (gen5-maudio1 of 11/14/2026): 24.
MEMOIRE_GO_DEFAUT = {"drama3": 24.0, "kokoro": 3.0, "qwen3tts": 7.0}
CACHE_LIMITE_GO_DEFAUT = 2.0
# First load: weights evaluated in chunks of this size (Mo) instead of a full model mx.eval
# (mlx_audio/utils.py:449-450, 20 GB at once for Drama 3) ; 0 = original loading.
MORCEAU_CHARGEMENT_MO_DEFAUT = 1024
ATTENTE_MAX_S = 600.0                 # a flag never removed does not block the child process forever
PAS_ATTENTE_S = 0.02
FRAMES_ENTRE_ATTENTES = 8             # Drama 3: a flag wait every 8 frames (about 0.36 s of audio: 22 frames/s measured on 15/11/2026)


def evaluer_par_morceaux(mx, modele, morceau_octets: int, entre=None) -> int:
    """Evaluates the model's lazy parameters in batches of at most morceau_octets, cache cleared after each batch.

    mx.load is lazy (MLX 0.32: nothing is read before evaluation): load_model(lazy=True) reads no weights,
    and the loading peak no longer exceeds the model plus one batch. To be called in the thread that loaded the model
    (lazy arrays are bound to its stream). entre() is called between two batches (wait for the general's flag).
    Returns the number of batches."""

    from mlx.utils import tree_flatten
    lot: list = []
    taille = n = 0
    for _, p in tree_flatten(modele.parameters()):
        lot.append(p)
        taille += p.nbytes
        if taille >= morceau_octets:
            mx.eval(*lot)
            lot, taille, n = [], 0, n + 1
            mx.clear_cache()
            if entre is not None:
                entre()
    if lot:
        mx.eval(*lot)
        n += 1
        mx.clear_cache()
    return n


class ErreurClient(Exception):
    """Invalid request: response 400."""


class Annulee(Exception):
    """The current request was cancelled by POST /annuler: response 499."""


class ErreurMemoire(Exception):
    """Available memory lower than the model's need: response 503, nothing is loaded."""


# ---------------------------------------------------------------- memory and flag (copied from superviseur.py: the child is standalone)

def analyser_vm_stat(texte: str) -> int:
    """Recoverable memory in bytes according to vm_stat: free + inactive + speculative + purgeable pages."""

    taille = re.search(r"page size of (\d+) bytes", texte)
    if not taille:
        raise ValueError("taille de page absente de vm_stat")
    pages = 0
    for ligne in texte.splitlines():
        m = re.match(r"\s*Pages (free|inactive|speculative|purgeable):\s+(\d+)\.?\s*$", ligne)
        if m:
            pages += int(m.group(2))
    return pages * int(taille.group(1))


def memoire_disponible_systeme() -> int:
    if shutil.which("vm_stat"):
        sortie = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10, check=True,
                                env={**os.environ, "LC_ALL": "C"}).stdout
        return analyser_vm_stat(sortie)
    with open("/proc/meminfo") as f:
        for ligne in f:
            if ligne.startswith("MemAvailable:"):
                return int(ligne.split()[1]) * 1024
    raise OSError("unavailable memory unreadable")


def attendre_drapeau(drapeau: str, *, max_s: float = ATTENTE_MAX_S, pas: float = PAS_ATTENTE_S, arret=None) -> int:
    """Waits as long as the flag file exists; returns the number of wait steps (0 if the path is free).
    arret() vrai interrompt l'attente."""

    if not drapeau:
        return 0
    debut, n = time.monotonic(), 0
    while os.path.exists(drapeau) and time.monotonic() - debut < max_s:
        if arret is not None and arret():
            break
        n += 1
        time.sleep(pas)
    return n


# ---------------------------------------------------------------- multipart analysis

def analyser_multipart(type_contenu: str, corps: bytes) -> list[dict]:
    """Parts of a multipart/form-data body: [{"nom", "fichier" (name or None), "type", "donnees"}].

    Written by hand (the cgi module no longer exists in Python 3.13). The content of the parts remains in bytes.
    """

    m = re.search(r'boundary=(?:"([^"]+)"|([^;\s]+))', type_contenu, re.I)
    if not m or "multipart/" not in type_contenu.lower():
        raise ErreurClient("corps multipart/form-data attendu")
    sep = b"--" + (m.group(1) or m.group(2)).encode("latin-1")
    # the delimiter is followed by « -- » (end) or a line break: a content starting the same way does not cut
    morceaux = re.split(rb"\r\n" + re.escape(sep) + rb"(?=--|[ \t]*\r\n)", b"\r\n" + corps)
    parties = []
    for morceau in morceaux[1:]:
        if morceau.startswith(b"--"):            # final delimiter
            break
        morceau = re.sub(rb"^[ \t]+", b"", morceau)
        if not morceau.startswith(b"\r\n"):
            raise ErreurClient("malformed multipart")
        entete, trouve, donnees = morceau[2:].partition(b"\r\n\r\n")
        if not trouve:
            raise ErreurClient("malformed multipart: part without end of headers")
        champs: dict[str, str] = {}
        for ligne in entete.decode("utf-8", "replace").split("\r\n"):
            cle, _, val = ligne.partition(":")
            champs[cle.strip().lower()] = val.strip()
        dispo = champs.get("content-disposition", "")
        nom = re.search(r'\bname="([^"]*)"', dispo)
        fichier = re.search(r'\bfilename="([^"]*)"', dispo)
        parties.append({"nom": nom.group(1) if nom else "", "fichier": fichier.group(1) if fichier else None,
                        "type": champs.get("content-type", ""), "donnees": donnees})
    return parties


# ---------------------------------------------------------------- WAV

def encoder_wav(audio: np.ndarray, taux: int) -> bytes:
    x = np.clip(np.asarray(audio, dtype=np.float32).reshape(-1), -1.0, 1.0)
    pcm = (x * 32767.0).astype("<i2")
    tampon = io.BytesIO()
    with wave.open(tampon, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(taux)
        w.writeframes(pcm.tobytes())
    return tampon.getvalue()


# ---------------------------------------------------------------- moteurs

class Moteur:
    """Common base: preemption by the flag, count of waits per request, memory check before loading."""

    simule = False

    def __init__(self, drapeau: str = "", memoire_disponible=None) -> None:
        self.drapeau = drapeau
        self.memoire_disponible = memoire_disponible or memoire_disponible_systeme
        self._local = threading.local()
        # a single thread for everything MLX does (streams bound to the thread); HTTP threads wait for the result
        self._executeur = ThreadPoolExecutor(max_workers=1, thread_name_prefix="mlx")
        self._verrou_annulation = threading.Lock()
        self._numero = 0
        self._courante: int | None = None     # number of the request currently running on the MLX thread
        self._annulee: int | None = None      # number cancelled by /annuler

    def debut_demande(self) -> None:
        self._local.attentes = 0

    def attentes_demande(self) -> int:
        return getattr(self._local, "attentes", 0)

    def executer(self, fonction, *args):
        """Executes fonction(*args) on the unique MLX thread and waits for its result (exceptions propagate as is).
        Flag waits counted per request are reported to the calling thread (attentes_demande)."""

        def travail():
            with self._verrou_annulation:
                self._numero += 1
                self._courante = self._numero
            self.debut_demande()
            try:
                return fonction(*args), self.attentes_demande()
            finally:
                with self._verrou_annulation:
                    self._courante = self._annulee = None

        resultat, attentes = self._executeur.submit(travail).result()
        self._local.attentes = attentes
        return resultat

    def annuler(self) -> bool:
        """Cancels the request currently running; false if there is none (already finished: nothing to do)."""

        with self._verrou_annulation:
            if self._courante is None:
                return False
            self._annulee = self._courante
            return True

    def annulation_demandee(self) -> bool:
        with self._verrou_annulation:
            return self._annulee is not None and self._annulee == self._courante

    def controler(self) -> None:
        if self.annulation_demandee():
            raise Annulee("request cancelled")

    def attendre(self) -> None:
        self.controler()
        self._local.attentes = self.attentes_demande() + attendre_drapeau(self.drapeau, arret=self.annulation_demandee)
        self.controler()

    def verifier_memoire(self, voix: dict) -> None:
        """Raises ErreurMemoire if available memory is below the model's requirement. An unreadable value does not refuse."""

        besoin = float(voix.get("memoire_go", MEMOIRE_GO_DEFAUT[voix["modele"]])) * GO
        try:
            libre = int(self.memoire_disponible())
        except Exception:  # noqa: BLE001
            return
        if libre < besoin:
            raise ErreurMemoire(f"insufficient memory to load {voix['modele']} : {libre / GO:.1f} GB available, "
                                f"{besoin / GO:.1f} GB required")


class MoteurSimule(Moteur):
    """Replaces mlx_audio : sine wave for the voice (produced in pieces), fixed text for transcription."""

    simule = True

    def __init__(self, drapeau: str = "", memoire_disponible=None, duree_morceau: float = 0.0,
                 duree_chargement: float = 0.0) -> None:
        super().__init__(drapeau, memoire_disponible or (lambda: 1 << 50))  # simulated : real memory does not count
        self.duree_morceau, self.duree_chargement = duree_morceau, duree_chargement
        self.cle_tts: str | None = None
        self.chargements = 0
        self.fils_utilises: list[int] = []        # threading.get_ident() of each job call (tests)
        self.en_cours = 0
        self.max_en_cours = 0

    def _entrer(self) -> None:
        self.fils_utilises.append(threading.get_ident())
        self.en_cours += 1
        self.max_en_cours = max(self.max_en_cours, self.en_cours)

    def synthese(self, voix: dict, texte: str, vitesse: float, options: dict):
        return self.executer(self._synthese, voix, texte, vitesse, options)

    def _synthese(self, voix: dict, texte: str, vitesse: float, options: dict):
        self._entrer()
        try:
            return self._synthese_simulee(voix, texte)
        finally:
            self.en_cours -= 1

    def _synthese_simulee(self, voix: dict, texte: str):
        if self.cle_tts != voix["modele"]:
            self.cle_tts = None                               # the previous one is unloaded before the measure, as in MoteurMlx
            self.verifier_memoire(voix)
            time.sleep(self.duree_chargement)
            self.cle_tts = voix["modele"]
            self.chargements += 1
        taux = 24000
        duree = min(5.0, max(0.2, 0.04 * len(texte)))
        freq = 220.0 + 40.0 * (sum(map(ord, voix["_nom"])) % 8)
        t = np.arange(int(duree * taux)) / taux
        n = max(1, len(texte) // 20)                          # one piece per 20-character chunk
        morceaux = []
        for partie in np.array_split(t, n):
            self.attendre()
            time.sleep(self.duree_morceau)
            morceaux.append((0.3 * np.sin(2 * np.pi * freq * partie)).astype(np.float32))
        return np.concatenate(morceaux), taux

    def transcription(self, chemin: str, langue: str | None) -> str:
        return self.executer(self._transcription, chemin, langue)

    def _transcription(self, chemin: str, langue: str | None) -> str:
        self._entrer()
        try:
            self.controler()
            time.sleep(self.duree_morceau)
            self.controler()
            return "simulated text"
        finally:
            self.en_cours -= 1


class MoteurMlx(Moteur):
    """Loads mlx_audio models on first request ; only one synthesis model at a time."""

    simule = False

    def __init__(self, depot_asr: str, drapeau: str = "", memoire_disponible=None,
                 cache_limit_go: float = CACHE_LIMITE_GO_DEFAUT,
                 morceau_chargement_mo: int = MORCEAU_CHARGEMENT_MO_DEFAUT) -> None:
        super().__init__(drapeau, memoire_disponible)
        self.depot_asr = depot_asr
        self.cache_limit_go = cache_limit_go
        self.morceau_chargement_mo = morceau_chargement_mo
        self.lots_chargement = 0
        self._limite_posee = False
        self.cle_tts: tuple | None = None
        self.modele_tts = None
        self.asr = None

    def _mx(self):
        """mlx.core, with the buffer cache limited before any loading (without a limit, it keeps tens of Go)."""

        import mlx.core as mx
        if not self._limite_posee:
            if self.cache_limit_go >= 0:
                mx.set_cache_limit(int(self.cache_limit_go * GO))
            self._limite_posee = True
        return mx

    def _vider(self) -> None:
        gc.collect()
        self._mx().clear_cache()

    # synthesis -------------------------------------------------
    def _depot(self, voix: dict) -> str:
        depot = voix.get("repo") or DEPOTS.get(voix["modele"])
        if not depot:
            raise ErreurClient(f"voice « {voix['_nom']} » : « repo » missing for model {voix['modele']}")
        return depot

    def decharger_tts(self) -> None:
        self.modele_tts = None
        self.cle_tts = None
        self._vider()

    def _charger_tts(self, voix: dict):
        cle = (voix["modele"], self._depot(voix))
        if self.cle_tts != cle:
            if self.modele_tts is not None:
                self.decharger_tts()
            self.verifier_memoire(voix)
            self._mx()                                      # cache limit set before the first loading
            from mlx_audio.tts.utils import load_model     # tts/utils.py:108
            if self.morceau_chargement_mo > 0:
                # lazy=True : utils.py:449-450 evaluates nothing ; weights are read here in batches (peak of the first
                # loading : swap +0.7 to 1.4 Go with an mx.eval of all Drama 3, gen5-bench4 from 11/13).
                modele = load_model(cle[1], lazy=True)
                if voix["modele"] == "drama3":
                    self._preparer_drama3(modele, voix)
                self.lots_chargement = evaluer_par_morceaux(self._mx(), modele, self.morceau_chargement_mo * 1024**2,
                                                            entre=self.attendre)
            else:
                modele = load_model(cle[1])
                if voix["modele"] == "drama3":
                    self._preparer_drama3(modele, voix)
            # utils.py:431-447: the "weights" dict (all safetensors, codec included) remains referenced until the return of
            # load_model and its freed buffers go into the MLX cache. Reloading takes as long as the first
            # loading (23.4 Go of active, gen5-maudio1) : the « 13 Go afterwards » of gen5-audio1 was an RSS, not the active.
            self._vider()
            if voix["modele"] == "drama3":
                self._brancher_preemption(modele)
            self.modele_tts = modele
            self.cle_tts = cle
        return self.modele_tts

    @staticmethod
    def _references_drama3(voix: dict) -> tuple[list[str], list[str]]:
        """Paths and texts of the references, as _synthese passes them to generate (same voice cache key)."""

        refs = voix.get("references") or []
        return ([os.path.expanduser(r[0]) for r in refs],
                [open(os.path.expanduser(r[1]), encoding="utf-8").read().strip() for r in refs])

    def _preparer_drama3(self, modele, voix: dict) -> None:
        """Before evaluating the backbone and tower weights (bench pli5-drama3-memoire from 11/15 : 29.70 -> 26.46 Go
        of maximum fingerprint with the cache limit, its identical sample-by-sample to the same seeds) :
        1. folded codec kernels evaluated, then weight_v and weight_g replaced by a scalar : only _folded serves
           inference (fish_s1_dac.py:148-164) ; -0.30 Go ;
        2. reference voice encoded by the codec while other weights are not read : the encoding peak
           (about 2.2 Go for 12 s of reference, more with a long reference) no longer adds to the weights ; -2.87 Go.
           drama3.py:198-226 keeps the encoded voice (voice cache, key = samples + text) : the following request does
           not re-encode it. Another Drama 3 voice requested later is encoded with weights loaded, as before."""

        mx = self._mx()
        codec = getattr(modele, "codec", None)
        if codec is not None:
            plies = [m for m in codec.modules() if "_folded" in m]
            if plies:
                mx.eval(*[m["_folded"] for m in plies])
                for m in plies:
                    m.weight_v = mx.zeros((1,), dtype=m.weight_v.dtype)
                    m.weight_g = mx.zeros((1,), dtype=m.weight_g.dtype)
            mx.eval(codec.parameters())
            mx.clear_cache()
        chemins, textes = self._references_drama3(voix)
        if chemins and hasattr(modele, "_voice"):
            voix_encodee = modele._voice(chemins, textes)
            mx.eval([c for _, c in voix_encodee.references])
            mx.clear_cache()

    def _brancher_preemption(self, modele) -> None:
        """Drama 3 produces its frames in modele._frames (drama3.py:406, called at 505) : a wait every N frames."""

        if not hasattr(modele, "_frames"):
            return
        origine = modele._frames
        attendre = self.attendre

        def _frames(*a, **k):
            for i, trame in enumerate(origine(*a, **k)):
                if i % FRAMES_ENTRE_ATTENTES == 0:
                    attendre()
                yield trame

        object.__setattr__(modele, "_frames", _frames)

    def synthese(self, voix: dict, texte: str, vitesse: float, options: dict):
        return self.executer(self._synthese, voix, texte, vitesse, options)

    def _synthese(self, voix: dict, texte: str, vitesse: float, options: dict):
        self.attendre()      # gen5-bench3 : a reload during a general request cost 1k 35 j/s instead of 50
        modele = self._charger_tts(voix)
        args: dict = {"text": texte}
        kind = voix["modele"]
        refs = voix.get("references") or []
        if kind == "drama3":
            # fish_drama3/drama3.py:450 : speed != 1 raises ValueError ; voice and lang_code would be ignored
            if vitesse != 1.0:
                raise ErreurClient("Drama 3 does not set the speed (speed must be 1)")
            args["ref_audio"], args["ref_text"] = self._references_drama3(voix)
        elif kind == "kokoro":
            # kokoro/kokoro.py:295 : voice, speed, lang_code
            args.update(voice=voix.get("voice"), speed=vitesse, lang_code=voix.get("lang_code", "a"))
        else:
            # qwen3_tts/qwen3_tts.py:1151 : voice, lang_code, ref_audio, ref_text, speed
            args.update(speed=vitesse, lang_code=voix.get("lang_code", "auto"))
            if voix.get("voice"):
                args["voice"] = voix["voice"]
            if refs:
                args["ref_audio"] = os.path.expanduser(refs[0][0])
                args["ref_text"] = open(os.path.expanduser(refs[0][1]), encoding="utf-8").read().strip()
        if kind in ("drama3", "qwen3tts"):
            for k in ("temperature", "top_p"):
                if options.get(k) is not None:
                    args[k] = float(options[k])
        morceaux, taux = [], 24000
        self.attendre()
        try:
            for r in modele.generate(**args):      # GenerationResult : .audio (mx.array), .sample_rate
                morceaux.append(np.array(r.audio, dtype=np.float32).reshape(-1))
                taux = r.sample_rate
                self.attendre()                    # entre deux morceaux (Kokoro, Qwen3-TTS, morceaux de texte de Drama 3)
        finally:
            self._vider()
        if not morceaux:
            raise RuntimeError("aucun audio produit")
        return np.concatenate(morceaux), taux

    # transcription --------------------------------------------
    def transcription(self, chemin: str, langue: str | None) -> str:
        return self.executer(self._transcription, chemin, langue)

    def _transcription(self, chemin: str, langue: str | None) -> str:
        self.controler()
        if self.asr is None:
            self._mx()
            from mlx_audio.stt.utils import load_model as charger_stt     # stt/utils.py:154
            self.asr = charger_stt(self.depot_asr)
            self._vider()
        self.controler()
        # qwen3_asr.py:1243 : generate(audio, language=...) returns an STTOutput (.text)
        try:
            return self.asr.generate(chemin, language=langue).text.strip()
        finally:
            self._vider()


# ---------------------------------------------------------------- serveur

class Catalogue(dict):
    """Voices by name, including aliases (an alias returns its voice entry); ``defaut``: voice by model and « * »."""

    def __init__(self, voix: dict[str, dict] | None = None, defaut: dict[str, str] | None = None) -> None:
        super().__init__(voix or {})
        self.defaut = dict(defaut or {})

    def noms(self) -> list[str]:
        """The names of the voices, without the aliases."""

        return sorted(n for n, v in self.items() if v.get("_nom", n) == n)

    def voix_par_defaut(self, modele: str | None) -> str | None:
        """Voice of a request without « voice »: the requested model's, else « * »; the catalog comes before the code."""

        cle = modele if modele in MODELES_TTS else "*"
        return self.defaut.get(cle) or VOIX_DEFAUT.get(cle)


def charger_catalogue(chemin: str | None) -> Catalogue:
    """Voice catalog: {"voix": {NOM: {...}}, "defaut": {...}}; an object without « voix » is read as the list of voices."""

    defaut: dict = {}
    if chemin is None:
        brut = CATALOGUE_DEFAUT
    else:
        with open(chemin, encoding="utf-8") as f:
            donnees = json.load(f)
        if isinstance(donnees, dict) and "voix" in donnees:
            brut, defaut = donnees["voix"], donnees.get("defaut") or {}
        else:
            brut = donnees
    if not isinstance(brut, dict):
        raise ValueError("catalogue de voix : objet attendu")
    catalogue = Catalogue()
    for nom, v in brut.items():
        if not isinstance(v, dict) or v.get("modele") not in MODELES_TTS:
            raise ValueError(f"voix « {nom} » : « modele » doit valoir {', '.join(MODELES_TTS)}")
        if "memoire_go" in v and not (isinstance(v["memoire_go"], (int, float)) and v["memoire_go"] > 0):
            raise ValueError(f"voice « {nom} »: « memoire_go » must be a positive number of GB")
        catalogue[nom] = {**v, "_nom": nom}
    for nom, v in brut.items():
        alias = v.get("alias", [])
        if isinstance(alias, str):
            alias = [alias]
        if not isinstance(alias, list) or not all(isinstance(a, str) and a for a in alias):
            raise ValueError(f"voice « {nom} »: « alias » must be a list of names")
        for a in alias:
            if a in catalogue:
                raise ValueError(f"voice « {nom} »: alias « {a} » is already a voice or alias name")
            catalogue[a] = catalogue[nom]
    if not isinstance(defaut, dict):
        raise ValueError("voice catalog: « defaut » must be an object {model or *: voice}")
    for cle, voix in defaut.items():
        if cle != "*" and cle not in MODELES_TTS:
            raise ValueError(f"defaut: unknown key « {cle} » (expected: {', '.join(MODELES_TTS)} or *)")
        if voix not in catalogue:
            raise ValueError(f"defaut: voice « {voix} » missing from the catalog")
        if cle != "*" and catalogue[voix]["modele"] != cle:
            raise ValueError(f"defaut: « {voix} » is a {catalogue[voix]['modele']} voice, not {cle}")
    catalogue.defaut = defaut
    return catalogue


def noms_voix(catalogue: dict) -> list[str]:
    return catalogue.noms() if isinstance(catalogue, Catalogue) else sorted(catalogue)


def voix_par_defaut(catalogue: dict, modele: str | None) -> str | None:
    if isinstance(catalogue, Catalogue):
        return catalogue.voix_par_defaut(modele)
    cle = modele if modele in MODELES_TTS else "*"
    return VOIX_DEFAUT.get(cle)


def faire_serveur(port: int, catalogue: dict[str, dict], moteur, *, hote: str = "127.0.0.1") -> ThreadingHTTPServer:
    class Gestionnaire(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _envoyer(self, code: int, corps: bytes, type_: str, extra: dict | None = None) -> None:
            self.send_response(code)
            self.send_header("Content-Type", type_)
            for cle, val in (extra or {}).items():
                self.send_header(cle, str(val))
            self.send_header("Content-Length", str(len(corps)))
            self.end_headers()
            self.wfile.write(corps)
            self.wfile.flush()

        def _json(self, code: int, objet: dict) -> None:
            self._envoyer(code, json.dumps(objet, ensure_ascii=False).encode(), "application/json")

        def _erreur(self, code: int, message: str, type_: str = "invalid_request_error") -> None:
            self._json(code, {"error": {"message": message, "type": type_}})

        def do_GET(self) -> None:
            if self.path.split("?", 1)[0] == "/sante":
                cle = getattr(moteur, "cle_tts", None)        # loaded synthesis model (read by the supervisor)
                self._json(200, {"ok": True, "simule": moteur.simule, "mlx_audio_importe": "mlx_audio" in sys.modules,
                                 "tts_charge": cle[0] if cle else None,
                                 "voix": noms_voix(catalogue),
                                 "defaut": {"*": voix_par_defaut(catalogue, None),
                                            **{m: voix_par_defaut(catalogue, m) for m in MODELES_TTS
                                               if voix_par_defaut(catalogue, m)}},
                                 "memoire_go": {n: float(v.get("memoire_go", MEMOIRE_GO_DEFAUT[v["modele"]]))
                                                for n, v in sorted(catalogue.items()) if v.get("_nom", n) == n}})
            else:
                self._erreur(404, "inconnu")

        def _corps(self) -> bytes:
            longueur = int(self.headers.get("Content-Length") or 0)
            if longueur > LIMITE_CORPS:
                raise ErreurClient("corps trop gros")
            return self.rfile.read(longueur) if longueur else b""

        def do_POST(self) -> None:
            chemin = self.path.split("?", 1)[0].rstrip("/")
            try:
                if chemin == "/v1/audio/speech":
                    self._speech(self._corps())
                elif chemin == "/v1/audio/transcriptions":
                    self._transcriptions(self._corps())
                elif chemin == "/annuler":
                    self._corps()
                    self._json(200, {"annulee": moteur.annuler()})
                else:
                    self._erreur(404, "inconnu")
            except Annulee as exc:
                self._erreur(499, str(exc), "annulee")
            except ErreurClient as exc:
                self._erreur(400, str(exc))
            except ErreurMemoire as exc:
                self._erreur(503, str(exc), "memoire")
            except Exception as exc:                       # noqa: BLE001 - any engine failure becomes a 500
                self._erreur(500, f"{type(exc).__name__}: {exc}", "server_error")

        def _speech(self, corps: bytes) -> None:
            try:
                d = json.loads(corps or b"{}")
            except ValueError:
                raise ErreurClient("corps JSON invalide")
            if not isinstance(d, dict):
                raise ErreurClient("corps JSON : objet attendu")
            texte = d.get("input")
            if not isinstance(texte, str) or not texte.strip():
                raise ErreurClient("« input » (texte non vide) obligatoire")
            modele = d.get("model")
            voix_nom = d.get("voice") or voix_par_defaut(catalogue, modele)
            if not voix_nom:
                raise ErreurClient(f"no default {modele} voice: specify « voice », or add "
                                   f"« defaut »: {{\"{modele}\": NOM}} to the voice catalog")
            voix = catalogue.get(voix_nom)
            if voix is None:
                raise ErreurClient(f"voix inconnue « {voix_nom} » (disponibles : {', '.join(noms_voix(catalogue))})")
            if modele in MODELES_TTS and modele != voix["modele"]:
                raise ErreurClient(f"voice « {voix_nom} » is a {voix['modele']} voice, not {modele}")
            fmt = d.get("response_format") or "wav"
            if fmt not in ("wav", "pcm"):
                raise ErreurClient("response_format : wav ou pcm seulement")
            try:
                vitesse = float(d.get("speed") if d.get("speed") is not None else 1.0)
            except (TypeError, ValueError):
                raise ErreurClient("speed : nombre attendu")
            moteur.debut_demande()
            audio, taux = moteur.synthese(voix, texte, vitesse, d)
            extra = {"X-PLI-Attentes": moteur.attentes_demande()}
            if fmt == "pcm":
                pcm = (np.clip(audio, -1, 1) * 32767.0).astype("<i2").tobytes()
                self._envoyer(200, pcm, "audio/pcm", extra)
            else:
                self._envoyer(200, encoder_wav(audio, taux), "audio/wav", extra)

        def _transcriptions(self, corps: bytes) -> None:
            parties = analyser_multipart(self.headers.get("Content-Type", ""), corps)
            champs = {p["nom"]: p for p in parties}
            fichier = champs.get("file")
            if fichier is None or not fichier["donnees"]:
                raise ErreurClient("champ « file » (audio) obligatoire")
            langue = champs.get("language", {}).get("donnees", b"").decode("utf-8", "replace").strip() or None
            if langue:
                langue = LANGUES.get(langue.lower(), langue)
            suffixe = os.path.splitext(fichier["fichier"] or "")[1] or ".wav"
            if not re.fullmatch(r"\.[A-Za-z0-9]{1,8}", suffixe):
                suffixe = ".wav"
            with tempfile.NamedTemporaryFile(suffix=suffixe, delete=False) as f:
                f.write(fichier["donnees"])
                nom = f.name
            try:
                texte = moteur.transcription(nom, langue)
            finally:
                os.unlink(nom)
            self._json(200, {"text": texte})

    return ThreadingHTTPServer((hote, port), Gestionnaire)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--voix", default=None, help="JSON voice catalog (default: built-in voices)")
    p.add_argument("--asr", default=ASR_DEFAUT, help="Hugging Face repository of the transcription model")
    p.add_argument("--simuler", action="store_true", help="dummy models, without importing mlx_audio")
    p.add_argument("--cache-limit-go", type=float, default=CACHE_LIMITE_GO_DEFAUT,
                   help="MLX buffer cache limit in GB (negative: no limit)")
    p.add_argument("--memoire-disponible-go", type=float, default=None,
                   help="imposed available memory (tests); otherwise read with vm_stat")
    p.add_argument("--duree-morceau", type=float, default=0.0, help="--simuler: dummy duration of a track (s)")
    p.add_argument("--morceau-chargement-mo", type=int, default=MORCEAU_CHARGEMENT_MO_DEFAUT,
                   help="first loading of a voice: weights evaluated in batches of N MB (0: a single mx.eval)")
    a = p.parse_args()
    catalogue = charger_catalogue(a.voix)
    drapeau = os.environ.get("PLI_GEN_DRAPEAU", "")
    memoire = (lambda: int(a.memoire_disponible_go * GO)) if a.memoire_disponible_go is not None else None
    moteur = (MoteurSimule(drapeau, memoire, duree_morceau=a.duree_morceau) if a.simuler
              else MoteurMlx(a.asr, drapeau, memoire, a.cache_limit_go, a.morceau_chargement_mo))
    faire_serveur(a.port, catalogue, moteur).serve_forever()


if __name__ == "__main__":
    main()
