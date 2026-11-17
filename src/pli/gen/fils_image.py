"""Image generation child process (PLI 5): standalone OpenAI HTTP server, one mflux model loaded at a time.

    VENV_MFLUX/bin/python fils_image.py --port N --catalogue catalogue.json [--simuler]

Standalone: does not import pli. Standard library only in --simuler mode; in real mode, mflux, mlx, numpy and PIL
are imported on first request (mflux venv). Launched by the supervisor (pli.gen.superviseur), which provides it with
PLI_GEN_DRAPEAU: as long as this file exists, the general model works and the child waits BETWEEN two denoising steps.

Routes : GET /sante, GET /etat, POST /v1/images/generations, POST /v1/images/edits (501 : voir ROUTE_EDITS),
POST /annuler.

Cancellation: POST /annuler (empty body) cancels the running request; 200 {"annulee": true|false}. The child process sees it
at the next wait point (before the first step, between two steps, during the flag wait), releases the model
(MLX cache cleared, the next request reloads) and responds to the cancelled request 499 {"error": {"type": "cancelled"}}. 499
("client closed", nginx usage) rather than 409: the caller gave up, there is no state conflict. The loading of the
model and a denoising step already started are not interrupted.

Catalogue (JSON) :
  {"defaut": "qwen21",
   "modeles": {"qwen21": {"classe": "qwen21|qwen2512|flux3klein|krea2", "chemin": "...", "horaire": "linear",
                          "etapes": 40, "guidance": 4.0, "taille": "1024x1024", "quantize": null,
                          "cache_limit_go": 4, "liberer_encodeur": true, "low_ram": false,
                          "cibles": [{"source": "img_in", "modele": "img_in"}]}},
  « quantize »: bits passed to mflux at loading (3, 4, 5, 6 or 8; null = weights as is, already quantified or not).
  krea2 (Krea 2 Turbo, mflux 0.24 models/krea2): schedules er_sde (default, « linear » leads there: krea2.py:209-214) or
  euler; 8 steps and guidance 1.0 for Turbo (krea2_generate.py:15-20); the negative prompt is only encoded if
  guidance != 1.0, otherwise refusal 400 rather than an ignored negative.
   "lora":    {"lightning-v1": {"chemin": "...", "base": ["qwen21"], "force": 1.0,
                                "horaire": "lightning1", "etapes": 4, "guidance": 1.0, "cibles": []}}}
"""

from __future__ import annotations

import argparse
import base64
import gc
import io
import json
import os
import random
import re
import shutil
import struct
import subprocess
import threading
import time
import zlib
from contextlib import contextmanager
from dataclasses import dataclass
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable

ATTENTE_MAX_S = 600.0       # a flag never removed does not block the child forever (like fils_simule)
PAS_ATTENTE_S = 0.02
LIMITE_CORPS = 8 * 1024**2
CLASSES = ("qwen21", "qwen2512", "flux3klein", "krea2")
HORAIRES = {"qwen21": ("linear", "lightning1", "lightning2"), "qwen2512": ("linear",), "flux3klein": ("linear",),
            "krea2": ("linear", "er_sde", "euler")}
CLASSES_CIBLES = ("qwen21", "qwen2512")       # additional LoRA targets: see _cibles_ajoutees
QUANTIFICATIONS = (3, 4, 5, 6, 8)
ROUTE_EDITS = ("POST /v1/images/edits: not implemented. mflux has editing variants (QwenImage21Edit, Flux3KleinEdit) "
               "but they require reference images in multipart and other classes to load; "
               "Qwen-Image-2512 has no editing. Use /v1/images/generations.")
TAILLE_MIN, TAILLE_MAX, PAS_TAILLE = 256, 2048, 16
GO = 1024**3
# memory need (GB) per class, measured on an M1 Ultra 64 GB (peak fingerprint); « memoire_go » of the model replaces it
# krea2: measured on 11/13/2026 (krea2-essai1, 1024x1024, 8 steps, quantize 8): child peak fingerprint 28.2 to 28.4 GB
# (MLX peak 27.7 to 28.1 GiB with LoRA), hence 29 GB.
MEMOIRE_GO_CLASSE = {"qwen21": 26.0, "qwen2512": 34.0, "flux3klein": 16.0, "krea2": 29.0}


class Annulee(Exception):
    """The current request was cancelled by POST /annuler: response 499."""


class ErreurDemande(Exception):
    def __init__(self, code: int, message: str, type_: str = "invalid_request_error") -> None:
        super().__init__(message)
        self.code, self.message, self.type = code, message, type_


# ---------------------------------------------------------------- catalogue

def catalogue_prive():
    """The gen/catalogue_prive.py module, loaded by its path: this child does not import pli (other venv)."""

    import importlib.util
    import sys
    nom = "_pli_catalogue_prive"
    if nom not in sys.modules:
        chemin = os.path.join(os.path.dirname(os.path.abspath(__file__)), "catalogue_prive.py")
        spec = importlib.util.spec_from_file_location(nom, chemin)
        module = importlib.util.module_from_spec(spec)
        sys.modules[nom] = module
        spec.loader.exec_module(module)
    return sys.modules[nom]


def charger_catalogue(chemin: str, environ=os.environ) -> dict[str, Any]:
    """The JSON catalogue, plus the private LoRAs of PLI_GEN_CATALOGUE_PRIVE (gen/catalogue_prive.py), then validated."""

    with open(chemin, encoding="utf-8") as f:
        c = json.load(f)
    if isinstance(c, dict):
        catalogue_prive().fusionner(c, "image", environ)
    return valider_catalogue(c)


def valider_catalogue(c: dict[str, Any]) -> dict[str, Any]:
    modeles, loras = c.get("modeles"), c.get("lora", {})
    if not isinstance(modeles, dict) or not modeles:
        raise ValueError("catalogue: « modeles » key missing or empty")
    if not isinstance(loras, dict):
        raise ValueError("catalogue: « lora » must be an object")
    for nom, m in modeles.items():
        if m.get("classe") not in CLASSES:
            raise ValueError(f"model {nom}: class {m.get('classe')!r} unknown (expected: {', '.join(CLASSES)})")
        if not m.get("chemin"):
            raise ValueError(f"model {nom}: « chemin » missing")
        if "memoire_go" in m and not (isinstance(m["memoire_go"], (int, float)) and m["memoire_go"] > 0):
            raise ValueError(f"model {nom}: « memoire_go » must be a positive number of GB")
        if m.get("horaire", "linear") not in HORAIRES[m["classe"]]:
            raise ValueError(f"model {nom}: schedule {m['horaire']!r} not supported for {m['classe']}")
        _lire_taille(m.get("taille", "1024x1024"))
        if m.get("cibles") and m["classe"] not in CLASSES_CIBLES:
            raise ValueError(f"model {nom}: unhandled extra LoRA targets for {m['classe']}")
        if m.get("quantize") is not None and m["quantize"] not in QUANTIFICATIONS:
            raise ValueError(f"model {nom}: quantize {m['quantize']!r} (expected null or {QUANTIFICATIONS})")
    for nom, lo in loras.items():
        if not lo.get("chemin"):
            raise ValueError(f"LoRA {nom} : « chemin » manquant")
        bases = lo.get("base")
        bases = [bases] if isinstance(bases, str) else bases
        if not bases or any(b not in modeles for b in bases):
            raise ValueError(f"LoRA {nom}: 'base' must name models from the catalog")
        lo["base"] = bases
        for b in bases:
            h = lo.get("horaire")
            if h is not None and h not in HORAIRES[modeles[b]["classe"]]:
                raise ValueError(f"LoRA {nom}: unhandled schedule {h!r} for {b}")
            if lo.get("cibles") and modeles[b]["classe"] not in CLASSES_CIBLES:
                raise ValueError(f"LoRA {nom}: unhandled extra targets for {modeles[b]['classe']}")
        d = lo.get("defauts", {})
        if not isinstance(d, dict) or set(d) - {"passes", "taille", "guidance"}:
            raise ValueError(f"LoRA {nom} : « defauts » : passes, taille, guidance seulement")
        if "passes" in d and (isinstance(d["passes"], bool) or not isinstance(d["passes"], int) or not 1 <= d["passes"] <= 200):
            raise ValueError(f"LoRA {nom}: defauts.passes: integer between 1 and 200")
        if "taille" in d:
            _lire_taille(d["taille"])
    defaut = c.get("defaut") or next(iter(modeles))
    if defaut not in modeles:
        raise ValueError(f"catalog: default {defaut!r} missing from models")
    c["defaut"] = defaut
    c["lora"] = loras
    return c


def _lire_taille(texte: Any) -> tuple[int, int]:
    m = re.fullmatch(r"(\d+)x(\d+)", str(texte))
    if not m:
        raise ErreurDemande(400, f"size {texte!r}: expected 'WIDTHxHEIGHT', e.g. 1024x1024")
    w, h = int(m.group(1)), int(m.group(2))
    for v in (w, h):
        if not TAILLE_MIN <= v <= TAILLE_MAX or v % PAS_TAILLE:
            raise ErreurDemande(400, f"size {texte!r}: sides between {TAILLE_MIN} and {TAILLE_MAX}, multiples of {PAS_TAILLE}")
    return w, h


# ---------------------------------------------------------------- request

@dataclass
class Plan:
    modele: str
    loras: list[tuple[str, float]]
    horaire: str
    etapes: int
    guidance: float
    largeur: int
    hauteur: int
    seed: int
    prompt: str
    negatif: str | None

    @property
    def cle(self) -> tuple:
        """Loading key: model + LoRA and forces (order irrelevant). Schedule, steps and seed do not reload."""

        return (self.modele, tuple(sorted(self.loras)))


def _nombre(v: Any, nom: str) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise ErreurDemande(400, f"{nom} : nombre attendu")
    return float(v)


def resoudre(demande: dict[str, Any], cat: dict[str, Any]) -> Plan:
    prompt = demande.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise ErreurDemande(400, "prompt : texte non vide attendu")
    nom_modele = demande.get("model") or cat["defaut"]
    m = cat["modeles"].get(nom_modele)
    if m is None:
        raise ErreurDemande(400, f"model {nom_modele!r} not in catalog (available: {', '.join(cat['modeles'])})")
    if demande.get("n", 1) != 1:
        raise ErreurDemande(400, "n: only value 1 is supported (one model, one image per request)")
    if demande.get("response_format", "b64_json") != "b64_json":
        raise ErreurDemande(400, "response_format: only b64_json is supported")
    classe = m["classe"]

    loras: list[tuple[str, float]] = []
    horaire, etapes, guidance = m.get("horaire", "linear"), int(m.get("etapes", 0) or 0), m.get("guidance")
    impose_etapes = impose_guidance = None
    brut = demande.get("lora") or []
    if not isinstance(brut, list):
        raise ErreurDemande(400, "lora: list expected, e.g. [{\"nom\": \"...\", \"force\": 1.0}]")
    vus: set[str] = set()
    horaire_lora = None
    defauts: dict[str, Any] = {}           # default settings from the first LoRA providing them ("defauts")
    for item in brut:
        nom, force = (item, None) if isinstance(item, str) else (item.get("nom"), item.get("force")) if isinstance(item, dict) else (None, None)
        lo = cat["lora"].get(nom) if isinstance(nom, str) else None
        if lo is None:
            raise ErreurDemande(400, f"LoRA {nom!r} hors catalogue (disponibles : {', '.join(cat['lora']) or 'aucun'})")
        if nom in vus:
            raise ErreurDemande(400, f"LoRA {nom} given twice")
        vus.add(nom)
        if nom_modele not in lo["base"]:
            raise ErreurDemande(400, f"LoRA {nom} incompatible with {nom_modele} (bases: {', '.join(lo['base'])})")
        force = float(lo.get("force", 1.0)) if force is None else _nombre(force, f"lora {nom} : force")
        loras.append((nom, force))
        for cle, valeur in (lo.get("defauts") or {}).items():
            defauts.setdefault(cle, valeur)
        if lo.get("horaire"):
            if horaire_lora not in (None, lo["horaire"]):
                raise ErreurDemande(400, f"LoRA {nom}: schedule {lo['horaire']} conflicts with {horaire_lora}")
            horaire_lora = horaire = lo["horaire"]
            if lo.get("etapes"):
                etapes = impose_etapes = int(lo["etapes"])
            if lo.get("guidance") is not None:
                guidance = impose_guidance = float(lo["guidance"])

    if demande.get("steps") is not None:
        s = _nombre(demande["steps"], "steps")
        if s != int(s) or not 1 <= s <= 200:
            raise ErreurDemande(400, "steps: integer between 1 and 200")
        if impose_etapes is not None and int(s) != impose_etapes:
            raise ErreurDemande(400, f"steps: schedule {horaire} requires {impose_etapes} steps")
        etapes = int(s)
    elif defauts.get("passes") and impose_etapes is None:
        etapes = int(defauts["passes"])
    if demande.get("guidance") is None and defauts.get("guidance") is not None and impose_guidance is None \
            and classe != "flux3klein":
        guidance = float(defauts["guidance"])
    if demande.get("guidance") is not None:
        if classe == "flux3klein":
            raise ErreurDemande(400, "guidance: flux3klein (distilled) has no guidance")
        g = _nombre(demande["guidance"], "guidance")
        if g < 0:
            raise ErreurDemande(400, "guidance : positif attendu")
        if impose_guidance is not None and g != impose_guidance:
            raise ErreurDemande(400, f"guidance : l'horaire {horaire} impose {impose_guidance:g}")
        guidance = g
    if etapes <= 0:
        etapes = 16
    negatif = demande.get("negative_prompt") or None
    if negatif is not None and (classe == "flux3klein" or not isinstance(negatif, str)):
        raise ErreurDemande(400, "negative_prompt: text, and not supported by flux3klein")
    if negatif is not None and classe == "krea2" and (guidance is None or float(guidance) == 1.0):
        # Krea2PromptEncoder only encodes the negative branch if guidance != 1.0 (krea2_generate.py:21-28)
        raise ErreurDemande(400, "negative_prompt: Krea 2 ignores it with guidance 1.0; set guidance different from 1")
    seed = demande.get("seed")
    if seed is None:
        seed = random.randrange(2**31)
    elif isinstance(seed, bool) or not isinstance(seed, int):
        raise ErreurDemande(400, "seed : entier attendu")
    w, h = _lire_taille(demande.get("size") or defauts.get("taille") or m.get("taille", "1024x1024"))
    return Plan(nom_modele, loras, horaire, etapes, float(guidance) if guidance is not None else 1.0, w, h, seed, prompt,
                negatif)


# ---------------------------------------------------------------- memory (copied from superviseur.py: the child is standalone)

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
    raise OSError("available memory unreadable")


def besoin_go(cat: dict[str, Any], nom_modele: str) -> float:
    m = cat["modeles"][nom_modele]
    return float(m.get("memoire_go", MEMOIRE_GO_CLASSE[m["classe"]]))


# ---------------------------------------------------------------- preemption

def attendre_drapeau(drapeau: str, *, max_s: float = ATTENTE_MAX_S, pas: float = PAS_ATTENTE_S, arret=None) -> int:
    """Waits while the flag file exists; returns the number of waiting steps (0 if the path is free).
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


# ---------------------------------------------------------------- moteurs

class Moteur:
    """A single loaded model. The loading key changes (model, LoRA or forces): unload then reload."""

    def __init__(self, cat: dict[str, Any], drapeau: str, memoire_disponible: Callable[[], int] | None = None) -> None:
        self.cat, self.drapeau = cat, drapeau
        self.memoire_disponible = memoire_disponible or memoire_disponible_systeme
        self.cle: tuple | None = None
        self.chargements = 0
        self.verrou = threading.Lock()
        self.attentes = 0
        self._verrou_annulation = threading.Lock()
        self._numero = 0
        self._courante: int | None = None     # number of the request currently running
        self._annulee: int | None = None      # number cancelled by /annuler

    @contextmanager
    def _demande(self):
        """One request at a time (lock); numbers the request so that /annuler targets the one running."""

        with self.verrou:
            with self._verrou_annulation:
                self._numero += 1
                self._courante = self._numero
            try:
                yield
            finally:
                with self._verrou_annulation:
                    self._courante = self._annulee = None

    def annuler(self) -> bool:
        """Cancels the request currently running; false if there is none."""

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
        self.attentes += attendre_drapeau(self.drapeau, arret=self.annulation_demandee)
        self.controler()

    def verifier_memoire(self, plan: "Plan") -> None:
        """503 'memoire' if available memory is below the model's need. An unreadable value does not refuse."""

        besoin = besoin_go(self.cat, plan.modele) * GO
        try:
            libre = int(self.memoire_disponible())
        except Exception:  # noqa: BLE001
            return
        if libre < besoin:
            raise ErreurDemande(503, f"insufficient memory to load {plan.modele}: {libre / GO:.1f} GB available, "
                                     f"{besoin / GO:.1f} GB required", "memoire")

    def etat(self) -> dict[str, Any]:
        return {"cle": None if self.cle is None else [self.cle[0], [list(x) for x in self.cle[1]]],
                "chargements": self.chargements}

    def generer(self, plan: Plan) -> tuple[bytes, dict[str, Any]]:
        with self._demande():
            t0 = time.monotonic()
            charge = False
            if self.cle != plan.cle:
                self._liberer()
                self.verifier_memoire(plan)              # the old model is already unloaded: we measure what remains
                self._charger(plan)
                self.cle, charge = plan.cle, True
                self.chargements += 1
            t1 = time.monotonic()
            self.attentes = 0
            try:
                png, extra = self._produire(plan)
            except BaseException:
                self._liberer()               # uncertain state: the next request reloads
                raise
            if self._jetable(plan):
                self._liberer()
            infos = {"chargements": self.chargements, "charge": charge, "attentes": self.attentes,
                     "secondes_chargement": round(t1 - t0, 2), "secondes_generation": round(time.monotonic() - t1, 2),
                     "etapes": plan.etapes, "horaire": plan.horaire, **extra}
            return png, infos

    def _liberer(self) -> None:
        if self.cle is not None:
            self._decharger()
        self.cle = None

    def _jetable(self, plan: Plan) -> bool:
        return False

    def _charger(self, plan: Plan) -> None: ...
    def _decharger(self) -> None: ...
    def _produire(self, plan: Plan) -> tuple[bytes, dict[str, Any]]: ...


def png_uni(largeur: int, hauteur: int, rvb: tuple[int, int, int]) -> bytes:
    """Valid PNG of a single color (standard library only)."""

    def bloc(type_: bytes, donnees: bytes) -> bytes:
        return struct.pack(">I", len(donnees)) + type_ + donnees + struct.pack(">I", zlib.crc32(type_ + donnees))

    ligne = b"\x00" + bytes(rvb) * largeur
    return (b"\x89PNG\r\n\x1a\n" + bloc(b"IHDR", struct.pack(">IIBBBBB", largeur, hauteur, 8, 2, 0, 0, 0))
            + bloc(b"IDAT", zlib.compress(ligne * hauteur, 6)) + bloc(b"IEND", b""))


class MoteurSimule(Moteur):
    """No model: fake step loop that waits for the flag before each step, like the real engine."""

    def __init__(self, cat, drapeau, *, duree_etape: float = 0.0, duree_chargement: float = 0.0,
                 memoire_disponible=None) -> None:
        super().__init__(cat, drapeau, memoire_disponible or (lambda: 1 << 50))  # simulated: the machine's real memory does not count
        self.duree_etape, self.duree_chargement = duree_etape, duree_chargement

    def _charger(self, plan: Plan) -> None:
        time.sleep(self.duree_chargement)

    def _produire(self, plan: Plan) -> tuple[bytes, dict[str, Any]]:
        for _ in range(plan.etapes):
            self.attendre()
            time.sleep(self.duree_etape)
        self.attendre()
        c = (plan.seed * 37 % 256, plan.etapes * 11 % 256, len(plan.loras) * 60 % 256)
        return png_uni(min(plan.largeur, 64), min(plan.hauteur, 64), c), {"simule": True}


class MoteurMflux(Moteur):
    """mflux 0.24: see file:line references in the comments (venv mflux 0.24)."""

    def __init__(self, cat, drapeau, memoire_disponible=None) -> None:
        super().__init__(cat, drapeau, memoire_disponible)
        import mlx.core as mx          # noqa: PLC0415 - late import: simulated mode does not have mflux
        self.mx = mx
        self.modele: Any = None
        self.jetable = False
        _enregistrer_horaires()

    def _jetable(self, plan: Plan) -> bool:
        return self.jetable

    def _charger(self, plan: Plan) -> None:
        mx = self.mx
        m = self.cat["modeles"][plan.modele]
        classe = m["classe"]
        cibles = list(m.get("cibles") or [])
        for nom, _ in plan.loras:
            cibles += self.cat["lora"][nom].get("cibles") or []
        chemins = [self.cat["lora"][n]["chemin"] for n, _ in plan.loras] or None
        forces = [f for _, f in plan.loras] or None
        limite = float(m.get("cache_limit_go", 4))
        low_ram = bool(m.get("low_ram", False))
        liberer = bool(m.get("liberer_encodeur", True))
        bits = m.get("quantize")                                     # None : poids tels quels
        mx.reset_peak_memory()
        with _cibles_ajoutees(classe, cibles):
            if classe == "qwen21":
                from mflux.models.common.config import ModelConfig
                from mflux.models.qwen21.variants.txt2img.qwen_image_21 import QwenImage21
                modele = QwenImage21(quantize=bits, model_path=m["chemin"], model_config=ModelConfig.qwen_image_21(),
                                     lora_paths=chemins, lora_scales=forces, bake_lora=False)
            elif classe == "qwen2512":
                from mflux.models.common.config import ModelConfig
                from mflux.models.qwen.variants.txt2img.qwen_image import QwenImage
                modele = QwenImage(quantize=bits, model_path=m["chemin"], model_config=ModelConfig.qwen_image(),
                                   lora_paths=chemins, lora_scales=forces, bake_lora=False)
            elif classe == "krea2":
                # krea2.py:31-49; LoRA ai-toolkit (diffusion_model.blocks.N.*, txtfusion.*) covered by
                # Krea2LoRAMapping (512 keys out of 512 for the 3 LoRAs waiting, verified on 11/13)
                from mflux.models.common.config import ModelConfig
                from mflux.models.krea2.variants.txt2img.krea2 import Krea2
                modele = Krea2(quantize=bits, model_path=m["chemin"], model_config=ModelConfig.krea2(),
                               lora_paths=chemins, lora_scales=forces, bake_lora=False)
            else:
                from mflux.models.flux3.variants.txt2img.flux3_klein import Flux3Klein
                modele = Flux3Klein(quantize=bits, model_path=m["chemin"], lora_paths=chemins, lora_scales=forces,
                                    bake_lora=False)
        # Memory: same wiring as mflux/callbacks/callback_manager.py:92-129 (_register_memory_saver).
        from mflux.callbacks.instances.memory_saver import MemorySaver
        limite_octets = int(limite * 1000**3)                       # callback_manager.py:137 (decimal GB)
        if low_ram:      # callback_manager.py:92-106: transformer freed after the loop, limited cache, VAE tiling
            saver = MemorySaver(model=modele, keep_transformer=False, cache_limit_bytes=limite_octets or 1000**3,
                                num_seeds=1, keep_text_encoder=False)
        else:            # callback_manager.py:107-129: encoder freed after encoding (memory_saver.py:50), limited cache
            saver = MemorySaver(model=modele, keep_transformer=True, cache_limit_bytes=None, num_seeds=1,
                                keep_text_encoder=not liberer)
            mx.set_cache_limit(limite_octets)
            mx.clear_cache()
            mx.reset_peak_memory()
        # Before the saver: the registry calls callbacks in registration order (callback_registry.py:16-22).
        modele.callbacks.register(_EvaluerEmbeddings(mx, modele))
        modele.callbacks.register(saver)
        modele.callbacks.register(_Preemption(mx, self.attendre))
        self.modele = modele
        # Encoder or transformer freed: the model is no longer needed for another prompt, it must be reloaded.
        self.jetable = low_ram or liberer
        self.pic_charge = mx.get_peak_memory()

    def _decharger(self) -> None:
        self.modele = None
        gc.collect()
        self.mx.clear_cache()

    def _produire(self, plan: Plan) -> tuple[bytes, dict[str, Any]]:
        mx = self.mx
        classe = self.cat["modeles"][plan.modele]["classe"]
        mx.reset_peak_memory()
        if classe == "flux3klein":
            img = self.modele.generate_image(seed=plan.seed, prompt=plan.prompt, num_inference_steps=plan.etapes,
                                             height=plan.hauteur, width=plan.largeur)
        else:            # qwen21, qwen2512 and krea2 (krea2.py:51-65: same argument names)
            img = self.modele.generate_image(seed=plan.seed, prompt=plan.prompt, num_inference_steps=plan.etapes,
                                             height=plan.hauteur, width=plan.largeur, guidance=plan.guidance,
                                             scheduler=plan.horaire, negative_prompt=plan.negatif)
        tampon = io.BytesIO()
        img.image.save(tampon, format="PNG")
        return tampon.getvalue(), {"pic_gio": round(mx.get_peak_memory() / 2**30, 2)}


def _tableaux(valeur) -> list:
    """MLX arrays from a prompt cache entry (tuple, list or array only); the rest is ignored."""

    if isinstance(valeur, (tuple, list)):
        return [t for v in valeur for t in _tableaux(v)]
    return [valeur] if hasattr(valeur, "dtype") and hasattr(valeur, "shape") else []


class _EvaluerEmbeddings:
    """mflux callback registered BEFORE MemorySaver: evaluates prompt embeddings (positive and negative).

    qwen21 (qwen_image_21.py:105-112, qwen21_prompt_encoder.py:32), qwen2512 (qwen_image.py:92-102, qwen_prompt_encoder.py:41)
    and flux3klein (flux3_klein_core.py:84-85,158-164) store lazy embeddings in model.prompt_cache: the
    graph keeps the encoder weights alive until the mx.eval of the first step, even after MemorySaver
    (memory_saver.py:50-51,80-100) sets text_encoder to None. Evaluated here, before the saver, the encoder is freed by it
    before the first step. krea2 already evaluates its embeddings (krea2.py:86-88)."""

    def __init__(self, mx, modele) -> None:
        self._mx, self._modele = mx, modele

    def call_before_loop(self, seed, prompt, latents, config, canny_image=None, depth_image=None, control_images=None):
        cache = getattr(self._modele, "prompt_cache", None)
        tableaux = [t for v in (cache or {}).values() for t in _tableaux(v)]
        if tableaux:
            self._mx.eval(*tableaux)


class _Preemption:
    """mflux callback: waits for the flag before the first step (call_before_loop) then after each step, so before
    the next one (call_in_loop). Hook: mflux/models/qwen/variants/txt2img/qwen_image.py:102,130;
    qwen21/variants/txt2img/qwen_image_21.py:112,147 ; flux3/variants/flux3_klein_core.py:97,115 ;
    krea2/variants/txt2img/krea2.py:92,101; dispatch in mflux/callbacks/generation_context.py:30-69. Krea 2 passes
    "denoised" to in_loop as well, but only to callbacks that declare it: call_in_loop does not declare it.

    mflux calls ctx.in_loop BEFORE mx.eval(latents) (qwen_image_21.py:147-148): without mx.eval here, the step that just
    finished computing would still be queued when waiting. We evaluate first, so the GPU is free during the wait.
    """

    def __init__(self, mx, attendre: Callable[[], None]) -> None:
        self._mx, self._attendre = mx, attendre

    def call_before_loop(self, seed, prompt, latents, config, canny_image=None, depth_image=None, control_images=None):
        self._mx.eval(latents)
        self._attendre()

    def call_in_loop(self, t, seed, prompt, latents, config, time_steps):
        self._mx.eval(latents)
        self._attendre()


_HORAIRES_FAITS: list[bool] = []


def _enregistrer_horaires() -> None:
    """Lightning schedules of Qwen-Image-2.1, copied from the trial script gen21.py:21-59 of the validated machine (12/11)."""

    if _HORAIRES_FAITS:
        return
    import mlx.core as mx
    from mflux.models.common.schedulers import register_contrib
    from mflux.models.common.schedulers.base_scheduler import BaseScheduler
    from mflux.models.common.schedulers.linear_scheduler import LinearScheduler

    class _Fixe(BaseScheduler):
        NODES: tuple = ()
        SHIFT = False

        def __init__(self, config):
            self.config = config
            steps = config.num_inference_steps
            if steps != len(self.NODES):
                raise ValueError(f"{type(self).__name__}: {len(self.NODES)} required steps, received {steps}")
            raw = mx.array(list(self.NODES), dtype=mx.float32)
            if self.SHIFT:
                self._sigmas = LinearScheduler.shift_sigmas(raw, config, use_terminal=False)
            else:
                self._sigmas = mx.concatenate([raw, mx.zeros(1)])
            self._timesteps = mx.arange(steps, dtype=mx.float32)

        @property
        def sigmas(self):
            return self._sigmas

        @property
        def timesteps(self):
            return self._timesteps

        def step(self, noise, timestep, latents, **kwargs):
            dt = (self._sigmas[timestep + 1] - self._sigmas[timestep]).astype(latents.dtype)
            return latents + noise.astype(latents.dtype) * dt

    class Lightning1(_Fixe):
        NODES, SHIFT = (1.0, 0.75, 0.5, 0.25), True

    class Lightning2(_Fixe):
        NODES, SHIFT = (1.0, 0.95418, 0.89508, 0.704534), False

    register_contrib(Lightning1, "lightning1")
    register_contrib(Lightning2, "lightning2")
    _HORAIRES_FAITS.append(True)


@contextmanager
def _cibles_ajoutees(classe: str, cibles: list[dict[str, str]]):
    """Adds LoRA targets to the mflux mapping for the duration of loading (gen21.py:62-76; q2512lora.py:6-14), then restores."""

    if not cibles:
        yield
        return
    from mflux.models.common.lora.mapping.lora_mapping import LoRATarget
    if classe == "qwen21":
        from mflux.models.qwen21.weights.qwen21_lora_mapping import Qwen21LoRAMapping as Mapping

        def cible(c):
            return LoRATarget(model_path=c["modele"], possible_up_patterns=[f"transformer.{c['source']}.lora_B.weight"],
                              possible_down_patterns=[f"transformer.{c['source']}.lora_A.weight"])
    elif classe == "qwen2512":
        from mflux.models.qwen.weights.qwen_lora_mapping import QwenLoRAMapping as Mapping

        def cible(c):
            pre = "diffusion_model.transformer_blocks.{block}."
            return LoRATarget(model_path="transformer_blocks.{block}." + c["modele"],
                              possible_up_patterns=[f"{pre}{c['source']}.lora_B.weight"],
                              possible_down_patterns=[f"{pre}{c['source']}.lora_A.weight"])
    else:
        raise ValueError(f"unhandled additional LoRA targets for {classe}")
    original = Mapping.__dict__["get_mapping"]
    extras = [cible(c) for c in cibles]
    Mapping.get_mapping = staticmethod(lambda: list(original.__func__()) + extras)
    try:
        yield
    finally:
        Mapping.get_mapping = original


# ---------------------------------------------------------------- HTTP

def faire_serveur(port: int, moteur: Moteur, cat: dict[str, Any], hote: str = "127.0.0.1",
                  sorties: str = "") -> ThreadingHTTPServer:
    class Gestionnaire(BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            pass

        def _json(self, code: int, objet: Any) -> None:
            corps = json.dumps(objet, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(corps)))
            self.end_headers()
            self.wfile.write(corps)
            self.wfile.flush()

        def _erreur(self, code: int, message: str, type_: str) -> None:
            self._json(code, {"error": {"message": message, "type": type_}})

        def do_GET(self) -> None:
            chemin = self.path.split("?", 1)[0]
            if chemin == "/sante":
                self._json(200, {"ok": True})
            elif chemin == "/etat":
                self._json(200, {**moteur.etat(), "modeles": list(cat["modeles"]), "lora": list(cat["lora"]),
                                 "memoire_go": {n: besoin_go(cat, n) for n in cat["modeles"]}})
            else:
                self._erreur(404, "inconnu", "not_found")

        def do_POST(self) -> None:
            chemin = self.path.split("?", 1)[0].rstrip("/")
            try:
                longueur = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                longueur = -1
            if not 0 <= longueur <= LIMITE_CORPS:
                self.close_connection = True
                self._erreur(400, "Content-Length absent ou trop grand", "invalid_request_error")
                return
            corps = self.rfile.read(longueur)
            if chemin == "/annuler":
                self._json(200, {"annulee": moteur.annuler()})
                return
            if chemin == "/v1/images/edits":
                self._erreur(501, ROUTE_EDITS, "not_implemented")
                return
            if chemin != "/v1/images/generations":
                self._erreur(404, "inconnu", "not_found")
                return
            try:
                demande = json.loads(corps or b"{}")
                if not isinstance(demande, dict):
                    raise ValueError
            except ValueError:
                self._erreur(400, "corps JSON (objet) attendu", "invalid_request_error")
                return
            try:
                plan = resoudre(demande, cat)
            except ErreurDemande as exc:
                self._erreur(exc.code, exc.message, exc.type)
                return
            try:
                png, infos = moteur.generer(plan)
            except Annulee as exc:
                self._erreur(499, str(exc), "annulee")
                return
            except ErreurDemande as exc:
                self._erreur(exc.code, exc.message, exc.type)
                return
            except Exception as exc:  # noqa: BLE001 - any mflux failure becomes a JSON error
                self._erreur(500, f"{type(exc).__name__} : {exc}", "generation")
                return
            if sorties:
                try:
                    infos["fichier"] = garder_sortie(sorties, png, demande, infos)
                except OSError as exc:  # the image is still rendered; the copy failure is reported
                    infos["fichier_erreur"] = str(exc)
            self._json(200, {"created": int(time.time()),
                             "data": [{"b64_json": base64.b64encode(png).decode()}], "pli": infos})

    serveur = ThreadingHTTPServer((hote, port), Gestionnaire)
    serveur.daemon_threads = True
    return serveur


def garder_sortie(dossier: str, png: bytes, demande: dict[str, Any], infos: dict[str, Any]) -> str:
    """Keeps each produced image: DOSSIER/AAAAMMJJ/HHMMSS-modele-graine.png,
    with a .json next to it (request without attached images, generation info). Personal use, local disk only."""
    t = time.localtime()
    jour = os.path.join(dossier, time.strftime("%Y%m%d", t))
    os.makedirs(jour, exist_ok=True)
    nom = re.sub(r"[^A-Za-z0-9._-]+", "_", str(demande.get("model") or "defaut"))[:40]
    base = os.path.join(jour, f"{time.strftime('%H%M%S', t)}-{nom}-{demande.get('seed', 'x')}")
    n, chemin = 1, base + ".png"
    while os.path.exists(chemin):
        n += 1
        chemin = f"{base}-{n}.png"
    with open(chemin, "wb") as f:
        f.write(png)
    meta = {k: v for k, v in demande.items() if k not in ("image", "mask", "b64_json")}
    with open(chemin[:-4] + ".json", "w", encoding="utf-8") as f:
        json.dump({"demande": meta, "pli": infos}, f, ensure_ascii=False, indent=1)
    return chemin


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--port", type=int, required=True)
    p.add_argument("--catalogue", required=True)
    p.add_argument("--hote", default="127.0.0.1")
    p.add_argument("--simuler", action="store_true", help="no mflux import, fake steps")
    p.add_argument("--duree-etape", type=float, default=0.0)
    p.add_argument("--duree-chargement", type=float, default=0.0)
    p.add_argument("--memoire-disponible-go", type=float, default=None,
                   help="imposed available memory (tests); otherwise read with vm_stat")
    p.add_argument("--sorties", default=os.environ.get("PLI_GEN_SORTIES", ""),
                   help="folder to keep each produced image in (empty: no copy)")
    a = p.parse_args()
    cat = charger_catalogue(a.catalogue)
    drapeau = os.environ.get("PLI_GEN_DRAPEAU", "")
    memoire = (lambda: int(a.memoire_disponible_go * GO)) if a.memoire_disponible_go is not None else None
    moteur = (MoteurSimule(cat, drapeau, duree_etape=a.duree_etape, duree_chargement=a.duree_chargement,
                           memoire_disponible=memoire)
              if a.simuler else MoteurMflux(cat, drapeau, memoire))
    faire_serveur(a.port, moteur, cat, a.hote, sorties=a.sorties).serve_forever()


if __name__ == "__main__":
    main()
