"""Private LoRA catalogues (PLI_GEN_CATALOGUE_PRIVE): TOML files outside the repository, added to the public catalogues.

PLI_GEN_CATALOGUE_PRIVE: one or more TOML file paths, separated by « : » (os.pathsep). Absent or empty:
nothing changes. The supervisor sets it in the child processes' environment from the « catalogue_prive » key of
PLI_GEN_CONFIG (gen/config.py): in the server environment, this PLI_* would change the checkpoints' keys. Standard library only (tomllib, Python 3.11 or newer): this module is also loaded by
path by the image child process, which runs in a different Python environment and does not import pli.

One section per LoRA, under [image.NOM] or [video.NOM]:

    [image.mon-lora]
    fichier = "/chemin/mon-lora.safetensors"      # obligatoire
    base = "krea2"                                # model(s) from the image catalogue: text or list (required)
    force = 1.0                                   # optional (default 1.0)
    cibles = [{source = "img_mod.1", modele = "img_mod_linear"}]   # facultatif (qwen21, qwen2512)
    defauts = {passes = 8, taille = "1024x1024", guidance = 1.0}   # facultatif

    [video.mon-lora-video]
    fichier = "/chemin/lora.safetensors"          # obligatoire
    base = "h3"                                   # optional (default: h3)
    force = 1.0
    mode = "sauvegardee"                          # or « a_la_demande » (default)
    taille_go = 62                                # optional: disk space for a new merge
    defauts = {passes = 8, taille = "512x512", secondes = 2}

« defauts »: values used when a request names this LoRA without providing them (steps, size, guidance for image;
steps, size, seconds for video); the request always takes precedence. A name already present in the public catalogue, or
in two private files, is refused (ValueError): never silent overwriting. Any unknown key is refused.
Entries keep « prive »: true. The route guard (gen/garde.py) applies to requests as before.
"""

from __future__ import annotations

import os
import re
from typing import Any

ENV = "PLI_GEN_CATALOGUE_PRIVE"
FAMILLES = ("image", "video")
RE_TAILLE = re.compile(r"^\d{2,5}x\d{2,5}$")
CLES = {
    "image": {"fichier", "base", "force", "cibles", "defauts"},
    "video": {"fichier", "base", "force", "mode", "taille_go", "defauts"},
}
DEFAUTS = {"image": {"passes", "taille", "guidance"}, "video": {"passes", "taille", "secondes"}}
MODES = ("sauvegardee", "a_la_demande", "en_memoire")


def chemins(environ=os.environ) -> list[str]:
    brut = environ.get(ENV, "") or ""
    return [os.path.expanduser(c.strip()) for c in brut.split(os.pathsep) if c.strip()]


def _toml(chemin: str) -> dict[str, Any]:
    try:
        import tomllib                                    # Python 3.11 and newer
    except ImportError:                                   # pragma: no cover - Python 3.10 and older
        try:
            import tomli as tomllib
        except ImportError:
            raise ValueError(f"{ENV} : reading TOML requires Python 3.11 or newer (or the tomli package)") from None
    try:
        with open(chemin, "rb") as f:
            return tomllib.load(f)
    except OSError as e:
        raise ValueError(f"{ENV} : {chemin} illisible ({e.strerror})") from None
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"{ENV} : {chemin} : invalid TOML ({e})") from None


def _nombre(v: Any, ou: str, bas: float, haut: float, entier: bool = False) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)) or (entier and not isinstance(v, int)) or not bas <= v <= haut:
        genre = "an integer" if entier else "a number"
        raise ValueError(f"{ou} : {genre} between {bas:g} and {haut:g} expected, not {v!r}")
    return v


def _entree(famille: str, nom: str, s: Any, ou: str) -> dict[str, Any]:
    if not isinstance(s, dict):
        raise ValueError(f"{ou} : [{famille}.{nom}] must be a table")
    inconnues = set(s) - CLES[famille]
    if inconnues:
        raise ValueError(f"{ou} : [{famille}.{nom}] : unknown key(s) {', '.join(sorted(inconnues))} "
                         f"(allowed: {', '.join(sorted(CLES[famille]))})")
    if not isinstance(s.get("fichier"), str) or not s["fichier"]:
        raise ValueError(f"{ou} : [{famille}.{nom}] : « fichier » (LoRA path) required")
    base = s.get("base", "h3" if famille == "video" else None)
    base = [base] if isinstance(base, str) else base
    if not isinstance(base, list) or not base or not all(isinstance(b, str) and b for b in base):
        raise ValueError(f"{ou} : [{famille}.{nom}] : « base » (model or list of models) required")
    e: dict[str, Any] = {"chemin": os.path.expanduser(s["fichier"]), "base": base, "prive": True,
                         "force": float(_nombre(s.get("force", 1.0), f"{ou} : [{famille}.{nom}] force", -10, 10))}
    if famille == "image" and "cibles" in s:
        c = s["cibles"]
        if not isinstance(c, list) or not all(isinstance(x, dict) and set(x) == {"source", "modele"} for x in c):
            raise ValueError(f"{ou} : [image.{nom}] : « cibles »: list of {{source = ..., modele = ...}}")
        e["cibles"] = [dict(x) for x in c]
    if famille == "video":
        e["mode"] = s.get("mode", "a_la_demande")
        if e["mode"] not in MODES:
            raise ValueError(f"{ou} : [video.{nom}] : « mode » one of {', '.join(MODES)}")
        if "taille_go" in s:
            e["taille_go"] = float(_nombre(s["taille_go"], f"{ou} : [video.{nom}] taille_go", 0.001, 10000))
    d = s.get("defauts", {})
    if not isinstance(d, dict):
        raise ValueError(f"{ou} : [{famille}.{nom}] : « defauts » must be a table")
    inconnues = set(d) - DEFAUTS[famille]
    if inconnues:
        raise ValueError(f"{ou} : [{famille}.{nom}].defauts : unknown key(s) {', '.join(sorted(inconnues))} "
                         f"(allowed: {', '.join(sorted(DEFAUTS[famille]))})")
    defauts: dict[str, Any] = {}
    if "passes" in d:
        defauts["passes"] = int(_nombre(d["passes"], f"{ou} : [{famille}.{nom}] passes", 1, 200 if famille == "image" else 1000, True))
    if "taille" in d:
        if not isinstance(d["taille"], str) or not RE_TAILLE.match(d["taille"]):
            raise ValueError(f"{ou} : [{famille}.{nom}] taille: « WIDTHxHEIGHT » expected, not {d['taille']!r}")
        defauts["taille"] = d["taille"]
    if "guidance" in d:
        defauts["guidance"] = float(_nombre(d["guidance"], f"{ou} : [image.{nom}] guidance", 0, 100))
    if "secondes" in d:
        defauts["secondes"] = float(_nombre(d["secondes"], f"{ou} : [video.{nom}] secondes", 0.01, 3600))
    if defauts:
        e["defauts"] = defauts
    return e


def lire(liste: list[str]) -> dict[str, dict[str, dict[str, Any]]]:
    """File entries, by family; a name present in two files is refused."""

    out: dict[str, dict[str, dict[str, Any]]] = {f: {} for f in FAMILLES}
    origine: dict[tuple[str, str], str] = {}
    for chemin in liste:
        donnees = _toml(chemin)
        inconnues = set(donnees) - set(FAMILLES)
        if inconnues:
            raise ValueError(f"{ENV} : {chemin} : unknown section(s) {', '.join(sorted(inconnues))} "
                             f"(allowed: [image.NOM], [video.NOM])")
        for famille in FAMILLES:
            sections = donnees.get(famille, {})
            if not isinstance(sections, dict):
                raise ValueError(f"{ENV} : {chemin} : [{famille}] must contain tables [{famille}.NOM]")
            for nom, s in sections.items():
                if (famille, nom) in origine:
                    raise ValueError(f"{ENV} : LoRA {famille} « {nom} » defined twice ({origine[(famille, nom)]} and {chemin})")
                origine[(famille, nom)] = chemin
                out[famille][nom] = _entree(famille, nom, s, f"{ENV} : {chemin}")
    return out


def entrees(famille: str, environ=os.environ) -> dict[str, dict[str, Any]]:
    liste = chemins(environ)
    return lire(liste)[famille] if liste else {}


def fusionner(catalogue: dict[str, Any], famille: str, environ=os.environ) -> dict[str, Any]:
    """Adds the family's private LoRAs to the catalog (before validation); a public name is rejected."""

    privees = entrees(famille, environ)
    if not privees:
        return catalogue
    loras = catalogue.get("lora") or {}
    if not isinstance(loras, dict):
        raise ValueError("catalog: 'lora' must be an object")
    doublons = sorted(set(privees) & set(loras))
    if doublons:
        raise ValueError(f"{ENV}: LoRA {famille} already in the public catalog: {', '.join(doublons)} "
                         f"(a private name does not replace a public name)")
    catalogue["lora"] = {**loras, **privees}
    return catalogue


__all__ = ["ENV", "chemins", "lire", "entrees", "fusionner"]
