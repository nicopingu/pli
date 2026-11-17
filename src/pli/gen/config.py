"""Configuration of generation families (PLI_GEN_CONFIG, JSON file) and activation (PLI_GEN=1).

File format:

    {"reserve_go": 6,
     "familles": {"image": {"argv": ["{python}", "-m", "pli.gen.fils_simule", "--port", "{port}"],
                            "port": 18101, "memoire_go": 8, "delai_decharge_s": 300, "sante_delai_s": 60}}}

Famille video : {"argv": ["{python}", "-m", "pli.gen.fils_video", "--port", "{port}", "--catalogue", CHEMIN,
"--sorties", DOSSIER], "memoire_go": 12, "delai_decharge_s": 0, ...} (see gen/exemple_config.json and
docs/gen/catalogue-video.exemple.json); 12 GB = need with --ssd-streaming, the video then runs next to the general model.

« {python} » becomes the server interpreter, « {port} » the family port. Memory: « memoire_go » or
« memoire_octets ». Optional file key: seuil_echange_mo.
Optional catalogue_prive key (path or list of private LoRA TOML files, gen/catalogue_prive.py): passed to child processes
only (PLI_GEN_CATALOGUE_PRIVE in their environment), to avoid changing the server checkpoint key.
The voice assistant's role during a video (optional file keys, absent = previous behavior):
  video_parallele_general (integer >= 1): general model generations at once while a video job runs;
  video_reserve_voix (boolean): a video only starts if memory also covers the audio family (otherwise it waits,
    « memoire_voix »); video_reserve_voix_go (reserve, default: that of the audio family); video_attente_memoire_s;
  video_pause_voix (boolean): h3 paused during a voice or transcription.
Room asked of the bench queue (PLI 5.6.4, optional): place_fichier (path of the request file the queue reads),
  place_attente_s (wait for the memory, default 120). Optional keys: relais_delai_s, surveiller_echange, reserve_go (family reserve, replaces file one).
Family memory: the requirement of its smallest useful model; the child then compares available memory to
each model's requirement (catalogue « memoire_go ») before loading it and responds 503 « memoire » otherwise.
"""

from __future__ import annotations

import atexit
import json
import os
import sys
from typing import Any

from pli.gen.superviseur import GO, Famille, Superviseur

NOMS = ("audio", "image", "video")


def actif() -> bool:
    return os.environ.get("PLI_GEN") == "1"


def familles_depuis(donnees: dict[str, Any]) -> dict[str, Famille]:
    brut = donnees.get("familles")
    if not isinstance(brut, dict) or not brut:
        raise ValueError("PLI_GEN_CONFIG: « familles » key missing or empty")
    familles: dict[str, Famille] = {}
    for nom, c in brut.items():
        if nom not in NOMS:
            raise ValueError(f"PLI_GEN_CONFIG : famille inconnue « {nom} » (attendu : {', '.join(NOMS)})")
        if not isinstance(c.get("argv"), list) or not c["argv"] or not isinstance(c.get("port"), int):
            raise ValueError(f"PLI_GEN_CONFIG: « {nom} » requires « argv » (list) and « port » (integer)")
        argv = [sys.executable if a == "{python}" else str(a) for a in c["argv"]]
        octets = int(c["memoire_octets"]) if "memoire_octets" in c else int(float(c.get("memoire_go", 0)) * GO)
        extra = {k: c[k] for k in ("delai_decharge_s", "sante_delai_s", "relais_delai_s", "surveiller_echange") if k in c}
        if "delai_decharge_s" in extra:
            extra["delai_decharge_s"] = float(extra["delai_decharge_s"])
        if "reserve_go" in c:
            extra["reserve_octets"] = int(float(c["reserve_go"]) * GO)
        familles[nom] = Famille(nom=nom, argv=argv, port=c["port"], memoire_octets=octets, **extra)
    return familles


def verifier_catalogues_prives(familles: dict[str, Famille], environ=os.environ) -> dict[str, list[str]]:
    """At startup: reads PLI_GEN_CATALOGUE_PRIVE (gen/catalogue_prive.py) and refuses a name already public.

    Public catalogues are those received by child processes (« --catalogue CHEMIN » in their argv). The child redoes the
    same merge at its launch; this control makes startup fail rather than the first request. Returns
    private names by family (log)."""

    from pli.gen import catalogue_prive

    liste = catalogue_prive.chemins(environ)
    if not liste:
        return {}
    privees = catalogue_prive.lire(liste)
    noms: dict[str, list[str]] = {}
    for famille, entrees in privees.items():
        if not entrees:
            continue
        noms[famille] = sorted(entrees)
        fam = familles.get(famille)
        if fam is None:
            print(f"[gen] {catalogue_prive.ENV}: private LoRA {famille} without configured family {famille}: "
                  f"{', '.join(sorted(entrees))}", file=sys.stderr, flush=True)
            continue
        if "--catalogue" in fam.argv and fam.argv.index("--catalogue") + 1 < len(fam.argv):
            with open(fam.argv[fam.argv.index("--catalogue") + 1], encoding="utf-8") as f:
                public = json.load(f)
            catalogue_prive.fusionner(public, famille, environ)      # ValueError on a name already public
    return noms


def superviseur_depuis_env() -> Superviseur:
    chemin = os.environ.get("PLI_GEN_CONFIG", "")
    if not chemin:
        raise ValueError("PLI_GEN=1 requires PLI_GEN_CONFIG (path to the families JSON file)")
    with open(chemin, encoding="utf-8") as f:
        donnees = json.load(f)
    reserve = None
    if "PLI_GEN_RESERVE_GO" not in os.environ and "reserve_go" in donnees:
        reserve = int(float(donnees["reserve_go"]) * GO)
    extra_sup = {}
    if "seuil_echange_mo" in donnees:          # tolerated swap increase before stopping a child (default 64 Mo)
        extra_sup["seuil_echange_octets"] = int(float(donnees["seuil_echange_mo"]) * 1024 * 1024)
    if "echange_dispo_min_go" in donnees:     # swap only stops a child if available memory is also below this threshold
        extra_sup["dispo_min_echange_octets"] = int(float(donnees["echange_dispo_min_go"]) * GO)
    if "video_parallele_general" in donnees:
        n = int(donnees["video_parallele_general"])
        if n < 1:
            raise ValueError("PLI_GEN_CONFIG: « video_parallele_general » must be at least 1")
        extra_sup["parallele_video"] = n
    for cle, nom in (("video_reserve_voix", "reserve_voix"), ("video_pause_voix", "pause_voix")):
        if cle in donnees:
            if not isinstance(donnees[cle], bool):
                raise ValueError(f"PLI_GEN_CONFIG : « {cle} » doit valoir true ou false")
            extra_sup[nom] = donnees[cle]
    if "video_reserve_voix_go" in donnees:
        extra_sup["reserve_voix_octets"] = int(float(donnees["video_reserve_voix_go"]) * GO)
    if "video_attente_memoire_s" in donnees:
        extra_sup["attente_memoire_s"] = float(donnees["video_attente_memoire_s"])
    # PLI 5.6.4: room asked of the bench queue when a child lacks memory (file read by the queue's probe); in this
    # file rather than as PLI_* variables, which would change the checkpoint key
    if donnees.get("place_fichier"):
        if not isinstance(donnees["place_fichier"], str):
            raise ValueError("PLI_GEN_CONFIG : « place_fichier » : chemin du fichier de demande de place")
        extra_sup["place_fichier"] = donnees["place_fichier"]
    if "place_attente_s" in donnees:
        extra_sup["place_attente_s"] = float(donnees["place_attente_s"])
    familles = familles_depuis(donnees)
    # « catalogue_prive » key (path or list): passed to child processes only, via PLI_GEN_CATALOGUE_PRIVE in their
    # environment. The server does not receive it as a variable: an extra PLI_* in its environment would change
    # the settings fingerprint of the checkpoint key (pli/code_key.py), and checkpoints would no longer resume.
    env_fils = {}
    cp = donnees.get("catalogue_prive")
    if cp:
        liste = [cp] if isinstance(cp, str) else cp
        if not isinstance(liste, list) or not all(isinstance(x, str) and x for x in liste):
            raise ValueError("PLI_GEN_CONFIG : « catalogue_prive » : chemin ou liste de chemins de fichiers TOML")
        if os.environ.get("PLI_GEN_CATALOGUE_PRIVE"):
            raise ValueError("PLI_GEN_CONFIG : « catalogue_prive » et la variable PLI_GEN_CATALOGUE_PRIVE : une seule des deux")
        env_fils["PLI_GEN_CATALOGUE_PRIVE"] = os.pathsep.join(os.path.expanduser(x) for x in liste)
    if env_fils:
        extra_sup["env"] = env_fils
    prives = verifier_catalogues_prives(familles, {**os.environ, **env_fils})
    if prives:
        print("[gen] Private LoRAs (PLI_GEN_CATALOGUE_PRIVE) : "
              + " ; ".join(f"{f} {len(n)}" for f, n in sorted(prives.items())), file=sys.stderr, flush=True)
    superviseur = Superviseur(familles, reserve_octets=reserve, **extra_sup)
    atexit.register(superviseur.arreter)              # safety net: exit without going through cli.py's finally
    return superviseur
