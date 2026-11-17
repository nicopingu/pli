"""Guard for image and video routes (PLI 5): rejects requests targeting a real person or mentioning a minor.

Rule decided on 11/12/2026: adult content is allowed (personal adult use), but any request naming or targeting
an identifiable real person, and any request involving or mentioning minors, is rejected (400, type "guard").
The rejection message never states which word or reason triggered it. The log (JSONL) keeps neither the text nor the reason:
timestamp, family, generic reason, sha256 fingerprint of the request text.

Three levels, from fastest to slowest:
  1. local filter (word and pattern lists, French and English): always active for image and video families;
  2. real names list (JSON file, PLI_GEN_GARDE_NOMS) and "photo de / of + First Name Last Name" heuristic;
  3. optional second opinion from the general model (PLI_GEN_GARDE_LLM=1): injectable function, see brancher_avis.

Settings: PLI_GEN_GARDE_NOMS (JSON: list of strings, or {"noms": [...]}, reloaded when the file changes),
PLI_GEN_GARDE_JOURNAL (JSONL path, otherwise no log), PLI_GEN_GARDE_LLM=1 (disabled by default),
PLI_GEN_GARDE_LLM_DELAI_S (5 by default). Second opinion down, unreadable or too slow: the request passes (the guard by
rules has already filtered) and the incident is logged (avis_erreur, avis_delai) with the fingerprint only.

Known limits, by design (word filter, no understanding):
  false positives: "il y a 10 ans", "2 ans d expérience" (number < 18 followed by "ans" / "years"), "jeune fille",
  "lolita" (fashion, novel), "baby" (except baby blue/pink/oil/powder), "24 mois" (any number of months under 48), "kid"/"child" in an
  adult context ("inner child"), "teenage mutant ninja turtles" (coded exception), "photo of New York" (places and
  works from the EXCEPTIONS_NOMS list excepted, other two-word places in uppercase are rejected).
  false negatives: unlisted synonyms, heavy spelling mistakes, other languages, descriptions without keyword
  ("very small, cherubic face"), real people described without name ("the president of..."), proper names without
  "photo de" or name in the list, names written in lowercase, text encoded in an attached image (not read).
  The second opinion from the general model serves to catch some of these cases.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
import time
import unicodedata
from dataclasses import dataclass
from email.parser import BytesParser
from email.policy import default as _politique
from pathlib import Path
from typing import Any, Callable, Iterable

MINEUR = "mineur"
PERSONNE_REELLE = "personne_reelle"
AVIS_MODELE = "avis_modele"
AVIS_ERREUR = "avis_erreur"
AVIS_DELAI = "avis_delai"
DELAI_AVIS_S = 5.0
TEXTE_AVIS_MAX = 2000

MESSAGE = "demande refusée par la garde de génération (personne réelle ou mineur)"

# JSON keys whose value is a file or an image (base64, data URL): never read as text
CLES_IGNOREES = frozenset({"image", "images", "mask", "input_reference", "b64_json", "image_url", "reference", "file"})


def normaliser(texte: str) -> str:
    """Lowercase, without accents or invisible characters, punctuation and hyphens as single spaces."""

    t = unicodedata.normalize("NFKD", texte)
    t = "".join(c for c in t if not unicodedata.combining(c) and unicodedata.category(c) not in ("Cf", "Cc") or c in " \n\t")
    t = t.lower().replace("’", "'")
    t = re.sub(r"[^\w']+|_", " ", t)
    return re.sub(r"\s+", " ", t).strip()


# --- mineurs ---------------------------------------------------------------------------------------------------------

# written numbers: 0 to 17 (French and English, including without accent, after normalization)
_NOMBRES_FR = ("zero un une deux trois quatre cinq six sept huit neuf dix onze douze treize quatorze quinze seize "
               "dix sept dix huit").split()
_NOMBRES_EN = ("zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen "
               "seventeen").split()
_NOMBRES = sorted({"un", "une", "deux", "trois", "quatre", "cinq", "six", "sept", "huit", "neuf", "dix", "onze", "douze",
                   "treize", "quatorze", "quinze", "seize", "dix sept", "zero", *_NOMBRES_EN}, key=len, reverse=True)
# not preceded by a ten or a hundred: "vingt huit", "twenty one", "quatre vingt dix" are adults
_DIZAINES = ("vingt", "trente", "quarante", "cinquante", "soixante", "cent", "et", "dix", "twenty", "thirty", "forty",
             "fifty", "sixty", "seventy", "eighty", "ninety", "hundred")
_NOMBRE = "".join(f"(?<!{d} )" for d in _DIZAINES) + "(?:" + "|".join(re.escape(n) for n in _NOMBRES) + ")"
_UNITE = r"(?:ans?|annees?|mois|yo|y o|yrs?|years?|yr olds?|year olds?|years old|months?|month olds?|months old|yrs old|ans d age)"

_AGE_CHIFFRES = re.compile(rf"(?<![\d.,])(?P<n>\d{{1,2}})\s*(?P<u>{_UNITE})\b")
_AGE_LETTRES = re.compile(rf"\b{_NOMBRE}\s*(?:{_UNITE})\b")
# "aged 12", "aged 14", "age 13", "de 15 ans" is already seen by the numbers rule
_AGE_MOT = re.compile(r"\b(?:age|agee?|ages?|aged|agees?)\s*(?:de|d'|of|:)?\s*(?P<n>\d{1,2})\b(?!\s*(?:%|h|min|s\b|px|k\b))")
_AGE_MOT_LETTRES = re.compile(rf"\b(?:age|agee?|aged)\s*(?:de|d'|of)?\s*{_NOMBRE}\b")

# simple words (word boundaries), after accent-free normalization
MOTS_MINEUR = (
    # French
    "enfant", "enfants", "enfance", "enfantin", "mineur", "mineure", "mineurs", "mineures", "ado", "ados", "adolescent",
    "adolescente", "adolescents", "adolescentes", "preado", "preados", "preadolescent", "preadolescente",
    "collegien", "collegienne", "collegiens", "collegiennes", "lyceen", "lyceenne", "lyceens", "lyceennes", "lycee",
    "ecolier", "ecoliere", "ecoliers", "ecolieres", "bebe", "bebes", "gamin", "gamine", "gamins", "gamines", "gosse",
    "gosses", "fillette", "fillettes", "garconnet", "petite fille", "petit garcon", "petits garcons", "petites filles",
    "jeune fille", "jeunes filles", "jeune garcon", "jeunes garcons", "prepubere", "prepuberes", "nourrisson",
    "nourrissons", "tout petit", "tout petits", "bambin", "bambins", "minot", "minots", "minette", "ecole primaire",
    "ecole maternelle", "maternelle", "mineurs d age", "pedophile", "pedophilie", "pedo", "pedos",
    "pubere", "jeune adolescente", "jeune adolescent",
    # anglais
    "child", "children", "childs", "kid", "kids", "kiddo", "kiddie", "kiddies", "teen", "teens", "teenager", "teenagers",
    "teenage", "teenaged", "preteen", "preteens", "pre teen", "tween", "tweens", "underage", "under age", "minors",
    "toddler", "toddlers", "infant", "infants", "newborn", "newborns", "baby", "babies", "juvenile", "juveniles",
    "youngster", "youngsters", "loli", "lolis", "lolicon", "shota", "shotas", "shotacon", "jailbait", "schoolgirl",
    "schoolgirls", "schoolboy", "schoolboys", "school girl", "school girls", "school boy", "school boys", "high school",
    "highschool", "highschooler", "middle school", "middleschool", "elementary school", "kindergarten",
    "preschool", "preschooler", "primary school", "little girl", "little girls", "little boy", "little boys",
    "young girl", "young girls", "young boy", "young boys", "small girl", "small boy", "girl child", "boy child",
    "barely legal", "paedophile", "pedophile", "pedophilia", "paedo", "minor girl", "minor boy", "minor child",
    "minor female", "minor male", "young minor", "child abuse",
)
# volontairement absents (trop de faux positifs) : « cp » (cours préparatoire), « cub » (jeune ours), « primaire » (couleurs),
# "college" (adult students), "minor" alone ("a minor detail"), "elementary" alone, "girl"/"boy" alone

# forms written with repeated or separated letters ("l o l i", "lolii"): spelling variants of the clearest terms
_VARIANTES = re.compile(r"\bl o l i\b|\bs h o t a\b|\bl+o+l+i+(?:s|con|ta|tas)?\b|\bs+h+o+t+a+(?:s|con)?\b|\bj+a+i+l+b+a+i+t+\b|\bp+[ae]+d+o+p+h+i+l")
# unambiguous terms, also searched in text reduced to its letters ("u n d e r a g e", "l.o.l.i.c.o.n")
_SANS_AMBIGUITE = ("lolicon", "shotacon", "jailbait", "underage", "pedophile", "paedophile", "prepubere", "preteen")

# explicit exceptions, removed from the text before word search
EXCEPTIONS_TEXTE = (
    "teenage mutant ninja turtles", "teenage mutant", "baby blue", "baby pink", "baby oil", "baby powder", "baby blues",
    "baby back ribs", "baby driver", "baby spinach", "baby yoda", "baby doll dress", "inner child", "child of god",
    "enfant terrible", "high school musical", "bebe cadum",
)
_EXCEPTIONS = re.compile(r"\b(?:" + "|".join(re.escape(e) for e in sorted(EXCEPTIONS_TEXTE, key=len, reverse=True)) + r")\b")


def _motifs_mots(mots: Iterable[str]) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(re.escape(m) for m in sorted(set(mots), key=len, reverse=True)) + r")\b")


_MOTS = _motifs_mots(MOTS_MINEUR)


def evoque_mineur(texte: str) -> bool:
    """True if the text contains a word, an age under 18, or a spelling variant evoking a minor."""

    n = normaliser(texte)
    n = _EXCEPTIONS.sub(" ", n)
    for m in _AGE_CHIFFRES.finditer(n):
        mois = m.group("u").startswith(("mois", "month"))
        if int(m.group("n")) < (48 if mois else 18):          # "30 months" is a toddler
            return True
    for m in _AGE_MOT.finditer(n):
        if int(m.group("n")) < 18:
            return True
    if _AGE_LETTRES.search(n) or _AGE_MOT_LETTRES.search(n):
        return True
    if _MOTS.search(n) or _VARIANTES.search(n):
        return True
    compact = re.sub(r"[^a-z]", "", n)
    return any(t in compact for t in _SANS_AMBIGUITE)


# --- real persons -----------------------------------------------------------------------------------------------

_MAJ = r"[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ'’]+(?:-[A-ZÀ-ÖØ-Þ][a-zà-öø-ÿ'’]+)*"
_PARTICULE = r"(?:(?:de|du|des|van|von|di|da|del|le|la|d'|De|Van|Von|Di|Da|Del|Le|La)\s+)?"
_TITRE = r"(?:M\.|Mme|Mr|Mrs|Ms|Dr|Pr|Sir|Lady|President|Président|Présidente|Prince|Princess|Queen|King|Roi|Reine)"
_NOM_COMPLET = rf"(?:{_TITRE}\s+{_MAJ}(?:\s+{_PARTICULE}{_MAJ})*|{_MAJ}(?:\s+{_PARTICULE}{_MAJ})+)"
_DECLENCHEURS = (
    r"(?i:photos?|portraits?|pictures?|images?|photographs?|photographies?|selfies?|dessins?|drawings?|paintings?|"
    r"peintures?|videos?|vidéos?|films?|clips?|nudes?|nues?|nu|deepfakes?|deep fakes?|face swaps?|faceswaps?|lookalikes?|"
    r"look alikes?|visages?|faces?|likeness|effigie|caricatures?|statues?|bustes?|busts?)"
    r"\s+(?i:de|d'|du|of|from|with|avec|featuring|starring|de la|de l')\s*"
)
_RESSEMBLE = (
    r"(?i:ressembl\w+\s+(?:à|a|au)|looks?\s+like|looking\s+like|lookalike\s+of|in\s+the\s+likeness\s+of|as\s+played\s+by|"
    r"joué\s+par|jouée\s+par|incarné\s+par|incarnée\s+par|impersonating|imitant|dans\s+le\s+rôle\s+de)\s+"
)
HEURISTIQUE_NOM = re.compile(rf"(?:{_DECLENCHEURS}|{_RESSEMBLE}){_NOM_COMPLET}")

# words that, in a capitalized name following "photo of", indicate a place, a work or an object, not a person
EXCEPTIONS_NOMS = frozenset((
    "tour", "tower", "bridge", "pont", "mont", "mount", "mountain", "mountains", "lake", "lac", "rue", "avenue",
    "boulevard", "place", "new", "los", "san", "las", "notre", "dame", "golden", "gate", "statue", "liberty", "mona",
    "lisa", "eiffel", "great", "grand", "canyon", "palais", "chateau", "château", "cathedrale", "cathédrale", "arc",
    "triomphe", "big", "ben", "park", "parc", "square", "island", "ile", "île", "beach", "plage", "valley", "vallee",
    "vallée", "river", "fleuve", "ocean", "océan", "sea", "mer", "city", "ville", "town", "village", "castle", "temple",
    "church", "eglise", "église", "museum", "musee", "musée", "northern", "southern", "western", "eastern", "north",
    "south", "west", "east", "hong", "kong", "united", "states", "kingdom", "france", "paris", "london", "rome", "tokyo",
    "milky", "way", "lights", "monde", "world", "machu", "picchu", "taj", "mahal", "sagrada", "familia", "forbidden",
    "empire", "state", "building", "white", "house", "blue", "red", "black", "green", "mer", "rouge", "noire", "morte",
))


def _heuristique_nom(texte: str) -> bool:
    for m in HEURISTIQUE_NOM.finditer(texte):
        nom = re.search(rf"{_NOM_COMPLET}$", m.group(0))
        jetons = normaliser(nom.group(0) if nom else m.group(0)).split()
        if not any(j in EXCEPTIONS_NOMS for j in jetons):
            return True
    return False


class ListeNoms:
    """Forbidden real names, read from a JSON file (list, or {"noms": [...]}), re-read when the file changes."""

    def __init__(self, chemin: str | os.PathLike[str] | None) -> None:
        self.chemin = str(chemin) if chemin else ""
        self._signature: tuple[float, int] | None = None
        self._motif: re.Pattern[str] | None = None
        self._verrou = threading.Lock()

    def _lire(self) -> re.Pattern[str] | None:
        if not self.chemin:
            return None
        try:
            st = os.stat(self.chemin)
        except OSError:
            self._signature = None
            return None
        signature = (st.st_mtime, st.st_size)
        if signature == self._signature:
            return self._motif
        try:
            with open(self.chemin, encoding="utf-8") as f:
                donnees = json.load(f)
        except (OSError, ValueError) as exc:
            # unreadable file: keep the previous list rather than raising the guard
            print(f"[tensorfold] garde : liste de noms illisible ({type(exc).__name__})", file=sys.stderr, flush=True)
            return self._motif
        noms = donnees.get("noms", []) if isinstance(donnees, dict) else donnees
        formes: set[str] = set()
        for brut in noms if isinstance(noms, list) else []:
            if not isinstance(brut, str):
                continue
            n = normaliser(brut)
            if len(n) < 3:
                continue
            formes.add(n)
            jetons = n.split()
            if len(jetons) == 2:
                formes.add(f"{jetons[1]} {jetons[0]}")        # "First Last"
        self._signature = signature
        self._motif = _motifs_mots(formes) if formes else None
        return self._motif

    def contient(self, texte: str) -> bool:
        with self._verrou:
            motif = self._lire()
        return bool(motif and motif.search(normaliser(texte)))


def vise_personne_reelle(texte: str, noms: ListeNoms | None = None) -> bool:
    return bool((noms is not None and noms.contient(texte)) or _heuristique_nom(texte))


# --- request text extraction ---------------------------------------------------------------------------------

def _chaines(valeur: Any, cle: str = "") -> Iterable[str]:
    if cle in CLES_IGNOREES:
        return
    if isinstance(valeur, str):
        if not valeur.startswith("data:"):
            yield valeur
    elif isinstance(valeur, dict):
        for k, v in valeur.items():
            yield from _chaines(v, str(k))
    elif isinstance(valeur, list):
        for v in valeur:
            yield from _chaines(v, cle)


def textes_de(corps: bytes, type_contenu: str = "") -> list[str]:
    """Request texts: strings from a JSON body (excluding files), or non-file fields from a multipart body."""

    if not corps:
        return []
    if "multipart/" in type_contenu.lower():
        brut = b"Content-Type: " + type_contenu.encode("latin-1", "replace") + b"\r\nMIME-Version: 1.0\r\n\r\n" + corps
        try:
            message = BytesParser(policy=_politique).parsebytes(brut)
            textes = []
            for partie in message.iter_parts():
                if partie.get_filename() is None and partie.get_content_maintype() in ("text", "application"):
                    nom = partie.get_param("name", header="content-disposition") or ""
                    if nom in CLES_IGNOREES:
                        continue
                    charge = partie.get_payload(decode=True) or b""
                    textes.append(charge.decode("utf-8", "replace"))
            return textes
        except Exception:                                    # malformed body: raw text read, without file bytes
            return [corps.decode("utf-8", "replace")[:200_000]]
    try:
        donnees = json.loads(corps)
    except ValueError:
        return []                  # the child will reject this body (400): it cannot read it as a request either
    return list(_chaines(donnees))


# --- log and second opinion --------------------------------------------------------------------------------------------

def empreinte(textes: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(textes).encode("utf-8", "replace")).hexdigest()


class Journal:
    """JSONL log of refusals: never the text, nor the word that triggered the refusal."""

    def __init__(self, chemin: str | os.PathLike[str] | None) -> None:
        self.chemin = str(chemin) if chemin else ""
        self._verrou = threading.Lock()

    def ecrire(self, famille: str, raison: str, sha256: str) -> None:
        if not self.chemin:
            return
        ligne = json.dumps({"horodatage": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "famille": famille, "raison": raison,
                            "sha256": sha256}, ensure_ascii=False)
        try:
            with self._verrou:
                Path(self.chemin).parent.mkdir(parents=True, exist_ok=True)
                with open(self.chemin, "a", encoding="utf-8") as f:
                    f.write(ligne + "\n")
        except OSError as exc:
            print(f"[tensorfold] garde : journal illisible ({type(exc).__name__})", file=sys.stderr, flush=True)


QUESTION_AVIS = ("Réponds par un seul mot, OUI ou NON. Cette demande de génération d image ou de vidéo décrit-elle un "
                 "mineur, une personne réelle identifiable (célébrité, personnalité, personne privée nommée) ou du "
                 "contenu illégal ?\n\nDemande :\n")

# Second opinion function: receives the request text, returns True to refuse. Set by the application (see
# brancher_avis); if absent, the second opinion has no effect even with PLI_GEN_GARDE_LLM=1.
_avis: Callable[[str], bool] | None = None


def brancher_avis(fonction: Callable[[str], bool] | None) -> None:
    global _avis
    _avis = fonction


def _oui(reponse: dict[str, Any]) -> bool:
    return normaliser(str(reponse.get("content") or "")).startswith(("oui", "yes"))


def avis_par_chat(chat: Callable[..., dict[str, Any]]) -> Callable[[str], bool]:
    """Builds the opinion function from a chat call: closed question, 4 tokens, temperature 0."""

    def avis(texte: str) -> bool:
        reponse = chat([{"role": "user", "content": QUESTION_AVIS + texte[:TEXTE_AVIS_MAX]}], max_tokens=4, temperature=0.0)
        return _oui(reponse)

    return avis


def delai_avis() -> float:
    try:
        return max(0.1, float(os.environ.get("PLI_GEN_GARDE_LLM_DELAI_S", DELAI_AVIS_S)))
    except ValueError:
        return DELAI_AVIS_S


def avis_general(app: Any, superviseur: Any, delai_s: float | None = None) -> Callable[[str], bool]:
    """Opinion from the 27B of the same process: ChatApp.chat (server/app.py:246) without reflection (enable_thinking False,
    app.py:315), temperature 0, 4 tokens, with timeout (TimeoutError beyond: the request is cancelled in the engine).

    The preemption flag (superviseur.general_debut / general_fin, like a chat POST in http.py:223) is
    raised only during the question, in the worker thread, and always returned in its finally: image and video children
    wait at most the duration of the question, never indefinitely. The guard is not called with the flag raised
    (routes.servir does not raise it before the guard): no deadlock."""

    def avis(texte: str) -> bool:
        from pli.server.cancellation import Cancellation

        annulation = Cancellation()
        sortie: dict[str, Any] = {}

        def travail() -> None:
            superviseur.general_debut()
            try:
                sortie["reponse"] = app.chat(
                    [{"role": "user", "content": QUESTION_AVIS + texte[:TEXTE_AVIS_MAX]}], max_tokens=4,
                    temperature=0.0, sampling={"enable_thinking": False}, cancellation=annulation)
            except BaseException as exc:                     # noqa: BLE001 - raised in the calling thread
                sortie["erreur"] = exc
            finally:
                superviseur.general_fin()

        fil = threading.Thread(target=travail, name="garde-avis", daemon=True)
        fil.start()
        fil.join(delai_s if delai_s is not None else delai_avis())
        if fil.is_alive():
            annulation.cancel()
            raise TimeoutError("second avis trop lent")
        if "erreur" in sortie:
            raise sortie["erreur"]
        return _oui(sortie["reponse"])

    return avis


# --- garde ---------------------------------------------------------------------------------------------------------------

@dataclass
class Verdict:
    refuse: bool
    raison: str = ""
    sha256: str = ""


FAMILLES_GARDEES = frozenset(("image", "video"))


class Garde:
    def __init__(self, noms: ListeNoms | None = None, journal: Journal | None = None, llm: bool = False,
                 avis: Callable[[str], bool] | None = None) -> None:
        self.noms, self.journal, self.llm, self.avis = noms, journal, llm, avis

    @classmethod
    def depuis_env(cls) -> "Garde":
        return cls(ListeNoms(os.environ.get("PLI_GEN_GARDE_NOMS")), Journal(os.environ.get("PLI_GEN_GARDE_JOURNAL")),
                   os.environ.get("PLI_GEN_GARDE_LLM") == "1")

    def verifier(self, famille: str, corps: bytes, type_contenu: str = "",
                 avis: Callable[[str], bool] | None = None) -> Verdict:
        if famille not in FAMILLES_GARDEES:
            return Verdict(False)
        textes = [t for t in textes_de(corps, type_contenu) if t.strip()]
        if not textes:
            return Verdict(False)
        sha = empreinte(textes)
        tout = "\n".join(textes)
        raison = ""
        incident = ""
        if evoque_mineur(tout):
            raison = MINEUR
        elif vise_personne_reelle(tout, self.noms):
            raison = PERSONNE_REELLE
        elif self.llm:
            fonction = self.avis or avis or _avis
            if fonction is None:
                print("[tensorfold] garde : PLI_GEN_GARDE_LLM=1 mais aucun avis branché", file=sys.stderr, flush=True)
            else:
                try:
                    if fonction(tout):
                        raison = AVIS_MODELE
                except Exception as exc:                     # opinion down or too slow: the request passes, the incident is logged
                    print(f"[tensorfold] garde : avis en échec ({type(exc).__name__})", file=sys.stderr, flush=True)
                    incident = AVIS_DELAI if isinstance(exc, TimeoutError) else AVIS_ERREUR
        if not raison:
            if incident and self.journal is not None:
                self.journal.ecrire(famille, incident, sha)
            return Verdict(False, sha256=sha)
        if self.journal is not None:
            self.journal.ecrire(famille, raison, sha)
        return Verdict(True, raison, sha)


__all__ = ["Garde", "Verdict", "ListeNoms", "Journal", "MESSAGE", "brancher_avis", "avis_par_chat", "avis_general", "evoque_mineur",
           "vise_personne_reelle", "textes_de", "normaliser"]
