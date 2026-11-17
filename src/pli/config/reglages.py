"""Settings table, conversions, reading and writing of ``pli.toml``, and preparation of ``pli serve``.

The table (``table.json``) is produced by ``tools/pli-reglages-inventaire``. A ``family.setting`` key in it names either
a ``pli serve`` option (``cache.context`` : ``--context``) or a variable read by the code
(``cache.checkpoint_bits`` : ``PLI_CHECKPOINT_BITS``). This module only translates to those names: the code that reads
the settings does not change.

Priority order: option > variable > pli.toml > machine profile > code default (docs/reglages.md, 2.4).
"""

from __future__ import annotations

import difflib
import json
import os
import re
import subprocess
import tomllib
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Callable, MutableMapping

TABLE = Path(__file__).with_name("table.json")
VARIABLE_FICHIER = "PLI_CONFIG"
FICHIER_DEFAUT = "pli.toml"           # for config:* commands only; pli serve only reads a named file


# ---------------------------------------------------------------------------------------------------- table
@lru_cache(maxsize=1)
def table() -> dict:
    return json.loads(TABLE.read_text())


def reglages() -> list[dict]:
    return table()["reglages"]


def familles() -> list[str]:
    return table()["familles"]


@lru_cache(maxsize=1)
def _index() -> tuple[dict, dict, dict, dict, dict]:
    cles = {r["cle"]: r for r in reglages()}
    derives = {r["variable"]: r for r in reglages() if r["source"] == "derive"}
    variables = {r["variable"]: r for r in reglages() if r["source"] == "variable"}
    alias: dict[str, dict] = {}
    for r in reglages():
        nom = r.get("alias") if r["source"] == "variable" else r.get("variable")
        if nom:
            alias[nom] = r
    options = {o: r for r in reglages() if r["source"] == "option" for o in r["options"]}
    return cles, variables, alias, options, derives


def par_derive(nom: str) -> dict | None:
    """A setting that the code does not read under its name (attention.per_row_always), by its variable."""

    return _index()[4].get(nom)


def derives() -> list[dict]:
    return list(_index()[4].values())


def statuts() -> tuple[str, ...]:
    return ("recommande", "experimental", "obsolete")


def par_cle(cle: str) -> dict | None:
    return _index()[0].get(cle)


def par_variable(nom: str) -> dict | None:
    return _index()[1].get(nom)


def par_alias(nom: str) -> dict | None:
    return _index()[2].get(nom)


def anciens_noms() -> dict[str, str]:
    from pli.additions import env_names

    return env_names.NEW_NAME


def trouver(nom: str) -> dict | None:
    """A setting by its key, its variable (code name, name derived from the key or old name) or its option."""

    r = par_cle(nom) or par_variable(nom) or par_alias(nom) or _index()[3].get(nom) or par_derive(nom)
    if r is None and nom in anciens_noms():
        r = par_variable(anciens_noms()[nom])
    if r is None and nom.startswith("--"):
        r = par_cle(nom[2:])
    return r


def proposer(nom: str) -> str:
    connus = list(_index()[0]) + list(_index()[1])
    proches = difflib.get_close_matches(nom, connus, n=3, cutoff=0.6)
    return f" (did you mean: {', '.join(proches)}?)" if proches else ""


def famille_et_nom(r: dict) -> tuple[str, str]:
    return r["famille"], r["cle"][len(r["famille"]) + 1:]


# ---------------------------------------------------------------------------------------------------- valeurs
_ENTIER = re.compile(r"-?(0|[1-9][0-9]*)")
VRAI = {"1", "true", "yes", "on", "oui"}
FAUX = {"0", "false", "no", "off", "non"}


def vers_texte(r: dict, v: Any) -> str:
    """Value from pli.toml or the command line -> text of the variable read by the code."""

    if isinstance(v, bool):
        return "1" if v else "0"
    if isinstance(v, int):
        return str(v)
    if isinstance(v, float):
        return repr(v)
    if isinstance(v, str):
        return v
    raise ValueError(f"{r['cle']} : a simple value is expected (text, number, true/false), not {type(v).__name__}")


def valeur_option(r: dict, v: Any) -> Any:
    """Value from pli.toml, a variable or the command line -> parser default for the setting's option."""

    action = r.get("action", "")
    if action in ("_StoreTrueAction", "BooleanOptionalAction"):
        if isinstance(v, bool):
            return v
        t = str(v).strip().lower()
        if t in VRAI:
            return True
        if t in FAUX:
            return False
        raise ValueError(f"{r['cle']} : true or false expected, not {v!r}")
    if action == "_AppendAction":
        if isinstance(v, list):
            return [str(x) for x in v]
        return [x.strip() for x in str(v).split(",") if x.strip()]
    if isinstance(v, (list, dict)):
        raise ValueError(f"{r['cle']} : a simple value is expected")
    texte = vers_texte(r, v)
    erreurs, _ = verifier(r, texte)
    if erreurs:
        raise ValueError("; ".join(erreurs))
    return texte                                    # argparse applies the option type to a text default


def verifier(r: dict, texte: str) -> tuple[list[str], list[str]]:
    """(errors, warnings) for the value ``texte`` of the setting ``r``."""

    erreurs: list[str] = []
    avert: list[str] = []
    typ = r["type"]
    t = texte.strip()
    if typ == "entier" and t and not _ENTIER.fullmatch(t.lstrip("+")):
        erreurs.append(f"{r['cle']} : integer expected, not {texte!r}")
    elif typ == "reel" and t:
        try:
            float(t)
        except ValueError:
            erreurs.append(f"{r['cle']} : number expected, not {texte!r}")
    elif typ == "drapeau" and r["source"] == "variable":
        admis = {"0", "1", ""} | set(r.get("choix") or ())
        if t not in admis:
            avert.append(f"{r['cle']} : the code compares this value to {'/'.join(sorted(admis - {''}))} ; "
                         f"{texte!r} is not part of it")
    elif typ == "choix":
        admis = set(r.get("choix") or ())
        if r["source"] == "option" and t not in admis:
            erreurs.append(f"{r['cle']} : a value among {', '.join(sorted(admis))} is expected, not {texte!r}")
        elif r["source"] == "variable" and t not in admis | {"", "0"}:
            avert.append(f"{r['cle']} : the code compares this value to {', '.join(sorted(admis))} ; "
                         f"{texte!r} is not part of it")
    return erreurs, avert


def depuis_texte(r: dict, texte: str) -> Any:
    """Variable text -> value to write in pli.toml, such that vers_texte renders it identically."""

    if r["type"] == "drapeau" and texte in ("0", "1"):
        v: Any = texte == "1"
    elif _ENTIER.fullmatch(texte):
        v = int(texte)
    else:
        try:
            f = float(texte)
            v = f if repr(f) == texte else texte
        except ValueError:
            v = texte
    return v if vers_texte(r, v) == texte else texte


# ---------------------------------------------------------------------------------------------------- pli.toml
def lire_toml(chemin: Path | str) -> dict[str, Any]:
    """{dotted key: value} from a pli.toml (nested tables flattened)."""

    donnees = tomllib.loads(Path(chemin).expanduser().read_text())
    sortie: dict[str, Any] = {}

    def plat(d: dict, prefixe: str) -> None:
        for k, v in d.items():
            cle = f"{prefixe}.{k}" if prefixe else k
            if isinstance(v, dict):
                plat(v, cle)
            else:
                sortie[cle] = v

    plat(donnees, "")
    return sortie


def cles_inconnues(valeurs: dict[str, Any]) -> list[str]:
    return [c for c in valeurs if par_cle(c) is None]


def valeur_toml(v: Any) -> str:
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(valeur_toml(x) for x in v) + "]"
    return json.dumps(str(v), ensure_ascii=False)


def texte_toml(valeurs: dict[str, Any], entete: str = "") -> str:
    """A complete pli.toml: one section per family, in family order, keys sorted."""

    lignes = [f"# {l}" if l else "#" for l in entete.splitlines()] if entete else []
    for fam in familles():
        cles = sorted(c for c in valeurs if par_cle(c) and par_cle(c)["famille"] == fam)
        if not cles:
            continue
        if lignes:
            lignes.append("")
        lignes.append(f"[{fam}]")
        for c in cles:
            lignes.append(f"{famille_et_nom(par_cle(c))[1]} = {valeur_toml(valeurs[c])}")
    return "\n".join(lignes) + "\n"


def ecrire_cle(chemin: Path | str, cle: str, valeur: Any | None) -> None:
    """Set (or remove, value None) a key in pli.toml while keeping the rest of the file and its comments."""

    r = par_cle(cle)
    if r is None:
        raise ValueError(f"unknown key: {cle}{proposer(cle)}")
    chemin = Path(chemin).expanduser()
    lignes = chemin.read_text().splitlines() if chemin.is_file() else []
    fam, nom = famille_et_nom(r)
    debut = next((i for i, l in enumerate(lignes) if l.strip() == f"[{fam}]"), None)
    nouvelle = None if valeur is None else f"{nom} = {valeur_toml(valeur)}"
    if debut is None:
        if nouvelle is not None:
            if lignes and lignes[-1].strip():
                lignes.append("")
            lignes += [f"[{fam}]", nouvelle]
    else:
        fin = next((i for i in range(debut + 1, len(lignes)) if lignes[i].lstrip().startswith("[")), len(lignes))
        motif = re.compile(rf"\s*{re.escape(nom)}\s*=")
        lieu = next((i for i in range(debut + 1, fin) if motif.match(lignes[i])), None)
        if lieu is not None:
            if nouvelle is None:
                del lignes[lieu]
            else:
                lignes[lieu] = nouvelle
        elif nouvelle is not None:
            dernier = max([i for i in range(debut, fin) if lignes[i].strip()] or [debut])
            lignes.insert(dernier + 1, nouvelle)
    texte = "\n".join(lignes) + "\n"
    try:
        lu = lire_toml_texte(texte)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{chemin} : the key cannot be set without breaking the file ({exc}) ; "
                         "edit the file manually") from None
    if (valeur is None and cle in lu) or (valeur is not None and lu.get(cle) != valeur):
        raise ValueError(f"{chemin} : the key {cle} is written elsewhere in the file (dot form?); "
                         "edit the file manually")
    chemin.write_text(texte)


def lire_toml_texte(texte: str) -> dict[str, Any]:
    import tempfile

    with tempfile.NamedTemporaryFile("w", suffix=".toml", delete=False) as f:
        f.write(texte)
    try:
        return lire_toml(f.name)
    finally:
        os.unlink(f.name)


# ---------------------------------------------------------------------------------------------------- .env -> toml
def lire_env_shell(fichiers: list[Path | str]) -> dict[str, str]:
    """Variables set by ``set -a; . A; . B`` (like the production launcher), read by bash itself."""

    def lancer(noms: list[str]) -> dict[str, str]:
        # one ". "$1" " line per file, no loop variable (set -a would export it)
        script = "set -a; " + "".join(f'. "${{{i}}}"; ' for i in range(1, len(noms) + 1)) + "env -0"
        sortie = subprocess.run(["/bin/bash", "-c", script, "bash", *noms],
                                env={"PATH": "/usr/bin:/bin"}, capture_output=True, check=True).stdout
        return dict(e.split("=", 1) for e in sortie.decode().split("\0") if "=" in e)

    base = lancer([])
    tout = lancer([str(Path(f).expanduser()) for f in fichiers])
    return {k: v for k, v in tout.items() if base.get(k) != v}


def env_vers_cles(env: dict[str, str]) -> tuple[dict[str, Any], list[tuple[str, str]]]:
    """Variables (including old names) -> ({key: value pli.toml}, [(ignored name, reason)])."""

    nouveau = anciens_noms()
    canon: dict[str, str] = {}
    ignores = []
    for k, v in env.items():                       # like env_names.apply: the new name wins
        if k in nouveau:
            if nouveau[k] not in env:
                canon[nouveau[k]] = v
        elif par_alias(k) is not None and par_alias(k)["source"] == "variable":
            canon.setdefault(par_alias(k)["variable"], v)
        elif par_derive(k) is not None:
            canon[k] = v
        else:
            canon[k] = v
    valeurs = {}
    for k, v in canon.items():
        r = par_variable(k) or par_derive(k)
        if r is None:
            ignores.append((k, "unknown to PLI" if k.startswith("PLI_") else "not a PLI setting (remains in "
                                                                              "l'environnement)"))
            continue
        valeurs[r["cle"]] = depuis_texte(r, v)
    return valeurs, ignores


def options_vers_cles(options: list[str]) -> dict[str, Any]:
    """Options for ``pli serve`` (without the model) -> {key: value} of options that differ from the default."""

    from pli import cli

    parser = cli.build_parser()
    args = parser.parse_args(["serve", "MODELE", *options])
    defauts = parser.parse_args(["serve", "MODELE"])
    valeurs = {}
    for r in reglages():
        if r["source"] != "option":
            continue
        v = getattr(args, r["dest"])
        if v != getattr(defauts, r["dest"]) and v is not None:
            valeurs[r["cle"]] = v
    return valeurs


# ---------------------------------------------------------------------------------------------------- pli serve
@dataclass
class Preparation:
    argv: list[str]
    defauts: dict[str, Any] = field(default_factory=dict)
    fichier: Path | None = None
    du_fichier: int = 0
    de_la_ligne: int = 0
    des_alias: int = 0
    anciens: list[tuple[str, str]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def annoncer(self, ecrire: Callable[[str], None] = lambda t: print(t, flush=True)) -> None:
        if self.anciens:
            def cle(n: str) -> str:
                r = par_variable(n)
                return f", {r['cle']}" if r else ""
            ecrire("[pli] old setting names, still accepted: " +
                   ", ".join(f"{o} (-> {n}{cle(n)})" for o, n in self.anciens) + " ; voir pli config:check")
        parts = []
        if self.du_fichier:
            parts.append(f"{self.du_fichier} de {self.fichier}")
        if self.de_la_ligne:
            parts.append(f"{self.de_la_ligne} from the command line")
        if self.des_alias:
            parts.append(f"{self.des_alias} by names derived from keys")
        if parts:
            ecrire("[pli] settings: " + ", ".join(parts))
        for n in self.notes:
            ecrire("[pli] " + n)


def _valeur_dans_argv(argv: list[str], option: str) -> str | None:
    for i, a in enumerate(argv):
        if a == option and i + 1 < len(argv):
            return argv[i + 1]
        if a.startswith(option + "="):
            return a.split("=", 1)[1]
    return None


def extraire(argv: list[str]) -> tuple[list[str], list[tuple[dict, Any]]]:
    """Removes ``--family.setting [value]`` from ``argv``; returns the rest and [(setting, value)]."""

    reste: list[str] = []
    trouves: list[tuple[dict, Any]] = []
    i = 0
    while i < len(argv):
        a = argv[i]
        nom, egal, val = a.partition("=")
        r = par_cle(nom[2:]) if a.startswith("--") and "." in nom else None
        if r is None:
            if a.startswith("--") and "." in nom and nom[2:].split(".")[0] in familles():
                raise ValueError(f"unknown setting: {nom}{proposer(nom[2:])}")
            reste.append(a)
            i += 1
            continue
        drapeau = r["type"] == "drapeau"
        if egal:
            trouves.append((r, val))
            i += 1
        elif drapeau and (i + 1 >= len(argv) or argv[i + 1].lower() not in VRAI | FAUX):
            trouves.append((r, True))
            i += 1
        elif i + 1 < len(argv):
            trouves.append((r, argv[i + 1]))
            i += 2
        else:
            raise ValueError(f"{nom}: a value is expected")
    return reste, trouves


def _drapeau(r: dict, v: Any) -> Any:
    """``--ane.ffn`` alone or ``--ane.ffn true`` -> "1" for a flag variable."""

    if r["type"] == "drapeau" and isinstance(v, str) and v.lower() in VRAI | FAUX:
        return v.lower() in VRAI
    return v


def _option_en_jetons(r: dict, v: Any) -> list[str]:
    """An option setting given as ``--family.setting`` -> tokens of the existing option."""

    v = valeur_option(r, v)
    option = r["options"][0]
    action = r.get("action", "")
    if action == "_StoreTrueAction":
        return [option] if v else []
    if action == "BooleanOptionalAction":
        return [option] if v else ["--no-" + option[2:]]
    if action == "_AppendAction":
        return [x for item in v for x in (option, item)]
    return [option, v]


def preparer(argv: list[str], environ: MutableMapping[str, str] | None = None) -> Preparation:
    """Before the ``pli serve`` parser: pli.toml, ``--family.setting``, names derived from keys, old names.

    Without a file (neither ``--config`` nor ``PLI_CONFIG``), without ``--famille.reglage`` and without a name taken from a key, ``argv``
    and the environment come out unchanged.
    """

    environ = os.environ if environ is None else environ
    from pli.additions import env_names

    prep = Preparation(argv=list(argv))
    prep.anciens = [(o, n) for o, n in env_names.RENAMED if o in environ]
    # the file; PLI_CONFIG leaves the environment (it would otherwise enter the settings fingerprint)
    de_env = environ.pop(VARIABLE_FICHIER, None)
    nomme = _valeur_dans_argv(argv, "--config") or de_env
    reste, de_ligne = extraire(argv[1:])
    options_donnees = {a.split("=", 1)[0] for a in reste if a.startswith("--")}
    # names derived from keys: set the code name, then leave the environment
    options_env: dict[str, Any] = {}
    for nom in [n for n in list(environ) if par_alias(n) is not None and par_variable(n) is None]:
        r = par_alias(nom)
        valeur = environ.pop(nom)
        prep.des_alias += 1
        if r["source"] == "option":
            options_env[r["dest"]] = valeur_option(r, valeur)
        elif r["variable"] not in environ:
            environ[r["variable"]] = valeur
        elif environ[r["variable"]] != valeur:
            prep.notes.append(f"{nom}={valeur!r} ignored: {r['variable']}={environ[r['variable']]!r} is set and wins")
    # pli.toml : sous l'environnement
    if nomme:
        prep.fichier = Path(nomme).expanduser()
        if not prep.fichier.is_file():
            raise FileNotFoundError(f"settings file not found: {prep.fichier}")
        try:
            valeurs = lire_toml(prep.fichier)
        except tomllib.TOMLDecodeError as exc:
            raise ValueError(f"{prep.fichier} : {exc}") from None
        inconnues = cles_inconnues(valeurs)
        if inconnues:
            raise ValueError(f"{prep.fichier}: unknown key: " +
                             ", ".join(f"{c}{proposer(c)}" for c in inconnues) + " (pli config:check)")
        for cle, v in valeurs.items():
            r = par_cle(cle)
            prep.du_fichier += 1
            if r["source"] == "derive":
                continue                            # lower down, after the environment
            if r["source"] == "option":
                if r.get("action") == "_AppendAction" and options_donnees & set(r["options"]):
                    continue                        # --alias from the line replaces the file list
                prep.defauts[r["dest"]] = valeur_option(r, v)
            else:
                texte = vers_texte(r, v)
                erreurs, _ = verifier(r, texte)
                if erreurs:
                    raise ValueError(f"{prep.fichier} : " + "; ".join(erreurs))
                environ.setdefault(r["variable"], texte)
    prep.defauts.update(options_env)                 # variable d'option > pli.toml
    # derived settings: line > variable > file; they set the code variables they name
    voulus: dict[str, tuple[Any, bool]] = {}          # key -> (value, comes from file)
    if nomme:
        voulus.update({c: (v, True) for c, v in valeurs.items() if par_cle(c)["source"] == "derive"})
    for r in derives():
        if r["variable"] in environ:
            voulus[r["cle"]] = (environ.pop(r["variable"]), False)
    for r, v in de_ligne:
        if r["source"] == "derive":
            voulus[r["cle"]] = (_drapeau(r, v), False)
    for cle, (v, du_fichier) in voulus.items():
        r = par_cle(cle)
        texte = vers_texte(r, _drapeau(r, v))
        if texte not in ("0", "1", ""):
            raise ValueError(f"{cle} : expected true or false, not {v!r}")
        if texte != "1":
            continue
        for var, val in r["pose"].items():
            if var in environ and environ[var] != val:
                if du_fichier:
                    continue                        # the deployed variable takes precedence over the file
                raise ValueError(f"{cle} requires {var}={val}, but {var}={environ[var]!r} is deployed: choose one "
                                 "or the other")
            environ[var] = val
    # --family.setting: above everything else
    ajouts: list[str] = []
    for r, v in de_ligne:
        prep.de_la_ligne += 1
        if r["source"] == "derive":
            continue
        if r["source"] == "option":
            ajouts += [str(x) for x in _option_en_jetons(r, v)]
        else:
            texte = vers_texte(r, _drapeau(r, v))
            erreurs, _ = verifier(r, texte)
            if erreurs:
                raise ValueError("; ".join(erreurs))
            environ[r["variable"]] = texte
    prep.argv = [argv[0], *reste, *ajouts]
    for dest, v in prep.defauts.items():
        r = next(x for x in reglages() if x.get("dest") == dest)
        if r["type"] == "choix" and str(v) not in r["choix"]:
            raise ValueError(f"{r['cle']} : expected one of {', '.join(r['choix'])}, not {v!r}")
    return prep


# ---------------------------------------------------------------------------------------------------- where it comes from
def origine(r: dict, environ: MutableMapping[str, str], fichier: dict[str, Any] | None, nom_fichier: str = "",
            profil: dict[str, str] | None = None, nom_profil: str = "") -> tuple[Any, str]:
    """(value, source) of a setting outside the command line: variable, pli.toml, profile, code default."""

    fichier = fichier or {}
    profil = profil or {}
    if r["source"] == "derive":
        if r["variable"] in environ:
            return environ[r["variable"]], f"variable {r['variable']}"
        if r["cle"] in fichier:
            return vers_texte(r, fichier[r["cle"]]), f"pli.toml {nom_fichier}".strip()
        return r["defaut"], "default"
    if r["source"] == "variable":
        var = r["variable"]
        vieux = [o for o, n in anciens_noms().items() if n == var and o in environ]
        if var in environ:
            if vieux and all(environ[o] == environ[var] for o in vieux):
                return environ[var], f"variable {vieux[0]} (old name of {var})"
            return environ[var], f"variable {var}"
        if r.get("alias") and r["alias"] in environ:
            return environ[r["alias"]], f"variable {r['alias']}"
        if r["cle"] in fichier:
            return vers_texte(r, fichier[r["cle"]]), f"pli.toml {nom_fichier}".strip()
        if var in profil:
            return profil[var], f"profil {nom_profil}".strip()
        return r["defaut"], "default (code)"
    if r.get("variable") and r["variable"] in environ:
        return environ[r["variable"]], f"variable {r['variable']}"
    if r["cle"] in fichier:
        return fichier[r["cle"]], f"pli.toml {nom_fichier}".strip()
    return r["defaut"], "default (option)"
