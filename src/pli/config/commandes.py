"""Commandes ``pli config:list``, ``config:get``, ``config:set``, ``config:check``, ``config:explain``, ``config:import-env``.

They load no model and do not import MLX. Target file: ``--fichier``, otherwise ``PLI_CONFIG``, otherwise
``./pli.toml``.
"""

from __future__ import annotations

import argparse
import os
import shlex
import sys
import tomllib
from pathlib import Path
from typing import Any

from pli.config import reglages as R

NOMS = ("config:list", "config:get", "config:set", "config:check", "config:explain", "config:import-env")


def _fichier(args: argparse.Namespace) -> Path:
    return Path(args.fichier or os.environ.get(R.VARIABLE_FICHIER) or R.FICHIER_DEFAUT).expanduser()


def _lire(chemin: Path) -> dict[str, Any]:
    if not chemin.is_file():
        return {}
    try:
        return R.lire_toml(chemin)
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"{chemin} : {exc}") from None


def _profil(args: argparse.Namespace) -> tuple[dict[str, str], str]:
    nom = getattr(args, "profile", None) or os.environ.get("PLI_PROFILE", "")
    if not nom:
        return {}, ""
    from pli.additions import profiles

    valeurs: dict[str, str] = {}
    lus = profiles.apply(nom, getattr(args, "famille_modele", None), environ=valeurs)
    return valeurs, "+".join(lus)


def _contexte(args: argparse.Namespace) -> tuple[Path, dict[str, Any], dict[str, str], str]:
    chemin = _fichier(args)
    profil, nom_profil = _profil(args)
    return chemin, _lire(chemin), profil, nom_profil


def _montrer(v: Any) -> str:
    if v is None:
        return "(not deployed)"
    if v == "":
        return '""'
    if isinstance(v, list):
        return " ".join(str(x) for x in v) or "[]"
    return str(v)


def cmd_list(args: argparse.Namespace) -> int:
    chemin, fichier, profil, nom_profil = _contexte(args)
    if args.famille and args.famille not in R.familles():
        raise ValueError(f"famille inconnue : {args.famille} (familles : {', '.join(R.familles())})")
    if args.statut and args.statut not in R.statuts():
        raise ValueError(f"statut inconnu : {args.statut} (statuts : {', '.join(R.statuts())})")
    lignes = []
    for r in R.reglages():
        if args.famille and r["famille"] != args.famille and not r["famille"].startswith(args.famille + "."):
            continue
        if args.statut and r["statut"] != args.statut:
            continue
        valeur, source = R.origine(r, os.environ, fichier, str(chemin), profil, nom_profil)
        if args.poses and source.startswith("default"):
            continue
        lignes.append((r["cle"], r["statut"], _montrer(valeur), source))
    if not lignes:
        print("no setting deployed" if args.poses else "no setting")
        return 0
    large = max(len(c) for c, _, _, _ in lignes)
    largv = min(40, max(len(v) for _, _, v, _ in lignes))
    for c, s, v, src in lignes:
        print(f"{c:<{large}}  {s:<12}  {v:<{largv}}  {src}")
    return 0


def cmd_get(args: argparse.Namespace) -> int:
    r = R.trouver(args.cle)
    if r is None:
        print(f"unknown key: {args.cle}{R.proposer(args.cle)}", file=sys.stderr)
        return 1
    chemin, fichier, profil, nom_profil = _contexte(args)
    valeur, _ = R.origine(r, os.environ, fichier, str(chemin), profil, nom_profil)
    print("" if valeur is None else _montrer(valeur) if isinstance(valeur, list) else valeur)
    return 0


def cmd_set(args: argparse.Namespace) -> int:
    r = R.par_cle(args.cle)
    if r is None:
        print(f"unknown key: {args.cle}{R.proposer(args.cle)}", file=sys.stderr)
        return 1
    chemin = _fichier(args)
    if args.retirer:
        R.ecrire_cle(chemin, r["cle"], None)
        print(f"{r['cle']} removed from {chemin}")
        return 0
    if args.valeur is None:
        print("config:set: a value is expected (or --retirer)", file=sys.stderr)
        return 1
    brut = args.valeur
    valeur: Any
    if r["source"] == "option":
        valeur = R.valeur_option(r, brut)
        if r.get("action") not in ("_StoreTrueAction", "BooleanOptionalAction", "_AppendAction"):
            valeur = R.depuis_texte(r, valeur)
    else:
        if r["type"] == "drapeau" and brut.lower() in R.VRAI | R.FAUX:
            brut = "1" if brut.lower() in R.VRAI else "0"
        erreurs, avert = R.verifier(r, brut)
        if erreurs:
            print("\n".join(erreurs), file=sys.stderr)
            return 1
        for a in avert:
            print("attention : " + a, file=sys.stderr)
        valeur = R.depuis_texte(r, brut)
    R.ecrire_cle(chemin, r["cle"], valeur)
    print(f"{r['cle']} = {R.valeur_toml(valeur)} written to {chemin}")
    nom = r["variable"] if r["source"] == "variable" else ""
    if nom and nom in os.environ:
        print(f"warning: {nom} is set in the environment ({os.environ[nom]!r}) and takes precedence over {chemin}",
              file=sys.stderr)
    return 0


def verifier_tout(fichier: dict[str, Any], environ: dict[str, str] | os._Environ,
                  nom_fichier: str = "pli.toml") -> tuple[list[str], list[str]]:
    """(errors, warnings) from a read pli.toml and an environment."""

    erreurs: list[str] = []
    avert: list[str] = []
    for cle, v in fichier.items():
        r = R.par_cle(cle)
        if r is None:
            erreurs.append(f"{nom_fichier}: unknown key {cle}{R.proposer(cle)}")
            continue
        avert += _statut(r, f"{nom_fichier} : {cle}")
        try:
            if r["source"] == "option":
                R.valeur_option(r, v)
                continue
            e, a = R.verifier(r, R.vers_texte(r, v))
        except ValueError as exc:
            erreurs.append(f"{nom_fichier} : {exc}")
            continue
        erreurs += [f"{nom_fichier} : {x}" for x in e]
        avert += [f"{nom_fichier} : {x}" for x in a]
    anciens = R.anciens_noms()
    for nom in sorted(environ):
        if nom == R.VARIABLE_FICHIER:
            continue
        r = R.par_variable(nom) or R.par_alias(nom) or R.par_derive(nom)
        if nom in anciens:
            neuf = R.par_variable(anciens[nom])
            avert.append(f"variable {nom}: old name for {anciens[nom]}" + (f" (key {neuf['cle']})" if neuf else ""))
            continue
        if r is None:
            if nom.startswith("PLI_"):
                erreurs.append(f"variable {nom} : unknown to PLI{R.proposer(nom)}")
            continue
        avert += _statut(r, f"variable {nom} ({r['cle']})")
        if r["source"] == "option":
            try:
                R.valeur_option(r, environ[nom])
            except ValueError as exc:
                erreurs.append(f"variable {nom} : {exc}")
            continue
        e, a = R.verifier(r, environ[nom])
        erreurs += [f"variable {nom} : {x}" for x in e]
        avert += [f"variable {nom} : {x}" for x in a]
        if nom == "PLI_ATTENTION_PER_ROW" and environ[nom].strip() == "1":
            avert.append("variable PLI_ATTENTION_PER_ROW=1: one call per line from the first key; "
                         "attention.per_row_always says it as a flag")
    return erreurs, avert


def _statut(r: dict, ou: str) -> list[str]:
    if r["statut"] == "experimental":
        return [f"{ou}: experimental setting deployed (effect not guaranteed; {r['statut_raison']})"]
    if r["statut"] == "obsolete":
        return [f"{ou}: obsolete setting deployed ({r['statut_raison']})"]
    return []


def cmd_check(args: argparse.Namespace) -> int:
    chemin = _fichier(args)
    try:
        fichier = _lire(chemin)
    except ValueError as exc:
        print(f"error: {exc}")
        return 1
    environ = {} if args.sans_env else dict(os.environ)
    erreurs, avert = verifier_tout(fichier, environ, str(chemin))
    for e in erreurs:
        print("error: " + e)
    for a in avert:
        print("warning: " + a)
    vus = f"{len(fichier)} keys from {chemin}" if chemin.is_file() else f"no file {chemin}"
    print(f"{vus}, {sum(1 for n in environ if n.startswith('PLI_') or n.startswith('MLX_'))} variables: "
          f"{len(erreurs)} error(s), {len(avert)} warning(s)")
    return 1 if erreurs else 0


def cmd_explain(args: argparse.Namespace) -> int:
    r = R.trouver(args.cle)
    if r is None:
        print(f"unknown key: {args.cle}{R.proposer(args.cle)}", file=sys.stderr)
        return 1
    chemin, fichier, profil, nom_profil = _contexte(args)
    valeur, source = R.origine(r, os.environ, fichier, str(chemin), profil, nom_profil)
    fam, nom = R.famille_et_nom(r)
    print(r["cle"])
    if r.get("aide"):
        print(f"  aide       {r['aide']}")
    print(f"  statut     {r['statut']} ({r['statut_raison']})")
    if r.get("doute"):
        print(f"  to decide {r['doute']}")
    if r.get("note"):
        print(f"  note       {r['note']}")
    typ = r["type"] + (f" ({', '.join(r['choix'])})" if r.get("choix") else "")
    print(f"  type       {typ}")
    print(f"  default   {_montrer(r['defaut'])}")
    print(f"  read from {', '.join(r['lu'][:3])}" + (f" (+{len(r['lu']) - 3})" if len(r["lu"]) > 3 else ""))
    print(f"  pli.toml   [{fam}] {nom} = ...")
    if r["source"] == "derive":
        print(f"  option     --{r['cle']}")
        print(f"  variable   {r['variable']} (pose " + ", ".join(f"{k}={v}" for k, v in r["pose"].items()) + ")")
    elif r["source"] == "option":
        print(f"  option     {', '.join(r['options'])}, --{r['cle']}")
        if r.get("variable"):
            print(f"  variable   {r['variable']}")
    else:
        print(f"  option     --{r['cle']}")
        noms = [r["variable"]] + ([r["alias"]] if r.get("alias") else [])
        print(f"  variable   {' or '.join(noms)}" +
              (f" (anciens noms : {', '.join(r['anciens'])})" if r.get("anciens") else ""))
    for a in r.get("alertes") or []:
        print(f"  to check  {a}")
    print(f"  valeur     {_montrer(valeur)}  ({source})")
    print("  priority  option > variable > pli.toml > machine profile > default")
    return 0


def cmd_import_env(args: argparse.Namespace) -> int:
    env = R.lire_env_shell(args.fichiers) if args.fichiers else {}
    valeurs, ignores = R.env_vers_cles(env)
    if args.options:
        valeurs.update(R.options_vers_cles(shlex.split(args.options)))
    entete = ("pli.toml produced by pli config:import-env from: " + ", ".join(str(f) for f in args.fichiers)
              + (" and options" if args.options else ""))
    for nom, raison in ignores:
        entete += f"\nnon repris : {nom} ({raison})"
    texte = R.texte_toml(valeurs, entete)
    if args.sortie == "-":
        sys.stdout.write(texte)
    else:
        sortie = Path(args.sortie).expanduser()
        if sortie.exists() and not args.remplacer:
            print(f"{sortie} exists: --remplacer to overwrite it, or -o - to write it to the screen", file=sys.stderr)
            return 1
        sortie.write_text(texte)
        print(f"{len(valeurs)} settings written to {sortie}" + (f", {len(ignores)} not taken back" if ignores else ""))
    return 0


def register(commands: argparse._SubParsersAction) -> None:
    def commun(p: argparse.ArgumentParser, profil: bool = True) -> None:
        p.add_argument("--fichier", "-f", default=None, metavar="FICHIER",
                       help=f"pli.toml to read or write (default: ${R.VARIABLE_FICHIER}, else ./{R.FICHIER_DEFAUT})")
        if profil:
            p.add_argument("--profile", default=None, metavar="NAME", help="count a machine profile (e.g. m1-ultra)")
            p.add_argument("--famille-modele", default=None, metavar="MODEL_TYPE",
                           help="the model family of the profile (e.g. qwen3_5)")

    p = commands.add_parser("config:list", help="list the settings of a family (or all) with value and source")
    p.add_argument("famille", nargs="?", default=None, help="a family: " + ", ".join(R.familles()))
    p.add_argument("--poses", action="store_true", help="only the settings that are set")
    p.add_argument("--statut", default=None, help="only this status: recommande, experimental, obsolete")
    commun(p)
    p.set_defaults(func=cmd_list)
    p = commands.add_parser("config:get", help="print the value of a setting (key famille.reglage)")
    p.add_argument("cle")
    commun(p)
    p.set_defaults(func=cmd_get)
    p = commands.add_parser("config:set", help="write a setting into pli.toml (checked first)")
    p.add_argument("cle")
    p.add_argument("valeur", nargs="?", default=None)
    p.add_argument("--retirer", action="store_true", help="remove the setting from the file")
    commun(p, profil=False)
    p.set_defaults(func=cmd_set)
    p = commands.add_parser("config:check", help="check types and values of pli.toml and of the environment")
    p.add_argument("--sans-env", action="store_true", help="check the file only")
    commun(p, profil=False)
    p.set_defaults(func=cmd_check)
    p = commands.add_parser("config:explain", help="help, default, where it is read, and where its value comes from")
    p.add_argument("cle", help="a key, a variable (old names too) or an option")
    commun(p)
    p.set_defaults(func=cmd_explain)
    p = commands.add_parser("config:import-env", help="turn .env files (and serve options) into a pli.toml")
    p.add_argument("fichiers", nargs="*", help=".env files, read in order as `set -a; . FILE` (the last one wins)")
    p.add_argument("--options", default="", help="pli serve options to carry over, as one string (without the model)")
    p.add_argument("-o", "--sortie", default="-", help="file to write (default: -, the screen)")
    p.add_argument("--remplacer", action="store_true", help="overwrite the output file")
    p.set_defaults(func=cmd_import_env)
