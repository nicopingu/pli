"""Family settings (docs/reglages.md): table, pli.toml, --family.setting, priorities, config:* commands.

Key test: the production configuration (.env files read like the launcher, config.yaml options) and the pli.toml
derived from ``pli config:import-env`` yield the same PLI_* environment variable by variable and the same arguments.
Cleaned copies from tests/fixtures/reglages are used everywhere; ``REGLAGES_PROD=<servers directory>`` reruns the test
on the real machine files (read-only). No model is loaded.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import json
import os
import shlex
from pathlib import Path

import pytest

from pli import cli
from pli.config import commandes, reglages as R
from pli.additions import env_names

RACINE = Path(__file__).resolve().parent.parent
FIX = Path(__file__).resolve().parent / "fixtures" / "reglages"


def _script():
    chemin = RACINE / "tools" / "pli-reglages-inventaire"
    chargeur = importlib.machinery.SourceFileLoader("pli_reglages_inventaire", str(chemin))
    spec = importlib.util.spec_from_loader(chargeur.name, chargeur)
    module = importlib.util.module_from_spec(spec)
    chargeur.exec_module(module)
    return module


# ------------------------------------------------------------------------------------------------ the table
def test_table_suit_le_code():
    """table.json is the one produced by the script today (otherwise: rerun tools/pli-reglages-inventaire)."""

    attendu = _script().construire_table()
    assert json.loads(R.TABLE.read_text()) == json.loads(json.dumps(attendu))


def test_cles_et_noms_uniques():
    regs = R.reglages()
    cles = [r["cle"] for r in regs]
    assert len(cles) == len(set(cles))
    assert all(r["famille"] in R.familles() and r["cle"].startswith(r["famille"] + ".") for r in regs)
    noms = [r["variable"] for r in regs if r["source"] == "variable"]
    noms += [r["alias"] for r in regs if r.get("alias")]
    noms += [r["variable"] for r in regs if r["source"] == "option" and r["variable"]]
    assert len(noms) == len(set(noms))
    # no name derived from a key is the old name of another variable
    for r in regs:
        a = r.get("alias") if r["source"] == "variable" else r.get("variable")
        if a in env_names.NEW_NAME:
            assert R.par_variable(env_names.NEW_NAME[a]) is r


def test_chaque_variable_lue_et_chaque_ancien_nom_ont_une_cle():
    for vieux, neuf in env_names.RENAMED:
        assert R.par_variable(neuf) is not None, neuf


def test_le_nouveau_code_reste_hors_de_la_cle_des_points():
    from pli.additions import code_key

    hashes = {p.relative_to(code_key.ROOT).as_posix() for p in code_key.files()}
    assert not any(h.startswith("config/") for h in hashes)
    assert "cli.py" not in hashes and "cli_args.py" not in hashes


def test_valeurs_du_profil_m1_ultra_valides():
    from pli.additions import profiles

    for famille in (None, "qwen3_5", "qwen3_5_moe"):
        env: dict[str, str] = {}
        profiles.apply("m1-ultra", famille, environ=env)
        erreurs, avert = commandes.verifier_tout({}, env)
        assert erreurs == [] and avert == []


# ------------------------------------------------------------------------------------------------ equivalence
BASE = {"HOME": "/home/exemple", "PATH": "/usr/bin:/bin", "HF_HUB_OFFLINE": "1", "TENSORFOLD_NO_UPDATE_CHECK": "1",
        "PYTHONUNBUFFERED": "1"}


def _comme_le_lanceur(fichiers: list[Path]) -> dict[str, str]:
    """pli.sh: exported base, ``set -a; . pli.env; . pli-NOM.env``, then the pli import (old names)."""

    env = dict(BASE)
    env.update(R.lire_env_shell(fichiers))
    env_names.apply(env)
    return env


def _options(chemin: Path) -> tuple[str, list[str]]:
    jetons = shlex.split(chemin.read_text())
    return jetons[0], jetons[1:]


def _parse(argv: list[str], defauts: dict | None = None):
    parser = cli.build_parser()
    if defauts:
        cli.serve_parser(parser).set_defaults(**defauts)
    ns = vars(parser.parse_args(argv))
    ns.pop("func")
    return ns


def _prouver(fichiers: list[Path], options: list[str], modele: str, tmp_path: Path) -> dict:
    """Production against pli.toml: same environment and same arguments. Returns the report."""

    from pli.additions import code_key

    avant = _comme_le_lanceur(fichiers)
    argv_avant = ["serve", modele, *options, "--host", "127.0.0.1", "--port", "9080", "--no-update-check"]
    args_avant = _parse(argv_avant)

    # the pli.toml derived by config:import-env from the same files and same options
    toml = tmp_path / "pli.toml"
    ns = cli.build_parser().parse_args(["config:import-env", *map(str, fichiers), "--options", shlex.join(options),
                                        "-o", str(toml)])
    assert ns.func(ns) == 0
    apres = dict(BASE)
    apres["PLI_CONFIG"] = str(toml)
    env_names.apply(apres)
    prep = R.preparer(["serve", modele, "--host", "127.0.0.1", "--port", "9080", "--no-update-check"], apres)
    args_apres = _parse(prep.argv, prep.defauts)

    # 1. each variable read by the code: same value (or absent on both sides)
    lues = [r["variable"] for r in R.reglages() if r["source"] == "variable"]
    diff = {v: (avant.get(v), apres.get(v)) for v in lues if avant.get(v) != apres.get(v)}
    assert diff == {}
    # 2. the only extra variables before are the old names (not read by the code); nothing extra after
    assert set(avant) - set(apres) <= set(env_names.NEW_NAME)
    assert set(apres) - set(avant) == set()
    for k in set(avant) & set(apres):
        assert avant[k] == apres[k], k
    # 3. same PLI_* settings seen by the checkpoint key, same fingerprint
    assert env_names.pli_settings(avant) == env_names.pli_settings(apres)
    assert code_key.settings_fingerprint(avant) == code_key.settings_fingerprint(apres)
    # 4. same pli serve arguments
    assert args_avant == args_apres
    return {"variables": len([v for v in lues if v in avant]), "anciens": sorted(set(avant) - set(apres)),
            "options": sum(1 for k in args_avant if args_avant[k] != _parse(["serve", modele])[k])}


@pytest.mark.parametrize("service", ["27b", "35b"])
def test_production_nettoyee_et_pli_toml_donnent_le_meme_environnement(service, tmp_path):
    modele, options = _options(FIX / f"options-{service}.txt")
    compte = _prouver([FIX / "commun.env", FIX / f"{service}.env"], options, modele, tmp_path)
    assert compte["variables"] >= (60 if service == "27b" else 18)


@pytest.mark.skipif(not os.environ.get("REGLAGES_PROD"), reason="REGLAGES_PROD=<servers directory> for real production")
@pytest.mark.parametrize("service", ["27b", "35b"])
def test_vraie_production(service, tmp_path):
    racine = Path(os.environ["REGLAGES_PROD"]).expanduser()
    lignes = (racine / "protocole" / "config.yaml").read_text().splitlines()
    i = next(i for i, l in enumerate(lignes) if l.strip() == "${proto}/pli.sh " + service + " ${PORT}")
    retrait = lignes[i][: len(lignes[i]) - len(lignes[i].lstrip())]
    pris = []
    for l in lignes[i + 1:]:
        if not l.startswith(retrait) or not l.strip():
            break
        pris.append(l.strip())
    jetons = shlex.split(" ".join(pris))
    fichiers = [racine / "pli" / "pli.env", racine / "pli" / f"pli-{service}.env"]
    compte = _prouver([f for f in fichiers if f.is_file()], jetons[1:], jetons[0], tmp_path)
    print(f"\n[{service}] {compte['variables']} variables lues identiques, {compte['options']} options identiques, "
          f"old names only before: {len(compte['anciens'])}")


def test_sans_rien_l_environnement_et_les_arguments_ne_changent_pas():
    env = _comme_le_lanceur([FIX / "commun.env", FIX / "27b.env"])
    copie = dict(env)
    modele, options = _options(FIX / "options-27b.txt")
    argv = ["serve", modele, *options]
    prep = R.preparer(argv, env)
    assert prep.argv == argv and prep.defauts == {} and env == copie
    assert prep.du_fichier == prep.de_la_ligne == prep.des_alias == 0
    assert ("PLI_ARBITRE", "PLI_ARBITER") in prep.anciens


def test_avertissement_des_anciens_noms_une_seule_ligne():
    env = {"PLI_ARBITRE": "1", "PLI_ARBITER": "1", "TENSORFOLD_NO_UPDATE_CHECK": "1"}
    prep = R.preparer(["serve", "/m"], env)
    lignes: list[str] = []
    prep.annoncer(lignes.append)
    assert len(lignes) == 1
    assert "PLI_ARBITRE (-> PLI_ARBITER, parallele.arbiter)" in lignes[0]
    assert "TENSORFOLD_NO_UPDATE_CHECK" in lignes[0]


# ------------------------------------------------------------------------------------------------ priorities
def test_ordre_option_variable_fichier_profil_defaut(tmp_path):
    toml = tmp_path / "pli.toml"
    toml.write_text("[arbres]\ndeltanet = 2\n[ane]\nblocks = 1\n[cache]\ncontext = 1000\nspill_gib = 4\n")
    # pli.toml only
    env = {"PLI_CONFIG": str(toml)}
    prep = R.preparer(["serve", "/m"], env)
    assert env == {"PLI_TREE_DELTANET": "2", "PLI_ANE_BLOCKS": "1"}           # PLI_CONFIG removed
    args = _parse(prep.argv, prep.defauts)
    assert args["context"] == 1000 and args["spill_gib"] == 4.0
    # the variable wins over the file (code name, old name, name derived from key, option variable)
    env = {"PLI_CONFIG": str(toml), "PLI_TREE_DELTANET": "3", "PLI_ARBRE_DN": "3", "PLI_ANE_BLOCKS": "2",
           "PLI_CACHE_CONTEXT": "2000"}
    prep = R.preparer(["serve", "/m"], env)
    assert env["PLI_TREE_DELTANET"] == "3" and env["PLI_ANE_BLOCKS"] == "2" and "PLI_CACHE_CONTEXT" not in env
    assert _parse(prep.argv, prep.defauts)["context"] == 2000
    # the option wins over everything
    env = {"PLI_CONFIG": str(toml), "PLI_TREE_DELTANET": "3", "PLI_CACHE_CONTEXT": "2000"}
    prep = R.preparer(["serve", "/m", "--arbres.deltanet", "1", "--context", "3000"], env)
    assert env["PLI_TREE_DELTANET"] == "1"
    assert _parse(prep.argv, prep.defauts)["context"] == 3000
    # the profile comes after the file (it only does setdefault)
    from pli.additions import profiles

    env = {"PLI_CONFIG": str(toml)}
    R.preparer(["serve", "/m"], env)
    profiles.apply("m1-ultra", "qwen3_5", environ=env)
    assert env["PLI_ANE_BLOCKS"] == "1" and env["PLI_ANE_FFN"] == "1"


def test_nom_tire_de_la_cle_pose_le_nom_du_code_puis_s_en_va():
    env = {"PLI_ARBRES_DELTANET": "3", "PLI_BROUILLONS_MTP_HEAD_DRAFTS": "5"}
    prep = R.preparer(["serve", "/m"], env)
    assert env == {"PLI_TREE_DELTANET": "3", "PLI_MTP_DRAFTS": "5"} and prep.des_alias == 2


def test_options_a_points():
    env: dict[str, str] = {}
    prep = R.preparer(["serve", "/m", "--ane.ffn", "--attention.reduced=10", "--cache.context", "4096",
                       "--modele.thinking", "false", "--brouillons.no_drafts", "--modele.alias", "a,b"], env)
    assert env == {"PLI_ANE_FFN": "1", "PLI_ATTENTION_REDUCED": "10"}
    args = _parse(prep.argv, prep.defauts)
    assert args["context"] == 4096 and args["thinking"] is False and args["no_drafts"] is True
    assert args["alias"] == ["a", "b"]


def test_alias_de_la_ligne_remplace_la_liste_du_fichier(tmp_path):
    toml = tmp_path / "pli.toml"
    toml.write_text('[modele]\nalias = ["x", "y"]\n')
    prep = R.preparer(["serve", "/m", "--config", str(toml), "--alias", "z"], {})
    assert _parse(prep.argv, prep.defauts)["alias"] == ["z"]
    prep = R.preparer(["serve", "/m", "--config", str(toml)], {})
    assert _parse(prep.argv, prep.defauts)["alias"] == ["x", "y"]


def test_refus_clair(tmp_path):
    toml = tmp_path / "pli.toml"
    toml.write_text("[ane]\nfffn = 1\n")
    with pytest.raises(ValueError, match="ane.ffn"):
        R.preparer(["serve", "/m", "--config", str(toml)], {})
    with pytest.raises(ValueError, match="unknown setting"):
        R.preparer(["serve", "/m", "--ane.fnn", "1"], {})
    toml.write_text("[cache]\ncheckpoint_bits = \"quatre\"\n")
    with pytest.raises(ValueError, match="integer"):
        R.preparer(["serve", "/m", "--config", str(toml)], {})
    toml.write_text("[modele]\nreasoning_effort = \"max\"\n")
    with pytest.raises(ValueError, match="among"):
        R.preparer(["serve", "/m", "--config", str(toml)], {})
    with pytest.raises(FileNotFoundError):
        R.preparer(["serve", "/m", "--config", str(tmp_path / "absent.toml")], {})


def test_main_refuse_sans_charger_de_modele(tmp_path, capsys):
    toml = tmp_path / "pli.toml"
    toml.write_text("[ane]\nfffn = 1\n")
    assert cli.main(["serve", "/m", "--config", str(toml)]) == 1
    assert "unknown key" in capsys.readouterr().err


# ------------------------------------------------------------------------------------------------ valeurs
@pytest.mark.parametrize("texte", ["1", "0", "15000", "4.5", "0.25", "256:14,512:16,1536:18", "f16", "", "015", "1e3",
                                   "120.0"])
def test_aller_retour_des_valeurs(texte):
    for cle in ("ane.ffn", "attention.per_row", "arbres.token_threshold", "ane.steps"):
        r = R.par_cle(cle)
        assert R.vers_texte(r, R.depuis_texte(r, texte)) == texte


# ------------------------------------------------------------------------------------------------ commandes
def _lancer(capsys, *argv) -> tuple[int, str, str]:
    code = cli.main(list(argv))
    sortie = capsys.readouterr()
    return code, sortie.out, sortie.err


def test_set_get_explain_check(tmp_path, capsys, monkeypatch):
    for nom in [n for n in os.environ if n.startswith(("PLI_", "MLX_"))]:
        monkeypatch.delenv(nom)
    f = tmp_path / "pli.toml"
    f.write_text("# my setting\n[ane]\nffn = true  # kept\n")
    assert _lancer(capsys, "config:set", "arbres.deltanet", "3", "-f", str(f))[0] == 0
    assert _lancer(capsys, "config:set", "ane.blocks", "2", "-f", str(f))[0] == 0
    assert _lancer(capsys, "config:set", "cache.context", "229376", "-f", str(f))[0] == 0
    assert _lancer(capsys, "config:set", "modele.thinking", "false", "-f", str(f))[0] == 0
    texte = f.read_text()
    assert "# my setting" in texte and "[arbres]\ndeltanet = 3" in texte and "blocks = 2" in texte
    assert R.lire_toml(f) == {"ane.ffn": True, "ane.blocks": 2, "arbres.deltanet": 3, "cache.context": 229376,
                              "modele.thinking": False}
    code, out, _ = _lancer(capsys, "config:get", "arbres.deltanet", "-f", str(f))
    assert code == 0 and out == "3\n"
    code, out, _ = _lancer(capsys, "config:get", "PLI_ARBRE_DN", "-f", str(f))     # an old name finds its key
    assert out == "3\n"
    monkeypatch.setenv("PLI_TREE_DELTANET", "1")
    code, out, _ = _lancer(capsys, "config:explain", "arbres.deltanet", "-f", str(f))
    assert "valeur     1  (variable PLI_TREE_DELTANET)" in out and "PLI_ARBRE_DN" in out
    monkeypatch.delenv("PLI_TREE_DELTANET")
    code, out, _ = _lancer(capsys, "config:list", "arbres", "--poses", "-f", str(f))
    assert out.split() == ["arbres.deltanet", "recommande", "3", "pli.toml", str(f)]
    assert _lancer(capsys, "config:set", "arbres.deltanet", "--retirer", "-f", str(f))[0] == 0
    assert "arbres.deltanet" not in R.lire_toml(f)
    assert _lancer(capsys, "config:set", "cache.checkpoint_bits", "quatre", "-f", str(f))[0] == 1
    assert _lancer(capsys, "config:set", "ane.fffn", "1", "-f", str(f))[0] == 1
    code, out, _ = _lancer(capsys, "config:check", "-f", str(f))
    assert code == 0 and "0 error(s)" in out
    monkeypatch.setenv("PLI_POINTS_BITS", "4")
    monkeypatch.setenv("PLI_TREE_DELTANNET", "1")
    code, out, _ = _lancer(capsys, "config:check", "-f", str(f))
    assert code == 1 and "PLI_TREE_DELTANNET : unknown to PLI" in out and "PLI_TREE_DELTANET" in out
    assert "old name for PLI_CHECKPOINT_BITS (key cache.checkpoint_bits)" in out


def test_import_env_ecrit_un_toml_lisible(tmp_path, capsys):
    code, out, _ = _lancer(capsys, "config:import-env", str(FIX / "commun.env"), str(FIX / "27b.env"),
                           "--options", "--context 229376 --parallel 4 --no-thinking")
    assert code == 0
    assert "[ane]\n" in out and "ffn = true" in out and "[cache]\n" in out and "context = 229376" in out
    assert "thinking = false" in out and 'parallel = "4"' in out
    assert "non repris : MLX" not in out


def test_mlx_n_est_pas_importe_avant_la_preparation(tmp_path):
    """H1 (docs/reglages.md 2.6): pli.toml acts as the shell, including MLX_MAX_*, because nothing imports MLX before
    cli.main; the preparation itself does not import it either."""

    import subprocess
    import sys

    toml = tmp_path / "pli.toml"
    toml.write_text("[memoire]\nmlx_max_mb_per_buffer = 500\n")
    code = ("import sys, os\nfrom pli import cli\nfrom pli.config import reglages\n"
            f"reglages.preparer(['serve', '/m', '--config', {str(toml)!r}])\n"
            "print('mlx' in sys.modules, os.environ.get('MLX_MAX_MB_PER_BUFFER'))")
    sortie = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True,
                            env={**os.environ, "PYTHONPATH": str(RACINE / "src")}).stdout.split()
    assert sortie == ["False", "500"]


# ------------------------------------------------------------------------------------------------ statuts
def test_statuts():
    assert {r["statut"] for r in R.reglages()} == set(R.statuts())
    variables, options = _script().production_reference()
    for r in R.reglages():
        en_prod = (r["source"] == "variable" and r["variable"] in variables) or \
                  (r["source"] == "option" and options & set(r["options"]))
        if en_prod and r["cle"] not in _script().STATUTS_FORCES:
            assert r["statut"] == "recommande", r["cle"]
    assert R.par_cle("modele.backend")["statut"] == "obsolete"
    assert R.par_cle("arbres.noise")["statut"] == "experimental"


def test_list_par_statut_et_check_previent(tmp_path, capsys, monkeypatch):
    for nom in [n for n in os.environ if n.startswith(("PLI_", "MLX_"))]:
        monkeypatch.delenv(nom)
    code, out, _ = _lancer(capsys, "config:list", "arbres", "--statut", "recommande", "-f", str(tmp_path / "x.toml"))
    assert code == 0 and all(l.split()[1] == "recommande" for l in out.splitlines())
    monkeypatch.setenv("PLI_TREE_NOISE", "0.1")
    code, out, _ = _lancer(capsys, "config:check", "-f", str(tmp_path / "x.toml"))
    assert code == 0 and "experimental setting deployed" in out
    code, out, _ = _lancer(capsys, "config:explain", "arbres.noise", "-f", str(tmp_path / "x.toml"))
    assert "statut     experimental" in out


# ------------------------------------------------------------------------------------------------ PER_ROW en deux
def test_per_row_always_vaut_l_ancien_1():
    env = {"PLI_ATTENTION_PER_ROW_ALWAYS": "1"}
    R.preparer(["serve", "/m"], env)
    assert env == {"PLI_ATTENTION_PER_ROW": "1"}               # exactly what the old "1" deployed
    env = {}
    R.preparer(["serve", "/m", "--attention.per_row_always"], env)
    assert env == {"PLI_ATTENTION_PER_ROW": "1"}
    env = {"PLI_ATTENTION_PER_ROW_ALWAYS": "0", "PLI_ATTENTION_PER_ROW": "15000"}
    R.preparer(["serve", "/m"], env)
    assert env == {"PLI_ATTENTION_PER_ROW": "15000"}            # the production threshold, intact


def test_per_row_always_contre_un_seuil(tmp_path):
    with pytest.raises(ValueError, match="choose"):
        R.preparer(["serve", "/m"], {"PLI_ATTENTION_PER_ROW_ALWAYS": "1", "PLI_ATTENTION_PER_ROW": "15000"})
    toml = tmp_path / "pli.toml"
    toml.write_text("[attention]\nper_row_always = true\n")
    env = {"PLI_ATTENTION_PER_ROW": "15000"}
    R.preparer(["serve", "/m", "--config", str(toml)], env)     # the variable wins over the file
    assert env == {"PLI_ATTENTION_PER_ROW": "15000"}
    env = {}
    R.preparer(["serve", "/m", "--config", str(toml)], env)
    assert env == {"PLI_ATTENTION_PER_ROW": "1"}


def test_per_row_dans_le_noyau_meme_lecture():
    """The kernel reads the same text: what the old "1" and the new flag deploy yield the same constants."""

    import ast

    source = (RACINE / "src/pli/kernels/qwen/dense/v1/exact_attention.py").read_text()
    lignes = [l for l in source.splitlines() if "PLI_ATTENTION_PER_ROW" in l and "environ" in l]
    assert len(lignes) == 2                                      # kernel unchanged: two reads of the same name
    for valeur in ("1", "15000", "0"):
        espace = {"os": type("os", (), {"environ": {"PLI_ATTENTION_PER_ROW": valeur}})}
        code = "\n".join(l for l in source.splitlines()[:40] if l.startswith(("GROUP_QUERIES", "try:", "    PER_ROW",
                                                                                "except", "if PER_ROW", "    GROUP")))
        exec(compile(ast.parse(code), "exact_attention", "exec"), espace)
        if valeur == "1":
            assert espace["GROUP_QUERIES"] is False
        if valeur == "15000":
            assert espace["GROUP_QUERIES"] is True and espace["PER_ROW_THRESHOLD"] == 15000
