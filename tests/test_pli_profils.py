"""Profils machine de PLI : fichiers lus, priorité de l'environnement, famille, profil inconnu, arguments de la CLI."""

import pytest

from tensorfold.pli import profils


def test_m1_ultra_existe_et_porte_deux_blocs_pour_le_27b():
    assert "m1-ultra" in profils.disponibles()
    env = {}
    lus = profils.appliquer("m1-ultra", "qwen3_5", environ=env)
    assert lus == ["m1-ultra/commun.env", "m1-ultra/qwen3_5.env"]
    assert env["PLI_ANE_BLOCS"] == "2" and env["PLI_ARBITRE"] == "1" and env["PLI_GARDE_HOTE"] == "1"
    assert "PLI_HOTES_ADMIS" not in env and "PLI_EVENEMENTS" not in env      # réglages propres à la machine


def test_l_environnement_l_emporte_sur_le_profil():
    env = {"PLI_ANE_BLOCS": "1"}
    profils.appliquer("m1-ultra", "qwen3_5", environ=env)
    assert env["PLI_ANE_BLOCS"] == "1"


def test_sans_famille_seul_le_commun_est_lu():
    env = {}
    assert profils.appliquer("m1-ultra", None, environ=env) == ["m1-ultra/commun.env"]
    assert "PLI_ANE_FFN" not in env


def test_profil_inconnu_refuse():
    with pytest.raises(ValueError, match="profil inconnu"):
        profils.appliquer("m9-max", "qwen3_5", environ={})


def test_type_de_modele(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "qwen3_5"}')
    assert profils.type_de_modele(tmp_path) == "qwen3_5" and profils.type_de_modele(None) is None


def test_option_profile_de_serve():
    from tensorfold import cli

    assert cli.build_parser().parse_args(["serve", "x/y", "--profile", "m1-ultra"]).profile == "m1-ultra"
