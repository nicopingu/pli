"""PLI 3.7.0 renamed its environment variables to English: the old names and old values keep working."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from tensorfold.pli import env_names, profiles

SRC = str(Path(__file__).resolve().parents[1] / "src")


def test_table_is_one_to_one():
    olds = [old for old, _ in env_names.RENAMED]
    news = [new for _, new in env_names.RENAMED]
    assert len(set(olds)) == len(olds) == len(set(news)) == len(news) == 174
    assert not set(olds) & set(news)


def test_old_name_sets_the_new_name():
    env = {"PLI_ARBRE_SEUL": "1", "TF_FLASH_MTP": "2", "TENSORFOLD_NO_LIVE": "1"}
    assert sorted(env_names.apply(env)) == ["PLI_FLASH_MTP", "PLI_NO_LIVE", "PLI_TREE_ALONE"]
    assert env["PLI_TREE_ALONE"] == "1" and env["PLI_FLASH_MTP"] == "2" and env["PLI_NO_LIVE"] == "1"
    assert env["PLI_ARBRE_SEUL"] == "1"                     # the old name stays, for older tools


def test_new_name_wins_over_old_name():
    env = {"PLI_ARBRE_TAU": "9", "PLI_TREE_TAU": "2"}
    assert env_names.apply(env) == []
    assert env["PLI_TREE_TAU"] == "2"


def test_prefixes_are_distinct_names():
    env = {"PLI_ANE_GARDE_MS": "5", "PLI_TRACE_DOSSIER": "/tmp/t"}
    env_names.apply(env)
    assert env["PLI_ANE_KEEPAWAKE_MS"] == "5" and "PLI_ANE_KEEPAWAKE" not in env
    assert env["PLI_TRACE_DIR"] == "/tmp/t" and "PLI_TRACE" not in env


def test_import_applies_old_names_first():
    env = {**os.environ, "PYTHONPATH": SRC, "PLI_ARBRE_SEUL": "1", "PLI_PROFIL": "m1-ultra"}
    env.pop("PLI_TREE_ALONE", None)
    env.pop("PLI_PROFILE", None)
    out = subprocess.run([sys.executable, "-c", "import os, tensorfold; print(os.environ['PLI_TREE_ALONE'], "
                          "os.environ['PLI_PROFILE'])"], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["1", "m1-ultra"]


def test_old_name_wins_over_the_profile():
    env = {"PLI_ANE_BLOCS": "1"}          # set by the user under the old name, before the profile is read
    env_names.apply(env)
    profiles.apply("m1-ultra", "qwen3_5", environ=env)
    assert env["PLI_ANE_BLOCKS"] == "1"


def test_profile_file_with_old_names(tmp_path, monkeypatch):
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "commun.env").write_text("PLI_ARBITRE=1\nPLI_ANE_BLOCKS=2\n")
    monkeypatch.setattr(profiles, "FOLDER", tmp_path)
    env = {}
    profiles.apply("old", None, environ=env)
    assert env == {"PLI_ARBITER": "1", "PLI_ANE_BLOCKS": "2"}


def test_old_values_of_prefill_cpu_mode():
    assert env_names.value("PLI_PREFILL_CPU_MODE", "decale") == "shifted"
    assert env_names.value("PLI_PREFILL_CPU_MODE", "couche") == "layer"
    assert env_names.value("PLI_PREFILL_CPU_MODE", "layer") == "layer"
    assert env_names.value("PLI_OTHER", "decale") == "decale"


@pytest.mark.parametrize("name", ["PLI_LECTURE_CPU_MODE", "PLI_PREFILL_CPU_MODE"])
@pytest.mark.parametrize("text,expected", [("decale", "shifted"), ("couche", "layer"), ("shifted", "shifted"),
                                           ("layer", "layer"), (None, "shifted")])
def test_prefill_cpu_mode_reads_old_names_and_values(monkeypatch, name, text, expected):
    pytest.importorskip("mlx.core")
    from tensorfold.families.qwen3_5 import prefill_cpu

    monkeypatch.delenv("PLI_LECTURE_CPU_MODE", raising=False)
    monkeypatch.delenv("PLI_PREFILL_CPU_MODE", raising=False)
    if text is not None:
        monkeypatch.setenv(name, text)
        env_names.apply()
    assert prefill_cpu.mode() == expected


def test_pop_removes_both_names():
    env = {"TENSORFOLD_API_KEY": "secret"}
    env_names.apply(env)
    assert env_names.pop("PLI_API_KEY", environ=env) == "secret" and env == {}
    assert env_names.pop("PLI_API_KEY", environ={}) == ""


def test_settings_fingerprint_counts_each_setting_once():
    old = {"PLI_ARBRE_SEUL": "1", "PLI_XF_BF16": "1", "TF_FLASH_MTP": "3"}
    env_names.apply(old)
    new = {"PLI_TREE_ALONE": "1", "PLI_XF_BF16": "1", "PLI_FLASH_MTP": "3"}
    assert env_names.pli_settings(old) == env_names.pli_settings(new) == [("PLI_TREE_ALONE", "1"), ("PLI_XF_BF16", "1")]
    assert env_names.pli_settings({"PLI_INSTANTANE_REGLAGES": "1"}, exclude=("PLI_SNAPSHOT_SETTINGS",)) == []


def test_service_profiles_accept_the_new_names():
    from tensorfold.control.config import validate_env

    env = {"PLI_HYBRID": "1", "PLI_MTP_HEAD": "/models/mtp.safetensors", "TF_FLASH_MTP": "2"}
    assert validate_env(env, secrets=False) == env


def test_settings_of_4_0_0_read_under_their_old_names():
    env = {"PLI_BROUILLON_PILE": "1", "PLI_FOURCHE_MIN": "0", "PLI_DEVERSE_INVITE": "0", "PLI_TETE_AVANT_COPIE": "3"}
    env_names.apply(env)
    assert (env["PLI_DRAFT_STACK"], env["PLI_FORK_MIN"], env["PLI_SPILL_PROMPT"], env["PLI_HEAD_BEFORE_COPY"]) == ("1", "0", "0", "3")


def test_extended_list_old_file_name(tmp_path):
    from tensorfold.pli import extended_list

    (tmp_path / "extended-list-2048.txt").write_text("5\n3\n5\n")
    assert extended_list._path(str(tmp_path / "liste-plus-2048.txt")) == str(tmp_path / "extended-list-2048.txt")
    assert extended_list._path(str(tmp_path / "other.txt")) == str(tmp_path / "other.txt")
    assert (Path(extended_list.__file__).parent / "extended-list-2048.txt").exists()
