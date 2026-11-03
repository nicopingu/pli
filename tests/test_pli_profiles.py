"""PLI machine profiles: files read, precedence of the environment, family, unknown profile, CLI arguments."""

import pytest

from tensorfold.pli import profiles


def test_m1_ultra_exists_and_carries_two_blocks_for_the_27b():
    assert "m1-ultra" in profiles.available()
    env = {}
    read = profiles.apply("m1-ultra", "qwen3_5", environ=env)
    assert read == ["m1-ultra/common.env", "m1-ultra/qwen3_5.env"]
    assert env["PLI_ANE_BLOCKS"] == "2" and env["PLI_ARBITER"] == "1" and env["PLI_HOST_GUARD"] == "1"
    assert "PLI_ALLOWED_HOSTS" not in env and "PLI_EVENTS" not in env      # machine-specific settings


def test_the_environment_wins_over_the_profile():
    env = {"PLI_ANE_BLOCKS": "1"}
    profiles.apply("m1-ultra", "qwen3_5", environ=env)
    assert env["PLI_ANE_BLOCKS"] == "1"


def test_without_family_only_common_is_read():
    env = {}
    assert profiles.apply("m1-ultra", None, environ=env) == ["m1-ultra/common.env"]
    assert "PLI_ANE_FFN" not in env


def test_unknown_profile_refused():
    with pytest.raises(ValueError, match="unknown profile"):
        profiles.apply("m9-max", "qwen3_5", environ={})


def test_old_common_file_name_still_read(tmp_path, monkeypatch):
    (tmp_path / "old").mkdir()
    (tmp_path / "old" / "commun.env").write_text("PLI_ARBITER=1\n")
    monkeypatch.setattr(profiles, "FOLDER", tmp_path)
    env = {}
    assert profiles.apply("old", None, environ=env) == ["old/commun.env"] and env == {"PLI_ARBITER": "1"}


def test_model_type_of(tmp_path):
    (tmp_path / "config.json").write_text('{"model_type": "qwen3_5"}')
    assert profiles.model_type_of(tmp_path) == "qwen3_5" and profiles.model_type_of(None) is None


def test_serve_profile_option():
    from tensorfold import cli

    assert cli.build_parser().parse_args(["serve", "x/y", "--profile", "m1-ultra"]).profile == "m1-ultra"
