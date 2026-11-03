"""pli setup, the service verbs, pli uninstall, pli bench and install.sh, the Hub, the Mac and pli service simulated.

Nothing is downloaded, no service is installed: the Hub's API, the Mac's checks, `pli pull`, `pli service` and the
MTP head writer are replaced by fakes that record what they are asked.
"""

from __future__ import annotations

import builtins
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from tensorfold import cli
from tensorfold.install import commands, downloads, machine

ROOT = Path(__file__).resolve().parents[1]
MODEL, DRAFTER, MTP = downloads.MODEL, downloads.DRAFTER, downloads.MTP_SOURCE
HUB = {
    MODEL: {"commit": "10c35caafbb80f7dc6a7a432cdd11af10a6d4818", "license": "apache-2.0",
            "files": [("config.json", 2_000, "c0"), ("model-00001-of-00002.safetensors", 8_000_000_000, "s1"),
                      ("model-00002-of-00002.safetensors", 8_081_488_933, "s2")]},
    DRAFTER: {"commit": "50307d4c4cde6860d4eee73e2547cd786fe8e8a4", "license": "apache-2.0",
              "files": [("config.json", 1_000, "d0"), ("model.safetensors", 3_849_112_002, "d1")]},
    MTP: {"commit": "e678963cb51fb7d9f99ff9bf52c3e0694b7e009d", "license": "",
          "files": [("model.safetensors", 849_407_885, "m1")]},
}
ULTRA = machine.Mac("Darwin", "arm64", "27.0.1", 64, "Apple M1 Ultra", 500, True)


class Recorder:
    def __init__(self):
        self.pli: list[tuple[tuple[str, ...], bool]] = []
        self.run: list[tuple[str, ...]] = []
        self.snapshots: list[str] = []

    def service_install(self) -> list[str] | None:
        found = [args for args, _ in self.pli if args[:2] == ("service", "install")]
        return list(found[-1]) if found else None

    def pulls(self) -> list[tuple[str, ...]]:
        return [args for args, _ in self.pli if args[0] == "pull"]


@pytest.fixture
def fake(tmp_path, monkeypatch):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("HOME", str(home))
    hub = tmp_path / "hub"
    monkeypatch.setattr(downloads, "hub_cache", lambda: hub)
    monkeypatch.setattr(downloads, "hub_info", lambda repo: HUB[repo])
    monkeypatch.setattr(downloads, "free_bytes", lambda path: 500 * 10**9)
    monkeypatch.setattr(downloads, "cached_complete", lambda repo: False)
    monkeypatch.setattr(machine, "probe", lambda: ULTRA)
    monkeypatch.setattr(machine, "check_alone", lambda: None)
    rec = Recorder()

    def pli(*args, quiet=False):
        rec.pli.append((args, quiet))
        return 0

    def run(*args):
        rec.run.append(args)
        return 0

    def snapshot(repo):
        rec.snapshots.append(repo)
        return tmp_path / "snapshots" / repo.replace("/", "--")

    monkeypatch.setattr(commands, "_pli", pli)
    monkeypatch.setattr(commands, "_run", run)
    monkeypatch.setattr(commands, "_snapshot", snapshot)
    monkeypatch.setattr(commands, "_health", lambda port: True)
    monkeypatch.setattr(commands, "_sleep", lambda s: None)
    monkeypatch.setattr(commands, "_python", lambda: "/venv/bin/python")
    rec.home, rec.hub, rec.tmp = home, hub, tmp_path
    return rec


def terminal(monkeypatch, answer: str | None):
    """A terminal on stdin that answers ``answer`` (None: no terminal)."""

    class Stdin:
        def isatty(self):
            return answer is not None

    monkeypatch.setattr(sys, "stdin", Stdin())
    asked = []

    def ask(prompt=""):
        asked.append(prompt)
        if answer is None:
            raise EOFError
        return answer

    monkeypatch.setattr(builtins, "input", ask)
    return asked


def expected_service(prefix: Path, *, start: bool = True, hybrid: bool = True) -> list[str]:
    """install.sh's `pli service install` on an M1 Ultra with 64 GB and 500 GB free, option for option."""

    args = ["service", "install", MODEL, "--name", "pli", "--python", "/venv/bin/python", "--port", "8080",
            "--drafter", "auto", "--env", f"PLI_EVENTS={prefix}/logs/events.jsonl", "--replace",
            "--env", "PLI_ANE_FFN=1", "--env", "MLX_MAX_MB_PER_BUFFER=500", "--env", "MLX_MAX_OPS_PER_BUFFER=100",
            "--arg=--profile=m1-ultra", "--context", "229376", "--parallel", "4", "--arg=--spill-gib=40"]
    if hybrid:
        args += ["--env", "PLI_HYBRID=1", "--env", f"PLI_MTP_HEAD={prefix}/models/mtp-head-4bit.safetensors"]
    if start:
        args.append("--start")
    return args


# ---------------------------------------------------------------- pli alone

def test_pli_alone_prints_a_short_help_with_examples(capsys):
    assert cli.main([]) == 0
    out = capsys.readouterr().out
    for example in ("pli setup", "pli serve MODEL", "pli status", "pli bench"):
        assert example in out
    assert "usage:" not in out and len(out.splitlines()) < 15


def test_the_new_commands_are_in_pli_help(capsys):
    with pytest.raises(SystemExit):
        cli.main(["--help"])
    out = capsys.readouterr().out
    for name in ("setup", "status", "start", "stop", "restart", "logs", "uninstall", "bench", "serve", "service"):
        assert name in out


# ---------------------------------------------------------------- service command

def test_service_command_is_install_sh_s_on_an_ultra(tmp_path):
    prefix = tmp_path / "pli"
    head = prefix / "models" / "mtp-head-4bit.safetensors"
    args = commands.service_args(model=MODEL, python="/venv/bin/python", prefix=prefix, port=8080,
                                 profile="m1-ultra", ane=1, context=None, free_gb=500, hybrid=True, mtp_head=head,
                                 start=True, drafter="auto")
    assert args == expected_service(prefix)


def test_service_command_without_profile_tools_or_disk(tmp_path):
    args = commands.service_args(model=MODEL, python="/p", prefix=tmp_path, port=9000, profile="", ane=0,
                                 context=32768, free_gb=40, hybrid=False, mtp_head=None, start=False, drafter="auto")
    assert args == ["service", "install", MODEL, "--name", "pli", "--python", "/p", "--port", "9000",
                    "--drafter", "auto", "--env", f"PLI_EVENTS={tmp_path}/logs/events.jsonl", "--replace",
                    "--context", "32768"]


def test_ultra_without_command_line_tools_turns_the_neural_engine_off(capsys):
    profile, ane = machine.check(machine.Mac("Darwin", "arm64", "15.1", 128, "Apple M3 Ultra", 900, False))
    assert (profile, ane) == ("m1-ultra", 0)
    assert "Command Line Tools missing" in capsys.readouterr().err


def test_a_small_mac_gets_warnings_and_no_profile(capsys):
    profile, ane = machine.check(machine.Mac("Darwin", "arm64", "14.6", 16, "Apple M1 Pro", 30, True))
    assert (profile, ane) == ("", 1)
    err = capsys.readouterr().err
    assert "needs about 48 GB" in err and "70 GB of free disk" in err and "m1-ultra profile" in err


@pytest.mark.parametrize("mac, message", [
    (machine.Mac("Linux", "x86_64", "", 0, "", 0, False), "PLI runs on macOS only"),
    (machine.Mac("Darwin", "x86_64", "14.0", 64, "Intel", 100, True), "Apple Silicon"),
    (machine.Mac("Darwin", "arm64", "13.6", 64, "Apple M1 Ultra", 100, True), "macOS 14 or later"),
])
def test_setup_refuses_a_system_pli_does_not_run_on(fake, monkeypatch, capsys, mac, message):
    monkeypatch.setattr(machine, "probe", lambda: mac)
    assert cli.main(["setup", "--yes"]) == 1
    assert message in capsys.readouterr().err
    assert fake.pli == [] and fake.pulls() == []


# ---------------------------------------------------------------- the download screen and the consent

def test_setup_yes_lists_every_download_then_installs_as_install_sh(fake, capsys):
    assert cli.main(["setup", "--yes"]) == 0
    out = capsys.readouterr().out
    prefix = fake.home / "pli"
    for repo in (MODEL, DRAFTER, MTP):
        assert f"https://huggingface.co/{repo}, revision main" in out
        assert str(fake.hub / f"models--{repo.replace('/', '--')}") in out
    assert "commit 10c35ca" in out and "license: apache-2.0" in out
    assert "license: see model card" in out                          # the MTP repo's card names no license
    assert "16.1 GB to download" in out and "3.8 GB to download" in out and "849 MB to download" in out
    assert "To download: 20.8 GB. Free disk: 500.0 GB" in out
    assert "written on this Mac from the MTP head above; nothing more to download" in out
    assert f"about 239 MB, into {prefix}/models/mtp-head-4bit.safetensors" in out
    assert "--no-hybrid" in out and "--no-drafter" in out and "--no-models" in out
    assert "Download accepted (--yes)." in out
    assert fake.pulls() == [("pull", MODEL, DRAFTER)]
    assert fake.snapshots == [MTP, MODEL]
    snap = fake.tmp / "snapshots"
    assert fake.run == [("-m", "tensorfold.install.mtp_head", "--model", str(snap / MODEL.replace("/", "--")),
                         str(snap / MTP.replace("/", "--") / "model.safetensors"),
                         str(prefix / "models" / "mtp-head-4bit.safetensors"))]
    assert (("service", "stop", "pli"), True) in fake.pli
    assert fake.service_install() == expected_service(prefix)
    assert "PLI is ready on port 8080." in out
    conf = (fake.home / ".config" / "pli" / "install.conf").read_text()
    assert conf == f'PREFIX="{prefix}"\nPORT="8080"\nMODEL="{MODEL}"\n'
    assert (prefix / "logs").is_dir()


def test_setup_asks_and_downloads_on_yes(fake, monkeypatch, capsys):
    asked = terminal(monkeypatch, "y")
    assert cli.main(["setup"]) == 0
    assert asked == ["Download them? [y/N] "]
    assert fake.pulls() == [("pull", MODEL, DRAFTER)]
    assert fake.service_install() == expected_service(fake.home / "pli")


@pytest.mark.parametrize("answer", ["", "n", "no", "maybe"])
def test_setup_downloads_nothing_unless_the_answer_is_yes(fake, monkeypatch, capsys, answer):
    asked = terminal(monkeypatch, answer)
    assert cli.main(["setup"]) == 0
    assert len(asked) == 1
    assert "Nothing downloaded." in capsys.readouterr().out
    assert fake.pulls() == [] and fake.run == [] and fake.snapshots == [] and fake.service_install() is None
    assert not (fake.home / ".config" / "pli" / "install.conf").exists()


def test_setup_without_a_terminal_stops_and_says_what_to_add(fake, monkeypatch, capsys):
    asked = terminal(monkeypatch, None)
    assert cli.main(["setup"]) == 1
    captured = capsys.readouterr()
    assert "Before anything is downloaded" in captured.out                # the list is shown all the same
    assert "add --yes" in captured.err and "--no-models" in captured.err
    assert asked == [] and fake.pulls() == [] and fake.service_install() is None


def test_files_already_downloaded_are_marked_and_not_counted(fake, capsys):
    blobs = downloads.repo_dir(DRAFTER) / "blobs"
    blobs.mkdir(parents=True)
    for blob in ("d0", "d1"):
        (blobs / blob).write_bytes(b"")
    (downloads.repo_dir(MODEL) / "blobs").mkdir(parents=True)
    (downloads.repo_dir(MODEL) / "blobs" / "s1").write_bytes(b"")
    assert cli.main(["setup", "--yes"]) == 0
    out = capsys.readouterr().out
    assert f"already downloaded (3.8 GB in {downloads.repo_dir(DRAFTER)})" in out
    assert "8.1 GB to download (of 16.1 GB; the rest is already downloaded)" in out
    assert "To download: 8.9 GB." in out                                  # 8.08 + 0.85: the draft is not counted


def test_nothing_to_download_asks_nothing(fake, monkeypatch, capsys):
    asked = terminal(monkeypatch, None)
    for repo, info in HUB.items():
        blobs = downloads.repo_dir(repo) / "blobs"
        blobs.mkdir(parents=True)
        for _, _, blob in info["files"]:
            (blobs / blob).write_bytes(b"")
    assert cli.main(["setup"]) == 0
    assert "Nothing to download: every file is already on this Mac." in capsys.readouterr().out
    assert asked == [] and fake.service_install() == expected_service(fake.home / "pli")


def test_sizes_are_approximate_when_the_hub_does_not_answer(fake, monkeypatch, capsys):
    def offline(repo):
        raise OSError("no network")

    monkeypatch.setattr(downloads, "hub_info", offline)
    assert cli.main(["setup", "--yes"]) == 0
    captured = capsys.readouterr()
    assert "about 16.1 GB to download" in captured.out and "about 3.8 GB to download" in captured.out
    assert "To download: about 20.8 GB." in captured.out
    assert "license: see model card" in captured.out and "commit" not in captured.out
    assert "sizes are approximate" in captured.err


def test_not_enough_free_disk_stops_before_downloading(fake, monkeypatch, capsys):
    monkeypatch.setattr(downloads, "free_bytes", lambda path: 10 * 10**9)
    assert cli.main(["setup", "--yes"]) == 1
    assert "not enough free disk: 20.8 GB to download, 10.0 GB free" in capsys.readouterr().err
    assert fake.pulls() == []


def test_hub_info_reads_sizes_license_and_blobs_without_downloading(monkeypatch):
    class Lfs(dict):
        pass

    class Sibling:
        def __init__(self, name, size, blob, lfs=None):
            self.rfilename, self.size, self.blob_id, self.lfs = name, size, blob, lfs

    class Info:
        sha = "abc1234def"
        card_data = {"license": "other", "license_name": "qwen"}
        siblings = [Sibling("config.json", 10, "git1"), Sibling("model.safetensors", 99, "git2", Lfs(sha256="lfs2"))]

    calls = []

    class Api:
        def model_info(self, repo, **kw):
            calls.append((repo, kw))
            return Info()

    import huggingface_hub

    monkeypatch.setattr(huggingface_hub, "HfApi", Api)
    info = downloads.hub_info("owner/name")
    assert calls == [("owner/name", {"revision": "main", "files_metadata": True})]
    assert info == {"commit": "abc1234def", "license": "other (qwen)",
                    "files": [("config.json", 10, "git1"), ("model.safetensors", 99, "lfs2")]}


# ---------------------------------------------------------------- leaving items out

def test_no_hybrid_downloads_no_mtp_head(fake, capsys):
    assert cli.main(["setup", "--yes", "--no-hybrid"]) == 0
    out = capsys.readouterr().out
    assert MTP not in out and "To download: 19.9 GB." in out
    assert fake.pulls() == [("pull", MODEL, DRAFTER)] and fake.run == [] and fake.snapshots == []
    assert fake.service_install() == expected_service(fake.home / "pli", hybrid=False)


def test_no_drafter_serves_without_any_draft(fake, capsys):
    assert cli.main(["setup", "--yes", "--no-drafter"]) == 0
    out = capsys.readouterr().out
    assert DRAFTER not in out.split("Leave items out")[0] and MTP not in out
    assert fake.pulls() == [("pull", MODEL)] and fake.run == []
    args = fake.service_install()
    assert args[args.index("--drafter") + 1] == "none" and "--arg=--no-drafts" in args
    assert not any(a.startswith(("PLI_HYBRID", "PLI_MTP_HEAD")) for a in args)


def test_no_models_downloads_nothing_and_does_not_start(fake, capsys):
    assert cli.main(["setup", "--no-models"]) == 0
    out = capsys.readouterr().out
    assert "Before anything is downloaded" not in out
    assert f"pull {MODEL} {DRAFTER}'" in out
    assert "For the hybrid draft (DFlash2 + MTP head), run pli setup again without --no-models." in out
    assert fake.pulls() == [] and fake.run == [] and fake.snapshots == []
    assert fake.service_install() == expected_service(fake.home / "pli", start=False, hybrid=False)


def test_no_models_with_files_already_here_starts_with_them(fake, capsys):
    model, drafter, head = fake.tmp / "m", fake.tmp / "d", fake.tmp / "head.safetensors"
    model.mkdir()
    drafter.mkdir()
    head.write_bytes(b"x")
    code = cli.main(["setup", "--no-models", "--model", str(model), "--drafter", str(drafter),
                     "--mtp-head", str(head), "--port", "9000"])
    assert code == 0
    assert "the service uses the files already on this Mac" in capsys.readouterr().out
    args = fake.service_install()
    assert args[2] == str(model) and args[args.index("--drafter") + 1] == str(drafter)
    assert args[args.index("--port") + 1] == "9000"
    assert f"PLI_MTP_HEAD={head}" in args and "PLI_HYBRID=1" in args and args[-1] == "--start"
    assert fake.pulls() == [] and fake.run == []


def test_a_local_model_is_not_downloaded(fake, capsys):
    model = fake.tmp / "my-27b"
    model.mkdir()
    assert cli.main(["setup", "--yes", "--model", str(model)]) == 0
    out = capsys.readouterr().out
    assert f"Model (required): {model}" in out and "on this Mac already, not downloaded" in out
    assert fake.pulls() == [("pull", DRAFTER)]
    assert fake.snapshots == [MTP]                                       # the head's shapes come from the local model
    assert fake.run[0][3] == str(model)


def test_mtp_head_cannot_go_with_no_hybrid(fake, capsys):
    head = fake.tmp / "h.safetensors"
    head.write_bytes(b"x")
    assert cli.main(["setup", "--yes", "--no-hybrid", "--mtp-head", str(head)]) == 1
    assert "--mtp-head" in capsys.readouterr().err and fake.pli == []


def test_a_second_setup_keeps_the_recorded_prefix_port_and_model(fake, capsys):
    conf = fake.home / ".config" / "pli" / "install.conf"
    conf.parent.mkdir(parents=True)
    other = fake.tmp / "elsewhere"
    conf.write_text(f'PREFIX="{other}"\nPORT="9100"\nMODEL="{MODEL}"\n')
    assert cli.main(["setup", "--yes", "--no-start"]) == 0
    args = fake.service_install()
    assert args[args.index("--port") + 1] == "9100" and f"PLI_EVENTS={other}/logs/events.jsonl" in args
    assert "--start" not in args


# ---------------------------------------------------------------- status, start, stop, restart, logs

@pytest.mark.parametrize("verb, expected", [
    ("status", ("service", "status", "pli")),
    ("stop", ("service", "stop", "pli")),
    ("start", ("service", "start", "pli")),
    ("restart", ("service", "restart", "pli")),
    ("logs", ("service", "logs", "pli", "--lines", "80")),
])
def test_service_verbs_are_pli_service_on_the_pli_profile(fake, capsys, verb, expected):
    assert cli.main([verb]) == 0
    assert fake.pli == [(expected, False)]
    waited = "PLI is ready on port 8080." in capsys.readouterr().out
    assert waited == (verb in ("start", "restart"))


def test_start_waits_on_the_recorded_port_and_warns_after_three_minutes(fake, monkeypatch, capsys):
    conf = fake.home / ".config" / "pli" / "install.conf"
    conf.parent.mkdir(parents=True)
    conf.write_text('PREFIX="/x"\nPORT="9200"\nMODEL="m/m"\n')
    ports = []
    monkeypatch.setattr(commands, "_health", lambda port: ports.append(port) and False)
    assert cli.main(["start"]) == 0
    assert ports == [9200] * 90
    assert "did not answer within 3 minutes: pli logs" in capsys.readouterr().err


# ---------------------------------------------------------------- uninstall

def _installed(fake):
    prefix = fake.home / "pli"
    (prefix / "venv" / "bin").mkdir(parents=True)
    commands.write_conf(prefix, 8080, MODEL)
    for folder in (fake.home / ".cache" / "tensorfold", fake.home / "Library" / "Logs" / "TensorFold",
                   fake.home / "Library" / "Application Support" / "TensorFold"):
        folder.mkdir(parents=True)
    for repo in (MODEL, DRAFTER, MTP):
        blobs = downloads.repo_dir(repo) / "blobs"
        blobs.mkdir(parents=True)
        (blobs / "x").write_bytes(b"0" * 1000)
    return prefix


def test_uninstall_yes_removes_everything(fake, capsys):
    prefix = _installed(fake)
    assert cli.main(["uninstall", "--yes"]) == 0
    assert fake.pli == [(("service", "stop", "pli"), True), (("service", "uninstall", "pli", "--yes"), False)]
    assert not prefix.exists() and not (fake.home / ".config" / "pli").exists()
    assert not (fake.home / ".cache" / "tensorfold").exists()
    assert not (fake.home / "Library" / "Logs" / "TensorFold").exists()
    assert not any(downloads.repo_dir(r).exists() for r in (MODEL, DRAFTER, MTP))
    assert "PLI is uninstalled." in capsys.readouterr().out


def test_uninstall_keep_models_and_asking_per_model(fake, monkeypatch, capsys):
    prefix = _installed(fake)
    answers = iter(["y", "n", "y", ""])
    monkeypatch.setattr(builtins, "input", lambda prompt="": next(answers))
    assert cli.main(["uninstall"]) == 0                       # yes to uninstall, then no, yes, no for the models
    assert not prefix.exists()
    assert downloads.repo_dir(MODEL).exists() and not downloads.repo_dir(DRAFTER).exists()
    assert downloads.repo_dir(MTP).exists()


def test_uninstall_no_keeps_everything(fake, monkeypatch):
    prefix = _installed(fake)
    monkeypatch.setattr(builtins, "input", lambda prompt="": "")
    assert cli.main(["uninstall"]) == 0
    assert prefix.exists() and fake.pli == []


def test_uninstall_refuses_the_home_folder(fake, capsys):
    assert cli.main(["uninstall", "--yes", "--prefix", str(fake.home)]) == 1
    assert "refusing to delete" in capsys.readouterr().err and fake.home.exists()


# ---------------------------------------------------------------- pli bench

def test_the_bench_suite_is_package_data():
    from tensorfold.bench import pli_bench

    suite = json.loads(pli_bench.SUITE.read_text())
    assert suite["suite"] == "pli-bench-v1" and (pli_bench.SUITE.parent / suite["corpus"]).is_file()
    pyproject = (ROOT / "pyproject.toml").read_text()
    assert '"tensorfold.bench.data" = ["*.json", "*.txt"]' in pyproject


def test_pli_bench_runs_the_suite_on_the_local_port_with_thinking_off(fake, monkeypatch):
    from tensorfold.bench import pli_bench

    seen = []
    monkeypatch.setattr(pli_bench, "run_suite", lambda a: seen.append(a))
    commands.write_conf(fake.home / "pli", 8090, MODEL)
    assert cli.main(["bench", "--only", "agent", "--salt", "pub"]) == 0
    (a,) = seen
    assert a.url == "http://127.0.0.1:8090/v1" and a.only == "agent" and a.salt == "pub"
    assert json.loads(a.extra) == {"chat_template_kwargs": {"enable_thinking": False}}
    assert a.suite == str(pli_bench.SUITE)


def test_pli_bench_help_is_the_bench_s(capsys):
    with pytest.raises(SystemExit) as stop:
        cli.main(["bench", "--help"])
    assert stop.value.code == 0
    out = capsys.readouterr().out
    assert out.startswith("usage: pli bench") and "--sizes" in out


def test_tools_pli_bench_still_runs_from_a_clone():
    spec = importlib.util.spec_from_file_location("pli_bench_tool", ROOT / "tools" / "pli_bench.py")
    tool = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(tool)
    assert callable(tool.load_suite) and callable(tool.piece) and callable(tool.main)
    help_text = subprocess.run([sys.executable, "-I", str(ROOT / "tools" / "pli_bench.py"), "--help"],
                               capture_output=True, text=True, check=True).stdout
    assert "src/tensorfold/bench/data/suite-v1.json" in help_text


# ---------------------------------------------------------------- install.sh

def _install_sh(home: Path, *args: str) -> subprocess.CompletedProcess:
    env = {"HOME": str(home), "PATH": os.environ.get("PATH", "/usr/bin:/bin")}
    return subprocess.run(["bash", str(ROOT / "install.sh"), *args], capture_output=True, text=True, env=env,
                          check=False)


def test_install_sh_passes_the_service_commands_to_pli(tmp_path):
    home = tmp_path / "home"
    prefix = tmp_path / "somewhere"
    pli = prefix / "venv" / "bin" / "pli"
    pli.parent.mkdir(parents=True)
    pli.write_text('#!/bin/sh\necho "pli $*"\n')
    pli.chmod(0o755)
    (home / ".config" / "pli").mkdir(parents=True)
    (home / ".config" / "pli" / "install.conf").write_text(f'PREFIX="{prefix}"\nPORT="8080"\nMODEL="{MODEL}"\n')
    for verb in ("status", "start", "stop", "restart", "logs"):
        done = _install_sh(home, verb)
        assert (done.returncode, done.stdout) == (0, f"pli {verb}\n")
    done = _install_sh(home, "uninstall", "--keep-models", "--yes")
    assert done.stdout == "pli uninstall --keep-models --yes\n"


def test_install_sh_without_pli_and_without_a_command(tmp_path):
    done = _install_sh(tmp_path, "status")
    assert done.returncode == 1 and "PLI is not installed" in done.stderr
    done = _install_sh(tmp_path)
    assert done.returncode == 1 and "./install.sh install [options]" in done.stdout


def _fake_python(tmp_path: Path) -> tuple[Path, Path]:
    """A python3.12 on PATH whose `-m venv DIR` makes a venv of fake pip and pli; both log their arguments."""

    bin_dir, templates, log = tmp_path / "bin", tmp_path / "templates", tmp_path / "log"
    bin_dir.mkdir()
    templates.mkdir()
    for name in ("pip", "pli"):
        (templates / name).write_text(f'#!/bin/sh\necho "{name} $*" >> "{log}"\n')
        (templates / name).chmod(0o755)
    python = bin_dir / "python3.12"
    python.write_text(f"""#!/bin/sh
case "$1" in
    -c) exit 0 ;;
    -V) echo "Python 3.12.9" ;;
    -m) mkdir -p "$3/bin" && cp "{templates}/pip" "{templates}/pli" "$3/bin/" ;;
esac
""")
    python.chmod(0o755)
    return bin_dir, log


def _run_install_sh(tmp_path: Path, *args: str, answer: str = "") -> tuple[subprocess.CompletedProcess, list[str]]:
    bin_dir, log = _fake_python(tmp_path)
    env = {"HOME": str(tmp_path / "home"), "PATH": f"{bin_dir}:/usr/bin:/bin"}
    done = subprocess.run(["bash", str(ROOT / "install.sh"), "install", *args], input=answer, capture_output=True,
                          text=True, env=env, check=False)
    return done, log.read_text().splitlines() if log.exists() else []


apple_silicon = pytest.mark.skipif(sys.platform != "darwin" or os.uname().machine != "arm64",
                                   reason="install.sh install checks for macOS on Apple Silicon first")


@apple_silicon
def test_install_sh_creates_the_venv_then_runs_pli_setup_with_every_option(tmp_path):
    prefix = tmp_path / "p"
    done, log = _run_install_sh(tmp_path, "--prefix", str(prefix), "--no-models", "--yes", "--port", "9000")
    assert done.returncode == 0, done.stderr
    assert done.stdout.startswith(f"Python: {tmp_path}/bin/python3.12 (Python 3.12.9)\n")
    assert f"Installing PLI from {ROOT} into {prefix}/venv..." in done.stdout
    assert log == ["pip install --quiet --upgrade pip", f"pip install --quiet {ROOT}", "pli --version",
                   f"pli setup --prefix {prefix} --no-models --yes --port 9000"]


@apple_silicon
@pytest.mark.parametrize("answer, expected", [
    ("n\n", []),
    ("y\n", ["pli service stop pli", "pip install --quiet --upgrade pip"]),
])
def test_install_sh_asks_before_reinstalling(tmp_path, answer, expected):
    prefix = tmp_path / "p"
    (prefix / "venv" / "bin").mkdir(parents=True)
    bin_dir, log = _fake_python(tmp_path)
    (prefix / "venv" / "bin" / "pli").write_text((tmp_path / "templates" / "pli").read_text())
    (prefix / "venv" / "bin" / "pli").chmod(0o755)
    done = subprocess.run(["bash", str(ROOT / "install.sh"), "install", "--prefix", str(prefix)], input=answer,
                          capture_output=True, text=True, check=False,
                          env={"HOME": str(tmp_path / "home"), "PATH": f"{bin_dir}:/usr/bin:/bin"})
    assert done.returncode == 0, done.stderr
    lines = log.read_text().splitlines() if log.exists() else []
    assert lines[:len(expected)] == expected and (lines == [] if not expected else len(lines) == 5)
