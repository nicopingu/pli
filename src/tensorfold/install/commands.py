"""``pli setup``, ``pli status|start|stop|restart|logs``, ``pli uninstall`` and ``pli bench``.

``pli setup`` does what ``./install.sh install`` did after creating the virtual environment: it checks the Mac,
shows what it will download and asks first, downloads the models with ``pli pull``, writes the MTP head in 4 bits,
and installs and starts the per-user service with ``pli service`` (profile name "pli"). The other commands are
``pli service`` verbs on that profile. The choices of ``pli setup`` are kept in ~/.config/pli/install.conf, the file
install.sh wrote.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import time
import urllib.request

from tensorfold.install import downloads, lsa_map, machine
from tensorfold.install.machine import SetupError, say, warn

NAME = "pli"                                    # `pli service` profile name
COMMANDS = ("setup", "status", "start", "stop", "restart", "logs", "uninstall", "bench")
PORT = 8080
THINKING_OFF = '{"chat_template_kwargs": {"enable_thinking": false}}'


# ---------------------------------------------------------------- settings kept between commands

def conf_file() -> Path:
    return Path.home() / ".config" / "pli" / "install.conf"


def read_conf() -> dict[str, str]:
    """PREFIX, PORT and MODEL from the last ``pli setup`` (KEY="value" lines, as install.sh wrote them)."""

    found: dict[str, str] = {}
    try:
        text = conf_file().read_text()
    except OSError:
        return found
    for line in text.splitlines():
        key, sep, value = line.strip().partition("=")
        if sep and key.isidentifier():
            found[key] = value.strip().strip('"')
    return found


def write_conf(prefix: Path, port: int, model: str) -> None:
    conf_file().parent.mkdir(parents=True, exist_ok=True)
    conf_file().write_text(f'PREFIX="{prefix}"\nPORT="{port}"\nMODEL="{model}"\n')


def _port(conf: dict[str, str]) -> int:
    try:
        return int(conf.get("PORT") or PORT)
    except ValueError:
        return PORT


# ---------------------------------------------------------------- what the commands run

def _pli(*args: str, quiet: bool = False) -> int:
    """Run ``pli ARGS`` with this Python (``pli service``, ``pli pull``); returns its exit code."""

    out = subprocess.DEVNULL if quiet else None
    return subprocess.run([sys.executable, "-m", "tensorfold", *args], stdout=out, stderr=out, check=False).returncode


def _run(*args: str) -> int:
    return subprocess.run([sys.executable, *args], check=False).returncode


def _snapshot(repo: str) -> Path:
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(repo))


def _health(port: int) -> bool:
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=5) as reply:
            return 200 <= reply.status < 300
    except Exception:  # noqa: BLE001 - not answering yet
        return False


_sleep = time.sleep


def _python() -> str:
    """The serving venv's Python: PREFIX/venv/bin/python when pli runs from a virtual environment."""

    venv = Path(sys.prefix) / "bin" / "python"
    return str(venv) if sys.prefix != sys.base_prefix and venv.exists() else sys.executable


def _pli_path() -> str:
    return str(Path(sys.executable).parent / "pli")


def ask(question: str, yes: bool) -> bool:
    if yes:
        return True
    try:
        answer = input(f"{question} [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def wait_ready(port: int) -> bool:
    say("Waiting for PLI (loading the model takes one to two minutes)...")
    for _ in range(90):
        if _health(port):
            say(f"PLI is ready on port {port}.")
            return True
        _sleep(2)
    warn("PLI did not answer within 3 minutes: pli logs")
    return False


# ---------------------------------------------------------------- pli setup

def service_args(*, model: str, python: str, prefix: Path, port: int, profile: str, ane: int, context: int | None,
                 free_gb: int, hybrid: bool, mtp_head: Path | None, start: bool, drafter: str,
                 no_drafts: bool = False, a1_map: Path | None = None) -> list[str]:
    """The ``pli service install`` command of install.sh, option for option."""

    args = ["service", "install", model, "--name", NAME, "--python", python, "--port", str(port),
            "--drafter", drafter, "--env", f"PLI_EVENTS={prefix}/logs/events.jsonl", "--replace"]
    if no_drafts:
        args.append("--arg=--no-drafts")
    if profile == "m1-ultra":
        # the Neural Engine path is switched on by the profile; without the Command Line Tools it is turned off here
        args += ["--env", f"PLI_ANE_FFN={ane}"]
        # MLX reads its command-buffer limits when it starts, before the profile: they go in the service environment
        args += ["--env", "MLX_MAX_MB_PER_BUFFER=500", "--env", "MLX_MAX_OPS_PER_BUFFER=100"]
    if profile:
        args.append(f"--arg=--profile={profile}")
    if context is not None:
        args += ["--context", str(context)]
    elif profile == "m1-ultra":
        args += ["--context", "229376", "--parallel", "4"]
    if free_gb >= 70:
        args.append("--arg=--spill-gib=40")
    if hybrid and mtp_head is not None:
        args += ["--env", "PLI_HYBRID=1", "--env", f"PLI_MTP_HEAD={mtp_head}"]
    if a1_map is not None:
        # A1 reads prompt chunks past 16 384 positions with the map; the profile keeps A1 to the prompt
        args += ["--env", f"PLI_A1_LSA_MAP={a1_map}"]
        args += [x for key, value in lsa_map.ENV.items() for x in ("--env", f"{key}={value}")]
    if start:
        args.append("--start")
    return args


def setup(a: argparse.Namespace) -> int:
    conf = read_conf()                  # a new install keeps the earlier prefix, port and model unless told otherwise
    prefix = Path(a.prefix or conf.get("PREFIX") or Path.home() / "pli").expanduser().absolute()
    port = a.port if a.port is not None else _port(conf)
    model = a.setup_model or conf.get("MODEL") or downloads.MODEL
    if a.mtp_head and (a.no_hybrid or a.no_drafter):
        raise SetupError("--mtp-head is the hybrid draft's head: it cannot go with --no-hybrid or --no-drafter")
    mac = machine.probe()
    profile, ane = machine.check(mac)
    say(f"Python: {sys.executable} (Python {platform.python_version()})")
    machine.check_alone()

    models = not a.no_models
    start = not a.no_start
    drafter = None if a.no_drafter else a.drafter
    # --no-models writes no MTP head: the hybrid draft needs one given with --mtp-head
    hybrid = not a.no_hybrid and drafter is not None and (models or bool(a.mtp_head))
    # A1's map is distilled for Qwen3.8-27B and its settings are tuned for the M1 Ultra: only with the m1-ultra profile
    a1 = profile == "m1-ultra" and not a.no_a1
    items = downloads.plan(model, drafter, a.mtp_head, hybrid, prefix, a1=a1)
    if models:
        downloads.measure(items)
        say(downloads.screen(items))
        downloads.consent(items, a.yes)
    write_conf(prefix, port, model)

    head = next((i.path for i in items if i.role == "mtp4"), None)
    if models:
        _download(items, prefix)
    else:
        missing = [i.repo for i in items if i.repo and not downloads.cached_complete(i.repo)]
        if missing:
            start = False
            say(f"Models not downloaded: run '{_pli_path()} pull {' '.join(missing)}', then {_pli_path()} start.")
        else:
            say("Models not downloaded: the service uses the files already on this Mac.")
    a1_map = _a1_map(items, models)

    choice = "none" if drafter is None else "auto" if drafter == downloads.DRAFTER else _local(drafter)
    _pli("service", "stop", NAME, quiet=True)          # a service from an earlier install is replaced, not doubled
    args = service_args(model=model, python=_python(), prefix=prefix, port=port, profile=profile, ane=ane,
                        context=a.context, free_gb=mac.free_gb, hybrid=hybrid, mtp_head=head, start=start,
                        drafter=choice, no_drafts=drafter is None, a1_map=a1_map)
    (prefix / "logs").mkdir(parents=True, exist_ok=True)
    code = _pli(*args)
    if code:
        raise SetupError(f"pli service install failed (exit code {code})", code)
    if start:
        wait_ready(port)
    say(f"Done. API: http://127.0.0.1:{port}/v1 (OpenAI) and http://127.0.0.1:{port} (Anthropic).")
    say(f"Settings: {_pli_path()} service status {NAME}; every PLI_* variable is described in docs/switches.md.")
    if not models and drafter is not None and not hybrid and not a.no_hybrid:
        say("The service uses the DFlash2 draft alone. For the hybrid draft (DFlash2 + MTP head), run pli setup "
            "again without --no-models.")
    return 0


def _a1_map(items: list[downloads.Item], models: bool) -> Path | None:
    """A1's map for the service: downloaded and converted (or already here), for a model it fits; None otherwise."""

    item = next((i for i in items if i.role == "lsa"), None)
    if item is None or item.path is None:
        return None
    if item.missing and models:
        say(f"Downloading A1's map ({lsa_map.FILE}), checking its SHA-256 and converting it...")
        try:
            lsa_map.download(item.path)
        except Exception as exc:  # noqa: BLE001 - network, checksum: PLI runs without A1, setup goes on
            warn(f"A1's map was not installed ({type(exc).__name__}: {exc}); PLI runs without A1. "
                 f"Run pli setup again to retry.")
            return None
    if not item.path.is_file():
        say("A1's map not downloaded: PLI runs without A1 (run pli setup again without --no-models for it).")
        return None
    model = next(i for i in items if i.role == "model")
    if model.path is not None:
        model_dir = model.path
    elif models or downloads.cached_complete(model.repo):
        model_dir = _snapshot(model.repo)
    else:
        model_dir = None
    if not lsa_map.fits(model_dir):
        warn("A1's map is distilled for Qwen3.8-27B; this model's config.json does not match it: PLI runs without A1")
        return None
    return item.path


def _local(name: str) -> str:
    path = Path(name).expanduser()
    return str(path.absolute()) if path.is_dir() else name


def _download(items: list[downloads.Item], prefix: Path) -> None:
    repos = [i.repo for i in items if i.repo and i.role in ("model", "drafter")]
    if repos:
        say(f"Downloading {' and '.join(repos)}...")
        code = _pli("pull", *repos)
        if code:
            raise SetupError(f"pli pull failed (exit code {code}); pli setup downloads what is missing when run again",
                             code)
    if any(i.role == "mtp" for i in items):
        head = next(i for i in items if i.role == "mtp4").path
        model = next(i for i in items if i.role == "model")
        say(f"Downloading the MTP head ({downloads.MTP_SOURCE}) and writing it in 4 bits...")
        head.parent.mkdir(parents=True, exist_ok=True)
        source = _snapshot(downloads.MTP_SOURCE)
        model_dir = model.path if model.path is not None else _snapshot(model.repo)
        code = _run("-m", "tensorfold.install.mtp_head", "--model", str(model_dir),
                    str(source / "model.safetensors"), str(head))
        if code:
            raise SetupError(f"writing the MTP head in 4 bits failed (exit code {code})", code)


# ---------------------------------------------------------------- pli uninstall

def _bytes(directory: Path) -> int:
    total = 0
    for root, _, files in os.walk(directory):
        for name in files:
            try:
                total += os.lstat(os.path.join(root, name)).st_size
            except OSError:
                pass
    return total


def uninstall(a: argparse.Namespace) -> int:
    conf = read_conf()
    home = Path.home()
    prefix = Path(a.prefix or conf.get("PREFIX") or home / "pli").expanduser().absolute()
    model = conf.get("MODEL") or downloads.MODEL
    if prefix.resolve() in (Path("/"), home.resolve()):
        raise SetupError(f"refusing to delete {prefix}: give the folder PLI was installed in with --prefix")
    hub = downloads.hub_cache()                        # read before the program's own folder goes
    say(f"This removes the service, {prefix} (program, MTP head, logs), its settings and logs, and the snapshots on "
        "disk.")
    if not ask("Uninstall PLI?", a.yes):
        return 0
    _pli("service", "stop", NAME, quiet=True)
    _pli("service", "uninstall", NAME, "--yes")
    for folder in (prefix, conf_file().parent, home / ".cache" / "tensorfold",
                   home / "Library" / "Application Support" / "TensorFold", home / "Library" / "Logs" / "TensorFold"):
        shutil.rmtree(folder, ignore_errors=True)
    if not a.keep_models:
        for repo in (model, downloads.DRAFTER, downloads.MTP_SOURCE):
            if not downloads._REPO_ID.match(repo):
                continue
            folder = hub / f"models--{repo.replace('/', '--')}"
            if folder.is_dir() and ask(f"Delete the downloaded model {repo} ({downloads.human(_bytes(folder))})?",
                                       a.yes):
                shutil.rmtree(folder, ignore_errors=True)
    say("PLI is uninstalled.")
    return 0


# ---------------------------------------------------------------- the service verbs and the bench

def service_verb(a: argparse.Namespace) -> int:
    if a.command == "logs":
        return _pli("service", "logs", NAME, "--lines", "80")
    code = _pli("service", a.command, NAME)
    if code == 0 and a.command in ("start", "restart"):
        wait_ready(_port(read_conf()))
    return code


def bench(argv: list[str]) -> int:
    """``pli bench``: the public suite against the local server (the port of ``pli setup``), thinking off."""

    from tensorfold.bench import pli_bench

    pli_bench.main(argv, prog="pli bench", url=f"http://127.0.0.1:{_port(read_conf())}/v1", extra=THINKING_OFF)
    return 0


def _guarded(command):
    def run(a: argparse.Namespace) -> int:
        try:
            return command(a)
        except SetupError as exc:
            if exc.code:
                print(f"error: {exc}", file=sys.stderr, flush=True)
            else:
                say(str(exc))
            return exc.code
        except KeyboardInterrupt:
            print("", file=sys.stderr)
            return 130
    return run


def register(commands) -> None:
    """Add setup, status, start, stop, restart, logs, uninstall and bench to the ``pli`` parser."""

    setup_parser = commands.add_parser(
        "setup", help="check this Mac, download the models (after asking), install and start the service",
        description="Check this Mac, list what will be downloaded and ask first, download the models, write the MTP "
                    "head in 4 bits, then install and start PLI as a per-user service (profile name pli).")
    setup_parser.add_argument("--prefix", metavar="DIR", help="folder for the MTP head and the logs (default: ~/pli, "
                              "or the folder of the last pli setup)")
    setup_parser.add_argument("--port", type=int, default=None, metavar="N", help="server port (default: 8080)")
    setup_parser.add_argument("--model", dest="setup_model", metavar="REPO|DIR",
                              help=f"Hugging Face model, or a model directory already on this Mac, not downloaded "
                                   f"(default: {downloads.MODEL})")
    setup_parser.add_argument("--drafter", default=downloads.DRAFTER, metavar="REPO|DIR",
                              help="DFlash2 draft model, or its directory on this Mac (default: %(default)s)")
    setup_parser.add_argument("--mtp-head", metavar="FILE",
                              help="an MTP head already written (4-bit or bf16 safetensors): nothing downloaded for it")
    setup_parser.add_argument("--context", type=int, default=None, metavar="N",
                              help="context window in tokens (default: 229376 with an Ultra chip and 64 GB, else "
                                   "chosen by PLI)")
    setup_parser.add_argument("--no-hybrid", action="store_true", help="DFlash2 draft only, without the MTP head")
    setup_parser.add_argument("--no-a1", action="store_true",
                              help="no A1 map (Latent Sparse Attention): long prompts are read without it")
    setup_parser.add_argument("--no-drafter", action="store_true",
                              help="no draft at all: neither DFlash2 nor the MTP head (slower, the same text)")
    setup_parser.add_argument("--no-start", action="store_true",
                              help="install the service without starting it now (it starts at the next login)")
    setup_parser.add_argument("--no-models", action="store_true",
                              help="download nothing: use --model, --drafter and --mtp-head for files already here")
    setup_parser.add_argument("--yes", "-y", action="store_true",
                              help="answer yes to every question, the download included")
    setup_parser.set_defaults(func=_guarded(setup))
    for verb, text in (("status", "is the service loaded, is the server answering, which model"),
                       ("start", "start the service and wait until the server answers"),
                       ("stop", "stop the service (it finishes writing its snapshots)"),
                       ("restart", "restart the service and wait until the server answers"),
                       ("logs", "the last lines of the server log")):
        verb_parser = commands.add_parser(verb, help=text)
        verb_parser.set_defaults(func=_guarded(service_verb))
    remove = commands.add_parser("uninstall", help="remove the service, its folder, settings, logs and snapshots; "
                                 "asks before deleting each downloaded model")
    remove.add_argument("--keep-models", action="store_true", help="keep the downloaded models")
    remove.add_argument("--yes", "-y", action="store_true", help="answer yes to every question")
    remove.add_argument("--prefix", metavar="DIR", help="the folder of pli setup (default: the one it recorded)")
    remove.set_defaults(func=_guarded(uninstall))
    bench_parser = commands.add_parser("bench", add_help=False,
                                       help="run the public bench suite pli-bench-v1 against the local server "
                                            "(pli bench --help)")
    bench_parser.add_argument("bench_args", nargs=argparse.REMAINDER)
    bench_parser.set_defaults(func=lambda a: bench(a.bench_args))
