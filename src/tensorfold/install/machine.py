"""What ``pli setup`` checks on the Mac before it installs anything: system, chip, memory, disk, Command Line Tools."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import platform
import shutil
import subprocess
import sys


class SetupError(Exception):
    """Stops ``pli setup`` or ``pli uninstall``: the message is printed as ``error: ...`` (code 0: no error)."""

    def __init__(self, message: str, code: int = 1):
        super().__init__(message)
        self.code = code


def say(text: str) -> None:
    print(text, flush=True)


def warn(text: str) -> None:
    print(f"warning: {text}", file=sys.stderr, flush=True)


@dataclass
class Mac:
    system: str           # platform.system(): Darwin on macOS
    arch: str             # platform.machine(): arm64 on Apple Silicon
    macos: str            # sw_vers -productVersion
    ram_gb: int
    chip: str             # machdep.cpu.brand_string
    free_gb: int          # free disk in the home folder
    command_line_tools: bool


def _out(*command: str) -> str:
    try:
        return subprocess.run(command, capture_output=True, text=True, check=False).stdout.strip()
    except OSError:
        return ""


def probe() -> Mac:
    """This Mac (or this system: the checks then refuse it)."""

    system, arch = platform.system(), platform.machine()
    if system != "Darwin":
        return Mac(system, arch, "", 0, "", 0, False)
    try:
        tools = subprocess.run(["xcode-select", "-p"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                               check=False).returncode == 0
    except OSError:
        tools = False
    return Mac(system, arch, _out("sw_vers", "-productVersion"),
               int(_out("sysctl", "-n", "hw.memsize") or 0) // 1024**3,
               _out("sysctl", "-n", "machdep.cpu.brand_string"),
               shutil.disk_usage(Path.home()).free // 1024**3, tools)


def check(mac: Mac) -> tuple[str, int]:
    """Refuse a system PLI does not run on; print the Mac and its warnings. Returns (profile, Neural Engine 1 or 0)."""

    if mac.system != "Darwin":
        raise SetupError("PLI runs on macOS only")
    if mac.arch != "arm64":
        raise SetupError("PLI needs Apple Silicon (arm64)")
    try:
        major = int(mac.macos.split(".")[0])
    except ValueError:
        major = 0
    if major < 14:
        raise SetupError(f"macOS 14 or later is needed (this Mac: {mac.macos})")
    say(f"Mac: {mac.chip}, {mac.ram_gb} GB memory, macOS {mac.macos}, {mac.free_gb} GB free on disk")
    if mac.ram_gb < 48:
        warn(f"the 27B model needs about 48 GB of memory; with {mac.ram_gb} GB use a smaller model")
    if mac.free_gb < 70:
        warn("about 70 GB of free disk is advised (models 20 GB, snapshots up to 40 GB)")
    profile = ""
    if "Ultra" in mac.chip and mac.ram_gb >= 64:
        profile = "m1-ultra"
    else:
        warn("the m1-ultra profile is tuned for the M1 Ultra with 64 GB and given to Ultra chips with 64 GB or more; "
             "PLI will run with its defaults")
    ane = 1
    if not mac.command_line_tools:
        warn("Xcode Command Line Tools missing (xcode-select --install): the Neural Engine path is turned off")
        ane = 0
    return profile, ane


def check_alone() -> None:
    """Warn about programs that use 5% of memory or more: PLI works best on a Mac that runs nothing else."""

    busy = []
    for line in _out("ps", "-Ao", "%mem=,comm=").splitlines():
        fields = line.split()
        try:
            heavy = bool(fields) and float(fields[0].replace(",", ".")) >= 5
        except ValueError:
            continue
        if heavy and "/pli" not in line and "tensorfold" not in line:
            busy.append(line)
    if busy:
        warn("these programs use 5% of memory or more; PLI works best on a Mac that runs nothing else:")
        print("\n".join(busy), file=sys.stderr, flush=True)
