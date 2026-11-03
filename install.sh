#!/usr/bin/env bash
# PLI installer for a dedicated Apple Silicon Mac: installs PLI in its own virtual environment, then runs `pli setup`.
#
#   ./install.sh install [options]   install PLI in ~/pli/venv (or --prefix DIR), then run: pli setup [options]
#   ./install.sh start | stop | restart | status | logs
#   ./install.sh uninstall [--keep-models] [--yes]
#
# The other commands are those of PLI: ./install.sh status is pli status, and so on. Every option of install goes to
# `pli setup` (pli setup --help): --prefix DIR, --port N, --model REPO|DIR, --drafter REPO|DIR, --mtp-head FILE,
# --context N, --no-hybrid, --no-drafter, --no-start, --no-models, --yes. pli setup checks the Mac, lists what it
# will download (source, license, size, where it goes) and asks before downloading, writes the MTP head in 4 bits,
# and installs and starts the per-user service.
#
# PLI is made for a Mac that does nothing else: it keeps about 36 GB of memory wired, uses the whole GPU and the
# Neural Engine, and writes up to 40 GB of conversation snapshots to disk. Other heavy programs running at the same
# time slow it down and can make it refuse requests for lack of memory.
#
# The service is a per-user LaunchAgent managed by `pli service` (profile name "pli"): it starts at login and
# restarts after a crash. On a Mac without a screen, turn on automatic login so that it starts after a reboot.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE="$HOME/.config/pli/install.conf"          # written by pli setup: gives the prefix to the other commands
PREFIX="$HOME/pli"; YES=0

say()  { printf '%s\n' "$*"; }
die()  { printf 'error: %s\n' "$*" >&2; exit 1; }
ask()  {                                             # ask "question": 0 for yes
    [ "$YES" = 1 ] && return 0
    local r; read -r -p "$1 [y/N] " r || true
    case "$r" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
}

load_conf() {
    # shellcheck source=/dev/null
    [ -f "$CONF_FILE" ] && . "$CONF_FILE"
    return 0
}

parse() {                                            # only --prefix and --yes; pli reads every option
    while [ $# -gt 0 ]; do
        case "$1" in
            --prefix) [ $# -ge 2 ] || die "--prefix needs a folder"; PREFIX="$2"; shift ;;
            --prefix=*) PREFIX="${1#--prefix=}" ;;
            --yes|-y) YES=1 ;;
        esac
        shift
    done
}

find_python() {
    local p
    for p in python3.12 python3.13 python3.14 python3.11 /opt/homebrew/bin/python3.12 /opt/homebrew/bin/python3 python3; do
        if command -v "$p" >/dev/null 2>&1 &&
           "$p" -c 'import sys; sys.exit(0 if (3, 11) <= sys.version_info[:2] <= (3, 14) else 1)' 2>/dev/null; then
            command -v "$p"; return 0
        fi
    done
    return 1
}

cmd_install() {
    # before pip: MLX has no wheel for another system, and pip's own error would not say why
    [ "$(uname -s)" = Darwin ] || die "PLI runs on macOS only"
    [ "$(uname -m)" = arm64 ] || die "PLI needs Apple Silicon (arm64)"
    local python
    python="$(find_python)" || die "Python 3.11 to 3.14 is needed: MLX 0.32.3 has no wheel for later versions (for example: brew install python@3.12)"
    say "Python: $python ($("$python" -V 2>&1))"
    if [ -e "$PREFIX/venv" ]; then
        ask "PLI is already installed in $PREFIX. Reinstall it?" || exit 0
        "$PREFIX/venv/bin/pli" service stop pli >/dev/null 2>&1 || true
    fi
    say "Installing PLI from $REPO_DIR into $PREFIX/venv..."
    mkdir -p "$PREFIX"
    "$python" -m venv "$PREFIX/venv"
    "$PREFIX/venv/bin/pip" install --quiet --upgrade pip
    "$PREFIX/venv/bin/pip" install --quiet "$REPO_DIR"
    "$PREFIX/venv/bin/pli" --version
    exec "$PREFIX/venv/bin/pli" setup "$@"
}

main() {
    local cmd="${1:-}"; [ $# -gt 0 ] && shift
    load_conf
    parse "$@"
    case "$cmd" in
        install) cmd_install "$@" ;;
        start|stop|restart|status|logs)
            [ -x "$PREFIX/venv/bin/pli" ] || die "PLI is not installed in $PREFIX: ./install.sh install"
            exec "$PREFIX/venv/bin/pli" "$cmd" ;;
        uninstall)
            [ -x "$PREFIX/venv/bin/pli" ] || die "PLI is not installed in $PREFIX: ./install.sh install"
            exec "$PREFIX/venv/bin/pli" uninstall "$@" ;;
        *) awk 'NR > 1 && /^#/ { sub(/^# ?/, ""); print; next } NR > 1 { exit }' "$0"; exit 1 ;;
    esac
}

main "$@"
