#!/usr/bin/env bash
# PLI installer for a dedicated Apple Silicon Mac.
#
#   ./install.sh install [options]   check the Mac, install PLI, download the models, start the service
#   ./install.sh start | stop | restart | status | logs
#   ./install.sh uninstall [--keep-models]
#
# Install options:
#   --prefix DIR       where PLI lives (default: ~/pli)
#   --port N           server port (default: 8080)
#   --model REPO       Hugging Face model (default: mlx-community/Qwen3.8-27B-4bit)
#   --context N        context window in tokens (default: 229376 with an Ultra chip and 64 GB, else chosen by PLI)
#   --no-hybrid        DFlash2 draft only, without the model's MTP head
#   --no-start         install the service without starting it now (it starts at the next login)
#   --no-models        do not download the models now
#   --yes              answer yes to every question
#
# PLI is made for a Mac that does nothing else: it keeps about 36 GB of memory wired, uses the whole GPU and the
# Neural Engine, and writes up to 40 GB of conversation snapshots to disk. Other heavy programs running at the same
# time slow it down and can make it refuse requests for lack of memory.
#
# The service is a per-user LaunchAgent managed by `pli service` (profile name "pli"): it starts at login and
# restarts after a crash. On a Mac without a screen, turn on automatic login so that it starts after a reboot.
set -euo pipefail

NAME="pli"                                           # `pli service` profile name
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONF_FILE="$HOME/.config/pli/install.conf"          # remembers the choices for start, stop and uninstall
DRAFTER="z-lab/Qwen3.8-27B-DFlash2"
MTP_SOURCE="EigenLabs/Qwen3.8-27B-MTP-bf16"
CACHE_DIRS=("$HOME/.cache/tensorfold")               # conversation snapshots and compiled extensions
SERVICE_DIRS=("$HOME/Library/Application Support/TensorFold" "$HOME/Library/Logs/TensorFold")

PREFIX="$HOME/pli"; PORT=8080; MODEL="mlx-community/Qwen3.8-27B-4bit"; CONTEXT=""
HYBRID=1; START=1; MODELS=1; YES=0; KEEP_MODELS=0

say()  { printf '%s\n' "$*"; }
warn() { printf 'warning: %s\n' "$*" >&2; }
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

parse() {
    while [ $# -gt 0 ]; do
        case "$1" in
            --prefix) PREFIX="$2"; shift ;;
            --port) PORT="$2"; shift ;;
            --model) MODEL="$2"; shift ;;
            --context) CONTEXT="$2"; shift ;;
            --no-hybrid) HYBRID=0 ;;
            --no-start) START=0 ;;
            --no-models) MODELS=0 ;;
            --keep-models) KEEP_MODELS=1 ;;
            --yes|-y) YES=1 ;;
            *) die "unknown option: $1 (see the top of install.sh)" ;;
        esac
        shift
    done
}

pli() { "$PREFIX/venv/bin/pli" "$@"; }

# ---------------------------------------------------------------- checks

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

check_mac() {
    [ "$(uname -s)" = Darwin ] || die "PLI runs on macOS only"
    [ "$(uname -m)" = arm64 ] || die "PLI needs Apple Silicon (arm64)"
    local major; major="$(sw_vers -productVersion | cut -d. -f1)"
    [ "$major" -ge 14 ] || die "macOS 14 or later is needed (this Mac: $(sw_vers -productVersion))"
    RAM_GB=$(( $(sysctl -n hw.memsize) / 1024 / 1024 / 1024 ))
    CHIP="$(sysctl -n machdep.cpu.brand_string)"
    FREE_GB=$(( $(df -k "$HOME" | awk 'NR == 2 {print $4}') / 1024 / 1024 ))
    say "Mac: $CHIP, ${RAM_GB} GB memory, macOS $(sw_vers -productVersion), ${FREE_GB} GB free on disk"
    [ "$RAM_GB" -ge 48 ] || warn "the 27B model needs about 48 GB of memory; with ${RAM_GB} GB use a smaller model"
    [ "$FREE_GB" -ge 70 ] || warn "about 70 GB of free disk is advised (models 20 GB, snapshots up to 40 GB)"
    PROFILE=""
    if [[ "$CHIP" == *Ultra* ]] && [ "$RAM_GB" -ge 64 ]; then
        PROFILE="m1-ultra"
    else
        warn "the m1-ultra profile is tuned for the M1 Ultra with 64 GB and given to Ultra chips with 64 GB or more; PLI will run with its defaults"
    fi
    ANE=1
    if ! xcode-select -p >/dev/null 2>&1; then
        warn "Xcode Command Line Tools missing (xcode-select --install): the Neural Engine path is turned off"
        ANE=0
    fi
    PYTHON="$(find_python)" || die "Python 3.11 to 3.14 is needed: MLX 0.32.3 has no wheel for later versions (for example: brew install python@3.12)"
    say "Python: $PYTHON ($("$PYTHON" -V 2>&1))"
}

check_alone() {
    local busy
    busy="$(ps -Ao %mem=,comm= | awk '$1 >= 5' | grep -v -e '/pli' -e tensorfold || true)"
    if [ -n "$busy" ]; then
        warn "these programs use 5% of memory or more; PLI works best on a Mac that runs nothing else:"
        printf '%s\n' "$busy" >&2
    fi
}

# ---------------------------------------------------------------- install

download_models() {
    say "Downloading $MODEL and $DRAFTER (about 20 GB)..."
    pli pull "$MODEL" "$DRAFTER"
    if [ "$HYBRID" = 1 ]; then
        say "Downloading the MTP head ($MTP_SOURCE) and writing it in 4 bits..."
        mkdir -p "$PREFIX/models"
        local src model_dir
        src="$("$PREFIX/venv/bin/python" -c "from huggingface_hub import snapshot_download as d; print(d('$MTP_SOURCE'))")"
        model_dir="$("$PREFIX/venv/bin/python" -c "from huggingface_hub import snapshot_download as d; print(d('$MODEL'))")"
        "$PREFIX/venv/bin/python" "$REPO_DIR/tools/quantize_mtp_head.py" --model "$model_dir" \
            "$src/model.safetensors" "$PREFIX/models/mtp-head-4bit.safetensors"
    fi
}

install_service() {
    local args=(service install "$MODEL" --name "$NAME" --python "$PREFIX/venv/bin/python" --port "$PORT"
                --drafter auto --env "PLI_EVENTS=$PREFIX/logs/events.jsonl" --replace)
    # the Neural Engine path is switched on by the profile; without the Command Line Tools it is turned off here
    [ "$PROFILE" = m1-ultra ] && args+=(--env "PLI_ANE_FFN=$ANE")
    # MLX reads its command-buffer limits when it starts, before the profile: they go in the service environment
    [ "$PROFILE" = m1-ultra ] && args+=(--env MLX_MAX_MB_PER_BUFFER=500 --env MLX_MAX_OPS_PER_BUFFER=100)
    [ -n "$PROFILE" ] && args+=(--arg="--profile=$PROFILE")
    if [ -n "$CONTEXT" ]; then args+=(--context "$CONTEXT")
    elif [ "$PROFILE" = m1-ultra ]; then args+=(--context 229376 --parallel 4); fi
    [ "$FREE_GB" -ge 70 ] && args+=(--arg=--spill-gib=40)
    if [ "$HYBRID" = 1 ]; then
        args+=(--env PLI_HYBRID=1 --env "PLI_MTP_HEAD=$PREFIX/models/mtp-head-4bit.safetensors")
    fi
    [ "$START" = 1 ] && args+=(--start)
    mkdir -p "$PREFIX/logs"
    pli "${args[@]}"
}

wait_ready() {
    say "Waiting for PLI (loading the model takes one to two minutes)..."
    local _
    for _ in $(seq 1 90); do
        if curl -fsS "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then say "PLI is ready on port $PORT."; return 0; fi
        sleep 2
    done
    warn "PLI did not answer within 3 minutes: ./install.sh logs"
}

cmd_install() {
    check_mac
    check_alone
    if [ -e "$PREFIX/venv" ]; then
        ask "PLI is already installed in $PREFIX. Reinstall it?" || exit 0
        pli service stop "$NAME" >/dev/null 2>&1 || true
    fi
    say "Installing PLI from $REPO_DIR into $PREFIX/venv..."
    mkdir -p "$PREFIX"
    "$PYTHON" -m venv "$PREFIX/venv"
    "$PREFIX/venv/bin/pip" install --quiet --upgrade pip
    "$PREFIX/venv/bin/pip" install --quiet "$REPO_DIR"
    pli --version
    mkdir -p "$(dirname "$CONF_FILE")"
    printf 'PREFIX="%s"\nPORT="%s"\nMODEL="%s"\n' "$PREFIX" "$PORT" "$MODEL" > "$CONF_FILE"
    if [ "$MODELS" = 1 ]; then
        download_models
    else
        START=0
        HYBRID=0                                     # no MTP head written: DFlash2 draft alone
        say "Models not downloaded: run '$PREFIX/venv/bin/pli pull $MODEL $DRAFTER', then ./install.sh start."
    fi
    install_service
    [ "$START" = 1 ] && wait_ready
    say "Done. API: http://127.0.0.1:$PORT/v1 (OpenAI) and http://127.0.0.1:$PORT (Anthropic)."
    say "Settings: $PREFIX/venv/bin/pli service status $NAME; every PLI_* variable is described in docs/switches.md."
    if [ "$MODELS" = 0 ]; then
        say "The service uses the DFlash2 draft alone. For the hybrid draft (DFlash2 + MTP head), run ./install.sh install again."
    fi
}

# ---------------------------------------------------------------- uninstall

cmd_uninstall() {
    say "This removes the service, $PREFIX (program, MTP head, logs), its settings and logs, and the snapshots on disk."
    ask "Uninstall PLI?" || exit 0
    if [ -x "$PREFIX/venv/bin/pli" ]; then
        pli service stop "$NAME" >/dev/null 2>&1 || true
        pli service uninstall "$NAME" --yes || true
    fi
    rm -rf "$PREFIX" "$(dirname "$CONF_FILE")" "${CACHE_DIRS[@]}" "${SERVICE_DIRS[@]}"
    local hub="$HOME/.cache/huggingface/hub" r d
    if [ "$KEEP_MODELS" = 0 ]; then
        for r in "$MODEL" "$DRAFTER" "$MTP_SOURCE"; do
            d="$hub/models--${r//\//--}"
            if [ -d "$d" ] && ask "Delete the downloaded model $r ($(du -sh "$d" | cut -f1))?"; then rm -rf "$d"; fi
        done
    fi
    say "PLI is uninstalled."
}

main() {
    local cmd="${1:-}"; [ $# -gt 0 ] && shift
    load_conf
    parse "$@"
    case "$cmd" in
        install) cmd_install ;;
        start) pli service start "$NAME"; wait_ready ;;
        stop) pli service stop "$NAME" ;;
        restart) pli service restart "$NAME"; wait_ready ;;
        status) pli service status "$NAME" ;;
        logs) pli service logs "$NAME" --lines 80 ;;
        uninstall) cmd_uninstall ;;
        *) sed -n '2,23p' "$0" | sed 's/^# \{0,1\}//'; exit 1 ;;
    esac
}

main "$@"
