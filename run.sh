#!/usr/bin/env bash
#
# run.sh — Run commands locally (via uv/bash) or on a remote GPU server
#           (via pre-built venv + tmux), with efficient code syncing.
#
# Local mode:  runs the given command in bash (e.g. uv run python ... or any script).
# Remote mode: ensures a machine is running (via provider or direct SSH),
#              builds a Docker image locally (for caching heavy deps like
#              flash-attn), extracts the .venv, transfers it to the remote,
#              syncs code via rsync, sets up credentials, and runs the command in tmux.
#              No Docker daemon is needed on the remote — just CUDA + Python.
#
# Usage:
#   ./run.sh [OPTIONS] [-- <command>]
#
# The command after -- is run as a bash command (local and remote). Examples:
#   uv run python src/trainer.py config.yaml
#   uv run python src/data_processor/download.py --download_dir=./data && uv run python src/trainer.py experiment/autoencoder/compression_ae.yaml
#   ./scripts/my_script.sh
#
# Options:
#   --ip HOST             Remote server IP/hostname
#   -p PORT               SSH port (default: 22)
#   -i IDENTITY_FILE      SSH identity file
#   --user USER           SSH user (default: root)
#   --provider NAME       Cloud provider: runpod, vast.ai
#   --pod-id ID           Existing pod/instance ID (for restart)
#   --pod-config FILE     Pod configuration file (default: deploy/runpod/pod_a40.conf)
#                         If omitted, deploy/runpod/pod_configs.list is used to try
#                         configs in order until one succeeds (fallback on create failure).
#   --remote-project-dir PATH   Remote workspace root (default: /workspace)
#   --remote-home PATH          Remote home directory (default: /root)
#   --pulse INTERVAL      Install pulse cron job for monitoring + auto-restart
#                         Supports: 1m, 5m, 1h (minimum: 1m, cron-based)
#   --pulse-idle-stop     Stop pod and remove cron if GPU stays below 5% for 3
#                         consecutive pulse checks (requires --pulse)
#   --help                Show this help
#
# Examples:
#   # Local — run a single Python module:
#   ./run.sh -- uv run python -m src.train --config configs/default.yaml
#
#   # Local — download then train (chained command; quote so shell doesn't split on &&):
#   ./run.sh -- "uv run python src/data_processor/download.py --download_dir=./data --dataset_name=div2k && uv run python src/trainer.py experiment/autoencoder/compression_ae.yaml"
#
#   # Local — run a bash script:
#   ./run.sh -- ./scripts/prepare_data.sh
#
#   # Remote — existing server with known IP:
#   ./run.sh --ip 1.2.3.4 -p 22222 -- uv run python -m src.train --config configs/default.yaml
#
#   # Remote — auto-create a RunPod pod from config:
#   ./run.sh --provider runpod -- uv run python -m src.train --config configs/default.yaml
#
#   # Remote — auto-create a Vast.ai instance:
#   ./run.sh --provider vast.ai -- uv run python -m src.train --config configs/default.yaml
#
#   # Remote — restart an existing RunPod pod:
#   ./run.sh --provider runpod --pod-id abc123 -- uv run python -m src.train --config configs/default.yaml
#
#   # Remote — with pulse auto-restart monitoring:
#   ./run.sh --ip 1.2.3.4 -p 22222 --provider runpod --pod-id abc123 --pulse 5m -- uv run python -m src.train
#
#   # Remote — with pulse + idle-stop (shuts down when GPU goes idle):
#   ./run.sh --provider runpod --pulse 5m --pulse-idle-stop -- uv run python -m src.train
#
#   # Remote — setup only (no command = no execution):
#   ./run.sh --ip 1.2.3.4 -p 22222
#

set -euo pipefail

# ════════════════════════════════════════════════════════════════════════════
# ── Source deploy scripts ──────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

DEPLOY_DIR="$(cd "$(dirname "$0")/deploy" && pwd)"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"

source "${DEPLOY_DIR}/common.sh"
source "${DEPLOY_DIR}/build_venv.sh"
source "${DEPLOY_DIR}/push_image.sh"
source "${DEPLOY_DIR}/ensure_machine.sh"
source "${DEPLOY_DIR}/sync_venv.sh"
source "${DEPLOY_DIR}/sync_code.sh"
source "${DEPLOY_DIR}/setup_creds.sh"
source "${DEPLOY_DIR}/setup_uv.sh"
source "${DEPLOY_DIR}/run_remote.sh"

# ════════════════════════════════════════════════════════════════════════════
# ── Defaults ───────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

REMOTE_IP=""
REMOTE_PORT=""
REMOTE_USER="root"
SSH_IDENTITY=""
PROVIDER=""
POD_ID=""
POD_CONFIG=""
REMOTE_PROJECT_DIR="/workspace"
REMOTE_HOME="/root"
PULSE_INTERVAL=""
PULSE_IDLE_STOP=false
# Command to run (everything after --); stored as array for quoting/session state
PYTHON_ARGS=()

# ════════════════════════════════════════════════════════════════════════════
# ── Usage ──────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

usage() {
    cat <<'USAGE'
Usage: ./run.sh [OPTIONS] [-- <command>]

Run a bash command locally or on a remote server (via pre-built venv + tmux).
The command after -- is executed as-is (e.g. "uv run python ..." or "./script.sh").
For chained commands (e.g. with &&), quote the whole command: -- "cmd1 && cmd2".
If no command is provided, remote setup runs without execution.

Options:
  --ip HOST             Remote server IP/hostname
  -p PORT               SSH port (default: 22)
  -i IDENTITY_FILE      SSH identity file
  --user USER           SSH user (default: root)
  --provider NAME       Cloud provider: runpod, vast.ai
  --pod-id ID           Existing pod/instance ID (for restart)
  --pod-config FILE     Pod config file (default: deploy/runpod/pod_a40.conf).
                        If omitted, deploy/runpod/pod_configs.list is used to try
                        configs in order until pod creation succeeds (fallbacks).
  --remote-project-dir PATH   Remote workspace root (default: /workspace)
  --remote-home PATH          Remote home directory (default: /root)
  --pulse INTERVAL      Install pulse cron job for monitoring + auto-restart (e.g. 5m, 1h)
  --pulse-idle-stop     Stop pod and remove cron if GPU idle <5% for 3 checks (requires --pulse)
  --help                Show this help

Examples:
  # Local (single command):
  ./run.sh -- uv run python -m src.train --config configs/default.yaml

  # Local (chained commands; quote so the shell doesn't split on &&):
  ./run.sh -- "uv run python src/data_processor/download.py --download_dir=./data --dataset_name=div2k && uv run python src/trainer.py experiment/autoencoder/compression_ae.yaml"

  # Local (bash script):
  ./run.sh -- ./scripts/prepare_data.sh

  # Remote (existing server):
  ./run.sh --ip 1.2.3.4 -p 22222 -- uv run python -m src.train --config configs/default.yaml

  # Remote (auto-create RunPod pod):
  ./run.sh --provider runpod -- uv run python -m src.train --config configs/default.yaml

  # Remote (auto-create Vast.ai instance):
  ./run.sh --provider vast.ai -- uv run python -m src.train --config configs/default.yaml

  # Remote (restart existing pod + pulse monitoring):
  ./run.sh --provider runpod --pod-id abc123 --pulse 5m -- uv run python -m src.train

  # Remote (pulse + idle-stop):
  ./run.sh --provider runpod --pulse 5m --pulse-idle-stop -- uv run python -m src.train

  # Remote (setup only):
  ./run.sh --ip 1.2.3.4 -p 22222
USAGE
    exit 0
}

# ════════════════════════════════════════════════════════════════════════════
# ── Parse arguments ────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

while [[ $# -gt 0 ]]; do
    case "$1" in
        --ip)
            [[ $# -lt 2 ]] && { log_error "--ip requires HOST argument."; exit 1; }
            REMOTE_IP="$2"; shift 2 ;;
        -p|--port)
            [[ $# -lt 2 ]] && { log_error "-p requires PORT argument."; exit 1; }
            REMOTE_PORT="$2"; shift 2 ;;
        -i)
            [[ $# -lt 2 ]] && { log_error "-i requires IDENTITY_FILE argument."; exit 1; }
            SSH_IDENTITY="$2"; shift 2 ;;
        --user)
            [[ $# -lt 2 ]] && { log_error "--user requires USER argument."; exit 1; }
            REMOTE_USER="$2"; shift 2 ;;
        --provider)
            [[ $# -lt 2 ]] && { log_error "--provider requires NAME argument."; exit 1; }
            PROVIDER="$2"; shift 2 ;;
        --pod-id)
            [[ $# -lt 2 ]] && { log_error "--pod-id requires ID argument."; exit 1; }
            POD_ID="$2"; shift 2 ;;
        --pod-config)
            [[ $# -lt 2 ]] && { log_error "--pod-config requires FILE argument."; exit 1; }
            POD_CONFIG="$2"; shift 2 ;;
        --remote-project-dir)
            [[ $# -lt 2 ]] && { log_error "--remote-project-dir requires PATH argument."; exit 1; }
            REMOTE_PROJECT_DIR="$2"; shift 2 ;;
        --remote-home)
            [[ $# -lt 2 ]] && { log_error "--remote-home requires PATH argument."; exit 1; }
            REMOTE_HOME="$2"; shift 2 ;;
        --pulse)
            [[ $# -lt 2 ]] && { log_error "--pulse requires INTERVAL argument."; exit 1; }
            PULSE_INTERVAL="$2"; shift 2 ;;
        --pulse-idle-stop)
            PULSE_IDLE_STOP=true; shift ;;
        --help|-h)
            usage ;;
        --)
            shift
            PYTHON_ARGS=("$@")
            break ;;
        *)
            log_error "Unknown option: $1"
            echo "Use --help for usage information."
            exit 1 ;;
    esac
done

# ════════════════════════════════════════════════════════════════════════════
# ── Local mode ─────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

if [[ -z "$REMOTE_IP" && -z "$PROVIDER" ]]; then
    cd "$REPO_ROOT"

    # # ── Build Docker image + venv (same as remote pipeline) ──────────────
    # log_success "▶ Building venv via Docker …"
    # build_venv

    # # ── Extract .venv from Docker to local .venv if needed ───────────────
    # local_hash_file="${REPO_ROOT}/.venv/.deps_hash"
    # local_hash=""
    # [[ -f "$local_hash_file" ]] && local_hash=$(cat "$local_hash_file")

    # if [[ "$DEPS_CHANGED" == true ]] || [[ "$local_hash" != "$CURRENT_HASH" ]]; then
    #     log_step "Extracting .venv from Docker image to local .venv …"

    #     container_id=$(docker create "${DOCKER_IMAGE}:latest" true)

    #     rm -rf "${REPO_ROOT}/.venv"
    #     mkdir -p "${REPO_ROOT}/.venv"

    #     if ! docker cp "$container_id:/app/.venv" - | \
    #             tar xf - -C "${REPO_ROOT}/.venv" --strip-components=1; then
    #         docker rm "$container_id" >/dev/null 2>&1 || true
    #         log_error "Failed to extract .venv from Docker image."
    #         exit 1
    #     fi

    #     docker rm "$container_id" >/dev/null 2>&1 || true

    #     echo "$CURRENT_HASH" > "${REPO_ROOT}/.venv/.deps_hash"
    #     log_done "Local .venv updated from Docker."
    # else
    #     log_done "Local .venv is up-to-date — skipping extraction."
    # fi

    if [[ ${#PYTHON_ARGS[@]} -eq 0 ]]; then
        log_done "Nothing to run (no command after --)."
        exit 0
    fi

    log_success "▶ Running locally …"

    # Expose Python headers so torch.compile / triton can find Python.h
    _py_include="$(uv run python -c 'import sysconfig; print(sysconfig.get_path("include"))')"
    export CFLAGS="-I${_py_include} ${CFLAGS:-}"
    export CPATH="${_py_include}${CPATH:+:$CPATH}"

    run_cmd_quoted=$(printf '%q ' "${PYTHON_ARGS[@]}")
    exec bash -c "$run_cmd_quoted"
fi

# ════════════════════════════════════════════════════════════════════════════
# ── Remote mode ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

if [[ ${#PYTHON_ARGS[@]} -eq 0 ]]; then
    log_warn "Note: No command provided — will sync/setup only (no remote execution)."
fi

# SSH retry and timeout for transient "Network is unreachable" and slow pod bring-up
export SSH_CONNECT_TIMEOUT="${SSH_CONNECT_TIMEOUT:-20}"
export SSH_RETRY_ATTEMPTS="${SSH_RETRY_ATTEMPTS:-5}"
export SSH_RETRY_INTERVAL="${SSH_RETRY_INTERVAL:-10}"

# ════════════════════════════════════════════════════════════════════════════
# ── Deployment pipeline ──────────────────────────────────────────────────
# Provider + GITHUB_USERNAME: build → push to ghcr.io → ensure machine with
#   that image → symlink venv on remote. Otherwise: ensure machine → build →
#   stream venv → sync code (existing behavior).
# ════════════════════════════════════════════════════════════════════════════

USE_REGISTRY_VENV=false
if [[ -n "${PROVIDER:-}" && -n "${GITHUB_USERNAME:-}" ]]; then
    USE_REGISTRY_VENV=true
    export GHRC_IMAGE
fi

if [[ -n "${PROVIDER:-}" && -n "${GITHUB_USERNAME:-}" ]]; then
    # ── Provider path: build and push image, then create instance with it ───
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  Step 1/7 — Build venv locally${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    build_venv

    if [[ "$DEPS_CHANGED" == true || "${FORCE_PUSH:-0}" == "1" ]]; then
        echo ""
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${CYAN}  Step 2/7 — Push image to ghcr.io${NC}"
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        push_image
    else
        echo ""
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        echo -e "${CYAN}  Step 2/7 — Push image to ghcr.io (skipped, up-to-date)${NC}"
        echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
        log_done "Image unchanged — using existing registry image."
    fi

    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  Step 3/7 — Ensure machine is running${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    ensure_machine
else
    # ── Manual --ip or provider without GITHUB_USERNAME: ensure machine first ─
    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  Step 1/7 — Ensure machine is running${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    ensure_machine

    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  Step 2/7 — Build venv locally${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    build_venv
fi
# ensure_machine sets/updates: REMOTE_IP, REMOTE_PORT, POD_ID
# It also calls build_ssh_opts and verifies SSH connectivity.

# Setup derived remote paths
setup_remote_paths

echo ""
echo -e "${BLUE}Remote host:${NC}        ${REMOTE_HOST}"
echo -e "${BLUE}Remote code dir:${NC}    ${REMOTE_CODE_DIR}"
echo -e "${BLUE}Remote venv:${NC}        ${REMOTE_VENV_DIR}"
echo -e "${BLUE}Remote logs:${NC}        ${REMOTE_LOGS_DIR}"
echo -e "${BLUE}Remote home:${NC}        ${REMOTE_HOME}"
[[ -n "$POD_ID" ]] && echo -e "${BLUE}Pod ID:${NC}             ${POD_ID}"
[[ -n "$PROVIDER" ]] && echo -e "${BLUE}Provider:${NC}           ${PROVIDER}"
if [[ "${USE_REGISTRY_VENV:-false}" == true && -n "${GHRC_IMAGE:-}" ]]; then
    echo -e "${BLUE}Registry image:${NC}    ${GHRC_IMAGE}"
fi

# ── Step 3/7 (manual) or 4/7 (provider): Sync venv to remote ─────────────────
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  Step 3/7 — Sync venv to remote${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
sync_venv

# ── Step 4: Sync code via rsync ─────────────────────────────────────────────
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  Step 4/7 — Sync code to remote${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
sync_code

# ── Step 5: Setup credentials on remote ─────────────────────────────────────
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  Step 5/7 — Setup credentials${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
setup_creds

# ── Step 6: Ensure uv is on remote ─────────────────────────────────────────
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  Step 6/7 — Ensure uv on remote${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
setup_uv

# ── Step 7: Run command in tmux on remote ───────────────────────────────────
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  Step 7/7 — Launch remote session${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
run_remote
# run_remote sets: SESSION_NAME (if a command was provided)

# ── Save session state (for pulse auto-restart) ────────────────────────────
save_session_state

# ── Install pulse cron job if requested ─────────────────────────────────────
if [[ "$PULSE_IDLE_STOP" == true && -z "$PULSE_INTERVAL" ]]; then
    log_warn "--pulse-idle-stop has no effect without --pulse. Ignoring."
fi

if [[ -n "$PULSE_INTERVAL" && ${#PYTHON_ARGS[@]} -gt 0 ]]; then
    # Warn if we have a RunPod provider but no pod ID (happens when the user
    # provides --ip manually, bypassing pod creation).  Without a pod ID,
    # pulse can only do SSH-based monitoring and cannot restart via the API.
    if [[ -z "$POD_ID" && "${PROVIDER:-}" == "runpod" ]]; then
        log_warn "No pod ID available — pulse will use SSH monitoring only (cannot restart via RunPod API)."
        log_info "Hint: pass --pod-id <ID>, or omit --ip to let the script create the pod automatically."
    fi
    if [[ -z "$POD_ID" && "${PROVIDER:-}" == "vast.ai" ]]; then
        log_warn "No instance ID available — pulse will use SSH monitoring only (cannot restart via Vast.ai API)."
        log_info "Hint: pass --pod-id <ID>, or omit --ip to let the script create the instance automatically."
    fi

    echo ""
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${CYAN}  Installing Pulse cron job (auto-restart enabled)${NC}"
    echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"

    PULSE_CMD=("${DEPLOY_DIR}/pulse.sh"
        --install-cron
        --session-file "$SESSION_STATE_FILE"
        --interval "$PULSE_INTERVAL"
        --auto-restart
    )
    [[ "$PULSE_IDLE_STOP" == true ]] && PULSE_CMD+=(--idle-stop)

    # Target info (pod ID, SSH host, port) is read from the session file at
    # each cron invocation, not baked into the cron entry.  This is essential
    # for RECOVERY_POLICY="new" where auto-restart creates a fresh pod with
    # a new ID/IP each time.

    "${PULSE_CMD[@]}"
fi

# ── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
echo -e "${GREEN}  All done!${NC}"
echo -e "${GREEN}════════════════════════════════════════════════════${NC}"
if [[ -n "${SESSION_NAME:-}" ]]; then
    echo -e "  ${BLUE}Attach:${NC}   ssh${SSH_DISPLAY} ${REMOTE_HOST} -t 'tmux attach -t ${SESSION_NAME}'"
    echo -e "  ${BLUE}Logs:${NC}     ssh${SSH_DISPLAY} ${REMOTE_HOST} 'tail -f ${REMOTE_LOGS_DIR}/${SESSION_NAME}.log'"
    echo -e "  ${BLUE}Sessions:${NC} ssh${SSH_DISPLAY} ${REMOTE_HOST} 'tmux ls'"
fi
[[ -n "$POD_ID" && "${PROVIDER:-}" == "runpod" ]] && echo -e "  ${BLUE}Pod:${NC}      curl -s -H 'Authorization: Bearer \$RUNPOD_API_KEY' https://rest.runpod.io/v1/pods/${POD_ID} | jq ."
[[ -n "$POD_ID" && "${PROVIDER:-}" == "runpod" ]] && echo -e "  ${BLUE}Stop pod:${NC} ./deploy/runpod/stop_pod.sh ${POD_ID}"
[[ -n "$POD_ID" && "${PROVIDER:-}" == "vast.ai" ]] && echo -e "  ${BLUE}Instance:${NC} curl -s -H 'Authorization: Bearer \$VAST_API_KEY' https://console.vast.ai/api/v0/instances/${POD_ID}/ | jq ."
[[ -n "$POD_ID" && "${PROVIDER:-}" == "vast.ai" ]] && echo -e "  ${BLUE}Stop instance:${NC} ./deploy/vastai/stop_instance.sh ${POD_ID}"
[[ -n "$POD_ID" && "${PROVIDER:-}" == "vast.ai" ]] && echo -e "  ${BLUE}Start instance:${NC} ./deploy/vastai/start_instance.sh ${POD_ID}"
[[ -n "$PULSE_INTERVAL" ]] && echo -e "  ${BLUE}Stop pulse:${NC} ./deploy/pulse.sh --uninstall-cron"
echo ""
