#!/usr/bin/env bash
#
# run.sh — Run Python commands locally (via uv) or on a remote GPU server
#           (via pre-built venv + tmux), with efficient code syncing.
#
# Local mode:  uses uv directly (no remote args needed).
# Remote mode: ensures a machine is running (via provider or direct SSH),
#              builds a Docker image locally (for caching heavy deps like
#              flash-attn), extracts the .venv, transfers it to the remote,
#              syncs code via rsync, sets up credentials, and runs in tmux.
#              No Docker daemon is needed on the remote — just CUDA + Python.
#
# Usage:
#   ./run.sh [OPTIONS] [-- <python args>]
#
# Options:
#   --ip HOST             Remote server IP/hostname
#   -p PORT               SSH port (default: 22)
#   -i IDENTITY_FILE      SSH identity file
#   --user USER           SSH user (default: root)
#   --provider NAME       Cloud provider: runpod, vast.ai
#   --pod-id ID           Existing pod/instance ID (for restart)
#   --pod-config FILE     Pod configuration file (default: deploy/runpod/pod.conf)
#   --remote-project-dir PATH   Remote workspace root (default: /workspace)
#   --remote-home PATH          Remote home directory (default: /root)
#   --pulse INTERVAL      Install pulse cron job for monitoring + auto-restart
#                         Supports: 1m, 5m, 1h (minimum: 1m, cron-based)
#   --pulse-idle-stop     Stop pod and remove cron if GPU stays below 5% for 3
#                         consecutive pulse checks (requires --pulse)
#   --help                Show this help
#
# Examples:
#   # Local — run via uv:
#   ./run.sh -- -m src.train --config configs/default.yaml
#
#   # Remote — existing server with known IP:
#   ./run.sh --ip 1.2.3.4 -p 22222 -- -m src.train --config configs/default.yaml
#
#   # Remote — auto-create a RunPod pod from config:
#   ./run.sh --provider runpod -- -m src.train --config configs/default.yaml
#
#   # Remote — restart an existing RunPod pod:
#   ./run.sh --provider runpod --pod-id abc123 -- -m src.train --config configs/default.yaml
#
#   # Remote — with pulse auto-restart monitoring:
#   ./run.sh --ip 1.2.3.4 -p 22222 --provider runpod --pod-id abc123 --pulse 5m -- -m src.train
#
#   # Remote — with pulse + idle-stop (shuts down when GPU goes idle):
#   ./run.sh --provider runpod --pulse 5m --pulse-idle-stop -- -m src.train
#
#   # Remote — setup only (no python args = no execution):
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
PYTHON_ARGS=()

# ════════════════════════════════════════════════════════════════════════════
# ── Usage ──────────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

usage() {
    cat <<'USAGE'
Usage: ./run.sh [OPTIONS] [-- <python args>]

Run a Python command locally (via uv) or on a remote server (via pre-built
venv + tmux). If no python args are provided, remote setup runs without execution.

Options:
  --ip HOST             Remote server IP/hostname
  -p PORT               SSH port (default: 22)
  -i IDENTITY_FILE      SSH identity file
  --user USER           SSH user (default: root)
  --provider NAME       Cloud provider: runpod, vast.ai
  --pod-id ID           Existing pod/instance ID (for restart)
  --pod-config FILE     Pod config file (default: deploy/runpod/pod.conf)
  --remote-project-dir PATH   Remote workspace root (default: /workspace)
  --remote-home PATH          Remote home directory (default: /root)
  --pulse INTERVAL      Install pulse cron job for monitoring + auto-restart (e.g. 5m, 1h)
  --pulse-idle-stop     Stop pod and remove cron if GPU idle <5% for 3 checks (requires --pulse)
  --help                Show this help

Examples:
  # Local:
  ./run.sh -- -m src.train --config configs/default.yaml

  # Remote (existing server):
  ./run.sh --ip 1.2.3.4 -p 22222 -- -m src.train --config configs/default.yaml

  # Remote (auto-create RunPod pod):
  ./run.sh --provider runpod -- -m src.train --config configs/default.yaml

  # Remote (restart existing pod + pulse monitoring):
  ./run.sh --provider runpod --pod-id abc123 --pulse 5m -- -m src.train

  # Remote (pulse + idle-stop):
  ./run.sh --provider runpod --pulse 5m --pulse-idle-stop -- -m src.train

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
        log_done "Venv ready. Nothing to run (no python args)."
        exit 0
    fi

    log_success "▶ Running locally via uv …"

    # Expose Python headers so torch.compile / triton can find Python.h
    _py_include="$(uv run python -c 'import sysconfig; print(sysconfig.get_path("include"))')"
    export CFLAGS="-I${_py_include} ${CFLAGS:-}"
    export CPATH="${_py_include}${CPATH:+:$CPATH}"

    exec uv run python "${PYTHON_ARGS[@]}"
fi

# ════════════════════════════════════════════════════════════════════════════
# ── Remote mode ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

if [[ ${#PYTHON_ARGS[@]} -eq 0 ]]; then
    log_warn "Note: No python arguments provided — will sync/setup only (no remote execution)."
fi

# ════════════════════════════════════════════════════════════════════════════
# ── Deployment pipeline ──────────────────────────────────────────────────
# The full auto-create/restart pipeline:
#   1. Ensure machine is running (create/start pod if needed, verify SSH)
#   2. Build venv locally (Docker-based, cached by dep hashes)
#   3. Transfer venv to remote (skipped if already up-to-date)
#   4. Sync code to remote via rsync
#   5. Setup credentials on remote (WANDB, etc.)
#   6. Ensure uv package manager on remote
#   7. Launch training session in tmux
# ════════════════════════════════════════════════════════════════════════════

# ── Step 1: Ensure machine is running + SSH-reachable ─────────────────────
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  Step 1/7 — Ensure machine is running${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
ensure_machine
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

# ── Step 2: Build Docker image + venv locally ───────────────────────────────
echo ""
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN}  Step 2/7 — Build venv locally${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
build_venv

# ── Step 3: Transfer .venv to remote ────────────────────────────────────────
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
# run_remote sets: SESSION_NAME (if python args were provided)

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
    if [[ -z "$POD_ID" && "$PROVIDER" == "runpod" ]]; then
        log_warn "No pod ID available — pulse will use SSH monitoring only (cannot restart via RunPod API)."
        log_info "Hint: pass --pod-id <ID>, or omit --ip to let the script create the pod automatically."
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
[[ -n "$POD_ID" ]] && echo -e "  ${BLUE}Pod:${NC}      curl -s -H 'Authorization: Bearer \$RUNPOD_API_KEY' https://rest.runpod.io/v1/pods/${POD_ID} | jq ."
[[ -n "$POD_ID" ]] && echo -e "  ${BLUE}Stop pod:${NC} ./deploy/runpod/stop_pod.sh ${POD_ID}"
[[ -n "$PULSE_INTERVAL" ]] && echo -e "  ${BLUE}Stop pulse:${NC} ./deploy/pulse.sh --uninstall-cron"
echo ""
