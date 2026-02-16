#!/usr/bin/env bash
#
# pulse.sh — Periodically check if a server/instance is active.
#             Optionally auto-restart and re-deploy on failure.
#
# Supports multiple backends (RunPod via runpodctl, generic SSH) and logs
# status over time. When --auto-restart is enabled and the server dies,
# pulse will restart the machine, re-sync code/venv, and re-launch the
# training session automatically.
#
# Usage:
#   ./deploy/pulse.sh [OPTIONS]
#
# Options:
#   --session-file FILE   Session state file (written by run.sh)
#   --pod POD_ID          Monitor a RunPod pod by ID
#   --ssh USER@HOST       Monitor a generic server via SSH
#   -p PORT               SSH port (default: 22)
#   -i IDENTITY_FILE      SSH identity file
#   --interval DURATION   Check interval (default: 60s)
#                         Supports suffixes: s (seconds), m (minutes), h (hours)
#   --count N             Number of checks (default: unlimited)
#   --quiet               Only print on status change
#   --auto-restart        Auto-restart and re-deploy on failure
#   --help                Show this help
#
# Examples:
#   # Basic monitoring
#   ./deploy/pulse.sh --pod abc123xyz456 --interval 5m
#   ./deploy/pulse.sh --ssh root@192.168.1.100 -p 2222 --interval 30s
#
#   # Auto-restart (reads session state from run.sh)
#   ./deploy/pulse.sh --session-file deploy/logs/.session_state --interval 5m --auto-restart
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# ── Defaults ────────────────────────────────────────────────────────────────
POD_ID=""
SSH_TARGET=""
SSH_PORT=""
SSH_IDENTITY=""
INTERVAL="60"
MAX_COUNT=0
QUIET=false
AUTO_RESTART=false
SESSION_FILE=""

# ── Usage ───────────────────────────────────────────────────────────────────
usage() {
    cat <<'USAGE'
Usage: ./deploy/pulse.sh [OPTIONS]

Periodically check if a server/instance is active and reachable.
Optionally auto-restart and re-deploy on failure.

Options:
  --session-file FILE   Session state file (from run.sh, enables auto-restart)
  --pod POD_ID          Monitor a RunPod pod by ID
  --ssh USER@HOST       Monitor a generic server via SSH
  -p PORT               SSH port (default: 22)
  -i IDENTITY_FILE      SSH identity file
  --interval DURATION   Check interval (default: 60s, supports: 30s, 5m, 1h)
  --count N             Number of checks (default: unlimited)
  --quiet               Only print on status change
  --auto-restart        Auto-restart + re-deploy on failure
  --help                Show this help

Examples:
  ./deploy/pulse.sh --pod abc123xyz456 --interval 5m
  ./deploy/pulse.sh --ssh root@gpu-server -p 2222 --interval 30s
  ./deploy/pulse.sh --session-file deploy/logs/.session_state --interval 5m --auto-restart
USAGE
    exit 0
}

# ── Parse duration ──────────────────────────────────────────────────────────
parse_duration() {
    local input="$1"
    local value="${input%[smh]}"
    local unit="${input##*[0-9]}"
    [[ -z "$unit" ]] && unit="s"
    case "$unit" in
        s) echo "$value" ;;
        m) echo $((value * 60)) ;;
        h) echo $((value * 3600)) ;;
        *) log_error "Invalid duration unit: $unit (use s, m, or h)"; exit 1 ;;
    esac
}

format_interval() {
    local secs="$1"
    if [[ $secs -ge 3600 ]]; then
        echo "$((secs / 3600))h $((secs % 3600 / 60))m"
    elif [[ $secs -ge 60 ]]; then
        echo "$((secs / 60))m $((secs % 60))s"
    else
        echo "${secs}s"
    fi
}

# ── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --session-file)
            [[ $# -lt 2 ]] && { log_error "--session-file requires FILE."; exit 1; }
            SESSION_FILE="$2"; shift 2 ;;
        --pod)
            [[ $# -lt 2 ]] && { log_error "--pod requires POD_ID."; exit 1; }
            POD_ID="$2"; shift 2 ;;
        --ssh)
            [[ $# -lt 2 ]] && { log_error "--ssh requires USER@HOST."; exit 1; }
            SSH_TARGET="$2"; shift 2 ;;
        -p)
            [[ $# -lt 2 ]] && { log_error "-p requires PORT."; exit 1; }
            SSH_PORT="$2"; shift 2 ;;
        -i)
            [[ $# -lt 2 ]] && { log_error "-i requires IDENTITY_FILE."; exit 1; }
            SSH_IDENTITY="$2"; shift 2 ;;
        --interval)
            [[ $# -lt 2 ]] && { log_error "--interval requires DURATION."; exit 1; }
            INTERVAL=$(parse_duration "$2"); shift 2 ;;
        --count)
            [[ $# -lt 2 ]] && { log_error "--count requires N."; exit 1; }
            MAX_COUNT="$2"; shift 2 ;;
        --quiet)     QUIET=true; shift ;;
        --auto-restart) AUTO_RESTART=true; shift ;;
        --help|-h)   usage ;;
        *)           log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Load session state if provided ──────────────────────────────────────────
if [[ -n "$SESSION_FILE" && -f "$SESSION_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$SESSION_FILE"
    # Map session vars to pulse vars
    [[ -z "$POD_ID" && -n "${SESSION_POD_ID:-}" ]] && POD_ID="$SESSION_POD_ID"
    [[ -z "$SSH_TARGET" && -n "${SESSION_IP:-}" ]] && SSH_TARGET="${SESSION_USER:-root}@${SESSION_IP}"
    [[ -z "$SSH_PORT" && -n "${SESSION_PORT:-}" ]] && SSH_PORT="$SESSION_PORT"
    [[ -z "$SSH_IDENTITY" && -n "${SESSION_SSH_IDENTITY:-}" ]] && SSH_IDENTITY="$SESSION_SSH_IDENTITY"
fi

# ── Validate ────────────────────────────────────────────────────────────────
if [[ -z "$POD_ID" && -z "$SSH_TARGET" ]]; then
    log_error "Specify a target with --pod, --ssh, or --session-file."
    echo "Use --help for usage."
    exit 1
fi

if [[ "$AUTO_RESTART" == true && -z "$SESSION_FILE" ]]; then
    log_warn "Auto-restart requires --session-file to know how to re-deploy."
    log_warn "Will monitor only (no auto-restart)."
    AUTO_RESTART=false
fi

# Build SSH options for generic SSH checks
_PULSE_SSH_OPTS=()
[[ -n "$SSH_PORT" ]] && _PULSE_SSH_OPTS+=(-p "$SSH_PORT")
[[ -n "$SSH_IDENTITY" ]] && _PULSE_SSH_OPTS+=(-i "$SSH_IDENTITY")

# ── Check functions ─────────────────────────────────────────────────────────

check_runpod() {
    local pod_id="$1"
    local info
    info=$(runpodctl get pod "$pod_id" 2>&1) || { echo "error"; return; }
    if echo "$info" | grep -qi "running"; then echo "running"
    elif echo "$info" | grep -qi "exited"; then echo "exited"
    elif echo "$info" | grep -qi "stopped"; then echo "stopped"
    elif echo "$info" | grep -qi "starting"; then echo "starting"
    elif echo "$info" | grep -qi "created"; then echo "created"
    else echo "unknown"; fi
}

check_ssh_target() {
    local target="$1"
    if ssh "${_PULSE_SSH_OPTS[@]}" -o ConnectTimeout=10 -o BatchMode=yes \
           -o StrictHostKeyChecking=no "$target" "echo ok" &>/dev/null; then
        echo "reachable"
    else
        echo "unreachable"
    fi
}

# ── Auto-restart pipeline ──────────────────────────────────────────────────
# Sources all pipeline scripts and re-runs the full setup + launch.
do_auto_restart() {
    log_warn "Auto-restart triggered — re-provisioning and re-deploying …"

    # Load session state
    # shellcheck source=/dev/null
    source "$SESSION_FILE"

    # Map session vars to pipeline vars
    REMOTE_IP="${SESSION_IP:-}"
    REMOTE_PORT="${SESSION_PORT:-22}"
    SSH_IDENTITY="${SESSION_SSH_IDENTITY:-}"
    REMOTE_USER="${SESSION_USER:-root}"
    PROVIDER="${SESSION_PROVIDER:-}"
    POD_ID="${SESSION_POD_ID:-}"
    POD_CONFIG="${SESSION_POD_CONFIG:-}"
    REMOTE_PROJECT_DIR="${SESSION_REMOTE_PROJECT_DIR:-/workspace}"
    REMOTE_HOME="${SESSION_REMOTE_HOME:-/root}"

    # Reconstruct PYTHON_ARGS from the saved string
    eval "PYTHON_ARGS=( ${SESSION_PYTHON_ARGS:-} )"

    # Source pipeline scripts
    source "${DEPLOY_DIR}/build_venv.sh"
    source "${DEPLOY_DIR}/ensure_machine.sh"
    source "${DEPLOY_DIR}/sync_venv.sh"
    source "${DEPLOY_DIR}/sync_code.sh"
    source "${DEPLOY_DIR}/setup_creds.sh"
    source "${DEPLOY_DIR}/setup_uv.sh"
    source "${DEPLOY_DIR}/run_remote.sh"

    # Step 1: Ensure machine is running (restart/create as needed)
    if ! ensure_machine; then
        log_error "Failed to restart machine. Will retry on next pulse."
        return 1
    fi

    # Update session state with new IP/port
    setup_remote_paths

    # Step 2: Build venv locally
    build_venv

    # Step 3: Transfer venv
    sync_venv

    # Step 4: Sync code
    sync_code

    # Step 5: Setup credentials
    setup_creds

    # Step 6: Setup uv
    setup_uv

    # Step 7: Launch training in tmux
    run_remote

    # Step 8: Save updated session state
    save_session_state

    # Update pulse monitoring targets with new connection info
    if [[ -n "${SESSION_POD_ID:-}" ]]; then
        POD_ID="$SESSION_POD_ID"
    fi
    SSH_TARGET="${REMOTE_USER}@${REMOTE_IP}"
    SSH_PORT="$REMOTE_PORT"
    _PULSE_SSH_OPTS=()
    [[ -n "$SSH_PORT" ]] && _PULSE_SSH_OPTS+=(-p "$SSH_PORT")
    [[ -n "$SSH_IDENTITY" ]] && _PULSE_SSH_OPTS+=(-i "$SSH_IDENTITY")

    log_done "Auto-restart complete. Resuming monitoring."
    return 0
}

# ── Setup logging ───────────────────────────────────────────────────────────
mkdir -p "$DEPLOY_LOGS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)

if [[ -n "$POD_ID" ]]; then
    TARGET_LABEL="pod:${POD_ID}"
    LOG_FILE="${DEPLOY_LOGS_DIR}/pulse_pod_${POD_ID}_${TIMESTAMP}.log"
else
    SAFE_TARGET=$(echo "$SSH_TARGET" | tr '@:.' '_')
    TARGET_LABEL="ssh:${SSH_TARGET}"
    LOG_FILE="${DEPLOY_LOGS_DIR}/pulse_ssh_${SAFE_TARGET}_${TIMESTAMP}.log"
fi

{
    echo "# Pulse Monitor Log"
    echo "# Started: $(date -Iseconds)"
    echo "# Target:  ${TARGET_LABEL}"
    echo "# Interval: $(format_interval "$INTERVAL")"
    echo "# Auto-restart: ${AUTO_RESTART}"
    [[ $MAX_COUNT -gt 0 ]] && echo "# Max checks: ${MAX_COUNT}"
    echo ""
} > "$LOG_FILE"

# ── Print header ────────────────────────────────────────────────────────────
echo -e "${CYAN}╔═══════════════════════════════════════════════╗${NC}"
echo -e "${CYAN}║  Pulse Monitor                                ║${NC}"
echo -e "${CYAN}╠═══════════════════════════════════════════════╣${NC}"
echo -e "${CYAN}║${NC}  Target:       ${GREEN}${TARGET_LABEL}${NC}"
echo -e "${CYAN}║${NC}  Interval:     ${GREEN}$(format_interval "$INTERVAL")${NC}"
echo -e "${CYAN}║${NC}  Auto-restart: ${GREEN}${AUTO_RESTART}${NC}"
[[ $MAX_COUNT -gt 0 ]] && echo -e "${CYAN}║${NC}  Checks:       ${GREEN}${MAX_COUNT}${NC}"
echo -e "${CYAN}║${NC}  Log:          ${GREEN}${LOG_FILE}${NC}"
echo -e "${CYAN}╚═══════════════════════════════════════════════╝${NC}"
echo ""
echo -e "${YELLOW}Press Ctrl+C to stop monitoring.${NC}"
echo ""

# ── Main monitoring loop ────────────────────────────────────────────────────
PREV_STATUS=""
CHECK_NUM=0

trap 'echo -e "\n${YELLOW}Pulse stopped.${NC}"; echo "# Stopped: $(date -Iseconds)" >> "$LOG_FILE"; exit 0' INT TERM

while true; do
    CHECK_NUM=$((CHECK_NUM + 1))
    NOW=$(date '+%Y-%m-%d %H:%M:%S')

    # Perform check
    if [[ -n "$POD_ID" ]]; then
        STATUS=$(check_runpod "$POD_ID")
    else
        STATUS=$(check_ssh_target "$SSH_TARGET")
    fi

    # Determine status colour
    case "$STATUS" in
        running|reachable)  STATUS_COLOR="${GREEN}" ;;
        starting|created)   STATUS_COLOR="${YELLOW}" ;;
        stopped|exited|unreachable|error) STATUS_COLOR="${RED}" ;;
        *)                  STATUS_COLOR="${YELLOW}" ;;
    esac

    # Log the check
    echo "[${NOW}] #${CHECK_NUM} status=${STATUS}" >> "$LOG_FILE"

    # Print (respect --quiet: only print on change)
    if [[ "$QUIET" == false ]] || [[ "$STATUS" != "$PREV_STATUS" ]]; then
        if [[ "$STATUS" != "$PREV_STATUS" && -n "$PREV_STATUS" ]]; then
            echo -e "  ${YELLOW}⚡ Status changed: ${PREV_STATUS} → ${STATUS}${NC}"
        fi
        echo -e "  [${NOW}] ${BLUE}#${CHECK_NUM}${NC}  ${STATUS_COLOR}${STATUS}${NC}"
    fi

    # Auto-restart if server is down
    if [[ "$AUTO_RESTART" == true ]]; then
        case "$STATUS" in
            stopped|exited|unreachable|error)
                echo "[${NOW}] AUTO-RESTART triggered (status=${STATUS})" >> "$LOG_FILE"
                if do_auto_restart; then
                    echo "[${NOW}] AUTO-RESTART succeeded" >> "$LOG_FILE"
                    STATUS="restarted"
                else
                    echo "[${NOW}] AUTO-RESTART failed — will retry" >> "$LOG_FILE"
                    log_warn "Auto-restart failed. Will retry on next pulse."
                fi
                ;;
        esac
    fi

    PREV_STATUS="$STATUS"

    # Check if we've reached the limit
    if [[ $MAX_COUNT -gt 0 && $CHECK_NUM -ge $MAX_COUNT ]]; then
        echo ""
        log_done "Completed ${MAX_COUNT} checks."
        echo "# Completed: $(date -Iseconds) (${MAX_COUNT} checks)" >> "$LOG_FILE"
        break
    fi

    sleep "$INTERVAL"
done
