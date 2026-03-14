#!/usr/bin/env bash
#
# pulse.sh — Check if a server/instance is active, optionally auto-restart.
#             Designed to be run as a cron job for periodic monitoring.
#
# Single-check mode (default): performs one status check and exits.
# When --auto-restart is enabled and the server is down, pulse will
# restart the machine, re-sync code/venv, and re-launch the session.
#
# Multi-session support: each run.sh invocation creates a unique session
# state file (.session_state_<id>). Pulse installs a separate cron job
# per session, each with its own lock/state files. Multiple sessions can
# be monitored in parallel.
#
# Use --install-cron / --uninstall-cron to manage scheduled monitoring.
#
# Usage:
#   ./deploy/pulse.sh [OPTIONS]
#
# Commands:
#   --install-cron        Install a cron job for periodic monitoring
#   --uninstall-cron      Remove pulse cron job(s)
#                         With --session-file: removes cron for that session only
#                         Without --session-file: removes ALL pulse cron jobs
#   --list-sessions       List all tracked sessions and their cron status
#   (default)             Run a single health check
#
# Options:
#   --session-file FILE   Session state file (written by run.sh)
#   --pod POD_ID          Monitor a RunPod pod by ID
#   --ssh USER@HOST       Monitor a generic server via SSH
#   -p PORT               SSH port (default: 22)
#   -i IDENTITY_FILE      SSH identity file
#   --interval DURATION   Cron check interval (for --install-cron, default: 5m)
#                         Supports suffixes: s (seconds), m (minutes), h (hours)
#                         Minimum for cron: 1m
#   --auto-restart        Auto-restart and re-deploy on failure
#   --max-retries N       Max consecutive restart attempts before disabling (default: 5)
#   --idle-stop           Stop pod and remove cron if GPU usage stays below 5% for
#                         3 consecutive checks (requires SSH access to the machine)
#   --help                Show this help
#
# Examples:
#   # Install cron-based monitoring (every 5 minutes):
#   ./deploy/pulse.sh --session-file deploy/logs/.session_state_train-123 --interval 5m --auto-restart --install-cron
#
#   # Install with idle-stop (shuts down pod when GPU goes idle for 3 checks):
#   ./deploy/pulse.sh --session-file deploy/logs/.session_state_train-123 --interval 5m --auto-restart --idle-stop --install-cron
#
#   # Single manual check:
#   ./deploy/pulse.sh --session-file deploy/logs/.session_state_train-123 --auto-restart
#
#   # Remove cron for a specific session:
#   ./deploy/pulse.sh --uninstall-cron --session-file deploy/logs/.session_state_train-123
#
#   # Remove ALL pulse cron jobs:
#   ./deploy/pulse.sh --uninstall-cron
#
#   # List all tracked sessions:
#   ./deploy/pulse.sh --list-sessions
#

# Only use set -e during setup phase, NOT in the check section.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/common.sh"

# ── Defaults ────────────────────────────────────────────────────────────────
POD_ID=""
SSH_TARGET=""
SSH_PORT=""
SSH_IDENTITY=""
INTERVAL="300"  # 5 minutes default (for cron scheduling)
AUTO_RESTART=false
SESSION_FILE=""
MAX_RETRIES=5
INSTALL_CRON=false
UNINSTALL_CRON=false
IDLE_STOP=false
LIST_SESSIONS=false
GPU_IDLE_THRESHOLD=5      # percent — below this is considered idle
GPU_IDLE_MAX_STRIKES=3    # consecutive idle checks before auto-stop

# ── Usage ───────────────────────────────────────────────────────────────────
usage() {
    cat <<'USAGE'
Usage: ./deploy/pulse.sh [OPTIONS]

Check if a server/instance is active and optionally auto-restart on failure.
Designed to be run as a cron job for periodic monitoring.

Commands:
  --install-cron        Install a cron job for periodic monitoring
  --uninstall-cron      Remove pulse cron job(s)
                        With --session-file: removes cron for that session only
                        Without --session-file: removes ALL pulse cron jobs
  --list-sessions       List all tracked sessions and their cron status
  (default)             Run a single health check

Options:
  --session-file FILE   Session state file (from run.sh, enables auto-restart)
  --pod POD_ID          Monitor a RunPod pod by ID
  --ssh USER@HOST       Monitor a generic server via SSH
  -p PORT               SSH port (default: 22)
  -i IDENTITY_FILE      SSH identity file
  --interval DURATION   Cron interval (default: 5m, supports: 1m, 5m, 1h)
  --auto-restart        Auto-restart + re-deploy on failure
  --max-retries N       Max consecutive restart attempts (default: 5)
  --idle-stop           Stop pod and remove cron if GPU stays below 5% for 3 checks
  --help                Show this help

Examples:
  ./deploy/pulse.sh --session-file deploy/logs/.session_state_train-123 --interval 5m --auto-restart --install-cron
  ./deploy/pulse.sh --session-file deploy/logs/.session_state_train-123 --interval 5m --auto-restart --idle-stop --install-cron
  ./deploy/pulse.sh --session-file deploy/logs/.session_state_train-123 --auto-restart
  ./deploy/pulse.sh --uninstall-cron --session-file deploy/logs/.session_state_train-123
  ./deploy/pulse.sh --uninstall-cron
  ./deploy/pulse.sh --list-sessions
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

# ── Convert seconds to cron expression ──────────────────────────────────────
seconds_to_cron() {
    local secs="$1"
    if [[ $secs -lt 60 ]]; then
        log_warn "Cron minimum interval is 1 minute. Rounding up to 1m."
        echo "* * * * *"
    elif [[ $secs -lt 3600 ]]; then
        local mins=$((secs / 60))
        if [[ $mins -eq 1 ]]; then
            echo "* * * * *"
        else
            echo "*/${mins} * * * *"
        fi
    else
        local hours=$((secs / 3600))
        if [[ $hours -eq 1 ]]; then
            echo "0 * * * *"
        else
            echo "0 */${hours} * * *"
        fi
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
        --max-retries)
            [[ $# -lt 2 ]] && { log_error "--max-retries requires N."; exit 1; }
            MAX_RETRIES="$2"; shift 2 ;;
        --auto-restart)  AUTO_RESTART=true; shift ;;
        --idle-stop)     IDLE_STOP=true; shift ;;
        --install-cron)  INSTALL_CRON=true; shift ;;
        --uninstall-cron) UNINSTALL_CRON=true; shift ;;
        --list-sessions) LIST_SESSIONS=true; shift ;;
        --help|-h)   usage ;;
        *)           log_error "Unknown option: $1"; exit 1 ;;
    esac
done

# ── Derive session ID from a session file ───────────────────────────────────
# Extracts SESSION_ID by reading the file (prefers SESSION_ID, falls back to
# SESSION_NAME, then filename). Does NOT source the file to avoid side effects.
derive_session_id_from_file() {
    local file="$1" sid=""
    if [[ -f "$file" ]]; then
        sid=$(grep '^SESSION_ID=' "$file" 2>/dev/null | head -1 | cut -d'"' -f2)
        [[ -z "$sid" ]] && sid=$(grep '^SESSION_NAME=' "$file" 2>/dev/null | head -1 | cut -d'"' -f2)
    fi
    if [[ -z "$sid" ]]; then
        local bname
        bname=$(basename "$file")
        [[ "$bname" == .session_state_* ]] && sid="${bname#.session_state_}"
    fi
    echo "$sid"
}

# ── Cron tag helpers ────────────────────────────────────────────────────────
# Session-specific cron tag: elemental_pulse_<SESSION_ID>
# Legacy (single-session) tag: elemental_pulse
cron_tag_for_session() {
    local sid="$1"
    if [[ -n "$sid" ]]; then
        echo "elemental_pulse_${sid}"
    else
        echo "elemental_pulse"
    fi
}

# ── List sessions ──────────────────────────────────────────────────────────
list_sessions() {
    local files
    files=$(compgen -G "${DEPLOY_LOGS_DIR}/.session_state_*" 2>/dev/null) || files=""

    if [[ -z "$files" ]]; then
        echo "No active sessions found in ${DEPLOY_LOGS_DIR}/"
        return 0
    fi

    local crontab_content
    crontab_content=$(crontab -l 2>/dev/null) || crontab_content=""

    echo ""
    printf "  %-28s %-10s %-15s %-25s %s\n" "SESSION" "PROVIDER" "POD/INSTANCE" "HOST" "CRON"
    printf "  %-28s %-10s %-15s %-25s %s\n" "───────" "────────" "────────────" "────" "────"

    while IFS= read -r f; do
        [[ -f "$f" ]] || continue

        local sid provider pod_id ip port
        sid=$(grep '^SESSION_ID=' "$f" 2>/dev/null | head -1 | cut -d'"' -f2)
        [[ -z "$sid" ]] && sid=$(grep '^SESSION_NAME=' "$f" 2>/dev/null | head -1 | cut -d'"' -f2)
        [[ -z "$sid" ]] && sid=$(basename "$f" | sed 's/^\.session_state_//')
        provider=$(grep '^SESSION_PROVIDER=' "$f" 2>/dev/null | head -1 | cut -d'"' -f2)
        pod_id=$(grep '^SESSION_POD_ID=' "$f" 2>/dev/null | head -1 | cut -d'"' -f2)
        ip=$(grep '^SESSION_IP=' "$f" 2>/dev/null | head -1 | cut -d'"' -f2)
        port=$(grep '^SESSION_PORT=' "$f" 2>/dev/null | head -1 | cut -d'"' -f2)

        local host="${ip}"
        [[ -n "$port" && "$port" != "22" ]] && host="${host}:${port}"

        local cron_status="inactive"
        local tag
        tag=$(cron_tag_for_session "$sid")
        if echo "$crontab_content" | grep -q "# ${tag}$"; then
            cron_status="active"
        fi

        printf "  %-28s %-10s %-15s %-25s %s\n" \
            "$sid" "${provider:-ssh}" "${pod_id:--}" "${host:--}" "$cron_status"
    done <<< "$files"

    echo ""
    echo "  Session files: ${DEPLOY_LOGS_DIR}/.session_state_<SESSION>"
    echo "  Uninstall one: ./deploy/pulse.sh --uninstall-cron --session-file <FILE>"
    echo "  Uninstall all: ./deploy/pulse.sh --uninstall-cron"
    echo ""
}

# ── Uninstall cron: single session ─────────────────────────────────────────
uninstall_cron_session() {
    local sid="$1"
    local tag
    tag=$(cron_tag_for_session "$sid")

    local existing filtered
    existing=$(crontab -l 2>/dev/null) || existing=""
    filtered=$(echo "$existing" | grep -v "# ${tag}$" || true)
    filtered=$(echo "$filtered" | sed '/^[[:space:]]*$/d')
    if [[ -z "$filtered" ]]; then
        crontab -r 2>/dev/null || true
    else
        echo "$filtered" | crontab -
    fi

    rm -f "${DEPLOY_LOGS_DIR}/.pulse_failures_${sid}" 2>/dev/null || true
    rm -f "${DEPLOY_LOGS_DIR}/.pulse_low_gpu_${sid}" 2>/dev/null || true
    rm -f "${DEPLOY_LOGS_DIR}/.pulse_cron_wrapper_${sid}.sh" 2>/dev/null || true
    rm -f "${DEPLOY_LOGS_DIR}/.pulse.lock_${sid}" 2>/dev/null || true
    rm -f "${DEPLOY_LOGS_DIR}/.session_state_${sid}" 2>/dev/null || true
    log_done "Pulse cron job removed for session: ${sid}"
}

# ── Uninstall cron: all sessions ───────────────────────────────────────────
uninstall_cron_all() {
    local existing filtered
    existing=$(crontab -l 2>/dev/null) || existing=""
    # Match both legacy "# elemental_pulse" and session-specific "# elemental_pulse_*"
    filtered=$(echo "$existing" | grep -v "# elemental_pulse" || true)
    filtered=$(echo "$filtered" | sed '/^[[:space:]]*$/d')
    if [[ -z "$filtered" ]]; then
        crontab -r 2>/dev/null || true
    else
        echo "$filtered" | crontab -
    fi

    # Clean up all pulse state files (legacy + session-specific)
    rm -f "${DEPLOY_LOGS_DIR}"/.pulse_failures* 2>/dev/null || true
    rm -f "${DEPLOY_LOGS_DIR}"/.pulse_low_gpu* 2>/dev/null || true
    rm -f "${DEPLOY_LOGS_DIR}"/.pulse_cron_wrapper* 2>/dev/null || true
    rm -f "${DEPLOY_LOGS_DIR}"/.pulse.lock* 2>/dev/null || true
    # Also remove legacy state file without suffix
    rm -f "${DEPLOY_LOGS_DIR}/.pulse_failures" 2>/dev/null || true
    rm -f "${DEPLOY_LOGS_DIR}/.pulse_low_gpu" 2>/dev/null || true
    rm -f "${DEPLOY_LOGS_DIR}/.pulse_cron_wrapper.sh" 2>/dev/null || true
    log_done "All pulse cron jobs removed."
}

# ── Install cron ───────────────────────────────────────────────────────────
install_cron() {
    if [[ -z "$SESSION_ID" ]]; then
        log_error "Cannot install cron without a session ID. Provide --session-file."
        exit 1
    fi

    local cron_tag
    cron_tag=$(cron_tag_for_session "$SESSION_ID")

    local cron_expr
    cron_expr=$(seconds_to_cron "$INTERVAL")

    # Resolve session file to absolute path
    local abs_session_file="$SESSION_FILE"
    if [[ -n "$abs_session_file" && "${abs_session_file:0:1}" != "/" ]]; then
        abs_session_file="$(cd "$(dirname "$SESSION_FILE")" && pwd)/$(basename "$SESSION_FILE")"
    fi

    # Resolve SSH identity to absolute path
    local abs_ssh_identity="$SSH_IDENTITY"
    if [[ -n "$abs_ssh_identity" && "${abs_ssh_identity:0:1}" != "/" ]]; then
        abs_ssh_identity="$(cd "$(dirname "$SSH_IDENTITY")" && pwd)/$(basename "$SSH_IDENTITY")"
    fi

    # Build the pulse command args.
    # Target info (pod ID, SSH host, port) is read from the session file at
    # each invocation — NOT baked into the cron entry. Critical for
    # RECOVERY_POLICY="new" where auto-restart creates a fresh pod.
    local pulse_args=""
    [[ -n "$abs_session_file" ]] && pulse_args+=" --session-file ${abs_session_file}"
    if [[ -z "$abs_session_file" ]]; then
        [[ -n "$POD_ID" ]] && pulse_args+=" --pod ${POD_ID}"
        [[ -n "$SSH_TARGET" ]] && pulse_args+=" --ssh ${SSH_TARGET}"
        [[ -n "$SSH_PORT" ]] && pulse_args+=" -p ${SSH_PORT}"
        [[ -n "$abs_ssh_identity" ]] && pulse_args+=" -i ${abs_ssh_identity}"
    fi
    [[ "$AUTO_RESTART" == true ]] && pulse_args+=" --auto-restart"
    pulse_args+=" --max-retries ${MAX_RETRIES}"
    [[ "$IDLE_STOP" == true ]] && pulse_args+=" --idle-stop"

    local log_file="${DEPLOY_LOGS_DIR}/pulse_cron.log"
    mkdir -p "$DEPLOY_LOGS_DIR"

    # Session-specific wrapper script (cron has line-length limits).
    local wrapper_path="${DEPLOY_LOGS_DIR}/.pulse_cron_wrapper_${SESSION_ID}.sh"
    local abs_wrapper_path="$wrapper_path"
    [[ "${abs_wrapper_path:0:1}" != "/" ]] && abs_wrapper_path="$(cd "$(dirname "$wrapper_path")" && pwd)/$(basename "$wrapper_path")"
    local abs_repo_root="$REPO_ROOT"
    [[ "${abs_repo_root:0:1}" != "/" ]] && abs_repo_root="$(cd "$REPO_ROOT" && pwd)"
    local abs_script_dir="$SCRIPT_DIR"
    [[ "${abs_script_dir:0:1}" != "/" ]] && abs_script_dir="$(cd "$SCRIPT_DIR" && pwd)"
    local abs_log_file="$log_file"
    [[ "${abs_log_file:0:1}" != "/" ]] && abs_log_file="$(cd "$(dirname "$log_file")" && pwd)/$(basename "$log_file")"

    cat > "$wrapper_path" <<WRAPPER_EOF
#!/usr/bin/env bash
# Generated by pulse.sh --install-cron for session ${SESSION_ID}; do not edit.
set -euo pipefail
export PATH="${PATH}"
export HOME="${HOME}"
cd "${abs_repo_root}"
exec "${abs_script_dir}/pulse.sh"${pulse_args}
WRAPPER_EOF
    chmod +x "$wrapper_path"

    # Cron line: schedule + wrapper + redirect, tagged for identification.
    local cron_line="${cron_expr} ${abs_wrapper_path} >> ${abs_log_file} 2>&1 # ${cron_tag}"

    # Remove any existing entry for THIS session, then add the new one.
    local existing filtered
    existing=$(crontab -l 2>/dev/null) || existing=""
    filtered=$(echo "$existing" | grep -v "# ${cron_tag}$" || true)
    filtered=$(echo "$filtered" | sed '/^[[:space:]]*$/d')

    if [[ -n "$filtered" ]]; then
        printf '%s\n%s\n' "$filtered" "$cron_line" | crontab -
    else
        echo "$cron_line" | crontab -
    fi

    # Reset failure counter on fresh install
    echo 0 > "${DEPLOY_LOGS_DIR}/.pulse_failures_${SESSION_ID}"

    echo ""
    log_done "Pulse cron job installed for session: ${SESSION_ID}"
    log_info "  Schedule:  ${cron_expr} (every $(format_interval "$INTERVAL"))"
    log_info "  Log file:  ${log_file}"
    log_info "  Uninstall: ./deploy/pulse.sh --uninstall-cron --session-file ${abs_session_file}"
    echo ""
}

# ═══════════════════════════════════════════════════════════════════════════
# ── Handle --list-sessions (before loading session state) ─────────────────
# ═══════════════════════════════════════════════════════════════════════════

if [[ "$LIST_SESSIONS" == true ]]; then
    list_sessions
    exit 0
fi

# ═══════════════════════════════════════════════════════════════════════════
# ── Load session state if provided ────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

if [[ -n "$SESSION_FILE" && -f "$SESSION_FILE" ]]; then
    # shellcheck source=/dev/null
    source "$SESSION_FILE"
    # Derive session ID: prefer SESSION_ID from file, fall back to SESSION_NAME/filename
    if [[ -z "${SESSION_ID:-}" ]]; then
        SESSION_ID="${SESSION_NAME:-}"
    fi
    if [[ -z "$SESSION_ID" ]]; then
        SESSION_ID=$(derive_session_id_from_file "$SESSION_FILE")
    fi
    # Map session vars to pulse vars
    [[ -z "$POD_ID" && -n "${SESSION_POD_ID:-}" ]] && POD_ID="$SESSION_POD_ID"
    [[ -z "$SSH_TARGET" && -n "${SESSION_IP:-}" ]] && SSH_TARGET="${SESSION_USER:-root}@${SESSION_IP}"
    [[ -z "$SSH_PORT" && -n "${SESSION_PORT:-}" ]] && SSH_PORT="$SESSION_PORT"
    [[ -z "$SSH_IDENTITY" && -n "${SESSION_SSH_IDENTITY:-}" ]] && SSH_IDENTITY="$SESSION_SSH_IDENTITY"
    # Pick up API keys from session state
    [[ -z "${RUNPOD_API_KEY:-}" && -n "${SESSION_RUNPOD_API_KEY:-}" ]] && export RUNPOD_API_KEY="$SESSION_RUNPOD_API_KEY"
    [[ -z "${VAST_API_KEY:-}" && -n "${SESSION_VAST_API_KEY:-}" ]] && export VAST_API_KEY="$SESSION_VAST_API_KEY"
elif [[ -n "$SESSION_FILE" && ! -f "$SESSION_FILE" ]]; then
    # File doesn't exist — try to derive session ID from filename for uninstall
    SESSION_ID=$(derive_session_id_from_file "$SESSION_FILE")
fi

# ═══════════════════════════════════════════════════════════════════════════
# ── Handle --uninstall-cron ───────────────────────────────────────────────
# ═══════════════════════════════════════════════════════════════════════════

if [[ "$UNINSTALL_CRON" == true ]]; then
    if [[ -n "$SESSION_FILE" && -n "$SESSION_ID" ]]; then
        uninstall_cron_session "$SESSION_ID"
    elif [[ -n "$SESSION_FILE" ]]; then
        log_error "Cannot determine session ID from: ${SESSION_FILE}"
        log_info "Use --list-sessions to see active sessions, or omit --session-file to remove all."
        exit 1
    else
        uninstall_cron_all
    fi
    exit 0
fi

# Handle --install-cron after loading session (so we have all target info)
if [[ "$INSTALL_CRON" == true ]]; then
    install_cron
    exit 0
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

# Source provider API when monitoring a pod/instance by ID
if [[ -n "$POD_ID" ]]; then
    if [[ "${SESSION_PROVIDER:-}" == "vast.ai" ]]; then
        source "${DEPLOY_DIR}/vastai/api.sh"
        if [[ -z "${VAST_API_KEY:-}" ]]; then
            log_error "VAST_API_KEY is required to monitor a Vast.ai instance."
            log_info "Set it with: export VAST_API_KEY=your_api_key_here"
            exit 1
        fi
    else
        source "${DEPLOY_DIR}/runpod/api.sh"
        if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
            log_error "RUNPOD_API_KEY is required to monitor a RunPod pod."
            log_info "Set it with: export RUNPOD_API_KEY=your_api_key_here"
            exit 1
        fi
    fi
fi

# ── Lock file (prevent concurrent runs for the SAME session) ────────────────
mkdir -p "$DEPLOY_LOGS_DIR"
if [[ -n "$SESSION_ID" ]]; then
    LOCK_FILE="${DEPLOY_LOGS_DIR}/.pulse.lock_${SESSION_ID}"
else
    LOCK_FILE="${DEPLOY_LOGS_DIR}/.pulse.lock"
fi
exec 200>"$LOCK_FILE"
if ! flock -n 200; then
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] Skipped — another pulse instance is running for session ${SESSION_ID:-default}."
    exit 0
fi

# ── Disable set -e for the check and restart phases ─────────────────────────
# The health check MUST be resilient to transient errors (network blips,
# API timeouts, etc.).  We handle errors explicitly from here on.
set +e

# ── Check functions ─────────────────────────────────────────────────────────

# Check RunPod pod status via the REST API.
# Returns a status string.  On transient API errors, returns "api_error"
# instead of crashing the whole script.
check_runpod() {
    local pod_id="$1"
    local info status

    info=$(runpod_get_pod "$pod_id" 2>/dev/null)
    _runpod_sync_http_code

    if [[ "$RUNPOD_API_HTTP_CODE" != "200" ]]; then
        echo "api_error"
        return 0
    fi

    # ── Check actual runtime status first ────────────────────────────────
    # desiredStatus is what was *requested* (e.g. RUNNING), but if the
    # container has crashed, the runtime object reflects the real state.
    local runtime_status
    runtime_status=$(echo "$info" | jq -r '(.runtime // {}).status // empty' 2>/dev/null)

    if [[ -n "$runtime_status" ]]; then
        echo "$runtime_status" | tr '[:upper:]' '[:lower:]'
        return 0
    fi

    # Fall back to desiredStatus
    status=$(echo "$info" | jq -r '.desiredStatus // empty' 2>/dev/null)
    if [[ -n "$status" ]]; then
        echo "$status" | tr '[:upper:]' '[:lower:]'
        return 0
    fi

    echo "unknown"
}

# Check Vast.ai instance status via the REST API.
# Returns a status string. On transient API errors, returns "api_error".
check_vastai() {
    local instance_id="$1"
    local status
    status=$(vastai_get_instance_status "$instance_id" 2>/dev/null)
    vastai_sync_http_code
    if [[ "$VAST_API_HTTP_CODE" != "200" ]]; then
        echo "api_error"
        return 0
    fi
    echo "$status"
}

# Check SSH reachability.  Returns "reachable" or "unreachable".
# Retries up to 3 times with 5s interval to avoid false "unreachable" on transient network blips.
# Never crashes on error — always returns a status string.
check_ssh_target() {
    local target="$1"
    local attempt=1
    local max_attempts=3
    local interval=5

    while [[ $attempt -le $max_attempts ]]; do
        if ssh "${_PULSE_SSH_OPTS[@]}" -o ConnectTimeout=10 -o BatchMode=yes \
               -o StrictHostKeyChecking=no "$target" "echo ok" &>/dev/null; then
            echo "reachable"
            return 0
        fi
        [[ $attempt -lt $max_attempts ]] && sleep "$interval"
        attempt=$((attempt + 1))
    done
    echo "unreachable"
}

# ── Auto-restart pipeline ──────────────────────────────────────────────────
# Sources all pipeline scripts and re-runs the full setup + launch.
# Runs with set +e so individual step failures don't kill pulse.
do_auto_restart() {
    log_warn "Auto-restart triggered — re-provisioning and re-deploying …"

    # Load session state
    if [[ ! -f "$SESSION_FILE" ]]; then
        log_error "Session file not found: $SESSION_FILE"
        return 1
    fi

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
    RECOVERY_POLICY="${SESSION_RECOVERY_POLICY:-restart}"

    # Restore API keys that cron's minimal environment doesn't have
    [[ -n "${SESSION_WANDB_API_KEY:-}" ]] && export WANDB_API_KEY="$SESSION_WANDB_API_KEY"
    [[ -n "${SESSION_RUNPOD_API_KEY:-}" ]] && export RUNPOD_API_KEY="$SESSION_RUNPOD_API_KEY"
    [[ -n "${SESSION_VAST_API_KEY:-}" ]] && export VAST_API_KEY="$SESSION_VAST_API_KEY"

    # Restore registry venv settings so sync_venv uses the same path as the
    # original run.sh invocation (symlink from /app/.venv vs Docker streaming).
    GITHUB_USERNAME="${SESSION_GITHUB_USERNAME:-}"
    GHRC_IMAGE_NAME="${SESSION_GHRC_IMAGE_NAME:-loom-env}"
    GHRC_TAG="${SESSION_GHRC_TAG:-latest}"
    if [[ -n "$GITHUB_USERNAME" ]]; then
        GHRC_IMAGE="ghcr.io/${GITHUB_USERNAME}/${GHRC_IMAGE_NAME}:${GHRC_TAG}"
    else
        GHRC_IMAGE=""
    fi
    USE_REGISTRY_VENV=false
    [[ -n "$PROVIDER" && -n "$GITHUB_USERNAME" ]] && USE_REGISTRY_VENV=true

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
    if ! build_venv; then
        log_error "Failed to build venv. Will retry on next pulse."
        return 1
    fi

    # Step 3: Transfer venv
    if ! sync_venv; then
        log_error "Failed to sync venv. Will retry on next pulse."
        return 1
    fi

    # Step 4: Sync code
    if ! sync_code; then
        log_error "Failed to sync code. Will retry on next pulse."
        return 1
    fi

    # Step 5: Setup credentials
    if ! setup_creds; then
        log_error "Failed to setup credentials. Will retry on next pulse."
        return 1
    fi

    # Step 6: Setup uv
    if ! setup_uv; then
        log_error "Failed to setup uv. Will retry on next pulse."
        return 1
    fi

    # Step 7: Launch training in tmux
    if ! run_remote; then
        log_error "Failed to launch remote session. Will retry on next pulse."
        return 1
    fi

    # Step 8: Save updated session state (preserving session-specific file path)
    if [[ -n "$SESSION_ID" ]]; then
        set_session_state_file "$SESSION_ID"
    fi
    save_session_state || true

    # Update pulse monitoring targets with new connection info
    # (POD_ID may have been updated by _runpod_create / _vastai_create if RECOVERY_POLICY="new")
    SSH_TARGET="${REMOTE_USER}@${REMOTE_IP}"
    SSH_PORT="$REMOTE_PORT"
    _PULSE_SSH_OPTS=()
    [[ -n "$SSH_PORT" ]] && _PULSE_SSH_OPTS+=(-p "$SSH_PORT")
    [[ -n "$SSH_IDENTITY" ]] && _PULSE_SSH_OPTS+=(-i "$SSH_IDENTITY")

    log_done "Auto-restart complete."
    return 0
}

# ── Failure tracking (session-specific) ──────────────────────────────────────
if [[ -n "$SESSION_ID" ]]; then
    FAILURE_COUNT_FILE="${DEPLOY_LOGS_DIR}/.pulse_failures_${SESSION_ID}"
else
    FAILURE_COUNT_FILE="${DEPLOY_LOGS_DIR}/.pulse_failures"
fi
CONSECUTIVE_FAILURES=0
[[ -f "$FAILURE_COUNT_FILE" ]] && CONSECUTIVE_FAILURES=$(cat "$FAILURE_COUNT_FILE" 2>/dev/null || echo 0)

# ── GPU idle tracking (session-specific) ──────────────────────────────────────
if [[ -n "$SESSION_ID" ]]; then
    GPU_IDLE_COUNT_FILE="${DEPLOY_LOGS_DIR}/.pulse_low_gpu_${SESSION_ID}"
else
    GPU_IDLE_COUNT_FILE="${DEPLOY_LOGS_DIR}/.pulse_low_gpu"
fi
GPU_IDLE_STRIKES=0
[[ -f "$GPU_IDLE_COUNT_FILE" ]] && GPU_IDLE_STRIKES=$(cat "$GPU_IDLE_COUNT_FILE" 2>/dev/null || echo 0)

# Query GPU activity on a remote host via SSH.
# Returns an integer (0–100) on stdout, or -1 if nvidia-smi is unavailable
# or the SSH connection fails.  Never crashes.
#
# Strategy (most-to-least reliable):
#   1. Active compute-process check — if any PID is attached to the GPU,
#      training is running regardless of what utilisation.gpu reads.
#      Returns 100 immediately (avoids sampling noise entirely).
#   2. Multi-sample peak utilisation — take N rapid samples and return
#      the MAX.  A single nvidia-smi reading is an instantaneous snapshot
#      that lands at 0 between CUDA kernel launches even under heavy load.
GPU_IDLE_SAMPLES=5        # number of utilisation samples to take
GPU_IDLE_SAMPLE_INTERVAL=2  # seconds between samples
check_gpu_usage() {
    local target="$1"
    local ssh_cmd="ssh ${_PULSE_SSH_OPTS[*]+"${_PULSE_SSH_OPTS[*]}"} \
        -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no \
        $target"

    # ── 1. Check for active compute processes ────────────────────────────
    # If any PID is using the GPU, training is running — no need to sample.
    local compute_procs
    compute_procs=$(${ssh_cmd} \
        "nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null \
         | grep -c '^[0-9]' || echo 0" 2>/dev/null)

    if [[ $? -ne 0 ]]; then
        # SSH itself failed — nvidia-smi may not exist or host is unreachable
        echo -1
        return 0
    fi

    compute_procs=$(echo "$compute_procs" | tr -d '[:space:]')
    if [[ "$compute_procs" =~ ^[0-9]+$ && "$compute_procs" -gt 0 ]]; then
        echo 100
        return 0
    fi

    # ── 2. Multi-sample peak utilisation ─────────────────────────────────
    # No compute processes found; sample utilisation several times and take
    # the peak to catch any burst activity the process list may have missed.
    local sample_cmd="for i in \$(seq ${GPU_IDLE_SAMPLES}); do \
        nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits 2>/dev/null; \
        sleep ${GPU_IDLE_SAMPLE_INTERVAL}; \
    done"

    local output
    output=$(${ssh_cmd} "$sample_cmd" 2>/dev/null)

    if [[ -z "$output" ]]; then
        echo -1
        return 0
    fi

    local max_util=0 util
    while IFS= read -r line; do
        util=$(echo "$line" | tr -d '[:space:]')
        [[ "$util" =~ ^[0-9]+$ ]] || continue
        [[ $util -gt $max_util ]] && max_util=$util
    done <<< "$output"

    echo "$max_util"
}

# Stop the monitored machine and remove the cron job for this session.
# For RunPod pods: calls the stop API.
# For SSH-only targets: issues a remote shutdown.
do_idle_stop() {
    local now
    now=$(date '+%Y-%m-%d %H:%M:%S')

    echo "[${now}] GPU has been idle (<${GPU_IDLE_THRESHOLD}%) for ${GPU_IDLE_MAX_STRIKES} consecutive checks."
    log_warn "Idle-stop triggered — stopping machine and removing cron job."

    if [[ -n "$POD_ID" ]]; then
        local result
        if [[ "${SESSION_PROVIDER:-}" == "vast.ai" ]]; then
            result=$(vastai_stop_instance "$POD_ID" 2>/dev/null)
            vastai_sync_http_code
            if [[ "$VAST_API_HTTP_CODE" == "200" ]]; then
                echo "[${now}] Instance ${POD_ID} stopped successfully."
            else
                echo "[${now}] Warning: stop API returned HTTP ${VAST_API_HTTP_CODE} — instance may already be stopped."
            fi
        else
            result=$(runpod_stop_pod "$POD_ID" 2>/dev/null)
            _runpod_sync_http_code
            if [[ "$RUNPOD_API_HTTP_CODE" == "200" ]]; then
                echo "[${now}] Pod ${POD_ID} stopped successfully."
            else
                echo "[${now}] Warning: stop API returned HTTP ${RUNPOD_API_HTTP_CODE} — pod may already be stopped."
            fi
        fi
    elif [[ -n "$SSH_TARGET" ]]; then
        ssh "${_PULSE_SSH_OPTS[@]}" \
            -o ConnectTimeout=10 -o BatchMode=yes -o StrictHostKeyChecking=no \
            "$SSH_TARGET" "sudo shutdown -h now" &>/dev/null || true
        echo "[${now}] Shutdown command sent to ${SSH_TARGET}."
    fi

    if [[ -n "$SESSION_ID" ]]; then
        uninstall_cron_session "$SESSION_ID"
    else
        uninstall_cron_all
    fi
}

# ── Determine target label ──────────────────────────────────────────────────
TARGET_LABEL=""
[[ -n "$POD_ID" ]] && TARGET_LABEL="pod:${POD_ID}"
[[ -z "$TARGET_LABEL" && -n "$SSH_TARGET" ]] && TARGET_LABEL="ssh:${SSH_TARGET}"
[[ -n "$SESSION_ID" ]] && TARGET_LABEL="${TARGET_LABEL} [${SESSION_ID}]"

# ── Perform single health check ────────────────────────────────────────────
    NOW=$(date '+%Y-%m-%d %H:%M:%S')

    STATUS=""
    if [[ -n "$POD_ID" ]]; then
        if [[ "${SESSION_PROVIDER:-}" == "vast.ai" ]]; then
            STATUS=$(check_vastai "$POD_ID" 2>/dev/null) || STATUS="check_error"
        else
            STATUS=$(check_runpod "$POD_ID" 2>/dev/null) || STATUS="check_error"
        fi
        # Fallback: when API returns unknown or api_error, check SSH to saved IP.
        # If the instance is still reachable, treat as up (avoids false restarts).
        if [[ -n "$SSH_TARGET" && ( "$STATUS" == "unknown" || "$STATUS" == "api_error" ) ]]; then
            ssh_reach=$(check_ssh_target "$SSH_TARGET" 2>/dev/null)
            if [[ "$ssh_reach" == "reachable" ]]; then
                echo "[$(date '+%Y-%m-%d %H:%M:%S')] API status=${STATUS}; SSH fallback: reachable"
                STATUS="reachable"
            fi
        fi
    else
        STATUS=$(check_ssh_target "$SSH_TARGET" 2>/dev/null) || STATUS="check_error"
    fi
    [[ -z "$STATUS" ]] && STATUS="check_error"

# Status icon for log readability
    case "$STATUS" in
    running|reachable)      STATUS_ICON="✔" ;;
    starting|created)       STATUS_ICON="…" ;;
    api_error|check_error)  STATUS_ICON="?" ;;
    *)                      STATUS_ICON="✘" ;;
    esac

echo "[${NOW}] ${STATUS_ICON} ${TARGET_LABEL} status=${STATUS}"

# ── Handle status ───────────────────────────────────────────────────────────
        case "$STATUS" in
    running|reachable)
        # All good — reset failure counter
        echo 0 > "$FAILURE_COUNT_FILE"

        # ── GPU idle check (only when --idle-stop is enabled) ────────────
        if [[ "$IDLE_STOP" == true && -n "$SSH_TARGET" ]]; then
            GPU_UTIL=$(check_gpu_usage "$SSH_TARGET" 2>/dev/null)
            if [[ "$GPU_UTIL" -lt 0 ]]; then
                echo "[${NOW}] GPU check skipped (nvidia-smi unavailable or SSH error) — idle strikes reset"
                echo 0 > "$GPU_IDLE_COUNT_FILE"
            elif [[ "$GPU_UTIL" -lt "$GPU_IDLE_THRESHOLD" ]]; then
                GPU_IDLE_STRIKES=$((GPU_IDLE_STRIKES + 1))
                echo "$GPU_IDLE_STRIKES" > "$GPU_IDLE_COUNT_FILE"
                echo "[${NOW}] GPU idle — peak_util=${GPU_UTIL}% over ${GPU_IDLE_SAMPLES}x${GPU_IDLE_SAMPLE_INTERVAL}s samples, no compute processes (strike ${GPU_IDLE_STRIKES}/${GPU_IDLE_MAX_STRIKES})"
                if [[ $GPU_IDLE_STRIKES -ge $GPU_IDLE_MAX_STRIKES ]]; then
                    do_idle_stop
                fi
            else
                echo "[${NOW}] GPU active — peak_util=${GPU_UTIL}% (idle strikes reset)"
                echo 0 > "$GPU_IDLE_COUNT_FILE"
            fi
        fi
        ;;
            # unknown = instance deleted/gone (e.g. Vast.ai returns 200 but no status fields)
            stopped|exited|unreachable|error|unknown)
                CONSECUTIVE_FAILURES=$((CONSECUTIVE_FAILURES + 1))
        echo "$CONSECUTIVE_FAILURES" > "$FAILURE_COUNT_FILE"

        if [[ "$AUTO_RESTART" == true ]]; then
                if [[ $CONSECUTIVE_FAILURES -ge $MAX_RETRIES ]]; then
                echo "[${NOW}] Giving up after ${MAX_RETRIES} consecutive failures. Removing cron job."
                log_error "Auto-restart failed ${MAX_RETRIES} times consecutively. Removing cron job — manual intervention needed."
                log_info "Re-install with: ./deploy/pulse.sh --install-cron --session-file <FILE> --interval <INTERVAL> --auto-restart"
                if [[ -n "$SESSION_ID" ]]; then
                    uninstall_cron_session "$SESSION_ID"
                else
                    uninstall_cron_all
                fi
                else
                echo "[${NOW}] AUTO-RESTART triggered (attempt ${CONSECUTIVE_FAILURES}/${MAX_RETRIES})"
                    if do_auto_restart; then
                    echo "[${NOW}] AUTO-RESTART succeeded"
                    echo 0 > "$FAILURE_COUNT_FILE"
                    echo 0 > "$GPU_IDLE_COUNT_FILE"
                    else
                    echo "[${NOW}] AUTO-RESTART failed (attempt ${CONSECUTIVE_FAILURES}/${MAX_RETRIES}) — will retry next check"
                fi
                    fi
                fi
                ;;
            api_error|check_error)
        # Transient error — don't count as a failure, don't restart
        echo "[${NOW}] Transient error (${STATUS}) — will retry next check"
                ;;
        esac
