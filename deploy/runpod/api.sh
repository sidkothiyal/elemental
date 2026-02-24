#!/usr/bin/env bash
#
# api.sh — RunPod REST API helpers using curl + jq.
#
# Provides functions to create, start, stop, and query RunPod pods
# via the REST API (https://rest.runpod.io/v1).
#
# Requires: curl, jq, RUNPOD_API_KEY environment variable
#
# Usage:
#   source deploy/runpod/api.sh
#   require_runpod_api
#   runpod_api GET /pods
#
# Reference: https://docs.runpod.io/api-reference/pods/POST/pods
#

RUNPOD_API_BASE="https://rest.runpod.io/v1"

# ── HTTP code persistence across subshells ───────────────────────────────────
# Bash subshells (command substitution) can't propagate globals back to the
# parent. We write the HTTP status code to a temp file so callers can read it
# after capturing stdout with $(…).
RUNPOD_API_HTTP_CODE=""
_RUNPOD_HTTP_CODE_FILE="${TMPDIR:-/tmp}/.runpod_http_code.$$"
trap 'rm -f "$_RUNPOD_HTTP_CODE_FILE"' EXIT

# Read the HTTP code back from the temp file into RUNPOD_API_HTTP_CODE.
# Call this after every  output=$(runpod_xxx …)  command substitution.
_runpod_sync_http_code() {
    RUNPOD_API_HTTP_CODE=$(cat "$_RUNPOD_HTTP_CODE_FILE" 2>/dev/null || echo "000")
}

# ── Colour helpers (use existing if already defined) ─────────────────────────
RED="${RED:-\033[0;31m}"
GREEN="${GREEN:-\033[0;32m}"
YELLOW="${YELLOW:-\033[1;33m}"
BLUE="${BLUE:-\033[0;34m}"
NC="${NC:-\033[0m}"

# ── Logging helpers (use existing if already defined) ────────────────────────
type log_info    &>/dev/null || log_info()    { echo -e "${BLUE}$*${NC}"; }
type log_success &>/dev/null || log_success() { echo -e "${GREEN}$*${NC}"; }
type log_warn    &>/dev/null || log_warn()    { echo -e "${YELLOW}$*${NC}"; }
type log_error   &>/dev/null || log_error()   { echo -e "${RED}Error:${NC} $*"; }
type log_step    &>/dev/null || log_step()    { echo -e "${YELLOW}⚙ $*${NC}"; }
type log_done    &>/dev/null || log_done()    { echo -e "${GREEN}✔ $*${NC}"; }

# ── Prerequisite check ───────────────────────────────────────────────────────
require_runpod_api() {
    local ok=true

    if ! command -v curl &>/dev/null; then
        log_error "curl is not installed."
        ok=false
    fi

    if ! command -v jq &>/dev/null; then
        log_error "jq is not installed."
        echo -e "${YELLOW}Install:${NC} sudo apt-get install jq  (or brew install jq)"
        ok=false
    fi

    if [[ -z "${RUNPOD_API_KEY:-}" ]]; then
        log_error "RUNPOD_API_KEY environment variable is not set."
        echo ""
        echo -e "${YELLOW}Set your API key:${NC}"
        echo "  export RUNPOD_API_KEY=your_api_key_here"
        echo ""
        echo "  Generate an API key at: https://www.runpod.io/console/user/settings"
        ok=false
    fi

    [[ "$ok" == false ]] && return 1

    log_done "RunPod API ready (curl + jq + API key)."
    return 0
}

# ════════════════════════════════════════════════════════════════════════════
# ── Generic API call ───────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
# Usage: runpod_api METHOD /path [JSON_BODY]
# Returns: JSON response body on stdout
# Sets:    RUNPOD_API_HTTP_CODE (global)
runpod_api() {
    local method="$1"
    local path="$2"
    local body="${3:-}"
    local url="${RUNPOD_API_BASE}${path}"

    local curl_args=(
        -s
        -w '\n%{http_code}'
        -X "$method"
        -H "Authorization: Bearer ${RUNPOD_API_KEY}"
        -H "Content-Type: application/json"
    )
    [[ -n "$body" ]] && curl_args+=(-d "$body")

    local response
    response=$(curl "${curl_args[@]}" "$url")

    # Last line is the HTTP status code
    RUNPOD_API_HTTP_CODE=$(echo "$response" | tail -1)
    # Persist for parent shells to read after $(…) subshell calls
    echo "$RUNPOD_API_HTTP_CODE" > "$_RUNPOD_HTTP_CODE_FILE"
    # Everything before that is the JSON body
    echo "$response" | sed '$d'
}

# ════════════════════════════════════════════════════════════════════════════
# ── Pod queries ────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

# Get full pod JSON.
# Usage: runpod_get_pod POD_ID
runpod_get_pod() {
    local pod_id="$1"
    runpod_api GET "/pods/${pod_id}"
}

# Get pod status string (running, exited, stopped, etc.).
# Checks runtime.status first (actual container state), then falls back
# to desiredStatus (the requested state).  This distinction matters: a
# pod whose container crashed may still have desiredStatus=RUNNING while
# its runtime.status is DEAD/EXITED.
# Usage: runpod_get_pod_status POD_ID
runpod_get_pod_status() {
    local pod_id="$1"
    local info
    info=$(runpod_get_pod "$pod_id")
    _runpod_sync_http_code

    if [[ "$RUNPOD_API_HTTP_CODE" != "200" ]]; then
        echo "error"
        return
    fi

    # Prefer actual runtime status over desired status
    local runtime_status
    runtime_status=$(echo "$info" | jq -r '(.runtime // {}).status // empty' 2>/dev/null)
    if [[ -n "$runtime_status" ]]; then
        echo "$runtime_status" | tr '[:upper:]' '[:lower:]'
        return
    fi

    # Fall back to desiredStatus (the requested state)
    local status
    status=$(echo "$info" | jq -r '.desiredStatus // empty' 2>/dev/null)
    if [[ -n "$status" ]]; then
        echo "$status" | tr '[:upper:]' '[:lower:]'
        return
    fi

    echo "unknown"
}

# Extract SSH connection info (publicIp + portMappings) from a pod JSON blob.
# Usage: runpod_extract_ssh_info "$pod_json"
# Sets: REMOTE_IP, REMOTE_PORT (globals)
# Returns 0 on success, 1 if not yet available.
runpod_extract_ssh_info() {
    local pod_json="$1"

    local ip port
    ip=$(echo "$pod_json" | jq -r '.publicIp // empty' 2>/dev/null)

    # portMappings is like {"22": 10341}  — get the mapped port for internal port 22
    port=$(echo "$pod_json" | jq -r '.portMappings["22"] // empty' 2>/dev/null)

    # Fallback: grab the first port mapping if 22 not found
    if [[ -z "$port" ]]; then
        port=$(echo "$pod_json" | jq -r '(.portMappings // {}) | to_entries | .[0].value // empty' 2>/dev/null)
    fi

    if [[ -n "$ip" && -n "$port" && "$ip" != "null" && "$port" != "null" ]]; then
        REMOTE_IP="$ip"
        REMOTE_PORT="$port"
        return 0
    fi

    return 1
}

# ════════════════════════════════════════════════════════════════════════════
# ── Pod lifecycle ──────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

# Create a new pod.
# Usage: runpod_create_pod JSON_BODY
# Returns: JSON response (Pod object) on stdout.
runpod_create_pod() {
    local body="$1"
    runpod_api POST "/pods" "$body"
}

# Start / resume a stopped pod.
# Usage: runpod_start_pod POD_ID
# Note: The REST API start endpoint accepts no request body.
runpod_start_pod() {
    local pod_id="$1"
    runpod_api POST "/pods/${pod_id}/start"
}

# Stop a running pod (keeps storage).
# Usage: runpod_stop_pod POD_ID
runpod_stop_pod() {
    local pod_id="$1"
    runpod_api POST "/pods/${pod_id}/stop"
}

# Permanently remove a pod.
# Usage: runpod_remove_pod POD_ID
runpod_remove_pod() {
    local pod_id="$1"
    runpod_api DELETE "/pods/${pod_id}"
}

# ════════════════════════════════════════════════════════════════════════════
# ── Polling helpers ────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

# Wait for a pod to reach RUNNING status.
# Args: $1 = pod_id, $2 = max_wait (default 300s)
runpod_wait_ready() {
    local pod_id="$1"
    local max_wait="${2:-300}"
    local elapsed=0
    local interval=10

    log_step "Waiting for pod ${pod_id} to become ready …"
    while [[ $elapsed -lt $max_wait ]]; do
        local status
        status=$(runpod_get_pod_status "$pod_id")
        if [[ "$status" == "running" ]]; then
            log_done "Pod is running."
            return 0
        fi
        echo -e "  Waiting … status=${status} (${elapsed}s / ${max_wait}s)"
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    log_warn "Pod did not become ready within ${max_wait}s."
    return 1
}

# Poll until SSH connection info (publicIp + portMappings) is available.
# Ports can take ~60s after RUNNING to be assigned.
# Args: $1 = pod_id, $2 = max_wait (default 120s)
# Sets: REMOTE_IP, REMOTE_PORT
runpod_wait_ssh_info() {
    local pod_id="$1"
    local max_wait="${2:-120}"
    local elapsed=0
    local interval=10

    log_step "Retrieving SSH connection info for pod ${pod_id} …"

    while [[ $elapsed -lt $max_wait ]]; do
        local info
        info=$(runpod_get_pod "$pod_id")
        _runpod_sync_http_code

        if [[ "$RUNPOD_API_HTTP_CODE" == "200" ]] && runpod_extract_ssh_info "$info"; then
            # Log what we found
            local deploy_logs="${DEPLOY_LOGS_DIR:-./deploy/logs}"
            mkdir -p "$deploy_logs"
            local log_file="${deploy_logs}/pod_${pod_id}_$(date +%Y%m%d_%H%M%S).log"
            {
                echo "# RunPod Pod Info"
                echo "# Timestamp: $(date -Iseconds)"
                echo "POD_ID=${pod_id}"
                echo "SSH_HOST=${REMOTE_IP}"
                echo "SSH_PORT=${REMOTE_PORT}"
                echo ""
                echo "# Raw API response:"
                echo "$info" | jq . 2>/dev/null || echo "$info"
            } > "$log_file"

            echo ""
            echo -e "${BLUE}═══════════════════════════════════════════${NC}"
            echo -e "${BLUE}  SSH Connection Details${NC}"
            echo -e "${BLUE}═══════════════════════════════════════════${NC}"
            echo -e "  ${GREEN}Host:${NC}    $REMOTE_IP"
            echo -e "  ${GREEN}Port:${NC}    $REMOTE_PORT"
            echo -e "  ${GREEN}Connect:${NC} ssh root@${REMOTE_IP} -p ${REMOTE_PORT}"
            echo -e "${BLUE}═══════════════════════════════════════════${NC}"
            echo -e "  ${GREEN}Log:${NC} $log_file"
            return 0
        fi

        echo -e "  Waiting for port info … (${elapsed}s / ${max_wait}s)"
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done

    # Fallback: log whatever we have
    local info
    info=$(runpod_get_pod "$pod_id" 2>/dev/null || true)
    local deploy_logs="${DEPLOY_LOGS_DIR:-./deploy/logs}"
    mkdir -p "$deploy_logs"
    local log_file="${deploy_logs}/pod_${pod_id}_$(date +%Y%m%d_%H%M%S).log"
    {
        echo "# RunPod Pod Info (ports not yet available)"
        echo "# Timestamp: $(date -Iseconds)"
        echo "POD_ID=${pod_id}"
        echo ""
        echo "# Raw API response:"
        echo "$info" | jq . 2>/dev/null || echo "$info"
    } > "$log_file"

    log_warn "SSH port info not available after ${max_wait}s. Check log: $log_file"
    return 1
}

# ════════════════════════════════════════════════════════════════════════════
# ── JSON builder for pod creation ──────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
# Build the JSON payload for POST /pods from pod.conf-style shell variables.
# Expects the pod.conf to have been sourced already.
# Usage: build_pod_create_json
# Returns: JSON string on stdout.
build_pod_create_json() {
    local json
    json=$(jq -n \
        --arg name "${POD_NAME}" \
        --arg gpuType "${GPU_TYPE}" \
        --argjson gpuCount "${GPU_COUNT:-1}" \
        --arg cloudType "$(echo "${CLOUD_TIER:-community}" | tr '[:lower:]' '[:upper:]')" \
        --argjson containerDisk "${CONTAINER_DISK_SIZE:-20}" \
        '{
            name: $name,
            gpuTypeIds: [$gpuType],
            gpuCount: $gpuCount,
            cloudType: $cloudType,
            containerDiskInGb: $containerDisk
        }')

    # Spot / interruptible instance
    if [[ "${SPOT:-false}" == true ]]; then
        json=$(echo "$json" | jq '.interruptible = true')
    fi

    # Optional fields — add only when set
    [[ -n "${TEMPLATE_ID:-}" ]]        && json=$(echo "$json" | jq --arg v "$TEMPLATE_ID"        '.templateId = $v')
    [[ -n "${IMAGE_NAME:-}" ]]         && json=$(echo "$json" | jq --arg v "$IMAGE_NAME"         '.imageName = $v')
    [[ -n "${VOLUME_SIZE:-}" ]]        && json=$(echo "$json" | jq --argjson v "$VOLUME_SIZE"    '.volumeInGb = $v')
    [[ -n "${VOLUME_PATH:-}" ]]        && json=$(echo "$json" | jq --arg v "$VOLUME_PATH"        '.volumeMountPath = $v')
    [[ -n "${NETWORK_VOLUME_ID:-}" ]]  && json=$(echo "$json" | jq --arg v "$NETWORK_VOLUME_ID"  '.networkVolumeId = $v')

    # Ports — space-separated string → JSON array
    if [[ -n "${PORTS:-}" ]]; then
        local ports_json="[]"
        for p in $PORTS; do
            ports_json=$(echo "$ports_json" | jq --arg p "$p" '. + [$p]')
        done
        json=$(echo "$json" | jq --argjson p "$ports_json" '.ports = $p')
    fi

    # ── Environment variables ─────────────────────────────────────────────
    local env_json="{}"
    local has_env=false

    # SSH public keys — read files listed in SSH_PUBLIC_KEY_FILES and
    # concatenate them into a single PUBLIC_KEY env var (newline-separated).
    if [[ -n "${SSH_PUBLIC_KEY_FILES:-}" ]]; then
        local pub_key_content=""
        for key_path in ${SSH_PUBLIC_KEY_FILES}; do
            # Expand tilde
            key_path="${key_path/#\~/$HOME}"
            if [[ -f "$key_path" ]]; then
                local key_data
                key_data=$(<"$key_path")
                if [[ -n "$pub_key_content" ]]; then
                    pub_key_content="${pub_key_content}"$'\n'"${key_data}"
                else
                    pub_key_content="${key_data}"
                fi
            else
                log_warn "SSH public key file not found: $key_path (skipped)"
            fi
        done
        if [[ -n "$pub_key_content" ]]; then
            env_json=$(echo "$env_json" | jq --arg v "$pub_key_content" '.PUBLIC_KEY = $v')
            has_env=true
        fi
    fi

    # Extra env vars — semicolon-separated KEY=VALUE pairs
    if [[ -n "${ENV_VARS:-}" ]]; then
        IFS=';' read -ra _env_arr <<< "$ENV_VARS"
        for env_var in "${_env_arr[@]}"; do
            env_var="$(printf '%s' "$env_var" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
            [[ -z "$env_var" ]] && continue
            local key="${env_var%%=*}"
            local val="${env_var#*=}"
            # Interpret escape sequences (\n → newline, etc.)
            val="$(printf '%b' "$val")"
            env_json=$(echo "$env_json" | jq --arg k "$key" --arg v "$val" '.[$k] = $v')
            has_env=true
        done
    fi

    [[ "$has_env" == true ]] && json=$(echo "$json" | jq --argjson e "$env_json" '.env = $e')

    # Docker start command
    if [[ -n "${CONTAINER_ARGS:-}" ]]; then
        # Split on spaces into an array
        local cmd_json="[]"
        for arg in $CONTAINER_ARGS; do
            cmd_json=$(echo "$cmd_json" | jq --arg a "$arg" '. + [$a]')
        done
        json=$(echo "$json" | jq --argjson c "$cmd_json" '.dockerStartCmd = $c')
    fi

    echo "$json"
}

# ── Run directly if not sourced ──────────────────────────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    require_runpod_api
fi
