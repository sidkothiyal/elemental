#!/usr/bin/env bash
#
# api.sh — Vast.ai REST API helpers using curl + jq.
#
# Provides functions to create, start, stop, and query Vast.ai instances
# via the REST API (https://console.vast.ai/api/v0).
#
# Endpoints used: GET/PUT/DELETE /instances/{id}/, PUT /asks/{id}/, POST /bundles/
# Reference: https://docs.vast.ai/api-reference/
#
# Requires: curl, jq, VAST_API_KEY environment variable
#
# Usage:
#   source deploy/vastai/api.sh
#   require_vastai_api
#   vastai_get_instance 12345
#

VAST_API_BASE="https://console.vast.ai/api/v0"

# ── HTTP code persistence across subshells ───────────────────────────────────
VAST_API_HTTP_CODE=""
_VAST_HTTP_CODE_FILE="${TMPDIR:-/tmp}/.vast_http_code.$$"
trap 'rm -f "$_VAST_HTTP_CODE_FILE"' EXIT

vastai_sync_http_code() {
    VAST_API_HTTP_CODE=$(cat "$_VAST_HTTP_CODE_FILE" 2>/dev/null || echo "000")
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
require_vastai_api() {
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

    if [[ -z "${VAST_API_KEY:-}" ]]; then
        log_error "VAST_API_KEY environment variable is not set."
        echo ""
        echo -e "${YELLOW}Set your API key:${NC}"
        echo "  export VAST_API_KEY=your_api_key_here"
        echo ""
        echo "  Get your API key at: https://cloud.vast.ai/account/"
        ok=false
    fi

    [[ "$ok" == false ]] && return 1

    log_done "Vast.ai API ready (curl + jq + API key)."
    return 0
}

# ════════════════════════════════════════════════════════════════════════════
# ── Generic API call ───────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
# Usage: vastai_api METHOD PATH [JSON_BODY]
# Returns: JSON response body on stdout
# Sets:    VAST_API_HTTP_CODE (global)
vastai_api() {
    local method="$1"
    local path="$2"
    local body="${3:-}"
    local url="${VAST_API_BASE}${path}"

    local curl_args=(
        -s
        -w '\n%{http_code}'
        -X "$method"
        -H "Authorization: Bearer ${VAST_API_KEY}"
        -H "Content-Type: application/json"
    )
    [[ -n "$body" ]] && curl_args+=(-d "$body")

    local response
    response=$(curl "${curl_args[@]}" "$url")

    VAST_API_HTTP_CODE=$(echo "$response" | tail -1)
    echo "$VAST_API_HTTP_CODE" > "$_VAST_HTTP_CODE_FILE"
    echo "$response" | sed '$d'
}

# ════════════════════════════════════════════════════════════════════════════
# ── Instance queries ────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

# Get full instance JSON. API returns { "instances": { ... } }.
# Usage: vastai_get_instance INSTANCE_ID
# Returns: instance object as JSON on stdout (extracted from .instances)
vastai_get_instance() {
    local instance_id="$1"
    local response
    response=$(vastai_api GET "/instances/${instance_id}/")
    # Response may be { "instances": <object> } — extract the instance
    if echo "$response" | jq -e '.instances' &>/dev/null; then
        echo "$response" | jq '.instances'
    else
        echo "$response"
    fi
}

# Get instance status string (running, stopped, etc.).
# Usage: vastai_get_instance_status INSTANCE_ID
vastai_get_instance_status() {
    local instance_id="$1"
    local info
    info=$(vastai_get_instance "$instance_id")
    vastai_sync_http_code

    if [[ "$VAST_API_HTTP_CODE" != "200" ]]; then
        echo "error"
        return
    fi

    local status
    status=$(echo "$info" | jq -r '.cur_state // .actual_status // .intended_status // empty' 2>/dev/null)
    if [[ -n "$status" ]]; then
        echo "$status" | tr '[:upper:]' '[:lower:]'
        return
    fi
    echo "unknown"
}

# Extract SSH connection info (ssh_host, ssh_port) from instance JSON.
# Usage: vastai_extract_ssh_info "$instance_json"
# Sets: REMOTE_IP, REMOTE_PORT (globals)
# Returns 0 on success, 1 if not yet available.
vastai_extract_ssh_info() {
    local instance_json="$1"
    local host port
    host=$(echo "$instance_json" | jq -r '.ssh_host // empty' 2>/dev/null)
    port=$(echo "$instance_json" | jq -r '.ssh_port // empty' 2>/dev/null)

    if [[ -n "$host" && -n "$port" && "$host" != "null" && "$port" != "null" ]]; then
        REMOTE_IP="$host"
        REMOTE_PORT="$port"
        return 0
    fi
    return 1
}

# ════════════════════════════════════════════════════════════════════════════
# ── Search offers ──────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
# Usage: vastai_search_offers JSON_BODY
# Returns: JSON with .offers array on stdout
vastai_search_offers() {
    local body="$1"
    vastai_api POST "/bundles/" "$body"
}

# ════════════════════════════════════════════════════════════════════════════
# ── Instance lifecycle ─────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

# Create a new instance by accepting an offer.
# Usage: vastai_create_instance OFFER_ID JSON_BODY
# Returns: full response on stdout; sets VAST_API_HTTP_CODE. Caller parses .new_contract for instance ID.
vastai_create_instance() {
    local offer_id="$1"
    local body="$2"
    vastai_api PUT "/asks/${offer_id}/" "$body"
}

# Start a stopped instance.
# Usage: vastai_start_instance INSTANCE_ID
vastai_start_instance() {
    local instance_id="$1"
    vastai_api PUT "/instances/${instance_id}/" '{"state": "running"}'
}

# Stop a running instance (preserves data).
# Usage: vastai_stop_instance INSTANCE_ID
vastai_stop_instance() {
    local instance_id="$1"
    vastai_api PUT "/instances/${instance_id}/" '{"state": "stopped"}'
}

# Permanently destroy an instance.
# Usage: vastai_destroy_instance INSTANCE_ID
vastai_destroy_instance() {
    local instance_id="$1"
    vastai_api DELETE "/instances/${instance_id}/"
}

# ════════════════════════════════════════════════════════════════════════════
# ── Polling helpers ────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

# Wait for an instance to reach running status.
# Args: $1 = instance_id, $2 = max_wait (default 300s)
vastai_wait_ready() {
    local instance_id="$1"
    local max_wait="${2:-300}"
    local elapsed=0
    local interval=10

    log_step "Waiting for instance ${instance_id} to become ready …"
    while [[ $elapsed -lt $max_wait ]]; do
        local status
        status=$(vastai_get_instance_status "$instance_id")
        if [[ "$status" == "running" ]]; then
            log_done "Instance is running."
            return 0
        fi
        echo -e "  Waiting … status=${status} (${elapsed}s / ${max_wait}s)"
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    log_warn "Instance did not become ready within ${max_wait}s."
    return 1
}

# Remove host from ~/.ssh/known_hosts for re-used hosts.
_vastai_ssh_remove_known_hosts() {
    local host="$1"
    local port="${2:-22}"
    [[ -z "$host" ]] && return 0
    local known_hosts="${HOME:-/root}/.ssh/known_hosts"
    [[ ! -f "$known_hosts" ]] && return 0
    if [[ -n "$port" && "$port" != "22" ]]; then
        ssh-keygen -R "[${host}]:${port}" -f "$known_hosts" 2>/dev/null || true
    fi
    ssh-keygen -R "$host" -f "$known_hosts" 2>/dev/null || true
}

# Poll until SSH connection info is available.
# Args: $1 = instance_id, $2 = max_wait (default 120s)
# Sets: REMOTE_IP, REMOTE_PORT
vastai_wait_ssh_info() {
    local instance_id="$1"
    local max_wait="${2:-120}"
    local elapsed=0
    local interval=10

    log_step "Retrieving SSH connection info for instance ${instance_id} …"

    while [[ $elapsed -lt $max_wait ]]; do
        local info
        info=$(vastai_get_instance "$instance_id")
        vastai_sync_http_code

        if [[ "$VAST_API_HTTP_CODE" == "200" ]] && vastai_extract_ssh_info "$info"; then
            local deploy_logs="${DEPLOY_LOGS_DIR:-./deploy/logs}"
            mkdir -p "$deploy_logs"
            local log_file="${deploy_logs}/instance_${instance_id}_$(date +%Y%m%d_%H%M%S).log"
            {
                echo "# Vast.ai Instance Info"
                echo "# Timestamp: $(date -Iseconds)"
                echo "INSTANCE_ID=${instance_id}"
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
            _vastai_ssh_remove_known_hosts "$REMOTE_IP" "${REMOTE_PORT:-22}"
            return 0
        fi

        echo -e "  Waiting for SSH info … (${elapsed}s / ${max_wait}s)"
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done

    local deploy_logs="${DEPLOY_LOGS_DIR:-./deploy/logs}"
    mkdir -p "$deploy_logs"
    local log_file="${deploy_logs}/instance_${instance_id}_$(date +%Y%m%d_%H%M%S).log"
    {
        echo "# Vast.ai Instance Info (SSH not yet available)"
        echo "# Timestamp: $(date -Iseconds)"
        echo "INSTANCE_ID=${instance_id}"
        echo ""
        echo "# Raw API response:"
        info=$(vastai_get_instance "$instance_id" 2>/dev/null || true)
        echo "$info" | jq . 2>/dev/null || echo "$info"
    } > "$log_file"

    log_warn "SSH port info not available after ${max_wait}s. Check log: $log_file"
    return 1
}

# ════════════════════════════════════════════════════════════════════════════
# ── JSON builder for instance creation ──────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════
# Build the JSON payload for PUT /asks/{id}/ from instance_*.conf shell variables.
# Expects config to have been sourced already.
# Usage: build_vastai_create_json
# Returns: JSON string on stdout.
build_vastai_create_json() {
    local json
    json=$(jq -n \
        --arg image "${IMAGE:-vastai/base-image:latest}" \
        --argjson disk "${DISK_GB:-50}" \
        --arg label "${INSTANCE_LABEL:-elemental-training}" \
        --arg runtype "${RUNTYPE:-ssh}" \
        --arg target_state "${TARGET_STATE:-running}" \
        '{
            image: $image,
            disk: $disk,
            label: $label,
            runtype: $runtype,
            target_state: $target_state
        }')

    [[ -n "${PRICE:-}" ]] && json=$(echo "$json" | jq --argjson v "$PRICE" '.price = $v')
    [[ -n "${ONSTART:-}" ]] && json=$(echo "$json" | jq --arg v "$ONSTART" '.onstart = $v')
    [[ -n "${ENV:-}" ]] && json=$(echo "$json" | jq --arg v "$ENV" '.env = $v')
    echo "$json"
}

# Build search-offers body from config (GPU_NAME, NUM_GPUS, OFFER_TYPE, etc.).
build_vastai_search_body() {
    local type="${OFFER_TYPE:-ondemand}"
    local limit="${OFFER_LIMIT:-20}"
    local body
    body=$(jq -n \
        --argjson limit "$limit" \
        --arg type "$type" \
        '{
            limit: $limit,
            type: $type,
            verified: {"eq": true},
            rentable: {"eq": true},
            rented: {"eq": false}
        }')
    [[ -n "${GPU_NAME:-}" ]] && body=$(echo "$body" | jq --arg v "$GPU_NAME" '.gpu_name = {"eq": $v}')
    [[ -n "${NUM_GPUS:-}" ]] && body=$(echo "$body" | jq --argjson v "$NUM_GPUS" '.num_gpus = {"gte": $v}')
    [[ -n "${DISK_GB:-}" ]] && body=$(echo "$body" | jq --argjson v "$DISK_GB" '.disk_space = {"gte": $v}')
    [[ -n "${RELIABILITY_MIN:-}" ]] && body=$(echo "$body" | jq --argjson v "$RELIABILITY_MIN" '.reliability = {"gte": $v}')
    [[ -n "${DPH_MAX:-}" ]] && body=$(echo "$body" | jq --argjson v "$DPH_MAX" '.dph_total = {"lte": $v}')
    echo "$body"
}

# ── Run directly if not sourced ──────────────────────────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    require_vastai_api
fi
