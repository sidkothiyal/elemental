#!/usr/bin/env bash
#
# ensure_machine.sh — Ensure a remote machine is running and reachable.
#
# Provider-aware: can create/start machines on RunPod (and vast.ai in future).
# When given an IP+port, checks SSH reachability first. If the machine is down
# and a provider is specified, attempts to restart it. If no IP is given,
# creates or starts a machine via the provider.
#
# RunPod operations use the REST API (curl + jq) — no runpodctl needed.
#
# Expects (env):
#   REMOTE_IP       (optional) — IP/hostname of existing machine
#   REMOTE_PORT     (optional) — SSH port, default 22
#   REMOTE_USER     (optional) — SSH user, default root
#   SSH_IDENTITY    (optional) — SSH identity file
#   PROVIDER        (optional) — "runpod" or "vast.ai"
#   POD_ID          (optional) — existing pod/instance ID for restart
#   POD_CONFIG      (optional) — path to pod config file; if unset, fallbacks from
#                   deploy/runpod/pod_configs.list are tried in order on create failure
#   POD_FALLBACK_PAUSE_SECONDS (optional) — seconds to wait after a create failure
#                   before trying the next config (default: 15)
#   RUNPOD_API_KEY  (required for provider=runpod)
#
# Sets (globals): REMOTE_IP, REMOTE_PORT, POD_ID (may be updated)
#
# Usage:
#   source deploy/common.sh && source deploy/ensure_machine.sh
#   ensure_machine
#

[[ -z "${DEPLOY_DIR:-}" ]] && source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# ── RunPod helpers ──────────────────────────────────────────────────────────

_runpod_require_api() {
    source "${DEPLOY_DIR}/runpod/api.sh"
    require_runpod_api || {
        log_error "RunPod REST API prerequisites not met (need curl, jq, RUNPOD_API_KEY)."
        return 1
    }
}

# Start an existing RunPod pod.
# Args: $1 = pod_id
_runpod_start() {
    local pod_id="$1"
    log_step "Starting RunPod pod ${pod_id} …"

    local output
    output=$(runpod_start_pod "$pod_id")
    _runpod_sync_http_code
    if [[ "$RUNPOD_API_HTTP_CODE" != "200" && "$RUNPOD_API_HTTP_CODE" != "201" ]]; then
        log_error "Failed to start pod (HTTP ${RUNPOD_API_HTTP_CODE}): $(echo "$output" | jq -r '.message // .error // empty' 2>/dev/null || echo "$output")"
        return 1
    fi
    echo "$output" | jq . 2>/dev/null || echo "$output"
    runpod_wait_ready "$pod_id"
    runpod_wait_ssh_info "$pod_id"
}

# Create a new RunPod pod from config.
# Uses POD_CONFIG or default deploy/runpod/pod_a40.conf
# When POD_CONFIG is unset, reads deploy/runpod/pod_configs.list and tries each
# config in order until one succeeds (fallback on create failure).
_runpod_create() {
    local config_file="${POD_CONFIG:-${DEPLOY_DIR}/runpod/pod_a40.conf}"
    if [[ ! -f "$config_file" ]]; then
        log_error "Pod config not found: $config_file"
        return 1
    fi

    log_info "Loading pod config: $config_file"
    # shellcheck source=runpod/pod.conf
    source "$config_file"

    # Validate
    [[ -z "${POD_NAME:-}" ]] && { log_error "POD_NAME is required in config."; return 1; }
    [[ -z "${GPU_TYPE:-}" ]] && { log_error "GPU_TYPE is required in config."; return 1; }

    # Build JSON payload
    local payload
    payload=$(build_pod_create_json)

    log_step "Creating RunPod pod …"
    echo -e "  ${BLUE}Payload:${NC}"
    echo "$payload" | jq . 2>/dev/null
    echo ""

    local output
    output=$(runpod_create_pod "$payload")
    _runpod_sync_http_code

    if [[ "$RUNPOD_API_HTTP_CODE" != "200" && "$RUNPOD_API_HTTP_CODE" != "201" ]]; then
        log_error "Failed to create pod (HTTP ${RUNPOD_API_HTTP_CODE}): $(echo "$output" | jq -r '.message // .error // empty' 2>/dev/null || echo "$output")"
        return 1
    fi

    # Extract pod ID from response
    POD_ID=$(echo "$output" | jq -r '.id // empty' 2>/dev/null)
    if [[ -z "$POD_ID" ]]; then
        log_warn "Could not parse pod ID from response. Check output above."
        echo "$output" | jq . 2>/dev/null || echo "$output"
        return 1
    fi

    log_done "Pod created: ${POD_ID}"
    runpod_wait_ready "$POD_ID"
    runpod_wait_ssh_info "$POD_ID"
}

# Try to create a pod using primary config, then fallbacks from pod_configs.list.
# When POD_CONFIG is set (e.g. --pod-config), uses only that config.
# When unset, reads deploy/runpod/pod_configs.list and tries each line until one succeeds.
_runpod_create_with_fallbacks() {
    local runpod_dir="${DEPLOY_DIR}/runpod"
    local list_file="${runpod_dir}/pod_configs.list"

    if [[ -n "${POD_CONFIG:-}" ]]; then
        _runpod_create
        return $?
    fi

    if [[ ! -f "$list_file" ]]; then
        POD_CONFIG="${runpod_dir}/pod_a40.conf"
        _runpod_create
        return $?
    fi

    local tried=()
    while IFS= read -r line || [[ -n "$line" ]]; do
        line=$(echo "$line" | sed 's/#.*//; s/^[[:space:]]*//; s/[[:space:]]*$//')
        [[ -z "$line" ]] && continue
        local config_file="${runpod_dir}/${line}"
        if [[ ! -f "$config_file" ]]; then
            log_warn "Pod config from list not found: $config_file (skipping)"
            continue
        fi
        POD_CONFIG="$config_file"
        tried+=("$config_file")
        if _runpod_create; then
            return 0
        fi
        log_warn "Pod creation failed with config $config_file, trying next fallback …"
        local pause_sec="${POD_FALLBACK_PAUSE_SECONDS:-15}"
        if [[ $pause_sec -gt 0 ]]; then
            log_step "Waiting ${pause_sec}s before trying next config …"
            sleep "$pause_sec"
        fi
    done < "$list_file"

    log_error "All pod configs failed (tried: ${tried[*]})."
    return 1
}

# ── SSH readiness check ────────────────────────────────────────────────────
# After a pod is created/restarted and IP/port are known, the network path may
# not be routable yet ("Network is unreachable") and sshd may need extra time.
# This helper optionally waits an initial delay, then polls until SSH succeeds.
# Args: $1 = max_wait (default 180s), $2 = initial_delay (default 15s, 0 to skip)
_wait_for_ssh() {
    local max_wait="${1:-180}"
    local initial_delay="${2:-15}"
    local elapsed=0
    local interval=10

    if [[ "$initial_delay" -gt 0 ]]; then
        log_step "Waiting ${initial_delay}s for pod network to become routable …"
        sleep "$initial_delay"
        elapsed=$initial_delay
    fi

    log_step "Verifying SSH connectivity to ${REMOTE_HOST} …"
    while [[ $elapsed -lt $max_wait ]]; do
        if ssh_check; then
            log_done "SSH connection verified."
            return 0
        fi
        echo -e "  Waiting for SSH … (${elapsed}s / ${max_wait}s)"
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    log_warn "SSH not reachable after ${max_wait}s."
    return 1
}

# ── Main function ───────────────────────────────────────────────────────────

ensure_machine() {
    # Load pod config to pick up RECOVERY_POLICY (and other settings)
    if [[ "${PROVIDER:-}" == "runpod" && -z "${RECOVERY_POLICY:-}" ]]; then
        local _conf="${POD_CONFIG:-${DEPLOY_DIR}/runpod/pod_a40.conf}"
        [[ -f "$_conf" ]] && source "$_conf"
    fi

    # Case 1: IP + port provided → check if reachable (with brief retries for transient unreachable)
    if [[ -n "${REMOTE_IP:-}" ]]; then
        REMOTE_PORT="${REMOTE_PORT:-22}"
        build_ssh_opts
        ssh_remove_from_known_hosts

        log_step "Checking if ${REMOTE_IP}:${REMOTE_PORT} is reachable …"
        local check_attempt=1
        local check_max=3
        local check_interval=10
        while [[ $check_attempt -le $check_max ]]; do
            if ssh_check; then
                log_done "Machine is reachable at ${REMOTE_IP}:${REMOTE_PORT}."
                return 0
            fi
            [[ $check_attempt -lt $check_max ]] && { echo -e "  Retrying in ${check_interval}s … (${check_attempt}/${check_max})"; sleep "$check_interval"; }
            check_attempt=$((check_attempt + 1))
        done

        log_warn "Machine at ${REMOTE_IP}:${REMOTE_PORT} is unreachable after ${check_max} attempts."

        # Try to recover via provider if available
        if [[ "${PROVIDER:-}" == "runpod" && -n "${POD_ID:-}" ]]; then
            _runpod_require_api || return 1

            if [[ "${RECOVERY_POLICY:-restart}" == "new" ]]; then
                log_info "Recovery policy is 'new' — creating a fresh pod …"
                # Best-effort stop of the old pod (avoid charges on stale pods)
                log_step "Stopping old pod ${POD_ID} …"
                runpod_stop_pod "$POD_ID" 2>/dev/null || true
                # Clear stale connection info so _runpod_create gets fresh values
                REMOTE_IP=""
                _runpod_create_with_fallbacks || return 1
            else
                log_info "Attempting to restart existing pod via RunPod …"
            _runpod_start "$POD_ID" || return 1
            fi

            build_ssh_opts  # rebuild with new IP/port
            ssh_remove_from_known_hosts
            _wait_for_ssh || return 1
            return 0
        elif [[ "${PROVIDER:-}" == "vast.ai" ]]; then
            log_error "vast.ai auto-restart is not yet implemented."
            log_info "Please restart the instance manually and re-run with the new IP/port."
            return 1
        else
            log_error "Machine is unreachable and no provider configured for auto-restart."
            [[ -z "${PROVIDER:-}" ]] && log_info "Hint: use --provider runpod --pod-id <ID> to enable auto-restart."
            return 1
        fi
    fi

    # Case 2: No IP — use provider to create/start a machine
    if [[ -n "${PROVIDER:-}" ]]; then
        case "$PROVIDER" in
            runpod)
                _runpod_require_api || return 1
                if [[ -n "${POD_ID:-}" && "${RECOVERY_POLICY:-restart}" != "new" ]]; then
                    # Start existing pod
                    _runpod_start "$POD_ID" || return 1
                else
                    # Create new pod (tries fallbacks from pod_configs.list when POD_CONFIG unset)
                    _runpod_create_with_fallbacks || return 1
                fi
                build_ssh_opts  # rebuild with new IP/port
                ssh_remove_from_known_hosts
                _wait_for_ssh || return 1
                return 0
                ;;
            vast.ai)
                log_error "vast.ai provider is not yet implemented."
                log_info "Once implemented, this will create/start vast.ai instances automatically."
                log_info "For now, create an instance manually and pass --ip and --port."
                return 1
                ;;
            *)
                log_error "Unknown provider: $PROVIDER (supported: runpod, vast.ai)"
                return 1
                ;;
        esac
    fi

    # Case 3: Neither IP nor provider
    log_error "No remote IP or provider specified. Cannot ensure machine."
    log_info "Use --ip HOST --port PORT for an existing machine,"
    log_info "or --provider runpod to create/start one automatically."
    return 1
}

# ── Standalone mode ─────────────────────────────────────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

    # Simple arg parsing for standalone use
    while [[ $# -gt 0 ]]; do
        case "$1" in
            --ip)       REMOTE_IP="$2"; shift 2 ;;
            --port|-p)  REMOTE_PORT="$2"; shift 2 ;;
            --user)     REMOTE_USER="$2"; shift 2 ;;
            -i)         SSH_IDENTITY="$2"; shift 2 ;;
            --provider) PROVIDER="$2"; shift 2 ;;
            --pod-id)   POD_ID="$2"; shift 2 ;;
            --pod-config) POD_CONFIG="$2"; shift 2 ;;
            --help|-h)
                echo "Usage: $0 [--ip HOST] [--port PORT] [--provider NAME] [--pod-id ID] [--pod-config FILE]"
                exit 0 ;;
            *) log_error "Unknown option: $1"; exit 1 ;;
        esac
    done

    ensure_machine
    echo ""
    echo "REMOTE_IP=${REMOTE_IP:-}"
    echo "REMOTE_PORT=${REMOTE_PORT:-}"
    echo "POD_ID=${POD_ID:-}"
fi
