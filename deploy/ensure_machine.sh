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
#   POD_CONFIG      (optional) — path to pod config file
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
# Uses POD_CONFIG or default deploy/runpod/pod.conf
_runpod_create() {
    local config_file="${POD_CONFIG:-${DEPLOY_DIR}/runpod/pod.conf}"
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

# ── SSH readiness check ────────────────────────────────────────────────────
# After a pod is created/restarted and IP/port are known, sshd inside the
# container may take additional seconds to start.  This helper polls until
# an SSH connection succeeds.
# Args: $1 = max_wait (default 90s)
_wait_for_ssh() {
    local max_wait="${1:-90}"
    local elapsed=0
    local interval=5

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
        local _conf="${POD_CONFIG:-${DEPLOY_DIR}/runpod/pod.conf}"
        [[ -f "$_conf" ]] && source "$_conf"
    fi

    # Case 1: IP + port provided → check if reachable
    if [[ -n "${REMOTE_IP:-}" ]]; then
        REMOTE_PORT="${REMOTE_PORT:-22}"
        build_ssh_opts

        log_step "Checking if ${REMOTE_IP}:${REMOTE_PORT} is reachable …"
        if ssh_check; then
            log_done "Machine is reachable at ${REMOTE_IP}:${REMOTE_PORT}."
            return 0
        fi

        log_warn "Machine at ${REMOTE_IP}:${REMOTE_PORT} is unreachable."

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
                _runpod_create || return 1
            else
                log_info "Attempting to restart existing pod via RunPod …"
            _runpod_start "$POD_ID" || return 1
            fi

            build_ssh_opts  # rebuild with new IP/port
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
                    # Create new pod
                    _runpod_create || return 1
                fi
                build_ssh_opts  # rebuild with new IP/port
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
