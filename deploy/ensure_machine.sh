#!/usr/bin/env bash
#
# ensure_machine.sh — Ensure a remote machine is running and reachable.
#
# Provider-aware: can create/start machines on RunPod (and vast.ai in future).
# When given an IP+port, checks SSH reachability first. If the machine is down
# and a provider is specified, attempts to restart it. If no IP is given,
# creates or starts a machine via the provider.
#
# Expects (env):
#   REMOTE_IP       (optional) — IP/hostname of existing machine
#   REMOTE_PORT     (optional) — SSH port, default 22
#   REMOTE_USER     (optional) — SSH user, default root
#   SSH_IDENTITY    (optional) — SSH identity file
#   PROVIDER        (optional) — "runpod" or "vast.ai"
#   POD_ID          (optional) — existing pod/instance ID for restart
#   POD_CONFIG      (optional) — path to pod config file
#
# Sets (globals): REMOTE_IP, REMOTE_PORT, POD_ID (may be updated)
#
# Usage:
#   source deploy/common.sh && source deploy/ensure_machine.sh
#   ensure_machine
#

[[ -z "${DEPLOY_DIR:-}" ]] && source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

# ── RunPod helpers ──────────────────────────────────────────────────────────

_runpod_require_cli() {
    source "${DEPLOY_DIR}/runpod/check_cli.sh"
    require_runpodctl || {
        log_error "runpodctl is required for provider=runpod."
        return 1
    }
}

# Wait for a RunPod pod to reach "running" status.
# Args: $1 = pod_id, $2 = max_wait (default 300s)
_runpod_wait_ready() {
    local pod_id="$1"
    local max_wait="${2:-300}"
    local elapsed=0
    local interval=10

    log_step "Waiting for pod ${pod_id} to become ready …"
    while [[ $elapsed -lt $max_wait ]]; do
        local info
        info=$(runpodctl get pod "$pod_id" 2>&1 || true)
        if echo "$info" | grep -qi "running"; then
            log_done "Pod is running."
            return 0
        fi
        echo -e "  Waiting … (${elapsed}s / ${max_wait}s)"
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done
    log_warn "Pod did not become ready within ${max_wait}s."
    return 1
}

# Extract SSH connection info from runpodctl get pod --allfields output.
# Polls until the public TCP port mapping appears (ports take ~60s after RUNNING).
# Sets REMOTE_IP and REMOTE_PORT globals.
# Args: $1 = pod_id, $2 = max_wait (default 120s)
_runpod_get_ssh_info() {
    local pod_id="$1"
    local max_wait="${2:-120}"
    local elapsed=0
    local interval=10

    log_step "Retrieving SSH connection info for pod ${pod_id} …"

    while [[ $elapsed -lt $max_wait ]]; do
        local info pod_line pub_tcp
        info=$(runpodctl get pod --allfields 2>&1 || true)
        # Sanitize: runpodctl outputs non-breaking spaces (U+00A0, \xc2\xa0) that break regex
        info=$(printf '%s\n' "$info" | sed "s/$(printf '\xc2\xa0')/ /g")

        # Find the line for our specific pod
        pod_line=$(echo "$info" | grep "$pod_id" || true)

        if [[ -n "$pod_line" ]]; then
            # RunPod PORTS column format:
            #   100.65.18.122:60119->19123 (prv,http),194.68.245.19:22113->22222 (pub,tcp)
            # We want the (pub,tcp) entry: IP:EXTERNAL_PORT->INTERNAL_PORT (pub,tcp)
            pub_tcp=$(echo "$pod_line" | grep -oP '\d+\.\d+\.\d+\.\d+:\d+->\d+\s*\(pub,tcp\)' || true)

            if [[ -n "$pub_tcp" ]]; then
                REMOTE_IP=$(echo "$pub_tcp" | grep -oP '^\d+\.\d+\.\d+\.\d+')
                REMOTE_PORT=$(echo "$pub_tcp" | grep -oP '(?<=:)\d+(?=->)')

                # Log what we found
                mkdir -p "$DEPLOY_LOGS_DIR"
                local log_file="${DEPLOY_LOGS_DIR}/pod_${pod_id}_$(date +%Y%m%d_%H%M%S).log"
                {
                    echo "# RunPod Pod Info"
                    echo "# Timestamp: $(date -Iseconds)"
                    echo "POD_ID=${pod_id}"
                    echo "SSH_HOST=${REMOTE_IP}"
                    echo "SSH_PORT=${REMOTE_PORT}"
                    echo ""
                    echo "# Raw output:"
                    echo "$info"
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
        fi

        echo -e "  Waiting for port info … (${elapsed}s / ${max_wait}s)"
        sleep "$interval"
        elapsed=$((elapsed + interval))
    done

    # Fallback: log whatever we have
    local info
    info=$(runpodctl get pod --allfields 2>&1 || true)
    info=$(printf '%s\n' "$info" | sed "s/$(printf '\xc2\xa0')/ /g")
    mkdir -p "$DEPLOY_LOGS_DIR"
    local log_file="${DEPLOY_LOGS_DIR}/pod_${pod_id}_$(date +%Y%m%d_%H%M%S).log"
    {
        echo "# RunPod Pod Info (ports not parsed)"
        echo "# Timestamp: $(date -Iseconds)"
        echo "POD_ID=${pod_id}"
        echo ""
        echo "# Raw output:"
        echo "$info"
    } > "$log_file"

    log_warn "SSH port info not available after ${max_wait}s. Check log: $log_file"
    log_info "You can check manually: runpodctl get pod --allfields"
    return 1
}

# Start an existing RunPod pod.
# Args: $1 = pod_id
_runpod_start() {
    local pod_id="$1"
    log_step "Starting RunPod pod ${pod_id} …"
    local output
    output=$(runpodctl start pod "$pod_id" 2>&1) || {
        log_error "Failed to start pod: $output"
        return 1
    }
    echo "$output"
    _runpod_wait_ready "$pod_id"
    _runpod_get_ssh_info "$pod_id"
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


    # Build command
    local cmd=(runpodctl create pod)
    cmd+=(--name "$POD_NAME")
    cmd+=(--gpuType "$GPU_TYPE")
    cmd+=(--gpuCount "${GPU_COUNT:-1}")

    case "${CLOUD_TIER:-community}" in
        secure)    cmd+=(--secureCloud) ;;
        community) cmd+=(--communityCloud) ;;
        *) log_error "Invalid CLOUD_TIER: ${CLOUD_TIER}"; return 1 ;;
    esac

    [[ -n "${TEMPLATE_ID:-}" ]] && cmd+=(--templateId "$TEMPLATE_ID")
    [[ -n "${IMAGE_NAME:-}" ]] && cmd+=(--imageName "$IMAGE_NAME")

    cmd+=(--containerDiskSize "${CONTAINER_DISK_SIZE:-20}")
    [[ -n "${VOLUME_SIZE:-}" ]] && cmd+=(--volumeSize "${VOLUME_SIZE}")
    [[ -n "${VOLUME_PATH:-}" ]] && cmd+=(--volumePath "$VOLUME_PATH")
    [[ -n "${NETWORK_VOLUME_ID:-}" ]] && cmd+=(--networkVolumeId "$NETWORK_VOLUME_ID")
    cmd+=(--mem "${MIN_MEMORY_GB:-20}")
    cmd+=(--vcpu "${MIN_VCPU:-1}")
    [[ -n "${MAX_COST:-}" ]] && cmd+=(--cost "$MAX_COST")

    if [[ -n "${PORTS:-}" ]]; then
        for port in $PORTS; do
            cmd+=(--ports "$port")
        done
    fi
    if [[ -n "${ENV_VARS:-}" ]]; then
        IFS=';' read -ra _env_arr <<< "$ENV_VARS"
        for env_var in "${_env_arr[@]}"; do
            env_var="$(printf '%s' "$env_var" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
            # Interpret escape sequences (\n → newline, etc.)
            [[ -n "$env_var" ]] && cmd+=(--env "$(printf '%b' "$env_var")")
        done
    fi
    [[ -n "${CONTAINER_ARGS:-}" ]] && cmd+=(--args "$CONTAINER_ARGS")

    # Create
    log_step "Creating RunPod pod …"
    echo -e "  ${BLUE}Command:${NC} ${cmd[*]}"
    local output
    output=$("${cmd[@]}" 2>&1) || {
        log_error "Failed to create pod: $output"
        return 1
    }
    echo "$output"

    # Extract pod ID: runpodctl outputs something like: pod "abc123" created for $X.XX / hr
    POD_ID=$(echo "$output" | grep -oP '(?<=pod ")([^"]+)' | head -1 || true)
    if [[ -z "$POD_ID" ]]; then
        POD_ID=$(echo "$output" | grep -oP '[a-z0-9]{20,}' | head -1 || true)
    fi
    if [[ -z "$POD_ID" ]]; then
        log_warn "Could not parse pod ID from output. Check output above."
        return 1
    fi

    log_done "Pod created: ${POD_ID}"
    _runpod_wait_ready "$POD_ID"
    _runpod_get_ssh_info "$POD_ID"
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

        # Try to restart via provider if available
        if [[ "${PROVIDER:-}" == "runpod" && -n "${POD_ID:-}" ]]; then
            log_info "Attempting to restart via RunPod …"
            _runpod_require_cli || return 1
            _runpod_start "$POD_ID" || return 1
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
                _runpod_require_cli || return 1
                if [[ -n "${POD_ID:-}" ]]; then
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
