#!/usr/bin/env bash
#
# create_pod.sh — Create a new RunPod pod using configuration from pod.conf.
#
# Reads pod configuration from a config file, creates the pod via runpodctl,
# retrieves SSH connection details, and saves them to the deploy logs.
#
# Usage:
#   ./deploy/runpod/create_pod.sh [CONFIG_FILE]
#
# Arguments:
#   CONFIG_FILE   Path to pod config file (default: deploy/runpod/pod.conf)
#
# Examples:
#   ./deploy/runpod/create_pod.sh
#   ./deploy/runpod/create_pod.sh deploy/runpod/my_custom_pod.conf
#

set -euo pipefail

# ── Colour helpers ──────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOY_LOGS_DIR="${REPO_ROOT}/deploy/logs"

# ── Ensure runpodctl is available ───────────────────────────────────────────
source "${SCRIPT_DIR}/check_cli.sh"
require_runpodctl

# ── Load configuration ──────────────────────────────────────────────────────
CONFIG_FILE="${1:-${SCRIPT_DIR}/pod.conf}"

if [[ ! -f "$CONFIG_FILE" ]]; then
    echo -e "${RED}Error:${NC} Config file not found: $CONFIG_FILE"
    exit 1
fi

echo -e "${BLUE}Loading config:${NC} $CONFIG_FILE"
# shellcheck source=pod.conf
source "$CONFIG_FILE"

# ── Validate required fields ────────────────────────────────────────────────
if [[ -z "${POD_NAME:-}" ]]; then
    echo -e "${RED}Error:${NC} POD_NAME is required in config file."
    exit 1
fi
if [[ -z "${GPU_TYPE:-}" ]]; then
    echo -e "${RED}Error:${NC} GPU_TYPE is required in config file."
    exit 1
fi

# ── Build runpodctl command ─────────────────────────────────────────────────
CMD=(runpodctl create pod)
CMD+=(--name "$POD_NAME")
CMD+=(--gpuType "$GPU_TYPE")
CMD+=(--gpuCount "${GPU_COUNT:-1}")

# Cloud tier
case "${CLOUD_TIER:-community}" in
    secure)  CMD+=(--secureCloud) ;;
    community) CMD+=(--communityCloud) ;;
    *) echo -e "${RED}Error:${NC} Invalid CLOUD_TIER: ${CLOUD_TIER}. Use 'secure' or 'community'."; exit 1 ;;
esac

# Image or template
[[ -n "${TEMPLATE_ID:-}" ]] && CMD+=(--templateId "$TEMPLATE_ID")
[[ -n "${IMAGE_NAME:-}" ]] && CMD+=(--imageName "$IMAGE_NAME")

# Storage
CMD+=(--containerDiskSize "${CONTAINER_DISK_SIZE:-20}")
[[ -n "${VOLUME_SIZE:-}" ]] && CMD+=(--volumeSize "${VOLUME_SIZE}")
[[ -n "${VOLUME_PATH:-}" ]] && CMD+=(--volumePath "$VOLUME_PATH")
[[ -n "${NETWORK_VOLUME_ID:-}" ]] && CMD+=(--networkVolumeId "$NETWORK_VOLUME_ID")

# System resources
CMD+=(--mem "${MIN_MEMORY_GB:-20}")
CMD+=(--vcpu "${MIN_VCPU:-1}")

# Cost ceiling
[[ -n "${MAX_COST:-}" ]] && CMD+=(--cost "$MAX_COST")

# Ports
if [[ -n "${PORTS:-}" ]]; then
    for port in $PORTS; do
        CMD+=(--ports "$port")
    done
fi

# Environment variables (semicolon-separated KEY=VALUE pairs)
if [[ -n "${ENV_VARS:-}" ]]; then
    IFS=';' read -ra _env_arr <<< "$ENV_VARS"
    for env_var in "${_env_arr[@]}"; do
        env_var="$(printf '%s' "$env_var" | sed 's/^[[:space:]]*//;s/[[:space:]]*$//')"
        # Interpret escape sequences (\n → newline, etc.)
        [[ -n "$env_var" ]] && CMD+=(--env "$(printf '%b' "$env_var")")
    done
fi

# Container arguments
[[ -n "${CONTAINER_ARGS:-}" ]] && CMD+=(--args "$CONTAINER_ARGS")

# ── Create the pod ──────────────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}⚙ Creating pod …${NC}"
echo -e "  ${BLUE}Command:${NC} ${CMD[*]}"
echo ""

OUTPUT=$("${CMD[@]}" 2>&1) || {
    echo -e "${RED}Error:${NC} Failed to create pod."
    echo "$OUTPUT"
    exit 1
}

echo "$OUTPUT"

# ── Extract pod ID from output ──────────────────────────────────────────────
# runpodctl create pod typically outputs something like:
#   pod "abc123xyz456" created for $X.XX / hr
POD_ID=$(echo "$OUTPUT" | grep -oP '(?<=pod ")([^"]+)' | head -1 || true)

if [[ -z "$POD_ID" ]]; then
    # Try alternative parsing (some versions output differently)
    POD_ID=$(echo "$OUTPUT" | grep -oP '[a-z0-9]{20,}' | head -1 || true)
fi

if [[ -z "$POD_ID" ]]; then
    echo -e "${YELLOW}⚠ Could not parse pod ID from output.${NC}"
    echo -e "  Check the output above and note down the pod ID manually."
    exit 0
fi

echo ""
echo -e "${GREEN}✔ Pod created:${NC} ${POD_ID}"

# ── Wait for pod to be ready ─────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}⚙ Waiting for pod to become ready …${NC}"

MAX_WAIT=300  # 5 minutes
ELAPSED=0
INTERVAL=10

while [[ $ELAPSED -lt $MAX_WAIT ]]; do
    POD_INFO=$(runpodctl get pod "$POD_ID" 2>&1 || true)

    # Check if the pod is running
    if echo "$POD_INFO" | grep -qi "running"; then
        echo -e "${GREEN}✔ Pod is running.${NC}"
        break
    fi

    echo -e "  Waiting … (${ELAPSED}s / ${MAX_WAIT}s)"
    sleep "$INTERVAL"
    ELAPSED=$((ELAPSED + INTERVAL))
done

if [[ $ELAPSED -ge $MAX_WAIT ]]; then
    echo -e "${YELLOW}⚠ Pod did not become ready within ${MAX_WAIT}s.${NC}"
    echo -e "  Pod ID: ${POD_ID}"
    echo -e "  Check status with: runpodctl get pod ${POD_ID}"
fi

# ── Wait for SSH port info to become available ──────────────────────────────
# RunPod exposes SSH via a proxied host/port in the PORTS column.
# Ports take ~45-60s after the pod reaches RUNNING to be assigned.
# Format: IP:EXTERNAL_PORT->INTERNAL_PORT (pub,tcp)
echo ""
echo -e "${YELLOW}⚙ Waiting for SSH port info …${NC}"

SSH_HOST=""
SSH_PORT=""
PORT_WAIT=120  # up to 2 minutes for port info
PORT_ELAPSED=0

while [[ $PORT_ELAPSED -lt $PORT_WAIT ]]; do
    ALL_INFO=$(runpodctl get pod --allfields 2>&1 || true)
    # Sanitize: runpodctl outputs non-breaking spaces (U+00A0, \xc2\xa0) that break regex
    ALL_INFO=$(printf '%s\n' "$ALL_INFO" | sed "s/$(printf '\xc2\xa0')/ /g")
    POD_LINE=$(echo "$ALL_INFO" | grep "$POD_ID" || true)

    if [[ -n "$POD_LINE" ]]; then
        PUB_TCP=$(echo "$POD_LINE" | grep -oP '\d+\.\d+\.\d+\.\d+:\d+->\d+\s*\(pub,tcp\)' || true)
        if [[ -n "$PUB_TCP" ]]; then
            SSH_HOST=$(echo "$PUB_TCP" | grep -oP '^\d+\.\d+\.\d+\.\d+')
            SSH_PORT=$(echo "$PUB_TCP" | grep -oP '(?<=:)\d+(?=->)')
            echo -e "${GREEN}✔ SSH port info received.${NC}"
            break
        fi
    fi

    echo -e "  Waiting for port info … (${PORT_ELAPSED}s / ${PORT_WAIT}s)"
    sleep "$INTERVAL"
    PORT_ELAPSED=$((PORT_ELAPSED + INTERVAL))
done

# ── Save pod info to deploy logs ────────────────────────────────────────────
mkdir -p "$DEPLOY_LOGS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${DEPLOY_LOGS_DIR}/pod_${POD_ID}_${TIMESTAMP}.log"

{
    echo "# RunPod Pod Log"
    echo "# Created: $(date -Iseconds)"
    echo "# Config:  $CONFIG_FILE"
    echo ""
    echo "POD_ID=${POD_ID}"
    echo "POD_NAME=${POD_NAME}"
    echo "GPU_TYPE=${GPU_TYPE}"
    echo "GPU_COUNT=${GPU_COUNT:-1}"
    echo "IMAGE=${IMAGE_NAME:-${TEMPLATE_ID:-}}"
    [[ -n "$SSH_HOST" ]] && echo "SSH_HOST=${SSH_HOST}"
    [[ -n "$SSH_PORT" ]] && echo "SSH_PORT=${SSH_PORT}"
    echo ""
    echo "# ── Creation Output ──"
    echo "$OUTPUT"
    echo ""
    echo "# ── Pod Info (allfields) ──"
    echo "${ALL_INFO:-}"
} > "$LOG_FILE"

echo ""
echo -e "${GREEN}✔ Pod info saved to:${NC} $LOG_FILE"

# ── Display SSH connection info ─────────────────────────────────────────────
echo ""
echo -e "${BLUE}═══════════════════════════════════════════${NC}"
echo -e "${BLUE}  SSH Connection Details${NC}"
echo -e "${BLUE}═══════════════════════════════════════════${NC}"
if [[ -n "$SSH_HOST" && -n "$SSH_PORT" ]]; then
    echo -e "  ${GREEN}Host:${NC} $SSH_HOST"
    echo -e "  ${GREEN}Port:${NC} $SSH_PORT"
    echo -e "  ${GREEN}Connect:${NC} ssh root@${SSH_HOST} -p ${SSH_PORT}"
else
    echo -e "  ${YELLOW}SSH details not yet available.${NC}"
    echo -e "  Run: runpodctl get pod --allfields"
fi
echo -e "${BLUE}═══════════════════════════════════════════${NC}"

echo ""
echo -e "${GREEN}Pod ID:${NC} ${POD_ID}"
echo -e "${GREEN}Log:${NC}    ${LOG_FILE}"
echo -e ""
echo -e "Useful commands:"
echo -e "  ${BLUE}Status:${NC}  runpodctl get pod ${POD_ID}"
echo -e "  ${BLUE}Stop:${NC}    ./deploy/runpod/stop_pod.sh ${POD_ID}"
echo -e "  ${BLUE}Start:${NC}   ./deploy/runpod/start_pod.sh ${POD_ID}"
