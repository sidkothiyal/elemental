#!/usr/bin/env bash
#
# create_pod.sh — Create a new RunPod pod using configuration from pod.conf.
#
# Reads pod configuration from a config file, creates the pod via the RunPod
# REST API, retrieves SSH connection details, and saves them to the deploy logs.
#
# Requires: curl, jq, RUNPOD_API_KEY env var
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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOY_LOGS_DIR="${REPO_ROOT}/deploy/logs"

# ── Source API helpers ───────────────────────────────────────────────────────
source "${SCRIPT_DIR}/api.sh"
require_runpod_api

# ── Load configuration ──────────────────────────────────────────────────────
CONFIG_FILE="${1:-${SCRIPT_DIR}/pod.conf}"

if [[ ! -f "$CONFIG_FILE" ]]; then
    log_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

echo -e "${BLUE}Loading config:${NC} $CONFIG_FILE"
# shellcheck source=pod.conf
source "$CONFIG_FILE"

# ── Validate required fields ────────────────────────────────────────────────
if [[ -z "${POD_NAME:-}" ]]; then
    log_error "POD_NAME is required in config file."
    exit 1
fi
if [[ -z "${GPU_TYPE:-}" ]]; then
    log_error "GPU_TYPE is required in config file."
    exit 1
fi

# ── Build JSON payload ──────────────────────────────────────────────────────
PAYLOAD=$(build_pod_create_json)

echo ""
log_step "Creating RunPod pod …"
echo -e "  ${BLUE}Payload:${NC}"
echo "$PAYLOAD" | jq .
echo ""

# ── Create the pod ──────────────────────────────────────────────────────────
OUTPUT=$(runpod_create_pod "$PAYLOAD")
_runpod_sync_http_code

if [[ "$RUNPOD_API_HTTP_CODE" != "200" && "$RUNPOD_API_HTTP_CODE" != "201" ]]; then
    log_error "Failed to create pod (HTTP ${RUNPOD_API_HTTP_CODE})."
    echo "$OUTPUT" | jq . 2>/dev/null || echo "$OUTPUT"
    exit 1
fi

# ── Extract pod ID from response ────────────────────────────────────────────
POD_ID=$(echo "$OUTPUT" | jq -r '.id // empty' 2>/dev/null)

if [[ -z "$POD_ID" ]]; then
    log_warn "Could not parse pod ID from response."
    echo "$OUTPUT" | jq . 2>/dev/null || echo "$OUTPUT"
    exit 1
fi

echo ""
log_done "Pod created: ${POD_ID}"

# ── Wait for pod to be ready ─────────────────────────────────────────────────
echo ""
runpod_wait_ready "$POD_ID" 300

# ── Wait for SSH port info ───────────────────────────────────────────────────
echo ""
runpod_wait_ssh_info "$POD_ID" 120
SSH_HOST="${REMOTE_IP:-}"
SSH_PORT="${REMOTE_PORT:-}"

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
    echo "# ── Creation Payload ──"
    echo "$PAYLOAD" | jq . 2>/dev/null || echo "$PAYLOAD"
    echo ""
    echo "# ── API Response ──"
    echo "$OUTPUT" | jq . 2>/dev/null || echo "$OUTPUT"
} > "$LOG_FILE"

echo ""
log_done "Pod info saved to: $LOG_FILE"

# ── Display connection info ──────────────────────────────────────────────────
if [[ -z "$SSH_HOST" || -z "$SSH_PORT" ]]; then
    echo ""
    echo -e "${YELLOW}SSH details not yet available.${NC}"
    echo -e "  Check with: curl -s -H 'Authorization: Bearer \$RUNPOD_API_KEY' ${RUNPOD_API_BASE}/pods/${POD_ID} | jq '.publicIp, .portMappings'"
fi

echo ""
echo -e "${GREEN}Pod ID:${NC} ${POD_ID}"
echo -e "${GREEN}Log:${NC}    ${LOG_FILE}"
echo ""
echo -e "Useful commands:"
echo -e "  ${BLUE}Status:${NC}  curl -s -H 'Authorization: Bearer \$RUNPOD_API_KEY' ${RUNPOD_API_BASE}/pods/${POD_ID} | jq .desiredStatus"
echo -e "  ${BLUE}Stop:${NC}    ./deploy/runpod/stop_pod.sh ${POD_ID}"
echo -e "  ${BLUE}Start:${NC}   ./deploy/runpod/start_pod.sh ${POD_ID}"
