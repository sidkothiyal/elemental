#!/usr/bin/env bash
#
# start_pod.sh — Start a stopped RunPod pod and retrieve SSH connection info.
#
# Uses the RunPod REST API (no runpodctl needed).
#
# Requires: curl, jq, RUNPOD_API_KEY env var
#
# Usage:
#   ./deploy/runpod/start_pod.sh <POD_ID> [OPTIONS]
#
# Examples:
#   ./deploy/runpod/start_pod.sh abc123xyz456
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOY_LOGS_DIR="${REPO_ROOT}/deploy/logs"

# ── Source API helpers ───────────────────────────────────────────────────────
source "${SCRIPT_DIR}/api.sh"
require_runpod_api

# ── Parse arguments ─────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    log_error "Pod ID is required."
    echo ""
    echo "Usage: $0 <POD_ID>"
    exit 1
fi

POD_ID="$1"
shift

if [[ $# -gt 0 ]]; then
    log_error "Unknown option: $1"
    exit 1
fi

# ── Start the pod ───────────────────────────────────────────────────────────
log_step "Starting pod ${POD_ID} …"
echo -e "  ${BLUE}API:${NC}  POST ${RUNPOD_API_BASE}/pods/${POD_ID}/start"
echo ""

OUTPUT=$(runpod_start_pod "$POD_ID")
_runpod_sync_http_code

if [[ "$RUNPOD_API_HTTP_CODE" != "200" && "$RUNPOD_API_HTTP_CODE" != "201" ]]; then
    log_error "Failed to start pod (HTTP ${RUNPOD_API_HTTP_CODE})."
    echo "$OUTPUT" | jq . 2>/dev/null || echo "$OUTPUT"
    exit 1
fi

echo "$OUTPUT" | jq . 2>/dev/null || echo "$OUTPUT"
echo ""
log_done "Pod start requested."

# ── Wait for pod to be ready ─────────────────────────────────────────────────
echo ""
runpod_wait_ready "$POD_ID" 300

# ── Wait for SSH port info ───────────────────────────────────────────────────
echo ""
runpod_wait_ssh_info "$POD_ID" 120
SSH_HOST="${REMOTE_IP:-}"
SSH_PORT="${REMOTE_PORT:-}"

# ── Save to deploy logs ────────────────────────────────────────────────────
mkdir -p "$DEPLOY_LOGS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${DEPLOY_LOGS_DIR}/start_${POD_ID}_${TIMESTAMP}.log"

{
    echo "# RunPod Pod Start Log"
    echo "# Started: $(date -Iseconds)"
    echo ""
    echo "POD_ID=${POD_ID}"
    [[ -n "$SSH_HOST" ]] && echo "SSH_HOST=${SSH_HOST}"
    [[ -n "$SSH_PORT" ]] && echo "SSH_PORT=${SSH_PORT}"
    echo ""
    echo "# ── API Response ──"
    echo "$OUTPUT" | jq . 2>/dev/null || echo "$OUTPUT"
} > "$LOG_FILE"

# ── Display SSH connection info ─────────────────────────────────────────────
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
