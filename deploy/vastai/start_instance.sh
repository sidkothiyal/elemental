#!/usr/bin/env bash
#
# start_instance.sh — Start a stopped Vast.ai instance and retrieve SSH connection info.
#
# Uses the Vast.ai REST API.
#
# Requires: curl, jq, VAST_API_KEY env var
#
# Usage:
#   ./deploy/vastai/start_instance.sh <INSTANCE_ID>
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOY_LOGS_DIR="${REPO_ROOT}/deploy/logs"

# ── Source API helpers ───────────────────────────────────────────────────────
source "${SCRIPT_DIR}/api.sh"
require_vastai_api

# ── Parse arguments ─────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    log_error "Instance ID is required."
    echo ""
    echo "Usage: $0 <INSTANCE_ID>"
    exit 1
fi

INSTANCE_ID="$1"
shift

if [[ $# -gt 0 ]]; then
    log_error "Unknown option: $1"
    exit 1
fi

# ── Start the instance ──────────────────────────────────────────────────────
log_step "Starting instance ${INSTANCE_ID} …"
echo -e "  ${BLUE}API:${NC}  PUT ${VAST_API_BASE}/instances/${INSTANCE_ID}/"
echo ""

OUTPUT=$(vastai_start_instance "$INSTANCE_ID")
vastai_sync_http_code

if [[ "$VAST_API_HTTP_CODE" != "200" ]]; then
    log_error "Failed to start instance (HTTP ${VAST_API_HTTP_CODE})."
    echo "$OUTPUT" | jq . 2>/dev/null || echo "$OUTPUT"
    exit 1
fi

echo "$OUTPUT" | jq . 2>/dev/null || echo "$OUTPUT"
echo ""
log_done "Instance start requested."

# ── Wait for instance to be ready ───────────────────────────────────────────
echo ""
vastai_wait_ready "$INSTANCE_ID" 300

# ── Wait for SSH port info ───────────────────────────────────────────────────
echo ""
vastai_wait_ssh_info "$INSTANCE_ID" 120
SSH_HOST="${REMOTE_IP:-}"
SSH_PORT="${REMOTE_PORT:-}"

# ── Save to deploy logs ──────────────────────────────────────────────────────
mkdir -p "$DEPLOY_LOGS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${DEPLOY_LOGS_DIR}/start_${INSTANCE_ID}_${TIMESTAMP}.log"

{
    echo "# Vast.ai Instance Start Log"
    echo "# Started: $(date -Iseconds)"
    echo ""
    echo "INSTANCE_ID=${INSTANCE_ID}"
    [[ -n "$SSH_HOST" ]] && echo "SSH_HOST=${SSH_HOST}"
    [[ -n "$SSH_PORT" ]] && echo "SSH_PORT=${SSH_PORT}"
    echo ""
    echo "# ── API Response ──"
    echo "$OUTPUT" | jq . 2>/dev/null || echo "$OUTPUT"
} > "$LOG_FILE"

if [[ -z "$SSH_HOST" || -z "$SSH_PORT" ]]; then
    echo ""
    echo -e "${YELLOW}SSH details not yet available.${NC}"
    echo -e "  Check with: curl -s -H 'Authorization: Bearer \$VAST_API_KEY' ${VAST_API_BASE}/instances/${INSTANCE_ID}/ | jq '.instances | {ssh_host, ssh_port}'"
fi

echo ""
echo -e "${GREEN}Instance ID:${NC} ${INSTANCE_ID}"
echo -e "${GREEN}Log:${NC}       ${LOG_FILE}"
echo ""
echo -e "Useful commands:"
echo -e "  ${BLUE}Status:${NC}  curl -s -H 'Authorization: Bearer \$VAST_API_KEY' ${VAST_API_BASE}/instances/${INSTANCE_ID}/ | jq .instances.cur_state"
echo -e "  ${BLUE}Stop:${NC}    ./deploy/vastai/stop_instance.sh ${INSTANCE_ID}"
