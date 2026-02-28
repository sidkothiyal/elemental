#!/usr/bin/env bash
#
# stop_instance.sh — Stop a running Vast.ai instance.
#
# Stops compute; instance and data are preserved and can be started again.
#
# Uses the Vast.ai REST API.
#
# Requires: curl, jq, VAST_API_KEY env var
#
# Usage:
#   ./deploy/vastai/stop_instance.sh <INSTANCE_ID>
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

# ── Stop the instance ────────────────────────────────────────────────────────
log_step "Stopping instance ${INSTANCE_ID} …"
echo -e "  ${BLUE}API:${NC} PUT ${VAST_API_BASE}/instances/${INSTANCE_ID}/ (state: stopped)"
echo ""

OUTPUT=$(vastai_stop_instance "$INSTANCE_ID")
vastai_sync_http_code

if [[ "$VAST_API_HTTP_CODE" != "200" ]]; then
    log_error "Failed to stop instance (HTTP ${VAST_API_HTTP_CODE})."
    echo "$OUTPUT" | jq . 2>/dev/null || echo "$OUTPUT"
    exit 1
fi

echo "$OUTPUT" | jq . 2>/dev/null || echo "$OUTPUT"

# ── Save to deploy logs ──────────────────────────────────────────────────────
mkdir -p "$DEPLOY_LOGS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${DEPLOY_LOGS_DIR}/stop_${INSTANCE_ID}_${TIMESTAMP}.log"

{
    echo "# Vast.ai Instance Stop Log"
    echo "# Stopped: $(date -Iseconds)"
    echo ""
    echo "INSTANCE_ID=${INSTANCE_ID}"
    echo ""
    echo "# ── API Response ──"
    echo "$OUTPUT" | jq . 2>/dev/null || echo "$OUTPUT"
} > "$LOG_FILE"

echo ""
log_done "Instance ${INSTANCE_ID} stopped."
echo -e "${GREEN}Log:${NC} ${LOG_FILE}"
echo ""
echo -e "${YELLOW}Note:${NC} Instance is preserved. To start again: ./deploy/vastai/start_instance.sh ${INSTANCE_ID}"
echo -e "  To destroy permanently: curl -X DELETE -H 'Authorization: Bearer \$VAST_API_KEY' ${VAST_API_BASE}/instances/${INSTANCE_ID}/"
