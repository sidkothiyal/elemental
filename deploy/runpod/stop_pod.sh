#!/usr/bin/env bash
#
# stop_pod.sh — Stop a running RunPod pod.
#
# Stops compute on the pod. Persistent storage is retained but continues
# to incur charges until the pod is removed.
#
# Uses the RunPod REST API (no runpodctl needed).
#
# Requires: curl, jq, RUNPOD_API_KEY env var
#
# Usage:
#   ./deploy/runpod/stop_pod.sh <POD_ID>
#
# Examples:
#   ./deploy/runpod/stop_pod.sh abc123xyz456
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

# ── Stop the pod ────────────────────────────────────────────────────────────
log_step "Stopping pod ${POD_ID} …"
echo -e "  ${BLUE}API:${NC} POST ${RUNPOD_API_BASE}/pods/${POD_ID}/stop"
echo ""

OUTPUT=$(runpod_stop_pod "$POD_ID")
_runpod_sync_http_code

if [[ "$RUNPOD_API_HTTP_CODE" != "200" ]]; then
    log_error "Failed to stop pod (HTTP ${RUNPOD_API_HTTP_CODE})."
    echo "$OUTPUT" | jq . 2>/dev/null || echo "$OUTPUT"
    exit 1
fi

echo "$OUTPUT" | jq . 2>/dev/null || echo "$OUTPUT"

# ── Save to deploy logs ────────────────────────────────────────────────────
mkdir -p "$DEPLOY_LOGS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${DEPLOY_LOGS_DIR}/stop_${POD_ID}_${TIMESTAMP}.log"

{
    echo "# RunPod Pod Stop Log"
    echo "# Stopped: $(date -Iseconds)"
    echo ""
    echo "POD_ID=${POD_ID}"
    echo ""
    echo "# ── API Response ──"
    echo "$OUTPUT" | jq . 2>/dev/null || echo "$OUTPUT"
} > "$LOG_FILE"

echo ""
log_done "Pod ${POD_ID} stopped."
echo -e "${GREEN}Log:${NC} ${LOG_FILE}"
echo ""
echo -e "${YELLOW}Note:${NC} Persistent storage is retained and continues to incur charges."
echo -e "  To restart: ./deploy/runpod/start_pod.sh ${POD_ID}"
echo -e "  To remove:  curl -X DELETE -H 'Authorization: Bearer \$RUNPOD_API_KEY' ${RUNPOD_API_BASE}/pods/${POD_ID}"
