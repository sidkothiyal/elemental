#!/usr/bin/env bash
#
# stop_pod.sh — Stop a running RunPod pod.
#
# Stops compute on the pod. Persistent storage is retained but continues
# to incur charges until the pod is removed.
#
# Usage:
#   ./deploy/runpod/stop_pod.sh <POD_ID>
#
# Examples:
#   ./deploy/runpod/stop_pod.sh abc123xyz456
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

# ── Parse arguments ─────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo -e "${RED}Error:${NC} Pod ID is required."
    echo ""
    echo "Usage: $0 <POD_ID>"
    exit 1
fi

POD_ID="$1"

# ── Stop the pod ────────────────────────────────────────────────────────────
echo -e "${YELLOW}⚙ Stopping pod ${POD_ID} …${NC}"
echo -e "  ${BLUE}Command:${NC} runpodctl stop pod ${POD_ID}"
echo ""

OUTPUT=$(runpodctl stop pod "$POD_ID" 2>&1) || {
    echo -e "${RED}Error:${NC} Failed to stop pod."
    echo "$OUTPUT"
    exit 1
}

echo "$OUTPUT"

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
    echo "# ── Stop Output ──"
    echo "$OUTPUT"
} > "$LOG_FILE"

echo ""
echo -e "${GREEN}✔ Pod ${POD_ID} stopped.${NC}"
echo -e "${GREEN}Log:${NC} ${LOG_FILE}"
echo ""
echo -e "${YELLOW}Note:${NC} Persistent storage is retained and continues to incur charges."
echo -e "  To restart: ./deploy/runpod/start_pod.sh ${POD_ID}"
echo -e "  To remove:  runpodctl remove pod ${POD_ID}"
