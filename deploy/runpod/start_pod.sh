#!/usr/bin/env bash
#
# start_pod.sh — Start a stopped RunPod pod and retrieve SSH connection info.
#
# Usage:
#   ./deploy/runpod/start_pod.sh <POD_ID> [--bid PRICE]
#
# Arguments:
#   POD_ID      The ID of the pod to start
#   --bid PRICE Optional bid price per GPU ($/hr) for spot instances
#
# Examples:
#   ./deploy/runpod/start_pod.sh abc123xyz456
#   ./deploy/runpod/start_pod.sh abc123xyz456 --bid 0.50
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
    echo "Usage: $0 <POD_ID> [--bid PRICE]"
    exit 1
fi

POD_ID="$1"
shift

BID_PRICE=""
while [[ $# -gt 0 ]]; do
    case "$1" in
        --bid)
            [[ $# -lt 2 ]] && { echo -e "${RED}Error:${NC} --bid requires a PRICE argument."; exit 1; }
            BID_PRICE="$2"
            shift 2
            ;;
        *)
            echo -e "${RED}Error:${NC} Unknown option: $1"
            exit 1
            ;;
    esac
done

# ── Start the pod ───────────────────────────────────────────────────────────
CMD=(runpodctl start pod "$POD_ID")
[[ -n "$BID_PRICE" ]] && CMD+=(--bid "$BID_PRICE")

echo -e "${YELLOW}⚙ Starting pod ${POD_ID} …${NC}"
echo -e "  ${BLUE}Command:${NC} ${CMD[*]}"
echo ""

OUTPUT=$("${CMD[@]}" 2>&1) || {
    echo -e "${RED}Error:${NC} Failed to start pod."
    echo "$OUTPUT"
    exit 1
}

echo "$OUTPUT"
echo ""
echo -e "${GREEN}✔ Pod start requested.${NC}"

# ── Wait for pod to be ready ─────────────────────────────────────────────────
echo ""
echo -e "${YELLOW}⚙ Waiting for pod to become ready …${NC}"

MAX_WAIT=300  # 5 minutes
ELAPSED=0
INTERVAL=10

while [[ $ELAPSED -lt $MAX_WAIT ]]; do
    POD_INFO=$(runpodctl get pod "$POD_ID" 2>&1 || true)

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
    echo -e "  Check status with: runpodctl get pod ${POD_ID}"
fi

# ── Wait for SSH port info to become available ──────────────────────────────
# RunPod PORTS column format: IP:EXTERNAL_PORT->INTERNAL_PORT (pub,tcp)
# Ports take ~45-60s after RUNNING status to be assigned.
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

# ── Save to deploy logs ────────────────────────────────────────────────────
mkdir -p "$DEPLOY_LOGS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${DEPLOY_LOGS_DIR}/start_${POD_ID}_${TIMESTAMP}.log"

{
    echo "# RunPod Pod Start Log"
    echo "# Started: $(date -Iseconds)"
    echo ""
    echo "POD_ID=${POD_ID}"
    [[ -n "$BID_PRICE" ]] && echo "BID_PRICE=${BID_PRICE}"
    [[ -n "$SSH_HOST" ]] && echo "SSH_HOST=${SSH_HOST}"
    [[ -n "$SSH_PORT" ]] && echo "SSH_PORT=${SSH_PORT}"
    echo ""
    echo "# ── Start Output ──"
    echo "$OUTPUT"
    echo ""
    echo "# ── Pod Info (allfields) ──"
    echo "${ALL_INFO:-}"
} > "$LOG_FILE"

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
echo ""
echo -e "Useful commands:"
echo -e "  ${BLUE}Status:${NC}  runpodctl get pod ${POD_ID}"
echo -e "  ${BLUE}Stop:${NC}    ./deploy/runpod/stop_pod.sh ${POD_ID}"
