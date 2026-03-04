#!/usr/bin/env bash
#
# create_instance.sh — Create a new Vast.ai instance using configuration from instance_*.conf.
#
# Reads config, searches for an offer (or uses OFFER_ID), creates the instance
# via the Vast.ai API, retrieves SSH connection details, and saves to deploy logs.
#
# Requires: curl, jq, VAST_API_KEY env var
#
# Usage:
#   ./deploy/vastai/create_instance.sh [CONFIG_FILE]
#
# Arguments:
#   CONFIG_FILE   Path to instance config (default: deploy/vastai/instance_rtx.conf)
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOY_LOGS_DIR="${REPO_ROOT}/deploy/logs"
DEPLOY_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

# ── Source API helpers ───────────────────────────────────────────────────────
source "${SCRIPT_DIR}/api.sh"
require_vastai_api

# ── Load configuration ──────────────────────────────────────────────────────
CONFIG_FILE="${1:-${SCRIPT_DIR}/instance_rtx.conf}"

if [[ ! -f "$CONFIG_FILE" ]]; then
    log_error "Config file not found: $CONFIG_FILE"
    exit 1
fi

echo -e "${BLUE}Loading config:${NC} $CONFIG_FILE"
# shellcheck source=instance_rtx.conf
source "$CONFIG_FILE"

# ── Resolve offer_id: from config or search ───────────────────────────────────
OFFER_ID="${OFFER_ID:-}"
if [[ -z "$OFFER_ID" ]]; then
    log_step "Searching for offers …"
    SEARCH_BODY=$(build_vastai_search_body)
    SEARCH_RESPONSE=$(vastai_search_offers "$SEARCH_BODY")
    vastai_sync_http_code
    if [[ "$VAST_API_HTTP_CODE" != "200" ]]; then
        log_error "Search offers failed (HTTP ${VAST_API_HTTP_CODE})."
        echo "$SEARCH_RESPONSE" | jq . 2>/dev/null || echo "$SEARCH_RESPONSE"
        exit 1
    fi
    OFFER_ID=$(echo "$SEARCH_RESPONSE" | jq -r '.offers[0].id // empty' 2>/dev/null)
    if [[ -z "$OFFER_ID" || "$OFFER_ID" == "null" ]]; then
        log_error "No offers found matching config (GPU_NAME=${GPU_NAME:-}, etc.)."
        echo "$SEARCH_RESPONSE" | jq . 2>/dev/null || echo "$SEARCH_RESPONSE"
        exit 1
    fi
    log_done "Using offer ID: ${OFFER_ID}"
fi

# ── Build create payload and create (with optional bid increment) ──────────────
OUTPUT=""
if [[ "${OFFER_TYPE:-}" == "bid" && -n "${PRICE_MAX:-}" ]]; then
    current="${PRICE:-}"
    price_step="${PRICE_STEP:-0.005}"
    price_max="${PRICE_MAX}"
    while true; do
        PRICE="$current" PAYLOAD=$(build_vastai_create_json)
        echo ""
        log_step "Creating Vast.ai instance (bid \$${current}/hr) …"
        echo -e "  ${BLUE}Offer ID:${NC} ${OFFER_ID}"
        echo "$PAYLOAD" | jq . 2>/dev/null
        OUTPUT=$(vastai_create_instance "$OFFER_ID" "$PAYLOAD")
        vastai_sync_http_code
        if [[ "$VAST_API_HTTP_CODE" == "200" ]]; then
            break
        fi
        log_warn "Create failed (HTTP ${VAST_API_HTTP_CODE}). Trying higher bid …"
        echo "$OUTPUT" | jq . 2>/dev/null || echo "$OUTPUT"
        if [[ "$current" == "$price_max" ]]; then
            log_error "Failed to create instance after bidding up to \$${price_max}/hr."
            exit 1
        fi
        current=$(awk -v current="$current" -v price_step="$price_step" -v price_max="$price_max" 'BEGIN {n=current+price_step; if(n>price_max)n=price_max; printf "%.4f", n}' 2>/dev/null || echo "$price_max")
    done
else
    PAYLOAD=$(build_vastai_create_json)
    echo ""
    log_step "Creating Vast.ai instance …"
    echo -e "  ${BLUE}Offer ID:${NC} ${OFFER_ID}"
    echo -e "  ${BLUE}Payload:${NC}"
    echo "$PAYLOAD" | jq . 2>/dev/null
    echo ""
    OUTPUT=$(vastai_create_instance "$OFFER_ID" "$PAYLOAD")
    vastai_sync_http_code
fi

if [[ "$VAST_API_HTTP_CODE" != "200" ]]; then
    log_error "Failed to create instance (HTTP ${VAST_API_HTTP_CODE})."
    echo "$OUTPUT" | jq . 2>/dev/null || echo "$OUTPUT"
    exit 1
fi

INSTANCE_ID=$(echo "$OUTPUT" | jq -r '.new_contract // empty' 2>/dev/null)
if [[ -z "$INSTANCE_ID" || "$INSTANCE_ID" == "null" ]]; then
    log_warn "Could not parse instance ID from response."
    echo "$OUTPUT" | jq . 2>/dev/null || echo "$OUTPUT"
    exit 1
fi

echo ""
log_done "Instance created: ${INSTANCE_ID}"

# ── Wait for instance to be ready ───────────────────────────────────────────
echo ""
vastai_wait_ready "$INSTANCE_ID" 300

# ── Wait for SSH port info ───────────────────────────────────────────────────
echo ""
vastai_wait_ssh_info "$INSTANCE_ID" 120
SSH_HOST="${REMOTE_IP:-}"
SSH_PORT="${REMOTE_PORT:-}"

# ── Save instance info to deploy logs ───────────────────────────────────────
mkdir -p "$DEPLOY_LOGS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${DEPLOY_LOGS_DIR}/instance_${INSTANCE_ID}_${TIMESTAMP}.log"

{
    echo "# Vast.ai Instance Log"
    echo "# Created: $(date -Iseconds)"
    echo "# Config:  $CONFIG_FILE"
    echo ""
    echo "INSTANCE_ID=${INSTANCE_ID}"
    echo "INSTANCE_LABEL=${INSTANCE_LABEL:-}"
    echo "GPU_NAME=${GPU_NAME:-}"
    echo "IMAGE=${IMAGE:-}"
    [[ -n "$SSH_HOST" ]] && echo "SSH_HOST=${SSH_HOST}"
    [[ -n "$SSH_PORT" ]] && echo "SSH_PORT=${SSH_PORT}"
    echo ""
    echo "# ── Create Payload ──"
    echo "$PAYLOAD" | jq . 2>/dev/null || echo "$PAYLOAD"
    echo ""
    echo "# ── API Response ──"
    echo "$OUTPUT" | jq . 2>/dev/null || echo "$OUTPUT"
} > "$LOG_FILE"

echo ""
log_done "Instance info saved to: $LOG_FILE"

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
echo -e "  ${BLUE}Start:${NC}   ./deploy/vastai/start_instance.sh ${INSTANCE_ID}"
