#!/usr/bin/env bash
#
# cleanup_stopped_pods.sh — Permanently terminate (delete) all RunPod pods that
# are currently stopped (EXITED), while leaving running pods untouched.
#
# Stopped pods still incur storage charges until removed. This script calls the
# RunPod DELETE /pods/{id} API to permanently remove them. Data outside
# network volumes is lost; attached network volumes are not deleted.
#
# Uses the RunPod REST API (no runpodctl needed).
#
# Requires: curl, jq, RUNPOD_API_KEY env var
#
# Usage:
#   ./deploy/runpod/cleanup_stopped_pods.sh [--dry-run]
#
# Options:
#   --dry-run   List stopped pods that would be terminated without deleting them.
#
# Reference: https://docs.runpod.io/api-reference/pods/DELETE/pods/podId
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
DEPLOY_LOGS_DIR="${REPO_ROOT}/deploy/logs"

# ── Source API helpers ───────────────────────────────────────────────────────
source "${SCRIPT_DIR}/api.sh"
require_runpod_api

# ── Parse arguments ─────────────────────────────────────────────────────────
DRY_RUN=false
while [[ $# -gt 0 ]]; do
    case "$1" in
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            head -30 "${BASH_SOURCE[0]}" | tail -22
            exit 0
            ;;
        *)
            log_error "Unknown option: $1"
            exit 1
            ;;
    esac
done

# ── List stopped pods ────────────────────────────────────────────────────────
log_step "Fetching stopped pods (desiredStatus=EXITED) …"
output=$(runpod_list_pods "EXITED")
_runpod_sync_http_code

if [[ "$RUNPOD_API_HTTP_CODE" != "200" ]]; then
    log_error "Failed to list pods (HTTP ${RUNPOD_API_HTTP_CODE})."
    echo "$output" | jq . 2>/dev/null || echo "$output"
    exit 1
fi

# Extract pod IDs and names (same order)
ids=()
names=()
while IFS='|' read -r id name; do
    [[ -z "$id" || "$id" == "null" ]] && continue
    ids+=("$id")
    names+=("${name:-<unnamed>}")
done < <(echo "$output" | jq -r '.[] | "\(.id)|\(.name // "<unnamed>")"' 2>/dev/null)

count=${#ids[@]}
if [[ $count -eq 0 ]]; then
    log_done "No stopped pods found. Nothing to clean up."
    exit 0
fi

echo ""
echo -e "${BLUE}Stopped pods (${count}):${NC}"
for i in "${!ids[@]}"; do
    echo -e "  ${ids[$i]}  ${names[$i]:-<unnamed>}"
done
echo ""

if [[ "$DRY_RUN" == true ]]; then
    log_info "Dry run: would permanently terminate the ${count} pod(s) above."
    echo -e "${YELLOW}Run without --dry-run to delete them.${NC}"
    exit 0
fi

# ── Terminate each stopped pod ──────────────────────────────────────────────
log_step "Permanently terminating ${count} stopped pod(s) …"
failed=0
for i in "${!ids[@]}"; do
    pid="${ids[$i]}"
    pname="${names[$i]:-<unnamed>}"
    echo -n "  Deleting ${pid} (${pname}) … "
    del_output=$(runpod_remove_pod "$pid")
    _runpod_sync_http_code
    if [[ "$RUNPOD_API_HTTP_CODE" == "200" ]]; then
        echo -e "${GREEN}done${NC}"
    else
        echo -e "${RED}failed (HTTP ${RUNPOD_API_HTTP_CODE})${NC}"
        echo "$del_output" | jq . 2>/dev/null || echo "$del_output"
        failed=$((failed + 1))
    fi
done

echo ""
if [[ $failed -eq 0 ]]; then
    log_done "All ${count} stopped pod(s) have been terminated."
else
    log_warn "${failed} of ${count} pod(s) could not be terminated. Check output above."
    exit 1
fi

# ── Optional log ───────────────────────────────────────────────────────────
mkdir -p "$DEPLOY_LOGS_DIR"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="${DEPLOY_LOGS_DIR}/cleanup_stopped_pods_${TIMESTAMP}.log"
{
    echo "# RunPod Cleanup — Stopped Pods Terminated"
    echo "# $(date -Iseconds)"
    echo "# Pods deleted: ${count}"
    for i in "${!ids[@]}"; do
        echo "  ${ids[$i]}  ${names[$i]:-<unnamed>}"
    done
} > "$LOG_FILE"
echo -e "${BLUE}Log:${NC} ${LOG_FILE}"
