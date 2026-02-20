#!/usr/bin/env bash
#
# check_cli.sh — Verify that RunPod REST API prerequisites are met.
#
# Checks for: curl, jq, and RUNPOD_API_KEY environment variable.
# This replaces the old runpodctl-based check.
#
# Usage:
#   source deploy/runpod/check_cli.sh
#   require_runpod_api          # exits with error if prerequisites not met
#
#   # Or run directly:
#   ./deploy/runpod/check_cli.sh
#

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source the API helper which defines require_runpod_api
source "${SCRIPT_DIR}/api.sh"

# ── Run directly if not sourced ─────────────────────────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    require_runpod_api
fi
