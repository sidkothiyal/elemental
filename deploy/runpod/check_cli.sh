#!/usr/bin/env bash
#
# check_cli.sh — Verify that runpodctl is installed and configured.
#
# Sources this script to get the `require_runpodctl` function,
# or run it directly to check the CLI status.
#
# Usage:
#   source deploy/runpod/check_cli.sh
#   require_runpodctl          # exits with error if not installed
#
#   # Or run directly:
#   ./deploy/runpod/check_cli.sh
#

set -euo pipefail

# ── Colour helpers ──────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ── Check if runpodctl is installed and configured ──────────────────────────
require_runpodctl() {
    if ! command -v runpodctl &>/dev/null; then
        echo -e "${RED}Error:${NC} runpodctl is not installed."
        echo ""
        echo -e "${YELLOW}Installation instructions:${NC}"
        echo -e "  See: ${BLUE}https://docs.runpod.io/runpodctl/overview${NC}"
        echo ""
        echo -e "  ${BLUE}Linux:${NC}"
        echo "    wget --quiet --show-progress https://github.com/Run-Pod/runpodctl/releases/download/v1.14.3/runpodctl-linux-amd64 -O runpodctl && chmod +x runpodctl && sudo cp runpodctl /usr/bin/runpodctl"
        echo ""
        echo -e "  ${BLUE}macOS (Homebrew):${NC}"
        echo "    brew install runpod/runpodctl/runpodctl"
        echo ""
        echo -e "  ${BLUE}macOS (ARM / Apple Silicon):${NC}"
        echo "    wget --quiet --show-progress https://github.com/runpod/runpodctl/releases/download/v1.14.3/runpodctl-darwin-arm64 -O runpodctl && chmod +x runpodctl && sudo mv runpodctl /usr/local/bin/runpodctl"
        echo ""
        echo -e "  ${BLUE}Windows:${NC}"
        echo "    wget https://github.com/runpod/runpodctl/releases/download/v1.14.3/runpodctl-windows-amd64.exe -O runpodctl.exe"
        echo ""
        echo -e "${YELLOW}After installing, configure your API key:${NC}"
        echo -e "  See: ${BLUE}https://docs.runpod.io/runpodctl/reference/runpodctl-config${NC}"
        echo ""
        echo "    runpodctl config --apiKey YOUR_API_KEY"
        echo ""
        return 1
    fi

    # Verify that an API key is configured by checking the config file
    local config_file="${HOME}/.runpod/config.toml"
    if [[ ! -f "$config_file" ]]; then
        echo -e "${RED}Error:${NC} runpodctl is installed but not configured."
        echo ""
        echo -e "${YELLOW}Configure your API key:${NC}"
        echo -e "  See: ${BLUE}https://docs.runpod.io/runpodctl/reference/runpodctl-config${NC}"
        echo ""
        echo "    runpodctl config --apiKey YOUR_API_KEY"
        echo ""
        echo "  You can generate an API key from: https://www.runpod.io/console/user/settings"
        echo ""
        return 1
    fi

    echo -e "${GREEN}✔ runpodctl is installed and configured.${NC}"
    runpodctl version
    return 0
}

# ── Run directly if not sourced ─────────────────────────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    require_runpodctl
fi
