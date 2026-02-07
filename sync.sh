#!/usr/bin/env bash
#
# sync.sh — Sync the current repo state (committed + uncommitted) to a remote
#            host via rsync/SSH, respecting .gitignore exclusions.
#
# Usage:
#   ./sync.sh <user@host> [remote_path]
#
# Examples:
#   ./sync.sh alice@192.168.1.10                    # syncs to ~/elemental3/
#   ./sync.sh alice@192.168.1.10 /opt/project       # syncs to /opt/project/
#   ./sync.sh alice@myserver ~/work/elemental3       # syncs to ~/work/elemental3/

set -euo pipefail

# ── Colour helpers ──────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No colour

# ── Validate arguments ─────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
    echo -e "${RED}Error:${NC} Missing required argument."
    echo ""
    echo "Usage: $0 <user@host> [remote_path]"
    echo ""
    echo "  user@host    SSH destination (e.g. alice@192.168.1.10)"
    echo "  remote_path  Optional remote directory (default: ~/$(basename "$(pwd)")/ )"
    exit 1
fi

REMOTE_HOST="$1"
REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_NAME="$(basename "$REPO_ROOT")"
REMOTE_PATH="${2:-"~/${REPO_NAME}/"}"

# Ensure remote path ends with /
[[ "$REMOTE_PATH" != */ ]] && REMOTE_PATH="${REMOTE_PATH}/"

# ── Verify we are inside a git repository ──────────────────────────────────
if ! git -C "$REPO_ROOT" rev-parse --is-inside-work-tree &>/dev/null; then
    echo -e "${RED}Error:${NC} $REPO_ROOT is not a git repository."
    exit 1
fi

# ── Build rsync exclude list from git ──────────────────────────────────────
# We use git's own machinery to honour nested .gitignore files, global
# excludes, and .git/info/exclude — far more reliable than parsing
# .gitignore ourselves.
#
# `git ls-files --others --ignored --exclude-standard` lists every ignored
# file.  We feed those paths into rsync as excludes.  We also always exclude
# the .git directory itself.

echo -e "${YELLOW}Building exclude list from git …${NC}"

EXCLUDE_FILE="$(mktemp)"
trap 'rm -f "$EXCLUDE_FILE"' EXIT

# Always exclude .git metadata directory
echo ".git" > "$EXCLUDE_FILE"

# Append every file / directory that git considers ignored
git -C "$REPO_ROOT" ls-files --others --ignored --exclude-standard --directory \
    >> "$EXCLUDE_FILE"

NUM_EXCLUDES="$(wc -l < "$EXCLUDE_FILE")"
echo -e "  ${GREEN}${NUM_EXCLUDES}${NC} exclusion rule(s) collected."

# ── Ensure remote directory exists ─────────────────────────────────────────
echo -e "${YELLOW}Ensuring remote directory exists …${NC}"
ssh "$REMOTE_HOST" "mkdir -p ${REMOTE_PATH}"

# ── Run rsync ──────────────────────────────────────────────────────────────
echo -e "${YELLOW}Syncing${NC} ${REPO_ROOT}/ ${YELLOW}→${NC} ${REMOTE_HOST}:${REMOTE_PATH}"
echo ""

rsync -avz --progress --delete \
    --exclude-from="$EXCLUDE_FILE" \
    "$REPO_ROOT/" \
    "${REMOTE_HOST}:${REMOTE_PATH}"

echo ""
echo -e "${GREEN}✔ Sync complete.${NC}"
