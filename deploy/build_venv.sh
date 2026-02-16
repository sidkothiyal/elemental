#!/usr/bin/env bash
#
# build_venv.sh — Generate dependency hashes and build Docker image locally.
#
# The Docker image is only used as a build environment to produce the .venv.
# It is never run on the remote — only the extracted venv is transferred.
#
# Expects (env):  REPO_ROOT, DOCKER_IMAGE
# Sets (globals): DEPS_CHANGED (true/false), CURRENT_HASH
#
# Usage:
#   source deploy/common.sh
#   build_venv
#
#   # Or standalone:
#   ./deploy/build_venv.sh
#

[[ -z "${DEPLOY_DIR:-}" ]] && source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

build_venv() {
    local hash_file="${REPO_ROOT}/.deps_hash"
    CURRENT_HASH=$(cat "${REPO_ROOT}/pyproject.toml" "${REPO_ROOT}/uv.lock" | md5sum | cut -d' ' -f1)

    # Heavy-deps hash controls Docker Layer 1 cache invalidation.
    # Only regenerate when the overall deps hash has changed.
    if [[ ! -f "$hash_file" ]] || [[ "$(cat "$hash_file")" != "$CURRENT_HASH" ]]; then
        log_step "Generating dependency hashes …"
        uv run python3 -c "
import tomllib
with open('${REPO_ROOT}/pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
heavy = sorted(str(d) for d in data.get('dependency-groups', {}).get('heavy', []))
print('\n'.join(heavy))
" | md5sum | cut -d' ' -f1 > "${REPO_ROOT}/.heavy-deps-hash"
        echo "$CURRENT_HASH" > "$hash_file"
        DEPS_CHANGED=true
        echo -e "  ${GREEN}Hashes updated.${NC}"
    else
        DEPS_CHANGED=false
    fi

    # Build Docker image if deps changed
    if [[ "$DEPS_CHANGED" == true ]]; then
        log_step "Dependencies changed — rebuilding Docker image locally …"
        DOCKER_BUILDKIT=1 docker build -t "${DOCKER_IMAGE}:latest" "${REPO_ROOT}"
        log_done "Image built."
    else
        log_done "Dependencies unchanged — skipping Docker build."
    fi
}

# ── Standalone mode ─────────────────────────────────────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
    build_venv
    echo ""
    echo "DEPS_CHANGED=$DEPS_CHANGED"
    echo "CURRENT_HASH=$CURRENT_HASH"
fi
