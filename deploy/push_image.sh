#!/usr/bin/env bash
#
# push_image.sh — Build the venv Docker image and push it to GitHub Container Registry.
#
# Used when creating RunPod/Vast instances so they start with the image already
# containing /app/.venv. Requires GITHUB_USERNAME and DOCKER_GHRC_TOKEN
# (or GITHUB_TOKEN with read:packages/write:packages scope).
#
# Expects (env): REPO_ROOT, DOCKER_IMAGE, GHRC_IMAGE (from common.sh when GITHUB_USERNAME set)
#               DOCKER_GHRC_TOKEN or GITHUB_TOKEN for docker login
#
# Usage:
#   source deploy/common.sh && source deploy/build_venv.sh && source deploy/push_image.sh
#   push_image
#

[[ -z "${DEPLOY_DIR:-}" ]] && source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_venv.sh"

push_image() {
    if [[ -z "${GITHUB_USERNAME:-}" ]]; then
        log_error "GITHUB_USERNAME is not set. Set it to push to ghcr.io."
        return 1
    fi

    local token="${DOCKER_GHRC_TOKEN:-${GITHUB_TOKEN:-}}"
    if [[ -z "$token" ]]; then
        log_error "DOCKER_GHRC_TOKEN (or GITHUB_TOKEN) is not set. Required for docker login ghcr.io."
        echo "  Create a PAT with read:packages and write:packages at https://github.com/settings/tokens"
        return 1
    fi

    # Build local image (sets DEPS_CHANGED, CURRENT_HASH)
    build_venv

    log_step "Logging in to ghcr.io …"
    if ! echo "$token" | docker login ghcr.io -u "$GITHUB_USERNAME" --password-stdin; then
        log_error "docker login ghcr.io failed."
        return 1
    fi
    log_done "Logged in to ghcr.io."

    log_step "Tagging and pushing ${GHRC_IMAGE} …"
    docker tag "${DOCKER_IMAGE}:latest" "$GHRC_IMAGE" || { log_error "docker tag failed."; return 1; }
    docker push "$GHRC_IMAGE" || { log_error "docker push failed."; return 1; }
    log_done "Pushed ${GHRC_IMAGE}."
}

# ── Standalone mode ─────────────────────────────────────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/build_venv.sh"
    push_image
fi
