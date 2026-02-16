#!/usr/bin/env bash
#
# sync_venv.sh — Transfer the .venv from the local Docker image to the remote.
#
# Checks the remote venv hash against local; only transfers when outdated.
# Strategy: docker cp → gzip → ssh → tar extract (no Docker needed on remote).
#
# Expects (env): REMOTE_HOST, SSH_OPTS, REMOTE_CODE_DIR, REMOTE_VENV_DIR,
#                DOCKER_IMAGE, DEPS_CHANGED, CURRENT_HASH
#
# Usage:
#   source deploy/common.sh && source deploy/sync_venv.sh
#   sync_venv
#

[[ -z "${DEPLOY_DIR:-}" ]] && source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

sync_venv() {
    # Check remote venv hash
    local remote_hash
    remote_hash=$(ssh_cmd "cat '${REMOTE_VENV_DIR}/.deps_hash' 2>/dev/null || echo ''")

    if [[ "$DEPS_CHANGED" == true ]] || [[ "$remote_hash" != "$CURRENT_HASH" ]]; then
        [[ "$remote_hash" != "$CURRENT_HASH" ]] && \
            log_step "Venv not found or outdated on remote — transferring …"
        [[ "$DEPS_CHANGED" == true ]] && \
            log_step "Dependencies changed — transferring updated venv …"

        echo -e "  Extracting .venv from Docker image and streaming to remote …"

        # Create a temporary container (never started) to access image filesystem
        local container_id
        container_id=$(docker create "${DOCKER_IMAGE}:latest" true)

        # Prepare remote directory
        ssh_cmd "mkdir -p '${REMOTE_CODE_DIR}' && rm -rf '${REMOTE_VENV_DIR}' && mkdir -p '${REMOTE_VENV_DIR}'"

        # Stream venv: docker cp → gzip → ssh → gunzip → tar extract
        # --strip-components=1 removes the ".venv/" prefix
        # --no-same-owner avoids chown errors on network-attached storage (RunPod)
        if ! docker cp "$container_id:/app/.venv" - | gzip | \
                ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" \
                "gunzip | tar xf - -C '${REMOTE_VENV_DIR}' --strip-components=1 --no-same-owner"; then
            docker rm "$container_id" >/dev/null 2>&1 || true
            log_error "Failed to transfer venv to remote."
            return 1
        fi

        docker rm "$container_id" >/dev/null 2>&1 || true

        # Store deps hash on remote to skip transfer next time
        ssh_cmd "echo '${CURRENT_HASH}' > '${REMOTE_VENV_DIR}/.deps_hash'"
        log_done "Venv transferred to remote."
    else
        log_done "Venv on remote is up-to-date — skipping transfer."
    fi
}

# ── Standalone mode ─────────────────────────────────────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
    sync_venv
fi
