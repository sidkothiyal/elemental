#!/usr/bin/env bash
#
# sync_venv.sh — Transfer or link .venv on the remote.
#
# When USE_REGISTRY_VENV is set (provider + ghcr.io image): symlink
# REMOTE_VENV_DIR to /app/.venv from the container image. Otherwise: stream
# .venv from local Docker to remote (docker cp → gzip → ssh → tar).
#
# Expects (env): REMOTE_HOST, SSH_OPTS, REMOTE_CODE_DIR, REMOTE_VENV_DIR,
#                DOCKER_IMAGE, DEPS_CHANGED, CURRENT_HASH
#                USE_REGISTRY_VENV (true = symlink from /app/.venv)
#
# Usage:
#   source deploy/common.sh && source deploy/sync_venv.sh
#   sync_venv
#

[[ -z "${DEPLOY_DIR:-}" ]] && source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

sync_venv() {
    if [[ "${USE_REGISTRY_VENV:-false}" == true ]]; then
        # Instance was created from our ghcr.io image; venv is already at /app/.venv.
        log_step "Linking remote venv to container /app/.venv …"
        ssh_cmd bash <<SYMLINK_EOF
set -e
mkdir -p '${REMOTE_CODE_DIR}'
if [ -L '${REMOTE_VENV_DIR}' ]; then
  target=\$(readlink '${REMOTE_VENV_DIR}')
  if [ "\$target" = '/app/.venv' ]; then
    echo 'Venv already linked to /app/.venv.'
    exit 0
  fi
fi
if [ -d '${REMOTE_VENV_DIR}' ] && [ ! -L '${REMOTE_VENV_DIR}' ]; then
  rm -rf '${REMOTE_VENV_DIR}'
fi
ln -sfn /app/.venv '${REMOTE_VENV_DIR}'
echo 'Linked ${REMOTE_VENV_DIR} -> /app/.venv'
SYMLINK_EOF
        log_done "Venv linked from container image."
        return 0
    fi

    # ── Stream path: transfer .venv from local Docker image to remote ─────
    # Check remote venv hash (ssh_cmd has built-in retry)
    local remote_hash
    remote_hash=$(ssh_cmd "cat '${REMOTE_VENV_DIR}/.deps_hash' 2>/dev/null || echo ''")

    if [[ "$DEPS_CHANGED" == true ]] || [[ "$remote_hash" != "$CURRENT_HASH" ]]; then
        [[ "$remote_hash" != "$CURRENT_HASH" ]] && \
            log_step "Venv not found or outdated on remote — transferring …"
        [[ "$DEPS_CHANGED" == true ]] && \
            log_step "Dependencies changed — transferring updated venv …"

        echo -e "  Extracting .venv from Docker image and streaming to remote …"

        local max_attempts="${SSH_RETRY_ATTEMPTS:-5}"
        local retry_interval="${SSH_RETRY_INTERVAL:-10}"
        local attempt=1
        local transferred=false

        while [[ $attempt -le $max_attempts ]]; do
            # Create a temporary container (never started) to access image filesystem
            local container_id
            container_id=$(docker create "${DOCKER_IMAGE}:latest" true)

            # Prepare remote directory
            if ! ssh_cmd "mkdir -p '${REMOTE_CODE_DIR}' && rm -rf '${REMOTE_VENV_DIR}' && mkdir -p '${REMOTE_VENV_DIR}'"; then
                docker rm "$container_id" >/dev/null 2>&1 || true
                [[ $attempt -lt $max_attempts ]] && { log_warn "Prepare remote dir failed (attempt ${attempt}/${max_attempts}), retrying in ${retry_interval}s …"; sleep "$retry_interval"; attempt=$((attempt + 1)); continue; }
                log_error "Failed to prepare remote directory."
                return 1
            fi

            # Stream venv: docker cp → gzip → ssh → gunzip → tar extract
            # --strip-components=1 removes the ".venv/" prefix
            # --no-same-owner avoids chown errors on network-attached storage (RunPod)
            if docker cp "$container_id:/app/.venv" - | gzip | \
                    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" \
                    "gunzip | tar xf - -C '${REMOTE_VENV_DIR}' --strip-components=1 --no-same-owner"; then
                docker rm "$container_id" >/dev/null 2>&1 || true
                transferred=true
                break
            fi

            docker rm "$container_id" >/dev/null 2>&1 || true
            if [[ $attempt -lt $max_attempts ]]; then
                log_warn "Venv transfer failed (attempt ${attempt}/${max_attempts}), retrying in ${retry_interval}s …"
                sleep "$retry_interval"
            fi
            attempt=$((attempt + 1))
        done

        if [[ "$transferred" != true ]]; then
            log_error "Failed to transfer venv to remote after ${max_attempts} attempts."
            return 1
        fi

        # Store deps hash on remote to skip transfer next time
        ssh_cmd "echo '${CURRENT_HASH}' > '${REMOTE_VENV_DIR}/.deps_hash'"

        # Repair venv Python symlinks so they point to the remote's interpreter.
        # The venv was built in Docker; .venv/bin/python3 and bin/python point at
        # paths that don't exist on the remote, so uv would otherwise remove the
        # venv and reinstall everything on first "uv run".
        log_step "Repairing venv interpreter symlinks on remote …"
        ssh_cmd bash <<FIXUP_EOF
set -e
REMOTE_VENV_DIR='${REMOTE_VENV_DIR}'
REMOTE_PY=\$(command -v python3 || true)
if [[ -z "\$REMOTE_PY" ]]; then
  REMOTE_PY=\$(command -v python || true)
fi
if [[ -n "\$REMOTE_PY" && -d "\$REMOTE_VENV_DIR" ]]; then
  rm -f "\${REMOTE_VENV_DIR}/bin/python3" "\${REMOTE_VENV_DIR}/bin/python"
  ln -sf "\$REMOTE_PY" "\${REMOTE_VENV_DIR}/bin/python3"
  ln -sf python3 "\${REMOTE_VENV_DIR}/bin/python"
  PY_HOME=\$(dirname "\$(dirname "\$REMOTE_PY")")
  PY_VERSION=\$("\$REMOTE_PY" -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' 2>/dev/null || true)
  if [[ -n "\$PY_VERSION" && -f "\${REMOTE_VENV_DIR}/pyvenv.cfg" ]]; then
    sed -i "s|^home = .*|home = \$PY_HOME|" "\${REMOTE_VENV_DIR}/pyvenv.cfg"
    sed -i "s|^version = .*|version = \$PY_VERSION|" "\${REMOTE_VENV_DIR}/pyvenv.cfg"
  fi
fi
FIXUP_EOF
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
