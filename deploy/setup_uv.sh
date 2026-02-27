#!/usr/bin/env bash
#
# setup_uv.sh — Ensure uv is installed on the remote.
#
# Expects (env): REMOTE_HOST, SSH_OPTS, REMOTE_HOME
#
# Usage:
#   source deploy/common.sh && source deploy/setup_uv.sh
#   setup_uv
#

[[ -z "${DEPLOY_DIR:-}" ]] && source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

setup_uv() {
    log_step "Ensuring uv is available on remote …"
    ssh_cmd bash <<UV_EOF
export HOME='${REMOTE_HOME}'
if command -v uv >/dev/null 2>&1 || [[ -x '${REMOTE_HOME}/.local/bin/uv' ]]; then
    echo 'uv is already installed.'
else
    echo 'Installing uv …'
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
# Ensure UV_LINK_MODE=copy for interactive/manual uv run (avoids hardlink warning
# when cache and workspace are on different filesystems, e.g. overlayfs/NFS).
if [[ -f "\${HOME}/.bashrc" ]] && ! grep -q UV_LINK_MODE "\${HOME}/.bashrc" 2>/dev/null; then
    echo 'export UV_LINK_MODE=copy' >> "\${HOME}/.bashrc"
fi
UV_EOF
    log_done "uv ready."
}

# ── Standalone mode ─────────────────────────────────────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
    setup_uv
fi
