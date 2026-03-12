#!/usr/bin/env bash
#
# setup_creds.sh — Copy credentials (Kaggle, WandB, HuggingFace) to the remote.
#
# Expects (env): REMOTE_HOST, SSH_OPTS, REMOTE_HOME
#
# Usage:
#   source deploy/common.sh && source deploy/setup_creds.sh
#   setup_creds
#

[[ -z "${DEPLOY_DIR:-}" ]] && source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

setup_creds() {
    log_step "Setting up credentials on remote …"

    # ── Kaggle ──────────────────────────────────────────────────────────────
    local local_kaggle="${HOME}/.kaggle/kaggle.json"
    if [[ -f "$local_kaggle" ]]; then
        echo -e "  Copying Kaggle credentials to ${REMOTE_HOME}/.kaggle/ …"
        ssh_cmd "mkdir -p '${REMOTE_HOME}/.kaggle'"
        ssh_cmd \
            "cat > '${REMOTE_HOME}/.kaggle/kaggle.json' && chmod 600 '${REMOTE_HOME}/.kaggle/kaggle.json'" \
            < "$local_kaggle"
        log_done "Kaggle credentials copied."
    else
        log_warn "  ⚠ No local ~/.kaggle/kaggle.json found — skipping."
    fi

    # ── WandB ───────────────────────────────────────────────────────────────
    local wandb_key="${WANDB_API_KEY:-}"
    if [[ -n "$wandb_key" ]]; then
        echo -e "  Setting WANDB_API_KEY in ${REMOTE_HOME}/.bashrc …"
        ssh_cmd bash <<CRED_EOF
if grep -q '^export WANDB_API_KEY=' '${REMOTE_HOME}/.bashrc' 2>/dev/null; then
    sed -i 's|^export WANDB_API_KEY=.*|export WANDB_API_KEY=${wandb_key}|' '${REMOTE_HOME}/.bashrc'
else
    echo 'export WANDB_API_KEY=${wandb_key}' >> '${REMOTE_HOME}/.bashrc'
fi
CRED_EOF
        log_done "WANDB_API_KEY set."
    else
        log_warn "  ⚠ WANDB_API_KEY not set locally — skipping."
    fi

    # ── HuggingFace ─────────────────────────────────────────────────────────
    local hf_token="${HUGGING_FACE_HUB_TOKEN:-}"
    if [[ -n "$hf_token" ]]; then
        echo -e "  Setting HUGGING_FACE_HUB_TOKEN in ${REMOTE_HOME}/.bashrc …"
        ssh_cmd bash <<CRED_EOF
if grep -q '^export HUGGING_FACE_HUB_TOKEN=' '${REMOTE_HOME}/.bashrc' 2>/dev/null; then
    sed -i 's|^export HUGGING_FACE_HUB_TOKEN=.*|export HUGGING_FACE_HUB_TOKEN=${hf_token}|' '${REMOTE_HOME}/.bashrc'
else
    echo 'export HUGGING_FACE_HUB_TOKEN=${hf_token}' >> '${REMOTE_HOME}/.bashrc'
fi
CRED_EOF
        log_done "HUGGING_FACE_HUB_TOKEN set."
    else
        log_warn "  ⚠ HUGGING_FACE_HUB_TOKEN not set locally — skipping."
    fi
}

# ── Standalone mode ─────────────────────────────────────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
    setup_creds
fi
