#!/usr/bin/env bash
#
# run_remote.sh — Launch a command in a tmux session on the remote.
#
# Creates a run script on the remote and starts it inside a new tmux session.
# The command is run as-is (e.g. "uv run python ..." or "script.sh"); no Docker on remote.
#
# Expects (env): REMOTE_HOST, SSH_OPTS, SSH_DISPLAY, REMOTE_CODE_DIR,
#                REMOTE_LOGS_DIR, REMOTE_HOME, REPO_NAME, PYTHON_ARGS (array = command)
# Sets (globals): SESSION_NAME
#
# Usage:
#   source deploy/common.sh && source deploy/run_remote.sh
#   PYTHON_ARGS=(uv run python -m src.train --config configs/default.yaml)
#   run_remote
#

[[ -z "${DEPLOY_DIR:-}" ]] && source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

run_remote() {
    if [[ ${#PYTHON_ARGS[@]} -eq 0 ]]; then
        log_done "Setup complete — no command provided, skipping execution."
        return 0
    fi

    # Embed command for bash -c: join args with space, escape only single quotes so the
    # remote script can safely wrap in single quotes. Do not use printf '%q' — it wraps
    # the whole string in quotes, which then become part of the -c argument and make bash
    # try to exec the entire string as a program name (No such file or directory).
    local raw_cmd
    raw_cmd="${PYTHON_ARGS[*]}"
    local remote_cmd
    remote_cmd=$(printf '%s' "$raw_cmd" | sed "s/'/'\\\\''/g")
    SESSION_NAME="train-$(date +%s)"

    # Build optional WANDB export line for the run script
    local wandb_export=""
    [[ -n "${WANDB_API_KEY:-}" ]] && wandb_export="export WANDB_API_KEY=${WANDB_API_KEY}"

    log_step "Starting tmux session ${SESSION_NAME} on remote …"

    ssh_cmd bash <<REMOTE_EOF
set -euo pipefail

mkdir -p '${REMOTE_LOGS_DIR}'

# Write a run script so tmux has a clean command to execute
cat > '${REMOTE_CODE_DIR}/.run_session.sh' <<INNER_EOF
#!/bin/bash
export PATH="${REMOTE_HOME}/.local/bin:\\\$PATH"
export HOME="${REMOTE_HOME}"
# Use copy mode so uv does not try to hardlink from cache (fails when cache and
# workspace are on different filesystems, e.g. overlayfs or NFS).
export UV_LINK_MODE=copy
${wandb_export}
cd '${REMOTE_CODE_DIR}'

# Expose Python headers so torch.compile / triton can find Python.h
_py_include=\\\$(uv run python -c 'import sysconfig; print(sysconfig.get_path("include"))')
export CFLAGS="-I\\\${_py_include} \\\${CFLAGS:-}"
export CPATH="\\\${_py_include}\\\${CPATH:+:\\\$CPATH}"

# Ensure logs dir exists as symlink or real dir
mkdir -p '${REMOTE_LOGS_DIR}'
[ -e logs ] || ln -sfn '${REMOTE_LOGS_DIR}' logs

bash -c '${remote_cmd}' 2>&1 | tee '${REMOTE_LOGS_DIR}/${SESSION_NAME}.log'
INNER_EOF
chmod +x '${REMOTE_CODE_DIR}/.run_session.sh'

tmux new-session -d -s '${SESSION_NAME}' '${REMOTE_CODE_DIR}/.run_session.sh'
REMOTE_EOF

    echo ""
    log_done "Remote session started."
    echo ""
    echo -e "  ${BLUE}Attach:${NC}   ssh${SSH_DISPLAY} ${REMOTE_HOST} -t 'tmux attach -t ${SESSION_NAME}'"
    echo -e "  ${BLUE}Logs:${NC}     ssh${SSH_DISPLAY} ${REMOTE_HOST} 'tail -f ${REMOTE_LOGS_DIR}/${SESSION_NAME}.log'"
    echo -e "  ${BLUE}Sessions:${NC} ssh${SSH_DISPLAY} ${REMOTE_HOST} 'tmux ls'"
}

# ── Standalone mode ─────────────────────────────────────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
    run_remote
fi
