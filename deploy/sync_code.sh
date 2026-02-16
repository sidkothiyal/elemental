#!/usr/bin/env bash
#
# sync_code.sh — Sync local code to remote via rsync, respecting .gitignore.
#
# Expects (env): REMOTE_HOST, SSH_OPTS, RSYNC_SSH, REMOTE_CODE_DIR, REPO_ROOT
#
# Usage:
#   source deploy/common.sh && source deploy/sync_code.sh
#   sync_code
#

[[ -z "${DEPLOY_DIR:-}" ]] && source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"

sync_code() {
    log_step "Building exclude list from git …"

    local exclude_file
    exclude_file="$(mktemp)"

    # Always exclude .git metadata, local-only files, and remote-managed paths
    cat > "$exclude_file" <<'EXCLUDES'
.git
.deps_hash
.heavy-deps-hash
.venv
.run_session.sh
logs
deploy/logs
EXCLUDES

    # Append every file/directory that git considers ignored
    # (honours nested .gitignore, global excludes, .git/info/exclude)
    git -C "$REPO_ROOT" ls-files --others --ignored --exclude-standard --directory \
        >> "$exclude_file"

    local num_excludes
    num_excludes="$(wc -l < "$exclude_file")"
    echo -e "  ${GREEN}${num_excludes}${NC} exclusion rule(s)."

    log_step "Ensuring remote directory exists …"
    ssh_cmd "mkdir -p '${REMOTE_CODE_DIR}'"

    log_step "Syncing code → ${REMOTE_HOST}:${REMOTE_CODE_DIR}/ …"
    # --no-owner --no-group: don't attempt chown (fails on NAS / RunPod)
    # --omit-dir-times: skip setting dir timestamps (also fails on some NAS mounts)
    # --filter ':- .gitignore': additionally let rsync parse .gitignore files directly
    rsync -avz --progress --delete \
        --no-owner --no-group --omit-dir-times \
        --exclude-from="$exclude_file" \
        --filter=':- .gitignore' \
        -e "$RSYNC_SSH" \
        "$REPO_ROOT/" \
        "${REMOTE_HOST}:${REMOTE_CODE_DIR}/"

    rm -f "$exclude_file"
    log_done "Code synced."
}

# ── Standalone mode ─────────────────────────────────────────────────────────
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    set -euo pipefail
    source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/common.sh"
    sync_code
fi
