#!/usr/bin/env bash
#
# run.sh — Run Python commands locally (via uv) or on a remote GPU server
#           (via pre-built venv + tmux), with efficient code syncing.
#
# Local mode:  uses uv directly.
# Remote mode: builds a Docker image *locally* (for caching heavy deps like
#              flash-attn), extracts the .venv from it, transfers it to the
#              remote, syncs code via rsync, and runs in a tmux session.
#              No Docker daemon is needed on the remote — just CUDA + Python.
#
# Usage:
#   ./run.sh [OPTIONS] [-- <python args>]
#
# Options:
#   --remote USER@HOST    Run on remote server via venv + tmux
#   -p PORT               SSH port (default: 22)
#   -i IDENTITY_FILE      SSH identity file
#   --docker-root PATH    Remote workspace directory (default: /workspace)
#   --remote-home PATH    Remote home directory for config files (default: /root)
#   --help                Show this help
#
# Examples:
#   # Local — runs: uv run python -m src.train --config configs/default.yaml
#   ./run.sh -- -m src.train --config configs/default.yaml
#
#   # Remote — builds image locally, extracts venv, syncs code, runs in tmux
#   ./run.sh --remote user@gpu-server -- -m src.train --config configs/default.yaml
#   ./run.sh --remote user@gpu-server -p 2222 -- -m src.train --config configs/default.yaml
#   ./run.sh --remote user@gpu-server -i ~/.ssh/key --docker-root /data -- -m src.train

set -euo pipefail

# ── Colour helpers ──────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# ── Defaults ────────────────────────────────────────────────────────────────
REMOTE_HOST=""
SSH_PORT=""
SSH_IDENTITY=""
DOCKER_ROOT="/workspace"
REMOTE_HOME="/root"
PYTHON_ARGS=()

REPO_ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO_NAME="$(basename "$REPO_ROOT")"
DOCKER_IMAGE="elemental"

# ════════════════════════════════════════════════════════════════════════════
# ── Generate dependency hashes (always, for standalone docker build) ───────
# ════════════════════════════════════════════════════════════════════════════

HASH_FILE="${REPO_ROOT}/.deps_hash"
CURRENT_HASH=$(cat "${REPO_ROOT}/pyproject.toml" "${REPO_ROOT}/uv.lock" | md5sum | cut -d' ' -f1)

# Heavy-deps hash controls Docker Layer 1 cache invalidation.
# Only regenerate when the overall deps hash has changed.
if [[ ! -f "$HASH_FILE" ]] || [[ "$(cat "$HASH_FILE")" != "$CURRENT_HASH" ]]; then
    echo -e "${YELLOW}⚙ Generating dependency hashes …${NC}"
    uv run python3 -c "
import tomllib
with open('${REPO_ROOT}/pyproject.toml', 'rb') as f:
    data = tomllib.load(f)
heavy = sorted(str(d) for d in data.get('dependency-groups', {}).get('heavy', []))
print('\n'.join(heavy))
" | md5sum | cut -d' ' -f1 > "${REPO_ROOT}/.heavy-deps-hash"
    echo "$CURRENT_HASH" > "$HASH_FILE"
    DEPS_CHANGED=true
    echo -e "  ${GREEN}Hashes updated.${NC}"
else
    DEPS_CHANGED=false
fi

# ── Usage ───────────────────────────────────────────────────────────────────
usage() {
    cat <<'USAGE'
Usage: ./run.sh [OPTIONS] [-- <python args>]

Run a Python command locally (via uv) or on a remote server (via pre-built venv + tmux).
If no python args are provided, remote setup (sync, venv, credentials) runs without execution.

Options:
  --remote USER@HOST    Run on remote server via venv + tmux
  -p PORT               SSH port (default: 22)
  -i IDENTITY_FILE      SSH identity file
  --docker-root PATH    Remote workspace directory (default: /workspace)
  --remote-home PATH    Remote home directory for config files (default: /root)
  --help                Show this help

Examples:
  # Local:
  ./run.sh -- -m src.train --config configs/default.yaml

  # Remote (setup + run):
  ./run.sh --remote user@gpu-server -- -m src.train --config configs/default.yaml
  ./run.sh --remote user@gpu-server -p 2222 -- -m src.train --config configs/default.yaml

  # Remote (setup only — no execution):
  ./run.sh --remote user@gpu-server
USAGE
    exit 0
}

# ── Parse arguments ─────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --remote)
            [[ $# -lt 2 ]] && { echo -e "${RED}Error:${NC} --remote requires USER@HOST argument."; exit 1; }
            REMOTE_HOST="$2"
            shift 2
            ;;
        -p)
            [[ $# -lt 2 ]] && { echo -e "${RED}Error:${NC} -p requires PORT argument."; exit 1; }
            SSH_PORT="$2"
            shift 2
            ;;
        -i)
            [[ $# -lt 2 ]] && { echo -e "${RED}Error:${NC} -i requires IDENTITY_FILE argument."; exit 1; }
            SSH_IDENTITY="$2"
            shift 2
            ;;
        --docker-root)
            [[ $# -lt 2 ]] && { echo -e "${RED}Error:${NC} --docker-root requires PATH argument."; exit 1; }
            DOCKER_ROOT="$2"
            shift 2
            ;;
        --remote-home)
            [[ $# -lt 2 ]] && { echo -e "${RED}Error:${NC} --remote-home requires PATH argument."; exit 1; }
            REMOTE_HOME="$2"
            shift 2
            ;;
        --help|-h)
            usage
            ;;
        --)
            shift
            PYTHON_ARGS=("$@")
            break
            ;;
        *)
            echo -e "${RED}Error:${NC} Unknown option: $1"
            echo "Use --help for usage information."
            exit 1
            ;;
    esac
done

if [[ ${#PYTHON_ARGS[@]} -eq 0 ]]; then
    echo -e "${YELLOW}Note:${NC} No python arguments provided — will sync/setup only (no remote execution)."
fi

# ── Build SSH options ───────────────────────────────────────────────────────
SSH_OPTS=()
[[ -n "$SSH_PORT" ]] && SSH_OPTS+=(-p "$SSH_PORT")
[[ -n "$SSH_IDENTITY" ]] && SSH_OPTS+=(-i "$SSH_IDENTITY")

RSYNC_SSH="ssh"
[[ -n "$SSH_PORT" ]] && RSYNC_SSH="$RSYNC_SSH -p $SSH_PORT"
[[ -n "$SSH_IDENTITY" ]] && RSYNC_SSH="$RSYNC_SSH -i $SSH_IDENTITY"

# For display in post-run instructions
SSH_DISPLAY=""
[[ -n "$SSH_PORT" ]] && SSH_DISPLAY="$SSH_DISPLAY -p $SSH_PORT"
[[ -n "$SSH_IDENTITY" ]] && SSH_DISPLAY="$SSH_DISPLAY -i $SSH_IDENTITY"

# ════════════════════════════════════════════════════════════════════════════
# ── Local mode ──────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

if [[ -z "$REMOTE_HOST" ]]; then
    if [[ ${#PYTHON_ARGS[@]} -eq 0 ]]; then
        echo -e "${GREEN}✔ Nothing to do locally (no python args).${NC}"
        exit 0
    fi
    echo -e "${GREEN}▶ Running locally via uv …${NC}"
    cd "$REPO_ROOT"
    exec uv run python "${PYTHON_ARGS[@]}"
fi

# ════════════════════════════════════════════════════════════════════════════
# ── Remote mode ─────────────────────────────────────────────────────────────
# ════════════════════════════════════════════════════════════════════════════

REMOTE_PROJECT_DIR="${DOCKER_ROOT}/${REPO_NAME}"
REMOTE_VENV_DIR="${REMOTE_PROJECT_DIR}/.venv"
REMOTE_LOGS_DIR="${DOCKER_ROOT}/logs/${REPO_NAME}"

echo -e "${BLUE}Remote host:${NC}        ${REMOTE_HOST}"
echo -e "${BLUE}Remote project:${NC}     ${REMOTE_PROJECT_DIR}"
echo -e "${BLUE}Remote venv:${NC}        ${REMOTE_VENV_DIR}"
echo -e "${BLUE}Remote logs:${NC}        ${REMOTE_LOGS_DIR}"
echo -e "${BLUE}Remote home:${NC}        ${REMOTE_HOME}"
echo ""

# ── 1. Build Docker image locally if dependencies changed ──────────────────
# The Docker image is only used as a build environment to produce the .venv.
# It is never run on the remote — only the extracted venv is transferred.

if [[ "$DEPS_CHANGED" == true ]]; then
    echo -e "${YELLOW}⚙ Dependencies changed — rebuilding Docker image locally …${NC}"
    DOCKER_BUILDKIT=1 docker build -t "${DOCKER_IMAGE}:latest" "${REPO_ROOT}"
    echo -e "${GREEN}✔ Image built.${NC}"
else
    echo -e "${GREEN}✔ Dependencies unchanged — skipping build.${NC}"
fi

# ── 2. Transfer .venv into remote project dir if needed ──────────────────────
# The .venv is placed inside the project directory to mirror the local structure.
# We store a deps hash inside the remote .venv to know which version is deployed.
# Transfer only when: deps changed OR remote hash doesn't match.
# Strategy: create a temporary container, stream .venv via docker cp → gzip
# → ssh → tar extract.  No Docker daemon needed on the remote.

REMOTE_VENV_HASH=$(ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" \
    "cat '${REMOTE_VENV_DIR}/.deps_hash' 2>/dev/null || echo ''")

if [[ "$DEPS_CHANGED" == true ]] || [[ "$REMOTE_VENV_HASH" != "$CURRENT_HASH" ]]; then
    [[ "$REMOTE_VENV_HASH" != "$CURRENT_HASH" ]] && \
        echo -e "${YELLOW}⚙ Venv not found or outdated on remote — transferring …${NC}"
    [[ "$DEPS_CHANGED" == true ]] && \
        echo -e "${YELLOW}⚙ Dependencies changed — transferring updated venv …${NC}"

    echo -e "  Extracting .venv from Docker image and streaming to remote …"

    # Create a temporary container (never started) to access the image filesystem
    CONTAINER_ID=$(docker create "${DOCKER_IMAGE}:latest" true)

    # Prepare remote directory (create project dir if it doesn't exist yet)
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" \
        "mkdir -p '${REMOTE_PROJECT_DIR}' && rm -rf '${REMOTE_VENV_DIR}' && mkdir -p '${REMOTE_VENV_DIR}'"

    # Stream venv directly: docker cp → gzip → ssh → gunzip → tar extract
    # --strip-components=1 removes the ".venv/" prefix from the tar archive
    # --no-same-owner avoids chown errors on network-attached storage (RunPod)
    if ! docker cp "$CONTAINER_ID:/app/.venv" - | gzip | \
            ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" \
            "gunzip | tar xf - -C '${REMOTE_VENV_DIR}' --strip-components=1 --no-same-owner"; then
        docker rm "$CONTAINER_ID" >/dev/null 2>&1 || true
        echo -e "${RED}Error:${NC} Failed to transfer venv to remote."
        exit 1
    fi

    docker rm "$CONTAINER_ID" >/dev/null 2>&1 || true

    # Store the deps hash on the remote so we can skip transfer next time
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" \
        "echo '${CURRENT_HASH}' > '${REMOTE_VENV_DIR}/.deps_hash'"

    echo -e "${GREEN}✔ Venv transferred to remote.${NC}"
else
    echo -e "${GREEN}✔ Venv on remote is up-to-date — skipping transfer.${NC}"
fi

echo ""

# ── 3. Sync code via rsync (respecting .gitignore) ─────────────────────────

echo -e "${YELLOW}⚙ Building exclude list from git …${NC}"

EXCLUDE_FILE="$(mktemp)"
trap 'rm -f "$EXCLUDE_FILE"' EXIT

# Always exclude .git metadata, local-only files, and remote-managed paths
echo ".git" > "$EXCLUDE_FILE"
echo ".deps_hash" >> "$EXCLUDE_FILE"
echo ".heavy-deps-hash" >> "$EXCLUDE_FILE"
echo ".venv" >> "$EXCLUDE_FILE"
echo ".run_session.sh" >> "$EXCLUDE_FILE"
echo "logs" >> "$EXCLUDE_FILE"

# Append every file/directory that git considers ignored
# (honours nested .gitignore, global excludes, .git/info/exclude)
git -C "$REPO_ROOT" ls-files --others --ignored --exclude-standard --directory \
    >> "$EXCLUDE_FILE"


NUM_EXCLUDES="$(wc -l < "$EXCLUDE_FILE")"
echo -e "  ${GREEN}${NUM_EXCLUDES}${NC} exclusion rule(s)."

echo -e "${YELLOW}⚙ Ensuring remote directory exists …${NC}"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "mkdir -p '${REMOTE_PROJECT_DIR}'"

echo -e "${YELLOW}⚙ Syncing code → ${REMOTE_HOST}:${REMOTE_PROJECT_DIR}/ …${NC}"
# --no-owner --no-group: don't attempt chown (fails on network-attached storage / RunPod)
# --omit-dir-times: skip setting dir timestamps (also fails on some NAS mounts)
# --filter ':- .gitignore': additionally let rsync parse .gitignore files directly
rsync -avz --progress --delete \
    --no-owner --no-group --omit-dir-times \
    --exclude-from="$EXCLUDE_FILE" \
    --filter=':- .gitignore' \
    -e "$RSYNC_SSH" \
    "$REPO_ROOT/" \
    "${REMOTE_HOST}:${REMOTE_PROJECT_DIR}/"

echo -e "${GREEN}✔ Code synced.${NC}"
echo ""

# ── 4. Setup credentials on remote ─────────────────────────────────────────

echo -e "${YELLOW}⚙ Setting up credentials on remote …${NC}"

# Copy Kaggle credentials if local kaggle.json exists
LOCAL_KAGGLE_JSON="${HOME}/.kaggle/kaggle.json"
if [[ -f "$LOCAL_KAGGLE_JSON" ]]; then
    echo -e "  Copying Kaggle credentials to ${REMOTE_HOME}/.kaggle/ …"
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" "mkdir -p '${REMOTE_HOME}/.kaggle'"
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" \
        "cat > '${REMOTE_HOME}/.kaggle/kaggle.json' && chmod 600 '${REMOTE_HOME}/.kaggle/kaggle.json'" \
        < "$LOCAL_KAGGLE_JSON"
    echo -e "  ${GREEN}✔ Kaggle credentials copied.${NC}"
else
    echo -e "  ${YELLOW}⚠ No local ~/.kaggle/kaggle.json found — skipping.${NC}"
fi

# Copy WANDB_API_KEY to remote .bashrc if set locally
LOCAL_WANDB_KEY="${WANDB_API_KEY:-}"
if [[ -n "$LOCAL_WANDB_KEY" ]]; then
    echo -e "  Setting WANDB_API_KEY in ${REMOTE_HOME}/.bashrc …"
    ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" bash <<CRED_EOF
if grep -q '^export WANDB_API_KEY=' '${REMOTE_HOME}/.bashrc' 2>/dev/null; then
    sed -i 's|^export WANDB_API_KEY=.*|export WANDB_API_KEY=${LOCAL_WANDB_KEY}|' '${REMOTE_HOME}/.bashrc'
else
    echo 'export WANDB_API_KEY=${LOCAL_WANDB_KEY}' >> '${REMOTE_HOME}/.bashrc'
fi
CRED_EOF
    echo -e "  ${GREEN}✔ WANDB_API_KEY set.${NC}"
else
    echo -e "  ${YELLOW}⚠ WANDB_API_KEY not set locally — skipping.${NC}"
fi

echo ""

# ── 5. Ensure uv is available on remote ────────────────────────────────────

echo -e "${YELLOW}⚙ Ensuring uv is available on remote …${NC}"
ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" bash <<UV_EOF
export HOME='${REMOTE_HOME}'
if command -v uv >/dev/null 2>&1 || [[ -x '${REMOTE_HOME}/.local/bin/uv' ]]; then
    echo 'uv is already installed.'
else
    echo 'Installing uv …'
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
UV_EOF
echo -e "${GREEN}✔ uv ready.${NC}"
echo ""

# ── 6. Run with uv in a tmux session on remote ─────────────────────────────
# No Docker needed on remote — we use uv run with the pre-built .venv.
# RunPod pods already have CUDA drivers, GPU access, and system libraries.

if [[ ${#PYTHON_ARGS[@]} -eq 0 ]]; then
    echo -e "${GREEN}✔ Setup complete — no python args provided, skipping execution.${NC}"
    exit 0
fi

ESCAPED_ARGS=$(printf '%q ' "${PYTHON_ARGS[@]}")
SESSION_NAME="train-$(date +%s)"

# Build optional WANDB export line for the run script
WANDB_EXPORT_LINE=""
if [[ -n "$LOCAL_WANDB_KEY" ]]; then
    WANDB_EXPORT_LINE="export WANDB_API_KEY=${LOCAL_WANDB_KEY}"
fi

echo -e "${YELLOW}⚙ Starting tmux session ${SESSION_NAME} on remote …${NC}"

ssh "${SSH_OPTS[@]}" "$REMOTE_HOST" bash <<REMOTE_EOF
set -euo pipefail

mkdir -p '${REMOTE_LOGS_DIR}'

# Write a run script so tmux has a clean command to execute
cat > '${REMOTE_PROJECT_DIR}/.run_session.sh' <<INNER_EOF
#!/bin/bash
export PATH="${REMOTE_HOME}/.local/bin:\\\$PATH"
export HOME="${REMOTE_HOME}"
${WANDB_EXPORT_LINE}
cd '${REMOTE_PROJECT_DIR}'

# Ensure logs dir exists as symlink or real dir
mkdir -p '${REMOTE_LOGS_DIR}'
[ -e logs ] || ln -sfn '${REMOTE_LOGS_DIR}' logs

uv run python ${ESCAPED_ARGS} 2>&1 | tee '${REMOTE_LOGS_DIR}/${SESSION_NAME}.log'
INNER_EOF
chmod +x '${REMOTE_PROJECT_DIR}/.run_session.sh'

tmux new-session -d -s '${SESSION_NAME}' '${REMOTE_PROJECT_DIR}/.run_session.sh'
REMOTE_EOF

echo ""
echo -e "${GREEN}✔ Remote session started.${NC}"
echo ""
echo -e "  ${BLUE}Attach:${NC}   ssh${SSH_DISPLAY} ${REMOTE_HOST} -t 'tmux attach -t ${SESSION_NAME}'"
echo -e "  ${BLUE}Logs:${NC}     ssh${SSH_DISPLAY} ${REMOTE_HOST} 'tail -f ${REMOTE_LOGS_DIR}/${SESSION_NAME}.log'"
echo -e "  ${BLUE}Sessions:${NC} ssh${SSH_DISPLAY} ${REMOTE_HOST} 'tmux ls'"
