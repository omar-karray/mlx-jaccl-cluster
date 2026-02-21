#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# bootstrap_node.sh — Bootstrap a remote cluster node from Mac 1
# =============================================================================
# Run this script ON MAC 1 to fully set up Mac 2 (or any other node):
#   1. SSH into the remote Mac
#   2. Install Homebrew if missing
#   3. Install git + uv if missing
#   4. Clone THIS repo (from GitHub) to the SAME path as on Mac 1
#   5. Run setup.sh on the remote Mac
#
# Usage:
#   ./scripts/bootstrap_node.sh <remote-host>
#
# Examples:
#   ./scripts/bootstrap_node.sh mac2.local
#   ./scripts/bootstrap_node.sh 192.168.0.50
#
# Requirements on Mac 1:
#   - SSH access to the remote host (key-based auth recommended)
#   - This repo must be pushed to GitHub (origin remote)
# =============================================================================

# ── Colours ──────────────────────────────────────────────────────────────────
_c()      { printf "\033[%sm%s\033[0m\n" "$1" "$2"; }
info()    { _c "36"  "  → $*"; }
success() { _c "32"  "  ✓ $*"; }
warn()    { _c "33"  "  ! $*"; }
section() { printf "\n\033[1m%s\033[0m\n" "━━━ $* ━━━"; }
error()   { _c "31"  "  ✗ $*"; exit 1; }

# ── Args ─────────────────────────────────────────────────────────────────────
if [[ $# -lt 1 ]]; then
  echo ""
  echo "  Usage: ./scripts/bootstrap_node.sh <remote-host>"
  echo ""
  echo "  Examples:"
  echo "    ./scripts/bootstrap_node.sh mac2.local"
  echo "    ./scripts/bootstrap_node.sh 192.168.0.50"
  echo ""
  exit 1
fi

REMOTE_HOST="$1"

# ── Resolve local repo info ───────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

# The path we will clone into on the remote Mac (same as local)
REMOTE_REPO_PATH="$REPO_DIR"

# Get the GitHub repo URL from the local origin remote
REPO_URL=$(git -C "$REPO_DIR" remote get-url origin 2>/dev/null || echo "")
if [[ -z "$REPO_URL" ]]; then
  error "Could not determine repo URL from 'git remote get-url origin'."
fi

# ── Print plan ────────────────────────────────────────────────────────────────
echo ""
printf "\033[1m%s\033[0m\n" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "\033[1m%s\033[0m\n" "  MLX JACCL — Remote Node Bootstrap"
printf "\033[1m%s\033[0m\n" "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
info "Remote host  : $REMOTE_HOST"
info "Repo URL     : $REPO_URL"
info "Clone path   : $REMOTE_REPO_PATH"
echo ""

# ── Check SSH connectivity ────────────────────────────────────────────────────
section "SSH connectivity"

if ssh -o ConnectTimeout=5 -o BatchMode=yes "$REMOTE_HOST" 'echo ok' &>/dev/null; then
  success "SSH connection to $REMOTE_HOST OK"
else
  error "Cannot SSH into $REMOTE_HOST. Make sure:
    - The remote Mac is reachable
    - SSH is enabled (System Settings → General → Sharing → Remote Login)
    - Your SSH key is authorized on the remote Mac
    - Run: ssh-copy-id $REMOTE_HOST"
fi

# ── Remote bootstrap script ───────────────────────────────────────────────────
# We pass all variables as env vars and run a heredoc over SSH.
# This way there is no temp file to clean up.

section "Running remote bootstrap on $REMOTE_HOST"

ssh "$REMOTE_HOST" \
  REPO_URL="$REPO_URL" \
  REMOTE_REPO_PATH="$REMOTE_REPO_PATH" \
  bash -s << 'REMOTE_SCRIPT'

set -euo pipefail

# ── Colours (same as local) ────────────────────────────────────────────────
_c()      { printf "\033[%sm%s\033[0m\n" "$1" "$2"; }
info()    { _c "36"  "  → $*"; }
success() { _c "32"  "  ✓ $*"; }
warn()    { _c "33"  "  ! $*"; }
section() { printf "\n\033[1m%s\033[0m\n" "━━━ $* ━━━"; }
error()   { _c "31"  "  ✗ $*"; exit 1; }

echo ""
printf "\033[1m  Remote host : %s\033[0m\n" "$(hostname)"
echo ""

# ── Sanity: must be Apple Silicon macOS ───────────────────────────────────
section "System checks"

if [[ "$(uname)" != "Darwin" ]]; then
  error "Remote must be macOS."
fi
if [[ "$(uname -m)" != "arm64" ]]; then
  error "Remote must be Apple Silicon (arm64). Got: $(uname -m)"
fi
success "macOS $(sw_vers -productVersion) on arm64"

# ── Homebrew ──────────────────────────────────────────────────────────────
section "Homebrew"

# Make sure brew is on PATH (it may not be in a non-interactive SSH session)
if ! command -v brew &>/dev/null; then
  if [[ -x /opt/homebrew/bin/brew ]]; then
    eval "$(/opt/homebrew/bin/brew shellenv)"
  fi
fi

if command -v brew &>/dev/null; then
  success "Homebrew already installed: $(brew --version | head -1)"
else
  info "Homebrew not found — installing..."
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
  eval "$(/opt/homebrew/bin/brew shellenv)"
  success "Homebrew installed"
fi

# ── git ───────────────────────────────────────────────────────────────────
section "git"

if ! command -v git &>/dev/null; then
  info "git not found — installing via Homebrew..."
  brew install git
fi
success "git $(git --version | awk '{print $3}')"

# ── uv ────────────────────────────────────────────────────────────────────
section "uv"

# uv may be installed at ~/.local/bin which may not be in SSH PATH
export PATH="$HOME/.local/bin:$PATH"

if ! command -v uv &>/dev/null; then
  info "uv not found — installing via Homebrew..."
  brew install uv
fi
success "uv $(uv --version)"

# ── Clone or update repo ───────────────────────────────────────────────────
section "Repository"

PARENT_DIR="$(dirname "$REMOTE_REPO_PATH")"

if [[ -d "$REMOTE_REPO_PATH/.git" ]]; then
  warn "Repo already exists at $REMOTE_REPO_PATH — pulling latest..."
  git -C "$REMOTE_REPO_PATH" pull --ff-only
  success "Repo updated"
else
  info "Cloning $REPO_URL"
  info "  → $REMOTE_REPO_PATH"
  mkdir -p "$PARENT_DIR"
  git clone "$REPO_URL" "$REMOTE_REPO_PATH"
  success "Repo cloned"
fi

# ── Run setup.sh ───────────────────────────────────────────────────────────
section "Running setup.sh"

chmod +x "$REMOTE_REPO_PATH/scripts/setup.sh"
"$REMOTE_REPO_PATH/scripts/setup.sh"

REMOTE_SCRIPT

# ── Done ─────────────────────────────────────────────────────────────────────
echo ""
printf "\033[1m\033[32m"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Bootstrap complete on $REMOTE_HOST"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "\033[0m\n"
echo "  Next steps:"
echo "    1. Edit hostfiles/hosts-2node.json with your hostnames + IPs"
echo "    2. ./scripts/verify_cluster.sh"
echo "    3. uv run mlx.launch --backend jaccl \\"
echo "         --hostfile hostfiles/hosts-2node.json \\"
echo "         --env MLX_METAL_FAST_SYNCH=1 -- \\"
echo "         python scripts/rdma_test.py"
echo ""
