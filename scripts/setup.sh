#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# setup.sh — One-shot installer for MLX JACCL cluster nodes
# =============================================================================
# Run this script on EACH Mac in the cluster.
# It installs uv (if missing), creates a .venv in the repo root,
# and installs all Python dependencies.
#
# Usage:
#   ./scripts/setup.sh
#
# Run on Mac 2 remotely from Mac 1 (after cloning the repo):
#   ssh mac2 "cd ~/path/to/mlx-jaccl-cluster && ./scripts/setup.sh"
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VENV_DIR="$REPO_DIR/.venv"

# ── Colours ──────────────────────────────────────────────────────────────────
_c() { printf "\033[%sm%s\033[0m\n" "$1" "$2"; }
info()    { _c "36"   "  → $*"; }
success() { _c "32"   "  ✓ $*"; }
warn()    { _c "33"   "  ! $*"; }
section() { printf "\n\033[1m%s\033[0m\n" "━━━ $* ━━━"; }
error()   { _c "31"   "  ✗ $*"; exit 1; }

# ── System checks ─────────────────────────────────────────────────────────────
section "System checks"

if [[ "$(uname)" != "Darwin" ]]; then
  error "This project requires macOS (Apple Silicon)."
fi

if [[ "$(uname -m)" != "arm64" ]]; then
  error "This project requires Apple Silicon (arm64). Got: $(uname -m)"
fi

info "macOS $(sw_vers -productVersion) on $(uname -m) — $(hostname)"

# ── uv ────────────────────────────────────────────────────────────────────────
section "uv"

# uv may live at ~/.local/bin which is absent in non-interactive SSH sessions
export PATH="$HOME/.local/bin:/opt/homebrew/bin:$PATH"

if command -v uv &>/dev/null; then
  success "uv already installed: $(uv --version)"
else
  info "uv not found — installing via Homebrew..."
  if ! command -v brew &>/dev/null; then
    error "Homebrew not found. Install it first: https://brew.sh"
  fi
  brew install uv
  success "uv installed: $(uv --version)"
fi

# ── Python virtualenv ─────────────────────────────────────────────────────────
section "Python virtual environment"

if [[ -d "$VENV_DIR" ]]; then
  warn ".venv already exists at $VENV_DIR — reusing"
else
  info "Creating .venv with Python 3.12..."
  uv venv "$VENV_DIR" --python 3.12
  success ".venv created at $VENV_DIR"
fi

VENV_PYTHON="$VENV_DIR/bin/python"
VENV_PIP_INSTALL="uv pip install --python $VENV_PYTHON"

# ── Dependencies ──────────────────────────────────────────────────────────────
section "Installing Python dependencies"

info "MLX + mlx-lm..."
$VENV_PIP_INSTALL \
  "mlx>=0.30.4" \
  "mlx-lm>=0.30.5"

info "Server dependencies (FastAPI, uvicorn, pydantic)..."
$VENV_PIP_INSTALL \
  "fastapi>=0.110.0" \
  "uvicorn[standard]>=0.29.0" \
  "pydantic>=2.0"

info "Tokenizer + HuggingFace dependencies..."
$VENV_PIP_INSTALL \
  "transformers>=4.50.0" \
  "tokenizers" \
  "mistral_common" \
  "huggingface_hub"

# ── Verify imports ────────────────────────────────────────────────────────────
section "Verification"

"$VENV_PYTHON" - <<'PYEOF'
import importlib, sys

checks = [
    ("mlx",             "mlx.core"),
    ("mlx-lm",          "mlx_lm"),
    ("fastapi",         "fastapi"),
    ("uvicorn",         "uvicorn"),
    ("transformers",    "transformers"),
    ("huggingface_hub", "huggingface_hub"),
]

ok = True
for name, mod in checks:
    try:
        m = importlib.import_module(mod)
        ver = getattr(m, "__version__", "?")
        print(f"  \033[32m✓\033[0m  {name:<20} {ver}")
    except ImportError as e:
        print(f"  \033[31m✗\033[0m  {name:<20} MISSING — {e}")
        ok = False

if not ok:
    sys.exit(1)
PYEOF

# ── mlx.launch JACCL check ────────────────────────────────────────────────────
section "mlx.launch JACCL support"

MLX_LAUNCH="$VENV_DIR/bin/mlx.launch"

if [[ ! -f "$MLX_LAUNCH" ]]; then
  warn "mlx.launch not found at $MLX_LAUNCH"
  warn "Try: uv pip install --python $VENV_PYTHON 'mlx>=0.30.4'"
elif "$MLX_LAUNCH" --help 2>&1 | grep -qi "jaccl"; then
  success "mlx.launch supports jaccl backend ✓"
else
  warn "mlx.launch found but jaccl backend not listed in --help"
  warn "You may need a newer mlx build with JACCL support"
fi

# ── RDMA devices ──────────────────────────────────────────────────────────────
section "RDMA devices (Thunderbolt)"

if command -v ibv_devices &>/dev/null; then
  RDMA_DEVS=$(ibv_devices 2>/dev/null | grep -c "rdma_en" || true)
  if [[ "$RDMA_DEVS" -gt 0 ]]; then
    success "Found $RDMA_DEVS RDMA device(s):"
    ibv_devices 2>/dev/null | grep "rdma_en" | sed 's/^/      /'
    echo ""
    info "Active ports (PORT_ACTIVE = cable connected):"
    ibv_devinfo 2>/dev/null \
      | awk '/hca_id/{dev=$2} /state/{print "      " dev " → " $0}' \
      | grep -v "PORT_DOWN" || info "  (no active ports yet — normal before mlx.launch runs)"
  else
    warn "No rdma_en* devices found."
    warn "→ Boot into macOS Recovery and run: rdma_ctl enable"
    warn "→ Then reboot and re-run this script"
  fi
else
  warn "ibv_devices not found — cannot check RDMA status"
  warn "→ Boot into macOS Recovery and run: rdma_ctl enable"
fi

# ── Done ──────────────────────────────────────────────────────────────────────
printf "\n\033[1m\033[32m"
printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
printf "  Setup complete on %s\n" "$(hostname)"
printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
printf "\033[0m\n"

echo "  Repo        : $REPO_DIR"
echo "  Virtual env : $VENV_DIR"
echo "  Python      : $($VENV_PYTHON --version)"
echo "  mlx.launch  : $MLX_LAUNCH"
echo ""
echo "  Next steps:"
echo "    1. Run this script on every other Mac in the cluster"
echo "    2. Edit hostfiles/hosts-2node.json with your hostnames + IPs"
echo "    3. ./scripts/verify_cluster.sh"
echo "    4. Run the RDMA test (no model needed):"
echo "         uv run --python $VENV_PYTHON mlx.launch \\"
echo "           --backend jaccl \\"
echo "           --hostfile hostfiles/hosts-2node.json \\"
echo "           --env MLX_METAL_FAST_SYNCH=1 -- \\"
echo "           python scripts/rdma_test.py"
echo ""
