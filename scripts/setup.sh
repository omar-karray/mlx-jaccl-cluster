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
# Run on Mac 2 remotely from Mac 1:
#   ssh mac2.local "cd ~/path/to/mlx-jaccl-cluster && ./scripts/setup.sh"
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

# ── Checks ───────────────────────────────────────────────────────────────────
section "System checks"

# macOS only
if [[ "$(uname)" != "Darwin" ]]; then
  error "This project requires macOS (Apple Silicon)."
fi

# Apple Silicon only
if [[ "$(uname -m)" != "arm64" ]]; then
  error "This project requires Apple Silicon (arm64). Got: $(uname -m)"
fi

info "macOS $(sw_vers -productVersion) on $(uname -m)"

# ── Install uv if missing ────────────────────────────────────────────────────
section "uv"

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

# ── Create / update virtualenv ───────────────────────────────────────────────
section "Python virtual environment"

if [[ -d "$VENV_DIR" ]]; then
  warn ".venv already exists at $VENV_DIR — will reuse it"
else
  info "Creating .venv with Python 3.12..."
  # Python 3.12 is the sweet spot: >=3.10 (MLX req), stable, well-tested
  uv venv "$VENV_DIR" --python 3.12
  success ".venv created at $VENV_DIR"
fi

# ── Install dependencies ─────────────────────────────────────────────────────
section "Installing Python dependencies"

info "Installing MLX + mlx-lm..."
uv pip install --python "$VENV_DIR" \
  "mlx>=0.30.4" \
  "mlx-lm>=0.30.5"

info "Installing server dependencies..."
uv pip install --python "$VENV_DIR" \
  "fastapi>=0.110.0" \
  "uvicorn[standard]>=0.29.0" \
  "pydantic>=2.0"

info "Installing tokenizer / HuggingFace dependencies..."
uv pip install --python "$VENV_DIR" \
  "transformers>=4.50.0" \
  "tokenizers" \
  "mistral_common" \
  "huggingface_hub[cli]"

# ── Verify key installs ──────────────────────────────────────────────────────
section "Verification"

"$VENV_DIR/bin/python" - <<'PYEOF'
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

# ── Check mlx.launch / JACCL ─────────────────────────────────────────────────
section "mlx.launch JACCL support"

if "$VENV_DIR/bin/python" -m mlx --help 2>&1 | grep -qi "jaccl"; then
  success "mlx.launch supports jaccl backend"
else
  warn "Could not confirm jaccl backend — may need a newer mlx version"
  warn "Run: uv pip install --python .venv 'mlx>=0.30.4' to upgrade"
fi

# ── RDMA check ───────────────────────────────────────────────────────────────
section "RDMA devices (Thunderbolt)"

if command -v ibv_devices &>/dev/null; then
  RDMA_DEVS=$(ibv_devices 2>/dev/null | grep -c "rdma_en" || true)
  if [[ "$RDMA_DEVS" -gt 0 ]]; then
    success "Found $RDMA_DEVS RDMA device(s):"
    ibv_devices 2>/dev/null | grep "rdma_en" | sed 's/^/    /'
  else
    warn "No rdma_en* devices found."
    warn "→ Boot into macOS Recovery and run: rdma_ctl enable"
  fi
else
  warn "ibv_devices not found — cannot check RDMA status."
  warn "→ Boot into macOS Recovery and run: rdma_ctl enable"
fi

# ── Done ─────────────────────────────────────────────────────────────────────
printf "\n\033[1m\033[32m"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Setup complete on $(hostname)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
printf "\033[0m\n"

echo "  Virtual env : $VENV_DIR"
echo "  Activate    : source $VENV_DIR/bin/activate"
echo "  mlx.launch  : $VENV_DIR/bin/mlx.launch"
echo ""
echo "  Next steps:"
echo "    1. Run this script on every other Mac in the cluster"
echo "    2. Edit hostfiles/hosts-2node.json with your hostnames + IPs"
echo "    3. ./scripts/verify_cluster.sh"
echo "    4. ./scripts/rdma_test.py  (RDMA bandwidth test)"
echo ""
