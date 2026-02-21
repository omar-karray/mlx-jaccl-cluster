#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# setup.sh — One-shot installer for MLX JACCL cluster nodes
# =============================================================================
# Run this script on EACH Mac in the cluster.
# It installs uv (if missing), creates a .venv in the repo root,
# installs all Python dependencies, and prints a full hardware fingerprint.
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

# ── Colours ───────────────────────────────────────────────────────────────────
_c()      { printf "\033[%sm%s\033[0m\n" "$1" "$2"; }
info()    { _c "36"  "  → $*"; }
success() { _c "32"  "  ✓ $*"; }
warn()    { _c "33"  "  ! $*"; }
section() { printf "\n\033[1m%s\033[0m\n" "━━━ $* ━━━"; }
error()   { _c "31"  "  ✗ $*"; exit 1; }
kv()      { printf "  \033[2m%-22s\033[0m %s\n" "$1" "$2"; }

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
        print(f"  \033[32m✓\033[0m  {name:<22} {ver}")
    except ImportError as e:
        print(f"  \033[31m✗\033[0m  {name:<22} MISSING — {e}")
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
  warn "JACCL is a runtime-loaded backend — will confirm when mlx.launch runs"
  success "mlx.launch is present: $MLX_LAUNCH"
fi

# ── Hardware fingerprint ──────────────────────────────────────────────────────
section "Hardware fingerprint"

# --- macOS / chip info via system_profiler ---
HW_MODEL=$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Model Name/{print $2}' | xargs)
HW_CHIP=$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Chip/{print $2}' | xargs)
HW_MEM=$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Memory:/{print $2}' | xargs)
HW_CORES=$(system_profiler SPHardwareDataType 2>/dev/null | awk -F': ' '/Total Number of Cores/{print $2}' | xargs)
HW_MACOS=$(sw_vers -productVersion)
HW_BUILD=$(sw_vers -buildVersion)
HW_HOSTNAME=$(hostname)

kv "Hostname:"     "$HW_HOSTNAME"
kv "Model:"        "${HW_MODEL:-unknown}"
kv "Chip:"         "${HW_CHIP:-unknown}"
kv "Memory:"       "${HW_MEM:-unknown}"
kv "CPU Cores:"    "${HW_CORES:-unknown}"
kv "macOS:"        "$HW_MACOS ($HW_BUILD)"

echo ""

# --- MLX / Metal device info ---
"$VENV_PYTHON" - <<'PYEOF'
import mlx.core as mx

try:
    d = mx.device_info()
    mem_gb   = d["memory_size"]                   / (1024 ** 3)
    wset_gb  = d["max_recommended_working_set_size"] / (1024 ** 3)
    buf_gb   = d["max_buffer_length"]             / (1024 ** 3)
    arch     = d.get("architecture", "unknown")
    gpu_name = d.get("device_name", "unknown")

    kv = lambda k, v: print(f"  \033[2m{k:<22}\033[0m {v}")
    kv("GPU / Neural Engine:", gpu_name)
    kv("Architecture:",        arch)
    kv("Total unified RAM:",   f"{mem_gb:.0f} GB")
    kv("Max working set:",     f"{wset_gb:.1f} GB  (~{wset_gb/mem_gb*100:.0f}% of RAM)")
    kv("Max single buffer:",   f"{buf_gb:.1f} GB")
    kv("MLX version:",         mx.__version__)

    # Compute safe RDMA test sizes based on actual RAM
    # Peak per all_sum = tensor × 3 (input + RDMA buffer + output)
    # Cap at 25% of max working set to leave room for OS + other processes
    safe_bytes   = int(wset_gb * 1024**3 * 0.25)
    safe_elems   = safe_bytes // 4  # float32
    safe_mb      = safe_bytes / (1024**2)

    print("")
    print(f"  \033[2m{'Safe RDMA max tensor:':<22}\033[0m {safe_mb:.0f} MB  ({safe_elems:,} float32 elements)")
    print(f"  \033[2m{'  (25% of working set)':<22}\033[0m peak ~{safe_mb*3/1024:.1f} GB during all_sum")

except Exception as e:
    print(f"  \033[33m! Could not read MLX device info: {e}\033[0m")
PYEOF

# ── RDMA devices ──────────────────────────────────────────────────────────────
section "RDMA devices (Thunderbolt)"

if command -v ibv_devices &>/dev/null; then
  RDMA_DEVS=$(ibv_devices 2>/dev/null | grep -c "rdma_en" || true)
  if [[ "$RDMA_DEVS" -gt 0 ]]; then
    success "Found $RDMA_DEVS RDMA device(s):"
    ibv_devices 2>/dev/null | grep "rdma_en" | sed 's/^/      /'
    echo ""
    info "Port states (PORT_ACTIVE = cable connected and ready):"
    if command -v ibv_devinfo &>/dev/null; then
      ibv_devinfo 2>/dev/null \
        | awk '
            /hca_id/  { dev=$2 }
            /state/   {
              state=$0
              gsub(/.*state:[[:space:]]+/, "", state)
              color = (state ~ /PORT_ACTIVE/) ? "\033[32m" : "\033[2m"
              printf "      %s%-12s\033[0m  %s\n", color, dev, state
            }
          '
    fi
  else
    warn "No rdma_en* devices found."
    warn "→ Boot into macOS Recovery and run: rdma_ctl enable"
    warn "→ Then reboot and re-run this script"
  fi
else
  warn "ibv_devices not found — cannot check RDMA status"
  warn "→ Boot into macOS Recovery and run: rdma_ctl enable"
fi

# ── Write node fingerprint JSON ────────────────────────────────────────────────
section "Saving node fingerprint"

FINGERPRINT_FILE="$REPO_DIR/.node_fingerprint_$(hostname).json"

"$VENV_PYTHON" - <<PYEOF
import json, subprocess, os, sys
from pathlib import Path

def sp(key):
    try:
        out = subprocess.check_output(
            ["system_profiler", "SPHardwareDataType"], text=True, stderr=subprocess.DEVNULL
        )
        for line in out.splitlines():
            if key in line:
                return line.split(":", 1)[-1].strip()
    except Exception:
        pass
    return "unknown"

import mlx.core as mx
d = mx.device_info()

fingerprint = {
    "hostname":      subprocess.check_output(["hostname"], text=True).strip(),
    "model":         sp("Model Name"),
    "chip":          sp("Chip"),
    "memory_gb":     round(d["memory_size"] / (1024**3)),
    "cpu_cores":     sp("Total Number of Cores"),
    "macos":         subprocess.check_output(["sw_vers", "-productVersion"], text=True).strip(),
    "mlx_version":   mx.__version__,
    "gpu":           d.get("device_name", "unknown"),
    "architecture":  d.get("architecture", "unknown"),
    "max_wset_gb":   round(d["max_recommended_working_set_size"] / (1024**3), 1),
    "max_buffer_gb": round(d["max_buffer_length"] / (1024**3), 1),
    "rdma_active":   [],
}

# detect active RDMA ports
try:
    out = subprocess.check_output(["ibv_devinfo"], text=True, stderr=subprocess.DEVNULL)
    dev = None
    for line in out.splitlines():
        if "hca_id" in line:
            dev = line.split()[1]
        if "PORT_ACTIVE" in line and dev:
            fingerprint["rdma_active"].append(dev)
except Exception:
    pass

path = Path("$FINGERPRINT_FILE")
path.write_text(json.dumps(fingerprint, indent=2))
print(f"  \033[32m✓\033[0m  Fingerprint saved → {path.name}")
print(json.dumps(fingerprint, indent=4))
PYEOF

# ── Done ──────────────────────────────────────────────────────────────────────
printf "\n\033[1m\033[32m"
printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
printf "  Setup complete on %s\n" "$(hostname)"
printf "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
printf "\033[0m\n"

kv "Repo:"        "$REPO_DIR"
kv "Virtual env:" "$VENV_DIR"
kv "Python:"      "$($VENV_PYTHON --version)"
kv "mlx.launch:"  "$MLX_LAUNCH"
echo ""
echo "  Next steps:"
echo "    1. Run this script on every other Mac in the cluster"
echo "    2. Run cluster_info.sh to compare both nodes side by side"
echo "    3. Edit hostfiles/hosts-2node.json with your hostnames + IPs"
echo "    4. ./scripts/verify_cluster.sh"
echo "    5. Run the RDMA test (no model needed):"
echo "       .venv/bin/mlx.launch --backend jaccl \\"
echo "         --hostfile hostfiles/hosts-2node.json \\"
echo "         --env MLX_METAL_FAST_SYNCH=1 -- \\"
echo "         python scripts/rdma_test.py"
echo ""
