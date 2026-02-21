#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# cluster_info.sh — Side-by-side node alignment report
# =============================================================================
# SSHes into every node in the hostfile, collects hardware + software info,
# and prints a unified table. Highlights any version mismatches in red.
#
# Usage:
#   ./scripts/cluster_info.sh
#   HOSTFILE=hostfiles/hosts-2node.json ./scripts/cluster_info.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

HOSTFILE="${HOSTFILE:-$REPO_DIR/hostfiles/hosts-2node.json}"
VENV_PYTHON="$REPO_DIR/.venv/bin/python"

# ── Colours ───────────────────────────────────────────────────────────────────
GREEN="\033[32m"
RED="\033[31m"
YELLOW="\033[33m"
CYAN="\033[36m"
BOLD="\033[1m"
DIM="\033[2m"
RESET="\033[0m"

ok()   { printf "${GREEN}✓${RESET} %s\n" "$*"; }
warn() { printf "${YELLOW}!${RESET} %s\n" "$*"; }
err()  { printf "${RED}✗${RESET} %s\n" "$*"; }
sep()  { printf "${DIM}%s${RESET}\n" "────────────────────────────────────────────────────────────────────────────"; }

# ── Validate inputs ───────────────────────────────────────────────────────────
if [[ ! -f "$HOSTFILE" ]]; then
  err "Hostfile not found: $HOSTFILE"
  exit 1
fi

if [[ ! -f "$VENV_PYTHON" ]]; then
  err ".venv not found at $REPO_DIR/.venv"
  err "Run: ./scripts/setup.sh first"
  exit 1
fi

# ── Parse hostfile ────────────────────────────────────────────────────────────
HOSTS=$("$VENV_PYTHON" -c "
import json
with open('$HOSTFILE') as f:
    hosts = json.load(f)
print(' '.join(h['ssh'] for h in hosts))
")

NUM_HOSTS=$(echo "$HOSTS" | wc -w | tr -d ' ')

# ── Probe script — runs on each remote node via SSH ───────────────────────────
# Outputs key=value lines for easy parsing
PROBE_SCRIPT='
set -euo pipefail
REPO="$HOME/'"$(python3 -c "import os; print(os.path.relpath('$REPO_DIR', os.path.expanduser('~')))")"'"
VENV="$REPO/.venv/bin/python"

# Helper: silent fallback
sp() {
  system_profiler SPHardwareDataType 2>/dev/null \
    | awk -F": " -v key="$1" '$0 ~ key {print $2}' \
    | head -1 | xargs
}

echo "hostname=$(hostname)"
echo "model=$(sp "Model Name")"
echo "chip=$(sp "Chip")"
echo "memory=$(sp "Memory")"
echo "cores=$(sp "Total Number of Cores")"
echo "macos=$(sw_vers -productVersion)"
echo "build=$(sw_vers -buildVersion)"

if [[ -f "$VENV" ]]; then
  "$VENV" - <<PYEOF
import mlx.core as mx, sys

try:
    d = mx.device_info()
    mem_gb  = d["memory_size"] / (1024**3)
    wset_gb = d["max_recommended_working_set_size"] / (1024**3)
    buf_gb  = d["max_buffer_length"] / (1024**3)
    print(f"mlx_version={mx.__version__}")
    print(f"gpu={d.get('device_name','unknown')}")
    print(f"arch={d.get('architecture','unknown')}")
    print(f"memory_gb={mem_gb:.0f}")
    print(f"wset_gb={wset_gb:.1f}")
    print(f"buf_gb={buf_gb:.1f}")
    print(f"python={sys.version.split()[0]}")
except Exception as e:
    print(f"mlx_error={e}")
PYEOF
else
  echo "mlx_version=NOT_INSTALLED"
  echo "python=NOT_INSTALLED"
fi

# RDMA
if command -v ibv_devices &>/dev/null; then
  DEVS=$(ibv_devices 2>/dev/null | grep -c "rdma_en" || true)
  ACTIVE=$(ibv_devinfo 2>/dev/null | awk "/hca_id/{dev=\$2} /PORT_ACTIVE/{print dev}" | tr "\n" "," | sed "s/,$//")
  echo "rdma_devices=$DEVS"
  echo "rdma_active=${ACTIVE:-none}"
else
  echo "rdma_devices=0"
  echo "rdma_active=none"
fi

# mlx.launch
if [[ -f "$REPO/.venv/bin/mlx.launch" ]]; then
  echo "mlx_launch=present"
else
  echo "mlx_launch=MISSING"
fi
'

# ── Collect data from all nodes ───────────────────────────────────────────────
printf "\n${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
printf "${BOLD}  MLX JACCL Cluster — Node Alignment Report${RESET}\n"
printf "${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n"
printf "  Hostfile : ${DIM}%s${RESET}\n" "$HOSTFILE"
printf "  Nodes    : ${DIM}%s${RESET}\n" "$HOSTS"
printf "\n"

# Store raw data per node in associative arrays
declare -A NODE_DATA

echo "  Probing nodes..."
for h in $HOSTS; do
  printf "  ${DIM}→ %-20s${RESET} " "$h"
  if DATA=$(ssh -o ConnectTimeout=6 -o BatchMode=yes "$h" bash -s <<< "$PROBE_SCRIPT" 2>/dev/null); then
    NODE_DATA["$h"]="$DATA"
    printf "${GREEN}OK${RESET}\n"
  else
    NODE_DATA["$h"]="hostname=$h
error=SSH_FAILED"
    printf "${RED}FAILED${RESET}\n"
  fi
done

echo ""

# ── Helper: extract value from a node's data ─────────────────────────────────
get() {
  local host="$1" key="$2"
  echo "${NODE_DATA[$host]}" | grep "^${key}=" | cut -d'=' -f2- | head -1
}

# ── Build side-by-side table ──────────────────────────────────────────────────
# Column widths
COL_KEY=22
COL_VAL=24

# Header
printf "${BOLD}${CYAN}%-${COL_KEY}s${RESET}" "Property"
for h in $HOSTS; do
  HN=$(get "$h" "hostname")
  printf " ${BOLD}${CYAN}%-${COL_VAL}s${RESET}" "${HN:-$h}"
done
echo ""
sep

# ── Rows ──────────────────────────────────────────────────────────────────────
print_row() {
  local label="$1" key="$2" check_match="${3:-true}"

  printf "${DIM}%-${COL_KEY}s${RESET}" "$label"

  # Collect values
  declare -a vals=()
  for h in $HOSTS; do
    vals+=("$(get "$h" "$key")")
  done

  # Check if all values match
  local first="${vals[0]}"
  local all_match=true
  for v in "${vals[@]}"; do
    [[ "$v" != "$first" ]] && all_match=false && break
  done

  # Print each value with colour
  for v in "${vals[@]}"; do
    if [[ "$v" == "NOT_INSTALLED" || "$v" == "MISSING" || "$v" == "SSH_FAILED" ]]; then
      printf " ${RED}%-${COL_VAL}s${RESET}" "$v"
    elif [[ "$check_match" == "true" && "$all_match" == "false" ]]; then
      printf " ${YELLOW}%-${COL_VAL}s${RESET}" "$v"
    else
      printf " ${GREEN}%-${COL_VAL}s${RESET}" "$v"
    fi
  done

  # Mismatch warning
  if [[ "$check_match" == "true" && "$all_match" == "false" ]]; then
    printf "  ${YELLOW}⚠ MISMATCH${RESET}"
  fi

  echo ""
}

# Hardware
print_row "Model"           "model"    false
print_row "Chip"            "chip"     true
print_row "Memory"          "memory"   true
print_row "CPU Cores"       "cores"    false
sep

# Software
print_row "macOS"           "macos"    true
print_row "macOS Build"     "build"    true
print_row "Python"          "python"   true
print_row "MLX version"     "mlx_version"  true
print_row "GPU"             "gpu"      true
print_row "Architecture"    "arch"     true
sep

# Memory / compute capacity
print_row "Unified RAM (GB)"   "memory_gb"   true
print_row "Max working set GB" "wset_gb"     true
print_row "Max buffer GB"      "buf_gb"      true
sep

# RDMA / cluster
print_row "RDMA devices"    "rdma_devices"  false
print_row "RDMA active"     "rdma_active"   false
print_row "mlx.launch"      "mlx_launch"    true
sep

# ── Alignment verdict ─────────────────────────────────────────────────────────
echo ""
printf "${BOLD}  Alignment verdict${RESET}\n"
echo ""

ISSUES=0

check_align() {
  local label="$1" key="$2"
  declare -a vals=()
  for h in $HOSTS; do
    vals+=("$(get "$h" "$key")")
  done
  local first="${vals[0]}"
  local ok=true
  for v in "${vals[@]}"; do
    [[ "$v" != "$first" ]] && ok=false && break
  done
  if [[ "$ok" == "true" ]]; then
    printf "  ${GREEN}✓${RESET}  %-30s ${DIM}%s${RESET}\n" "$label" "$first"
  else
    printf "  ${RED}✗${RESET}  %-30s " "$label"
    for h in $HOSTS; do
      HN=$(get "$h" "hostname")
      V=$(get "$h" "$key")
      printf "${YELLOW}%s${RESET}=${RED}%s${RESET}  " "${HN:-$h}" "$V"
    done
    echo ""
    ISSUES=$((ISSUES + 1))
  fi
}

check_align "macOS version"         "macos"
check_align "MLX version"           "mlx_version"
check_align "Python version"        "python"
check_align "Chip"                  "chip"
check_align "Unified memory"        "memory_gb"
check_align "mlx.launch"           "mlx_launch"

echo ""

if [[ "$ISSUES" -eq 0 ]]; then
  printf "${BOLD}${GREEN}  ✓ All nodes are aligned — cluster is ready to run.${RESET}\n"
else
  printf "${BOLD}${RED}  ✗ %d alignment issue(s) found — resolve before running inference.${RESET}\n" "$ISSUES"
  echo ""
  echo "  Common fixes:"
  echo "    - MLX mismatch  → run ./scripts/setup.sh on the outdated node"
  echo "    - macOS mismatch → update via System Settings → Software Update"
  echo "    - Python mismatch → rm -rf .venv && ./scripts/setup.sh"
fi

printf "\n${BOLD}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${RESET}\n\n"
