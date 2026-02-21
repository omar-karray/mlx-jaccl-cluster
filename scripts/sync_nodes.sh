#!/usr/bin/env bash
# =============================================================================
# sync_nodes.sh — Pull latest git changes on all cluster nodes in one command
# =============================================================================
# Run this from Mac 1 (rank 0) after pushing changes to GitHub.
# It will git pull on every node in the hostfile in parallel,
# then print a summary of each node's current HEAD.
#
# Usage:
#   ./scripts/sync_nodes.sh
#   HOSTFILE=hostfiles/hosts-2node.json ./scripts/sync_nodes.sh
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"

HOSTFILE="${HOSTFILE:-$REPO_DIR/hostfiles/hosts-2node.json}"
VENV_PYTHON="$REPO_DIR/.venv/bin/python"

# ── Colours ───────────────────────────────────────────────────────────────────
G="\033[32m"
R="\033[31m"
Y="\033[33m"
C="\033[36m"
B="\033[1m"
D="\033[2m"
X="\033[0m"

# ── Validate ──────────────────────────────────────────────────────────────────
if [[ ! -f "$HOSTFILE" ]]; then
  printf "${R}✗${X} Hostfile not found: %s\n" "$HOSTFILE"; exit 1
fi

if [[ ! -f "$VENV_PYTHON" ]]; then
  printf "${R}✗${X} .venv not found — run ./scripts/setup.sh first\n"; exit 1
fi

# ── Parse hosts ───────────────────────────────────────────────────────────────
HOSTS=()
while IFS= read -r _h; do
  [[ -n "$_h" ]] && HOSTS+=("$_h")
done <<< "$("$VENV_PYTHON" -c "
import json
with open('$HOSTFILE') as f:
    hosts = json.load(f)
for h in hosts:
    print(h['ssh'])
")"

REPO_REL=$(python3 -c "import os; print(os.path.relpath('$REPO_DIR', os.path.expanduser('~')))")

# ── Header ────────────────────────────────────────────────────────────────────
printf "\n${B}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${X}\n"
printf "${B}  MLX JACCL — Sync all nodes${X}\n"
printf "${B}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${X}\n"
printf "  Hostfile : ${D}%s${X}\n" "$HOSTFILE"
printf "  Nodes    : ${D}%s${X}\n" "${HOSTS[*]}"
printf "  Repo     : ${D}~/%s${X}\n" "$REPO_REL"
printf "\n"

# ── Temp dir for per-node output ──────────────────────────────────────────────
TMPDIR_SYNC=$(mktemp -d)
trap 'rm -rf "$TMPDIR_SYNC"' EXIT

# ── Pull on all nodes in parallel ─────────────────────────────────────────────
printf "  Pulling...\n\n"

PIDS=()
for i in "${!HOSTS[@]}"; do
  h="${HOSTS[$i]}"
  OUTFILE="$TMPDIR_SYNC/node_${i}.txt"
  (
    ssh -o ConnectTimeout=8 -o BatchMode=yes "$h" \
      "cd \$HOME/$REPO_REL && git pull origin main 2>&1; echo '---HEAD---'; git log --oneline -1" \
      > "$OUTFILE" 2>&1
    echo $? > "$TMPDIR_SYNC/node_${i}.exit"
  ) &
  PIDS+=($!)
done

# Wait for all parallel pulls to finish
for pid in "${PIDS[@]}"; do
  wait "$pid"
done

# ── Print results ─────────────────────────────────────────────────────────────
ISSUES=0
for i in "${!HOSTS[@]}"; do
  h="${HOSTS[$i]}"
  OUTFILE="$TMPDIR_SYNC/node_${i}.txt"
  EXITFILE="$TMPDIR_SYNC/node_${i}.exit"
  EXIT_CODE=$(cat "$EXITFILE" 2>/dev/null || echo "1")

  HEAD=$(grep -A1 "^---HEAD---" "$OUTFILE" 2>/dev/null | tail -1)
  PULL_OUT=$(grep -v "^---HEAD---" "$OUTFILE" 2>/dev/null | grep -v "^$")

  if [[ "$EXIT_CODE" == "0" ]]; then
    # Detect if already up to date or actually updated
    if echo "$PULL_OUT" | grep -q "Already up to date"; then
      printf "  ${G}✓${X}  ${B}%-26s${X} ${D}already up to date${X}\n" "$h"
    else
      printf "  ${G}↓${X}  ${B}%-26s${X} ${G}updated${X}\n" "$h"
      echo "$PULL_OUT" | grep -v "^From\|^remote:" | sed 's/^/       /'
    fi
    printf "       ${D}HEAD: %s${X}\n" "$HEAD"
  else
    printf "  ${R}✗${X}  ${B}%-26s${X} ${R}FAILED${X}\n" "$h"
    cat "$OUTFILE" | sed 's/^/       /'
    ISSUES=$((ISSUES + 1))
  fi
  printf "\n"
done

# ── Verify all nodes on same commit ───────────────────────────────────────────
printf "  Checking commit alignment...\n\n"

COMMITS=()
for i in "${!HOSTS[@]}"; do
  OUTFILE="$TMPDIR_SYNC/node_${i}.txt"
  COMMIT=$(grep -A1 "^---HEAD---" "$OUTFILE" 2>/dev/null | tail -1 | awk '{print $1}')
  COMMITS+=("$COMMIT")
done

FIRST="${COMMITS[0]}"
ALL_SAME=true
for c in "${COMMITS[@]}"; do
  [[ "$c" != "$FIRST" ]] && ALL_SAME=false && break
done

if [[ "$ALL_SAME" == "true" && "$ISSUES" -eq 0 ]]; then
  printf "${B}${G}  ✓ All %d nodes on commit %s${X}\n" "${#HOSTS[@]}" "$FIRST"
else
  if [[ "$ALL_SAME" == "false" ]]; then
    printf "${B}${R}  ✗ Nodes are on different commits:${X}\n"
    for i in "${!HOSTS[@]}"; do
      h="${HOSTS[$i]}"
      printf "      ${Y}%-26s${X} %s\n" "$h" "${COMMITS[$i]:-unknown}"
    done
  fi
  if [[ "$ISSUES" -gt 0 ]]; then
    printf "${B}${R}  ✗ %d node(s) failed to pull${X}\n" "$ISSUES"
  fi
fi

printf "\n${B}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${X}\n\n"
