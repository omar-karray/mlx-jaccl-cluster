#!/usr/bin/env bash
set -euo pipefail

HOSTS="${HOSTS:-macstudio1.local macstudio2.local macstudio3.local macstudio4.local}"

echo "== SSH check =="
for h in $HOSTS; do
  echo "### $h"
  ssh -o ConnectTimeout=5 "$h" 'hostname'
done

echo
echo "== RDMA devices (rdma_en3/4/5) =="
for h in $HOSTS; do
  echo "### $h"
  ssh "$h" 'ibv_devices | egrep "rdma_en3|rdma_en4|rdma_en5" || echo "(missing rdma_en3/4/5)"'
done
