#!/usr/bin/env bash
set -euo pipefail

HOSTS="${HOSTS:-macstudio1.local macstudio2.local macstudio3.local macstudio4.local}"

for h in $HOSTS; do
  echo "### stopping on $h"
  ssh "$h" 'pkill -f openai_cluster_server.py || true'
done
