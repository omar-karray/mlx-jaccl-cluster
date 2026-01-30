# Session Notes - MLX Cluster Debugging

## Current Status (2026-01-30)

### Working Configuration
- **Backend**: `ring` (TCP/IP over Thunderbolt)
- **Model**: Qwen3-4B-Instruct-2507-4bit
- **Speed**: ~12.8 tokens/sec on 4-node cluster

```bash
# Start server (uses ring backend by default now)
/Users/alex/Code/mlx-jaccl-cluster/scripts/run_openai_cluster_server.sh

# Test request
curl -s http://localhost:8080/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"Qwen3-4B-Instruct-2507-4bit","messages":[{"role":"user","content":"Hello"}],"max_tokens":100}'
```

### RDMA/JACCL Issue (Unresolved)

**Problem**: `[jaccl] Couldn't allocate protection domain` error

**Root Cause**: RDMA ports showing `PORT_DOWN` on all nodes:
- macstudio1: ALL ports DOWN
- macstudio2: en3,4,5 ACTIVE (but peers are DOWN)
- macstudio3: only en4 ACTIVE
- macstudio4: en3,5 ACTIVE

No node pair has BOTH sides with PORT_ACTIVE status.

**Verification**:
```bash
# Check RDMA port status
for host in macstudio1.local macstudio2.local macstudio3.local macstudio4.local; do
  echo "=== $host ==="
  ssh $host 'ibv_devinfo 2>&1 | grep -E "hca_id|state" | head -12'
done

# Network interfaces are active (ping works), RDMA layer is not
```

**Workaround**: Use `ring` backend instead of `jaccl`
- Set `BACKEND=ring` in run_openai_cluster_server.sh (already done)
- Lower performance than RDMA but stable

**To try later**:
1. Re-cable Thunderbolt connections
2. Reboot all nodes simultaneously
3. Check Apple developer forums for RDMA issues
4. File bug report with Apple

---

## Previous Issue: Kimi-K2.5 Hanging

- Model: 182 shards, ~600GB
- Worked initially at ~26 tokens/sec
- Hung after 1-2 requests, GPU stuck at 100%

**Suspected causes** (not yet tested with ring backend):
1. Double `mx.distributed.init()` calls
2. `mx.eval(model.parameters())` loading entire model at once
3. Deadlock in worker synchronization

**To test**: Run Kimi-K2.5 with ring backend after RDMA issues resolved

---

## Stop server gracefully
```bash
pkill -TERM -f "openai_cluster_server"
# Or on all nodes:
for h in macstudio1.local macstudio2.local macstudio3.local macstudio4.local; do
  ssh $h 'pkill -TERM -f python' || true
done
```
