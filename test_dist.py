import os
import sys
print(f"Starting on rank env: {os.environ.get('RANK', 'not set')}", flush=True)
print(f"World size env: {os.environ.get('WORLD_SIZE', 'not set')}", flush=True)

import mlx.core as mx
print(f"MLX imported, attempting distributed init...", flush=True)
try:
    world = mx.distributed.init()
    print(f"SUCCESS: Rank {world.rank()} of {world.size()}", flush=True)
except Exception as e:
    print(f"FAILED: {e}", flush=True)
    sys.exit(1)
