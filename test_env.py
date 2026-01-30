import os
import json
print("MLX_JACCL_COORDINATOR:", os.environ.get('MLX_JACCL_COORDINATOR', 'NOT SET'))
print("MLX_IBV_DEVICES:", os.environ.get('MLX_IBV_DEVICES', 'NOT SET'))
print("RANK:", os.environ.get('RANK', 'NOT SET'))
import mlx.core as mx
print(f"MLX Version: {mx.__version__}")
