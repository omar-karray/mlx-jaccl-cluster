import os
import json
ibv_file = os.environ.get('MLX_IBV_DEVICES')
print(f"MLX_IBV_DEVICES points to: {ibv_file}")
if ibv_file and os.path.exists(ibv_file):
    with open(ibv_file) as f:
        content = f.read()
        print(f"Contents: {content}")
else:
    print("File not found or not set")
