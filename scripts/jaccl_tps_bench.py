import argparse, time
import mlx.core as mx
from mlx_lm.utils import sharded_load

# generate() import differs across mlx-lm branches
try:
    from mlx_lm.utils import generate
except Exception:
    from mlx_lm.generate import generate

def _token_count(tok, text: str) -> int:
    return len(tok.encode(text))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max-tokens", type=int, default=512)
    args = ap.parse_args()

    world = mx.distributed.init()
    rank = world.rank()

    model, tok = sharded_load(args.model)

    # warmup
    _ = generate(model, tok, "hi", max_tokens=8)
    mx.eval()

    t0 = time.time()
    out = generate(model, tok, args.prompt, max_tokens=args.max_tokens)
    mx.eval()
    t1 = time.time()

    prompt_tokens = _token_count(tok, args.prompt)
    out_tokens = _token_count(tok, out)
    gen_tokens = max(out_tokens - prompt_tokens, 1) if out.startswith(args.prompt) else max(out_tokens, 1)

    secs = max(t1 - t0, 1e-9)
    if rank == 0:
        print("==========")
        print(f"model={args.model}")
        print(f"world_size={world.size()}")
        print(f"prompt_tokens={prompt_tokens}")
        print(f"gen_tokens={gen_tokens}")
        print(f"seconds={secs:.3f}")
        print(f"tokens_per_sec={gen_tokens/secs:.3f}")

if __name__ == "__main__":
    main()
