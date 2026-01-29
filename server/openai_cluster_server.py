#!/usr/bin/env python3
import os
import time
import json
import socket
import struct
import threading
import asyncio
import uuid
from typing import Optional, Union

import mlx.core as mx
from mlx_lm.utils import sharded_load

# generate() import differs across mlx-lm branches
try:
    from mlx_lm.utils import generate
except Exception:
    from mlx_lm.generate import generate

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# -------------------------
# Configuration (env vars)
# -------------------------
MODEL_DIR = os.environ["MODEL_DIR"]  # REQUIRED
MODEL_ID = os.environ.get("MODEL_ID", os.path.basename(MODEL_DIR.rstrip("/")))

HOST = os.environ.get("HOST", "0.0.0.0")      # HTTP bind on rank0
PORT = int(os.environ.get("PORT", "8080"))    # HTTP port on rank0

# Control-plane (rank0 <-> workers) for coordinating "everyone call generate()"
CTRL_PORT = int(os.environ.get("CTRL_PORT", "18080"))

def _default_ctrl_host() -> str:
    c = os.environ.get("MLX_JACCL_COORDINATOR", "")
    if ":" in c:
        return c.split(":", 1)[0]
    return "macstudio1.local"

CTRL_HOST = os.environ.get("CTRL_HOST", _default_ctrl_host())

DEFAULT_MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))

# Backpressure / queueing
QUEUE_MAX = int(os.environ.get("QUEUE_MAX", "8"))          # max queued requests
REQ_TIMEOUT = float(os.environ.get("REQ_TIMEOUT", "120"))  # per request timeout (seconds)

# -------------------------
# Globals
# -------------------------
app = FastAPI()
_model = None
_tok = None
_world = None

_queue: asyncio.Queue = asyncio.Queue(maxsize=QUEUE_MAX)  # rank0 only uses it

# -------------------------
# Tiny framed JSON protocol
# -------------------------
def _recvall(sock: socket.socket, n: int) -> Optional[bytes]:
    buf = b""
    while len(buf) < n:
        chunk = sock.recv(n - len(buf))
        if not chunk:
            return None
        buf += chunk
    return buf

def send_msg(sock: socket.socket, obj: dict) -> None:
    data = json.dumps(obj).encode("utf-8")
    sock.sendall(struct.pack("!I", len(data)))
    sock.sendall(data)

def recv_msg(sock: socket.socket) -> Optional[dict]:
    hdr = _recvall(sock, 4)
    if hdr is None:
        return None
    (n,) = struct.unpack("!I", hdr)
    body = _recvall(sock, n)
    if body is None:
        return None
    return json.loads(body.decode("utf-8"))

# -------------------------
# OpenAI-ish schemas
# -------------------------
class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionsReq(BaseModel):
    model: Optional[str] = None
    messages: list[ChatMessage]
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

class CompletionsReq(BaseModel):
    model: Optional[str] = None
    prompt: Union[str, list[str]]
    max_tokens: Optional[int] = None
    stream: Optional[bool] = False

def _build_chat_prompt(messages: list[ChatMessage]) -> str:
    # Prefer tokenizer chat template when available
    if hasattr(_tok, "apply_chat_template"):
        msgs = [{"role": m.role, "content": m.content} for m in messages]
        return _tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)

    # Fallback: simple "ROLE: content" format
    parts = [f"{m.role.upper()}: {m.content}" for m in messages]
    parts.append("ASSISTANT:")
    return "\n".join(parts)

def _tok_len(text: str) -> int:
    return len(_tok.encode(text))

# -------------------------
# Rank0 worker connections
# -------------------------
_worker_socks: dict[int, socket.socket] = {}  # rank -> socket
_worker_lock = threading.Lock()

def rank0_accept_workers(expected_world_size: int) -> None:
    """
    Rank0 listens for worker control-plane connections.
    Each worker sends {"type":"hello","rank":N}.
    """
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((HOST, CTRL_PORT))
    srv.listen(16)
    print(f"[rank0] control-plane listening on {HOST}:{CTRL_PORT}", flush=True)

    while True:
        conn, addr = srv.accept()
        hello = recv_msg(conn)
        if not hello or hello.get("type") != "hello" or "rank" not in hello:
            conn.close()
            continue
        r = int(hello["rank"])
        with _worker_lock:
            _worker_socks[r] = conn
        print(f"[rank0] worker connected rank={r} from {addr}", flush=True)

def rank0_wait_for_workers(expected_world_size: int, timeout_s: int = 60) -> bool:
    t0 = time.time()
    while True:
        with _worker_lock:
            ok = all(r in _worker_socks for r in range(1, expected_world_size))
        if ok:
            print("[rank0] all workers connected", flush=True)
            return True
        if time.time() - t0 > timeout_s:
            return False
        time.sleep(0.1)

def rank0_broadcast_task(task: dict) -> None:
    """
    Send the same task to all worker ranks (1..N-1).
    """
    with _worker_lock:
        items = list(_worker_socks.items())
    for r, s in items:
        send_msg(s, {"type": "task", **task})

def rank0_wait_done(expected_world_size: int) -> None:
    """
    Wait for {"type":"done"} from all workers.
    """
    done: set[int] = set()
    while len(done) < (expected_world_size - 1):
        with _worker_lock:
            items = list(_worker_socks.items())
        for r, s in items:
            if r in done:
                continue
            s.settimeout(0.2)
            try:
                msg = recv_msg(s)
            except Exception:
                msg = None
            if msg and msg.get("type") == "done":
                done.add(r)

# -------------------------
# Worker loop
# -------------------------
def worker_loop(rank: int) -> None:
    """
    Workers connect to rank0 control-plane, block waiting for tasks.
    For each task: call generate() (so collectives match rank0), then send done.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect((CTRL_HOST, CTRL_PORT))
    send_msg(s, {"type": "hello", "rank": rank})
    print(f"[worker {rank}] connected to control-plane {CTRL_HOST}:{CTRL_PORT}", flush=True)

    while True:
        msg = recv_msg(s)
        if not msg:
            continue
        if msg.get("type") != "task":
            continue

        prompt = msg["prompt"]
        max_tokens = int(msg["max_tokens"])

        _ = generate(_model, _tok, prompt, max_tokens=max_tokens)
        mx.eval()
        send_msg(s, {"type": "done", "rank": rank})

# -------------------------
# Queue worker (rank0 only)
# -------------------------
async def _queue_worker() -> None:
    """
    Processes queued requests sequentially.
    Each request triggers:
      - broadcast task to workers
      - rank0 generate()
      - wait for worker completion
      - fulfill per-request future with an OpenAI-shaped response
    """
    while True:
        item = await _queue.get()
        if item is None:
            _queue.task_done()
            continue

        kind, prompt, max_t, fut = item  # kind: "chat" | "completions"
        try:
            rank0_broadcast_task({"prompt": prompt, "max_tokens": max_t})

            t0 = time.time()
            out_text = generate(_model, _tok, prompt, max_tokens=max_t)
            mx.eval()
            t1 = time.time()

            rank0_wait_done(_world.size())

            completion = out_text[len(prompt):] if out_text.startswith(prompt) else out_text
            pt = _tok_len(prompt)
            ct = _tok_len(completion)

            timing = {
                "seconds": round(t1 - t0, 3),
                "tokens_per_sec": round(ct / max(t1 - t0, 1e-9), 3),
            }

            if kind == "chat":
                resp = {
                    "id": f"chatcmpl-{uuid.uuid4().hex[:24]}",
                    "object": "chat.completion",
                    "created": int(time.time()),
                    "model": MODEL_ID,
                    "choices": [{
                        "index": 0,
                        "message": {"role": "assistant", "content": completion},
                        "finish_reason": "stop",
                    }],
                    "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
                    "timing": timing,
                }
            elif kind == "completions":
                resp = {
                    "id": f"cmpl-{uuid.uuid4().hex[:24]}",
                    "object": "text_completion",
                    "created": int(time.time()),
                    "model": MODEL_ID,
                    "choices": [{
                        "index": 0,
                        "text": completion,
                        "finish_reason": "stop",
                        "logprobs": None,
                    }],
                    "usage": {"prompt_tokens": pt, "completion_tokens": ct, "total_tokens": pt + ct},
                    "timing": timing,
                }
            else:
                raise RuntimeError(f"Unknown request kind: {kind}")

            fut.set_result(resp)

        except Exception as e:
            fut.set_exception(e)
        finally:
            _queue.task_done()

@app.on_event("startup")
async def _startup() -> None:
    # Only rank0 runs the HTTP server, so only rank0 starts the queue worker
    if _world and _world.rank() == 0:
        asyncio.create_task(_queue_worker())

# -------------------------
# HTTP endpoints (rank0 only)
# -------------------------
@app.get("/health")
def health() -> dict:
    return {
        "ok": True,
        "world_size": _world.size(),
        "rank": _world.rank(),
        "model": MODEL_ID,
        "queue_max": QUEUE_MAX,
        "queue_size": _queue.qsize(),
    }

@app.get("/v1/models")
def list_models() -> dict:
    return {"object": "list", "data": [{"id": MODEL_ID, "object": "model"}]}

@app.get("/queue")
def queue_status() -> dict:
    return {"size": _queue.qsize(), "max": QUEUE_MAX}

@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionsReq) -> dict:
    if req.stream:
        raise HTTPException(status_code=400, detail="stream=true not supported")
    if req.model and req.model != MODEL_ID:
        raise HTTPException(status_code=400, detail=f"Only model '{MODEL_ID}' is served")

    if _world.rank() != 0:
        raise HTTPException(status_code=500, detail="Rank != 0 received HTTP request")

    prompt = _build_chat_prompt(req.messages)
    max_t = req.max_tokens or DEFAULT_MAX_TOKENS

    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()

    try:
        _queue.put_nowait(("chat", prompt, max_t, fut))
    except asyncio.QueueFull:
        raise HTTPException(status_code=429, detail="Server busy (queue full). Try again later.")

    try:
        return await asyncio.wait_for(fut, timeout=REQ_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out")

@app.post("/v1/completions")
async def completions(req: CompletionsReq) -> dict:
    if req.stream:
        raise HTTPException(status_code=400, detail="stream=true not supported")
    if req.model and req.model != MODEL_ID:
        raise HTTPException(status_code=400, detail=f"Only model '{MODEL_ID}' is served")

    if _world.rank() != 0:
        raise HTTPException(status_code=500, detail="Rank != 0 received HTTP request")

    if isinstance(req.prompt, list):
        # Keep it simple + safe for distributed mode: one prompt at a time.
        if len(req.prompt) != 1:
            raise HTTPException(status_code=400, detail="Only a single prompt string is supported (prompt must be a string, or a list of length 1).")
        prompt = req.prompt[0]
    else:
        prompt = req.prompt

    max_t = req.max_tokens or DEFAULT_MAX_TOKENS

    loop = asyncio.get_running_loop()
    fut: asyncio.Future = loop.create_future()

    try:
        _queue.put_nowait(("completions", prompt, max_t, fut))
    except asyncio.QueueFull:
        raise HTTPException(status_code=429, detail="Server busy (queue full). Try again later.")

    try:
        return await asyncio.wait_for(fut, timeout=REQ_TIMEOUT)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out")

# -------------------------
# Main
# -------------------------
def main() -> None:
    global _model, _tok, _world
    _world = mx.distributed.init()
    _model, _tok = sharded_load(MODEL_DIR)

    if _world.rank() == 0:
        th = threading.Thread(target=rank0_accept_workers, args=(_world.size(),), daemon=True)
        th.start()

        if not rank0_wait_for_workers(_world.size(), timeout_s=60):
            raise RuntimeError("Workers did not connect to control-plane in time")

        uvicorn.run(app, host=HOST, port=PORT, log_level="info")
    else:
        worker_loop(_world.rank())

if __name__ == "__main__":
    main()

