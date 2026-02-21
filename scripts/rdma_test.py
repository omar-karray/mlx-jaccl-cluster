#!/usr/bin/env python3
"""
rdma_test.py — Minimal RDMA connectivity & bandwidth test for MLX JACCL clusters.

No model download required. Tests RDMA by running MLX distributed collective
operations (all_sum) over Thunderbolt RDMA between all ranks.

Usage (via mlx.launch):
    mlx.launch --backend jaccl \
      --hostfile hostfiles/hosts-2node.json \
      --env MLX_METAL_FAST_SYNCH=1 -- \
      python scripts/rdma_test.py

Optional env vars:
    RDMA_ROUNDS      Number of benchmark rounds    (default: 20)
    RDMA_SIZES       Comma-separated tensor sizes  (default: 1024,65536,1048576,16777216)
    RDMA_VERBOSE     Set to 1 for per-round timing (default: 0)
"""

import os
import struct
import sys
import time

import mlx.core as mx

# ─── Config ──────────────────────────────────────────────────────────────────
ROUNDS = int(os.environ.get("RDMA_ROUNDS", "20"))
SIZES_ENV = os.environ.get("RDMA_SIZES", "1024,65536,1048576,16777216")
SIZES = [int(s.strip()) for s in SIZES_ENV.split(",")]
VERBOSE = os.environ.get("RDMA_VERBOSE", "0") == "1"

# Bytes per float32 element
BYTES_PER_ELEM = 4

# ─── ANSI colours (disabled if not a tty) ────────────────────────────────────
_IS_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _IS_TTY else text


GREEN = lambda t: _c("32", t)
YELLOW = lambda t: _c("33", t)
CYAN = lambda t: _c("36", t)
BOLD = lambda t: _c("1", t)
RED = lambda t: _c("31", t)
DIM = lambda t: _c("2", t)


# ─── Helpers ─────────────────────────────────────────────────────────────────


def _sync_barrier(world: mx.distributed.Group) -> None:
    """Global barrier: all ranks must arrive before any continue."""
    x = mx.ones((1,), dtype=mx.float32)
    result = mx.distributed.all_sum(x, group=world)
    mx.eval(result)


def _human_bytes(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.2f} TB"


def _human_bw(bytes_per_sec: float) -> str:
    gb = bytes_per_sec / 1e9
    if gb >= 1.0:
        return f"{gb:.2f} GB/s"
    mb = bytes_per_sec / 1e6
    return f"{mb:.1f} MB/s"


# ─── Test 1: Correctness ─────────────────────────────────────────────────────


def test_correctness(world: mx.distributed.Group, rank: int, size: int) -> None:
    """
    Each rank fills a tensor with (rank+1). After all_sum the expected value
    is sum(1..world_size) = world_size*(world_size+1)//2 on every element.
    """
    world_size = world.size()
    expected = float(world_size * (world_size + 1) // 2)

    tensor = mx.full((size,), float(rank + 1), dtype=mx.float32)
    result = mx.distributed.all_sum(tensor, group=world)
    mx.eval(result)

    actual = float(result[0])
    passed = abs(actual - expected) < 1e-3

    if rank == 0:
        status = GREEN("PASS ✓") if passed else RED("FAIL ✗")
        label = _human_bytes(size * BYTES_PER_ELEM)
        print(
            f"  Correctness [{label:>8}]: {status}  "
            f"(expected {expected:.1f}, got {actual:.1f})"
        )

    if not passed:
        raise RuntimeError(
            f"Rank {rank}: correctness check FAILED — expected {expected}, got {actual}"
        )


# ─── Test 2: Latency (small tensor round-trip) ───────────────────────────────


def test_latency(world: mx.distributed.Group, rank: int) -> None:
    """
    Measure round-trip latency of a tiny all_sum (1 element = 4 bytes).
    """
    WARMUP = 5
    ITERS = 50

    tensor = mx.ones((1,), dtype=mx.float32)

    # Warmup
    for _ in range(WARMUP):
        r = mx.distributed.all_sum(tensor, group=world)
        mx.eval(r)

    _sync_barrier(world)
    t0 = time.perf_counter()
    for _ in range(ITERS):
        r = mx.distributed.all_sum(tensor, group=world)
        mx.eval(r)
    t1 = time.perf_counter()

    elapsed_us = (t1 - t0) / ITERS * 1e6

    if rank == 0:
        status = (
            GREEN("excellent")
            if elapsed_us < 100
            else YELLOW("good")
            if elapsed_us < 500
            else RED("slow")
        )
        print(f"  Latency (1 elem / 4 B):  {CYAN(f'{elapsed_us:.1f} µs')}  [{status}]")


# ─── Test 3: Bandwidth sweep ─────────────────────────────────────────────────


def test_bandwidth(
    world: mx.distributed.Group,
    rank: int,
    num_elems: int,
) -> float:
    """
    all_sum benchmark for a given tensor size. Returns achieved GB/s (rank 0).
    """
    WARMUP = 3

    tensor = mx.ones((num_elems,), dtype=mx.float32)

    # Warmup
    for _ in range(WARMUP):
        r = mx.distributed.all_sum(tensor, group=world)
        mx.eval(r)

    _sync_barrier(world)
    times = []
    for i in range(ROUNDS):
        t0 = time.perf_counter()
        r = mx.distributed.all_sum(tensor, group=world)
        mx.eval(r)
        t1 = time.perf_counter()
        dt = t1 - t0
        times.append(dt)
        if VERBOSE and rank == 0:
            bw = (num_elems * BYTES_PER_ELEM) / dt
            print(DIM(f"    round {i + 1:02d}: {dt * 1e3:.2f} ms  {_human_bw(bw)}"))

    times.sort()
    # Drop top 10% outliers
    trim = max(1, int(len(times) * 0.9))
    trimmed = times[:trim]
    avg_dt = sum(trimmed) / len(trimmed)
    min_dt = min(trimmed)

    payload = num_elems * BYTES_PER_ELEM
    avg_bw = payload / avg_dt
    peak_bw = payload / min_dt

    if rank == 0:
        label = _human_bytes(payload)
        print(
            f"  BW [{label:>8}]:  "
            f"avg {CYAN(_human_bw(avg_bw)):>14}   "
            f"peak {GREEN(_human_bw(peak_bw)):>14}   "
            f"avg_lat {avg_dt * 1e3:.2f} ms"
        )

    return avg_bw


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    world = mx.distributed.init()
    rank = world.rank()
    world_size = world.size()

    if rank == 0:
        print()
        print(BOLD("━" * 60))
        print(BOLD("  MLX JACCL RDMA Connectivity & Bandwidth Test"))
        print(BOLD("━" * 60))
        print(f"  Ranks     : {world_size}")
        print(f"  Rounds    : {ROUNDS}")
        print(
            f"  Sizes     : {', '.join(_human_bytes(s * BYTES_PER_ELEM) for s in SIZES)}"
        )
        print()

    # ── Phase 0: barrier smoke test ──────────────────────────────────────────
    if rank == 0:
        print(BOLD("[ Phase 0 ] Barrier smoke test"))
    _sync_barrier(world)
    if rank == 0:
        print(f"  {GREEN('All ranks reached barrier ✓')}")
        print()

    # ── Phase 1: correctness ─────────────────────────────────────────────────
    if rank == 0:
        print(BOLD("[ Phase 1 ] Correctness (all_sum value check)"))
    for size in SIZES:
        test_correctness(world, rank, size)
    if rank == 0:
        print()

    # ── Phase 2: latency ─────────────────────────────────────────────────────
    if rank == 0:
        print(BOLD("[ Phase 2 ] Latency (1-element all_sum)"))
    test_latency(world, rank)
    if rank == 0:
        print()

    # ── Phase 3: bandwidth sweep ─────────────────────────────────────────────
    if rank == 0:
        print(BOLD(f"[ Phase 3 ] Bandwidth sweep ({ROUNDS} rounds each)"))

    results = {}
    for size in SIZES:
        bw = test_bandwidth(world, rank, size)
        results[size] = bw

    # ── Summary ──────────────────────────────────────────────────────────────
    if rank == 0:
        print()
        print(BOLD("━" * 60))
        print(BOLD("  Summary"))
        print(BOLD("━" * 60))

        peak = max(results.values())
        for size, bw in results.items():
            bar_len = int((bw / peak) * 30)
            bar = "█" * bar_len + "░" * (30 - bar_len)
            print(
                f"  {_human_bytes(size * BYTES_PER_ELEM):>8}  {CYAN(bar)}  {_human_bw(bw)}"
            )

        print()
        best_bw = peak
        verdict = (
            GREEN("EXCELLENT — TB5 RDMA is flying 🚀")
            if best_bw > 5e9
            else GREEN("GREAT — strong RDMA performance ✓")
            if best_bw > 2e9
            else YELLOW("OK — RDMA working, but bandwidth is moderate")
            if best_bw > 500e6
            else RED("LOW — check RDMA enablement and cable")
        )
        print(f"  Overall: {verdict}")
        print(f"  Peak BW: {CYAN(_human_bw(best_bw))}")
        print(BOLD("━" * 60))
        print()


if __name__ == "__main__":
    main()
