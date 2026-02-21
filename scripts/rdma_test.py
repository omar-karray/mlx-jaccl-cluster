#!/usr/bin/env python3
"""
rdma_test.py — Minimal RDMA connectivity & bandwidth test for MLX JACCL clusters.

No model download required. Tests RDMA by running MLX distributed collective
operations (all_sum) over Thunderbolt RDMA between all ranks.

Usage (via mlx.launch):
    .venv/bin/mlx.launch --backend jaccl \
      --hostfile hostfiles/hosts-2node.json \
      --env MLX_METAL_FAST_SYNCH=1 -- \
      python scripts/rdma_test.py

Optional env vars:
    RDMA_ROUNDS      Number of benchmark rounds    (default: 20)
    RDMA_SIZES       Comma-separated tensor sizes  (default: 1024,65536,1048576,16777216)
    RDMA_VERBOSE     Set to 1 for per-round timing (default: 0)
    RDMA_MAX_MB      Safety cap on max tensor size in MB (default: 256)
"""

import gc
import os
import sys
import time

import mlx.core as mx

# ─── Config ───────────────────────────────────────────────────────────────────
ROUNDS = int(os.environ.get("RDMA_ROUNDS", "20"))
SIZES_ENV = os.environ.get("RDMA_SIZES", "1024,65536,1048576,16777216")
SIZES = [int(s.strip()) for s in SIZES_ENV.split(",")]
VERBOSE = os.environ.get("RDMA_VERBOSE", "0") == "1"
MAX_MB = int(os.environ.get("RDMA_MAX_MB", "256"))

# Bytes per float32 element
BYTES_PER_ELEM = 4

# ─── ANSI colours (disabled if not a tty) ─────────────────────────────────────
_IS_TTY = sys.stdout.isatty()


def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _IS_TTY else text


GREEN = lambda t: _c("32", t)
YELLOW = lambda t: _c("33", t)
CYAN = lambda t: _c("36", t)
BOLD = lambda t: _c("1", t)
RED = lambda t: _c("31", t)
DIM = lambda t: _c("2", t)

# ─── Memory helpers ───────────────────────────────────────────────────────────


def _mb(num_bytes: int) -> float:
    return num_bytes / (1024 * 1024)


def _free_buffers(*tensors) -> None:
    """
    Explicitly delete tensors, run Python GC, then clear MLX Metal cache.
    Call this between tests to avoid Metal buffer accumulation.
    """
    for t in tensors:
        del t
    gc.collect()
    mx.clear_cache()


def _cache_mb() -> float:
    return _mb(mx.get_cache_memory())


def _check_size_safe(num_elems: int, rank: int) -> bool:
    """
    Estimate peak memory for one all_sum operation:
      - input tensor   : num_elems × 4 bytes
      - RDMA buffer    : num_elems × 4 bytes  (staging copy for transfer)
      - output tensor  : num_elems × 4 bytes
      ─────────────────────────────────────────
      Peak per rank    : num_elems × 12 bytes

    Returns False and prints a warning on rank 0 if it exceeds RDMA_MAX_MB.
    """
    peak_bytes = num_elems * BYTES_PER_ELEM * 3
    peak_mb = _mb(peak_bytes)
    if peak_mb > MAX_MB:
        if rank == 0:
            print(
                RED(
                    f"  ⚠  Skipping {_human_bytes(num_elems * BYTES_PER_ELEM)}: "
                    f"estimated peak {peak_mb:.0f} MB exceeds RDMA_MAX_MB={MAX_MB} MB"
                )
            )
        return False
    return True


# ─── Formatting helpers ────────────────────────────────────────────────────────


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


# ─── Barrier ──────────────────────────────────────────────────────────────────


def _sync_barrier(world: mx.distributed.Group) -> None:
    """Global barrier: all ranks must arrive before any continue."""
    x = mx.ones((1,), dtype=mx.float32)
    result = mx.distributed.all_sum(x, group=world)
    mx.eval(result)
    _free_buffers(x, result)


# ─── Phase 0: Barrier smoke test ──────────────────────────────────────────────


def test_barrier(world: mx.distributed.Group, rank: int) -> None:
    _sync_barrier(world)
    if rank == 0:
        print(f"  {GREEN('All ranks reached barrier ✓')}")


# ─── Phase 1: Correctness ─────────────────────────────────────────────────────


def test_correctness(world: mx.distributed.Group, rank: int, num_elems: int) -> None:
    """
    Each rank fills a tensor with (rank+1).
    After all_sum the expected value is sum(1..world_size)
    = world_size*(world_size+1)//2 on every element.
    """
    if not _check_size_safe(num_elems, rank):
        return

    world_size = world.size()
    expected = float(world_size * (world_size + 1) // 2)

    tensor = mx.full((num_elems,), float(rank + 1), dtype=mx.float32)
    result = mx.distributed.all_sum(tensor, group=world)
    mx.eval(result)

    actual = float(result[0])
    passed = abs(actual - expected) < 1e-3

    if rank == 0:
        status = GREEN("PASS ✓") if passed else RED("FAIL ✗")
        label = _human_bytes(num_elems * BYTES_PER_ELEM)
        cache = _cache_mb()
        print(
            f"  Correctness [{label:>8}]: {status}  "
            f"(expected {expected:.1f}, got {actual:.1f})  "
            f"{DIM(f'cache {cache:.1f} MB')}"
        )

    # ── explicit cleanup ──
    _free_buffers(tensor, result)

    if not passed:
        raise RuntimeError(
            f"Rank {rank}: correctness FAILED — expected {expected}, got {actual}"
        )


# ─── Phase 2: Latency ─────────────────────────────────────────────────────────


def test_latency(world: mx.distributed.Group, rank: int) -> None:
    """
    Measure round-trip latency of a tiny all_sum (1 element = 4 bytes).
    Single tensor allocated once, result explicitly deleted each iteration.
    """
    WARMUP = 5
    ITERS = 50

    tensor = mx.ones((1,), dtype=mx.float32)

    # warmup
    for _ in range(WARMUP):
        r = mx.distributed.all_sum(tensor, group=world)
        mx.eval(r)
        del r

    _sync_barrier(world)

    t0 = time.perf_counter()
    for _ in range(ITERS):
        r = mx.distributed.all_sum(tensor, group=world)
        mx.eval(r)
        del r  # free result immediately — don't let results accumulate
    t1 = time.perf_counter()

    elapsed_us = (t1 - t0) / ITERS * 1e6

    if rank == 0:
        quality = (
            GREEN("excellent")
            if elapsed_us < 100
            else YELLOW("good")
            if elapsed_us < 500
            else RED("slow")
        )
        print(f"  Latency (1 elem / 4 B):  {CYAN(f'{elapsed_us:.1f} µs')}  [{quality}]")

    # ── explicit cleanup ──
    _free_buffers(tensor)


# ─── Phase 3: Bandwidth sweep ─────────────────────────────────────────────────


def test_bandwidth(
    world: mx.distributed.Group,
    rank: int,
    num_elems: int,
) -> float | None:
    """
    all_sum bandwidth benchmark for a given tensor size.
    - Input tensor allocated once and reused across rounds (no re-alloc noise).
    - Result deleted explicitly every iteration.
    - Cache cleared after the test.
    Returns achieved avg GB/s on rank 0, or None if skipped.
    """
    if not _check_size_safe(num_elems, rank):
        return None

    WARMUP = 3

    tensor = mx.ones((num_elems,), dtype=mx.float32)

    # warmup — not timed
    for _ in range(WARMUP):
        r = mx.distributed.all_sum(tensor, group=world)
        mx.eval(r)
        del r

    _sync_barrier(world)

    times = []
    for i in range(ROUNDS):
        t0 = time.perf_counter()
        r = mx.distributed.all_sum(tensor, group=world)
        mx.eval(r)
        t1 = time.perf_counter()

        del r  # free result immediately after timing
        gc.collect()  # nudge GC inside the loop for large tensors

        dt = t1 - t0
        times.append(dt)

        if VERBOSE and rank == 0:
            bw = (num_elems * BYTES_PER_ELEM) / dt
            print(DIM(f"    round {i + 1:02d}: {dt * 1e3:.2f} ms  {_human_bw(bw)}"))

    # drop top 10 % outliers before computing stats
    times.sort()
    trim = max(1, int(len(times) * 0.9))
    trimmed = times[:trim]
    avg_dt = sum(trimmed) / len(trimmed)
    min_dt = min(trimmed)

    payload = num_elems * BYTES_PER_ELEM
    avg_bw = payload / avg_dt
    peak_bw = payload / min_dt

    if rank == 0:
        label = _human_bytes(payload)
        cache = _cache_mb()
        print(
            f"  BW [{label:>8}]:  "
            f"avg {CYAN(_human_bw(avg_bw)):>14}   "
            f"peak {GREEN(_human_bw(peak_bw)):>14}   "
            f"avg_lat {avg_dt * 1e3:.2f} ms   "
            f"{DIM(f'cache {cache:.1f} MB')}"
        )

    # ── explicit cleanup — critical between large tensor tests ──
    _free_buffers(tensor)

    return avg_bw


# ─── Main ─────────────────────────────────────────────────────────────────────


def main() -> None:
    world = mx.distributed.init()
    rank = world.rank()
    world_size = world.size()

    if rank == 0:
        print()
        print(BOLD("━" * 64))
        print(BOLD("  MLX JACCL RDMA Connectivity & Bandwidth Test"))
        print(BOLD("━" * 64))
        print(f"  Ranks     : {world_size}")
        print(f"  Rounds    : {ROUNDS}")
        print(f"  Max MB    : {MAX_MB} MB  (safety cap per tensor, ~3× for peak)")
        print(
            f"  Sizes     : "
            f"{', '.join(_human_bytes(s * BYTES_PER_ELEM) for s in SIZES)}"
        )
        print()

    # ── Phase 0: barrier ──────────────────────────────────────────────────────
    if rank == 0:
        print(BOLD("[ Phase 0 ] Barrier smoke test"))
    test_barrier(world, rank)
    if rank == 0:
        print()

    # ── Phase 1: correctness ──────────────────────────────────────────────────
    if rank == 0:
        print(BOLD("[ Phase 1 ] Correctness  (all_sum value check)"))
    for size in SIZES:
        test_correctness(world, rank, size)
    if rank == 0:
        print()

    # ── Phase 2: latency ──────────────────────────────────────────────────────
    if rank == 0:
        print(BOLD("[ Phase 2 ] Latency  (1-element all_sum, 50 iters)"))
    test_latency(world, rank)
    if rank == 0:
        print()

    # ── Phase 3: bandwidth ────────────────────────────────────────────────────
    if rank == 0:
        print(BOLD(f"[ Phase 3 ] Bandwidth sweep  ({ROUNDS} rounds each)"))

    results: dict[int, float] = {}
    for size in SIZES:
        bw = test_bandwidth(world, rank, size)
        if bw is not None:
            results[size] = bw

    # ── Summary ───────────────────────────────────────────────────────────────
    if rank == 0 and results:
        print()
        print(BOLD("━" * 64))
        print(BOLD("  Summary"))
        print(BOLD("━" * 64))

        peak = max(results.values())
        for size, bw in results.items():
            bar_len = int((bw / peak) * 32)
            bar = "█" * bar_len + "░" * (32 - bar_len)
            print(
                f"  {_human_bytes(size * BYTES_PER_ELEM):>8}  "
                f"{CYAN(bar)}  {_human_bw(bw)}"
            )

        print()
        best_bw = peak
        verdict = (
            GREEN("EXCELLENT — TB5 RDMA is flying 🚀")
            if best_bw > 5e9
            else GREEN("GREAT — strong RDMA performance ✓")
            if best_bw > 2e9
            else YELLOW("OK — RDMA working, bandwidth moderate")
            if best_bw > 500e6
            else RED("LOW — check rdma_ctl enable and cable")
        )
        print(f"  Overall : {verdict}")
        print(f"  Peak BW : {CYAN(_human_bw(best_bw))}")
        print(BOLD("━" * 64))
        print()


if __name__ == "__main__":
    main()
