#!/usr/bin/env python3
"""
macmon_fmt.py — Format macmon JSON output for cluster monitoring.

Usage:
    macmon pipe -s 1 | python3 scripts/macmon_fmt.py --mode snap --host mac.home
    macmon pipe -s 1 | python3 scripts/macmon_fmt.py --mode line --host mac.home

Modes:
    snap   Detailed multi-line snapshot (for hw-snap)
    line   Compact single-line with bars (for hw-monitor)
"""

import argparse
import json
import sys

# ── ANSI helpers ─────────────────────────────────────────────────────────────
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"
GREEN = "\033[32m"
YELLOW = "\033[33m"
RED = "\033[31m"


def _color_threshold(val, lo, hi):
    """Green if below lo, yellow if below hi, red otherwise."""
    if val < lo:
        return GREEN
    if val < hi:
        return YELLOW
    return RED


def _bar(pct, width=20):
    """Simple bar: █░"""
    pct = max(0.0, min(100.0, pct))
    filled = int(pct / 100 * width)
    return "█" * filled + "░" * (width - filled)


# ── Formatters ───────────────────────────────────────────────────────────────
def fmt_snap(host, d):
    """Detailed multi-line snapshot."""
    mem = d.get("memory", {})
    temp = d.get("temp", {})
    e = d.get("ecpu_usage", [0, 0])
    p = d.get("pcpu_usage", [0, 0])
    g = d.get("gpu_usage", [0, 0])

    ram_used = mem.get("ram_usage", 0) / (1024**3)
    ram_total = mem.get("ram_total", 0) / (1024**3)
    ram_pct = (ram_used / ram_total * 100) if ram_total > 0 else 0

    cpu_t = temp.get("cpu_temp_avg", 0)
    gpu_t = temp.get("gpu_temp_avg", 0)

    lines = [
        f"  {BOLD}{host}{RESET}",
        f"    E-CPU       {e[1] * 100:5.1f}%   {int(e[0]):>4} MHz",
        f"    P-CPU       {p[1] * 100:5.1f}%   {int(p[0]):>4} MHz",
        f"    GPU         {g[1] * 100:5.1f}%   {int(g[0]):>4} MHz",
        f"    RAM         {ram_used:.1f} / {ram_total:.0f} GB  ({ram_pct:.0f}%)",
        f"    CPU temp    {cpu_t:.1f} °C",
        f"    GPU temp    {gpu_t:.1f} °C",
        f"    CPU power   {d.get('cpu_power', 0):.2f} W",
        f"    GPU power   {d.get('gpu_power', 0):.2f} W",
        f"    ANE power   {d.get('ane_power', 0):.2f} W",
        f"    Total SoC   {d.get('all_power', 0):.2f} W",
        f"    System      {d.get('sys_power', 0):.1f} W",
        "",
    ]
    return "\n".join(lines)


def fmt_line(host, d):
    """Compact multi-line with bars for live monitor."""
    mem = d.get("memory", {})
    temp = d.get("temp", {})
    p = d.get("pcpu_usage", [0, 0])
    g = d.get("gpu_usage", [0, 0])

    ram_used = mem.get("ram_usage", 0) / (1024**3)
    ram_total = mem.get("ram_total", 0) / (1024**3)
    ram_pct = (ram_used / ram_total * 100) if ram_total > 0 else 0

    cpu_t = temp.get("cpu_temp_avg", 0)
    gpu_t = temp.get("gpu_temp_avg", 0)
    soc_w = d.get("all_power", 0)
    sys_w = d.get("sys_power", 0)

    tc = _color_threshold(cpu_t, 60, 80)
    mc = _color_threshold(ram_pct, 70, 85)
    gc = _color_threshold(g[1] * 100, 50, 80)

    lines = [
        f"  {BOLD}{host:<18}{RESET}",
        f"    CPU  {_bar(p[1] * 100)}  {p[1] * 100:5.1f}%  {int(p[0]):>4} MHz",
        f"    GPU  {gc}{_bar(g[1] * 100)}{RESET}  {g[1] * 100:5.1f}%  {int(g[0]):>4} MHz",
        f"    RAM  {mc}{_bar(ram_pct)}{RESET}  {ram_used:.1f}/{ram_total:.0f} GB ({ram_pct:.0f}%)",
        f"    {tc}Temp: CPU {cpu_t:.0f}°C  GPU {gpu_t:.0f}°C{RESET}  ⚡ SoC {soc_w:.1f}W  Sys {sys_w:.0f}W",
        "",
    ]
    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Format macmon JSON for cluster display")
    ap.add_argument(
        "--mode",
        choices=["snap", "line"],
        default="snap",
        help="Output mode: snap (detailed) or line (compact bars)",
    )
    ap.add_argument("--host", default="unknown", help="Hostname label to display")
    args = ap.parse_args()

    raw = sys.stdin.read().strip()
    if not raw:
        print(f"  {BOLD}{args.host:<18}{RESET}  {RED}✗ no data from macmon{RESET}")
        sys.exit(1)

    # macmon may output multiple lines (samples), take the last one
    last_line = raw.strip().split("\n")[-1]

    try:
        d = json.loads(last_line)
    except json.JSONDecodeError as e:
        print(f"  {BOLD}{args.host:<18}{RESET}  {YELLOW}parse error: {e}{RESET}")
        sys.exit(1)

    if args.mode == "snap":
        print(fmt_snap(args.host, d))
    else:
        print(fmt_line(args.host, d))


if __name__ == "__main__":
    main()
