#!/usr/bin/env python3
"""
Live GPU monitor — shows utilization % and memory for all visible GPUs.
Updates every 2 seconds, designed to run in a dedicated tmux pane.
Kill with Ctrl-C or when the parent training process finishes.
"""

import subprocess
import sys
import time
import signal
import os


# ── ANSI helpers ──────────────────────────────────────────────────────────────
RESET  = "\033[0m"
BOLD   = "\033[1m"
GREEN  = "\033[92m"
YELLOW = "\033[93m"
RED    = "\033[91m"
CYAN   = "\033[96m"
DIM    = "\033[2m"

def color_pct(pct: int) -> str:
    if pct >= 80:
        return RED
    elif pct >= 50:
        return YELLOW
    return GREEN

def bar(pct: int, width: int = 20) -> str:
    filled = round(width * pct / 100)
    c = color_pct(pct)
    return f"{c}{'█' * filled}{DIM}{'░' * (width - filled)}{RESET}"

def mem_bar(used: int, total: int, width: int = 12) -> str:
    pct = used / total * 100 if total > 0 else 0
    return bar(int(pct), width)


# ── nvidia-smi query ──────────────────────────────────────────────────────────
QUERY = "index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit"
FMT   = "csv,noheader,nounits"

def query_gpus() -> list[dict]:
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={QUERY}", f"--format={FMT}"],
        capture_output=True, text=True,
    )
    gpus = []
    for line in result.stdout.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 6:
            continue
        idx, name, util, mem_used, mem_total, temp = parts[:6]
        pwr_draw  = parts[6] if len(parts) > 6 else "N/A"
        pwr_limit = parts[7] if len(parts) > 7 else "N/A"

        def safe_int(v, default=0):
            try:
                return int(float(v))
            except (ValueError, TypeError):
                return default

        gpus.append({
            "idx":       idx,
            "name":      name[:18],
            "util":      safe_int(util),
            "mem_used":  safe_int(mem_used),
            "mem_total": safe_int(mem_total),
            "temp":      safe_int(temp),
            "pwr_draw":  safe_int(pwr_draw),
            "pwr_limit": safe_int(pwr_limit),
        })
    return gpus


# ── render ─────────────────────────────────────────────────────────────────────
HEADER = (
    f"{BOLD}{CYAN}"
    f"  GPU  {'Name':<18}  {'Util':>4}  {'Util-bar':<20}  "
    f"{'Mem Used':>9} / {'Total':<9}  {'Mem-bar':<12}  "
    f"{'Temp':>5}  {'Power':>12}"
    f"{RESET}"
)
SEP = "─" * 110

def render(gpus: list[dict]) -> str:
    lines = [
        f"{BOLD}  LVR Training — GPU Monitor{RESET}  "
        f"{DIM}(refresh 2 s | Ctrl-C to stop){RESET}",
        SEP,
        HEADER,
        SEP,
    ]
    for g in gpus:
        mem_pct = g["mem_used"] / g["mem_total"] * 100 if g["mem_total"] > 0 else 0
        pwr_str = (
            f"{g['pwr_draw']:>4}W / {g['pwr_limit']:<4}W"
            if g["pwr_limit"] > 0
            else f"{g['pwr_draw']:>4}W"
        )
        lines.append(
            f"  {g['idx']:>3}  {g['name']:<18}  "
            f"{color_pct(g['util'])}{g['util']:>3}%{RESET}  "
            f"[{bar(g['util'])}]  "
            f"{g['mem_used']:>7} MiB / {g['mem_total']:<7} MiB  "
            f"[{mem_bar(g['mem_used'], g['mem_total'])}]  "
            f"{g['temp']:>3}°C  "
            f"{pwr_str}"
        )
    lines.append(SEP)
    return "\n".join(lines)


# ── main loop ─────────────────────────────────────────────────────────────────
def handle_exit(sig, frame):
    # Move cursor down past the display before exiting
    sys.stdout.write("\n")
    sys.exit(0)

signal.signal(signal.SIGTERM, handle_exit)
signal.signal(signal.SIGINT,  handle_exit)

prev_lines = 0

while True:
    try:
        gpus   = query_gpus()
        output = render(gpus)

        # Erase previous output
        if prev_lines > 0:
            sys.stdout.write(f"\033[{prev_lines}A\033[J")

        sys.stdout.write(output + "\n")
        sys.stdout.flush()
        prev_lines = output.count("\n") + 1

    except Exception as e:
        sys.stdout.write(f"[gpu_monitor] error: {e}\n")
        sys.stdout.flush()

    time.sleep(2)
