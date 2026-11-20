#!/usr/bin/env python3
"""The final framework stage of one tape with its backtests run in parallel shards, inside ONE notebook (child of scripts/cpu_session.py).

  python scripts/run_framework_sharded.py <run> --root <tree> --out <folder with fitted.pkl, manifest.json, ALL_METRICS_<run>.parquet> [--shards N]

  1. combine + forecast once (scripts/run_framework_final.py --skip-build --stop-after forecast): stored in the checkpoint folder (ORYXIS_CKPT_DIR, set by cpu_session);
  2. N separate processes, each ``run_framework_final.py --skip-build --task-shard k/N``: every N-th backtest (variant x headline set x delisting rule); each loads the combination and forecasts from
     the checkpoint and stores its backtests there (separate processes, no fork pool);
  3. the unsharded pass: takes every backtest from the checkpoint, runs the one no-world-lane comparison backtest, builds the metrics tables and writes framework-final/FINAL_REPORT.md.

At the notebook's cutoff the supervisor stops this process; the processes below are stopped with it and the checkpoint bundle holds every stored piece, so a second session continues where this one stopped.
Prints a line when a stage starts and ends and every 10 minutes while shards run.
"""

from __future__ import annotations

import argparse
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FINAL = HERE / "run_framework_final.py"
BEAT_SECONDS = 600
POLL_SECONDS = 5
processes: list[subprocess.Popen] = []


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] SHARDED {message}", flush=True)


def stop_children(signum=None, frame=None) -> None:
    for p in processes:
        if p.poll() is None:
            p.terminate()
    if signum is not None:
        sys.exit(128 + signum)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--config", default=str(HERE.parent / "config"))
    parser.add_argument("--root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--no-volu-calibration", action="store_true")
    parser.add_argument("--shards", type=int, default=0, help="parallel backtest processes; 0 = the number of CPUs, at most 4")
    args = parser.parse_args(argv)
    shards = args.shards or min(os.cpu_count() or 1, 4)
    base = [sys.executable, str(FINAL), args.run, "--skip-build", "--config", args.config, "--root", args.root, "--out", args.out, "--workers", "1", *(["--no-volu-calibration"] if args.no_volu_calibration else [])]
    signal.signal(signal.SIGTERM, stop_children)
    began = time.time()

    log("stage 1: combine and forecast (once)")
    status = subprocess.call([*base, "--stop-after", "forecast"])
    if status != 0:
        log(f"combine/forecast failed (exit {status})")
        return status
    log(f"stage 1 done in {time.time() - began:.0f} s; stage 2: {shards} backtest shards in parallel")

    began = time.time()
    for k in range(shards):
        processes.append(subprocess.Popen([*base, "--task-shard", f"{k}/{shards}"]))
    last = time.time()
    while any(p.poll() is None for p in processes):
        time.sleep(POLL_SECONDS)
        if time.time() - last >= BEAT_SECONDS:
            last = time.time()
            log(f"shards running {sum(p.poll() is None for p in processes)}/{shards}, {time.time() - began:.0f} s in")
    codes = [p.returncode for p in processes]
    if any(codes):
        log(f"a shard failed: exit codes {codes}")
        return 1
    log(f"stage 2 done in {time.time() - began:.0f} s; stage 3: no-world comparison backtest, metrics, report")

    status = subprocess.call(base)
    log(f"stage 3 finished with exit {status}")
    return status


if __name__ == "__main__":
    sys.exit(main())
