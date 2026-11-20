#!/usr/bin/env python3
"""Child of the Kaggle supervisor (scripts/cpu_session.py) for the whole final run of one tape, also runnable on a laptop.

  python scripts/run_framework_final.py <run> --root <tree> --out <results folder> [--workers N] [--smoke N] [--stop-after-build]

Stages in order: clean -> pooled sheaf (graphs, fit) -> scoring and ALL_METRICS -> ablations -> framework stage (road 2 personas, merge, optimiser, backtest, portfolio metrics).
The first part is scripts/run_final_build.py's code (finfm.build.run.run_all); the framework stage starts only when the sheaf fit is finished (a shard run stops after its own fits).
The framework stage refuses to start unless FREEZE_FRAMEWORK.json matches. It never opens the sealed window.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finfm.build.run import MANIFEST, run_all  # noqa: E402
from finfm.common.config import load_config  # noqa: E402
from finfm.framework.final_run import run_final  # noqa: E402

STAGE_LIST = ("clean", "pooled sheaf (graphs, fit)", "scoring and ALL_METRICS", "ablations", "framework stage (road 2 personas, merge, optimiser, backtest, portfolio metrics)")


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--config", default=str(ROOT / "config"))
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--out", default=None)
    parser.add_argument("--smoke", type=int, default=0)
    parser.add_argument("--fast-fit", action="store_true")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--open_test", action="store_true")
    parser.add_argument("--stop-after-build", action="store_true", help="run only the build stages (clean ... ablations)")
    parser.add_argument("--stop-after", choices=["forecast", "calibrate"], default=None, help="framework stage only: stop after the forecasts (stored in the checkpoint), or run only the dev-only calibration of road 2's volume dial (writes outputs/final-build/calibration/volu_calibration_<run>.json)")
    parser.add_argument("--no-volu-calibration", action="store_true", help="the off switch: road 2's volume dial stays raw (the ablation reports raw against calibrated)")
    parser.add_argument("--task-shard", default=None, help="k/N: run only every N-th backtest starting at k (separate processes, same FINFM_CKPT_DIR); no metrics are written by a shard")
    parser.add_argument("--skip-build", action="store_true", help="the build stages are finished in --out; run only the framework stage")
    args = parser.parse_args(argv)
    args.out = args.out or str(Path(args.root) / "outputs" / "final-build" / "runs" / args.run)
    log("stages: " + " -> ".join(STAGE_LIST))
    if not args.skip_build:
        status = run_all(args, log)
        if status != 0:
            return status
    out = Path(args.out)
    manifest = (out / MANIFEST)
    fit_done = manifest.exists() and '"fit"' in manifest.read_text() and '"status": "done"' in manifest.read_text().split('"fit"')[1][:200]
    if not fit_done:
        log("the sheaf fit is not finished here (a shard run): the framework stage is left to the assemble run")
        return 0
    if args.stop_after_build:
        return 0
    base = load_config(Path(args.config))
    workers = args.workers or 1
    shard = tuple(int(x) for x in args.task_shard.split("/")) if args.task_shard else None
    folder = run_final(Path(args.root), args.run, base, out, log, workers=workers, stop_after=args.stop_after, task_shard=shard, smoke=args.smoke, calibrate_volu=False if args.no_volu_calibration else None)
    log(f"framework stage finished: {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
