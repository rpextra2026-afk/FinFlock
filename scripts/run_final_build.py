"""Child of the Kaggle supervisor (scripts/cpu_session.py) for the final build, also runnable on a laptop.

  python scripts/run_final_build.py <run> --root <tree with dataset/ outputs/ store/> --out <results folder> [--smoke N] [--workers N]
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oryxis.build.run import run_all  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--config", default=str(ROOT / "config"))
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--out", default=None)
    parser.add_argument("--smoke", type=int, default=0, help="keep only this many events (a mechanics test, never a result)")
    parser.add_argument("--fast-fit", action="store_true", help="the smoke grid and steps on ALL events: a scale test of the score and metrics stages, never a result")
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument("--open_test", action="store_true")
    args = parser.parse_args(argv)
    args.out = args.out or str(Path(args.root) / "outputs" / "final-build" / "runs" / args.run)
    return run_all(args, log)


if __name__ == "__main__":
    sys.exit(main())
