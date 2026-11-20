#!/usr/bin/env python3
"""Child of the Kaggle supervisor for the OPEN-ONCE sealed-window stage (2024-01-01 to 2025-12-31) of a tessera tape. Never run it twice.

  python scripts/run_framework_sealed.py <run> --root <tree> --out <folder holding fitted.pkl and manifest.json> --i-open-the-sealed-window-once

It is the main framework stage with the sealed window as the test window (oryxis.framework.final_run.run_final(sealed=True), same code, same configuration). It starts only when
FREEZE_FRAMEWORK.json matches, takes an O_EXCL lock (SEALED_OPENED.json) in its own folder <out>/framework-sealed/ and refuses to start when that lock is there. The flag is required so that
the stage cannot be started by accident.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oryxis.common.config import load_config  # noqa: E402
from oryxis.framework.final_run import run_final  # noqa: E402


def log(message: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {message}", flush=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("run")
    parser.add_argument("--config", default=str(ROOT / "config"))
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--i-open-the-sealed-window-once", action="store_true", dest="confirmed")
    args = parser.parse_args(argv)
    if not args.confirmed:
        log("refusing to start: pass --i-open-the-sealed-window-once (the sealed window is opened once)")
        return 2
    base = load_config(Path(args.config))
    folder = run_final(Path(args.root), args.run, base, Path(args.out), log, workers=args.workers, sealed=True)
    log(f"sealed stage finished: {folder}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
