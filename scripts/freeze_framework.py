#!/usr/bin/env python3
"""Write FREEZE_FRAMEWORK.json (once; never overwritten): hashes of the cleaned tapes, the config, the code, the persona tapes and, when it exists, dev_fit.pkl.

  python scripts/freeze_framework.py [--runs db3-llm2 tessera-llm1 ...] [--dev-fit-dir <build out dir>] [--out <file>]

The file goes to outputs/final-build/clean/FREEZE_FRAMEWORK.json by default. The older FREEZE.json is not touched. A dev fit that is not made yet is recorded as null
(the run then pins the hash when it writes the file).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finfm.common.config import load_config  # noqa: E402
from finfm.framework import freeze as FZ  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", nargs="*", default=None)
    parser.add_argument("--root", default=str(ROOT))
    parser.add_argument("--out", default=None)
    parser.add_argument("--dev-fit-dir", default=None, help="folder holding <run>/framework-final/dev_fit.pkl (default: outputs/final-build/runs)")
    args = parser.parse_args(argv)
    base = load_config(ROOT / "config")
    root = Path(args.root)
    runs = args.runs or list(base.get("clean.runs"))
    path = Path(args.out) if args.out else root / str(base.get("clean.output_dir")) / str(base.get("framework.final.freeze_file"))
    dev_dir = Path(args.dev_fit_dir) if args.dev_fit_dir else root / "outputs" / "final-build" / "runs"
    dev = {run: dev_dir / run / str(base.get("framework.final.results_dir")) / str(base.get("framework.final.dev_fit_file")) for run in runs}
    hashes = FZ.current_hashes(base, root, ROOT, runs, dev)
    record = FZ.write_freeze(path, hashes, ROOT)
    print(f"wrote {path}")
    print({k: (v if not isinstance(v, dict) else {a: (str(b)[:12] if b else b) for a, b in v.items()}) for k, v in record.items() if k != "persona_tape_sha256"})
    return 0


if __name__ == "__main__":
    sys.exit(main())
