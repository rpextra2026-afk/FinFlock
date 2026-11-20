"""Write FREEZE.json: the content hashes of the cleaned tapes, the config hashes and the code hash (spec section 8). Run after the cleaning and before the sheaf run.

  python scripts/freeze_final_build.py [--runs a,b,...]

The sheaf run refuses to start when the cleaned tape, the config or the code no longer matches this file.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oryxis.build.run import code_hash  # noqa: E402
from oryxis.common.config import load_config  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", default="")
    args = parser.parse_args(argv)
    cfg = load_config(ROOT / "config")
    runs = [r for r in args.runs.split(",") if r] or list(cfg.get("clean.runs"))
    folder = ROOT / str(cfg.get("clean.output_dir"))
    content, sector = {}, {}
    for run in runs:
        report = json.loads((folder / f"clean_{run}_report.json").read_text())
        content[run], sector[run] = report["content_sha256"], report["sector_rows_sha256"]
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True, check=False).stdout.strip()
    freeze = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git_head": git, "clean_content_sha256": content, "sector_rows_sha256": sector,
              "config_hash": cfg.hash, "config_unhashed_hash": cfg.unhashed_hash, "code_sha256": code_hash(ROOT)}
    out = folder / "FREEZE.json"
    out.write_text(json.dumps(freeze, indent=2))
    print(f"wrote {out}\n" + json.dumps({k: (v if not isinstance(v, dict) else {a: b[:12] for a, b in v.items()}) for k, v in freeze.items()}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
