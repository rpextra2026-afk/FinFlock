"""Write a fixed shard plan for one run: finished units are left out, the rest is cut into equal numbered shards.

  python3 scripts/make_shard_plan.py reader db1 llm2 --bundles ../outputs --first-shard 4 --n-shards 5
  python3 scripts/make_shard_plan.py reader db3 llm2 --first-shard 0 --n-shards 13      # a fresh run: nothing finished yet

The plan lands in finflock/compute/shard_plans/<job>__<bucket>__<llm>.json and travels inside the notebook's code archive.
Shard numbers start at ``--first-shard`` (one number per account). Re-run it only before any account has started the new shards:
the cut depends on which units count as finished.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finflock.common.schema import STATUS, TERMINAL_STATUSES, UNIT_ID  # noqa: E402
from finflock.compute.bundle import make_job_id, read_consolidated, scan_bundles, unit_state  # noqa: E402
from finflock.common.config import load_config  # noqa: E402
from finflock.compute.shard import PLAN_DIR, units_fingerprint  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("job")
    parser.add_argument("bucket")
    parser.add_argument("llm")
    parser.add_argument("--bundles", default=None, help="folder with the finished bundles of this run (omit for a fresh run)")
    parser.add_argument("--units", help="prestaged units.parquet (default: dataset/finflock-upload/prestaged/<job>-<bucket>)")
    parser.add_argument("--first-shard", type=int, required=True)
    parser.add_argument("--n-shards", type=int, default=None)
    parser.add_argument("--part", nargs=3, type=int, action="append", metavar=("HASH_FROM", "HASH_BELOW", "N_SHARDS"), default=[],
                        help="a block of shards over the units whose event hash bucket is in [HASH_FROM, HASH_BELOW); repeat for further blocks, numbered in order")
    parser.add_argument("--retry-unparsed", action="store_true",
                        help="also make one extra shard of the finished units whose answer did not parse (a rerun with the fixed prompt)")
    args = parser.parse_args(argv)
    job_id = make_job_id(args.job, args.bucket, args.llm)
    units_path = Path(args.units) if args.units else ROOT / "dataset" / "finflock-upload" / "prestaged" / f"{args.job}-{args.bucket}" / "units.parquet"
    units = set(pd.read_parquet(units_path)[UNIT_ID].astype(str))
    done: list[str] = []
    if args.bundles:
        state = unit_state(scan_bundles([Path(args.bundles)], job_id).valid)
        done = sorted(set(state.loc[state[STATUS].isin(TERMINAL_STATUSES), UNIT_ID].astype(str)) & units)
    if bool(args.part) == (args.n_shards is not None):
        parser.error("give either --n-shards or one or more --part blocks")
    n_shards = sum(n for _, _, n in args.part) if args.part else args.n_shards
    plan = {"job_id": job_id, "first_shard": args.first_shard, "n_shards": n_shards, "n_units_total": len(units),
            "units_sha256": units_fingerprint(units), "done_unit_ids": done}
    if args.part:
        plan["parts"] = [{"hash_from": a, "hash_below": b, "n_shards": n} for a, b, n in args.part]
    if args.retry_unparsed and not args.bundles:
        parser.error("--retry-unparsed needs --bundles")
    if args.retry_unparsed:
        tape = read_consolidated(scan_bundles([Path(args.bundles)], job_id).valid)
        unparsed = tape[(tape[STATUS] == "ok") & tape["parse_ok"].map(lambda v: not pd.isna(v) and not bool(v))]
        plan["retry_unit_ids"] = sorted(set(unparsed[UNIT_ID].astype(str)) & units)
        plan["retry_shard"] = args.first_shard + args.n_shards
        plan["retry_reasons"] = {str(k): int(v) for k, v in unparsed["reason"].value_counts().items()}
    PLAN_DIR.mkdir(parents=True, exist_ok=True)
    path = PLAN_DIR / f"{job_id}.json"
    path.write_text(json.dumps(plan, separators=(",", ":")), encoding="utf-8")
    retry = plan.get("retry_unit_ids", [])
    left = len(units) - len(done)
    cfg = load_config(ROOT / "config")
    args.n_shards = n_shards
    per_shard = -(-left // args.n_shards)
    hours = per_shard * float(cfg.get(f"runs.est_seconds_per_unit.{args.job}")) / 3600.0
    print(f"{job_id}: {len(units)} units, {len(done)} finished, {left} left -> shards {args.first_shard}..{args.first_shard + args.n_shards - 1}"
          f" of about {per_shard} units, about {hours:.1f} h of projected work each at the configured speed; wrote {path.relative_to(ROOT)}")
    if retry:
        print(f"retry shard {plan['retry_shard']}: {len(retry)} units whose first answer did not parse {plan['retry_reasons']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
