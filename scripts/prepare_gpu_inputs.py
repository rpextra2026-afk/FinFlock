"""Unit tables for the minimum extra GPU work behind experiments D and F. Local only; nothing is uploaded or launched.

  python scripts/prepare_gpu_inputs.py reader  <bucket> <llm> <folder with the reader .bundle file(s)>
  python scripts/prepare_gpu_inputs.py personas <bucket> <llm> <reader tape folder> <personas tape folder>

reader   : the framework-sample events the reader tape has no row for, in the unit-table layout of prestaged/reader-<bucket>.
personas : the persona units (all personas) of framework-sample events whose reader row is clean (status ok, no reason)
           and that the persona tape has not done. Run it after the extra reader run, with its bundles added to the reader folder.
Output: <cpu.gpu_extra.folder>/prestaged/<job>-<bucket>/units.parquet (est_seconds set to the measured seconds per unit)
        and a JSON of counts beside it. The aliases file of the reader job is copied unchanged.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from finfm.common.config import load_config  # noqa: E402
from finfm.compute.bundle import make_job_id, read_consolidated, scan_bundles  # noqa: E402

EVENT_SEPARATOR = "|"
UNIT_ID = "unit_id"


def tape_of(folder: str, job: str, bucket: str, llm: str) -> pd.DataFrame:
    return read_consolidated(scan_bundles([folder], make_job_id(job, bucket, llm)).valid)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("job", choices=["reader", "personas"])
    parser.add_argument("bucket")
    parser.add_argument("llm")
    parser.add_argument("reader_tapes")
    parser.add_argument("persona_tapes", nargs="?")
    args = parser.parse_args(argv)
    cfg = load_config(ROOT / "config")
    upload = ROOT / str(cfg.get("cpu.folders.upload")) / "prestaged"
    seconds = cfg.get("cpu.gpu_extra.measured_seconds_per_unit")
    out = ROOT / str(cfg.get("cpu.gpu_extra.folder")) / "prestaged" / f"{args.job}-{args.bucket}"
    out.mkdir(parents=True, exist_ok=True)
    events = pd.read_parquet(upload / f"personas-{args.bucket}" / "events.parquet", columns=["event_id"])["event_id"]
    reader_tape = tape_of(args.reader_tapes, cfg.get("cpu.gpu_extra.reader_job"), args.bucket, args.llm)

    if args.job == "reader":
        units = pd.read_parquet(upload / f"reader-{args.bucket}" / "units.parquet")
        keep = units[units[UNIT_ID].isin(events) & ~units[UNIT_ID].isin(reader_tape[UNIT_ID])].copy()
        keep["est_seconds"] = float(seconds["reader"])
        shutil.copy(upload / f"reader-{args.bucket}" / "aliases.parquet", out / "aliases.parquet")
        counts = {"framework_events": int(len(events)), "reader_rows_done": int(len(reader_tape)), "reader_units_to_run": int(len(keep))}
    else:
        if not args.persona_tapes:
            parser.error("personas needs the persona tape folder")
        persona_tape = tape_of(args.persona_tapes, cfg.get("cpu.gpu_extra.persona_job"), args.bucket, args.llm)
        clean = reader_tape[(reader_tape["status"] == "ok") & reader_tape["reason"].isna()][UNIT_ID]
        done_events = set(persona_tape[UNIT_ID].str.split(EVENT_SEPARATOR).str[0])
        want = set(events) & set(clean)
        units = pd.read_parquet(upload / f"personas-{args.bucket}" / "units.parquet")
        event_of = units[UNIT_ID].str.split(EVENT_SEPARATOR).str[0]
        keep = units[event_of.isin(want - done_events) & ~units[UNIT_ID].isin(persona_tape[UNIT_ID])].copy()
        keep["est_seconds"] = float(seconds["personas"])
        counts = {"framework_events": int(len(events)), "clean_reader_framework_events": len(want),
                  "events_with_persona_rows": len(done_events), "persona_events_to_run": len(want - done_events),
                  "persona_units_to_run": int(len(keep))}
    counts["estimated_gpu_hours"] = round(float(keep["est_seconds"].sum()) / 3600, 2)
    keep.to_parquet(out / "units.parquet", index=False)
    shutil.copy(upload / f"reader-{args.bucket}" / "aliases.parquet", out.parents[1] / "aliases.parquet")   # the marker file the notebook looks for
    if args.job == "personas":                                  # only the events and states the chosen units use; skeletons are small
        used = set(keep[UNIT_ID].str.split(EVENT_SEPARATOR).str[0])
        source = upload / f"personas-{args.bucket}"
        for name in ("events", "states"):
            table = pd.read_parquet(source / f"{name}.parquet")
            table[table["event_id"].isin(used)].to_parquet(out / f"{name}.parquet", index=False)
        shutil.copy(source / "skeletons.json", out / "skeletons.json")
    (out / "counts.json").write_text(json.dumps(counts, indent=1), encoding="utf-8")
    print(json.dumps(counts, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
