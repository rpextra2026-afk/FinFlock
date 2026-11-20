"""The open-once test stage for the sealed window (2024-01-01 to 2025-12-31) of the tessera tapes.

It is the SAME code as the main framework stage (``finfm.framework.final_run.run_final(sealed=True)``): the cleaning steps, the dev fit, the merge fit, the optimiser and the metrics are not
re-written here. What differs is only the window the test events and the decision days come from. Nothing is fitted on the sealed window or on the 2022-2023 hold-out: the dev events, the
dev fit and the merge fit are the fit window's, with the embargo of the ORIGINAL hold-out start.

Safeguards, in the order they run:
  1. the run must be one of ``framework.sealed.runs``;
  2. FREEZE_FRAMEWORK.json must match (the main run's freeze check, unchanged);
  3. ``SEALED_OPENED.json`` is created with O_EXCL in the stage's own folder; if it exists the stage refuses to start, so a second run in the same folder is impossible. (A fresh Kaggle session
     has a fresh folder: the notebook also refuses when a ``SEALED_OPENED.json`` is found in any attached input, and the owner keeps the first result.);
  4. the sealed rows are cleaned with the same steps on the longer window; the rows of the units that the frozen clean file holds must reproduce its ``content_sha256`` exactly, otherwise the
     stage stops (a cleaning that changes when a later window is added would not be the frozen cleaning).
"""

from __future__ import annotations

import json
import os
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd

from finfm.clean import run as clean_run
from finfm.common.config import Config
from finfm.common.hashing import sha256_file

CLEAN_TAG = "_sealed_window"
SHORT = 12  # characters of a hash shown in messages
LOCK_TEXT_CHARS = 200
Log = Callable[[str], None]


def sealed_settings(base: Config, run: str) -> dict[str, Any]:
    spec = dict(base.get("framework.sealed"))
    if run not in spec["runs"]:
        raise SystemExit(f"{run} is not in framework.sealed.runs {spec['runs']}: only those tapes have a sealed window to open")
    return spec


def window_cfg(run_cfg: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    """The run's windows with the test window replaced by the sealed one (fit window and everything else unchanged)."""
    return {**run_cfg, "holdout_start": str(spec["start"]), "holdout_end": str(spec["end"])}


def take_lock(folder: Path, spec: dict[str, Any], run: str, freeze_note: dict[str, Any], git_head: str) -> Path:
    """Create the lock file; raises when it exists (the sealed window is opened once)."""
    path = folder / str(spec["lock_file"])
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        raise SystemExit(f"{path} exists: the sealed window of {run} was already opened ({path.read_text()[:LOCK_TEXT_CHARS]}). It is opened once; this run refuses to start.") from None
    with os.fdopen(descriptor, "w") as handle:
        handle.write(json.dumps({"run": run, "opened_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "window": [str(spec["start"]), str(spec["end"])], "freeze": freeze_note, "git_head": git_head}, indent=1))
    return path


def sealed_clean(root: Path, run: str, base: Config, folder: Path, spec: dict[str, Any], log: Log) -> Path:
    """Clean the tape on the longer window into ``folder`` and prove that the frozen part is reproduced unchanged. Returns the cleaned file."""
    target = folder / f"clean_{run}{CLEAN_TAG}.parquet"
    if target.exists():
        return target
    run_cfg = window_cfg(dict(base.get(f"clean.runs.{run}")), spec)
    log(f"sealed clean: {run}, window {spec['start']} to {spec['end']}")
    report = clean_run.clean_run(run, base, str(root), run_cfg=run_cfg, out_dir=folder, tag=CLEAN_TAG)
    frozen_report = json.loads((root / str(base.get("clean.output_dir")) / f"clean_{run}_report.json").read_text())
    frozen_rows = pd.read_parquet(root / str(base.get("clean.output_dir")) / f"clean_{run}.parquet", columns=["unit_id"])
    long = pd.read_parquet(target)
    same = long[long["unit_id"].isin(set(frozen_rows["unit_id"]))]
    digest = clean_run.content_hashes(same, int(base.get("clean.report.decimals")))["content_sha256"]
    if digest != frozen_report["content_sha256"]:
        raise SystemExit(f"the frozen part of the longer cleaning differs from the frozen clean file (content_sha256 {digest[:SHORT]} against {frozen_report['content_sha256'][:SHORT]}): the stage stops")
    log(f"sealed clean done: {report['items_out']} items, {report['rows_out']} rows; the {frozen_report['items_out']} frozen items reproduce content_sha256 {digest[:SHORT]}")
    (folder / "sealed_clean_check.json").write_text(json.dumps({"content_sha256_of_frozen_units": digest, "frozen_content_sha256": frozen_report["content_sha256"], "file_sha256": sha256_file(target),
                                                                  "items_out": report["items_out"]}, indent=1))
    return target
