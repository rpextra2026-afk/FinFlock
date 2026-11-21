"""Finished model outputs (tapes) from the folders the owner downloads bundles into."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from finflock.common.config import Config
from finflock.compute.bundle import make_job_id, read_consolidated, scan_bundles


class TapeError(RuntimeError):
    """Raised when a tape is missing or a bundle needed for it is damaged."""


def tape_dirs(cfg: Config, root: Path) -> list[Path]:
    return [(root / str(d)).resolve() for d in cfg.get("cpu.tape_dirs")]


READER_V2_JOB = "reader2"  # the final hardened reader (reader2/); its tapes carry the parsed columns themselves
READER_V2_MARK = "event_type_final"


def is_reader_v2(tape: pd.DataFrame) -> bool:
    return READER_V2_MARK in tape.columns


def load_reader_tape(cfg: Config, root: Path, bucket: str, llm: str) -> tuple[pd.DataFrame, list[Path]]:
    """The reader tape of this run: the final reader (job ``reader2``) when its bundles are present, otherwise the earlier reader (job ``reader``)."""
    if scan_bundles(tape_dirs(cfg, root), make_job_id(READER_V2_JOB, bucket, llm)).valid:
        return load_tape(cfg, root, READER_V2_JOB, bucket, llm)
    return load_tape(cfg, root, "reader", bucket, llm)


def load_tape(cfg: Config, root: Path, job: str, bucket: str, llm: str) -> tuple[pd.DataFrame, list[Path]]:
    """One row per unit over every valid bundle of the job, and the bundle files used."""
    job_id = make_job_id(job, bucket, llm)
    scan = scan_bundles(tape_dirs(cfg, root), job_id)
    if not scan.valid:
        raise TapeError(f"no valid bundle of {job_id} in {[str(d) for d in tape_dirs(cfg, root)]}; "
                        f"rejected: {[(p.name, why) for p, why in scan.rejected if job_id in p.name]}")
    return read_consolidated(scan.valid), [ref.path for ref in scan.valid]
