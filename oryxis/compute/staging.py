"""Staging a shard for a Kaggle session and the manifest that tells the worker what to run (spec 2.3).

Layout: ``<root>/<job>/<shard_id>/`` holds ``shard_manifest.json`` and the job's input files. The same
folder is attached to the session as a private dataset and read by ``oryxis.compute.runner``.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from oryxis.common.config import Config
from oryxis.common.hashing import canonical_json
from oryxis.common.schema import UNIT_ID
from oryxis.compute import payloads as pl
from oryxis.compute.shard import Shard

MANIFEST = "shard_manifest.json"
INFRASTRUCTURE_SECTIONS = ("kaggle", "cpu")  # session limits, quotas, accounts, local result folders: they cannot change a result


def stage_shard(
    root: str | Path, job: str, bucket: str, llm: str, shard: Shard, units: pd.DataFrame,
    write_inputs: Callable[[Path], None], cfg: Config,
) -> Path:
    """Write the shard's unit table, its job inputs and the manifest; returns the shard folder."""
    folder = Path(root) / job / shard.shard_id
    folder.mkdir(parents=True, exist_ok=True)
    chosen = units[units[UNIT_ID].isin(set(shard.unit_ids))]
    chosen.to_parquet(folder / pl.UNITS, index=False)
    write_inputs(folder)
    manifest = {
        "job": job, "bucket": bucket, "llm": llm, "shard_id": shard.shard_id, "job_id": shard.job_id,
        "unit_ids": list(shard.unit_ids), "projected_work_hours": shard.work_hours,
        "session_hours": shard.session_hours, "config_hash": cfg.hash,
        "science_hash": cfg.hash_without(INFRASTRUCTURE_SECTIONS),
    }
    if shard.explicit:
        manifest["explicit_shard"] = True  # fixed plan: the session must be told which shard to run
    (folder / MANIFEST).write_text(canonical_json(manifest), encoding="utf-8")
    return folder


def read_manifest(folder: str | Path) -> dict:
    return json.loads((Path(folder) / MANIFEST).read_text(encoding="utf-8"))
