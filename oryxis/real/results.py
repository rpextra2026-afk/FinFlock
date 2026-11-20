"""Results folders: tables as parquet/CSV beside a lineage file that says which tapes, config and commit produced them."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from oryxis.common.config import Config
from oryxis.common.hashing import sha256_file
from oryxis.common.provenance import collect_provenance

GIT_TIMEOUT_SECONDS_KEY = "runtime.git_timeout_seconds"
PACKAGES_KEY = "runtime.provenance_packages"


def write_lineage(folder: Path, cfg: Config, repo: Path, tapes: list[Path], extra: dict) -> None:
    prov = collect_provenance(repo, list(cfg.get(PACKAGES_KEY)), float(cfg.get(GIT_TIMEOUT_SECONDS_KEY)))
    # a plain write: the file is tiny and written last, and the atomic writer (fsync of a directory) fails on Windows
    write_json(folder, str(cfg.get("cpu.synthetic_flag_file")), {
        "synthetic": False, "config_hash": cfg.hash, "tapes": {p.name: sha256_file(p) for p in tapes},
        "provenance": prov.to_dict(), **extra})


def write_tables(folder: Path, tables: dict[str, pd.DataFrame]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for name, frame in tables.items():
        target = folder / name.replace("/", "__")
        frame.to_parquet(target.with_suffix(".parquet"))
        frame.to_csv(target.with_suffix(".csv"))


def write_json(folder: Path, name: str, obj: dict) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / name).write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
