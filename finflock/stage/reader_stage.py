"""Stage the reader shards of one run inside a session: raw news and market files in, shard folders out."""

from __future__ import annotations

import glob
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from finflock.common.config import Config
from finflock.common.timing import TradingCalendar
from finflock.compute import payloads as pl
from finflock.compute.bundle import make_job_id
from finflock.compute.shard import EST_SECONDS, plan_shards, select_shards
from finflock.compute.staging import stage_shard
from finflock.data.sp500 import members_on
from finflock.stage.reader_build import (
    basic_units,
    bucket_sources,
    build_items,
    build_reader_units,
    build_samples_of_bucket,
    cap_candidates,
)

GUARDIAN = "guardian"


def load_raw(cfg: Config, sources: list[str], hf_dir: Path, market_dir: Path) -> dict[str, pd.DataFrame]:
    """Raw frame of each requested source: Hugging Face parquet folders, and the Guardian files shipped in the market dataset."""
    raw = {}
    for source in sources:
        folder = market_dir / GUARDIAN if source == GUARDIAN else hf_dir / source
        files = sorted(glob.glob(str(folder / "*.parquet")))
        if files:
            raw[source] = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return raw


def sources_needed(cfg: Config, bucket: str) -> list[str]:
    """The bucket's own sources plus those of lower-numbered buckets (their titles are dropped here as duplicates)."""
    order = list(cfg.get("adapters.set_order"))
    return [s for b in order[: order.index(bucket) + 1] for s in bucket_sources(cfg, b)]


def _inputs(bucket: str, cfg: Config, config_dir: Path, market_dir: Path, hf_dir: Path, log: Callable[[str], None]):
    """Everything both jobs need: candidate items, aliases, a membership test, gazetteers and the calendar."""
    order = list(cfg.get("adapters.set_order"))
    raw = load_raw(cfg, sources_needed(cfg, bucket), hf_dir, market_dir)
    earlier = {b: build_items(raw, b, cfg) for b in order[: order.index(bucket)] if any(s in raw for s in bucket_sources(cfg, b))}
    calendar = TradingCalendar.from_config(cfg)
    items = cap_candidates(build_items(raw, bucket, cfg, earlier), calendar, bucket, cfg)
    log(f"{bucket}: {len(items)} candidate items from {sorted(items['source_name'].unique())}")
    snapshots = pd.read_parquet(market_dir / "sp500_snapshot.parquet")
    return (items, pd.read_parquet(market_dir / "aliases.parquet"), lambda node, date: node in members_on(snapshots, date).security_ids,
            pd.read_csv(config_dir / "countries.csv"), pd.read_csv(config_dir / "commodity_keywords.csv"), calendar)


PRESTAGED = "prestaged"
UNITS_FILE = "units.parquet"
ALIASES_FILE = "aliases.parquet"


def build_job_units(job: str, bucket: str, cfg: Config, config_dir: Path, market_dir: Path, hf_dir: Path, log: Callable[[str], None] = print):
    """``(units, aliases used)`` of a job from raw news; slow (tens of minutes), so the result is normally prebuilt once."""
    items, aliases, member_on, countries, commodities, calendar = _inputs(bucket, cfg, config_dir, market_dir, hf_dir, log)
    if job == "reader":
        return build_reader_units(items, aliases, member_on, countries, commodities, calendar, bucket, cfg)
    _, framework, used = build_samples_of_bucket(items, aliases, member_on, countries, commodities, calendar, bucket, cfg)
    names = {n: str(g["alias"].iloc[0]) for n, g in used[used["alias_type"] == "name"].groupby("node_id")}
    return basic_units(framework, names, member_on, calendar, cfg), used


def save_prestaged(market_dir: Path, job: str, bucket: str, units: pd.DataFrame, aliases: pd.DataFrame) -> Path:
    folder = market_dir / PRESTAGED / f"{job}-{bucket}"
    folder.mkdir(parents=True, exist_ok=True)
    units.to_parquet(folder / UNITS_FILE, index=False)
    aliases.to_parquet(folder / ALIASES_FILE, index=False)
    return folder


def stage_units(job: str, bucket: str, llm: str, units: pd.DataFrame, aliases: pd.DataFrame, cfg: Config, out_root: Path,
                log: Callable[[str], None] = print, only_shard: str | int | None = None) -> list[Path]:
    """Cut a unit table into shards and write the shard folders (seconds)."""
    units = units.assign(**{EST_SECONDS: float(cfg.get(f"runs.est_seconds_per_unit.{job}"))})  # the speed comes from the current config, not the table built long ago
    shards = select_shards(plan_shards(make_job_id(job, bucket, llm), units, cfg), only_shard)
    log(f"{bucket}: {len(units)} {job} units, staging {len(shards)} shard(s)")
    write = (lambda folder: aliases.to_parquet(folder / pl.ALIASES, index=False)) if job == "reader" else (lambda folder: None)
    return [stage_shard(out_root, job, bucket, llm, shard, units, write, cfg) for shard in shards]


def stage_job(job: str, bucket: str, llm: str, cfg: Config, config_dir: Path, market_dir: Path, hf_dir: Path | None, out_root: Path,
              log: Callable[[str], None] = print, only_shard: str | int | None = None) -> list[Path]:
    """Shards of a job: from the prebuilt unit table in the market dataset when present, else built from raw news."""
    pre = market_dir / PRESTAGED / f"{job}-{bucket}"
    if (pre / UNITS_FILE).exists():
        units, aliases = pd.read_parquet(pre / UNITS_FILE), pd.read_parquet(pre / ALIASES_FILE)
    else:
        units, aliases = build_job_units(job, bucket, cfg, config_dir, market_dir, hf_dir, log)
    return stage_units(job, bucket, llm, units, aliases, cfg, out_root, log, only_shard)
