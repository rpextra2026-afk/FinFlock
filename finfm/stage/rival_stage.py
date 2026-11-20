"""FinRipple-lite units: framework-sample finance events with their knowledge-graph text (built from the link snapshots)."""

from __future__ import annotations

import bisect
import json
from collections.abc import Callable
from pathlib import Path

import pandas as pd

from finfm.common.config import Config
from finfm.common.schema import UNIT_ID
from finfm.compute.bundle import make_job_id
from finfm.compute.shard import EST_SECONDS, plan_shards
from finfm.compute.staging import stage_shard
from finfm.reader.prompt import Tokenizer
from finfm.rivals.finripple_lite.run import KG, NEWS, kg_lines, kg_text, random_lines

KG_LINES = "kg_lines_json"
FIRST_ASOF_COLUMN = "asof_date"


def sp500_seeded_events(reader: pd.DataFrame, snapshots: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """The events FinRipple-lite can act on in a bucket whose framework sample is mostly about companies outside the index.

    FinRipple answers for companies of the S&P 500 (its prompt asks for "TICKER in SP500"), so its events are the finance events of the reader sample that
    pin a company that was an index member on the event's own day (the latest snapshot on or before that day: no later information). Every such event is
    kept: there is no further sampling and no look at prices or at the model's answers. Columns as the framework sample: ``event_id`` ...
    """
    stamps = sorted(pd.to_datetime(snapshots["snapshot_date"]).unique())
    members = {stamp: frozenset(group["security_id"]) for stamp, group in snapshots.groupby(pd.to_datetime(snapshots["snapshot_date"]))}

    def member_on(node: str, day: pd.Timestamp) -> bool:
        position = bisect.bisect_right(stamps, day.to_datetime64()) - 1
        return position >= 0 and node in members[stamps[position]]

    kind = str(cfg.get("cpu.finripple_build.domain_kind"))
    events = reader[reader["kind"] == kind]
    days = pd.to_datetime(events["available_utc"], utc=True).dt.tz_convert(None).dt.normalize()
    keep = [any(p["node_type"] == "company" and member_on(p["node_id"], day) for p in json.loads(pins)) for pins, day in zip(events["pins_json"], days, strict=True)]
    return events[keep].rename(columns={UNIT_ID: "event_id"}).reset_index(drop=True)


def finripple_units(framework: pd.DataFrame, links: pd.DataFrame, names: dict[str, str], cfg: Config) -> pd.DataFrame:
    """Prebuilt units: ``unit_id, news, kg_lines_json, asof_date``. Token truncation happens at staging, with the model's tokenizer."""
    labels = {n: f"{name} ({n})" for n, name in names.items()}
    rows = []
    for row in framework[framework["kind"] == "finance"].itertuples():
        asof = pd.Timestamp(row.available_utc).tz_convert(None).normalize() if pd.Timestamp(row.available_utc).tzinfo else pd.Timestamp(row.available_utc).normalize()
        seeds = [p["node_id"] for p in json.loads(row.pins_json) if p["node_type"] == "company" and p["node_id"] in names]
        lines = kg_lines(seeds, links, labels, asof, cfg) if seeds else random_lines(links, labels, asof, cfg, (row.event_id, "finripple"))
        rows.append({UNIT_ID: row.event_id, NEWS: f"{row.title}\n{row.text}", KG_LINES: json.dumps(lines[: int(cfg.get("rivals.finripple.stored_lines"))]), FIRST_ASOF_COLUMN: str(asof.date()),
                     EST_SECONDS: float(cfg.get("runs.est_seconds_per_unit.finripple"))})
    return pd.DataFrame(rows).sort_values(UNIT_ID).reset_index(drop=True)


def stage_finripple_units(bucket: str, llm: str, units: pd.DataFrame, tokenizer: Tokenizer, cfg: Config, out_root: Path,
                          log: Callable[[str], None] = print) -> list[Path]:
    """Truncate each unit's news and graph text to the token budgets and write the shard folders."""
    budget = int(cfg.get("rivals.finripple.kg_tokens"))
    news_budget = int(cfg.get("reader.max_text_tokens"))
    staged = units.assign(
        **{KG: [kg_text(json.loads(lines), tokenizer, budget) for lines in units[KG_LINES]],
           NEWS: [tokenizer.truncate(text, news_budget)[0] for text in units[NEWS]]}).drop(columns=[KG_LINES])
    shards = plan_shards(make_job_id("finripple", bucket, llm), staged, cfg)
    log(f"{bucket}: {len(staged)} finripple units in {len(shards)} shard(s)")
    return [stage_shard(out_root, "finripple", bucket, llm, shard, staged, lambda folder: None, cfg) for shard in shards]


PRESTAGED = "prestaged"
UNITS_FILE = "units.parquet"


def stage_finripple_job(bucket: str, llm: str, cfg: Config, market_dir: Path, tokenizer: Tokenizer, out_root: Path,
                        log: Callable[[str], None] = print) -> list[Path]:
    from finfm.compute.session import SessionError

    path = market_dir / PRESTAGED / f"finripple-{bucket}" / UNITS_FILE
    if not path.exists():
        raise SessionError(f"the market dataset has no prebuilt finripple units for {bucket} ({path})")
    return stage_finripple_units(bucket, llm, pd.read_parquet(path), tokenizer, cfg, out_root, log)
