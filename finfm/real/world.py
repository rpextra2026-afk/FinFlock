"""Everything a CPU run on real tapes reads, loaded once: market, factors, tapes, unit table, events, links, skeletons (shared by run_cpu.py and the ablation runs)."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from finfm.common.config import Config
from finfm.common.hashing import sha256_file
from finfm.common.timing import TradingCalendar
from finfm.common.windows import BucketWindows, bucket_windows
from finfm.compute.payloads import load_skeletons
from finfm.data.buckets.contract import finalize_events
from finfm.personas.skeletons import Skeleton
from finfm.reader.salvage import AlignmentError, config_for_mode, prepare_reader_tape
from finfm.real.market_load import RealMarket, load_real_market
from finfm.real.tapes import TapeError, load_reader_tape, load_tape
from finfm.stage.reader_stage import PRESTAGED

UNITS_FILE = "units.parquet"
FACTORS_FILE = "factors_daily.parquet"
LINKS_FILE = "links.parquet"
MANAGERS_FILE = "f13_managers.parquet"
EXIT_MISALIGNED = 4


@dataclass
class World:
    cfg: Config  # the run configuration of the reader mode
    mode: dict
    mode_name: str
    bucket: str
    llm: str
    rm: RealMarket
    factors: pd.DataFrame
    events: pd.DataFrame
    reader_tape: pd.DataFrame
    persona_tape: pd.DataFrame | None
    skeletons: dict[str, Skeleton]
    links: pd.DataFrame
    managers: pd.DataFrame
    windows: BucketWindows
    calendar: TradingCalendar
    used: list
    persona_used: list
    units_path: Path
    alignment: dict
    fill_info: dict


def load_world(root: Path, bucket: str, llm: str, cfg: Config, mode_name: str, with_personas: bool = True) -> World:
    """Reads the finished tapes and the market tables exactly as ``scripts/run_cpu.py framework`` does."""
    run_cfg, mode = config_for_mode(cfg, mode_name)
    market_dir = root / str(cfg.get("sources.output.dataset_folder"))
    reader_tape, used = load_reader_tape(cfg, root, bucket, llm)
    try:
        persona_tape, persona_used = load_tape(cfg, root, "personas", bucket, llm) if with_personas else (None, [])
    except TapeError as exc:
        print(f"no persona tape ({exc}); the game runs on the reader's consensus only")
        persona_tape, persona_used = None, []
    calendar = TradingCalendar.from_config(run_cfg)
    windows = bucket_windows(run_cfg, bucket, calendar)
    units_path = market_dir / PRESTAGED / f"reader-{bucket}" / UNITS_FILE
    units = pd.read_parquet(units_path).rename(columns={"unit_id": "event_id"})
    pins_json = dict(zip(units["event_id"], units["pins_json"], strict=True))
    try:
        reader_tape, alignment, fill_info = prepare_reader_tape(reader_tape, pins_json, run_cfg, mode)  # current parser on the stored raw answers
    except AlignmentError as exc:
        print(f"stopping before any result is written: {exc}", file=sys.stderr)
        raise SystemExit(EXIT_MISALIGNED) from exc
    if not alignment["ok"]:
        print(f"WARNING: unit table and reader tape are not fully aligned: {alignment}", flush=True)
    events, _ = finalize_events(units, calendar, windows, run_cfg)
    rm = load_real_market(market_dir, run_cfg)
    skeletons = load_skeletons(market_dir / PRESTAGED / f"personas-{bucket}")
    return World(run_cfg, mode, mode_name, bucket, llm, rm, pd.read_parquet(market_dir / FACTORS_FILE), events, reader_tape, persona_tape, skeletons,
                 pd.read_parquet(market_dir / LINKS_FILE), pd.read_parquet(market_dir / MANAGERS_FILE), windows, calendar, used, persona_used, units_path,
                 alignment, fill_info)


def units_sha256(world: World) -> str:
    return sha256_file(world.units_path)
