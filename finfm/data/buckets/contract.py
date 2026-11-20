"""The interface contract of the Tessera (tessera) and the common finishing step for every bucket.

``load_contract`` reads ``events.parquet``, ``items.parquet`` and ``pins.parquet`` in the schema of
dataset-build-spec Part 5, checks the required columns, and ``finalize_events`` recomputes timing with the
project rule, assigns the bucket's own split, era and hash bucket, and drops embargo and outside events.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from finfm.common.config import Config
from finfm.common.timing import TradingCalendar
from finfm.common.windows import BucketWindows
from finfm.data.buckets.adapters import era_of
from finfm.data.buckets.events import assign_day0, assign_split, hash_bucket, kind_of, time_known
from finfm.data.schema import EVENTS, SchemaError

REQUIRED_EVENT_COLUMNS = [c for c in EVENTS.columns if c not in {"text", "session_day0", "split", "era"}] + ["text_path"]
REQUIRED_ITEM_COLUMNS = ["item_id", "source_name", "kind", "available_utc", "headline", "family"]
REQUIRED_PIN_COLUMNS = ["item_id", "pin_type", "pin_id", "role", "confidence", "sp500_member_on_date"]


@dataclass
class BucketData:
    events: pd.DataFrame
    items: pd.DataFrame
    pins: pd.DataFrame
    dropped: dict[str, int]


def check_columns(frame: pd.DataFrame, required: list[str], name: str) -> None:
    missing = [c for c in required if c not in frame.columns]
    if missing:
        raise SchemaError(f"{name}: missing columns {missing}")


def load_contract(root: str | Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Read the three parquet files of the Tessera and check the contract columns."""
    base = Path(root)
    events, items, pins = (pd.read_parquet(base / f"{n}.parquet") for n in ("events", "items", "pins"))
    check_columns(events, REQUIRED_EVENT_COLUMNS, "events")
    check_columns(items, REQUIRED_ITEM_COLUMNS, "items")
    check_columns(pins, REQUIRED_PIN_COLUMNS, "pins")
    return events, items, pins


def read_text(path: str, text_root: str | Path) -> str:
    """Local article text of an event (text never leaves the machine); empty if the file is missing."""
    target = Path(text_root) / path
    return target.read_text(encoding="utf-8") if target.exists() else ""


def finalize_events(
    events: pd.DataFrame, calendar: TradingCalendar, windows: BucketWindows, cfg: Config
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Recompute day 0, split, era, hash bucket and kind; keep dev and test events only."""
    frame = events.copy()
    mapping = dict(cfg.get("events.family_kind"))
    frame["kind"] = [kind_of(f, mapping) or k for f, k in zip(frame["family"], frame["kind"], strict=True)]
    allowed = list(cfg.get("events.kinds"))
    unmapped = sorted(set(frame.loc[~frame["kind"].isin(allowed), "family"].astype(str)))
    if unmapped:  # a family with no kind would reach the persona feed as None and fail there, after the GPU time is spent
        raise SchemaError(f"events of families {unmapped} have no kind in {allowed}: add them to events.family_kind in config/events.yaml")
    frame["time_known"] = time_known(frame["available_utc"], cfg)
    frame["session_day0"] = assign_day0(frame, calendar, cfg)
    frame["split"] = assign_split(frame["session_day0"], windows)
    frame["era"] = [era_of(d, cfg) for d in frame["session_day0"]]
    frame["hash_bucket"] = frame["event_id"].map(hash_bucket)
    dropped = frame["split"].value_counts().to_dict()
    kept = frame[frame["split"].isin(["dev", "test"])].reset_index(drop=True)
    return kept, {"embargo": int(dropped.get("embargo", 0)), "outside": int(dropped.get("outside", 0))}
