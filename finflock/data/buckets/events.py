"""Event-layer helpers shared by every databucket: kinds, timing, splits, clustering, flags."""

from __future__ import annotations

import pandas as pd

from finflock.common.config import Config
from finflock.common.hashing import sha256_hex
from finflock.common.timing import TradingCalendar
from finflock.common.windows import BucketWindows

MIDNIGHT = pd.Timedelta(0)
HASH_BUCKETS = 100
HASH_CHARS = 8
HASH_BASE = 16


def kind_of(family: str, mapping: dict[str, str]) -> str | None:
    """Kind of a family by exact name or ``prefix*`` entry; ``None`` if the family is unknown."""
    if family in mapping:
        return mapping[family]
    for pattern, kind in mapping.items():
        if pattern.endswith("*") and family.startswith(pattern[:-1]):
            return kind
    return None


def hash_bucket(event_id: str) -> int:
    """Integer 0-99 = sha256(event_id) mod 100, so hashed samples can be drawn without recomputation."""
    return int(sha256_hex(event_id)[:HASH_CHARS], HASH_BASE) % HASH_BUCKETS


def time_known(available_utc: pd.Series, cfg: Config) -> pd.Series:
    """False for date-only items: a timestamp at exactly 00:00:00 UTC carries no time of day."""
    if not bool(cfg.get("events.midnight_means_date_only")):
        return pd.Series(True, index=available_utc.index)
    stamps = pd.to_datetime(available_utc, utc=True)
    return (stamps - stamps.dt.normalize()) != MIDNIGHT


def assign_day0(events: pd.DataFrame, calendar: TradingCalendar, cfg: Config) -> pd.Series:
    """Day-0 session with the project rule (the corpus's own date column is ignored)."""
    known = time_known(events["available_utc"], cfg)
    day0 = pd.Series(pd.NaT, index=events.index, dtype="datetime64[ns]")
    if known.any():
        day0[known] = calendar.day0(pd.to_datetime(events.loc[known, "available_utc"], utc=True))
    if (~known).any():
        stamps = pd.to_datetime(events.loc[~known, "available_utc"], utc=True).dt.tz_localize(None)
        day0[~known] = calendar.day0_date_only(stamps)
    return day0


def assign_split(day0: pd.Series, windows: BucketWindows) -> pd.Series:
    """``dev`` / ``embargo`` / ``test`` / ``outside`` from the bucket's own windows."""
    return windows.label(day0).set_axis(day0.index)


def cluster_ids(
    frame: pd.DataFrame, entity_col: str, kind_col: str, time_col: str, cfg: Config
) -> pd.Series:
    """Cluster id per item: the first item of a cluster defines it.

    Items with the same entity and kind belong to one cluster while each is within the kind's window of
    the cluster's FIRST item (finance 60 min, geo 24 h). Social items cluster by entity and day.
    """
    windows = {"finance": pd.Timedelta(minutes=float(cfg.get("events.cluster.finance_minutes"))),
               "geo": pd.Timedelta(minutes=float(cfg.get("events.cluster.geo_minutes")))}
    out = pd.Series("", index=frame.index, dtype=object)
    ordered = frame.sort_values(time_col, kind="mergesort")
    for (entity, kind), part in ordered.groupby([entity_col, kind_col], sort=False):
        if kind == "social":
            labels = part[time_col].dt.strftime("%Y-%m-%d")
            out.loc[part.index] = [f"{kind}|{entity}|{d}" for d in labels]
            continue
        window, head, head_label = windows[kind], None, ""
        for index, stamp in zip(part.index, part[time_col], strict=True):
            if head is None or stamp - head > window:
                head, head_label = stamp, f"{kind}|{entity}|{stamp.isoformat()}"
            out.loc[index] = head_label
    return out
