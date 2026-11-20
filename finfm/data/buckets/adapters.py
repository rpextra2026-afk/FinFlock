"""Adapters for the public databuckets (db1, db3): raw records to items to events (spec 3.6).

Timing is recomputed with the project rule; the corpus's own trading-date column is ignored. Names,
columns and licences live in ``config/adapters.yaml`` (VERIFY items); this module has no source names.
"""

from __future__ import annotations

import ast
import json
import re

import pandas as pd

from finfm.common.config import Config
from finfm.common.hashing import sha256_hex

ITEM_COLUMNS = ["item_id", "source_name", "kind", "family", "tier", "trust_tier", "published_utc", "available_utc",
                "time_precision", "title", "text", "tickers"]
EVENT_HASH_CHARS = 16
TITLE_KEY = re.compile(r"[^a-z0-9]+")


def normalised_title_hash(title: str) -> str:
    """Hash of the lower-cased alphanumeric title, for cross-set deduplication."""
    return sha256_hex(TITLE_KEY.sub(" ", str(title).lower()).strip())


def to_utc(values: pd.Series, source: dict) -> pd.Series:
    """Parse timestamps (ISO, naive-in-timezone, or epoch seconds) to UTC."""
    if source.get("time_unit") == "epoch_seconds":
        return pd.to_datetime(values, unit="s", utc=True)
    parsed = pd.to_datetime(values, errors="coerce")
    if parsed.dt.tz is None:
        return parsed.dt.tz_localize(str(source["timezone"]), ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC")
    return parsed.dt.tz_convert("UTC")


TITLE_SEPARATOR = "\n\n"
ID_HASH_CHARS = 16


def _stock_list(extra: str) -> list[str]:
    """Tickers of a record from the corpus's ``extra_fields`` JSON (the ``stocks`` entry is a stringified list)."""
    try:
        value = json.loads(extra).get("stocks")
        listed = ast.literal_eval(value) if isinstance(value, str) else (value or [])
    except (ValueError, SyntaxError, TypeError):
        return []
    return [str(t) for t in listed]


def from_hf_corpus(raw: pd.DataFrame) -> pd.DataFrame:
    """Generic ``id, date, title, body, stocks`` columns from the 3-column corpus (``date``, ``text``, ``extra_fields``).

    The title is the first paragraph of ``text`` and the body the rest (the title again when there is no rest). The corpus has
    no record id, so the id is a hash of the timestamp and the text.
    """
    parts = raw["text"].fillna("").astype(str).str.split(TITLE_SEPARATOR, n=1, expand=True)
    title = parts[0]
    body = parts[1].fillna(title) if parts.shape[1] > 1 else title
    ids = (raw["date"].astype(str) + "|" + raw["text"].fillna("").astype(str)).map(sha256_hex).str[:ID_HASH_CHARS]
    return pd.DataFrame({"id": ids.to_numpy(), "date": raw["date"].to_numpy(), "title": title.to_numpy(), "body": body.to_numpy(),
                         "stocks": raw["extra_fields"].map(_stock_list).to_numpy()})


def normalise_source(raw: pd.DataFrame, name: str, cfg: Config) -> pd.DataFrame:
    """Items of one source: ids, UTC times, kind/family/tier, text. Unparseable or empty records are dropped."""
    source = cfg.get(f"adapters.sources.{name}")
    columns = source["columns"]
    if source.get("hf_corpus"):
        raw = from_hf_corpus(raw)
    published = to_utc(raw[columns["time"]], source)
    frame = pd.DataFrame({
        "item_id": name + ":" + raw[columns["id"]].astype(str),
        "source_name": name, "kind": source["kind"], "family": source["family"], "tier": source["tier"],
        "trust_tier": source["trust_tier"], "published_utc": published, "available_utc": published,
        "title": raw[columns["title"]].fillna("").astype(str),
        "text": raw[columns["text"]].fillna("").astype(str),
        "tickers": raw[columns["tickers"]] if columns.get("tickers") else None,
    })
    midnight = (published - published.dt.normalize()) == pd.Timedelta(0)
    frame["time_precision"] = midnight.map({True: "date", False: "second"})
    keep = published.notna() & (frame["title"].str.len() >= int(cfg.get("adapters.min_title_chars")))
    return frame[keep].drop_duplicates("item_id")[ITEM_COLUMNS].reset_index(drop=True)


def drop_cross_set_duplicates(items_by_bucket: dict[str, pd.DataFrame], cfg: Config) -> dict[str, pd.DataFrame]:
    """Keep each normalised title in the lowest set number only (the multi-source corpus dedupes only within itself)."""
    seen: set[str] = set()
    out = {}
    for bucket in cfg.get("adapters.set_order"):
        frame = items_by_bucket.get(bucket)
        if frame is None:
            continue
        hashes = frame["title"].map(normalised_title_hash)
        fresh = ~hashes.isin(seen) & ~hashes.duplicated()
        seen |= set(hashes[fresh])
        out[bucket] = frame[fresh].reset_index(drop=True)
    return out


def era_of(date: pd.Timestamp, cfg: Config) -> str:
    for era in cfg.get("events.eras"):
        if pd.Timestamp(era["start"]) <= date <= pd.Timestamp(era["end"]):
            return str(era["name"])
    return "unknown"


def items_to_events(items: pd.DataFrame, primary_entity: pd.Series, cfg: Config) -> pd.DataFrame:
    """Cluster items into events. ``primary_entity`` gives each item's main entity (a pin id or ``none``).

    The event takes its availability from the earliest member (first seen), its text and title from that
    member, the best tier, and the source of the earliest trustworthy item.
    """
    from finfm.data.buckets.events import cluster_ids

    frame = items.assign(entity=primary_entity.reindex(items.index).fillna("none"))
    frame["cluster"] = cluster_ids(frame, "entity", "kind", "available_utc", cfg)
    tier_rank, trust_rank = cfg.get("events.tier_rank"), cfg.get("events.trust_rank")
    rows = []
    for cluster, part in frame.sort_values("available_utc", kind="mergesort").groupby("cluster", sort=False):
        first = part.iloc[0]
        trustworthy = part.assign(r=part["trust_tier"].map(trust_rank)).sort_values(["r", "available_utc"], kind="mergesort")
        rows.append({
            "event_id": "ev:" + sha256_hex(cluster)[:EVENT_HASH_CHARS], "member_item_ids": list(part["item_id"]),
            "first_seen_utc": first["available_utc"], "available_utc": first["available_utc"],
            "family": first["family"], "subtype": first["family"], "kind": first["kind"], "entities": [first["entity"]],
            "title": first["title"], "text": first["text"], "n_items": len(part), "n_sources": part["source_name"].nunique(),
            "best_tier": min(part["tier"], key=lambda t: tier_rank[t]), "trust_tier": trustworthy["trust_tier"].iloc[0],
            "source_name": trustworthy["source_name"].iloc[0], "flags": [],
        })
    return pd.DataFrame(rows)
