"""Build the reader's unit table of one databucket from raw news, the same way on every machine (spec 3.6-3.7, 4.1).

raw sources -> items (UTC, deduplicated) -> candidate cap -> rule pins (company names, countries, commodities) -> events
-> day 0 and dev/test split -> sample S_R -> one unit per event. Everything is deterministic for the same inputs and config,
so a notebook can rebuild the table in every session and always see the same shards.
"""

from __future__ import annotations

import json

import pandas as pd

from finfm.common.config import Config
from finfm.common.hashing import sha256_hex
from finfm.common.schema import AVAILABLE_DATE, FAMILY, KIND, PINS_JSON, TEXT, TITLE, UNIT_ID
from finfm.common.timing import TradingCalendar
from finfm.common.windows import bucket_windows
from finfm.compute.shard import EST_SECONDS
from finfm.data.buckets.adapters import drop_cross_set_duplicates, items_to_events, normalise_source
from finfm.data.buckets.contract import finalize_events
from finfm.data.buckets.pins import event_pins_json, pin_name_only_items
from finfm.data.buckets.sampling import build_samples
from finfm.data.names import drop_generic_aliases
from finfm.reader.masking import MaskerCache

NO_ENTITY = "none"


def bucket_sources(cfg: Config, bucket: str) -> list[str]:
    return [name for name, spec in cfg.get("adapters.sources").items() if spec["bucket"] == bucket]


def build_items(raw_by_source: dict[str, pd.DataFrame], bucket: str, cfg: Config, earlier: dict[str, pd.DataFrame] | None = None) -> pd.DataFrame:
    """Items of one bucket inside its data window; titles already seen in a lower-numbered bucket are dropped (kept there only)."""
    items = pd.concat([normalise_source(raw_by_source[s], s, cfg) for s in bucket_sources(cfg, bucket) if s in raw_by_source], ignore_index=True)
    if earlier:
        items = drop_cross_set_duplicates({**earlier, bucket: items}, cfg)[bucket]
    return items


def cap_candidates(items: pd.DataFrame, calendar: TradingCalendar, bucket: str, cfg: Config) -> pd.DataFrame:
    """Keep items inside the bucket's dev and test windows, then at most N per (source, year) in a fixed hashed order."""
    windows = bucket_windows(cfg, bucket, calendar)
    stamps = items["available_utc"].dt.tz_convert(None)
    inside = items[(stamps >= pd.Timestamp(windows.dev_start)) & (stamps <= pd.Timestamp(windows.test_end) + pd.Timedelta(days=1))].copy()
    inside["_year"] = inside["available_utc"].dt.year
    inside["_key"] = inside["item_id"].map(sha256_hex)
    cap = int(cfg.get("runs.candidate_items_per_source_year"))
    kept = inside.sort_values("_key").groupby(["source_name", "_year"], sort=False).head(cap)
    return kept.drop(columns=["_year", "_key"]).sort_values("item_id").reset_index(drop=True)


def primary_entities(pins: pd.DataFrame, items: pd.DataFrame) -> pd.Series:
    """Each item's main entity: its first company pin, else its first other pin, else ``none``."""
    ordered = pins.assign(_rank=pins["pin_type"].ne("company")).sort_values(["item_id", "_rank", "pin_id"], kind="mergesort")
    first = ordered.groupby("item_id")["pin_id"].first()
    return items["item_id"].map(first).fillna(NO_ENTITY).set_axis(items.index)


def build_samples_of_bucket(
    items: pd.DataFrame, aliases: pd.DataFrame, member_on: callable, countries: pd.DataFrame, commodities: pd.DataFrame,
    calendar: TradingCalendar, bucket: str, cfg: Config, tier: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(S_R, S_F, aliases used)``: the reader sample and the framework sample (a subset of it), with each event's pins."""
    sizes = cfg.get(f"sample.tiers.{tier or cfg.get('sample.active_tier')}.{bucket}")
    aliases = drop_generic_aliases(aliases, items["title"] + " " + items["text"], cfg)
    maskers = MaskerCache(aliases, label_prefix=str(cfg.get("reader.masked_company_prefix")), short_ticker_max_len=int(cfg.get("reader.short_ticker_max_len")))
    work = items.assign(available_date=items["available_utc"].dt.tz_convert(None).dt.normalize())
    pins = pin_name_only_items(work, maskers, countries, commodities, member_on, cfg)
    events = items_to_events(items, primary_entities(pins, items), cfg)
    item_pins = pins.set_index("item_id")
    events["entity_pins"] = [pd.concat([item_pins.loc[[i]] for i in ids if i in item_pins.index]).reset_index(drop=True)
                             if any(i in item_pins.index for i in ids) else pins.iloc[0:0] for ids in events["member_item_ids"]]
    events, _ = finalize_events(events, calendar, bucket_windows(cfg, bucket, calendar), cfg)
    companies = {n: n for n in aliases["node_id"].unique()}
    events["n_pins"] = events["entity_pins"].map(len)
    events["pins_json"] = [event_pins_json(p, companies, {}, cfg) for p in events["entity_pins"]]
    reader, framework = build_samples(events, int(sizes["reader_events"]), int(sizes["framework_dev"]), int(sizes["framework_test_cap"]), cfg, f"{bucket}|reader")
    return reader, framework, aliases


def reader_units(reader: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """The reader unit table ``unit_id, kind, family, available_date, title, text, pins_json, est_seconds``."""
    units = pd.DataFrame({
        UNIT_ID: reader["event_id"].to_numpy(), KIND: reader["kind"].to_numpy(), FAMILY: reader["family"].to_numpy(),
        AVAILABLE_DATE: pd.to_datetime(reader["available_utc"]).dt.strftime("%Y-%m-%d").to_numpy(),
        TITLE: reader["title"].to_numpy(), TEXT: reader["text"].to_numpy(), PINS_JSON: reader["pins_json"].to_numpy(),
    })
    units["available_utc"] = pd.to_datetime(reader["available_utc"]).astype(str).to_numpy()
    units[EST_SECONDS] = float(cfg.get("runs.est_seconds_per_unit.reader"))
    return units.sort_values(UNIT_ID).reset_index(drop=True)


def build_reader_units(
    items: pd.DataFrame, aliases: pd.DataFrame, member_on: callable, countries: pd.DataFrame, commodities: pd.DataFrame,
    calendar: TradingCalendar, bucket: str, cfg: Config, tier: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(units, aliases used)`` of one bucket's reader job."""
    reader, _, used = build_samples_of_bucket(items, aliases, member_on, countries, commodities, calendar, bucket, cfg, tier)
    return reader_units(reader, cfg), used


def basic_units(framework: pd.DataFrame, names: dict[str, str], member_on: callable, calendar: TradingCalendar, cfg: Config) -> pd.DataFrame:
    """Lopez-Lira and Tang units: finance events of the framework sample that name exactly one S&P 500 member."""
    from finfm.rivals.llt.run import COMPANY, HEADLINE, select_events

    frame = framework.copy()
    frame["session_day0"] = pd.to_datetime(frame["session_day0"])
    frame["decision_date"] = calendar.text_decision_date(frame["available_utc"]) if len(frame) else []
    # pins_json carries graph node ids for every bucket; entity_pins of the tessera package holds SEC numbers (CIK), which never match a security
    pins = {r.event_id: [p["node_id"] for p in json.loads(r.pins_json) if p["node_type"] == "company"] for r in frame.itertuples()}
    chosen = select_events(frame, pins, names, member_on, cfg)
    chosen[EST_SECONDS] = float(cfg.get("runs.est_seconds_per_unit.basic"))
    return chosen.sort_values(UNIT_ID).reset_index(drop=True)
