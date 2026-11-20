"""The tessera package (finfm_v2 candidate) as the event frame the reader and persona stagers use.

The package already holds clustered events, items, text and pins in four folders (``financial``,
``geopolitical``, ``social_reddit`` and the news supplement ``geo_supplement_records``). This module merges them
into the same event frame that ``items_to_events`` plus rule pinning give for db1 and db3, so everything after
it (``finalize_events``, ``build_samples``, the unit tables) is the same code. The package's own pins are used as
delivered; nothing is re-pinned and nothing in the package is edited. Company pins are SEC numbers (CIK) and are
turned into graph ids through the package's entity table and our ticker aliases; a CIK without a graph id stays
``cik:<number>`` (a non-S&P company, as in the other buckets).
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from finfm.common.config import Config

CATEGORIES = ("financial", "geopolitical", "social_reddit")
SUPPLEMENT = "geo_supplement_records"
INTERNAL_TEXT = Path("INTERNAL_ONLY_text_layer") / "internal_text.parquet"
PIN_COLUMNS = ["pin_type", "pin_id", "role", "confidence"]
TITLE_CHARS = 200
OTHER_ROLE = "other"
ISSUER_ROLE = "issuer"
SUPPLEMENT_TIER = "X2"
PACKAGE_TIER = "B"
TRUST_FOR_SOURCE = {"financial": "high", "geopolitical": "high", "social_reddit": "low", SUPPLEMENT: "medium"}


def cik_to_security(package: Path, aliases: pd.DataFrame) -> dict[str, str]:
    """CIK (as text) -> graph node id, through the package's current ticker and our ticker aliases."""
    entities = pd.read_parquet(package / "common" / "entities.parquet")
    nodes = set(aliases.loc[aliases["alias_type"] == "ticker", "node_id"])
    out = {}
    for cik, tickers in zip(entities["cik"], entities["tickers_current"].fillna(""), strict=True):
        for ticker in str(tickers).replace(";", ",").split(","):
            if ticker.strip() in nodes:
                out[str(int(cik))] = ticker.strip()
                break
    return out


def _title(text: str) -> str:
    return text.strip().split("\n", 1)[0][:TITLE_CHARS]


def _category_events(package: Path, category: str) -> pd.DataFrame:
    events = pd.read_parquet(package / category / "events.parquet")
    items = pd.read_parquet(package / category / "items.parquet", columns=["item_id", "source_id", "available_utc"])
    text = pd.read_parquet(package / category / "text.parquet").set_index("item_id")["text"]
    pins = pd.read_parquet(package / category / "pins.parquet")
    pins = pins[pins["pin_type"].isin(["company", "country", "commodity"])]
    by_item = {k: g[PIN_COLUMNS] for k, g in pins.assign(role=lambda f: f["role"].fillna(OTHER_ROLE)).groupby("item_id")}
    source = items.set_index("item_id")["source_id"]
    rows = []
    for e in events.itertuples(index=False):
        members = list(e.member_item_ids)
        first = members[0]
        got = [by_item[m] for m in members if m in by_item]
        rows.append({
            "event_id": e.event_id, "member_item_ids": members, "first_seen_utc": e.first_available_utc,
            "available_utc": e.first_available_utc, "family": e.family, "subtype": e.subtype, "kind": None,
            "entities": [], "title": e.title or _title(text.get(first, "")), "text": text.get(first, ""),
            "n_items": len(members), "n_sources": source.reindex(members).nunique(), "best_tier": PACKAGE_TIER,
            "trust_tier": TRUST_FOR_SOURCE[category], "source_name": source.get(first, category), "flags": [],
            "entity_pins": pd.concat(got).drop_duplicates().reset_index(drop=True) if got else pd.DataFrame(columns=PIN_COLUMNS),
        })
    return pd.DataFrame(rows)


def _supplement_events(package: Path) -> pd.DataFrame:
    records = pd.read_parquet(package / SUPPLEMENT / "records.parquet")
    text = pd.read_parquet(package / INTERNAL_TEXT).set_index("record_id")["text"]
    pins = pd.read_parquet(package / SUPPLEMENT / "pins.parquet")
    by_record = {k: g for k, g in pins.groupby("record_id")}
    rows = []
    for r in records.itertuples(index=False):
        body = text.get(r.record_id, "")
        found = by_record.get(r.record_id)
        parts = [pd.DataFrame({"pin_type": "company", "pin_id": [str(int(c)) for c in r.exposure_ciks], "role": OTHER_ROLE, "confidence": "medium"})]
        if found is not None:
            parts.append(pd.DataFrame({"pin_type": found["pin_type"], "pin_id": found["pin_id"], "role": OTHER_ROLE, "confidence": found["confidence"]}))
        rows.append({
            "event_id": r.event_id, "member_item_ids": [r.record_id], "first_seen_utc": r.available_utc, "available_utc": r.available_utc,
            "family": SUPPLEMENT, "subtype": SUPPLEMENT, "kind": None, "entities": [], "title": r.headline or _title(body), "text": body,
            "n_items": 1, "n_sources": 1, "best_tier": SUPPLEMENT_TIER, "trust_tier": TRUST_FOR_SOURCE[SUPPLEMENT], "source_name": r.source_name,
            "flags": [], "entity_pins": pd.concat(parts).drop_duplicates().reset_index(drop=True),
        })
    return pd.DataFrame(rows)


def package_events(package: str | Path, cfg: Config) -> pd.DataFrame:
    """All events of the package in the event-frame shape of the other buckets (before ``finalize_events``)."""
    base = Path(package)
    frames = [_category_events(base, c) for c in CATEGORIES] + [_supplement_events(base)]
    events = pd.concat(frames, ignore_index=True)
    events["available_utc"] = pd.to_datetime(events["available_utc"], utc=True)
    events["first_seen_utc"] = pd.to_datetime(events["first_seen_utc"], utc=True)
    events["n_pins"] = events["entity_pins"].map(len)
    return events


def with_pins_json(events: pd.DataFrame, package: str | Path, aliases: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Add ``pins_json`` (reader pins with graph node ids) after the events have been finalised."""
    from finfm.data.buckets.pins import event_pins_json

    mapping = cik_to_security(Path(package), aliases)
    out = events.copy()
    out["pins_json"] = [event_pins_json(p, mapping, {}, cfg) for p in out["entity_pins"]]
    out["n_pins"] = [len(json.loads(s)) for s in out["pins_json"]]
    return out


def build_samples_of_package(package: str | Path, aliases: pd.DataFrame, calendar, bucket: str, cfg: Config, log=print, tier: str | None = None):
    """``(S_R, S_F, aliases used)`` for the package bucket: same steps and sample rule as ``build_samples_of_bucket``."""
    from finfm.common.windows import bucket_windows
    from finfm.data.buckets.contract import finalize_events
    from finfm.data.buckets.sampling import build_samples
    from finfm.data.names import drop_generic_aliases

    events = package_events(package, cfg)
    log(f"{bucket}: {len(events)} package events")
    events, dropped = finalize_events(events, calendar, bucket_windows(cfg, bucket, calendar), cfg)
    log(f"{bucket}: {len(events)} dev/test events kept, dropped {dropped}")
    events = with_pins_json(events, package, aliases, cfg)
    used = drop_generic_aliases(aliases, events["title"] + " " + events["text"], cfg)
    sizes = cfg.get(f"sample.tiers.{tier or cfg.get('sample.active_tier')}.{bucket}")
    reader, framework = build_samples(events, int(sizes["reader_events"]), int(sizes["framework_dev"]), int(sizes["framework_test_cap"]), cfg, f"{bucket}|reader")
    return reader, framework, used
