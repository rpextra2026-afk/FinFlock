"""Table schemas of the store (oryxis-data-requirements.md section 7) and their validation."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class SchemaError(ValueError):
    """Raised when a table lacks required columns, has null keys or duplicate keys."""


@dataclass(frozen=True)
class TableSpec:
    name: str
    columns: tuple[str, ...]
    key: tuple[str, ...]


SECURITY_MASTER = TableSpec(
    "security_master",
    ("security_id", "cik", "cusip9", "figi", "instrument_type", "share_class", "issuer_id",
     "first_trade_date", "last_trade_date", "delist_reason", "match_quality"),
    ("security_id",),
)
SYMBOL_HISTORY = TableSpec(
    "security_symbol_history",
    ("security_id", "ticker", "exchange", "valid_from", "valid_to", "source", "asof_known"),
    ("security_id", "valid_from"),
)
PX_DAILY = TableSpec(
    "px_daily",
    ("security_id", "session_date", "open", "high", "low", "close_raw", "volume_raw", "vendor"),
    ("security_id", "session_date", "vendor"),
)
CORP_ACTION = TableSpec(
    "corp_action",
    ("security_id", "ex_date", "action_type", "ratio", "amount", "announce_date", "source"),
    ("security_id", "ex_date", "action_type"),
)
SHARES_OUT = TableSpec(
    "shares_out", ("security_id", "cover_date", "filed_utc", "shares", "accession", "concept"),
    ("security_id", "filed_utc"),
)
SP500_MEMBERSHIP = TableSpec(
    "sp500_membership", ("security_id", "start_date", "end_date", "source", "snapshot_date"),
    ("security_id", "start_date"),
)
SP500_SNAPSHOT = TableSpec(
    "sp500_snapshot", ("snapshot_date", "security_id", "source"), ("snapshot_date", "security_id"),
)
F13_FILING = TableSpec(
    "f13_filing",
    ("accession", "manager_cik", "period", "filing_date", "submission_type", "is_amendment",
     "available_from_session"),
    ("accession",),
)
F13_HOLDING = TableSpec(
    "f13_holding",
    ("accession", "infotable_sk", "cusip9", "shares", "value_usd", "unit_factor", "putcall", "sh_prn"),
    ("accession", "infotable_sk"),
)
MANAGER_CLASS = TableSpec(
    "manager_class", ("manager_cik", "period", "type", "n_positions", "book_usd", "turnover", "passive_flag"),
    ("manager_cik", "period"),
)
LINK = TableSpec(
    "link",
    ("src_node", "dst_node", "link_type", "available_from", "valid_to", "weight", "source",
     "src_traded", "dst_traded", "carried_forward"),
    ("src_node", "dst_node", "link_type", "available_from"),
)
LABEL = TableSpec(
    "label", ("event_id", "security_id", "window", "abn_ret", "log_vol_ratio", "log_dv_ratio"),
    ("event_id", "security_id", "window"),
)
EVENTS = TableSpec(
    "events",
    ("event_id", "member_item_ids", "first_seen_utc", "available_utc", "session_day0", "time_known",
     "family", "subtype", "kind", "entities", "title", "text", "n_items", "n_sources", "best_tier",
     "era", "split", "flags", "source_name", "trust_tier"),
    ("event_id",),
)
SPECS = {s.name: s for s in (
    SECURITY_MASTER, SYMBOL_HISTORY, PX_DAILY, CORP_ACTION, SHARES_OUT, SP500_MEMBERSHIP, SP500_SNAPSHOT,
    F13_FILING, F13_HOLDING, MANAGER_CLASS, LINK, LABEL, EVENTS,
)}


def validate(frame: pd.DataFrame, spec: TableSpec) -> pd.DataFrame:
    """Check columns, null keys and duplicate keys; returns the frame unchanged."""
    missing = [c for c in spec.columns if c not in frame.columns]
    if missing:
        raise SchemaError(f"{spec.name}: missing columns {missing}")
    if frame[list(spec.key)].isna().any().any():
        raise SchemaError(f"{spec.name}: null values in key columns {spec.key}")
    if frame.duplicated(list(spec.key)).any():
        raise SchemaError(f"{spec.name}: duplicate keys {spec.key}")
    return frame
