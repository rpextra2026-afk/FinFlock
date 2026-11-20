"""Rule-based pins: which nodes does an item touch? (dataset-build-spec 5.5). Contains NO model output.

Name-only sources are matched against point-in-time company aliases (word boundary, cashtags, tickers of
two letters or fewer only as cashtags), a country gazetteer, commodity keywords and ETF names.
"""

from __future__ import annotations

import concurrent.futures as futures
import json
import multiprocessing
import re
from collections.abc import Callable

import pandas as pd

from oryxis.common.config import Config
from oryxis.reader.masking import MaskerCache

PIN_COLUMNS = ["item_id", "pin_type", "pin_id", "role", "source_field", "method", "confidence", "sp500_member_on_date"]
WORD = r"A-Za-z0-9"


def _keyword_regex(table: pd.DataFrame, key: str, value: str) -> list[tuple[str, re.Pattern]]:
    return [
        (row[key], re.compile(rf"(?<![{WORD}]){re.escape(str(row[value]))}(?![{WORD}])", re.IGNORECASE))
        for _, row in table.iterrows()
    ]


_SHARED: dict = {}


def _pin_range(state: dict, start: int, stop: int) -> list[tuple]:
    """Pin rows of ``items[start:stop]`` in item order."""
    rows = []
    confidence = state["confidence"]
    for item in state["items"].iloc[start:stop].itertuples():
        text = f"{item.title}\n{item.text}"
        date = pd.Timestamp(item.available_date)
        for node in state["maskers"].for_date(date).find(text):
            rows.append((item.item_id, "company", node, "other", "alias", "rule", confidence["text_alias"], bool(state["is_member"](node, date))))
        for iso, pattern in state["country_rx"]:
            if pattern.search(text):
                rows.append((item.item_id, "country", iso, "other", "gazetteer", "rule", confidence["text_pattern"], None))
        for tag, pattern in state["commodity_rx"]:
            if pattern.search(text):
                rows.append((item.item_id, "commodity", tag, "other", "keyword", "rule", confidence["text_pattern"], None))
    return rows


def _pin_range_shared(start: int, stop: int) -> list[tuple]:
    return _pin_range(_SHARED, start, stop)


def pin_name_only_items(
    items: pd.DataFrame,
    maskers: MaskerCache,
    countries: pd.DataFrame,
    commodities: pd.DataFrame,
    is_member: Callable[[str, pd.Timestamp], bool],
    cfg: Config,
) -> pd.DataFrame:
    """Pins of name-only items. ``items`` needs ``item_id, title, text, available_date``.

    ``is_member(security_node, date)`` answers point-in-time S&P 500 membership; it is only asked for
    company pins and its answer is stored in ``sp500_member_on_date``.
    """
    state = {
        "items": items, "maskers": maskers, "is_member": is_member, "confidence": cfg.get("events.pins.rule_confidence"),
        "country_rx": _keyword_regex(countries, "iso", "alias"), "commodity_rx": _keyword_regex(commodities, "tag", "keyword"),
    }
    workers = int(cfg.get("events.pins.workers"))
    if workers <= 1 or len(items) < workers * int(cfg.get("events.pins.min_items_per_worker")):
        rows = _pin_range(state, 0, len(items))
    else:  # items are independent: contiguous chunks in parallel, joined in the original order, so the result equals the serial one
        bounds = [len(items) * k // workers for k in range(workers + 1)]
        _SHARED.update(state)  # forked workers inherit it: the masker, the membership test and the items are never pickled
        with futures.ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork")) as pool:
            parts = list(pool.map(_pin_range_shared, bounds[:-1], bounds[1:]))
        rows = [row for part in parts for row in part]
        _SHARED.clear()
    out = pd.DataFrame(rows, columns=PIN_COLUMNS).drop_duplicates(["item_id", "pin_type", "pin_id"])
    return out.reset_index(drop=True)


def pin_priority(pin_type: str, role: str, confidence: str, cfg: Config) -> float:
    """Priority used to keep the best pins when an event has more than ``reader.max_pins``."""
    roles = cfg.get("events.pins.role_priority")
    bonus = cfg.get("events.pins.confidence_bonus")
    return float(roles.get(role, roles["other"])) + float(bonus.get(confidence, 0.0))


def node_id_of(pin_type: str, pin_id: str, cfg: Config) -> str:
    """Graph node id of a pin: ``cik:...``, ``c:US``, ``m:oil``, ``e:XLE`` (S&P members use their security id)."""
    return f"{cfg.get('events.pins.node_prefix')[pin_type]}{pin_id}"


def event_pins_json(
    pins: pd.DataFrame, cik_to_security: dict[str, str], descriptors: dict[str, dict], cfg: Config
) -> str:
    """JSON list of an event's pins for the reader: node id, type, sector, layer, priority."""
    out = []
    for row in pins.itertuples():
        node = (
            cik_to_security.get(str(row.pin_id), node_id_of(row.pin_type, row.pin_id, cfg))
            if row.pin_type == "company" else node_id_of(row.pin_type, row.pin_id, cfg)
        )
        info = descriptors.get(node, {})
        out.append({
            "node_id": node, "node_type": row.pin_type,
            "sector": info.get("sector"), "layer": info.get("layer", cfg.get("reader.non_sp_layer")),
            "priority": pin_priority(row.pin_type, row.role, row.confidence, cfg),
        })
    unique = {p["node_id"]: p for p in sorted(out, key=lambda p: -p["priority"])}
    return json.dumps(list(unique.values()), sort_keys=True)
