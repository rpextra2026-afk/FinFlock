"""Masked-names sensitivity for FinRipple-lite (spec 10.1): a hashed share of the events with every company name replaced by COMPANY_A, COMPANY_B ...

The news and the graph text of one event share one label table (labels follow the order of first appearance: the news first, then the graph
lines), so the model sees a consistent world without real names. The table is stored with the unit so the scorer can map the model's answer
(``COMPANY_C``) back to a security. Masking is a leakage control, not a fix; the masked-text probe measures how often it fails.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Mapping

import pandas as pd

from finfm.common.config import Config
from finfm.reader.masking import MaskerCache, label_for
from finfm.rivals.finripple_lite.run import KG, NEWS, kg_lines, random_lines
from finfm.common.schema import UNIT_ID

SUFFIX = "~masked"
SOURCE, MASK_MAP = "source_unit_id", "mask_map_json"
HASH_DIGITS = 8
HEX_BASE = 16  # digits of the hash prefix are read as hexadecimal


def in_sample(unit_id: str, fraction: float) -> bool:
    """Hashed membership: the same units are chosen in every run and on every machine."""
    return int(hashlib.sha256(str(unit_id).encode()).hexdigest()[:HASH_DIGITS], HEX_BASE) / HEX_BASE**HASH_DIGITS < fraction


class LazyLabels(Mapping):
    """node -> ``COMPANY_X``. A node gets the next free label the first time it is asked for (labels already given by the news are kept)."""

    def __init__(self, given: Mapping[str, str], prefix: str) -> None:
        self.table, self.prefix = dict(given), prefix

    def get(self, node, default=None):
        if node not in self.table:
            self.table[node] = label_for(len(self.table), self.prefix)
        return self.table[node]

    def __getitem__(self, node):
        return self.get(node)

    def __iter__(self):
        return iter(self.table)

    def __len__(self):
        return len(self.table)


def masked_units(framework: pd.DataFrame, links: pd.DataFrame, aliases: pd.DataFrame, cfg: Config, fraction: float) -> pd.DataFrame:
    """Prebuilt masked units for the hashed share of the finance events: ``unit_id`` (source id + suffix), ``news``, ``kg_lines_json``,
    ``asof_date``, ``source_unit_id``, ``mask_map_json`` (label -> node). Same columns as the unmasked table, plus the last two."""
    prefix = str(cfg.get("reader.masked_company_prefix"))
    maskers = MaskerCache(aliases, label_prefix=prefix, short_ticker_max_len=int(cfg.get("reader.short_ticker_max_len")))
    stored = int(cfg.get("rivals.finripple.stored_lines"))
    rows = []
    for row in framework[framework["kind"] == "finance"].itertuples():
        if not in_sample(row.event_id, fraction):
            continue
        stamp = pd.Timestamp(row.available_utc)
        asof = (stamp.tz_convert(None) if stamp.tzinfo else stamp).normalize()
        masked = maskers.for_date(asof).mask(f"{row.title}\n{row.text}")
        seeds = [p["node_id"] for p in json.loads(row.pins_json) if p["node_type"] == "company"]
        labels = LazyLabels(masked.mapping, prefix)
        lines = kg_lines(seeds, links, labels, asof, cfg) if seeds else random_lines(links, labels, asof, cfg, (row.event_id, "finripple"))
        lines = lines[:stored]
        rows.append({UNIT_ID: row.event_id + SUFFIX, NEWS: masked.text, "kg_lines_json": json.dumps(lines), "asof_date": str(asof.date()),
                     "est_seconds": float(cfg.get("runs.est_seconds_per_unit.finripple")), SOURCE: row.event_id,
                     MASK_MAP: json.dumps({label: node for node, label in labels.table.items()}, sort_keys=True)})
    return pd.DataFrame(rows).sort_values(UNIT_ID).reset_index(drop=True)


LABEL = re.compile(r"COMPANY_[A-Z]+")


def label_in(text: str, prefix: str) -> str | None:
    """The mask label inside a model's company name (``COMPANY_C``, ``COMPANY_C (stock)``), if any."""
    found = re.search(re.escape(prefix) + r"[A-Z]+", str(text))
    return found.group(0) if found else None
