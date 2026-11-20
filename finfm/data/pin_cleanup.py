"""Stricter candidate assets for name-only news (finance and world news); social source pins are never touched.

Why: a candidate list built from word matching offered the reader assets that the story is not about (a tennis
*match* offered Match Group, the verb *target* offered Target Corp, a stray "gold" offered gold). The reader cannot
repair a candidate that should never have been offered, so these rules remove such candidates before the model runs.

Rules (all deterministic, no model output):

1. company: kept only if the text names it by something other than a generic one-word name, i.e. a ticker
   (cashtag for 1-2 letters), a full name, or "Name (TICKER)". The generic one-word names are the ones that
   ``drop_generic_aliases`` removes from the prompt masker.
2. company: dropped when it is only the *actor* of a rating headline ("Goldman Sachs Upgrades Merck ...").
3. country / commodity: kept only if its gazetteer word occurs in the title or the text the reader will see.

Checked against the 100 hand-reviewed POC rows: about 60 % of the offered assets judged wrong were removed while
losing only a few of the assets judged right (see ``tests/test_pin_cleanup.py`` and the POC pack report).
"""
from __future__ import annotations

import json
import re
from collections import Counter

import pandas as pd

NAME_ONLY_FAMILIES = frozenset({"ext_company_news", "ext_world_news"})
# One-word company names that the lower-case-share heuristic wrongly calls generic (it counts lower-case social posts):
# they are distinctive, so they stay valid evidence and stay masked in the prompt. Every other flagged name
# (Target, Match, Northern, Ball, Pool, Block, Nasdaq, Visa, Mosaic, News, Southern, Capri) really is an ordinary word.
NOT_GENERIC_NAMES = frozenset({"adobe", "amgen", "ebay", "intel", "nvidia", "tesla"})
WORD = r"A-Za-z0-9"
MAX_TICKER_CHARS = 5  # longest ticker accepted in "(TICKER)" form
TOP_NODES_REPORTED = 20
RATING_WORD = (r"(?:strong\s+)?(?:buy|sell|hold|neutral|overweight|underweight|outperform|underperform|equal[- ]weight|"
               r"market[- ]perform|sector[- ]perform|peer[- ]perform|in[- ]line|positive|negative|accumulate|reduce|top\s+pick|rating)")
RATING_ACTOR = re.compile(
    r"^(?:UPDATE:\s*)?(?P<actor>[^:;|]{2,60}?)\s+(?:upgrades|downgrades|"
    rf"(?:maintains|reiterates)\s+(?:a\s+|an\s+|the\s+|its\s+)?{RATING_WORD}|"
    r"(?:initiates|resumes|assumes|starts)\s+(?:coverage|at|on|with)|"
    r"(?:raises|lowers|cuts|boosts|trims)\s+(?:the\s+)?(?:price\s+target|pt|target)|"
    r"(?:raises|lowers|cuts|boosts)\s+.{1,40}?\s+(?:price\s+target|pt))\b",
    re.IGNORECASE,
)


def is_rating_headline(title: str) -> bool:
    """An analyst rating or price-target headline ("Morgan Stanley Maintains Equal-Weight on X, Raises PT to $50")."""
    return RATING_ACTOR.match(str(title).strip()) is not None


def restore_distinctive_names(aliases: pd.DataFrame, prompt_aliases: pd.DataFrame) -> pd.DataFrame:
    """``prompt_aliases`` plus the distinctive one-word names the heuristic dropped by mistake."""
    back = aliases[(aliases["alias_type"] == "name") & aliases["alias"].str.casefold().isin(NOT_GENERIC_NAMES)]
    keys = ["node_id", "alias", "alias_type"]
    return pd.concat([prompt_aliases, back], ignore_index=True).drop_duplicates(keys).reset_index(drop=True)


def _gazetteer(table: pd.DataFrame, key: str, value: str, prefix: str) -> dict[str, list[re.Pattern]]:
    patterns: dict[str, list[re.Pattern]] = {}
    for _, row in table.iterrows():
        rx = re.compile(rf"(?<![{WORD}]){re.escape(str(row[value]))}(?![{WORD}])", re.IGNORECASE)
        patterns.setdefault(f"{prefix}{row[key]}", []).append(rx)
    return patterns


def ticker_in_brackets(text: str, node: str) -> bool:
    """``Visa (V)`` or ``(NYSE: V)``: an explicit ticker even when the node is a one- or two-letter ticker."""
    if not node.isalpha() or not node.isupper() or len(node) > MAX_TICKER_CHARS:
        return False
    return re.search(rf"[(\[]\s*(?:[A-Za-z]+\s*:\s*)?{re.escape(node)}\s*[)\]]", text) is not None


def rating_actor_nodes(title: str, finder) -> set[str]:
    """Companies that only *issue* a rating in a headline such as ``Goldman Sachs Upgrades Merck to Buy``."""
    match = RATING_ACTOR.match(title.strip())
    return set(finder(match.group("actor"))) if match else set()


def clean_pins(units: pd.DataFrame, prompt_maskers, countries: pd.DataFrame, commodities: pd.DataFrame,
               node_prefix: dict[str, str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return ``(units with cleaned pins_json, one row per removed pin)``. Input is not modified."""
    country_rx = _gazetteer(countries, "iso", "alias", node_prefix["country"])
    commodity_rx = _gazetteer(commodities, "tag", "keyword", node_prefix["commodity"])
    gazetteer = {**country_rx, **commodity_rx}
    out = units.copy()
    removed = []
    for index, row in units.iterrows():
        if row["family"] not in NAME_ONLY_FAMILIES:
            continue
        pins = json.loads(row["pins_json"])
        if not pins:
            continue
        title, text = str(row["title"]), str(row["text"])
        whole = title + "\n" + text
        finder = prompt_maskers.for_date(pd.Timestamp(row["available_date"])).find
        named = set(finder(whole))
        actors = rating_actor_nodes(title, finder)
        kept = []
        for pin in pins:
            node, kind = str(pin["node_id"]), pin.get("node_type")
            reason = None
            if kind == "company":
                if node not in named and not ticker_in_brackets(whole, node):
                    reason = "generic_name_only"
                elif node in actors:
                    reason = "rating_issuer_only"
            elif node in gazetteer and not any(rx.search(whole) for rx in gazetteer[node]):
                reason = "no_visible_mention"
            if reason:
                removed.append({"unit_id": row["unit_id"], "family": row["family"], "title": title,
                                "node_id": node, "reason": reason})
            else:
                kept.append(pin)
        if len(kept) != len(pins):
            out.at[index, "pins_json"] = json.dumps(kept, sort_keys=True)
    return out, pd.DataFrame(removed, columns=["unit_id", "family", "title", "node_id", "reason"])


def summarize(before: pd.DataFrame, after: pd.DataFrame, removed: pd.DataFrame) -> dict:
    def pinless(frame: pd.DataFrame) -> pd.Series:
        return frame["pins_json"].map(lambda s: not json.loads(s))
    return {
        "rows": int(len(before)),
        "rows_changed": int(removed["unit_id"].nunique()),
        "pins_removed": int(len(removed)),
        "removed_by_reason": dict(Counter(removed["reason"])),
        "removed_by_node_top": dict(Counter(removed["node_id"]).most_common(TOP_NODES_REPORTED)),
        "pinless_rows_before": int(pinless(before).sum()),
        "pinless_rows_after": int(pinless(after).sum()),
        "rows_with_pins_after": int((~pinless(after)).sum()),
    }
