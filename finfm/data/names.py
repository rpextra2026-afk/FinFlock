"""Company aliases (name, shortened name, ticker) from the SEC company list, for masking and pinning.

The SEC list holds today's names, so it is a point-in-time approximation: a company that was renamed or has left the
list is matched by its current name only (flagged ``source=sec_current``). Former names are not available from this free source.
"""

from __future__ import annotations

import re
from collections import Counter

import pandas as pd

from finfm.common.config import Config

ALIAS_COLUMNS = ["node_id", "alias", "alias_type", "valid_from", "valid_to"]
WORD = re.compile(r"[A-Za-z0-9&']+")


def shortened(name: str, cfg: Config) -> str:
    """The name without trailing legal words, e.g. ``Apple Inc.`` -> ``Apple``."""
    words = WORD.findall(name)
    drop = {w.lower() for w in cfg.get("sources.names.strip_suffixes")}
    while words and words[-1].lower() in drop:
        words.pop()
    return " ".join(words)


def build_aliases(sec: pd.DataFrame, tickers: list[str], cfg: Config) -> pd.DataFrame:
    """Alias table for the given tickers (node id = ticker). ``sec`` has ``ticker`` and ``title`` columns."""
    by_ticker = {str(r.ticker).upper(): str(r.title) for r in sec.itertuples()}
    min_chars = int(cfg.get("sources.names.min_clean_chars"))
    rows = []
    for ticker in tickers:
        title = by_ticker.get(ticker.upper()) or by_ticker.get(ticker.upper().replace(".", "-"))
        rows.append((ticker, ticker, "ticker", None, None))
        if not title:
            continue
        nice = title.title() if title.isupper() else title
        rows.append((ticker, nice, "name", None, None))
        short = shortened(nice, cfg)
        if short and short.casefold() != nice.casefold() and len(short) >= min_chars:
            rows.append((ticker, short, "name", None, None))
    return pd.DataFrame(rows, columns=ALIAS_COLUMNS).drop_duplicates(["node_id", "alias", "alias_type"]).reset_index(drop=True)


def drop_generic_aliases(aliases: pd.DataFrame, texts: pd.Series, cfg: Config) -> pd.DataFrame:
    """Remove one-word name aliases that are mostly ordinary lower-case words in the news (``Block``, ``News``).

    The test uses the corpus itself: the share of a word's occurrences written in lower case. Tickers and multi-word
    names are untouched.
    """
    spec = cfg.get("sources.generic_alias")
    blocked = {str(name).casefold() for name in spec.get("blocked_one_word_names", [])}
    sample = texts.head(int(spec["sample_items"]))
    counts: Counter[str] = Counter(w for t in sample for w in WORD.findall(str(t)))
    drop = []
    for row in aliases.itertuples():
        if row.alias_type != "name" or " " in row.alias:
            continue
        if row.alias.casefold() in blocked:
            drop.append(row.Index)
            continue
        lower, cap = counts.get(row.alias.lower(), 0), counts.get(row.alias[:1].upper() + row.alias[1:].lower(), 0)
        total = lower + cap
        if total >= int(spec["min_count"]) and lower / total > float(spec["max_lowercase_share"]):
            drop.append(row.Index)
    kept = aliases.drop(index=drop).reset_index(drop=True)
    variants = []
    for node, names in spec.get("full_name_variants", {}).items():
        source = kept[(kept["node_id"] == node) & (kept["alias_type"] == "name") & kept["alias"].str.contains(" ", regex=False)]
        if source.empty:
            continue
        for name in names:
            row = source.iloc[0].copy()
            row["alias"] = str(name)
            variants.append(row)
    if variants:
        kept = pd.concat([kept, pd.DataFrame(variants)], ignore_index=True)
        kept = kept.drop_duplicates(["node_id", "alias", "alias_type"]).reset_index(drop=True)
    return kept
