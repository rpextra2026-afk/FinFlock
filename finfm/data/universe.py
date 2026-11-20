"""Layers, ETF eligibility and the liquid-only flag (spec 3.3).

Layers are terciles of market cap among the S&P 500 members at each monthly rebalance, fixed between
rebalances. No price or volume screen is applied by default; ``liquid_only`` is a robustness flag.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finfm.common.binning import rank_bins
from finfm.common.config import Config


def monthly_rebalance_dates(sessions: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """First session of each calendar month."""
    periods = sessions.to_period("M")
    first = pd.Series(sessions, index=sessions).groupby(periods).first()
    return pd.DatetimeIndex(first.values)


def assign_layers(mcap: pd.Series, labels: list[str]) -> pd.Series:
    """Tercile labels by rank of market cap; securities without a market cap get no layer."""
    bins = rank_bins(mcap, len(labels))
    return bins.map(dict(enumerate(labels)))


def liquid_flag(close_raw: pd.Series, median_dv: pd.Series, cfg: Config) -> pd.Series:
    """True when ``close >= min_price`` and the 60-day median dollar volume reaches the floor."""
    return (close_raw >= float(cfg.get("market.liquid_only.min_price"))) & (
        median_dv >= float(cfg.get("market.liquid_only.min_median_dv_usd"))
    )


def build_universe_table(
    members_by_rebalance: dict[pd.Timestamp, frozenset[str]],
    mcap: pd.DataFrame,
    dv: pd.DataFrame,
    close_raw: pd.DataFrame,
    cfg: Config,
) -> pd.DataFrame:
    """Rows ``(rebalance_date, security_id, layer, mktcap_usd, adv60_usd, passed_screen, layer_known)``."""
    window = int(cfg.get("market.etf.dv_window_sessions"))
    labels = list(cfg.get("market.layers.labels"))
    rows = []
    for date, members in sorted(members_by_rebalance.items()):
        ids = sorted(members)
        cap = mcap.loc[date].reindex(ids)
        layer = assign_layers(cap, labels)
        adv = dv.loc[:date].iloc[-window - 1 : -1].reindex(columns=ids)
        median_dv = adv.median()
        passed = liquid_flag(close_raw.shift(1).loc[date].reindex(ids), median_dv, cfg)
        for security in ids:
            rows.append({
                "rebalance_date": date, "security_id": security, "layer": layer[security],
                "mktcap_usd": cap[security], "adv60_usd": float(adv[security].mean()) if len(adv) else np.nan,
                "passed_screen": bool(passed[security]), "layer_known": bool(pd.notna(cap[security])),
            })
    return pd.DataFrame(rows)


def eligible_etfs(
    names: pd.Series, first_trade: pd.Series, dv: pd.DataFrame, date: pd.Timestamp,
    sessions: pd.DatetimeIndex, cfg: Config,
) -> list[str]:
    """ETF context nodes at a rebalance: listed long enough, liquid enough, not leveraged or inverse."""
    spec = cfg.get("market.etf")
    tokens = [t.upper() for t in spec["exclude_name_tokens"]]
    window = int(spec["dv_window_sessions"])
    median_dv = dv.loc[:date].iloc[-window - 1 : -1].median()
    today = sessions.searchsorted(pd.Timestamp(date))
    keep = []
    for security in names.index:
        if any(token in str(names[security]).upper() for token in tokens):
            continue
        listed = today - sessions.searchsorted(pd.Timestamp(first_trade[security]))
        if listed < int(spec["min_listing_sessions"]):
            continue
        if not median_dv.get(security, 0.0) >= float(spec["min_median_dv_usd"]):
            continue
        keep.append(security)
    return keep
