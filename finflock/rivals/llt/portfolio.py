"""LLT's published portfolio rule: equal-weighted long-short on the sign of the day's firm score.

Variants: 100% daily rebalancing (headline), 25% partial rebalancing, and a liquid-only screen. Scores of
one firm on one decision date are aggregated by mean (or the value of largest magnitude).
"""

from __future__ import annotations

import pandas as pd

from finflock.common.config import Config
from finflock.portfolio.market import MarketView
from finflock.rivals.common import (
    daily_scores,
    liquid_only,
    long_only,
    make_strategy,
    partial_step,
    sign_legs,
)


def llt_scores(signals: pd.DataFrame, aggregation: str = "mean") -> dict[pd.Timestamp, pd.Series]:
    """Firm scores per decision date from the signal table ``(decision_date, security_id, score)``."""
    table = daily_scores(signals, aggregation)
    return {d: part.set_index("security_id")["score"] for d, part in table.groupby("decision_date")}


def llt_strategy(
    scores_by_date: dict[pd.Timestamp, pd.Series], cfg: Config, *, rebalance: float, min_firms: int,
    screen: bool, long_leg_only: bool = False,
):
    """Target function for the executor implementing the published rule and its variants."""
    spec = cfg.get("rivals.llt")
    liquid = cfg.get("market.liquid_only")
    window = int(cfg.get("market.etf.dv_window_sessions"))

    def rule(scores: pd.Series, date: pd.Timestamp, view: MarketView, held: pd.Series) -> pd.Series:
        target = sign_legs(scores, float(spec["leg_weight"]), min_firms)
        if screen and len(target):
            target = liquid_only(target, view, float(liquid["min_price"]), float(liquid["min_median_dv_usd"]), window)
        if long_leg_only:
            target = long_only(target, float(spec["leg_weight"]) * 2.0)
        return partial_step(held, target, rebalance) if rebalance < 1.0 else target

    return make_strategy(rule, scores_by_date)
