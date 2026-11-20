"""FinRipple's published rule: each morning rank stocks by predicted impact, top 10% long, bottom 10% short."""

from __future__ import annotations

import pandas as pd

from finfm.common.config import Config
from finfm.portfolio.market import MarketView
from finfm.rivals.common import daily_scores, make_strategy, top_bottom_legs


def finripple_scores(signals: pd.DataFrame, aggregation: str = "mean") -> dict[pd.Timestamp, pd.Series]:
    """Per decision date, one predicted impact per stock (mean or sum of that day's scores)."""
    table = daily_scores(signals, aggregation)
    return {d: part.set_index("security_id")["score"] for d, part in table.groupby("decision_date")}


def finripple_strategy(scores_by_date: dict[pd.Timestamp, pd.Series], cfg: Config, top_fraction: float | None = None):
    spec = cfg.get("rivals.finripple")
    fraction = float(spec["top_fraction"]) if top_fraction is None else top_fraction

    def rule(scores: pd.Series, date: pd.Timestamp, view: MarketView, held: pd.Series) -> pd.Series:
        return top_bottom_legs(scores, fraction, float(spec["leg_weight"]))

    return make_strategy(rule, scores_by_date)
