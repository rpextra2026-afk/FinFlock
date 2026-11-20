"""MASS's published rule: top 20% of the pool by signal, equal weight, long-only, rebalanced weekly."""

from __future__ import annotations

import pandas as pd

from finfm.common.config import Config
from finfm.portfolio.market import MarketView
from finfm.rivals.common import make_strategy


def weekly_rebalance_dates(sessions: pd.DatetimeIndex) -> set[pd.Timestamp]:
    """The last session of each ISO week is a decision date (fill at the next open)."""
    frame = pd.Series(sessions, index=sessions)
    iso = sessions.isocalendar()
    return set(frame.groupby([iso["year"].to_numpy(), iso["week"].to_numpy()]).last())


def mass_strategy(signal_by_date: dict[pd.Timestamp, pd.Series], rebalance_dates: set[pd.Timestamp],
                  cfg: Config, top_fraction: float | None = None):
    """Hold the last target between rebalance dates; on a rebalance date take the top share by signal."""
    fraction = float(cfg.get("rivals.mass.top_fraction")) if top_fraction is None else top_fraction
    state: dict[str, pd.Series] = {"target": pd.Series(dtype=float)}

    def rule(scores: pd.Series, date: pd.Timestamp, view: MarketView, held: pd.Series) -> pd.Series:
        if date in rebalance_dates:
            k = max(int(round(fraction * len(scores))), 1)
            top = scores.sort_values(ascending=False, kind="mergesort").index[:k]
            state["target"] = pd.Series(1.0 / k, index=top)
        return state["target"]

    inner = make_strategy(rule, signal_by_date)

    def target(date: pd.Timestamp, view: MarketView, held: pd.Series) -> pd.Series:
        return inner(date, view, held) if date in signal_by_date else state["target"]

    return target
