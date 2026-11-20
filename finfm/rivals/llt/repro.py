"""LLT's own yardsticks (spec 10.2): hit rates, leg returns, Sharpe, drift regression, turnover, by year.

The reproduction pattern to look for on our data: high gross Sharpe that falls sharply with turnover and
cost, and a short leg stronger than the long. Absolute numbers are never compared with the paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finfm.portfolio.execute import BacktestResult


MIN_OBSERVATIONS = 3


def leg_returns(result: BacktestResult, returns: pd.DataFrame) -> pd.DataFrame:
    """Daily return of the long leg, the short leg and the long-short from recorded positions."""
    positions = result.positions.reindex(columns=returns.columns, fill_value=0.0)
    aligned = returns.reindex(positions.index)
    long_leg = (positions.clip(lower=0.0) * aligned).sum(axis=1)
    short_leg = (positions.clip(upper=0.0) * aligned).sum(axis=1)
    return pd.DataFrame({"long": long_leg, "short": short_leg, "long_short": long_leg + short_leg})


def annual_sharpe(daily: pd.Series, periods: int) -> float:
    return float(daily.mean() / daily.std() * np.sqrt(periods)) if daily.std() > 0 else float("nan")


def hit_rate(scores: pd.Series, forward: pd.Series) -> float:
    """Share of scored firm-days where the sign of the score matches the sign of the forward return."""
    both = pd.concat([scores, forward], axis=1, keys=["s", "r"]).dropna()
    both = both[both["s"] != 0]
    return float((np.sign(both["s"]) == np.sign(both["r"])).mean()) if len(both) else float("nan")


def drift_regression(scores: pd.Series, forward: pd.Series) -> dict[str, float]:
    """OLS of the forward return on the score: slope, t-statistic (heteroskedasticity-robust) and n."""
    both = pd.concat([scores, forward], axis=1, keys=["x", "y"]).dropna()
    n = len(both)
    if n < MIN_OBSERVATIONS or both["x"].nunique() < 2:
        return {"slope": float("nan"), "t": float("nan"), "n": n}
    x = both["x"].to_numpy() - both["x"].mean()
    y = both["y"].to_numpy() - both["y"].mean()
    slope = float((x * y).sum() / (x * x).sum())
    resid = y - slope * x
    se = float(np.sqrt((x * x * resid * resid).sum()) / (x * x).sum())
    return {"slope": slope, "t": slope / se if se > 0 else float("nan"), "n": n}


def yearly_sharpe(daily: pd.Series, periods: int) -> pd.Series:
    return daily.groupby(daily.index.year).apply(lambda s: annual_sharpe(s, periods))
