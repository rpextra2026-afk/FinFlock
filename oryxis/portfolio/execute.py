"""Backtest executor shared by all four frameworks (spec 9.4): same fills, same drift, same costs.

Decision on date ``d`` (after the close, using ``Market.asof(d)`` only) is filled at the open of the
next session ``f`` and held to the following open. Trades are measured against the drift-adjusted
previous holdings. Positions never depend on the scoring costs, so costs are applied afterwards to
the recorded trades (several cost models from one run).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from oryxis.portfolio.market import Market, MarketView

HALF = 0.5

TargetFn = Callable[[pd.Timestamp, MarketView, pd.Series], pd.Series]
CostModel = Callable[[pd.Series, pd.Timestamp], float]


@dataclass
class BacktestResult:
    fill_dates: pd.DatetimeIndex
    gross: pd.Series  # portfolio return per fill date before costs
    cash_weight: pd.Series
    positions: pd.DataFrame  # target weights by fill date
    trades: pd.DataFrame  # drift-adjusted trades by fill date
    delist_hits: pd.Series  # delisted holdings met per fill date
    missing_return_hits: pd.Series  # held assets with a missing return (treated as 0, counted)
    risk_scale: float = 1.0  # factor that put this backtest on the common risk target (1.0 = not scaled)


def run_backtest(
    market: Market,
    decision_dates: Sequence[pd.Timestamp],
    target_fn: TargetFn,
    *,
    delist_return: float | None,
    risk_free_daily: pd.Series | None = None,
) -> BacktestResult:
    """Run the receding-horizon strategy over consecutive decision dates."""
    held = pd.Series(dtype=float)
    fills, gross, cash, positions, trades, delist, missing = [], [], [], [], [], [], []
    panel_end = market.sessions[-1]
    for date in decision_dates:
        fill = market.next_session(date)
        if fill is None or fill >= panel_end:  # the open-to-open return of the last session needs a next open that does not exist: stop, never score it
            break
        target = target_fn(date, market.asof(date), held).astype(float)
        names = held.index.union(target.index)
        trade = target.reindex(names, fill_value=0.0) - held.reindex(names, fill_value=0.0)
        returns = market.fill_return(fill).reindex(target.index)
        nan_mask = returns.isna() & (target != 0)
        last = market.last_trading_date.reindex(target.index)
        delisted = nan_mask & last.lt(panel_end).fillna(False) & last.le(fill).fillna(False)  # still trading at the panel end = not delisted (the panel end is a cut, not an exit)
        returns = returns.copy()
        returns[delisted] = delist_return if delist_return is not None else 0.0
        missing_now = nan_mask & ~delisted
        returns = returns.fillna(0.0)
        rf = 0.0 if risk_free_daily is None else float(risk_free_daily.get(fill, 0.0))
        cash_w = 1.0 - target.sum()
        values = target * (1.0 + returns)
        total = values.sum() + cash_w * (1.0 + rf)
        gross.append(total - 1.0)
        held = values / total
        fills.append(fill)
        cash.append(cash_w)
        positions.append(target.rename(fill))
        trades.append(trade.rename(fill))
        delist.append(int(delisted.sum()))
        missing.append(int(missing_now.sum()))
    index = pd.DatetimeIndex(fills)
    return BacktestResult(
        fill_dates=index,
        gross=pd.Series(gross, index=index),
        cash_weight=pd.Series(cash, index=index),
        positions=pd.DataFrame(positions).fillna(0.0),
        trades=pd.DataFrame(trades).fillna(0.0),
        delist_hits=pd.Series(delist, index=index),
        missing_return_hits=pd.Series(missing, index=index),
    )


def net_returns(result: BacktestResult, models: Mapping[str, CostModel]) -> pd.DataFrame:
    """Net return series, one column per cost model, from one set of recorded trades."""
    costs = {
        name: pd.Series(
            [model(result.trades.loc[d], d) for d in result.fill_dates], index=result.fill_dates
        )
        for name, model in models.items()
    }
    return pd.DataFrame({name: result.gross - cost for name, cost in costs.items()})


def turnover(result: BacktestResult) -> pd.Series:
    """One-way turnover per day: half the sum of absolute drift-adjusted trades."""
    return HALF * result.trades.abs().sum(axis=1)


def vol_match(returns: pd.Series, target_annual_vol: float, annualisation_days: int) -> pd.Series:
    """Scale a return series to the common annualised volatility used for reporting."""
    realised = returns.std() * np.sqrt(annualisation_days)
    return returns * (target_annual_vol / realised)


def scale_to_risk_target(result: BacktestResult, target_annual_vol: float, annualisation_days: int, risk_free_daily: pd.Series | None = None) -> BacktestResult:
    """Put a backtest on the common risk target by scaling its WEIGHTS (so costs scale with the trades, not just the returns).

    With weights c * w and the rest in cash: gross' = c * gross + (1 - c) * rf, trades' = c * trades, cash' = 1 - c * (1 - cash). The factor is
    target / realised annual volatility of the gross series; it is stored in ``risk_scale`` so the leverage it implies is visible."""
    realised = float(result.gross.std() * np.sqrt(annualisation_days))
    if not np.isfinite(realised) or realised <= 0.0:
        return result
    c = target_annual_vol / realised
    rf = pd.Series(0.0, index=result.gross.index) if risk_free_daily is None else risk_free_daily.reindex(result.gross.index).fillna(0.0)
    return BacktestResult(result.fill_dates, c * result.gross + (1.0 - c) * rf, 1.0 - c * (1.0 - result.cash_weight), c * result.positions, c * result.trades,
                          result.delist_hits, result.missing_return_hits, risk_scale=c)
