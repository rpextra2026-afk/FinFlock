"""One summary row per backtest: the same columns for every framework (numbers only)."""

from __future__ import annotations

import pandas as pd

from finflock.common.config import Config
from finflock.eval.portfolio_metrics import bridge, extended_metrics, max_drawdown, sharpe
from finflock.portfolio.execute import BacktestResult


def summary_row(name: dict, result: BacktestResult, net: pd.DataFrame, cfg: Config, factors: pd.DataFrame | None = None) -> dict:
    periods = int(cfg.get("eval.annualisation_days"))
    headline = f"{int(cfg.get('portfolio.scoring.headline_round_trip_bp'))}bp"
    cost = result.gross - net[headline]
    row = {**name, "n_days": len(result.gross), "n_days_invested": int((result.positions.abs().sum(axis=1) > 0).sum()),
           "sharpe_gross": sharpe(result.gross, periods), "sharpe_net_headline": sharpe(net[headline], periods),
           "max_drawdown_net_headline": max_drawdown(net[headline]), "delist_hits": int(result.delist_hits.sum()),
           "missing_return_hits": int(result.missing_return_hits.sum()), "risk_scale": float(result.risk_scale)}
    row.update(bridge(result.gross, cost, result.trades, periods))
    row.update(extended_metrics(result.gross, net, result.positions, result.trades, factors, cfg))
    return row
