"""The common net-of-cost block (spec 11.4 B): Sharpe, drawdown, CVaR, alpha, bridge, break-even, capacity."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd
import statsmodels.api as sm

from finfm.common.config import Config


HALF = 0.5  # one-way turnover counts each traded dollar once (buy and sell sides are equal)

MIN_STD = 1e-12  # a series this flat has no Sharpe ratio


def sharpe(returns: pd.Series, periods: int) -> float:
    return float(returns.mean() / returns.std() * np.sqrt(periods)) if returns.std() > MIN_STD else float("nan")


def max_drawdown(returns: pd.Series) -> float:
    wealth = (1.0 + returns).cumprod()
    return float((wealth / wealth.cummax() - 1.0).min())


def cvar(returns: pd.Series, level: float) -> float:
    """Mean of the worst ``1 - level`` share of daily returns (a negative number)."""
    cutoff = returns.quantile(1.0 - level)
    return float(returns[returns <= cutoff].mean())


def factor_alpha(returns: pd.Series, factors: pd.DataFrame, lag: int, periods: int) -> dict[str, float]:
    """Annualised alpha, its HAC t-statistic and the betas of an OLS on the factors."""
    data = pd.concat([returns.rename("y"), factors], axis=1).dropna()
    fit = sm.OLS(data["y"], sm.add_constant(data[factors.columns])).fit(cov_type="HAC", cov_kwds={"maxlags": lag})
    out = {"alpha_annual": float(fit.params["const"] * periods), "t_alpha": float(fit.tvalues["const"]), "n": int(fit.nobs),
           "resid_vol_annual": float(fit.resid.std() * np.sqrt(periods))}
    out.update({f"beta_{c}": float(fit.params[c]) for c in factors.columns})
    return out


def spanning_alpha(ours: pd.Series, theirs: pd.Series, lag: int, periods: int) -> dict[str, float]:
    """Intercept of our net return on a rival's net return: what the rival does not already span."""
    return factor_alpha(ours, theirs.rename("rival").to_frame(), lag, periods)


def one_way_turnover(trades: pd.DataFrame) -> pd.Series:
    """Daily one-way turnover: half the sum of absolute drift-adjusted trades."""
    return HALF * trades.abs().sum(axis=1)


def bridge(gross: pd.Series, cost: pd.Series, trades: pd.DataFrame, periods: int) -> dict[str, float]:
    """Gross-to-net bridge: annual return, vol, gross Sharpe, cost, net, net Sharpe, turnover, leverage."""
    net = gross - cost
    return {
        "annual_return_gross": float(gross.mean() * periods), "annual_vol": float(gross.std() * np.sqrt(periods)),
        "sharpe_gross": sharpe(gross, periods), "annual_cost": float(cost.mean() * periods),
        "annual_return_net": float(net.mean() * periods), "sharpe_net": sharpe(net, periods),
        "turnover_annual": float(one_way_turnover(trades).mean() * periods),
        "leverage_gross_mean": float(trades.abs().sum(axis=1).mean()),
    }


def break_even_round_trip_bp(gross: pd.Series, trades: pd.DataFrame, bp_to_fraction: float) -> float:
    """Round-trip cost in bp at which the mean net return is zero (fixed-cost model)."""
    traded = trades.abs().sum(axis=1).mean()
    return float(2.0 * gross.mean() / traded / bp_to_fraction) if traded > 0 else float("inf")


def capacity(trades: pd.DataFrame, adv_usd: pd.DataFrame, gross: pd.Series, sigma_daily: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Share of ADV traded and square-root-impact cost drag at each AUM level (spec 9.4)."""
    y = float(cfg.get("portfolio.impact.y"))
    rows = []
    aligned_adv = adv_usd.reindex(index=trades.index, columns=trades.columns)
    aligned_sigma = sigma_daily.reindex(index=trades.index, columns=trades.columns)
    for aum in cfg.get("portfolio.aum.capacity_levels_usd"):
        dollars = trades.abs() * float(aum)
        share = (dollars / aligned_adv)
        impact = (y * aligned_sigma * np.sqrt(share) * dollars).sum(axis=1) / float(aum)
        rows.append({"aum_usd": float(aum), "max_adv_share": float(share.max().max()), "mean_adv_share": float(share.stack().mean()),
                     "annual_return_after_impact": float((gross - impact).mean() * float(cfg.get("eval.annualisation_days")))})
    return pd.DataFrame(rows)


def window_metrics(net: pd.Series, windows: Mapping[str, tuple[str, str]], periods: int, level: float) -> pd.DataFrame:
    """Return, drawdown and CVaR inside named windows (stress periods)."""
    rows = []
    for name, (start, end) in windows.items():
        part = net.loc[start:end]
        if len(part) > 1:
            rows.append({"window": name, "n_days": len(part), "total_return": float((1 + part).prod() - 1),
                         "max_drawdown": max_drawdown(part), "cvar": cvar(part, level)})
    return pd.DataFrame(rows)


FACTOR_COLUMNS = ("mkt_rf", "smb", "hml", "rmw", "cma", "mom")  # market, Fama-French 5, momentum
RF_COLUMN = "rf"
MARKET_COLUMN = "mkt_rf"


def longest_run(flags: pd.Series) -> int:
    """Longest run of consecutive True values (used for the longest time under water)."""
    best = run = 0
    for flag in flags.to_numpy():
        run = run + 1 if flag else 0
        best = max(best, run)
    return best


def return_shape(net: pd.Series, periods: int, level: float, lag: int) -> dict[str, float]:
    """Shape and tail metrics of the daily net return series: PnL, risk-adjusted ratios, tails, hit rate."""
    wealth = (1.0 + net).cumprod()
    drawdown = float((wealth / wealth.cummax() - 1.0).min())
    annual = float(net.mean() * periods)
    below = net.clip(upper=0.0)
    downside = float(np.sqrt((below**2).mean()) * np.sqrt(periods))
    gains, losses = net[net > 0].sum(), net[net < 0].sum()
    centred = sm.OLS(net.to_numpy(), np.ones(len(net))).fit(cov_type="HAC", cov_kwds={"maxlags": lag}) if len(net) > lag else None
    return {
        "cum_net_return": float(wealth.iloc[-1] - 1.0), "annual_vol_net": float(net.std() * np.sqrt(periods)),
        "sortino_net": annual / downside if downside > MIN_STD else float("nan"), "downside_dev_annual": downside,
        "calmar_net": annual / abs(drawdown) if drawdown < 0 else float("nan"),
        "max_drawdown_days": float(longest_run(wealth < wealth.cummax())),
        "cvar_net": cvar(net, level), "var_net": float(net.quantile(1.0 - level)),
        "hit_rate_net": float((net > 0).mean()), "profit_factor_net": float(gains / abs(losses)) if losses < 0 else float("nan"),
        "skew_net": float(net.skew()), "kurtosis_net": float(net.kurt()), "best_day_net": float(net.max()), "worst_day_net": float(net.min()),
        "t_stat_mean_net": float(centred.tvalues[0]) if centred is not None else float("nan"),
    }


def holdings_shape(positions: pd.DataFrame) -> dict[str, float]:
    """How concentrated and how invested the target portfolios are (per fill date, then averaged)."""
    absolute = positions.abs()
    return {"net_exposure_mean": float(positions.sum(axis=1).mean()), "n_positions_mean": float((absolute > 0).sum(axis=1).mean()),
            "max_weight_mean": float(absolute.max(axis=1).mean())}


def extended_metrics(gross: pd.Series, net_by_cost: pd.DataFrame, positions: pd.DataFrame, trades: pd.DataFrame, factors: pd.DataFrame | None,
                     cfg: Config) -> dict[str, float]:
    """Every metric recorded for one backtest beyond the gross-to-net bridge (the papers' portfolio metrics; ``report.main_metrics`` marks the headline ones).

    Net series come from ``net_by_cost`` (one column per round-trip cost in bp). PnL in dollars is the sum of daily net returns times the headline AUM
    (constant AUM, not compounded). Alpha is the intercept of the daily net excess return on the market, the Fama-French 5 factors and momentum.
    """
    periods = int(cfg.get("eval.annualisation_days"))
    headline = f"{int(cfg.get('portfolio.scoring.headline_round_trip_bp'))}bp"
    net = net_by_cost[headline]
    out = return_shape(net, periods, float(cfg.get("eval.cvar_level")), int(cfg.get("eval.hac_lag")))
    out["total_net_pnl_usd"] = float(net.sum() * float(cfg.get("portfolio.aum.headline_usd")))
    # the rivals' own yardsticks: FinRipple reports daily return, Sharpe, maximum drawdown and win rate without costs; MASS the cumulative excess return over an index
    out.update({"mean_daily_return_gross": float(gross.mean()), "mean_daily_return_net": float(net.mean()),
                "max_drawdown_gross": max_drawdown(gross), "hit_rate_gross": float((gross > 0).mean())})
    out["break_even_round_trip_bp"] = break_even_round_trip_bp(gross, trades, float(cfg.get("portfolio.scoring.bp_to_fraction")))
    for column in net_by_cost.columns:
        out[f"sharpe_net_{column}"] = sharpe(net_by_cost[column], periods)
    out.update(holdings_shape(positions))
    if factors is not None and RF_COLUMN in factors.columns:
        aligned = factors.reindex(net.index)
        index_return = aligned[MARKET_COLUMN] + aligned[RF_COLUMN]  # the market's total daily return: the index MASS measures against
        out["cum_excess_vs_market"] = float((1.0 + net).prod() - (1.0 + index_return.fillna(0.0)).prod())
        fit = factor_alpha(net - aligned[RF_COLUMN], aligned[list(FACTOR_COLUMNS)], int(cfg.get("eval.hac_lag")), periods)
        out.update({"alpha_annual": fit["alpha_annual"], "t_alpha": fit["t_alpha"], "n_alpha_days": float(fit["n"]),
                    "appraisal_ratio": fit["alpha_annual"] / fit["resid_vol_annual"] if fit["resid_vol_annual"] > MIN_STD else float("nan"),
                    **{k: v for k, v in fit.items() if k.startswith("beta_")}})
    return out
