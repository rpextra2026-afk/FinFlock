"""The appendix tables of the final run, all computed in the same run as the headline table (spec 11.4 B-C, metrics-final.md).

Every table has the metrics as rows and the backtests (or variants) as columns, so the report builder and the notebook print them the same way:
year-by-year net Sharpe and return (decay), bootstrap interval of the net Sharpe, the named stress windows, capacity at each AUM level, Sharpe
differences between variants with a studentised block-bootstrap interval, and the event-level rank IC of the forecasts. Numbers only, no verdicts.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from oryxis.common.config import Config
from oryxis.common.rng import make_rng
from oryxis.eval.lw_bootstrap import sharpe_difference_test
from oryxis.eval.portfolio_metrics import capacity, sharpe, window_metrics
from oryxis.eval.signal_metrics import ic_report
from oryxis.eval.stats import block_length, bootstrap_ci
from oryxis.framework.run import EventForecast

PREFIX = "appendix_"
MIN_EVENT_ASSETS = 3  # a rank correlation across fewer assets than this is not computed
ROW_COLUMN = "metric"


def _headline(cfg: Config) -> str:
    return f"{int(cfg.get('portfolio.scoring.headline_round_trip_bp'))}bp"


def yearly(net: dict[str, pd.Series], periods: int) -> dict[str, pd.DataFrame]:
    """Net Sharpe and total net return by calendar year, one column per backtest."""
    sharpes, returns = {}, {}
    for key, series in net.items():
        groups = series.groupby(series.index.year)
        sharpes[key] = groups.apply(lambda part: sharpe(part, periods))
        returns[key] = groups.apply(lambda part: float((1.0 + part).prod() - 1.0))
    return {f"{PREFIX}yearly_net_sharpe": pd.DataFrame(sharpes), f"{PREFIX}yearly_net_return": pd.DataFrame(returns)}


def sharpe_intervals(net: dict[str, pd.Series], cfg: Config) -> pd.DataFrame:
    """Net Sharpe with a circular block-bootstrap percentile interval."""
    periods, draws, level = int(cfg.get("eval.annualisation_days")), int(cfg.get("eval.bootstrap.draws")), float(cfg.get("eval.bootstrap.ci_level"))
    out = {}
    for key, series in net.items():
        values = series.to_numpy()
        point, low, high = bootstrap_ci(values, lambda x: float(x.mean() / x.std() * np.sqrt(periods)) if x.std() > 0 else float("nan"),
                                        block_length(len(values)), draws, level, make_rng("appendix", "sharpe_ci", key))
        out[key] = {"net_sharpe": point, "ci_low": low, "ci_high": high, "block_days": float(block_length(len(values)))}
    return pd.DataFrame(out)


def stress(net: dict[str, pd.Series], cfg: Config) -> pd.DataFrame:
    """Return, drawdown and CVaR inside the named stress windows of config ``eval.stress_windows`` that overlap the test window."""
    windows = {name: tuple(span) for name, span in dict(cfg.get("eval.stress_windows")).items()}
    periods, level = int(cfg.get("eval.annualisation_days")), float(cfg.get("eval.cvar_level"))
    columns = {}
    for key, series in net.items():
        table = window_metrics(series, windows, periods, level)
        flat = {f"{row['window']} | {metric}": row[metric] for _, row in table.iterrows() for metric in ("n_days", "total_return", "max_drawdown", "cvar")}
        columns[key] = flat
    return pd.DataFrame(columns)


def capacity_table(backtests: dict, rm, cfg: Config, treatment: str) -> pd.DataFrame:
    """Share of average daily volume traded and the return left after square-root impact, at every AUM level, per variant."""
    dollars = (rm.close * rm.volume).rolling(int(cfg.get("portfolio.impact.adv_window_days")), min_periods=1).mean().shift(1)
    sigma = rm.ret.rolling(int(cfg.get("portfolio.impact.vol_window_days")), min_periods=2).std().shift(1)
    columns = {}
    for key, result in backtests.items():
        if not key.endswith(f"/{treatment}"):
            continue
        table = capacity(result.trades, dollars, result.gross, sigma, cfg)
        columns[key.split("/")[0]] = {f"AUM {int(row['aum_usd']):,} | {metric}": row[metric]
                                      for _, row in table.iterrows() for metric in ("max_adv_share", "mean_adv_share", "annual_return_after_impact")}
    return pd.DataFrame(columns)


def sharpe_differences(net: dict[str, pd.Series], cfg: Config, treatment: str) -> pd.DataFrame:
    """Annualised Sharpe difference between pairs of variants (config ``eval.variant_pairs``) with the Ledoit-Wolf studentised block-bootstrap interval."""
    periods, draws, level = int(cfg.get("eval.annualisation_days")), int(cfg.get("eval.bootstrap.draws")), float(cfg.get("eval.bootstrap.ci_level"))
    columns = {}
    for first, second in cfg.get("eval.variant_pairs"):
        a, b = net.get(f"{first}/{treatment}"), net.get(f"{second}/{treatment}")
        if a is None or b is None or len(a) != len(b):
            continue
        test = sharpe_difference_test(a.to_numpy(), b.to_numpy(), periods, draws, level, make_rng("appendix", "sharpe_diff", first, second))
        columns[f"{first} minus {second}"] = {"sharpe_difference_annual": test.delta_annual, "ci_low": test.ci_low, "ci_high": test.ci_high,
                                              "p_value": test.p_value, "block_days": float(test.block)}
    return pd.DataFrame(columns)


def event_rank_ic(forecasts: dict[str, list[EventForecast]], rm, cfg: Config) -> pd.DataFrame:
    """Rank correlation, across the assets of each event, between the predicted day-1 increment and the realised day-1 return, summarised over events.

    This is an event-level (ripple) rank IC, not a daily-universe IC: it measures whether the forecast orders the linked assets of one event correctly.
    """
    from scipy.stats import spearmanr

    columns = {}
    for variant, items in forecasts.items():
        ics = {}
        for forecast in items:
            day1 = forecast.day0 + 1
            if day1 >= len(rm.ret) or len(forecast.assets) < MIN_EVENT_ASSETS:
                continue
            predicted = np.asarray(forecast.increments)[0]
            realised = rm.ret.iloc[day1, forecast.assets].to_numpy()
            keep = np.isfinite(predicted) & np.isfinite(realised)
            if keep.sum() >= MIN_EVENT_ASSETS and len(set(predicted[keep])) > 1 and len(set(realised[keep])) > 1:
                ics[forecast.event_id] = float(spearmanr(predicted[keep], realised[keep])[0])
        if ics:
            report = ic_report(pd.Series(ics), int(cfg.get("eval.hac_lag")))
            columns[variant] = {"rank_ic": report["ic"], "icir": report["icir"], "t_naive": report["t_naive"], "t_newey_west": report["t_newey_west"],
                                "n_events": float(report["n_dates"])}
    return pd.DataFrame(columns)


def appendix_tables(cfg: Config, rm, forecasts: dict, backtests: dict, net: dict) -> dict[str, pd.DataFrame]:
    """All appendix tables of the run; a table that cannot be built is left out and the reason is returned by the caller's diagnostics."""
    headline, treatment = _headline(cfg), str(cfg.get("eval.appendix_treatment"))
    series = {key: frame[headline] for key, frame in net.items()}
    periods = int(cfg.get("eval.annualisation_days"))
    out: dict[str, pd.DataFrame] = {}
    out.update(yearly(series, periods))
    out[f"{PREFIX}sharpe_interval"] = sharpe_intervals(series, cfg)
    out[f"{PREFIX}stress_windows"] = stress(series, cfg)
    out[f"{PREFIX}capacity"] = capacity_table(backtests, rm, cfg, treatment)
    out[f"{PREFIX}sharpe_differences"] = sharpe_differences(series, cfg, treatment)
    out[f"{PREFIX}event_rank_ic"] = event_rank_ic(forecasts, rm, cfg)
    return {name: frame.rename_axis(ROW_COLUMN).reset_index() for name, frame in out.items() if not frame.empty}
