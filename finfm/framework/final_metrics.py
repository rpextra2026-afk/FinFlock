"""Metrics of the final framework run that the older appendix did not have: the full cost grid, paired Sharpe differences between every pair of variants (and rivals)
with a Holm adjustment per family, deflated Sharpe, PBO, CPCV, event rank IC at several horizons with ICIR, the sheaf's linked-stock out-of-sample R2 per dial, and a
checklist of which required metric reaches the final table. Numbers only; no verdict words.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import norm, spearmanr

from finfm.common.config import Config
from finfm.common.rng import make_rng
from finfm.eval.dsr_pbo import _block_sums, _sharpes, cpcv_splits, deflated_sharpe, probability_of_backtest_overfitting
from finfm.eval.lw_bootstrap import sharpe_difference_test
from finfm.eval.portfolio_metrics import sharpe
from finfm.eval.signal_metrics import ic_report
from finfm.eval.stats import block_length, bootstrap_ci, holm
from finfm.portfolio.costs import fixed_cost

MIN_EVENT_ASSETS = 3
PBO_BLOCK_DAYS = 4  # at least this many days per PBO partition
MIN_TRIAL_DAYS = 64  # PBO / CPCV need this many aligned days


def cost_grid_table(backtests: Mapping[str, object], cfg: Config) -> pd.DataFrame:
    """Net Sharpe and net annual return at every round-trip cost of ``portfolio.scoring.round_trip_bp_grid_full`` (0 to 40 bp), from the recorded trades."""
    periods = int(cfg.get("eval.annualisation_days"))
    rows = []
    for key, result in backtests.items():
        for bp in cfg.get("portfolio.scoring.round_trip_bp_grid_full"):
            cost = pd.Series([fixed_cost(result.trades.loc[d].to_numpy(), float(bp), 1.0, cfg) for d in result.fill_dates], index=result.fill_dates)
            net = result.gross - cost
            rows.append({"key": key, "round_trip_bp": float(bp), "sharpe_net": sharpe(net, periods), "annual_return_net": float(net.mean() * periods), "annual_cost": float(cost.mean() * periods)})
    return pd.DataFrame(rows)


def paired_sharpe_table(series: Mapping[str, pd.Series], cfg: Config, family: str, pairs: Sequence[tuple[str, str]] | None = None) -> pd.DataFrame:
    """Ledoit-Wolf paired Sharpe differences with a date-block bootstrap interval, over every pair of ``series`` (or the given pairs); Holm-adjusted within ``family``."""
    periods, draws, level = int(cfg.get("eval.annualisation_days")), int(cfg.get("eval.bootstrap.draws")), float(cfg.get("eval.bootstrap.ci_level"))
    names = list(series)
    todo = list(pairs) if pairs is not None else list(combinations(names, 2))
    rows = []
    for first, second in todo:
        if first not in series or second not in series:
            continue
        both = pd.concat([series[first].rename("a"), series[second].rename("b")], axis=1).dropna()
        if len(both) < MIN_TRIAL_DAYS:
            continue
        test = sharpe_difference_test(both["a"].to_numpy(), both["b"].to_numpy(), periods, draws, level, make_rng("final", "sharpe_diff", family, first, second))
        rows.append({"family": family, "first": first, "second": second, "sharpe_difference_annual": test.delta_annual, "ci_low": test.ci_low, "ci_high": test.ci_high,
                     "p_value": test.p_value, "block_days": int(test.block), "n_days": int(len(both))})
    table = pd.DataFrame(rows)
    if len(table):
        adjusted = holm({f"{r.first}|{r.second}": float(r.p_value) for r in table.itertuples()})
        table["p_holm"] = [adjusted[f"{r.first}|{r.second}"] for r in table.itertuples()]
    return table


def trial_matrix(series: Mapping[str, pd.Series]) -> tuple[np.ndarray, list[str]]:
    """Time x configurations matrix of daily net returns on the days every configuration has."""
    frame = pd.concat(dict(series), axis=1).dropna()
    return frame.to_numpy(dtype=float), [str(c) for c in frame.columns]


def deflated_pbo_cpcv(series: Mapping[str, pd.Series], cfg: Config) -> dict[str, pd.DataFrame]:
    """Deflated Sharpe of each configuration against all configurations tried in this run, PBO (CSCV) and CPCV of the in-sample-best choice.

    The trials are the configurations of this run (variants x headline sets at one delisting rule); the number is printed with the table."""
    matrix, names = trial_matrix(series)
    out: dict[str, pd.DataFrame] = {}
    if matrix.shape[0] < MIN_TRIAL_DAYS or matrix.shape[1] < 2:
        return out
    gamma = float(cfg.get("eval.dsr.euler_gamma"))
    sharpes = matrix.mean(axis=0) / matrix.std(axis=0, ddof=1)
    out["deflated_sharpe"] = pd.DataFrame([{"config": n, **deflated_sharpe(matrix[:, i], sharpes, gamma)} for i, n in enumerate(names)])
    partitions = min(int(cfg.get("eval.pbo.partitions")), (matrix.shape[0] // PBO_BLOCK_DAYS) // 2 * 2)
    pbo = probability_of_backtest_overfitting(matrix, partitions)
    out["pbo"] = pd.DataFrame([{**pbo, "partitions": partitions, "trials": matrix.shape[1], "days": matrix.shape[0]}])
    groups, test_groups, embargo = int(cfg.get("eval.cpcv.groups")), int(cfg.get("eval.cpcv.test_groups")), int(cfg.get("eval.cpcv.embargo_sessions"))
    rows = []
    for train, test in cpcv_splits(matrix.shape[0], groups, test_groups, embargo):
        if len(train) < 2 or len(test) < 2:
            continue
        best = int(np.argmax(matrix[train].mean(axis=0) / matrix[train].std(axis=0, ddof=1)))
        part = matrix[test, best]
        rows.append({"chosen": names[best], "oos_sharpe_per_day": float(part.mean() / part.std(ddof=1))})
    split = pd.DataFrame(rows)
    if len(split):
        out["cpcv"] = pd.DataFrame([{"splits": len(split), "oos_sharpe_annual_mean": float(split["oos_sharpe_per_day"].mean() * np.sqrt(int(cfg.get("eval.annualisation_days")))),
                                     "oos_sharpe_annual_median": float(split["oos_sharpe_per_day"].median() * np.sqrt(int(cfg.get("eval.annualisation_days")))),
                                     "share_negative": float((split["oos_sharpe_per_day"] < 0).mean()), "trials": matrix.shape[1]}])
    return out


def event_rank_ic_horizons(forecasts: Mapping[str, list], rm, cfg: Config) -> pd.DataFrame:
    """Rank correlation across the assets of one event between the predicted cumulative increment over ``h`` sessions after day 0 and the realised cumulative return,
    for h in ``eval.ic_horizons``; mean over events, ICIR, Newey-West t and its p-value (Holm-adjusted within the rank_ic family)."""
    rows = []
    for variant, items in forecasts.items():
        for h in cfg.get("eval.ic_horizons"):
            ics = {}
            for forecast in items:
                d1 = forecast.day0 + 1
                if d1 + int(h) > len(rm.ret) or len(forecast.assets) < MIN_EVENT_ASSETS or np.asarray(forecast.increments).shape[0] < int(h):
                    continue
                predicted = np.asarray(forecast.increments)[: int(h)].sum(axis=0)
                realised = (1.0 + rm.ret.iloc[d1 : d1 + int(h), forecast.assets]).prod(axis=0).to_numpy() - 1.0
                ok = np.isfinite(predicted) & np.isfinite(realised)
                if ok.sum() >= MIN_EVENT_ASSETS and len(set(predicted[ok])) > 1 and len(set(realised[ok])) > 1:
                    ics[forecast.event_id] = float(spearmanr(predicted[ok], realised[ok])[0])
            if ics:
                rep = ic_report(pd.Series(ics), int(cfg.get("eval.hac_lag")))
                t = rep["t_newey_west"]
                rows.append({"variant": variant, "horizon_days": int(h), "rank_ic": rep["ic"], "icir": rep["icir"], "t_newey_west": t, "n_events": rep["n_dates"],
                             "p_value": float(2 * (1 - norm.cdf(abs(t)))) if np.isfinite(t) else float("nan")})
    table = pd.DataFrame(rows)
    if len(table):
        adjusted = holm({f"{r.variant}|{r.horizon_days}": float(r.p_value) for r in table.itertuples() if np.isfinite(r.p_value)})
        table["p_holm"] = [adjusted.get(f"{r.variant}|{r.horizon_days}", float("nan")) for r in table.itertuples()]
    return table


def sheaf_r2_per_dial(all_metrics: pd.DataFrame, method: str = "sheaf") -> pd.DataFrame:
    """Linked-stock out-of-sample R2 per dial (ret, vol, volu) with its interval, from the build's ALL_METRICS (view 'linked')."""
    part = all_metrics[(all_metrics["slice"] == "view") & (all_metrics["metric"] == "r2_os") & (all_metrics["method"] == method) & all_metrics["view"].astype(str).str.contains("link")]
    return part[["tape", "dial", "view", "metric", "value", "lo", "hi", "n"]].reset_index(drop=True)


ROLLING_VOL_DAYS = 63  # window of the rolling realised volatility
POSITION_QUANTILES = (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99)  # of the absolute target weights, for figures
ROLLING_LOW_QUANTILE = 0.05
ROLLING_HIGH_QUANTILE = 0.95
MIN_ACCURACY_PAIRS = 30  # event-asset pairs needed for a dial-accuracy row


def realised_vol_table(backtests: Mapping[str, object], nets: Mapping[str, pd.DataFrame], cfg: Config) -> pd.DataFrame:
    """Realised annual volatility of each backtest against the risk target ``portfolio.vol_match.target_annual_vol``.

    ``unscaled_annual_vol`` is what the optimiser's own weights realised before the common scale ``risk_scale`` was applied (the number the sheaf's variance inputs are responsible for);
    gross and net are after the scale (the gross volatility equals the target by construction on the window it is scaled on, so the rolling quantiles and the net row show the spread around it).
    Intervals: circular block bootstrap of the daily returns."""
    periods, draws, level = int(cfg.get("eval.annualisation_days")), int(cfg.get("eval.bootstrap.draws")), float(cfg.get("eval.bootstrap.ci_level"))
    target = float(cfg.get("portfolio.vol_match.target_annual_vol"))
    headline = f"{int(cfg.get('portfolio.scoring.headline_round_trip_bp'))}bp"
    annual = lambda x: float(np.std(x, ddof=1) * np.sqrt(periods)) if len(x) > 1 else float("nan")  # noqa: E731
    rows = []
    for key, result in backtests.items():
        gross = result.gross.dropna().to_numpy(float)
        net = nets[key][headline].dropna().to_numpy(float)
        if len(gross) < MIN_TRIAL_DAYS:
            continue
        rng = make_rng("final", "realised_vol", key)
        g, g_lo, g_hi = bootstrap_ci(gross, annual, block_length(len(gross)), draws, level, rng)
        n, n_lo, n_hi = bootstrap_ci(net, annual, block_length(len(net)), draws, level, make_rng("final", "realised_vol_net", key))
        rolling = pd.Series(gross).rolling(ROLLING_VOL_DAYS).std().dropna() * np.sqrt(periods)
        scale = float(getattr(result, "risk_scale", 1.0))
        rows.append({"key": key, "target_annual_vol": target, "risk_scale": scale, "unscaled_annual_vol": target / scale if scale > 0 else float("nan"),
                     "realised_annual_vol_gross": g, "gross_ci_low": g_lo, "gross_ci_high": g_hi, "realised_annual_vol_net": n, "net_ci_low": n_lo, "net_ci_high": n_hi,
                     "ratio_net_to_target": n / target, "rolling_vol_p05": float(rolling.quantile(ROLLING_LOW_QUANTILE)) if len(rolling) else float("nan"),
                     "rolling_vol_median": float(rolling.median()) if len(rolling) else float("nan"), "rolling_vol_p95": float(rolling.quantile(ROLLING_HIGH_QUANTILE)) if len(rolling) else float("nan"),
                     "rolling_window_days": ROLLING_VOL_DAYS, "n_days": int(len(gross))})
    return pd.DataFrame(rows)


def dial_accuracy_table(forecasts: Mapping[str, Sequence], labels: pd.DataFrame, columns: Sequence[str], cfg: Config, pairs_out: dict | None = None) -> pd.DataFrame:
    """Accuracy of the volatility and volume dials as the optimiser reads them, on the test events.

    ``vol``  : the log volatility ratio that scales the variances (covariance x exp(2 vol)), against the realised log volatility ratio of the label window (d01).
    ``volu`` : the log volume ratio that enters the cost multiplier, against the realised log dollar-volume ratio.
    ``cost_input``: vol - alpha x volu, the exponent of the cost multiplier exp(vol - alpha volu), against the realised exponent.
    Each row: pairs, events, bias (predicted minus realised), RMSE, correlation, out-of-sample R2 against the dial the optimiser would use with no information (zero), and the slope of the
    realised on the predicted; intervals from a bootstrap that resamples whole events (pairs of one event are not independent)."""
    alpha = float(cfg.get("portfolio.impact.alpha_headline"))
    draws, level = int(cfg.get("eval.bootstrap.draws")), float(cfg.get("eval.bootstrap.ci_level"))
    real = labels.set_index(["event_id", "security_id"])[["log_vol_ratio", "log_dv_ratio"]]
    rows = []
    for variant, items in forecasts.items():
        recs = []
        for f in items:
            ids = [columns[i] for i in f.assets]
            for j, sec in enumerate(ids):
                if (f.event_id, sec) in real.index:
                    y = real.loc[(f.event_id, sec)]
                    recs.append((f.event_id, float(np.asarray(f.vol_dial)[j]), float(np.asarray(f.volume_dial)[j]), float(y["log_vol_ratio"]), float(y["log_dv_ratio"])))
        frame = pd.DataFrame(recs, columns=["event_id", "pv", "pu", "rv", "ru"]).replace([np.inf, -np.inf], np.nan).dropna()
        if len(frame) < MIN_ACCURACY_PAIRS:
            continue
        frame["pc"], frame["rc"] = frame["pv"] - alpha * frame["pu"], frame["rv"] - alpha * frame["ru"]
        if pairs_out is not None:   # the raw predicted/realised pairs, for figures
            pairs_out[variant] = frame
        for dial, (pc, rc) in {"vol": ("pv", "rv"), "volu": ("pu", "ru"), "cost_input": ("pc", "rc")}.items():
            groups = [g[[pc, rc]].to_numpy() for _, g in frame.groupby("event_id")]
            stats_of = lambda parts: _dial_stats(np.vstack(parts))  # noqa: E731
            point = _dial_stats(np.vstack(groups))
            rng = make_rng("final", "dial_accuracy", variant, dial)
            boots = np.array([stats_of([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(draws)])
            tail = (1.0 - level) / 2.0
            lo, hi = np.nanquantile(boots, tail, axis=0), np.nanquantile(boots, 1.0 - tail, axis=0)
            for k, name in enumerate(DIAL_STAT_NAMES):
                rows.append({"variant": variant, "dial": dial, "statistic": name, "value": point[k], "ci_low": lo[k], "ci_high": hi[k], "n_pairs": int(len(frame)), "n_events": int(len(groups))})
    return pd.DataFrame(rows)


DIAL_STAT_NAMES = ("bias", "rmse", "correlation", "r2_os_vs_zero", "slope_realised_on_predicted")


def _dial_stats(xy: np.ndarray) -> np.ndarray:
    p, r = xy[:, 0], xy[:, 1]
    err = p - r
    sst = float((r**2).sum())
    var = float(((p - p.mean()) ** 2).sum())
    corr = float(np.corrcoef(p, r)[0, 1]) if p.std() > 0 and r.std() > 0 else float("nan")
    return np.array([float(err.mean()), float(np.sqrt((err**2).mean())), corr, float(1.0 - (err**2).sum() / sst) if sst > 0 else float("nan"),
                     float(((p - p.mean()) * (r - r.mean())).sum() / var) if var > 0 else float("nan")])


REQUIRED = {  # metric group -> the names of columns / tables that must show up
    "Portfolio": ["cum_net_return", "annual_return_net", "annual_return_gross", "annual_vol", "annual_vol_net", "sharpe_gross", "sharpe_net_5bp", "sharpe_net_10bp", "sharpe_net_20bp",
                  "max_drawdown_net_headline", "cvar_net", "hit_rate_net", "turnover_annual", "annual_cost", "break_even_round_trip_bp", "alpha_annual", "t_alpha", "total_net_pnl_usd"],
    "Inference": ["paired_sharpe_vs_variants", "paired_sharpe_vs_rivals", "sharpe_interval_block_bootstrap", "holm_within_family", "deflated_sharpe", "pbo", "cpcv"],
    "Forecast level": ["rank_ic_1d", "rank_ic_5d", "icir", "r2_os_ret", "r2_os_vol", "r2_os_volu"],
    "Windows, cost, capacity": ["stress_windows", "cost_grid_0_40bp", "capacity_aum_levels"],
    "The sheaf in the optimiser": ["realised_vol_vs_target", "dial_accuracy_vol_volu"],
    "Variants": ["road1", "road2", "both", "merged"],
    "Headline sets": ["long_short", "headline"],
}


def checklist(summary: pd.DataFrame, tables: Mapping[str, pd.DataFrame], added: Sequence[str], missing_reason: Mapping[str, str] | None = None) -> pd.DataFrame:
    """One row per required metric: present (already in the earlier code), added (new in this run) or missing, and where it is."""
    reasons = dict(missing_reason or {})
    columns = set(summary.columns) if len(summary) else set()
    variants = set(summary["variant"]) if "variant" in summary.columns else set()
    sets = set(summary["headline_set"]) if "headline_set" in summary.columns else set()
    rows = []
    for group, names in REQUIRED.items():
        for name in names:
            if group == "Portfolio":
                ok = name in columns
                where = "summary table"
            elif group == "Variants":
                ok, where = name in variants, "summary table"
            elif group == "Headline sets":
                ok, where = name in sets, "summary table"
            else:
                ok = name in tables and len(tables[name]) > 0
                where = f"table {name}"
            state = "missing" if not ok else ("added" if name in added else "present")
            rows.append({"group": group, "metric": name, "state": state, "where": where if ok else reasons.get(name, "not produced in this run")})
    return pd.DataFrame(rows)


def daily_series_table(key: str, result, net: pd.DataFrame) -> pd.DataFrame:
    """One row per fill date of one backtest, for figures: gross and net returns at every cost, cash weight, gross exposure, one-way turnover, number of positions, delisted holdings met."""
    frame = pd.DataFrame({"gross": result.gross, **{f"net_{c}": net[c] for c in net.columns}, "cash_weight": result.cash_weight, "gross_exposure": result.positions.abs().sum(axis=1),
                          "one_way_turnover": result.trades.abs().sum(axis=1) / 2.0, "n_positions": (result.positions.abs() > 0).sum(axis=1), "delist_hits": result.delist_hits})
    frame.insert(0, "key", key)
    frame.index.name = "fill_date"
    return frame.reset_index()


def position_size_table(key: str, result) -> pd.DataFrame:
    """Distribution of the absolute target weights of one backtest (quantiles over all held name-days) and the share of days at each gross exposure, for figures."""
    sizes = result.positions.abs().to_numpy().ravel()
    sizes = sizes[sizes > 0]
    qs = list(POSITION_QUANTILES)
    return pd.DataFrame({"key": key, "quantile": qs, "abs_weight": np.quantile(sizes, qs) if len(sizes) else np.nan, "n_name_days": int(len(sizes))})
