"""Metrics of the final build (spec section 6, families A to E): pure functions on arrays, and a collector that writes the long format.

Row format of the long table: tape, block, dial, horizon, method, view, slice, slice_value, metric, value, lo, hi, n
(the spec's columns plus ``method`` and ``view``: a method comparison and the linked / all / unlinked views need them).
Every number that decides anything (levels, bins, lags, draws) comes from ``build.metrics`` in the config.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from finfm.common.config import Config

ROOT = "build.metrics"
HALF = 0.5
TWO_SIDED = 2.0
COLUMNS = ["tape", "block", "dial", "horizon", "method", "view", "slice", "slice_value", "metric", "value", "lo", "hi", "n"]
NAN = float("nan")
MIN_POINTS = 3                  # fewer points than this give no statistic
PERCENT = 100.0
EXP_CLIP = 20.0                 # log-ratio errors are clipped here before the exponential in QLIKE
BASIS_POINT = 0.0001
LOGIT_CLIP = 30.0
PROB_FLOOR = 1e-9
NEWTON_TOL = 1e-8
LOG_FLOOR = 1e-12
CAL_RIDGE = 1e-6
CAL_STEPS = 25


def mcfg(cfg: Config, key: str) -> Any:
    return cfg.get(f"{ROOT}.{key}")


class Collector:
    """Rows of the long metrics table for one tape and one block."""

    def __init__(self, tape: str, block: str, horizon: str):
        self.tape, self.block, self.horizon = tape, block, horizon
        self.rows: list[dict[str, Any]] = []

    def add(self, dial: str, slice_name: str, slice_value: Any, metric: str, value: float, lo: float = NAN, hi: float = NAN, n: float = NAN, method: str = "", view: str = "") -> None:
        self.rows.append({"tape": self.tape, "block": self.block, "dial": dial, "horizon": self.horizon, "method": method, "view": view, "slice": slice_name, "slice_value": str(slice_value),
                          "metric": metric, "value": float(value) if value is not None else NAN, "lo": lo, "hi": hi, "n": n})

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=COLUMNS)


# ---- A. error and fit -----------------------------------------------------------------------------------------------------------------------

def error_metrics(y: np.ndarray, pred: np.ndarray, base: np.ndarray, scale: float, cfg: Config) -> dict[str, float]:
    """Errors of the forecast and of the no-ripple baseline, and the gain of one over the other (Campbell-Thompson form)."""
    err, err_b = y - pred, y - base
    sse, sse_b = float((err**2).sum()), float((err_b**2).sum())
    delta = float(mcfg(cfg, "huber_delta")) * scale
    floor = float(mcfg(cfg, "mape_floor")) * scale
    abs_err = np.abs(err)
    huber = np.where(abs_err <= delta, HALF * err**2, delta * (abs_err - HALF * delta))
    big = np.abs(y) >= floor
    smape_den = np.abs(y) + np.abs(pred)
    out = {"r2_os": 1.0 - sse / sse_b if sse_b > 0 else NAN, "gain_pct": PERCENT * (1.0 - sse / sse_b) if sse_b > 0 else NAN, "mse": sse / len(y), "mse_baseline": sse_b / len(y),
           "rmse": float(np.sqrt(sse / len(y))), "mae": float(abs_err.mean()), "median_abs_error": float(np.median(abs_err)), "huber": float(huber.mean()),
           "mape": float((abs_err[big] / np.abs(y[big])).mean()) if big.any() else NAN,
           "smape": float((2 * abs_err[smape_den > 0] / smape_den[smape_den > 0]).mean()) if (smape_den > 0).any() else NAN}
    clipped = np.clip(err, -EXP_CLIP, EXP_CLIP)
    out["qlike"] = float(np.mean(np.exp(2 * clipped) - 2 * clipped - 1))  # variance-ratio QLIKE: y and pred are log ratios of volatility (or volume)
    out["log_ratio_error"] = float(np.mean(err))
    slope = mz_regression(y, pred)
    out.update({"mz_intercept": slope[0], "mz_slope": slope[1], "mz_r2": slope[2]})
    return out


def mz_regression(y: np.ndarray, pred: np.ndarray) -> tuple[float, float, float]:
    """Mincer-Zarnowitz: y on pred; intercept, slope, R2."""
    if len(y) < MIN_POINTS or np.ptp(pred) == 0:
        return NAN, NAN, NAN
    slope, intercept, r, _, _ = stats.linregress(pred, y)
    return float(intercept), float(slope), float(r**2)


def day_sums(day_codes: np.ndarray, n_days: int, values: np.ndarray) -> np.ndarray:
    return np.bincount(day_codes, weights=values, minlength=n_days)


def gain_interval(sse_day: np.ndarray, sse_base_day: np.ndarray, cfg: Config, rng: np.random.Generator, block: bool, draws: int | None = None) -> tuple[float, float]:
    """Bootstrap interval of the relative gain over days (single days, or blocks of consecutive days)."""
    n = len(sse_day)
    draws = int(draws or mcfg(cfg, "bootstrap_draws"))
    if n < 2:
        return NAN, NAN
    if block:
        length = max(min(int(mcfg(cfg, "block_days")), n), 1)
        starts = rng.integers(0, n, size=(draws, int(np.ceil(n / length))))
        idx = ((starts[:, :, None] + np.arange(length)[None, None, :]) % n).reshape(draws, -1)[:, :n]
    else:
        idx = rng.integers(0, n, size=(draws, n))
    boot = 1.0 - sse_day[idx].sum(1) / sse_base_day[idx].sum(1)
    tail = (1.0 - float(mcfg(cfg, "ci_level"))) / TWO_SIDED
    low, high = np.quantile(boot, [tail, 1.0 - tail])
    return float(low), float(high)


# ---- B. tests -------------------------------------------------------------------------------------------------------------------------------

def newey_west_mean(x: np.ndarray, lag: int) -> tuple[float, float, float]:
    """Mean, Newey-West standard error and t-statistic of a series."""
    n = len(x)
    if n < 2:
        return NAN, NAN, NAN
    centred = x - x.mean()
    var = float(centred @ centred) / n
    for k in range(1, min(lag, n - 1) + 1):
        weight = 1.0 - k / (lag + 1.0)
        var += TWO_SIDED * weight * float(centred[k:] @ centred[:-k]) / n
    se = float(np.sqrt(max(var, 0.0) / n))
    mean = float(x.mean())
    return mean, se, mean / se if se > 0 else NAN


def p_two_sided(t: float, df: float) -> float:
    return float(TWO_SIDED * stats.t.sf(abs(t), df)) if np.isfinite(t) else NAN


def diebold_mariano(loss_model_day: np.ndarray, loss_base_day: np.ndarray, lag: int) -> dict[str, float]:
    """Daily loss differential (baseline minus model); positive mean means the model is better."""
    d = loss_base_day - loss_model_day
    mean, se, t = newey_west_mean(d, lag)
    return {"dm_mean_diff": mean, "dm_t": t, "dm_p": p_two_sided(t, len(d) - 1), "dm_effect_d": mean / float(d.std(ddof=1)) if len(d) > 1 and d.std(ddof=1) > 0 else NAN}


def clark_west(y_day_terms: np.ndarray, lag: int) -> dict[str, float]:
    """Clark-West adjusted differential per day: (y-base)^2 - ((y-pred)^2 - (base-pred)^2), averaged over rows of the day."""
    mean, se, t = newey_west_mean(y_day_terms, lag)
    return {"cw_mean": mean, "cw_t": t, "cw_p": p_two_sided(t, len(y_day_terms) - 1)}


def slope_tests(y: np.ndarray, pred: np.ndarray, day_codes: np.ndarray, n_days: int, lag: int, base: np.ndarray) -> dict[str, float]:
    """Slope of the outcome (minus baseline) on the signal (pred minus baseline), with day-clustered and Newey-West t-statistics."""
    x, z = pred - base, y - base
    sxx = float(x @ x)
    if sxx <= 0 or len(x) < MIN_POINTS:
        return {k: NAN for k in ("slope", "slope_t_clustered", "slope_t_nw", "slope_std")}
    slope = float(x @ z) / sxx
    resid = z - slope * x
    score_day = day_sums(day_codes, n_days, x * resid)
    var_cluster = float(score_day @ score_day) / sxx**2
    se_cluster = float(np.sqrt(var_cluster))
    sx_day = day_sums(day_codes, n_days, x * z)
    nw_mean, nw_se, nw_t = newey_west_mean(sx_day, lag)
    return {"slope": slope, "slope_t_clustered": slope / se_cluster if se_cluster > 0 else NAN, "slope_t_nw": nw_t, "slope_std": slope * float(x.std()) / float(z.std()) if z.std() > 0 else NAN}


def multiple_comparison(pvalues: Sequence[float]) -> tuple[np.ndarray, np.ndarray]:
    """Holm and Benjamini-Hochberg adjusted p-values (NaN stays NaN)."""
    p = np.asarray(pvalues, dtype=float)
    holm = np.full(len(p), NAN)
    bh = np.full(len(p), NAN)
    ok = np.isfinite(p)
    m = int(ok.sum())
    if m == 0:
        return holm, bh
    order = np.argsort(p[ok])
    ranked = p[ok][order]
    holm_adj = np.maximum.accumulate((m - np.arange(m)) * ranked)
    bh_adj = np.minimum.accumulate((ranked * m / (np.arange(m) + 1))[::-1])[::-1]
    index = np.flatnonzero(ok)[order]
    holm[index] = np.minimum(holm_adj, 1.0)
    bh[index] = np.minimum(bh_adj, 1.0)
    return holm, bh


# ---- C. rank and signal quality -------------------------------------------------------------------------------------------------------------

def daily_ic(y: np.ndarray, pred: np.ndarray, day_codes: np.ndarray, n_days: int, min_stocks: int) -> tuple[np.ndarray, np.ndarray]:
    """Pearson IC and rank IC for every day with at least ``min_stocks`` rows (NaN elsewhere)."""
    order = np.argsort(day_codes, kind="stable")
    codes = day_codes[order]
    bounds = np.flatnonzero(np.diff(codes)) + 1
    ic = np.full(n_days, NAN)
    ric = np.full(n_days, NAN)
    for part in np.split(order, bounds):
        if len(part) < min_stocks:
            continue
        a, b = y[part], pred[part]
        d = day_codes[part[0]]
        if np.ptp(a) > 0 and np.ptp(b) > 0:
            ic[d] = float(np.corrcoef(a, b)[0, 1])
            ric[d] = float(stats.spearmanr(a, b)[0])
    return ic, ric


def ic_summary(ic: np.ndarray) -> dict[str, float]:
    good = ic[np.isfinite(ic)]
    if len(good) < 2:
        return {"ic_mean": NAN, "ic_std": NAN, "icir": NAN, "ic_pos_share": NAN, "ic_days": float(len(good))}
    return {"ic_mean": float(good.mean()), "ic_std": float(good.std(ddof=1)), "icir": float(good.mean() / good.std(ddof=1)) if good.std(ddof=1) > 0 else NAN, "ic_pos_share": float((good > 0).mean()), "ic_days": float(len(good))}


def correlations(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    if len(y) < MIN_POINTS or np.ptp(pred) == 0 or np.ptp(y) == 0:
        return {"pearson": NAN, "spearman": NAN, "kendall": NAN}
    return {"pearson": float(np.corrcoef(y, pred)[0, 1]), "spearman": float(stats.spearmanr(y, pred)[0]), "kendall": float(stats.kendalltau(y, pred)[0])}


def bin_spread(y: np.ndarray, pred: np.ndarray, bins: int) -> dict[str, float]:
    """Mean outcome by signal bin, top minus bottom, and monotonicity (Spearman of bin number and bin mean)."""
    if len(y) < bins * 2 or np.ptp(pred) == 0:
        return {f"spread_{bins}": NAN, f"monotonicity_{bins}": NAN}
    ranks = stats.rankdata(pred, method="average") / len(pred)
    which = np.minimum((ranks * bins).astype(int), bins - 1)
    means = np.array([y[which == b].mean() if (which == b).any() else NAN for b in range(bins)])
    good = np.isfinite(means)
    mono = float(stats.spearmanr(np.arange(bins)[good], means[good])[0]) if good.sum() >= MIN_POINTS else NAN
    return {f"spread_{bins}": float(means[good][-1] - means[good][0]) if good.sum() > 1 else NAN, f"monotonicity_{bins}": mono}


def long_short(y: np.ndarray, pred: np.ndarray, node_codes: np.ndarray, day_codes: np.ndarray, n_days: int, share: float, costs_bp: Sequence[float], ann: int) -> dict[str, float]:
    """Daily long-short on the signal (long the top share, short the bottom share, equal weights, in label units): hit rate, mean, Sharpe, turnover, drawdown."""
    daily, turn = [], []
    prev_long: set[int] = set()
    prev_short: set[int] = set()
    order = np.argsort(day_codes, kind="stable")
    bounds = np.flatnonzero(np.diff(day_codes[order])) + 1
    for part in np.split(order, bounds):
        if len(part) < max(int(1 / share) * 2, 2):
            continue
        k = max(int(len(part) * share), 1)
        ranked = part[np.argsort(pred[part], kind="stable")]
        short, long = ranked[:k], ranked[-k:]
        daily.append(float(y[long].mean() - y[short].mean()))
        now_l, now_s = set(node_codes[long].tolist()), set(node_codes[short].tolist())
        turn.append(0.0 if not prev_long else 1.0 - len(now_l & prev_long) / max(len(now_l), 1) + 1.0 - len(now_s & prev_short) / max(len(now_s), 1))
        prev_long, prev_short = now_l, now_s
    if len(daily) < 2:
        return {}
    r = np.array(daily)
    out = {"ls_days": float(len(r)), "ls_hit_rate": float((r > 0).mean()), "ls_mean": float(r.mean()), "ls_turnover": float(np.mean(turn))}
    for bp in costs_bp:
        net = r - float(np.mean(turn)) * bp * BASIS_POINT
        out[f"ls_sharpe_{int(bp)}bp"] = float(net.mean() / net.std(ddof=1) * np.sqrt(ann)) if net.std(ddof=1) > 0 else NAN
    curve = np.cumsum(r)
    out["ls_max_drawdown"] = float((curve - np.maximum.accumulate(curve)).min())
    return out


# ---- D. direction and classification --------------------------------------------------------------------------------------------------------

def logistic_fit(score: np.ndarray, label: np.ndarray, steps: int, ridge: float) -> tuple[float, float]:
    """Intercept and slope of a one-variable logistic regression (Newton steps)."""
    a, b = 0.0, 0.0
    x = np.column_stack([np.ones(len(score)), score])
    beta = np.array([a, b])
    for _ in range(steps):
        p = 1.0 / (1.0 + np.exp(-np.clip(x @ beta, -LOGIT_CLIP, LOGIT_CLIP)))
        w = np.maximum(p * (1 - p), PROB_FLOOR)
        grad = x.T @ (label - p) - ridge * beta
        hess = (x * w[:, None]).T @ x + ridge * np.eye(2)
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.abs(step).max() < NEWTON_TOL:
            break
    return float(beta[0]), float(beta[1])


def classification_metrics(label: np.ndarray, score: np.ndarray, prob: np.ndarray, bins: int) -> dict[str, float]:
    """Direction metrics: label = outcome above zero, score = forecast, prob = calibrated probability of an up outcome."""
    pos = label.astype(bool)
    pred_pos = score > 0
    tp, fp, fn, tn = (int((pred_pos & pos).sum()), int((pred_pos & ~pos).sum()), int((~pred_pos & pos).sum()), int((~pred_pos & ~pos).sum()))
    n = tp + fp + fn + tn
    prec = tp / (tp + fp) if tp + fp else NAN
    rec = tp / (tp + fn) if tp + fn else NAN
    spec = tn / (tn + fp) if tn + fp else NAN
    den = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    out = {"accuracy": (tp + tn) / n if n else NAN, "balanced_accuracy": np.nanmean([rec, spec]) if np.isfinite([rec, spec]).any() else NAN, "precision": prec, "recall": rec,
           "f1": 2 * prec * rec / (prec + rec) if np.isfinite(prec) and np.isfinite(rec) and prec + rec > 0 else NAN, "mcc": (tp * tn - fp * fn) / den if den > 0 else NAN}
    if pos.any() and (~pos).any():
        ranks = stats.rankdata(score)
        out["auc"] = float((ranks[pos].sum() - pos.sum() * (pos.sum() + 1) / TWO_SIDED) / (pos.sum() * (~pos).sum()))
        order = np.argsort(-score, kind="stable")
        hits = pos[order]
        precision_at = np.cumsum(hits) / (np.arange(len(hits)) + 1)
        out["average_precision"] = float(precision_at[hits].mean())
    out["brier"] = float(np.mean((prob - label) ** 2))
    out["log_loss"] = float(-np.mean(label * np.log(np.clip(prob, LOG_FLOOR, 1)) + (1 - label) * np.log(np.clip(1 - prob, LOG_FLOOR, 1))))
    cal_a, cal_b = logistic_fit(np.log(np.clip(prob, PROB_FLOOR, 1 - PROB_FLOOR) / np.clip(1 - prob, PROB_FLOOR, 1)), label.astype(float), CAL_STEPS, CAL_RIDGE) if pos.any() and (~pos).any() else (NAN, NAN)
    out["calibration_intercept"], out["calibration_slope"] = cal_a, cal_b
    edges = np.minimum((prob * bins).astype(int), bins - 1)
    ece = 0.0
    for b in range(bins):
        m = edges == b
        if m.any():
            ece += m.mean() * abs(prob[m].mean() - label[m].mean())
    out["ece"] = float(ece)
    return out


def reliability_bins(label: np.ndarray, prob: np.ndarray, bins: int) -> list[tuple[int, float, float, int]]:
    edges = np.minimum((prob * bins).astype(int), bins - 1)
    return [(b, float(prob[edges == b].mean()), float(label[edges == b].mean()), int((edges == b).sum())) for b in range(bins) if (edges == b).any()]


# ---- E. uncertainty -------------------------------------------------------------------------------------------------------------------------

def interval_metrics(y: np.ndarray, pred: np.ndarray, sd: float, levels: Sequence[float], quantiles: Sequence[float], pit_bins: int) -> dict[str, float]:
    """Gaussian predictive distribution with the training residual sd: coverage, width, pinball, CRPS, PIT histogram."""
    out: dict[str, float] = {}
    z = (y - pred) / sd
    for level in levels:
        half = float(stats.norm.ppf(HALF + level / TWO_SIDED))
        out[f"coverage_{int(round(level * PERCENT))}"] = float((np.abs(z) <= half).mean())
        out[f"width_{int(round(level * PERCENT))}"] = float(TWO_SIDED * half * sd)
    for q in quantiles:
        qhat = pred + sd * float(stats.norm.ppf(q))
        diff = y - qhat
        out[f"pinball_{int(round(q * PERCENT)):02d}"] = float(np.mean(np.maximum(q * diff, (q - 1) * diff)))
    pdf, cdf = stats.norm.pdf(z), stats.norm.cdf(z)
    out["crps"] = float(np.mean(sd * (z * (TWO_SIDED * cdf - 1) + TWO_SIDED * pdf - 1 / np.sqrt(np.pi))))
    hist = np.bincount(np.minimum((cdf * pit_bins).astype(int), pit_bins - 1), minlength=pit_bins) / len(y)
    for b, share in enumerate(hist):
        out[f"pit_bin_{b:02d}"] = float(share)
    return out


# ---- one frame, every metric ----------------------------------------------------------------------------------------------------------------

def all_metrics(frame: pd.DataFrame, dial: str, scale: float, resid_sd: float, prob: np.ndarray, cfg: Config, rng: np.random.Generator, draws: int | None = None, full: bool = True) -> dict[str, tuple[float, float, float]]:
    """Every metric of families A to E for one frame (columns y, pred, base, day, node). Value, low, high."""
    y, pred, base = frame["y"].to_numpy(float), frame["pred"].to_numpy(float), frame["base"].to_numpy(float)
    days, day_codes = np.unique(frame["day"].to_numpy(), return_inverse=True)
    nd = len(days)
    node_codes = pd.factorize(frame["node"])[0]
    out: dict[str, tuple[float, float, float]] = {}
    for k, v in error_metrics(y, pred, base, scale, cfg).items():
        out[k] = (v, NAN, NAN)
    sse_d, sseb_d = day_sums(day_codes, nd, (y - pred) ** 2), day_sums(day_codes, nd, (y - base) ** 2)
    if nd >= 2:
        for name, block in (("gain_ci_day", False), ("gain_ci_block", True)):
            lo, hi = gain_interval(sse_d, sseb_d, cfg, rng, block, draws)
            out[name] = (1.0 - sse_d.sum() / sseb_d.sum(), lo, hi)
        lag = int(mcfg(cfg, "nw_lag"))
        for k, v in diebold_mariano(sse_d, sseb_d, lag).items():
            out[k] = (v, NAN, NAN)
        cw = day_sums(day_codes, nd, (y - base) ** 2 - ((y - pred) ** 2 - (base - pred) ** 2)) / np.maximum(np.bincount(day_codes, minlength=nd), 1)
        for k, v in clark_west(cw, lag).items():
            out[k] = (v, NAN, NAN)
        for k, v in slope_tests(y, pred, day_codes, nd, lag, base).items():
            out[k] = (v, NAN, NAN)
        t_cl = out["slope_t_clustered"][0]
        out["slope_p"] = (p_two_sided(t_cl, nd - 1), NAN, NAN)
    if not full:
        return out
    for k, v in correlations(y, pred).items():
        out[k] = (v, NAN, NAN)
    ic, ric = daily_ic(y, pred, day_codes, nd, int(mcfg(cfg, "min_stocks_per_day")))
    for k, v in ic_summary(ic).items():
        out[k] = (v, NAN, NAN)
    for k, v in ic_summary(ric).items():
        out["rank_" + k] = (v, NAN, NAN)
    for bins in (int(mcfg(cfg, "quantile_bins")), int(mcfg(cfg, "coarse_bins"))):
        for k, v in bin_spread(y, pred, bins).items():
            out[k] = (v, NAN, NAN)
    for k, v in long_short(y, pred, node_codes, day_codes, nd, float(mcfg(cfg, "long_short_share")), list(mcfg(cfg, "costs_bp")), int(mcfg(cfg, "annualisation_days"))).items():
        out[k] = (v, NAN, NAN)
    up = (y > base).astype(float)  # direction is measured against the baseline: an outcome above it, a forecast above it
    if 0 < up.sum() < len(up):
        for k, v in classification_metrics(up, pred - base, prob, int(mcfg(cfg, "calibration_bins"))).items():
            out["dir_" + k] = (v, NAN, NAN)
    for k, v in interval_metrics(y, pred, resid_sd, list(mcfg(cfg, "interval_levels")), list(mcfg(cfg, "pinball_quantiles")), int(mcfg(cfg, "pit_bins"))).items():
        out[k] = (v, NAN, NAN)
    out["per_day_mse_mean"] = (float((sse_d / np.maximum(np.bincount(day_codes, minlength=nd), 1)).mean()), NAN, NAN)
    out["per_day_mse_median"] = (float(np.median(sse_d / np.maximum(np.bincount(day_codes, minlength=nd), 1))), NAN, NAN)
    out["per_day_mse_p95"] = (float(np.quantile(sse_d / np.maximum(np.bincount(day_codes, minlength=nd), 1), float(mcfg(cfg, "ci_level")))), NAN, NAN)
    per_node = pd.Series((y - pred) ** 2).groupby(node_codes).mean()
    out["per_stock_mse_mean"] = (float(per_node.mean()), NAN, NAN)
    out["per_stock_mse_median"] = (float(per_node.median()), NAN, NAN)
    out["per_stock_mse_p95"] = (float(per_node.quantile(float(mcfg(cfg, "ci_level")))), NAN, NAN)
    return out


def write_metrics(collector: Collector, dial: str, slice_name: str, slice_value: Any, metrics: Mapping[str, tuple[float, float, float]], n: int, method: str = "", view: str = "") -> None:
    for name, (value, lo, hi) in metrics.items():
        collector.add(dial, slice_name, slice_value, name, value, lo, hi, n, method, view)
