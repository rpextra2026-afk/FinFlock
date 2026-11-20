"""Dev-only fits for the merge of the two roads (spec 9.1, "Lead, Correct, Brake").

* ``fit_merge`` is the headline fit: road 2's correction is fitted on road 1's leftover error with time-blocked
  cross-fitting, a trust that scales the correction is a ridge model shrunk toward 1 on at most four pre-event features
  (kept only if a Giacomini-White test on dev supports them), and the final rule is chosen out of sample among the plain
  sum, the 50/50 average and the context-weighted rule, the plain sum always in the race.
* ``fit_return_weights`` is the earlier non-negative least squares ``(w_A, w_B)``. It stays as the "current OLS rule" so the
  comparison of merges can show it beside the new rule; it is never the headline.
Everything reads dev rows only; no label of a test event enters.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import nnls

from oryxis.common.config import Config
from oryxis.common.cv import time_blocked_folds
from oryxis.common.timing import TradingCalendar
from oryxis.eval.stats import giacomini_white
from oryxis.framework.run import EventForecast
from oryxis.portfolio.combine import choose_by_loss, half_average, trust_value

RULE_PLAIN, RULE_HALF, RULE_LCB = "plain_sum", "half_average", "lcb"
MIN_ROWS_FLOOR = 8  # fewer rows than this cannot be cross-fitted at all


def realised_days_1_to_5(returns: pd.DataFrame, market: pd.Series, day0: int, assets: Sequence[int], days: int) -> np.ndarray:
    """Market-adjusted cumulative return over days ``1 .. days`` after day 0 for the given asset columns."""
    window = returns.iloc[day0 + 1 : day0 + 1 + days].to_numpy()
    return (window - market.iloc[day0 + 1 : day0 + 1 + days].to_numpy()[:, None]).sum(axis=0)[list(assets)]


def realised_between(returns: pd.DataFrame, market: pd.Series, day0: int, assets: Sequence[int], first: int, last: int) -> np.ndarray:
    """Market-adjusted cumulative return over days ``first .. last`` after day 0."""
    window = returns.iloc[day0 + first : day0 + last + 1].to_numpy()
    return (window - market.iloc[day0 + first : day0 + last + 1].to_numpy()[:, None]).sum(axis=0)[list(assets)]


def fit_return_weights(
    forecasts: Sequence[EventForecast], returns: pd.DataFrame, market: pd.Series, days: int, variant: str
) -> tuple[float, float, dict]:
    """Legacy rule. ``road1`` fits ``w_A`` alone, ``road2`` fits ``w_B`` alone, ``both`` fits both (non-negative least squares)."""
    rows, target = [], []
    for f in forecasts:
        realised = realised_days_1_to_5(returns, market, f.day0, f.assets, days)
        a, b = f.inc_a[:days].sum(axis=0), f.inc_b[:days].sum(axis=0)
        rows.append(np.column_stack([a, b]))
        target.append(realised)
    X, y = np.vstack(rows), np.concatenate(target)
    columns = {"road1": [0], "road2": [1], "both": [0, 1]}[variant]
    coef, residual = nnls(X[:, columns], y)
    weights = [0.0, 0.0]
    for position, column in enumerate(columns):
        weights[column] = float(coef[position])
    return weights[0], weights[1], {"n_rows": len(y), "residual_norm": float(residual)}


# --- Lead, Correct, Brake ------------------------------------------------------------------------------------------


@dataclass
class MergeRows:
    """Dev rows of one horizon bucket: one row per (event, asset)."""

    road1: np.ndarray  # road-1 forecast summed over the bucket's days
    road2_raw: np.ndarray  # road 2's correction before it is fitted on road 1's leftover (from the "both" variant)
    road2_only: np.ndarray  # road 2's own forecast (the "road2" variant)
    realised: np.ndarray  # market-adjusted return over the same days
    features: np.ndarray  # (n, k) trust features, pre-event only
    dates: pd.Series  # day 0 of each row


@dataclass
class TrustModel:
    """Trust for one horizon bucket: ``clip(1 + theta_0 + z @ theta, lo, hi)`` on standardised features."""

    slope: float
    theta: np.ndarray
    mean: np.ndarray
    scale: np.ndarray
    ridge: float
    bounds: tuple[float, float]
    columns: list[int] = field(default_factory=list)  # the kept features, as positions in the full feature list

    def trust(self, all_features: np.ndarray) -> np.ndarray:
        """Trust per row from the FULL feature matrix (this bucket picks its own kept columns)."""
        raw = np.atleast_2d(all_features)[:, self.columns] if self.columns else np.zeros((len(np.atleast_2d(all_features)), 0))
        z = (raw - self.mean) / self.scale if self.columns else raw
        return trust_value(z, self.theta, self.bounds)

    def correction(self, road2_raw: np.ndarray) -> np.ndarray:
        return self.slope * road2_raw


@dataclass
class MergeFit:
    rule: str
    models: list[TrustModel]
    feature_names: list[str]
    kept_features: list[str]
    within_variance: tuple[float, float]
    gap_quantile: float
    cv_loss: dict[str, float]
    notes: dict = field(default_factory=dict)


def _folds(rows: MergeRows, calendar: TradingCalendar, cfg: Config) -> list[tuple[np.ndarray, np.ndarray]]:
    spec = cfg.get("framework.merge")
    return time_blocked_folds(rows.dates, int(spec["cv_folds"]), int(spec["embargo_sessions"]), calendar)


def _slope(road1: np.ndarray, road2: np.ndarray, realised: np.ndarray, ridge: float) -> float:
    """Least-squares slope of road 1's leftover error on road 2's raw correction (through the origin, ridge-shrunk)."""
    return float(road2 @ (realised - road1) / (road2 @ road2 + ridge))


def cross_fitted_correction(rows: MergeRows, calendar: TradingCalendar, cfg: Config) -> tuple[np.ndarray, float]:
    """Out-of-fold fitted correction (no row's own label touches its slope) and the slope fitted on all dev rows."""
    ridge = float(cfg.get("framework.merge.slope_ridge"))
    out = np.zeros(len(rows.road1))
    for train, val in _folds(rows, calendar, cfg):
        slope = _slope(rows.road1[train], rows.road2_raw[train], rows.realised[train], ridge) if len(train) else 0.0
        out[val] = slope * rows.road2_raw[val]
    return out, _slope(rows.road1, rows.road2_raw, rows.realised, ridge)


def _standardise(z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = z.mean(axis=0) if z.shape[1] else np.zeros(0)
    scale = np.where(z.std(axis=0) > 0, z.std(axis=0), 1.0) if z.shape[1] else np.ones(0)
    return (z - mean) / scale, mean, scale


def _fit_theta(correction: np.ndarray, residual: np.ndarray, z: np.ndarray, ridge: float) -> np.ndarray:
    """Ridge on ``residual - correction`` with regressors ``correction * [1, z]``: shrinks theta to 0, i.e. trust to 1."""
    design = correction[:, None] * np.column_stack([np.ones(len(correction)), z])
    gram = design.T @ design + ridge * np.eye(design.shape[1])
    return np.linalg.solve(gram, design.T @ (residual - correction))


def _oof_trust_loss(rows: MergeRows, correction: np.ndarray, z: np.ndarray, ridge: float, bounds, folds) -> tuple[float, np.ndarray]:
    """Cross-validated squared error of ``road1 + trust * correction`` and the per-row out-of-fold squared errors."""
    errors = np.full(len(correction), np.nan)
    for train, val in folds:
        if len(train) <= z.shape[1] + 1:
            continue
        theta = _fit_theta(correction[train], rows.realised[train] - rows.road1[train], z[train], ridge)
        trust = trust_value(z[val], theta, bounds)
        errors[val] = (rows.realised[val] - rows.road1[val] - trust * correction[val]) ** 2
    return (float(np.nanmean(errors)) if np.isfinite(errors).any() else float("inf")), errors


def fit_bucket(rows: MergeRows, feature_names: Sequence[str], calendar: TradingCalendar, cfg: Config) -> tuple[TrustModel, list[str], dict[str, float]]:
    """One horizon bucket: correction slope, kept features (Giacomini-White), ridge by CV, and the candidate losses."""
    spec = cfg.get("framework.merge")
    bounds = (float(spec["trust_range"][0]), float(spec["trust_range"][1]))
    if len(rows.road1) < max(int(spec["min_rows"]), MIN_ROWS_FLOOR):
        return TrustModel(1.0, np.zeros(1), np.zeros(0), np.ones(0), 0.0, bounds), [], {RULE_PLAIN: float("nan")}  # too few rows: the plain sum
    folds = _folds(rows, calendar, cfg)
    correction, slope = cross_fitted_correction(rows, calendar, cfg)
    plain_error = (rows.realised - rows.road1 - rows.road2_raw) ** 2
    half_error = (rows.realised - half_average(rows.road1, rows.road2_only)) ** 2
    z_all, mean, scale = _standardise(rows.features)
    plain_loss = float(plain_error.mean())
    # a feature stays only if it improves the out-of-fold loss over a constant trust, tested conditionally (GW) on dev
    constant_loss, constant_errors = min((_oof_trust_loss(rows, correction, z_all[:, :0], float(r), bounds, folds)
                                          for r in spec["ridge_grid"]), key=lambda pair: pair[0])
    kept = []
    for j, name in enumerate(feature_names):
        loss, errors = min((_oof_trust_loss(rows, correction, z_all[:, [j]], float(r), bounds, folds) for r in spec["ridge_grid"]),
                           key=lambda pair: pair[0])
        valid = np.isfinite(errors) & np.isfinite(constant_errors)
        if loss < constant_loss and valid.sum() > MIN_ROWS_FLOOR:
            instrument = np.column_stack([np.ones(valid.sum()), z_all[valid, j]])
            _, p_value = giacomini_white(constant_errors[valid] - errors[valid], instrument, int(spec["gw_lag"]))
            if p_value < float(spec["gw_alpha"]):
                kept.append(name)
    columns = [feature_names.index(name) for name in kept]
    z = z_all[:, columns]
    best = min(((*_oof_trust_loss(rows, correction, z, float(r), bounds, folds), float(r)) for r in spec["ridge_grid"]), key=lambda t: t[0])
    lcb_loss, _, ridge = best
    theta = _fit_theta(slope * rows.road2_raw, rows.realised - rows.road1, z, ridge)
    model = TrustModel(slope, theta, mean[columns], scale[columns], ridge, bounds, columns)
    return model, kept, {RULE_PLAIN: plain_loss, RULE_HALF: float(half_error.mean()), RULE_LCB: lcb_loss}


def fit_merge(buckets: Sequence[MergeRows], feature_names: Sequence[str], calendar: TradingCalendar, cfg: Config) -> MergeFit:
    """Fit the trust of every horizon bucket and choose the final rule by out-of-sample dev loss (plain sum always in the race)."""
    models, kept_all, totals = [], set(), {RULE_PLAIN: 0.0, RULE_HALF: 0.0, RULE_LCB: 0.0}
    weights = [len(r.road1) for r in buckets]
    for rows, weight in zip(buckets, weights, strict=True):
        model, kept, losses = fit_bucket(rows, feature_names, calendar, cfg)
        models.append(model)
        kept_all.update(kept)
        for name in totals:
            totals[name] += weight * losses.get(name, float("nan"))
    total_rows = max(sum(weights), 1)
    cv_loss = {name: value / total_rows for name, value in totals.items()}
    candidates = {name: cv_loss[name] for name in cfg.get("framework.merge.candidates")}
    rule = choose_by_loss(candidates, must_include=RULE_PLAIN) if np.isfinite(cv_loss[RULE_PLAIN]) else RULE_PLAIN
    forced = cfg.get("framework.merge.rule")
    rule = str(forced) if forced else rule  # a rule written in config (pre-registered) overrides the dev choice
    road1 = np.concatenate([r.road1 for r in buckets])
    road2_only = np.concatenate([r.road2_only for r in buckets])
    realised = np.concatenate([r.realised for r in buckets])
    within = (float(np.mean((realised - road1) ** 2)), float(np.mean((realised - road2_only) ** 2)))
    relative_gap = np.abs(road1 - road2_only) / (np.maximum(np.abs(road1), np.abs(road2_only)) + float(cfg.get("framework.brake.disagreement_floor")))
    quantile = float(np.quantile(relative_gap, float(cfg.get("framework.brake.abstain_quantile")))) if len(relative_gap) else float("inf")
    return MergeFit(rule, models, list(feature_names), [n for n in feature_names if n in kept_all], within, quantile, cv_loss,
                    {"rows_per_bucket": weights})


def merge_rows_from_forecasts(
    road1: Sequence[EventForecast], both: Sequence[EventForecast], road2: Sequence[EventForecast],
    returns: pd.DataFrame, market: pd.Series, cfg: Config, feature_names: Sequence[str],
) -> list[MergeRows]:
    """Dev rows per horizon bucket from matched forecasts of the three variants of the same events (one row per event and asset)."""
    from oryxis.framework.run import trust_inputs

    out = []
    for first, last in cfg.get("framework.merge.horizons"):
        cols = {k: [] for k in ("a", "b", "r2", "y", "z", "date")}
        for f1, fb, f2 in zip(road1, both, road2, strict=True):
            rows = slice(first - 1, last)
            cols["a"].append(f1.inc_a[rows].sum(axis=0))
            cols["b"].append(fb.inc_b[rows].sum(axis=0))
            cols["r2"].append(f2.inc_b[rows].sum(axis=0))
            cols["y"].append(realised_between(returns, market, f1.day0, f1.assets, first, last))
            cols["z"].append(trust_inputs(f1, fb, feature_names))
            cols["date"].extend([returns.index[f1.day0]] * len(f1.assets))
        out.append(MergeRows(np.concatenate(cols["a"]), np.concatenate(cols["b"]), np.concatenate(cols["r2"]), np.concatenate(cols["y"]),
                             np.vstack(cols["z"]), pd.Series(pd.DatetimeIndex(cols["date"]))))
    return out
