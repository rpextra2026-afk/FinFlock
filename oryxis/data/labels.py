"""Labels: what the news did afterwards (spec 3.8). IMPORT-GUARDED.

No module under ``reader``, ``road1``, ``personas``, ``mfg`` or ``portfolio`` may import this one;
only ``fit_*`` functions called from scripts and the ``eval`` modules read labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from oryxis.common.config import Config

HALF = 0.5
SQRT_N = "sqrt_n"


def fit_residual_model(
    excess: np.ndarray, factors: np.ndarray, usable: np.ndarray, min_obs: int
) -> tuple[np.ndarray, float] | None:
    """OLS of excess returns on the factors over the usable estimation rows; ``None`` if too few rows."""
    X = np.column_stack([np.ones(len(excess)), factors])
    rows = usable & np.isfinite(excess) & np.isfinite(X).all(axis=1)
    if rows.sum() < max(min_obs, X.shape[1] + 1):
        return None
    beta, *_ = np.linalg.lstsq(X[rows], excess[rows], rcond=None)
    residual = excess[rows] - X[rows] @ beta
    sigma = float(residual.std(ddof=X.shape[1]))
    return beta, sigma


def garman_klass(open_: np.ndarray, high: np.ndarray, low: np.ndarray, close: np.ndarray, ln2_term: float) -> np.ndarray:
    """Daily Garman-Klass variance proxy ``0.5 ln(H/L)^2 - (2 ln 2 - 1) ln(C/O)^2``."""
    return HALF * np.log(high / low) ** 2 - ln2_term * np.log(close / open_) ** 2


def event_window_mask(n: int, event_days: np.ndarray, before: int, after: int) -> np.ndarray:
    """True on sessions inside any event window ``[day0 + before, day0 + after]`` of the asset."""
    mask = np.zeros(n, dtype=bool)
    for day in event_days:
        mask[max(day + before, 0) : min(day + after + 1, n)] = True
    return mask


def label_one(
    ret: np.ndarray, factors: np.ndarray, rf: np.ndarray, open_: np.ndarray, high: np.ndarray, low: np.ndarray,
    close: np.ndarray, dollar_volume: np.ndarray, day0: int, excluded: np.ndarray, cfg: Config,
) -> dict[str, float] | None:
    """Labels of one (event, asset) given session-indexed arrays and the position of day 0."""
    spec = cfg.get("labels")
    n = len(ret)
    end = day0 - int(spec["beta_gap_sessions"])
    start = max(end - int(spec["beta_window_sessions"]) + 1, 0)
    if end <= start or day0 + max(w[1] for w in spec["windows"].values()) >= n:
        return None
    usable = np.zeros(n, dtype=bool)
    usable[start : end + 1] = True
    fit = fit_residual_model(ret - rf, factors, usable & ~excluded, int(spec["min_beta_obs"]))
    fallback = False
    if fit is None:  # too many event windows: estimate on all days of the window, and say so
        fit, fallback = fit_residual_model(ret - rf, factors, usable, int(spec["min_beta_obs"])), True
    if fit is None or fit[1] <= 0:
        return None
    beta, sigma = fit
    residual = ret - rf - np.column_stack([np.ones(n), factors]) @ beta
    out: dict[str, float] = {"beta_fallback": float(fallback)}
    for name, (lo, hi) in spec["windows"].items():
        days = residual[day0 + lo : day0 + hi + 1]
        scale = sigma * (np.sqrt(len(days)) if spec["car_standardise"] == SQRT_N else 1.0)
        out[name] = float(days.sum() / scale)
    baseline = slice(max(day0 - int(spec["baseline_sessions"]), 0), day0)
    keep = ~excluded[baseline]
    gk = garman_klass(open_, high, low, close, float(spec["gk_ln2_term"]))
    base_var = np.nanmean(gk[baseline][keep]) if keep.any() else np.nan
    v_lo, v_hi = spec["vol_window"]
    event_var = np.nanmean(gk[day0 + v_lo : day0 + v_hi + 1])
    d_lo, d_hi = spec["dv_window"]
    base_dv = np.nanmean(dollar_volume[baseline][keep]) if keep.any() else np.nan
    event_dv = np.nanmean(dollar_volume[day0 + d_lo : day0 + d_hi + 1])
    with np.errstate(divide="ignore", invalid="ignore"):
        out["log_vol_ratio"] = float(HALF * np.log(event_var / base_var))
        out["log_dv_ratio"] = float(np.log(event_dv / base_dv))
    return out


def build_labels(
    events: pd.DataFrame, prices: dict[str, pd.DataFrame], factors: pd.DataFrame, cfg: Config
) -> pd.DataFrame:
    """Label rows ``(event_id, security_id, window, abn_ret, log_vol_ratio, log_dv_ratio)``.

    ``events``: ``event_id, security_id, day0`` (day-0 session). ``prices[security_id]`` is a frame on
    the common session index with ``ret, open, high, low, close_raw, volume_raw``. Assets or events
    without enough history are skipped (no row), never imputed.
    """
    spec = cfg.get("labels")
    before, after = (int(x) for x in spec["event_window_exclusion"])
    factor_names = list(spec["factors"])
    sessions = factors.index
    rows = []
    for security, part in events.groupby("security_id"):
        px = prices.get(security)
        if px is None:
            continue
        px = px.reindex(sessions)
        day_idx = sessions.get_indexer(pd.DatetimeIndex(part["day0"]))
        excluded = event_window_mask(len(sessions), day_idx[day_idx >= 0], before, after)
        dollar = (px["close_raw"] * px["volume_raw"]).to_numpy()
        for event_id, day0 in zip(part["event_id"], day_idx, strict=True):
            if day0 < 0:
                continue
            own = excluded.copy()
            own[max(day0 + before, 0) : day0 + after + 1] = False
            result = label_one(
                px["ret"].to_numpy(), factors[factor_names].to_numpy(), factors["rf"].to_numpy(),
                px["open"].to_numpy(), px["high"].to_numpy(), px["low"].to_numpy(), px["close_raw"].to_numpy(),
                dollar, int(day0), own, cfg,
            )
            if result is None:
                continue
            for window in spec["windows"]:
                rows.append({"event_id": event_id, "security_id": security, "window": window,
                             "abn_ret": result[window], "log_vol_ratio": result["log_vol_ratio"],
                             "log_dv_ratio": result["log_dv_ratio"], "beta_fallback": result["beta_fallback"]})
    return pd.DataFrame(rows, columns=["event_id", "security_id", "window", "abn_ret", "log_vol_ratio", "log_dv_ratio",
                                       "beta_fallback"])
