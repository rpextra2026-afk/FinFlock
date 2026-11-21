"""Rival signals through the SAME optimiser (equal terms): the rival's daily score becomes a one-day expected-return forecast, then the usual strategy, risk target and costs.

A rival hands over one frame ``decision_date, security_id, score`` (Basic, FinRipple-lite and MASS each have their own scorer; this module does not read their tapes).
The score is turned into a return forecast by one slope fitted on the rival's own scores BEFORE the test window (dev days): the day-1 open-to-open return regressed on the
score. No test day is read. The forecast is put in the first session after the decision date only, as the rivals' own rules hold for one day. Same events table, same decision days
and same optimiser as every variant (events_test.parquet and days_test.parquet).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from finflock.framework.run import EventForecast

MIN_DEV_POINTS = 200  # fewer scored (day, stock) pairs before the test window: the slope is not fitted and the rival is not run


def fit_slope(scores: pd.DataFrame, oo_ret: pd.DataFrame, test_start: pd.Timestamp) -> tuple[float, int]:
    """Slope of the next-session open-to-open return on the score, through the origin, over scores dated before ``test_start``. Returns (slope, n points)."""
    dev = scores[pd.to_datetime(scores["decision_date"]) < test_start]
    sessions = oo_ret.index
    xs, ys = [], []
    for date, part in dev.groupby("decision_date"):
        pos = sessions.searchsorted(pd.Timestamp(date), side="right")
        if pos >= len(sessions):
            continue
        realised = oo_ret.iloc[pos].reindex(part["security_id"]).to_numpy()
        keep = np.isfinite(realised) & np.isfinite(part["score"].to_numpy(dtype=float))
        xs.append(part["score"].to_numpy(dtype=float)[keep])
        ys.append(realised[keep])
    if not xs:
        return float("nan"), 0
    x, y = np.concatenate(xs), np.concatenate(ys)
    if len(x) < MIN_DEV_POINTS or float(x @ x) == 0.0:
        return float("nan"), int(len(x))
    return float((x @ y) / (x @ x)), int(len(x))


def forecasts_from_scores(name: str, scores: pd.DataFrame, days: Sequence[pd.Timestamp], sessions: pd.DatetimeIndex, global_assets: list[str], slope: float, horizon: int) -> list[EventForecast]:
    """One ``EventForecast`` per decision day of ``days`` that has scores: day-1 increment = slope x score, nothing after; dials and covariance shift are zero."""
    index = {a: i for i, a in enumerate(global_assets)}
    wanted = set(pd.DatetimeIndex(days))
    out = []
    for date, part in scores.groupby("decision_date"):
        stamp = pd.Timestamp(date)
        if stamp not in wanted or stamp not in sessions:
            continue
        part = part[part["security_id"].isin(index)]
        if part.empty:
            continue
        assets = np.array([index[s] for s in part["security_id"]], dtype=int)
        inc = np.zeros((horizon, len(assets)))
        inc[0] = slope * part["score"].to_numpy(dtype=float)
        out.append(EventForecast(f"{name}:{stamp.date()}", int(sessions.get_loc(stamp)), assets, inc, np.zeros(len(assets)), np.zeros(len(assets)), None, 1.0))
    return out
