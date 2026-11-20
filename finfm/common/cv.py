"""Time-blocked cross-validation with an embargo (spec 8.2): splits are by time, never random."""

from __future__ import annotations

import numpy as np
import pandas as pd

from finfm.common.timing import TradingCalendar


def time_blocked_folds(
    dates: pd.Series | pd.DatetimeIndex,
    n_folds: int,
    embargo_sessions: int,
    calendar: TradingCalendar,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Return ``(train_index, validation_index)`` per fold.

    Observations are ordered by date and cut into ``n_folds`` contiguous blocks of (nearly) equal
    size. Training excludes the validation block and every observation within ``embargo_sessions``
    sessions of it on either side, so no training label can overlap a validation window.
    """
    stamps = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    order = np.argsort(stamps.values, kind="stable")
    blocks = np.array_split(order, n_folds)
    folds = []
    for block in blocks:
        first, last = stamps[block].min(), stamps[block].max()
        lo = calendar.shift(first, -embargo_sessions)
        hi = calendar.shift(last, embargo_sessions)
        train = np.flatnonzero((stamps < lo) | (stamps > hi))
        folds.append((train, np.sort(block)))
    return folds
