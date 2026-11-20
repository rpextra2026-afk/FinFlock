"""Dev/test windows per databucket and the per-LLM split of the test window (spec 3.7, PLAN C10)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from oryxis.common.config import Config
from oryxis.common.timing import TradingCalendar

DEV = "dev"
EMBARGO = "embargo"
TEST = "test"
OUTSIDE = "outside"
POST_CUTOFF = "post_cutoff"
IN_CUTOFF = "in_cutoff"
TEST_FULL = "test_full"


class WindowError(ValueError):
    """Raised when configured windows break the embargo or data-end rules."""


@dataclass(frozen=True)
class BucketWindows:
    """Calendar-snapped dev and test windows of one databucket."""

    bucket: str
    dev_start: pd.Timestamp
    dev_end: pd.Timestamp
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    embargo_sessions: int

    def label(self, dates: pd.DatetimeIndex | pd.Series) -> pd.Series:
        """``dev``, ``embargo``, ``test`` or ``outside`` for each date."""
        index = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
        out = pd.Series(OUTSIDE, index=range(len(index)), dtype="object")
        out[(index >= self.dev_start) & (index <= self.dev_end)] = DEV
        out[(index > self.dev_end) & (index < self.test_start)] = EMBARGO
        out[(index >= self.test_start) & (index <= self.test_end)] = TEST
        return out


def bucket_windows(cfg: Config, bucket: str, calendar: TradingCalendar) -> BucketWindows:
    """Snap the configured dates to sessions and check the embargo and the data end."""
    raw = cfg.get(f"windows.buckets.{bucket}")
    embargo = int(cfg.get("windows.calendar_embargo_trading_days"))
    data_end = pd.Timestamp(cfg.get("windows.data_end"))
    dev_start = calendar.snap_forward(raw["dev_start"])
    dev_end = calendar.snap_backward(raw["dev_end"])
    test_start = calendar.snap_forward(raw["test_start"])
    test_end = calendar.snap_backward(raw["test_end"])
    if pd.Timestamp(raw["test_end"]) > data_end:
        raise WindowError(f"{bucket}: configured test_end is after the data end {data_end.date()}")
    if not dev_start < dev_end < test_start < test_end:
        raise WindowError(f"{bucket}: windows are not in order")
    gap = calendar.count_between(dev_end, test_start)
    if gap < embargo:
        raise WindowError(f"{bucket}: only {gap} sessions between dev and test, need {embargo}")
    return BucketWindows(bucket, dev_start, dev_end, test_start, test_end, embargo)


def clean_start(
    stated_cutoff: str, calendar: TradingCalendar, empirical_cutoff: str | None = None
) -> pd.Timestamp:
    """First session of the month after max(stated, empirical) cutoff (both given as ``YYYY-MM``)."""
    months = [pd.Period(stated_cutoff, freq="M")]
    if empirical_cutoff:
        months.append(pd.Period(empirical_cutoff, freq="M"))
    first_clean_month = max(months) + 1
    return calendar.snap_forward(first_clean_month.start_time)


def cutoff_part(
    dates: pd.DatetimeIndex | pd.Series, windows: BucketWindows, clean: pd.Timestamp
) -> pd.Series:
    """``post_cutoff`` / ``in_cutoff`` for test dates, ``None`` elsewhere."""
    index = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
    in_test = (index >= windows.test_start) & (index <= windows.test_end)
    out = pd.Series([None] * len(index), index=range(len(index)), dtype="object")
    out[in_test & (index >= clean)] = POST_CUTOFF
    out[in_test & (index < clean)] = IN_CUTOFF
    return out
