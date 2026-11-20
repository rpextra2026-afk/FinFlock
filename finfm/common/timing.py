"""Trading calendar and the day-0 timing rule (spec 3.6).

Day 0 of an event is the first session whose close is strictly after the moment the item became
available. A timestamp exactly at the close belongs to the next session. Date-only items
(``time_known=False``) use the session of that date, or the next one when the date is not a session.
"""

from __future__ import annotations

from collections.abc import Sequence
from functools import cached_property

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from finfm.common.config import Config

UTC = "UTC"


class TradingCalendar:
    """Thin wrapper over an exchange calendar with vectorised day-0 assignment."""

    def __init__(self, name: str, start: str | pd.Timestamp, end: str | pd.Timestamp):
        self.name = name
        self._cal = xcals.get_calendar(name, start=pd.Timestamp(start), end=pd.Timestamp(end))

    @classmethod
    def from_config(cls, cfg: Config) -> TradingCalendar:
        return cls(
            str(cfg.get("calendar.name")),
            str(cfg.get("calendar.start")),
            str(cfg.get("calendar.end")),
        )

    @cached_property
    def sessions(self) -> pd.DatetimeIndex:
        """All sessions as tz-naive dates."""
        return pd.DatetimeIndex(self._cal.sessions).tz_localize(None)

    @cached_property
    def closes_utc(self) -> pd.DatetimeIndex:
        """Close time of every session in UTC (early closes included)."""
        return pd.DatetimeIndex(self._cal.closes).tz_convert(UTC)

    @cached_property
    def opens_utc(self) -> pd.DatetimeIndex:
        """Open time of every session in UTC."""
        return pd.DatetimeIndex(self._cal.opens).tz_convert(UTC)

    def text_decision_date(self, available_utc: pd.Series | Sequence[pd.Timestamp]) -> pd.DatetimeIndex:
        """Session after whose close a text-only signal is decided (fill rule E1, spec 9.4).

        News available before the open of its day-0 session can be filled at that open, so it is decided
        at the close of the previous session; later news is filled at the open of the next session, so it
        is decided at the close of day 0.
        """
        stamps = pd.DatetimeIndex(pd.to_datetime(available_utc, utc=True))
        day0 = self.day0(stamps)
        position = self.sessions.get_indexer(day0)
        before_open = stamps < self.opens_utc[position]
        decision = np.where(before_open, position - 1, position)
        return self.sessions[np.maximum(decision, 0)]

    def is_session(self, date: str | pd.Timestamp) -> bool:
        return bool(self._cal.is_session(pd.Timestamp(date).normalize()))

    def day0(self, available_utc: Sequence[pd.Timestamp] | pd.Series) -> pd.DatetimeIndex:
        """Day-0 session (date) for each UTC timestamp: first close strictly after it."""
        stamps = pd.DatetimeIndex(pd.to_datetime(available_utc, utc=True))
        position = self.closes_utc.searchsorted(stamps, side="right")
        if (position >= len(self.sessions)).any():
            raise ValueError("an item is available after the last session of the calendar")
        return self.sessions[position]

    def day0_date_only(self, dates: Sequence[pd.Timestamp] | pd.Series) -> pd.DatetimeIndex:
        """Day-0 session for date-only items: that date's session, else the next session."""
        values = pd.DatetimeIndex(pd.to_datetime(dates)).normalize()
        position = self.sessions.searchsorted(values, side="left")
        if (position >= len(self.sessions)).any():
            raise ValueError("a date lies after the last session of the calendar")
        return self.sessions[position]

    def shift(self, date: str | pd.Timestamp, n_sessions: int) -> pd.Timestamp:
        """The session ``n_sessions`` after (or before, if negative) the session ``date``."""
        base = self.sessions.searchsorted(pd.Timestamp(date).normalize(), side="left")
        index = base + n_sessions
        if index < 0 or index >= len(self.sessions):
            raise ValueError("shift leaves the calendar")
        return self.sessions[index]

    def snap_forward(self, date: str | pd.Timestamp) -> pd.Timestamp:
        """The date itself when it is a session, else the next session."""
        return self.shift(date, 0)

    def snap_backward(self, date: str | pd.Timestamp) -> pd.Timestamp:
        """The date itself when it is a session, else the previous session."""
        target = pd.Timestamp(date).normalize()
        index = self.sessions.searchsorted(target, side="right") - 1
        if index < 0:
            raise ValueError("date precedes the calendar")
        return self.sessions[index]

    def sessions_between(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> pd.DatetimeIndex:
        """Sessions with ``start <= session <= end``."""
        lo = self.sessions.searchsorted(pd.Timestamp(start).normalize(), side="left")
        hi = self.sessions.searchsorted(pd.Timestamp(end).normalize(), side="right")
        return self.sessions[lo:hi]

    def count_between(self, start: str | pd.Timestamp, end: str | pd.Timestamp) -> int:
        """Number of sessions strictly between two dates."""
        lo = self.sessions.searchsorted(pd.Timestamp(start).normalize(), side="right")
        hi = self.sessions.searchsorted(pd.Timestamp(end).normalize(), side="left")
        return int(max(hi - lo, 0))


def to_utc(timestamps: Sequence[pd.Timestamp] | pd.Series, source_tz: str) -> pd.DatetimeIndex:
    """Localise naive timestamps in ``source_tz`` and convert to UTC (ambiguous times raise)."""
    index = pd.DatetimeIndex(pd.to_datetime(timestamps))
    if index.tz is None:
        index = index.tz_localize(source_tz, ambiguous="raise", nonexistent="raise")
    return index.tz_convert(UTC)

