"""Causal market access: signals see prices only through ``Market.asof`` (spec 9.4, 12.2 #12)."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


class CausalityError(RuntimeError):
    """Raised when code asks for prices dated after its decision date."""


@dataclass(frozen=True)
class MarketView:
    """Prices known at the close of ``asof``. Rows after ``asof`` do not exist for the caller."""

    asof: pd.Timestamp
    close: pd.DataFrame
    volume: pd.DataFrame

    def __post_init__(self) -> None:
        if len(self.close) and self.close.index.max() > self.asof:
            raise CausalityError("view contains rows after its as-of date")


class Market:
    """Daily panels indexed by session date, columns are security ids.

    ``open_to_open`` holds total returns for holding from the open of row ``t`` to the open of the next
    session; ``close`` and ``volume`` are as traded. Decision code gets a ``MarketView``; only the
    executor reads ``open_to_open`` (fill and mark-to-market).
    """

    def __init__(
        self,
        close: pd.DataFrame,
        volume: pd.DataFrame,
        open_to_open: pd.DataFrame,
        last_trading_date: pd.Series | None = None,
    ) -> None:
        if not close.index.is_monotonic_increasing:
            raise ValueError("market index must be sorted")
        self._close, self._volume, self._oo = close, volume, open_to_open
        self.sessions = close.index
        self.last_trading_date = last_trading_date if last_trading_date is not None else pd.Series(dtype="datetime64[ns]")

    def asof(self, date: pd.Timestamp) -> MarketView:
        stamp = pd.Timestamp(date)
        return MarketView(stamp, self._close.loc[:stamp], self._volume.loc[:stamp])

    def fill_return(self, fill_date: pd.Timestamp) -> pd.Series:
        """Total return from the open of ``fill_date`` to the next open (executor only)."""
        return self._oo.loc[pd.Timestamp(fill_date)]

    def next_session(self, date: pd.Timestamp) -> pd.Timestamp | None:
        position = self.sessions.searchsorted(pd.Timestamp(date), side="right")
        return self.sessions[position] if position < len(self.sessions) else None
