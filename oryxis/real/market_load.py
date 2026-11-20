"""Real market panels for the CPU stage: prices, total returns, the executor's ``Market`` and point-in-time membership.

Built from the tables ``fetch_market.py`` writes into the market dataset folder. The same pivots and the same
``total_returns`` as ``scripts/prestage.py``, so the stage sees exactly the prices the unit tables were built from.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from oryxis.common.config import Config
from oryxis.data.market import last_trading_dates, total_returns
from oryxis.data.sp500 import members_on
from oryxis.portfolio.market import Market

PX_FILE = "px_daily.parquet"
ACTIONS_FILE = "corp_action.parquet"
SNAPSHOT_FILE = "sp500_snapshot.parquet"
SESSION = "session_date"
SECURITY = "security_id"


@dataclass
class RealMarket:
    """Wide panels (sessions x securities) and the executor's market."""

    close: pd.DataFrame
    open_: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    ret: pd.DataFrame
    oo_ret: pd.DataFrame
    snapshots: pd.DataFrame
    market: Market

    @property
    def sessions(self) -> pd.DatetimeIndex:
        return self.close.index

    def members(self, date: pd.Timestamp) -> frozenset[str]:
        return members_on(self.snapshots, pd.Timestamp(date)).security_ids

    def universe(self) -> Callable[[pd.Timestamp], set[str]]:
        """Members on a decision date (the latest snapshot on or before it), restricted to securities with prices."""
        have = set(self.close.columns)
        cache: dict[pd.Timestamp, set[str]] = {}

        def of(date: pd.Timestamp) -> set[str]:
            key = pd.Timestamp(date)
            if key not in cache:
                cache[key] = set(self.members(key)) & have
            return cache[key]

        return of


def _panel(px: pd.DataFrame, column: str) -> pd.DataFrame:
    wide = px.pivot_table(index=SESSION, columns=SECURITY, values=column, aggfunc="last").sort_index()
    wide.index = pd.DatetimeIndex(wide.index)
    return wide


def load_real_market(market_dir: Path, cfg: Config, extra_px: pd.DataFrame | None = None, extra_actions: pd.DataFrame | None = None) -> RealMarket:
    """Read the market dataset folder into panels. Securities without a price never appear (counted by the caller).

    ``extra_px`` / ``extra_actions``: more price and corporate-action rows (the game's sleeve proxy) added to the panels only, never written back."""
    px = pd.read_parquet(market_dir / PX_FILE)
    actions = pd.read_parquet(market_dir / ACTIONS_FILE)
    if extra_px is not None:
        px = pd.concat([px[~px[SECURITY].isin(set(extra_px[SECURITY]))], extra_px[px.columns]], ignore_index=True)
    if extra_actions is not None and len(extra_actions):
        actions = pd.concat([actions[~actions[SECURITY].isin(set(extra_actions[SECURITY]))], extra_actions[actions.columns]], ignore_index=True)
    snapshots = pd.read_parquet(market_dir / SNAPSHOT_FILE)
    close, open_ = _panel(px, "close_raw"), _panel(px, "open")
    high, low, volume = _panel(px, "high"), _panel(px, "low"), _panel(px, "volume_raw")
    ret, oo_ret = total_returns(close, open_, actions)
    market = Market(close, volume, oo_ret, last_trading_dates(close))
    return RealMarket(close, open_, high, low, volume, ret, oo_ret, snapshots, market)
