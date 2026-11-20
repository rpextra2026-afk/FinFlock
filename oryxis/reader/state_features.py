"""State features of a pin reading that are known before the event (spec 4.5): the stock's own recent vol and volume ending on day -1,
the market-state terciles of the C5 rule, and the reader's scheduled/surprise flag. Nothing here reads day 0 or later, and nothing
here reads a label."""

from __future__ import annotations

import pandas as pd

from oryxis.common.config import Config
from oryxis.data.market import market_state_terciles, trailing_state

STATE_COLUMNS = ("own_vol_trail", "own_volu_trail", "mkt_vol_tercile", "mkt_volu_tercile")


def market_proxy(ret: pd.DataFrame, dollar_volume: pd.DataFrame, symbol: str) -> tuple[pd.Series, pd.Series]:
    """The broad-index return and dollar volume: the named index symbol when it is in the panel, else the equal-weighted S&P panel."""
    if symbol in ret.columns:
        return ret[symbol], dollar_volume[symbol] if symbol in dollar_volume.columns else dollar_volume.sum(axis=1)
    return ret.mean(axis=1), dollar_volume.sum(axis=1)


def add_state_features(pins: pd.DataFrame, ret: pd.DataFrame, dollar_volume: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Copy of ``pins`` (needs ``node_id`` and ``day0``) with the state columns added."""
    spec = cfg.get("reader.dials.state")
    out = pins.reset_index(drop=True).copy()
    own = trailing_state(ret, dollar_volume, out["node_id"].tolist(), out["day0"].tolist(), int(spec["window_sessions"]))
    mkt_ret, mkt_dv = market_proxy(ret, dollar_volume, str(spec["market_symbol"]))
    market = market_state_terciles(mkt_ret, mkt_dv, out["day0"].tolist(), int(spec["market_window_sessions"]), int(spec["market_bins"]),
                                   int(spec["market_min_history"]))
    for frame in (own, market):
        for column in frame.columns:
            out[column] = frame[column].to_numpy()
    return out


def dollar_volume_from_prices(prices: dict[str, pd.DataFrame], sessions: pd.DatetimeIndex, close: str = "close_raw", volume: str = "volume_raw") -> pd.DataFrame:
    """Sessions x securities dollar volume ``close_raw x volume_raw`` of the same date (never the adjusted close)."""
    return pd.DataFrame({s: p[close] * p[volume] for s, p in prices.items()}).reindex(sessions)
