"""From as-traded prices to returns, dollar volume and market cap (spec 3.2).

Rules that must hold exactly:

* returns are total returns from the as-traded price plus the cash dividend on the ex-date, with the
  split ratio applied on the ex-date: ``r_t = (P_t + D_t) S_t / P_{t-1} - 1``;
* dollar volume is ``close_asTraded x volume_asTraded`` of the same date, never ``adj_close x volume``
  (when a vendor split-adjusts volume retroactively it is rebuilt from the split table);
* market cap at ``t`` is the as-traded close of ``t-1`` times the latest share count filed by ``t-1``,
  split-adjusted from the cover date to ``t``.

Panels are wide: index = session date (ascending), columns = security id. A missing price gives a
missing return; gaps are never bridged silently.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

SPLIT = "split"
DIVIDEND = "dividend"


def wide(frame: pd.DataFrame, value: str, sessions: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    """Pivot a long price table (``security_id``, ``session_date``) into a sessions x securities panel."""
    panel = frame.pivot_table(index="session_date", columns="security_id", values=value, aggfunc="last")
    panel.index = pd.DatetimeIndex(panel.index)
    return panel.reindex(sessions) if sessions is not None else panel.sort_index()


def action_panel(
    actions: pd.DataFrame, action_type: str, value_col: str, sessions: pd.DatetimeIndex,
    securities: pd.Index, neutral: float,
) -> pd.DataFrame:
    """Sessions x securities panel of split ratios (neutral 1) or dividends (neutral 0) on ex-dates."""
    part = actions[actions["action_type"] == action_type]
    panel = pd.DataFrame(neutral, index=sessions, columns=securities, dtype=float)
    for row in part.itertuples():
        date = pd.Timestamp(row.ex_date)
        if row.security_id in panel.columns and date in panel.index:
            amount = float(getattr(row, value_col))
            panel.at[date, row.security_id] = panel.at[date, row.security_id] * amount if action_type == SPLIT else (
                panel.at[date, row.security_id] + amount
            )
    return panel


def total_returns(
    close: pd.DataFrame, open_: pd.DataFrame, actions: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Close-to-close return of each session and open-to-open return of holding from that open.

    Row ``t`` of the second panel is the return from the open of ``t`` to the open of the next session
    (it includes a dividend or split whose ex-date is the next session, because a holder at the open
    of ``t`` owns the share through that ex-date).
    """
    sessions, securities = close.index, close.columns
    splits = action_panel(actions, SPLIT, "ratio", sessions, securities, 1.0)
    dividends = action_panel(actions, DIVIDEND, "amount", sessions, securities, 0.0)
    close_ret = (close + dividends) * splits / close.shift(1) - 1.0
    next_open = open_.shift(-1)
    oo_ret = (next_open + dividends.shift(-1)) * splits.shift(-1) / open_ - 1.0
    return close_ret, oo_ret


def future_split_factor(actions: pd.DataFrame, sessions: pd.DatetimeIndex, securities: pd.Index) -> pd.DataFrame:
    """``F_t = prod of split ratios with ex-date after t``: how much a vendor inflates old share counts."""
    splits = action_panel(actions, SPLIT, "ratio", sessions, securities, 1.0)
    reverse_cumulative = splits[::-1].cumprod()[::-1]
    return reverse_cumulative.shift(-1).fillna(1.0)


def volume_as_traded(volume_vendor: pd.DataFrame, actions: pd.DataFrame, vendor_adjusts_volume: bool) -> pd.DataFrame:
    """Shares traded on each date as they were then (undo a vendor's retroactive split adjustment)."""
    if not vendor_adjusts_volume:
        return volume_vendor
    factor = future_split_factor(actions, volume_vendor.index, volume_vendor.columns)
    return volume_vendor / factor


def dollar_volume(close_raw: pd.DataFrame, volume_raw: pd.DataFrame) -> pd.DataFrame:
    """``DV = close_asTraded x volume_asTraded`` of the same date."""
    return close_raw * volume_raw


def shares_panel(
    shares_out: pd.DataFrame, actions: pd.DataFrame, sessions: pd.DatetimeIndex, securities: pd.Index
) -> pd.DataFrame:
    """Latest share count known on each session, split-adjusted from its cover date to that session.

    A count is known from the session after its ``filed_utc`` date (information through ``t - 1``);
    the returned panel is indexed by decision date ``t``.
    """
    out = pd.DataFrame(np.nan, index=sessions, columns=securities)
    split_by_security = {
        sec: part.sort_values("ex_date") for sec, part in actions[actions["action_type"] == SPLIT].groupby("security_id")
    }
    for security, part in shares_out.groupby("security_id"):
        if security not in out.columns:
            continue
        part = part.sort_values("filed_utc")
        filed = pd.to_datetime(part["filed_utc"], utc=True).dt.tz_localize(None).dt.normalize()
        known_from = sessions.searchsorted(filed.to_numpy(), side="right")  # first session strictly after filing
        for index, row in enumerate(part.itertuples()):
            start = known_from[index]
            stop = known_from[index + 1] if index + 1 < len(part) else len(sessions)
            if start >= len(sessions):
                continue
            window = sessions[start:stop]
            adjust = pd.Series(1.0, index=window)
            for split in split_by_security.get(security, pd.DataFrame(columns=["ex_date", "ratio"])).itertuples():
                if pd.Timestamp(row.cover_date) < pd.Timestamp(split.ex_date):
                    adjust[window >= pd.Timestamp(split.ex_date)] *= float(split.ratio)
            out.loc[window, security] = float(row.shares) * adjust
    return out


def market_cap(close_raw: pd.DataFrame, shares: pd.DataFrame) -> pd.DataFrame:
    """Market cap at ``t``: close of ``t - 1`` as traded times the share count known at ``t``."""
    return close_raw.shift(1) * shares


def last_trading_dates(close_raw: pd.DataFrame) -> pd.Series:
    """Last session with a close for each security (delisting date when it is before the panel end)."""
    return close_raw.apply(lambda col: col.last_valid_index())


def suspect_returns(returns: pd.DataFrame, limit: float) -> pd.DataFrame:
    """Boolean panel of returns whose absolute value exceeds ``limit`` (logged, never clipped)."""
    return returns.abs() > limit


# --- trailing state known before an event (no data from day 0 or later) --------------------------------------------


def trailing_window_end(index: pd.DatetimeIndex, day0: pd.Timestamp) -> int:
    """Position of the first session on or after ``day0``: rows before it (up to day -1) are the only ones a feature may read."""
    return int(index.searchsorted(pd.Timestamp(day0), side="left"))


def trailing_state(ret: pd.DataFrame, dollar_volume: pd.DataFrame, security_ids, day0s, window: int) -> pd.DataFrame:
    """Stock's own recent state ending on day -1: ``own_vol_trail`` = log std of its daily returns, ``own_volu_trail`` = log of its mean
    dollar volume, over the ``window`` sessions before day 0. Missing (NaN) when the stock has no price history there."""
    out = np.full((len(security_ids), 2), np.nan)
    for row, (security, day0) in enumerate(zip(security_ids, day0s, strict=True)):
        if security not in ret.columns:
            continue
        end = trailing_window_end(ret.index, day0)
        start = max(end - window, 0)
        returns = ret[security].iloc[start:end].dropna()
        volume = dollar_volume[security].iloc[start:end] if security in dollar_volume.columns else pd.Series(dtype=float)
        volume = volume[volume > 0].dropna()
        if len(returns) >= 2 and returns.std() > 0:
            out[row, 0] = np.log(returns.std())
        if len(volume):
            out[row, 1] = np.log(volume.mean())
    return pd.DataFrame(out, columns=["own_vol_trail", "own_volu_trail"])


def _expanding_tercile(values: pd.Series, bins: int, min_history: int) -> pd.Series:
    """Rank of each value among the EARLIER values only (expanding history), cut into ``bins`` equal parts: 0 .. bins-1."""
    out = pd.Series(np.nan, index=values.index)
    history: list[float] = []
    for stamp, value in values.items():
        if np.isfinite(value):
            if len(history) >= min_history:
                rank = np.searchsorted(np.sort(history), value, side="left") / len(history)
                out[stamp] = min(int(rank * bins), bins - 1)
            history.append(float(value))
    return out


def market_state_terciles(market_ret: pd.Series, market_dollar_volume: pd.Series, day0s, window: int, bins: int, min_history: int) -> pd.DataFrame:
    """Terciles (0 .. bins-1) of the market's trailing realised vol and trailing dollar volume on day -1, ranked against the history
    before that day only (the C5 rule). ``mkt_vol_tercile`` and ``mkt_volu_tercile`` per entry of ``day0s``; NaN while history is short."""
    vol = market_ret.rolling(window).std().shift(1)  # the value on date t uses sessions up to t-1
    volume = market_dollar_volume.rolling(window).mean().shift(1)
    vol_t = _expanding_tercile(vol, bins, min_history)
    volume_t = _expanding_tercile(volume, bins, min_history)
    positions = [market_ret.index.searchsorted(pd.Timestamp(d), side="left") for d in day0s]
    last = len(market_ret.index) - 1
    picks = [market_ret.index[min(p, last)] for p in positions]
    return pd.DataFrame({"mkt_vol_tercile": vol_t.reindex(picks).to_numpy(), "mkt_volu_tercile": volume_t.reindex(picks).to_numpy()})
