"""Where do the delisting rules differ, and why? One row per name that took a delisting return in a backtest.

The same strategy gives a very different result under the two rules (0 or the default delisting return). This trace lists the names the rule touched, how
many fill dates each was hit on, the return they contribute and the price rows they sit on (missing, stale, delisted, split), so that a data error can be
told from a holding that really was delisted. It changes nothing; numbers only.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finflock.portfolio.execute import BacktestResult

SPLIT_WINDOW_SESSIONS = 10  # a split this close to the last close is listed beside the name
TAIL_SESSIONS = 10  # trailing rows looked at for stale prices


def hit_table(result: BacktestResult, rm) -> pd.DataFrame:
    """Every (fill date, name) where a held name had no return and its last close is on or before the fill and before the panel end (the executor's delisting test)."""
    panel_end = rm.market.sessions[-1]
    last = rm.market.last_trading_date
    rows = []
    for fill in result.fill_dates:
        held = result.positions.loc[fill]
        held = held[held != 0]
        if held.empty:
            continue
        returns = rm.market.fill_return(fill).reindex(held.index)
        miss = returns.isna()
        for name in held.index[miss.to_numpy()]:
            if pd.notna(last.get(name)) and last[name] < panel_end and last[name] <= fill:
                rows.append({"fill_date": fill, "security_id": name, "weight": float(held[name]), "last_close": last[name], "sessions_after_last_close": int(rm.market.sessions.searchsorted(fill) - rm.market.sessions.searchsorted(last[name]))})
    return pd.DataFrame(rows, columns=["fill_date", "security_id", "weight", "last_close", "sessions_after_last_close"])


def price_rows(px: pd.DataFrame, actions: pd.DataFrame, name: str, last_close: pd.Timestamp, rm) -> dict:
    """What the price table holds around a name's last close: rows after it, missing closes, zero volume, unchanged closes (stale), splits."""
    part = px[px["security_id"] == name].sort_values("session_date")
    after = part[part["session_date"] > last_close]
    tail = part[part["session_date"] <= last_close].tail(TAIL_SESSIONS)
    close = tail["close_raw"].to_numpy(dtype=float)
    stale = int((np.diff(close) == 0).sum()) if len(close) > 1 else 0
    sessions = rm.market.sessions
    pos = sessions.searchsorted(last_close)
    lo, hi = sessions[max(pos - SPLIT_WINDOW_SESSIONS, 0)], sessions[min(pos + SPLIT_WINDOW_SESSIONS, len(sessions) - 1)]
    split = actions[(actions["security_id"] == name) & (actions["action_type"] == "split") & actions["ex_date"].between(lo, hi)]
    return {"px_rows_after_last_close": int(len(after)), "px_rows_after_with_no_close": int(after["close_raw"].isna().sum()) if len(after) else 0,
            "tail_missing_close": int(tail["close_raw"].isna().sum()), "tail_zero_volume": int((tail["volume_raw"].fillna(0) == 0).sum()), "tail_unchanged_close": stale,
            "split_near_last_close": bool(len(split)), "first_px_date": part["session_date"].min() if len(part) else pd.NaT, "last_px_date": part["session_date"].max() if len(part) else pd.NaT}


def classify(row: pd.Series) -> str:
    if row["px_rows_after_last_close"] > 0:
        return "price rows exist after the last close (check for a gap in the close column)"
    if row["split_near_last_close"]:
        return "split near the last close (check the adjustment)"
    if row["tail_missing_close"] > 0:
        return "missing closes in the last rows"
    if row["tail_unchanged_close"] >= TAIL_SESSIONS // 2 or row["tail_zero_volume"] > 0:
        return "stale or zero-volume tail"
    return "price table ends: delisted by the table"


def trace(zero: BacktestResult, default: BacktestResult, rm, px: pd.DataFrame, actions: pd.DataFrame, delist_return: float, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(per name, per hit) tables for a pair of backtests of the same strategy under the 0 rule and the default rule.

    ``contribution_gross`` is the sum over hit dates of weight x delisting return (the change in the gross return, in the units of the backtest, i.e. after the risk scale);
    ``contribution_sharpe_days`` is how many days of the gross series differ between the two backtests."""
    hits = hit_table(default, rm)
    diff = (default.gross - zero.gross).reindex(default.gross.index)
    differing = int((diff.abs() > 0).sum())
    if hits.empty:
        return pd.DataFrame(), hits
    hits = hits.assign(contribution_gross=hits["weight"] * delist_return, label=label)
    names = []
    for name, part in hits.groupby("security_id"):
        info = price_rows(px, actions, name, part["last_close"].iloc[0], rm)
        row = {"label": label, "security_id": name, "hit_dates": int(len(part)), "first_hit": part["fill_date"].min(), "last_hit": part["fill_date"].max(),
               "last_close": part["last_close"].iloc[0], "max_sessions_after_last_close": int(part["sessions_after_last_close"].max()),
               "hits_after_first_session": int((part["sessions_after_last_close"] > 0).sum()), "mean_weight": float(part["weight"].mean()), "max_abs_weight": float(part["weight"].abs().max()),
               "contribution_gross": float(part["contribution_gross"].sum()), "days_with_different_gross": differing, **info}
        row["data_check"] = classify(pd.Series(row))
        names.append(row)
    table = pd.DataFrame(names).sort_values("contribution_gross", key=np.abs, ascending=False).reset_index(drop=True)
    return table, hits
