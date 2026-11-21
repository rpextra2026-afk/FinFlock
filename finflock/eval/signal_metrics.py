"""Signal yardsticks common to all frameworks (spec 11.4 A): rank IC, ICIR and its Newey-West t."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from finflock.eval.stats import nw_tstat


def forward_returns(open_to_open: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Compounded open-to-open return over ``horizon`` sessions starting at the fill (row = decision date shifted by one)."""
    growth = (1.0 + open_to_open).rolling(horizon).apply(np.prod, raw=True) - 1.0
    return growth.shift(-(horizon - 1))


def daily_rank_ic(signals: pd.DataFrame, forward: pd.DataFrame, min_stocks: int) -> pd.Series:
    """Spearman correlation per decision date between the score and the forward return.

    ``signals``: ``decision_date, security_id, score``; ``forward``: dates x securities (row = fill date, the
    session after the decision). Days with fewer than ``min_stocks`` scored stocks are pooled within their
    calendar month instead of dropped (spec 11.4 A.1).
    """
    out, pooled = {}, []
    sessions = forward.index
    for date, part in signals.groupby("decision_date"):
        position = sessions.searchsorted(pd.Timestamp(date), side="right")
        if position >= len(sessions):
            continue
        fwd = forward.iloc[position].reindex(part["security_id"]).to_numpy()
        frame = pd.DataFrame({"s": part["score"].to_numpy(), "r": fwd, "date": pd.Timestamp(date)}).dropna()
        if len(frame) >= min_stocks and frame["s"].nunique() > 1 and frame["r"].nunique() > 1:
            out[pd.Timestamp(date)] = float(spearmanr(frame["s"], frame["r"])[0])
        else:
            pooled.append(frame)
    if pooled:
        everything = pd.concat(pooled)
        for _, month in everything.groupby(everything["date"].dt.to_period("M")):
            if len(month) >= min_stocks and month["s"].nunique() > 1 and month["r"].nunique() > 1:
                out[month["date"].max()] = float(spearmanr(month["s"], month["r"])[0])
    return pd.Series(out).sort_index()


def ic_report(ic: pd.Series, lag: int) -> dict[str, float]:
    """Mean IC, IC information ratio, its t-statistic ``ICIR sqrt(T)`` and the Newey-West t of the series."""
    clean = ic.dropna()
    icir = float(clean.mean() / clean.std()) if clean.std() > 0 else float("nan")
    return {"ic": float(clean.mean()), "icir": icir, "t_naive": icir * float(np.sqrt(len(clean))),
            "t_newey_west": nw_tstat(clean.to_numpy(), lag), "n_dates": int(len(clean))}
