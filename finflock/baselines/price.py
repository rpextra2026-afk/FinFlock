"""N3 short-term reversal and N4 12-1 momentum from total returns known at the decision session (no later data is read)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _window_return(cc: pd.DataFrame, security: str, date: pd.Timestamp, start_lag: int, end_lag: int) -> float:
    """Compounded total return over the sessions (t - start_lag, t - end_lag], t = the decision session; NaN when the history is short or missing."""
    if security not in cc.columns:
        return np.nan
    position = cc.index.searchsorted(pd.Timestamp(date), side="right")      # first session after the decision date
    last, first = position - end_lag, position - start_lag
    if first < 0 or last <= first:
        return np.nan
    part = cc[security].iloc[first:last]
    return float(np.nan) if part.isna().all() else float(np.expm1(np.log1p(part.fillna(0.0)).sum()))


def reversal(cc: pd.DataFrame, units: pd.DataFrame, days: int) -> pd.Series:
    """Minus the return of the last ``days`` sessions ending at the decision session."""
    return pd.Series([-_window_return(cc, s, d, days, 0) for s, d in zip(units["security_id"], units["decision_date"], strict=True)], index=units.index)


def momentum(cc: pd.DataFrame, units: pd.DataFrame, start_lag: int, skip: int) -> pd.Series:
    """Return from ``start_lag`` to ``skip`` sessions before the decision session."""
    return pd.Series([_window_return(cc, s, d, start_lag, skip) for s, d in zip(units["security_id"], units["decision_date"], strict=True)], index=units.index)
