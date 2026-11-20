"""Exact equal-count bins by rank (no floating-point edges)."""

from __future__ import annotations

import numpy as np
import pandas as pd


def rank_bins(values: pd.Series, n_bins: int) -> pd.Series:
    """Bin index ``0 .. n_bins-1`` by rank (ties broken by order); NaN values stay NaN.

    With ``n`` known values the bin of rank ``r`` (1-based) is ``floor((r - 1) * n_bins / n)``, so six
    values in three bins give two per bin.
    """
    known = values.dropna()
    ranks = known.rank(method="first").to_numpy()
    bins = np.floor((ranks - 1) * n_bins / len(known)).astype(int) if len(known) else np.array([], dtype=int)
    return pd.Series(bins, index=known.index).reindex(values.index)
