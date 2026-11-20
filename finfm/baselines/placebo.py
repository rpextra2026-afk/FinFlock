"""N5: random placebo scores, reproducible from a seed."""

from __future__ import annotations

import numpy as np
import pandas as pd

from finfm.common.rng import make_rng


def placebo_scores(index: pd.Index, seed: int, draw: int) -> pd.Series:
    return pd.Series(make_rng(seed, "placebo", draw).standard_normal(len(index)), index=index)


def band(values: list[float], low: float, high: float) -> tuple[float, float, float]:
    """(mean, low quantile, high quantile) of the finite values."""
    v = np.array([x for x in values if np.isfinite(x)])
    return (float(v.mean()), float(np.quantile(v, low)), float(np.quantile(v, high))) if len(v) else (np.nan, np.nan, np.nan)
