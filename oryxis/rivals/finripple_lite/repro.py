"""FinRipple's own yardsticks (spec 10.3): CAPM-residual regression on the predicted shock, refusal rate.

Pattern to expect from zero-shot-with-graph: a positive but small R2. Reported as measured, with the
refusal-to-answer rate and the share of named companies that could not be mapped.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


MIN_OBSERVATIONS = 3


def capm_residual(stock: pd.Series, market: pd.Series, rf: pd.Series, beta: float) -> pd.Series:
    """``eps = R_i - [R_f + beta (R_m - R_f)]``."""
    return stock - (rf + beta * (market - rf))


def impact_regression(shock: np.ndarray, residual: np.ndarray) -> dict[str, float]:
    """OLS of the standardised residual on the predicted shock: slope, p-value-free t, R2, F-based effect size."""
    ok = np.isfinite(shock) & np.isfinite(residual)
    x, y = shock[ok], residual[ok]
    n = len(x)
    if n < MIN_OBSERVATIONS or np.std(x) == 0:
        return {"slope": float("nan"), "t": float("nan"), "r2": float("nan"), "n": n}
    z = (y - y.mean()) / y.std()
    xc = x - x.mean()
    slope = float((xc * z).sum() / (xc * xc).sum())
    resid = z - z.mean() - slope * xc
    se = float(np.sqrt((resid * resid).sum() / (n - 2) / (xc * xc).sum()))
    r2 = 1.0 - float((resid * resid).sum() / ((z - z.mean()) ** 2).sum())
    return {"slope": slope, "t": slope / se if se > 0 else float("nan"), "r2": r2, "n": n}


def refusal_rate(parse_ok: pd.Series) -> float:
    """Share of answers that were refusals or malformed (the paper's refusal-to-answer yardstick)."""
    return float(1.0 - parse_ok.astype(bool).mean()) if len(parse_ok) else float("nan")
