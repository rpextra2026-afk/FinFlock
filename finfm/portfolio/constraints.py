"""Portfolio constraints applied after the band (spec 9.3): headline long-only and long-short."""

from __future__ import annotations

import numpy as np

from finfm.common.config import Config


HALF = 0.5


def _bisect_shift(
    values: np.ndarray, lower: float, upper: float, target: float, lo: float, hi: float,
    tol: float, max_iter: int,
) -> np.ndarray:
    """``clip(values - tau, lower, upper)`` with ``tau`` chosen so the sum equals ``target``."""
    for _ in range(max_iter):
        tau = HALF * (lo + hi)
        total = np.clip(values - tau, lower, upper).sum()
        if abs(total - target) <= tol:
            break
        if total > target:
            lo = tau
        else:
            hi = tau
    return np.clip(values - tau, lower, upper)


def project_long_only(n: np.ndarray, cap: float, max_gross: float, tol: float, max_iter: int) -> np.ndarray:
    """Euclidean projection onto ``0 <= n <= cap`` and ``sum(n) <= max_gross`` (cash is the rest)."""
    clipped = np.clip(n, 0.0, cap)
    if clipped.sum() <= max_gross:
        return clipped
    span = np.abs(n).max() + cap
    return _bisect_shift(n, 0.0, cap, max_gross, 0.0, span, tol, max_iter)


def project_long_short(n: np.ndarray, cap: float, max_gross: float, tol: float, max_iter: int) -> np.ndarray:
    """Dollar-neutral ``|n| <= cap``, ``sum(n) = 0``, then scaled to gross exposure ``<= max_gross``."""
    span = np.abs(n).max() + cap
    neutral = _bisect_shift(n, -cap, cap, 0.0, -span, span, tol, max_iter)
    gross = np.abs(neutral).sum()
    return neutral * (max_gross / gross) if gross > max_gross else neutral


def project(n: np.ndarray, cfg: Config, kind: str) -> np.ndarray:
    """Apply the named constraint set (``headline`` or ``long_short``) from config."""
    spec = cfg.get(f"portfolio.constraints.{kind}")
    tol = float(cfg.get("portfolio.constraints.projection_tol"))
    max_iter = int(cfg.get("portfolio.constraints.projection_max_iter"))
    cap, gross = float(spec["cap"]), float(spec["max_gross"])
    if spec["long_only"]:
        return project_long_only(n, cap, gross, tol, max_iter)
    return project_long_short(n, cap, gross, tol, max_iter)
