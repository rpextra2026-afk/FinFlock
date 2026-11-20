"""Statistical building blocks: HAC variance, Newey-West t, Holm, block bootstrap, Sharpe power (D22)."""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
from scipy.stats import chi2, norm, t as student_t

CUBE = 3
CUBE_ROOT = 1.0 / CUBE
HALF = 0.5

HALF = 0.5


def newey_west_variance(x: np.ndarray, lag: int) -> float:
    """Long-run variance of a series with Bartlett weights."""
    x = np.asarray(x, dtype=float)
    x = x - x.mean()
    n = len(x)
    total = float(x @ x) / n
    for k in range(1, min(lag, n - 1) + 1):
        total += 2.0 * (1.0 - k / (lag + 1.0)) * float(x[k:] @ x[:-k]) / n
    return total


def nw_tstat(x: np.ndarray, lag: int) -> float:
    """t-statistic of the mean with a Newey-West standard error."""
    x = np.asarray(x, dtype=float)
    se = np.sqrt(newey_west_variance(x, lag) / len(x))
    return float(x.mean() / se) if se > 0 else float("nan")


def holm(p_values: Mapping[str, float]) -> dict[str, float]:
    """Holm step-down adjusted p-values (monotone, capped at 1)."""
    names = sorted(p_values, key=lambda k: p_values[k])
    m = len(names)
    adjusted, running = {}, 0.0
    for rank, name in enumerate(names):
        running = max(running, (m - rank) * p_values[name])
        adjusted[name] = min(running, 1.0)
    return adjusted


def block_length(n: int) -> int:
    """Default circular block length ``floor(T^(1/3))`` (at least 1)."""
    root = int(round(n ** CUBE_ROOT))
    while root**CUBE > n:  # guard the float cube root (1000 ** (1/3) is 9.999...)
        root -= 1
    return max(root, 1)


def circular_block_indices(n: int, block: int, draws: int, rng: np.random.Generator) -> np.ndarray:
    """``(draws, n)`` indices of a circular block bootstrap."""
    blocks = -(-n // block)
    starts = rng.integers(0, n, size=(draws, blocks))
    idx = (starts[:, :, None] + np.arange(block)[None, None, :]) % n
    return idx.reshape(draws, -1)[:, :n]


def bootstrap_ci(
    data: np.ndarray, statistic: Callable[[np.ndarray], float], block: int, draws: int, level: float, rng: np.random.Generator,
) -> tuple[float, float, float]:
    """Point estimate and percentile interval of ``statistic`` over a circular block bootstrap of rows."""
    index = circular_block_indices(len(data), block, draws, rng)
    values = np.array([statistic(data[i]) for i in index])
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(values, [tail, 1.0 - tail])
    return float(statistic(data)), float(low), float(high)


def sharpe_difference_se(sr1: float, sr2: float, rho: float, years: float, days: int) -> float:
    """Standard error of a difference of two annual Sharpe ratios from daily data (D22, Memmel-corrected).

    ``Var = (days / n) [2 - 2 rho + (d1^2 + d2^2 - 2 d1 d2 rho^2) / 2]`` with daily Sharpe ``d = SR / sqrt(days)``
    and ``n = days x years`` observations.
    """
    d1, d2 = sr1 / np.sqrt(days), sr2 / np.sqrt(days)
    n = days * years
    return float(np.sqrt(days / n * (2.0 - 2.0 * rho + HALF * (d1**2 + d2**2 - 2.0 * d1 * d2 * rho**2))))


def smallest_detectable_gap(se: float, level: float) -> float:
    """The gap a two-sided interval of this level can just exclude zero for: ``z x se``."""
    return float(norm.ppf(HALF + level / 2.0) * se)


def giacomini_white(loss_difference: np.ndarray, instruments: np.ndarray, lag: int) -> tuple[float, float]:
    """Giacomini-White (2006) test of equal conditional predictive ability.

    ``loss_difference[t] = L(forecast A) - L(forecast B)``; ``instruments[t]`` is the information known before ``t`` (a constant
    plus the conditioning variables). Wald statistic of ``E[h_t d_t] = 0`` with a Newey-West covariance, chi-square with
    ``q`` degrees of freedom. Returns ``(statistic, p_value)``; a small p says the loss gap depends on the instruments."""
    d = np.asarray(loss_difference, dtype=float)
    h = np.asarray(instruments, dtype=float).reshape(len(d), -1)
    moment = h * d[:, None]
    n, q = moment.shape
    mean = moment.mean(axis=0)
    centred = moment - mean
    cov = centred.T @ centred / n
    for k in range(1, min(lag, n - 1) + 1):
        gamma = centred[k:].T @ centred[:-k] / n
        cov += (1.0 - k / (lag + 1.0)) * (gamma + gamma.T)
    stat = float(n * mean @ np.linalg.pinv(cov) @ mean)
    return stat, float(chi2.sf(stat, q))


def encompassing_test(error_a: np.ndarray, error_b: np.ndarray, lag: int, horizon: int = 1) -> tuple[float, float]:
    """Harvey-Leybourne-Newbold (1998) forecast-encompassing test that forecast A encompasses forecast B.

    Statistic ``mean(e_A (e_A - e_B))`` with a Newey-West variance and the small-sample correction; one-sided p-value for
    "B adds information". Returns ``(statistic, p_value)``."""
    e_a, e_b = np.asarray(error_a, dtype=float), np.asarray(error_b, dtype=float)
    d = e_a * (e_a - e_b)
    n = len(d)
    variance = newey_west_variance(d, lag) / n
    if variance <= 0:
        return float("nan"), float("nan")
    correction = np.sqrt((n + 1 - 2 * horizon + horizon * (horizon - 1) / n) / n)
    stat = float(correction * d.mean() / np.sqrt(variance))
    return stat, float(student_t.sf(stat, n - 1))


def agreement_table(road1: np.ndarray, road2_only: np.ndarray, merged: np.ndarray, realised: np.ndarray, floor: float,
                    bins: int) -> list[dict]:
    """Results by agreement level: terciles (``bins``) of the relative gap between the roads, lowest gap first.

    Per level: rows, mean relative gap, mean |realised|, correlation of each forecast with the realised return and the
    mean squared error of each. Disagreement is normalised by signal size so strong news is not penalised for being strong."""
    gap = np.abs(road1 - road2_only) / (np.maximum(np.abs(road1), np.abs(road2_only)) + floor)
    edges = np.quantile(gap, np.linspace(0.0, 1.0, bins + 1))
    level = np.clip(np.searchsorted(edges[1:-1], gap, side="right"), 0, bins - 1)
    rows = []
    for k in range(bins):
        pick = level == k
        if pick.sum() < 2:
            continue
        row = {"level": k + 1, "n": int(pick.sum()), "mean_gap": float(gap[pick].mean()), "mean_abs_realised": float(np.abs(realised[pick]).mean())}
        for name, forecast in (("road1", road1), ("road2", road2_only), ("merged", merged)):
            f, y = forecast[pick], realised[pick]
            row[f"corr_{name}"] = float(np.corrcoef(f, y)[0, 1]) if f.std() > 0 and y.std() > 0 else float("nan")
            row[f"mse_{name}"] = float(np.mean((y - f) ** 2))
        rows.append(row)
    return rows
