"""Deflated Sharpe ratio, probability of backtest overfitting (CSCV) and combinatorial purged CV splits.

Bailey and Lopez de Prado (2014) for the deflated Sharpe; Bailey, Borwein, Lopez de Prado and Zhu (2017)
for CSCV. The number of trials is the configuration count printed beside every headline.
"""

from __future__ import annotations

from itertools import combinations

import numpy as np
from scipy.stats import kurtosis, norm, skew

VARIANCE_FLOOR = 1e-12  # keeps the DSR denominator and trial variances positive

QUARTER = 0.25


def expected_max_sharpe(variance_of_sharpes: float, trials: int, euler_gamma: float) -> float:
    """``SR_0``: the best per-period Sharpe expected from ``trials`` skill-free configurations."""
    if trials < 2:
        return 0.0
    return float(np.sqrt(variance_of_sharpes) * ((1.0 - euler_gamma) * norm.ppf(1.0 - 1.0 / trials)
                                                  + euler_gamma * norm.ppf(1.0 - 1.0 / (trials * np.e))))


def deflated_sharpe(returns: np.ndarray, trial_sharpes: np.ndarray, euler_gamma: float) -> dict[str, float]:
    """Probability that the true Sharpe exceeds the best expected from skill-free trials (per-period units)."""
    n = len(returns)
    sr = float(returns.mean() / returns.std(ddof=1))
    sr0 = expected_max_sharpe(float(np.var(trial_sharpes, ddof=1)) if len(trial_sharpes) > 1 else 0.0,
                              len(trial_sharpes), euler_gamma)
    g3, g4 = float(skew(returns)), float(kurtosis(returns, fisher=False))
    denominator = np.sqrt(max(1.0 - g3 * sr + (g4 - 1.0) * QUARTER * sr**2, VARIANCE_FLOOR))
    return {"sharpe": sr, "sr0": sr0, "dsr": float(norm.cdf((sr - sr0) * np.sqrt(n - 1) / denominator)), "trials": len(trial_sharpes)}


def _block_sums(matrix: np.ndarray, partitions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per partition: count, sum and sum of squares of each configuration's returns."""
    blocks = np.array_split(np.arange(matrix.shape[0]), partitions)
    counts = np.array([len(b) for b in blocks], dtype=float)
    sums = np.stack([matrix[b].sum(axis=0) for b in blocks])
    squares = np.stack([(matrix[b] ** 2).sum(axis=0) for b in blocks])
    return counts, sums, squares


def _sharpes(counts, sums, squares, chosen: list[int]) -> np.ndarray:
    n = counts[chosen].sum()
    mean = sums[chosen].sum(axis=0) / n
    var = (squares[chosen].sum(axis=0) / n - mean**2) * n / (n - 1)
    return mean / np.sqrt(np.maximum(var, VARIANCE_FLOOR**2))


def probability_of_backtest_overfitting(matrix: np.ndarray, partitions: int) -> dict[str, float]:
    """CSCV: ``matrix`` is time x configurations. PBO is the share of splits where the in-sample best
    configuration ranks at or below the out-of-sample median."""
    counts, sums, squares = _block_sums(matrix, partitions)
    everything = list(range(partitions))
    logits = []
    n_configs = matrix.shape[1]
    for in_sample in combinations(everything, partitions // 2):
        out_sample = [p for p in everything if p not in in_sample]
        best = int(np.argmax(_sharpes(counts, sums, squares, list(in_sample))))
        oos = _sharpes(counts, sums, squares, out_sample)
        rank = float((oos < oos[best]).sum() + 1) / (n_configs + 1)
        logits.append(np.log(rank / (1.0 - rank)))
    logits = np.array(logits)
    return {"pbo": float(np.mean(logits <= 0.0)), "n_splits": len(logits), "median_logit": float(np.median(logits))}


def cpcv_splits(n: int, groups: int, test_groups: int, embargo: int):
    """Combinatorial purged CV: every choice of ``test_groups`` groups is a test set; training drops the
    ``embargo`` observations on each side of every test group (purging for overlapping labels)."""
    bounds = np.array_split(np.arange(n), groups)
    for chosen in combinations(range(groups), test_groups):
        test = np.concatenate([bounds[g] for g in chosen])
        banned = np.zeros(n, dtype=bool)
        for g in chosen:
            lo, hi = bounds[g][0], bounds[g][-1]
            banned[max(lo - embargo, 0) : min(hi + embargo + 1, n)] = True
        yield np.flatnonzero(~banned), test
