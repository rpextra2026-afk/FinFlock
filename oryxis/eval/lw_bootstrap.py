"""Ledoit-Wolf (2008) studentised circular block bootstrap for a difference of Sharpe ratios.

Statistic: ``Delta = SR_1 - SR_2`` on per-period returns. With ``y_t = (r1, r2, r1^2, r2^2)`` and
``v = E y``, ``Delta = f(v)`` and the standard error is ``s = sqrt(grad f' Psi grad f / T)`` with ``Psi`` the
Bartlett HAC covariance of ``y``. The bootstrap resamples the PAIRS of returns in circular blocks and
studentises each draw with its own standard error: ``d* = |Delta* - Delta| / s*``. The two-sided p-value is the
share of draws with ``d* >= |Delta| / s``, and a confidence interval is ``Delta +/- s q`` with ``q`` the
quantile of ``d*``. [VERIFY the interval construction against the paper before the freeze.]
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from oryxis.eval.stats import block_length, circular_block_indices

CUBE = 3  # exponent in the gradient of mu / sqrt(gamma - mu^2)

Y_COLUMNS = 4


@dataclass(frozen=True)
class SharpeDifference:
    delta: float  # per-period difference
    delta_annual: float
    se: float
    p_value: float
    ci_low: float
    ci_high: float
    block: int
    draws: int


def _features(r: np.ndarray) -> np.ndarray:
    return np.column_stack([r[..., 0], r[..., 1], r[..., 0] ** 2, r[..., 1] ** 2])


def _delta_and_gradient(v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``Delta`` and its gradient with respect to ``(mu1, mu2, gamma1, gamma2)`` for rows of ``v``."""
    mu1, mu2, g1, g2 = np.moveaxis(v, -1, 0)
    s1, s2 = np.sqrt(g1 - mu1**2), np.sqrt(g2 - mu2**2)
    delta = mu1 / s1 - mu2 / s2
    grad = np.stack([g1 / s1**CUBE, -g2 / s2**CUBE, -mu1 / (2.0 * s1**CUBE), mu2 / (2.0 * s2**CUBE)], axis=-1)
    return delta, grad


def _hac(y: np.ndarray, lag: int) -> np.ndarray:
    """Bartlett HAC covariance of the rows of ``y`` (axis 0 is time), batched over a leading axis if present."""
    y = y - y.mean(axis=-2, keepdims=True)
    n = y.shape[-2]
    psi = np.einsum("...ti,...tj->...ij", y, y) / n
    for k in range(1, lag + 1):
        gamma = np.einsum("...ti,...tj->...ij", y[..., k:, :], y[..., :-k, :]) / n
        psi = psi + (1.0 - k / (lag + 1.0)) * (gamma + np.swapaxes(gamma, -1, -2))
    return psi


def sharpe_difference_test(
    r1: np.ndarray, r2: np.ndarray, periods: int, draws: int, level: float, rng: np.random.Generator, block: int | None = None,
) -> SharpeDifference:
    """Test and interval for ``SR_1 - SR_2`` of two return series of equal length."""
    pairs = np.column_stack([np.asarray(r1, float), np.asarray(r2, float)])
    n = len(pairs)
    b = block or block_length(n)
    lag = b
    y = _features(pairs)
    delta, grad = _delta_and_gradient(y.mean(axis=0))
    se = float(np.sqrt(grad @ _hac(y, lag) @ grad / n))
    index = circular_block_indices(n, b, draws, rng)
    star = pairs[index]  # (draws, n, 2)
    y_star = np.stack([star[..., 0], star[..., 1], star[..., 0] ** 2, star[..., 1] ** 2], axis=-1)
    d_star, g_star = _delta_and_gradient(y_star.mean(axis=1))
    psi_star = _hac(y_star, lag)
    se_star = np.sqrt(np.einsum("bi,bij,bj->b", g_star, psi_star, g_star) / n)
    studentised = np.abs(d_star - delta) / se_star
    observed = abs(delta) / se
    p_value = float((1 + np.sum(studentised >= observed)) / (draws + 1))
    q = float(np.quantile(studentised, level))
    scale = np.sqrt(periods)
    return SharpeDifference(float(delta), float(delta * scale), se, p_value, float((delta - q * se) * scale),
                            float((delta + q * se) * scale), b, draws)
