"""The sheaf test (D10, spec 8.4): does the sheaf add anything beyond the best table?

``y_j = alpha + beta1 * yhat_table + beta2 * (yhat_sheaf - yhat_table) + eps`` over free-node dial
labels. The sheaf survives iff ``beta2 > 0`` with a date-clustered interval above 0. Intervals come
from a cluster bootstrap over event dates; a CR1 cluster-robust standard error is reported too.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


N_REGRESSORS = 3  # constant, table forecast, sheaf-minus-table difference
SHEAF_INDEX = 2
TABLE_INDEX = 1


@dataclass(frozen=True)
class D10Result:
    beta1: float
    beta2: float
    se_cluster: float
    ci_low: float
    ci_high: float
    p_two_sided: float
    n_obs: int
    n_clusters: int


def d10_regression(
    y: np.ndarray, y_table: np.ndarray, y_sheaf: np.ndarray, cluster: np.ndarray,
    rng: np.random.Generator, draws: int, level: float,
) -> D10Result:
    """OLS of the label on the table forecast and the sheaf-minus-table difference."""
    X = np.column_stack([np.ones_like(y), y_table, y_sheaf - y_table])
    labels, inverse = np.unique(cluster, return_inverse=True)
    g = len(labels)
    xtx = np.zeros((g, N_REGRESSORS, N_REGRESSORS))
    xty = np.zeros((g, N_REGRESSORS))
    np.add.at(xtx, inverse, X[:, :, None] * X[:, None, :])
    np.add.at(xty, inverse, X * y[:, None])
    total_xtx, total_xty = xtx.sum(axis=0), xty.sum(axis=0)
    beta = np.linalg.solve(total_xtx, total_xty)

    resid = y - X @ beta
    scores = np.zeros((g, N_REGRESSORS))
    np.add.at(scores, inverse, X * resid[:, None])
    bread = np.linalg.inv(total_xtx)
    n, k = len(y), N_REGRESSORS
    meat = scores.T @ scores
    cov = bread @ meat @ bread * (g / (g - 1)) * ((n - 1) / (n - k))
    counts = rng.multinomial(g, np.full(g, 1.0 / g), size=draws)
    boot_xtx = np.einsum("bg,gij->bij", counts, xtx)
    boot_xty = counts @ xty
    boot_beta = np.linalg.solve(boot_xtx, boot_xty[:, :, None])[:, :, 0]
    b2 = boot_beta[:, SHEAF_INDEX]
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(b2, [tail, 1.0 - tail])
    p = 2.0 * min(np.mean(b2 <= 0.0), np.mean(b2 >= 0.0))
    return D10Result(float(beta[TABLE_INDEX]), float(beta[SHEAF_INDEX]), float(np.sqrt(cov[SHEAF_INDEX, SHEAF_INDEX])), float(low),
                     float(high), float(min(p, 1.0)), n, g)
