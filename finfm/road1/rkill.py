"""The sheaf regime ``r`` (D10 Prop. 10.2): how strongly neighbours interact relative to grounding.

Split ``L_FF + lambda I = Delta - W`` with ``Delta`` the block diagonal (``d x d`` blocks) and ``W``
the off-diagonal remainder; ``r = rho(Delta^-1/2 W Delta^-1/2)``. It is reported as a distribution
(median, 90th percentile) and the bound ``r / (1 - r)``; it is a measurement, not a gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RStats:
    r: float  # spectral radius
    largest: float  # largest eigenvalue of Delta^-1/2 W Delta^-1/2
    smallest: float
    positive_definite: bool


def free_system(L: np.ndarray, n_pinned: int, d: int, grounding: float) -> np.ndarray:
    """``L_FF + lambda I`` from a full Laplacian whose pinned nodes come first."""
    p = n_pinned * d
    block = L[p:, p:]
    return block + grounding * np.eye(block.shape[0])


def inverse_sqrt_blocks(system: np.ndarray, d: int) -> np.ndarray:
    """Block-diagonal ``Delta^-1/2`` of the diagonal ``d x d`` blocks of ``system``."""
    n = system.shape[0] // d
    out = np.zeros_like(system)
    for i in range(n):
        block = system[i * d : (i + 1) * d, i * d : (i + 1) * d]
        values, vectors = np.linalg.eigh(block)
        out[i * d : (i + 1) * d, i * d : (i + 1) * d] = (vectors / np.sqrt(values)) @ vectors.T
    return out


def r_value(system: np.ndarray, d: int) -> RStats:
    """Spectral radius of ``Delta^-1/2 W Delta^-1/2`` for ``system = Delta - W``."""
    if system.shape[0] == 0:
        return RStats(0.0, 0.0, 0.0, True)
    n = system.shape[0] // d
    delta = np.zeros_like(system)
    for i in range(n):
        delta[i * d : (i + 1) * d, i * d : (i + 1) * d] = system[i * d : (i + 1) * d, i * d : (i + 1) * d]
    root = inverse_sqrt_blocks(system, d)
    normalised = root @ (delta - system) @ root
    eig = np.linalg.eigvalsh((normalised + normalised.T) / 2.0)
    return RStats(float(np.abs(eig).max()), float(eig.max()), float(eig.min()), bool(eig.max() < 1.0))


def r_bound(r: float) -> float:
    """``r / (1 - r)``, infinite when ``r >= 1``."""
    return float(r / (1.0 - r)) if r < 1.0 else float("inf")
