"""Tension of a multi-pin reading (D9): the energy cost of the pinned values being jointly inconsistent.

``T(s) = 1/2 s' S s`` with the Schur complement ``S = L_PP - L_PF (L_FF + lambda I)^-1 L_FP``. For
single-pin news it is a fixed quadratic of the source dials; only the interaction part on multi-pin
news carries new information.
"""

from __future__ import annotations

import numpy as np

HALF = 0.5


def tension_matrix(L: np.ndarray, n_pinned: int, d: int, grounding: float) -> np.ndarray:
    p = n_pinned * d
    L_PP, L_PF, L_FP = L[:p, :p], L[:p, p:], L[p:, :p]
    L_FF = L[p:, p:] + grounding * np.eye(L.shape[0] - p)
    if L_FF.shape[0] == 0:
        return L_PP
    return L_PP - L_PF @ np.linalg.solve(L_FF, L_FP)


def tension(L: np.ndarray, s: np.ndarray, grounding: float) -> float:
    """``1/2 s' S s`` for pinned values ``s`` of shape ``(n_pinned, d)``."""
    flat = s.reshape(-1)
    S = tension_matrix(L, s.shape[0], s.shape[1], grounding)
    return float(HALF * flat @ S @ flat)
