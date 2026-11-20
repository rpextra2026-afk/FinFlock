"""The dynamic path, the anchor to the observed day-0 move, and the expected path (7.6c, d, e)."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from oryxis.common.config import Config
from oryxis.mfg.precompute import MonthGame
from oryxis.mfg.push import PushResult, end_point
from oryxis.mfg.speed import EventKnobs


def price_path(
    push: PushResult, phi: np.ndarray, delta0: np.ndarray, horizon: int
) -> np.ndarray:
    """Prices for days ``0 .. horizon``: ``p_t = delta0 + sum_k (1 - phi_k^(t+1)) pi_k`` (D14b timing).

    First order in impact. Row ``t`` is day ``t``; the day before the event is ``delta0``. Because the
    types' pushes sum to ``y*`` the path ends at the static end point as ``t`` grows.
    """
    days = np.arange(horizon + 1)
    weights = 1.0 - phi[None, :] ** (days[:, None] + 1)  # (T, K)
    return delta0[None, :] + weights @ push.pi


def expected_path(path: np.ndarray, consensus: np.ndarray, varsigma: float) -> np.ndarray:
    """``E[p_t] = delta0_1 + (1 - varsigma)^(t+1) (p_t - delta0_1)``: the news may resolve (7.6e)."""
    decay = (1.0 - varsigma) ** (np.arange(path.shape[0]) + 1)
    return consensus[None, :] + decay[:, None] * (path - consensus[None, :])


def predicted_increments(expected: np.ndarray) -> np.ndarray:
    """Tradable increments ``m_t = E[p_t] - E[p_{t-1}]`` for ``t >= 1`` (row 0 is day 1)."""
    return np.diff(expected, axis=0)


@dataclass(frozen=True)
class Anchor:
    tau: np.ndarray  # shift of the outside belief at the pinned assets
    weak_id: np.ndarray  # True where the pinned price barely reacts to its own shift
    clipped: np.ndarray


def anchor_outside_belief(
    day0_pinned: Callable[[np.ndarray], np.ndarray],
    observed: np.ndarray,
    cfg: Config,
) -> Anchor:
    """Back out the outside belief shift from the observed day-0 move (Prop. 14.6).

    ``day0_pinned(tau)`` returns the day-0 prices of the pinned assets after shifting the outside
    belief of asset ``i`` by ``tau_i``. It is affine in ``tau``, so two evaluations per source give the
    map; the ``|I| x |I|`` system is solved with a small ridge. Sources whose own sensitivity is
    below ``weak_id_ratio * |s_i|`` get ``tau_i = 0`` and a ``weak_id`` flag; ``|tau|`` is clipped.
    """
    ridge = float(cfg.get("game.anchor.ridge"))
    weak_ratio = float(cfg.get("game.anchor.weak_id_ratio"))
    clip = float(cfg.get("game.anchor.tau_clip"))
    size = len(observed)
    base = day0_pinned(np.zeros(size))
    sensitivity = np.column_stack([day0_pinned(np.eye(size)[j]) - base for j in range(size)])
    weak = np.diag(sensitivity) < weak_ratio * np.abs(observed)
    tau = np.zeros(size)
    active = np.flatnonzero(~weak)
    if len(active):
        block = sensitivity[np.ix_(active, active)] + ridge * np.eye(len(active))
        tau[active] = np.linalg.solve(block, (observed - base)[active])
    clipped = np.abs(tau) > clip
    return Anchor(np.clip(tau, -clip, clip), weak, clipped)


def anchored_path(
    game: MonthGame,
    knobs: EventKnobs,
    deltas: np.ndarray,
    delta0: np.ndarray,
    pinned: np.ndarray,
    observed: np.ndarray,
    cfg: Config,
) -> tuple[np.ndarray, Anchor, PushResult]:
    """Path whose pinned day-0 prices match the observed moves, via the outside-belief shift."""
    horizon = int(cfg.get("game.horizon_days"))

    def shifted(tau: np.ndarray) -> np.ndarray:
        d0 = delta0.copy()
        d0[pinned] += tau
        push = end_point(game, knobs, deltas, d0)
        return price_path(push, knobs.phi, d0, horizon)[0][pinned]

    anchor = anchor_outside_belief(shifted, observed, cfg)
    d0 = delta0.copy()
    d0[pinned] += anchor.tau
    push = end_point(game, knobs, deltas, d0)
    return price_path(push, knobs.phi, d0, horizon), anchor, push
