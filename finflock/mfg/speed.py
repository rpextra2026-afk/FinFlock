"""Emotion, state and trait -> the game's knobs (spec 7.4, D18) and the trading speed (7.2)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finflock.common.config import Config
from finflock.mfg.precompute import MonthGame


DISCRIMINANT_FACTOR = 4.0  # b^2 - 4 a c of the quadratic a phi^2 - b phi + 1


@dataclass(frozen=True)
class EventKnobs:
    """Per-type knobs of one event (arrays have one entry per fund type)."""

    gamma: np.ndarray
    cost: np.ndarray
    omega: np.ndarray
    f: np.ndarray  # (K, N) fear term toward the minimum-variance portfolio, S&P block
    phi: np.ndarray


def stable_root(beta_k: np.ndarray, g: np.ndarray) -> np.ndarray:
    """The root in (0, 1) of ``beta phi^2 - (1 + beta + g) phi + 1 = 0``, ``g = gamma / cost``.

    This is the Garleanu-Pedersen trading-rate root: the share of the gap to the aim left untouched
    each day. It lies in (0, 1) for any ``beta in (0, 1]`` and ``g >= 0``.
    """
    b = 1.0 + beta_k + g
    return (b - np.sqrt(b * b - DISCRIMINANT_FACTOR * beta_k)) / (2.0 * beta_k)


def event_knobs(
    game: MonthGame,
    cfg: Config,
    *,
    fear_minus_anger: np.ndarray,
    arousal: np.ndarray,
    state_z: np.ndarray,
    trait_b: np.ndarray,
    mandate: np.ndarray | None = None,
) -> EventKnobs:
    """Risk aversion, urgency, belief sensitivity and trading speed of each type for one event.

    ``gamma = gamma_bar exp(m (kappa_f fa - kappa_s b z))``, ``cost = cost_bar exp(-m kappa_a ar)``,
    ``omega = rho / gamma`` and ``f = -(1 - gamma_bar / gamma)(x - x_mv)``. With all kappas zero this
    is the static panel.
    """
    types = game.types
    gamma_bar = np.array([t.gamma_bar for t in types])
    cost_bar = np.array([t.cost_bar for t in types])
    rho = np.array([t.rho for t in types])
    beta_k = np.array([t.beta_k for t in types])
    m = np.array([t.mandate for t in types]) if mandate is None else np.asarray(mandate)
    k_fear = float(cfg.get("game.kappa.fear"))
    k_arousal = float(cfg.get("game.kappa.arousal"))
    k_state = float(cfg.get("game.kappa.state"))
    gamma = gamma_bar * np.exp(m * (k_fear * fear_minus_anger - k_state * trait_b * state_z))
    cost = cost_bar * np.exp(-m * k_arousal * arousal)
    omega = float(cfg.get("game.omega_scale")) * rho / gamma
    tilt = np.stack([t.x_sp - t.x_mv_sp for t in types])
    f = -(1.0 - gamma_bar / gamma)[:, None] * tilt
    return EventKnobs(gamma, cost, omega, f, stable_root(beta_k, gamma / cost))
