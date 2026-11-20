"""The one fixed optimiser: multi-period mean-variance with quadratic costs (D21) and the band (R-8).

Maximise ``sum_t beta^t [n_t' mu_t - (gamma/2) n_t' Sigma_t n_t - 1/2 (n_t - n_{t-1})' Lambda_t (n_t - n_{t-1})]``.
Backward recursion: ``M_t = gamma Sigma_t + Lambda_t + beta A_{t+1}``, ``A_t = Lambda_t - Lambda_t M_t^-1 Lambda_t``,
``b_t = Lambda_t M_t^-1 (mu_t + beta b_{t+1})``, trade rule ``n_t = M_t^-1 (Lambda_t n_{t-1} + mu_t + beta b_{t+1})``.

The buy/hold band is the exact no-trade rule for a linear cost (D21 Prop. 21.2), applied asset by
asset: an asset keeps its weight while its aim is within ``h kappa_j / H_jj`` of it, otherwise its
trade is shortened by ``h kappa_j / (H_jj + Lambda_jj)``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from scipy.linalg import solve_discrete_are


@dataclass(frozen=True)
class HorizonInputs:
    """Forecast mean, covariance and cost matrix for each of the next ``T`` periods."""

    mu: np.ndarray  # (T, N)
    sigma: np.ndarray  # (T, N, N)
    lam: np.ndarray  # (T, N, N)


@dataclass(frozen=True)
class FirstStep:
    n_free: np.ndarray  # unconstrained D21 target for the first period
    aim: np.ndarray  # H^-1 (mu_0 + beta b_1): where the asset would sit without trading costs
    H: np.ndarray  # gamma Sigma_0 + beta A_1
    lam0: np.ndarray


def _continuation(sigma: np.ndarray, lam: np.ndarray, gamma: float, beta: float, A: np.ndarray) -> np.ndarray:
    return lam - lam @ np.linalg.solve(gamma * sigma + lam + beta * A, lam)


def stationary_value(
    sigma: np.ndarray, lam: np.ndarray, gamma: float, beta: float, tol: float, max_iter: int
) -> np.ndarray:
    """Fixed point ``A = Lambda - Lambda (gamma Sigma + Lambda + beta A)^-1 Lambda`` (calm continuation).

    This is a discrete algebraic Riccati equation (state: last holdings, control: the trade, stage cost
    ``u' Lambda u + (x + u)' gamma Sigma (x + u)``, discount ``beta``), solved directly. Plain iteration needs
    tens of thousands of steps when the cost matrix dominates the risk term (the real case), so it is kept only
    as the fallback when the direct solve is not accurate to ``tol`` relative to the size of ``A``.
    """
    size = sigma.shape[0]
    eye, root = np.eye(size), np.sqrt(beta)
    risk = gamma * sigma
    try:
        A = solve_discrete_are(root * eye, root * eye, risk, lam + risk, s=risk)
    except (np.linalg.LinAlgError, ValueError):
        A = np.zeros_like(sigma)
    for _ in range(max_iter):
        updated = _continuation(sigma, lam, gamma, beta, A)
        if np.abs(updated - A).max() <= tol * max(np.abs(updated).max(), 1.0):
            return updated
        A = updated
    raise RuntimeError("stationary continuation value did not converge")


def backward_pass(
    inputs: HorizonInputs, gamma: float, beta: float, A_terminal: np.ndarray, b_terminal: np.ndarray
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """Value-function matrices ``M_t``, ``A_{t+1}`` and ``b_{t+1}`` for every period ``t``."""
    horizon = inputs.mu.shape[0]
    A, b = A_terminal, b_terminal
    Ms: list[np.ndarray] = [None] * horizon  # type: ignore[list-item]
    A_next: list[np.ndarray] = [None] * horizon  # type: ignore[list-item]
    b_next: list[np.ndarray] = [None] * horizon  # type: ignore[list-item]
    for t in reversed(range(horizon)):
        lam = inputs.lam[t]
        M = gamma * inputs.sigma[t] + lam + beta * A
        Ms[t], A_next[t], b_next[t] = M, A, b
        rhs = inputs.mu[t] + beta * b
        b = lam @ np.linalg.solve(M, rhs)
        A = lam - lam @ np.linalg.solve(M, lam)
    return Ms, A_next, b_next


def plan_path(
    n_prev: np.ndarray, inputs: HorizonInputs, gamma: float, beta: float,
    A_terminal: np.ndarray, b_terminal: np.ndarray,
) -> np.ndarray:
    """Full optimal path ``n_0 .. n_{T-1}`` from the starting weights ``n_prev``."""
    Ms, _, b_next = backward_pass(inputs, gamma, beta, A_terminal, b_terminal)
    path, n = [], n_prev
    for t in range(inputs.mu.shape[0]):
        n = np.linalg.solve(Ms[t], inputs.lam[t] @ n + inputs.mu[t] + beta * b_next[t])
        path.append(n)
    return np.array(path)


def first_step(
    n_prev: np.ndarray, inputs: HorizonInputs, gamma: float, beta: float,
    A_terminal: np.ndarray, b_terminal: np.ndarray,
) -> FirstStep:
    """First-period target of the receding-horizon rule and the quantities the band needs."""
    Ms, A_next, b_next = backward_pass(inputs, gamma, beta, A_terminal, b_terminal)
    lam0 = inputs.lam[0]
    n_free = np.linalg.solve(Ms[0], lam0 @ n_prev + inputs.mu[0] + beta * b_next[0])
    H = gamma * inputs.sigma[0] + beta * A_next[0]
    aim = np.linalg.solve(H, inputs.mu[0] + beta * b_next[0])
    return FirstStep(n_free, aim, H, lam0)


def apply_band(
    n_prev: np.ndarray, step: FirstStep, one_way_cost: np.ndarray, band: float
) -> np.ndarray:
    """Band on the first-period trade; ``band = 0`` returns the unconstrained D21 target."""
    trade = step.n_free - n_prev
    if band == 0.0:
        return step.n_free
    diag_h = np.diag(step.H)
    width = band * one_way_cost / diag_h
    inside = np.abs(step.aim - n_prev) <= width
    shortened = np.maximum(np.abs(trade) - band * one_way_cost / (diag_h + np.diag(step.lam0)), 0.0)
    out_trade = np.sign(trade) * shortened
    return n_prev + np.where(inside, 0.0, out_trade)


def constant_inputs(
    mu_path: Sequence[np.ndarray], sigma: np.ndarray, lam: np.ndarray
) -> HorizonInputs:
    """Inputs with the same covariance and cost matrices in every period."""
    mu = np.array(mu_path)
    return HorizonInputs(mu, np.broadcast_to(sigma, (len(mu), *sigma.shape)).copy(),
                         np.broadcast_to(lam, (len(mu), *lam.shape)).copy())
