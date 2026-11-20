"""Exact solution of the dynamic game (D14b) used to validate the fast path (spec 7.8 #1).

The stacked finite-horizon block system is solved exactly on a small market. ``market`` is a plain
dict so tests can build markets without any data: keys ``N, K, Sig, Us, x, a, gam, rho, Lam,
vth, theta, beta, ell`` (``Lam`` is the list of per-type trading-cost matrices).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import scipy.linalg as sla
import scipy.sparse as sp
import scipy.sparse.linalg as spl

from oryxis.mfg.precompute import rebalancing_matrix


BLOCK_COUNT = 3  # A0, A1, A2 of the Euler system


def feasible_basis(universe: Sequence[int], n: int) -> np.ndarray:
    """Orthonormal basis of trades supported on ``universe`` that sum to zero."""
    embed = np.zeros((n, len(universe)))
    embed[list(universe), range(len(universe))] = 1.0
    return embed @ sla.null_space(np.ones((1, len(universe))))


def euler_system(market: Mapping[str, Any], psi_scale: float) -> dict[str, Any]:
    """Coefficient blocks ``A0 Z_{t-1} + A1 Z_t + A2 Z_{t+1} = b`` of the dynamic game."""
    n, k_types = market["N"], market["K"]
    bases = [feasible_basis(u, n) for u in market["Us"]]
    dims = [b.shape[1] for b in bases]
    offsets = np.cumsum([0, *dims])
    total = sum(dims)
    psi = np.diag(1.0 / market["ell"]) * psi_scale
    aggregate = np.hstack([market["a"][k] * bases[k] for k in range(k_types)])
    A0, A1, A2 = (np.zeros((total, total)) for _ in range(BLOCK_COUNT))
    for k in range(k_types):
        block = slice(offsets[k], offsets[k + 1])
        V = bases[k]
        rho, gamma, theta = market["rho"][k], market["gam"][k], market["theta"][k]
        vartheta = (1 - rho) * market["vth"][k]
        beta = market["beta"] * (1 - rho)
        cost = V.T @ market["Lam"][k] @ V
        rebalance = rebalancing_matrix(market["x"][k])
        A0[block, block] += -cost
        A1[block, block] += (1 + beta) * cost + gamma * V.T @ market["Sig"] @ V
        A1[block, :] += (
            V.T
            @ ((rho + vartheta) * np.eye(n) + theta * gamma * market["Sig"] @ rebalance)
            @ psi
            @ aggregate
        )
        A2[block, block] += -beta * cost
        A2[block, :] += -vartheta * V.T @ psi @ aggregate
    return {
        "bases": bases, "offsets": offsets, "dim": total, "psi": psi, "aggregate": aggregate,
        "A0": A0, "A1": A1, "A2": A2,
    }


def forcing(market: Mapping[str, Any], system: Mapping[str, Any], deltas, delta0) -> np.ndarray:
    b = np.zeros(system["dim"])
    for k in range(market["K"]):
        block = slice(system["offsets"][k], system["offsets"][k + 1])
        b[block] = system["bases"][k].T @ (
            market["rho"][k] * (deltas[k] - delta0)
            - market["theta"][k]
            * market["gam"][k]
            * market["Sig"]
            @ rebalancing_matrix(market["x"][k])
            @ delta0
        )
    return b


def exact_prices(
    market: Mapping[str, Any], deltas, delta0, horizon: int, psi_scale: float = 1.0
) -> np.ndarray:
    """Prices for days ``0 .. horizon - 1`` from the stacked solve with terminal ``Z_T = Z*``."""
    system = euler_system(market, psi_scale)
    D = system["dim"]
    b = forcing(market, system, deltas, delta0)
    steady = np.linalg.solve(system["A0"] + system["A1"] + system["A2"], b)
    matrix = sp.lil_matrix((horizon * D, horizon * D))
    rhs = np.tile(b, horizon)
    for t in range(horizon):
        rows = slice(t * D, (t + 1) * D)
        matrix[rows, rows] = system["A1"]
        if t > 0:
            matrix[rows, slice((t - 1) * D, t * D)] = system["A0"]
        if t < horizon - 1:
            matrix[rows, slice((t + 1) * D, (t + 2) * D)] = system["A2"]
        else:
            rhs[rows] = rhs[rows] - system["A2"] @ steady
    Z = spl.spsolve(matrix.tocsc(), rhs).reshape(horizon, D)
    return delta0[None, :] + (system["psi"] @ system["aggregate"] @ Z.T).T
