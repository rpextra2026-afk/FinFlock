"""The push: unconditional end point and the rule conditional on the observed day-0 move (7.6a, b)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from finfm.mfg.precompute import MonthGame
from finfm.mfg.speed import EventKnobs


@dataclass(frozen=True)
class PushResult:
    """Static end point of one event for all assets of the month."""

    y: np.ndarray  # price change from the outside belief, y* = G c
    p: np.ndarray  # end point, p* = delta0 + y*
    c: np.ndarray  # forcing term
    q: np.ndarray  # (K, N) steady-state trade of each type, in weights
    pi: np.ndarray  # (K, N) price push of each type; sums over types to y*
    sleeve_flow: float  # dollars moved into (+) or out of (-) the funds' non-S&P sleeves


def end_point(
    game: MonthGame, knobs: EventKnobs, deltas: np.ndarray, delta0: np.ndarray
) -> PushResult:
    """End point of the static game (D12/D16).

    ``deltas`` is ``(K, N)`` (belief of each type), ``delta0`` is ``(N,)`` (outside belief). The
    forcing is ``c = sum_k a_k omega_k S_k d_k - R delta0 + sum_k a_k f_k`` with ``d_k = delta_k -
    delta0``. The steady-state trade uses the baseline ``omega_bar`` on the price term so that the
    types' pushes sum to ``y*`` exactly (market clearing); with ``omega = omega_bar`` it is the
    textbook ``omega S (delta - p*) + f - theta R p*``.
    """
    types = game.types
    book = np.array([t.book for t in types])
    theta = np.array([t.theta for t in types])
    omega_bar = np.array([t.omega_bar for t in types])
    S = game.S_stack()
    R = game.R_stack()
    d = deltas - delta0[None, :]
    forcing = (
        np.einsum("k,kij,kj->i", book * knobs.omega, S, d)
        - game.R_sum @ delta0
        + (book[:, None] * knobs.f).sum(axis=0)
    )
    y = game.G @ forcing
    p = delta0 + y
    q = (
        knobs.omega[:, None] * np.einsum("kij,kj->ki", S, d)
        - omega_bar[:, None] * np.einsum("kij,j->ki", S, y)
        + knobs.f
        - theta[:, None] * np.einsum("kij,j->ki", R, p)
    )
    pi = (book[:, None] * q) / game.ell[None, :]
    sleeve_flow = 0.0
    if game.sleeve:  # sleeve price change and belief are 0, so only the S&P parts enter
        for k, fund in enumerate(types):
            q_sleeve = (
                knobs.omega[k] * fund.S_sleeve @ d[k]
                - fund.omega_bar * fund.S_sleeve @ y
                - fund.theta * fund.R_sleeve @ p
                - (1.0 - fund.gamma_bar / knobs.gamma[k]) * (fund.x_sleeve - fund.x_mv_sleeve)
            )
            sleeve_flow += fund.book * q_sleeve
    return PushResult(y=y, p=p, c=forcing, q=q, pi=pi, sleeve_flow=sleeve_flow)


def conditional_prices(
    game: MonthGame,
    forcing: np.ndarray,
    delta0: np.ndarray,
    pinned: np.ndarray,
    observed: np.ndarray,
) -> np.ndarray:
    """Prices of the free assets when the pinned ones are known (D12 Prop. 12.7, general form).

    ``p_F = delta0_F + (G_FF - G_FI G_II^-1 G_IF) c_F + G_FI G_II^-1 (s_I - delta0_I)``; pinned
    assets are set to their observed value ``s_I``.
    """
    n = game.n_assets
    free = np.setdiff1d(np.arange(n), pinned)
    G = game.G
    G_FF, G_FI = G[np.ix_(free, free)], G[np.ix_(free, pinned)]
    G_II, G_IF = G[np.ix_(pinned, pinned)], G[np.ix_(pinned, free)]
    solve = np.linalg.solve(G_II, np.column_stack([G_IF, observed - delta0[pinned]]))
    block_inverse = G_FF - G_FI @ solve[:, : len(free)]
    out = np.empty(n)
    out[pinned] = observed
    out[free] = delta0[free] + block_inverse @ forcing[free] + G_FI @ solve[:, -1]
    return out


def static_push_regressors(
    game: MonthGame, source: int, observed_move: float, type_belief: float
) -> tuple[np.ndarray, np.ndarray]:
    """Regressors of the static-push test for one pinned source (D12 test design).

    ``Z_reb = A_FF^-1 |R_FI| s`` (rebalancing: neighbours move WITH the source) and ``Z_sub =
    -A_FF^-1 |M_FI| (delta_hat - s)`` (substitution: neighbours move against under- or over-shoot).
    Returned over all assets except the source, in index order.
    """
    n = game.n_assets
    free = np.setdiff1d(np.arange(n), [source])
    G = game.G
    A_FF_inv = G[np.ix_(free, free)] - np.outer(G[free, source], G[source, free]) / G[source, source]
    reb = A_FF_inv @ np.abs(game.R_sum[free, source]) * observed_move
    sub = -A_FF_inv @ np.abs(game.M_bar[free, source]) * (type_belief - observed_move)
    return reb, sub
