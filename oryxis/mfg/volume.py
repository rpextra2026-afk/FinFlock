"""Road-2 volume and covariance shift (D17, D15; spec 7.7)."""

from __future__ import annotations

import numpy as np

from oryxis.mfg.precompute import MonthGame
from oryxis.mfg.push import PushResult


HALF = 0.5


def fund_trades(push: PushResult, phi: np.ndarray, horizon: int) -> np.ndarray:
    """Daily trades in weights, shape ``(K, T, N)``: ``u_kt = phi_k^t (1 - phi_k) q_k``."""
    days = np.arange(horizon + 1)
    decay = phi[:, None] ** days[None, :] * (1.0 - phi[:, None])  # (K, T)
    return decay[:, :, None] * push.q[:, None, :]


def event_volume(
    game: MonthGame,
    push: PushResult,
    phi: np.ndarray,
    path: np.ndarray,
    delta0: np.ndarray,
    baseline_dollar_volume: np.ndarray,
    window_days: int,
) -> dict[str, np.ndarray]:
    """Dollar volume from road 2 and its one-sided share.

    ``Vol_jt = (sum_k a_k |u_kjt| + ell_j |p_jt - p_j,t-1|) / 2`` (buys equal sells, so volume is the
    mean of gross fund flow and the net absorbed by outside liquidity). Returns the volume dial
    ``log(1 + sum_{t<window} Vol_t / (window * DV))`` and the one-sided share ``net / (gross + net)``.
    """
    book = np.array([t.book for t in game.types])
    u = fund_trades(push, phi, path.shape[0] - 1)  # (K, T, N)
    gross = np.einsum("k,ktn->tn", book, np.abs(u))
    previous = np.vstack([delta0[None, :], path[:-1]])
    net = game.ell[None, :] * np.abs(path - previous)
    volume = HALF * (gross + net)
    head = slice(0, window_days)
    dial = np.log1p(volume[head].sum(axis=0) / (window_days * baseline_dollar_volume))
    gross_sum, net_sum = gross[head].sum(axis=0), net[head].sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        one_sided = np.where(gross_sum + net_sum > 0, net_sum / (gross_sum + net_sum), 0.0)
    return {"volume": volume, "volume_dial": dial, "one_sided_share": one_sided}


class LowRankCov:
    """``centred' centred / horizon``, kept as the ``(H, n)`` factor instead of the dense ``(n, n)`` matrix (about 25x less memory per event)."""

    __slots__ = ("centred", "horizon")

    def __init__(self, centred: np.ndarray, horizon: int):
        self.centred, self.horizon = centred, horizon

    @property
    def shape(self) -> tuple[int, int]:
        n = self.centred.shape[1]
        return (n, n)

    def take(self, keep: np.ndarray) -> "LowRankCov":
        return LowRankCov(self.centred[:, keep], self.horizon)

    def dense(self) -> np.ndarray:
        return self.centred.T @ self.centred / self.horizon

    def __array__(self, dtype=None, copy=None):
        out = self.dense()
        return out if dtype is None else out.astype(dtype)


def as_dense(cov) -> np.ndarray:
    """The dense matrix of a covariance shift held either way (older checkpoints hold the dense matrix)."""
    return cov.dense() if isinstance(cov, LowRankCov) else cov


def covariance_factor(path: np.ndarray, delta0: np.ndarray, horizon: int) -> LowRankCov:
    increments = np.diff(np.vstack([delta0[None, :], path]), axis=0)[1 : horizon + 1]
    return LowRankCov(increments - increments.mean(axis=0, keepdims=True), horizon)


def covariance_shift(path: np.ndarray, delta0: np.ndarray, horizon: int) -> np.ndarray:
    """``Delta Sigma = sum_{t=1}^H (dp_t - mean dp)(dp_t - mean dp)' / H`` over the predicted path."""
    return covariance_factor(path, delta0, horizon).dense()
