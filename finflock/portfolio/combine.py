"""From event forecasts to the optimiser's daily inputs (spec 9.1, 9.2).

* Return forecast of an event: increments ``m_t`` (days ``t >= 1``) of the combined expected price path.
* Holding-period forecast: ``mu_h = (1 - psi) m_{h+1} + psi m_{h+2}``, with ``psi`` the overnight share
  of a day's move (a fill at the open of day ``+1`` earns the intraday part of day ``t`` and the
  overnight part of day ``t + 1``).
* Decision date ``d`` (after the close) adds up every active event: event ``e`` with day 0 at ``d_e``
  contributes its ``mu`` at offset ``d - d_e + h``.
* Confidence scales the forecast; the vol dial scales the variances; volume and vol dials set the cost.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from finflock.common.config import Config

EQUAL_WEIGHT = 0.5  # the 50/50 average and the equal-weight linear pool


def combine_dials(dial_a: np.ndarray, dial_b: np.ndarray, w_a: float, w_b: float) -> np.ndarray:
    """``dial = w_A dial_A + w_B dial_B``. Legacy fixed-weight rule (the "current OLS rule" in the merge comparison);
    the headline merge is ``lead_correct`` with a fitted trust."""
    return w_a * dial_a + w_b * dial_b


# --- Lead, Correct, Brake (spec 9.1) ---------------------------------------------------------------------------------


def horizon_bucket_rows(buckets: Sequence[Sequence[int]], horizon: int) -> list[slice]:
    """Row slices of an increments array (row 0 = day 1) for the trust horizons, e.g. ``[[1,1],[2,3],[4,10]]``."""
    return [slice(first - 1, min(last, horizon)) for first, last in buckets]


def trust_value(features: np.ndarray, theta: np.ndarray, bounds: Sequence[float]) -> np.ndarray:
    """``trust = clip(1 + theta_0 + features @ theta_1:, lo, hi)``; ``theta = 0`` gives trust 1, the plain sum."""
    z = np.atleast_2d(features)
    raw = 1.0 + theta[0] + z @ theta[1:]
    return np.clip(raw, float(bounds[0]), float(bounds[1]))


def lead_correct(road1: np.ndarray, correction: np.ndarray, trust: np.ndarray | float) -> np.ndarray:
    """Final forecast = road 1 + trust * road 2's correction (the correction is fitted on road 1's leftover error)."""
    return road1 + trust * correction


def half_average(road1: np.ndarray, road2_only: np.ndarray) -> np.ndarray:
    """The 50/50 average of the two roads' own forecasts."""
    return EQUAL_WEIGHT * road1 + EQUAL_WEIGHT * road2_only


def cautious_min(road1: np.ndarray, road2_only: np.ndarray) -> np.ndarray:
    """Trade the smaller of the two forecasts; zero where the signs differ."""
    same = np.sign(road1) == np.sign(road2_only)
    return np.where(same, np.sign(road1) * np.minimum(np.abs(road1), np.abs(road2_only)), 0.0)


def disagreement(road1: np.ndarray, road2_only: np.ndarray, floor: float) -> np.ndarray:
    """Gap between the two forecasts relative to the signal size, in [0, 2]: strong news is not penalised for being strong.

    ``|f1 - f2| / (max(|f1|, |f2|) + floor)`` per asset, using forecasts summed over the days that matter."""
    return np.abs(road1 - road2_only) / (np.maximum(np.abs(road1), np.abs(road2_only)) + floor)


def pooled_variance(within1: float, within2: float, road1: np.ndarray, road2_only: np.ndarray) -> np.ndarray:
    """Linear-pool variance with equal weights: average within-road error variance + between-road spread.

    Between = sum_i w_i (m_i - mbar)^2 = (f1 - f2)^2 / 4."""
    return EQUAL_WEIGHT * (within1 + within2) + EQUAL_WEIGHT * EQUAL_WEIGHT * (road1 - road2_only) ** 2


def soft_threshold(forecast: np.ndarray, gap: np.ndarray, kappa: float) -> np.ndarray:
    """Shrink the forecast toward zero by ``kappa * gap`` (never past zero)."""
    return np.sign(forecast) * np.maximum(np.abs(forecast) - kappa * gap, 0.0)


def abstain_mask(relative_gap: np.ndarray, threshold: float) -> np.ndarray:
    """True where the roads disagree above the dev-chosen high quantile: those forecasts are set to zero."""
    return relative_gap > threshold


def choose_by_loss(losses: Mapping[str, float], must_include: str = "plain_sum") -> str:
    """Pre-registered choice: the lowest out-of-sample loss; the plain sum must be in the race (forecast-combination puzzle)."""
    if must_include not in losses:
        raise ValueError(f"{must_include!r} must be among the candidates, got {sorted(losses)}")
    finite = {k: v for k, v in losses.items() if np.isfinite(v)}
    if must_include not in finite:
        raise ValueError(f"the loss of {must_include!r} is not finite")
    return min(finite, key=lambda k: (finite[k], k != must_include))


def holding_forecasts(increments: np.ndarray, psi: float) -> np.ndarray:
    """``mu_k = (1 - psi) m_{k+1} + psi m_{k+2}`` for ``k = 0 .. H-1`` from increments ``m_1 .. m_H``.

    ``increments`` has shape ``(H, N)`` (row ``0`` is day 1); increments beyond ``H`` are zero.
    """
    padded = np.vstack([increments, np.zeros((1, increments.shape[1]))])
    return (1.0 - psi) * padded[:-1] + psi * padded[1:]


def mu_panel(
    event_mu: Mapping[str, tuple[int, np.ndarray]], n_sessions: int, n_assets: int, horizon: int
) -> np.ndarray:
    """Calendar panel ``mu[d, h, asset]`` summed over active events.

    ``event_mu[e] = (day0 session index, mu array (H_e, N))`` where ``mu`` already includes the
    confidence scaling. Offsets beyond an event's horizon contribute nothing.
    """
    panel = np.zeros((n_sessions, horizon, n_assets))
    for day0, mu in event_mu.values():
        length = mu.shape[0]
        for d in range(day0, min(day0 + length, n_sessions)):
            offset = d - day0
            span = min(horizon, length - offset)
            if span > 0:
                panel[d, :span] += mu[offset : offset + span]
    return panel


def blend_covariance(
    event_cov: np.ndarray, rolling_cov: np.ndarray, intensity: float, floor: float
) -> np.ndarray:
    """``Sigma = intensity * rolling + (1 - intensity) * event`` projected to positive definite."""
    blended = intensity * rolling_cov + (1.0 - intensity) * event_cov
    symmetric = (blended + blended.T) / 2.0
    values, vectors = np.linalg.eigh(symmetric)
    return (vectors * np.maximum(values, floor)) @ vectors.T


def event_covariance(
    base_cov: np.ndarray, vol_dial: np.ndarray, shift: np.ndarray | None, ambiguity_var: np.ndarray | None
) -> np.ndarray:
    """Baseline covariance with each variance scaled by ``exp(2 r_sigma)``, ambiguity variance added on the
    diagonal and the game's covariance shift added off the diagonal (and on it)."""
    scale = np.exp(vol_dial)
    cov = base_cov * np.outer(scale, scale)
    if ambiguity_var is not None:
        cov = cov + np.diag(ambiguity_var)
    return cov + (shift if shift is not None else 0.0)


def confidence_scale(mu: np.ndarray, confidence: float) -> np.ndarray:
    """Scale a forecast by the calibrated confidence in [0, 1]."""
    return mu * float(confidence)


def active_dial(dials: Sequence[tuple[int, np.ndarray]], d: int, horizon: int) -> np.ndarray:
    """Per-asset dial on session ``d``: the largest value among events still inside their window
    (``0 <= d - day0 < horizon``); zero when none is active."""
    values = [dial for day0, dial in dials if 0 <= d - day0 < horizon]
    return np.max(values, axis=0) if values else None


def cost_multiplier_inputs(cfg: Config) -> tuple[float, float]:
    """Headline impact exponent and impact constant from config (shown beside every result)."""
    return float(cfg.get("portfolio.impact.alpha_headline")), float(cfg.get("portfolio.impact.y"))
