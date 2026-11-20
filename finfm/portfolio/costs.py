"""Trading costs: the optimiser's own matrix (D11) and the common scoring cost grid (spec 9.2, 9.4)."""

from __future__ import annotations

import numpy as np

from finfm.common.config import Config

HALF = 0.5


def lambda_base(
    sigma_daily: np.ndarray, adv_usd: np.ndarray, alpha: float, aum_usd: float, cfg: Config
) -> np.ndarray:
    """Diagonal of ``Lambda_base`` in weight space.

    Dollar cost of trading ``q`` is ``Y sigma (q / V)^alpha q``; its quadratic fit at the typical trade
    ``Q_bar`` is ``Lambda_$ = 2 Y sigma Q_bar^(alpha-1) V^(-alpha)``. In weights (``q = AUM dw``) the
    cost as a share of AUM has ``Lambda_w = AUM Lambda_$``.
    """
    y = float(cfg.get("portfolio.impact.y"))
    q_bar = float(cfg.get("portfolio.impact.typical_trade_adv_share")) * adv_usd
    lam_dollar = 2.0 * y * sigma_daily * q_bar ** (alpha - 1.0) * adv_usd ** (-alpha)
    return aum_usd * lam_dollar


def lognormal_cost_multiplier(
    var_vol: float, var_volume: float, cov_vol_volume: float, alpha: float
) -> float:
    """``E[exp(r_sigma - alpha r_V)]`` for jointly normal dials (D11 Prop. 11.2), mean already in the dials."""
    variance = var_vol + alpha**2 * var_volume - 2.0 * alpha * cov_vol_volume
    return float(np.exp(HALF * variance))


def fixed_cost(trade_weights: np.ndarray, round_trip_bp: float, multiplier: float, cfg: Config) -> float:
    """Fixed round-trip cost on traded notional, as a share of AUM (half the round trip per side)."""
    one_way = HALF * round_trip_bp * float(cfg.get("portfolio.scoring.bp_to_fraction"))
    return float(multiplier * one_way * np.abs(trade_weights).sum())


def one_way_cost_fraction(round_trip_bp: float, cfg: Config) -> float:
    """The band's ``kappa``: half the headline round trip as a fraction of notional."""
    return HALF * round_trip_bp * float(cfg.get("portfolio.scoring.bp_to_fraction"))


def sqrt_impact_cost(
    trade_weights: np.ndarray, sigma_daily: np.ndarray, adv_usd: np.ndarray, aum_usd: float,
    multiplier: float, cfg: Config,
) -> float:
    """Square-root impact: ``Y sigma sqrt(q / ADV) q`` summed over assets, as a share of AUM."""
    y = float(cfg.get("portfolio.impact.y"))
    dollars = aum_usd * np.abs(trade_weights)
    cost_dollars = y * sigma_daily * np.sqrt(dollars / adv_usd) * dollars
    return float(multiplier * cost_dollars.sum() / aum_usd)
