"""Per-month game objects: covariance, outside liquidity, fund types and their matrices (spec 7.1).

Notation follows ``math_for_paper.md``. ``N`` S&P 500 assets are priced; with the sleeve (R-8) every
fund's universe also holds one extra asset ``o`` for the rest of its book. Prices clear on the S&P
names only; the sleeve's price is fixed during an event.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
from sklearn.covariance import ledoit_wolf

from oryxis.common.config import Config


SP_COVER_TOLERANCE = 1e-9


class GameError(ValueError):
    """Raised when a month's matrices cannot be built (not positive definite, no assets...)."""


def projected_precision(sigma: np.ndarray, universe: np.ndarray) -> np.ndarray:
    """Projected precision on ``universe`` with the full-investment constraint (Lemma P).

    ``S = P - q q' / (1' q)`` with ``P = Sigma_U^-1`` and ``q = P 1``; zero outside ``universe``.
    Rows and columns of ``S`` sum to zero, so it is a signed graph Laplacian on the universe.
    """
    size = sigma.shape[0]
    out = np.zeros((size, size))
    if len(universe) < 2:
        return out
    precision = np.linalg.inv(sigma[np.ix_(universe, universe)])
    q = precision @ np.ones(len(universe))
    out[np.ix_(universe, universe)] = precision - np.outer(q, q) / q.sum()
    return out


def min_variance_weights(sigma: np.ndarray, universe: np.ndarray) -> np.ndarray:
    """Minimum-variance fully invested weights on ``universe``; zero elsewhere."""
    weights = np.zeros(sigma.shape[0])
    if len(universe) == 0:
        return weights
    solved = np.linalg.solve(sigma[np.ix_(universe, universe)], np.ones(len(universe)))
    weights[universe] = solved / solved.sum()
    return weights


def rebalancing_matrix(weights: np.ndarray) -> np.ndarray:
    """``diag(x) - x x'``: the drift-rebalancing Laplacian of a fully invested weight vector."""
    return np.diag(weights) - np.outer(weights, weights)


def estimate_covariance(returns: np.ndarray, ridge: float) -> np.ndarray:
    """Ledoit-Wolf shrinkage covariance, made positive definite if necessary."""
    if returns.ndim != 2 or returns.shape[0] < returns.shape[1] // 2:
        raise GameError("too few return observations for the number of assets")
    cov, _ = ledoit_wolf(returns)
    smallest = np.linalg.eigvalsh(cov).min()
    if smallest <= 0:
        cov = cov + (ridge - smallest) * np.eye(cov.shape[0])
    return cov


def amihud_liquidity(
    abs_returns: np.ndarray, dollar_volume: np.ndarray, scale: float, low: float, high: float
) -> np.ndarray:
    """Outside liquidity ``ell_j = s_ell / ILLIQ_j`` with winsorised Amihud illiquidity.

    Inputs are ``(days, assets)``; illiquidity is the mean of ``|r| / DV`` over the window.
    """
    valid = dollar_volume > 0
    counts = valid.sum(axis=0)
    if (counts == 0).any():
        raise GameError("Amihud illiquidity is missing for some assets (no positive dollar volume)")
    ratio = np.where(valid, abs_returns / np.where(valid, dollar_volume, 1.0), 0.0)
    illiq = ratio.sum(axis=0) / counts
    if (illiq <= 0).any():
        raise GameError("Amihud illiquidity is zero for some assets")
    lo, hi = np.quantile(illiq, [low, high])
    return scale / np.clip(illiq, lo, hi)


@dataclass(frozen=True)
class FundInput:
    """One game type (a persona cluster) as supplied by the persona and 13F layers."""

    name: str
    group: str  # key into game.types: mutual_pension, bank_dealer, hedge_fund
    book: float  # dollars, whole book
    weights: np.ndarray  # length N: S&P 500 weights as a share of the whole book
    turnover_pctile: float  # within its group, in [0, 1]


@dataclass(frozen=True)
class FundType:
    name: str
    group: str
    book: float
    sp_cover: float
    x_plus: np.ndarray  # weights on the S&P assets (+ sleeve last when sleeve is on)
    x_sp: np.ndarray  # S&P block of x_plus
    x_mv_sp: np.ndarray  # S&P block of the minimum-variance portfolio on its universe
    S: np.ndarray  # N x N S&P block of the projected precision
    R: np.ndarray  # N x N S&P block of the rebalancing matrix
    S_sleeve: np.ndarray  # (N,) sleeve row of the projected precision (zeros without sleeve)
    R_sleeve: np.ndarray  # (N,) sleeve row of the rebalancing matrix (zeros without sleeve)
    x_sleeve: float  # fund weight in the sleeve
    x_mv_sleeve: float  # minimum-variance weight in the sleeve
    gamma_bar: float
    rho: float
    theta: float
    vartheta: float
    mandate: float
    trade_rate_bar: float
    cost_bar: float
    beta_k: float
    omega_bar: float


@dataclass(frozen=True)
class MonthGame:
    """Everything the events of one month share."""

    assets: tuple[str, ...]
    sigma_plus: np.ndarray
    ell: np.ndarray
    types: tuple[FundType, ...]
    M_bar: np.ndarray
    R_sum: np.ndarray
    A: np.ndarray
    G: np.ndarray
    sleeve: bool

    @property
    def n_assets(self) -> int:
        return len(self.assets)

    def S_stack(self) -> np.ndarray:
        return np.stack([t.S for t in self.types])

    def R_stack(self) -> np.ndarray:
        return np.stack([t.R for t in self.types])


def discount_factor(cfg: Config) -> float:
    """``beta = 1 / (1 + annual rate / trading days)``."""
    return 1.0 / (
        1.0
        + float(cfg.get("game.discount_annual_rate")) / float(cfg.get("game.trading_days_per_year"))
    )


def trade_rate(cfg: Config, group: str, turnover_pctile: float) -> float:
    """Baseline daily trading rate ``1 - phi_bar``, scaled by turnover rank within its group."""
    base = float(cfg.get(f"game.types.{group}.trade_rate"))
    scaled = base * (
        float(cfg.get("game.trade_rate_turnover_base"))
        + float(cfg.get("game.trade_rate_turnover_slope")) * turnover_pctile
    )
    return float(
        np.clip(
            scaled,
            float(cfg.get("game.trade_rate_min")),
            float(cfg.get("game.trade_rate_max")),
        )
    )


def baseline_cost_scale(gamma_bar: float, phi_bar: float, beta_k: float) -> float:
    """``c_bar = gamma phi / ((1 - phi)(1 - beta phi))``: inverse of the stable-root equation."""
    return gamma_bar * phi_bar / ((1.0 - phi_bar) * (1.0 - beta_k * phi_bar))


def build_month_game(
    cfg: Config,
    assets: Sequence[str],
    sigma_plus: np.ndarray,
    ell: np.ndarray,
    funds: Sequence[FundInput],
    *,
    sleeve: bool | None = None,
) -> MonthGame:
    """Build the month's matrices. ``sigma_plus`` is ``(N+1) x (N+1)`` with the sleeve proxy last."""
    use_sleeve = bool(cfg.get("game.sleeve")) if sleeve is None else sleeve
    n = len(assets)
    if sigma_plus.shape != (n + 1, n + 1):
        raise GameError("sigma_plus must be (N+1)x(N+1) with the sleeve proxy as the last row")
    if len(ell) != n or not funds:
        raise GameError("ell must have one entry per asset and at least one fund is required")
    sigma = sigma_plus if use_sleeve else sigma_plus[:n, :n]
    min_weight = float(cfg.get("game.support_min_weight"))
    beta = discount_factor(cfg)
    types: list[FundType] = []
    for fund in funds:
        params = cfg.get(f"game.types.{fund.group}")
        sp_cover = float(fund.weights.sum())
        if not 0.0 < sp_cover <= 1.0 + SP_COVER_TOLERANCE:
            raise GameError(f"{fund.name}: S&P weights must sum to a share in (0, 1]")
        if use_sleeve:
            x_plus = np.concatenate([fund.weights, [max(1.0 - sp_cover, 0.0)]])
            book = fund.book
        else:
            x_plus = fund.weights / sp_cover
            book = fund.book * sp_cover
        universe = np.flatnonzero(x_plus >= min_weight)
        S_plus = projected_precision(sigma, universe)
        R_plus = rebalancing_matrix(x_plus)
        x_mv = min_variance_weights(sigma, universe)
        gamma_bar = float(params["gamma_bar"])
        rho = float(params["rho"])
        beta_k = beta * (1.0 - rho)
        phi_bar = 1.0 - trade_rate(cfg, fund.group, fund.turnover_pctile)
        types.append(
            FundType(
                name=fund.name,
                group=fund.group,
                book=book,
                sp_cover=sp_cover,
                x_plus=x_plus,
                x_sp=x_plus[:n],
                x_mv_sp=x_mv[:n],
                S=S_plus[:n, :n],
                R=R_plus[:n, :n],
                S_sleeve=S_plus[n, :n] if use_sleeve else np.zeros(n),
                R_sleeve=R_plus[n, :n] if use_sleeve else np.zeros(n),
                x_sleeve=float(x_plus[n]) if use_sleeve else 0.0,
                x_mv_sleeve=float(x_mv[n]) if use_sleeve else 0.0,
                gamma_bar=gamma_bar,
                rho=rho,
                theta=float(params["theta"]),
                vartheta=float(params["vartheta"]),
                mandate=float(params["mandate"]),
                trade_rate_bar=1.0 - phi_bar,
                cost_bar=baseline_cost_scale(gamma_bar, phi_bar, beta_k),
                beta_k=beta_k,
                omega_bar=float(cfg.get("game.omega_scale")) * rho / gamma_bar,
            )
        )
    strength_target = cfg.get("game.strength_target")
    if strength_target is not None:
        ell = normalise_liquidity(types, ell, float(strength_target))
    return assemble_month_game(assets, sigma_plus, ell, types, use_sleeve)


def normalise_liquidity(types: Sequence[FundType], ell: np.ndarray, target: float) -> np.ndarray:
    """Rescale outside liquidity so the median ratio of fund sensitivity to liquidity equals ``target``.

    The game mixes dollars (books, liquidity), daily covariances and belief units, so its raw scale is
    arbitrary. This fixes one overall scale from the data of the month; the fitted ``s_omega`` and
    ``s_ell`` of spec 7.9 then act as multipliers around it instead of having to span many orders.
    """
    sensitivity = np.diag(sum(t.book * t.omega_bar * t.S for t in types))
    ratio = np.median(sensitivity / ell)
    return ell * ratio / target


def assemble_month_game(
    assets: Sequence[str],
    sigma_plus: np.ndarray,
    ell: np.ndarray,
    types: Sequence[FundType],
    sleeve: bool,
) -> MonthGame:
    """Combine fund types into the month's matrices ``M, R, L, A = M + R + L`` and ``G = A^-1``."""
    M_bar = sum(t.book * t.omega_bar * t.S for t in types)
    R_sum = sum(t.theta * t.book * t.R for t in types)
    A = M_bar + R_sum + np.diag(ell)
    try:
        chol = np.linalg.cholesky(A)
    except np.linalg.LinAlgError as exc:
        raise GameError("M + R + L is not positive definite") from exc
    inv_chol = np.linalg.inv(chol)
    return MonthGame(
        assets=tuple(assets),
        sigma_plus=sigma_plus,
        ell=np.asarray(ell, dtype=float),
        types=tuple(types),
        M_bar=M_bar,
        R_sum=R_sum,
        A=A,
        G=inv_chol.T @ inv_chol,
        sleeve=sleeve,
    )


def type_params(cfg: Config) -> Mapping[str, Mapping[str, float]]:
    """The per-group parameter table (read-only convenience for reports)."""
    return cfg.get("game.types")
