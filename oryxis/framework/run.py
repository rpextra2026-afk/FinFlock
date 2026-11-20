"""One cell end to end on CPU: event forecasts, daily optimiser inputs, positions (spec 9).

Everything here reads frozen tapes and fitted objects; no LLM call and no label is used. Variants:
``road1`` (ripple dials with a common decay), ``road2`` (game on the reader's consensus only), ``both``
(game on the consensus including the ripple) and ``none`` (flat). The sheaf is one of the road-1 methods;
which one feeds the optimiser is a pre-registered config choice, never picked by outcome.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from oryxis.common.config import Config
from oryxis.framework.calibration import apply as apply_volu_calibration
from oryxis.framework.events import EventReading
from oryxis.framework.road1_stage import Road1Predictor
from oryxis.framework.road2_stage import PersonaEventAnswer, run_road2_event
from oryxis.mfg.precompute import MonthGame, estimate_covariance
from oryxis.mfg.volume import LowRankCov, as_dense
from oryxis.portfolio.combine import (
    abstain_mask,
    blend_covariance,
    cautious_min,
    confidence_scale,
    disagreement,
    event_covariance,
    half_average,
    holding_forecasts,
    horizon_bucket_rows,
    lead_correct,
    mu_panel,
    pooled_variance,
    soft_threshold,
)
from oryxis.portfolio.constraints import project
from oryxis.portfolio.costs import lambda_base, lognormal_cost_multiplier, one_way_cost_fraction
from oryxis.portfolio.market import MarketView
from oryxis.portfolio.optimiser import HorizonInputs, apply_band, first_step, stationary_value

VARIANTS = ("road1", "road2", "both", "none")
MERGED = "merged"  # Lead, Correct, Brake (spec 9.1); needs ``CellContext.merge`` fitted on dev
MERGE_RULES = ("plain_sum", "half_average", "lcb", "cautious")
BRAKES = ("variance_add", "cautious_band", "abstain", "none")


@dataclass
class EventForecast:
    event_id: str
    day0: int  # session index of day 0
    assets: np.ndarray  # global asset indices
    increments: np.ndarray  # (H, n) predicted tradable increments, day 1 first
    vol_dial: np.ndarray
    volume_dial: np.ndarray
    cov_shift: "LowRankCov | None"  # (H, n) factor of the (n, n) shift; see oryxis.mfg.volume.LowRankCov
    confidence: float
    flags: dict = field(default_factory=dict)
    inc_a: np.ndarray | None = None  # road-1 component before weighting, (H, n)
    inc_b: np.ndarray | None = None  # road-2 component before weighting, (H, n)
    aux: dict = field(default_factory=dict)  # pre-event inputs of the trust: link strength, persona spread, ... (per kept asset)
    vol_parts: tuple[np.ndarray, np.ndarray] | None = None  # (road 1, road 2) vol dial before weighting
    volu_parts: tuple[np.ndarray, np.ndarray] | None = None
    ambiguity_var: np.ndarray | None = None  # brake (a): extra forecast variance on the covariance diagonal
    relative_gap: np.ndarray | None = None  # disagreement of the two roads relative to signal size


@dataclass
class MonthContext:
    game: MonthGame
    asset_ids: list[str]
    uncovered: np.ndarray
    baseline_dv: np.ndarray


@dataclass
class CellContext:
    """Callables and fitted objects a cell needs. All are pure functions of data known at their date."""

    cfg: Config
    global_assets: list[str]
    readings: Mapping[str, EventReading]
    day0_index: Mapping[str, int]
    day0_date: Mapping[str, pd.Timestamp]
    source_dials: Callable[[str], pd.DataFrame]  # event -> dials of its pins (ret, vol, volu)
    road1: Road1Predictor | None
    month_context: Callable[[pd.Period], MonthContext]
    answers: Callable[[str], Mapping[str, PersonaEventAnswer]]
    retail_ids: list[str]
    state_z: Callable[[str], Mapping[str, float]]
    trait_b: Mapping[str, float]
    day0_move: Callable[[str, list[str]], Mapping[str, float]]  # event, S&P pinned nodes -> observed day-0 log returns
    confidence: Callable[[float], float] = lambda certainty: certainty
    dial_weights: tuple[float, float] = (1.0, 1.0)
    return_weights: dict[str, tuple[float, float]] = field(default_factory=lambda: {v: (1.0, 1.0) for v in VARIANTS})
    vol_kappas: tuple[float, float] = (0.0, 0.0)
    merge: Any = None  # fitted ``MergeFit`` (oryxis.fit.return_weights); needed for the "merged" variant
    merge_rule: str | None = None  # None = the rule the dev fit chose; or one of MERGE_RULES to report a rule side by side
    brake: str | None = None  # None = ``framework.brake.variant``; or one of BRAKES
    dial_trust: tuple[float, float] = (1.0, 1.0)  # trust on road 2's vol and volume corrections; 1 = plain sum until fitted
    volu_calibration: tuple[float, float] | None = None  # (scale, shift) of road 2's volume dial, fitted on the fit window (oryxis.framework.calibration); None = raw
    consensus_override: Callable[[str, list[str]], Mapping[str, float]] | None = None  # ablation: replaces the pinned consensus that road 2 reads (event, pinned S&P nodes -> value)


def _common_phi(game: MonthGame, cfg: Config) -> float:
    configured = cfg.get("framework.common_phi")
    if configured is not None:
        return float(configured)
    return float(np.mean([1.0 - t.trade_rate_bar for t in game.types]))


def persona_spread(answers: Mapping[str, PersonaEventAnswer], ids: list[str]) -> np.ndarray:
    """Standard deviation across the personas that saw the event of the value change they give each asset (percent)."""
    seen = [a for a in answers.values() if a.saw]
    if len(seen) < 2:
        return np.zeros(len(ids))
    return np.array([np.std([a.values.get(node, 0.0) for a in seen]) for node in ids])


def forecast_event(ctx: CellContext, event_id: str, variant: str) -> EventForecast | None:
    """Predicted increments and dials of one event over the month's S&P assets, or ``None``."""
    if variant == MERGED:
        return _forecast_merged(ctx, event_id)
    if variant == "none" or event_id not in ctx.readings:
        return None
    cfg, reading = ctx.cfg, ctx.readings[event_id]
    month = ctx.month_context(pd.Period(ctx.day0_date[event_id], freq="M"))
    game, ids = month.game, month.asset_ids
    index = {a: i for i, a in enumerate(ids)}
    n, horizon = len(ids), int(cfg.get("game.horizon_days"))
    pin_dials = ctx.source_dials(event_id)
    consensus = np.zeros(n)
    for pin, ret in zip(reading.pins, pin_dials["ret"], strict=True):
        if pin["node_id"] in index:
            consensus[index[pin["node_id"]]] = ret
    if ctx.consensus_override is not None:
        for node, value in ctx.consensus_override(event_id, [p["node_id"] for p in reading.pins if p["node_id"] in index]).items():
            consensus[index[node]] = value
    ripple = None
    if variant in ("road1", "both") and ctx.road1 is not None:
        ripple = ctx.road1.predict(reading, pin_dials, ctx.day0_date[event_id])
        for node, row in ripple.iterrows():
            if node in index and row["role"] == "ripple" and variant == "both":
                consensus[index[node]] = row["ret"]
    pinned_nodes = [p["node_id"] for p in reading.pins if p["node_id"] in index]
    flags = {"n_pinned_sp": len(pinned_nodes), "truncated": bool(ripple["truncated"].iloc[0]) if ripple is not None else False}
    source_size = float(np.max(np.abs(pin_dials["ret"]))) if len(pin_dials) else 0.0
    vol_a = np.zeros(n)
    volu_a = np.zeros(n)
    if ripple is not None:
        for node, row in ripple.iterrows():
            if node in index:
                vol_a[index[node]], volu_a[index[node]] = row["vol"], row["volu"]
    ret_a = np.zeros(n)  # road-1 return dial for every asset of the month (pinned: the reader's own value effect)
    if ripple is not None:
        for node, row in ripple.iterrows():
            if node in index:
                ret_a[index[node]] = row["ret"]
    phi_c = _common_phi(game, cfg)
    days = np.arange(1, horizon + 1)
    cumulative = ret_a[None, :] * (1.0 - phi_c ** (days[:, None] + 1))
    inc_a = np.diff(np.vstack([ret_a[None, :] * (1.0 - phi_c), cumulative]), axis=0)  # road-1 timing: common decay
    inc_b, cov_shift, vol_b, volu_b = np.zeros((horizon, n)), None, np.zeros(n), np.zeros(n)
    if variant in ("road2", "both"):
        moves = ctx.day0_move(event_id, pinned_nodes)
        pinned = np.array([index[a] for a in pinned_nodes if a in moves], dtype=int)
        observed = np.array([moves[a] for a in pinned_nodes if a in moves])
        result = run_road2_event(
            cfg, game, consensus, ctx.answers(event_id), ctx.retail_ids, index, month.uncovered,
            ctx.state_z(event_id), ctx.trait_b, pinned, observed, month.baseline_dv,
            int(cfg.get("game.volume_window_days")), ctx.vol_kappas,
        )
        inc_b, cov_shift, vol_b, volu_b = result.increments, result.cov_shift, result.vol_dial, apply_volu_calibration(result.volume_dial, ctx.volu_calibration)
        flags.update({"weak_id": bool(result.anchor.weak_id.any()) if result.anchor else False,
                      "sleeve_flow": result.sleeve_flow})
    if variant == "road1" and ripple is None:
        return None
    r_a, r_b = ctx.return_weights[variant]
    use_a, use_b = variant in ("road1", "both"), variant in ("road2", "both")
    increments = (r_a * inc_a if use_a else 0.0) + (r_b * inc_b if use_b else 0.0)
    w_a, w_b = ctx.dial_weights
    global_index = {a: i for i, a in enumerate(ctx.global_assets)}
    keep = np.array([i for i, a in enumerate(ids) if a in global_index], dtype=int)
    id_list = [ids[i] for i in keep]
    aux = {"multi_pin": np.full(len(keep), float(len(pinned_nodes) > 1)), "inst_share": (1.0 - month.uncovered)[keep],
           "link_strength": np.abs(ret_a)[keep] / max(source_size, float(cfg.get("framework.brake.disagreement_floor"))),
           "persona_spread": persona_spread(ctx.answers(event_id), id_list) if variant in ("road2", "both") else np.zeros(len(keep))}
    return EventForecast(
        event_id, ctx.day0_index[event_id], np.array([global_index[ids[i]] for i in keep], dtype=int),
        increments[:, keep], w_a * vol_a[keep] + w_b * vol_b[keep], w_a * volu_a[keep] + w_b * volu_b[keep],
        None if cov_shift is None else cov_shift.take(keep), ctx.confidence(reading.certainty), flags,
        inc_a[:, keep], inc_b[:, keep], aux, (vol_a[keep], vol_b[keep]), (volu_a[keep], volu_b[keep]),
    )


def trust_inputs(f1: EventForecast, fb: EventForecast, names) -> np.ndarray:
    """Trust features per asset, in the order of ``names``; all known before the event. Persona-based ones come from the
    ``both`` forecast, the rest from the road-1 forecast."""
    from_both = ("persona_spread", "inst_share")
    return np.column_stack([(fb if name in from_both else f1).aux[name] for name in names])


def merged_increments(ctx: CellContext, a: np.ndarray, b_raw: np.ndarray, r2_only: np.ndarray, features: np.ndarray, rule: str) -> np.ndarray:
    """Final daily increments ``(H, n)`` from road 1 ``a``, road 2's raw correction, road 2's own forecast and the trust features."""
    if rule == "plain_sum":
        return lead_correct(a, b_raw, 1.0)
    if rule == "half_average":
        return half_average(a, r2_only)
    if rule == "cautious":
        return cautious_min(a, r2_only)
    if rule != "lcb":
        raise ValueError(f"unknown merge rule {rule!r}; choose one of {MERGE_RULES}")
    horizon = a.shape[0]
    out = np.array(a, dtype=float)
    spec = ctx.cfg.get("framework.merge")
    for rows, model in zip(horizon_bucket_rows(spec["horizons"], horizon), ctx.merge.models, strict=True):
        out[rows] = lead_correct(a[rows], model.correction(b_raw[rows]), model.trust(features)[None, :] if len(a[rows]) else 1.0)
    return out


def _forecast_merged(ctx: CellContext, event_id: str) -> EventForecast | None:
    """Lead (road 1), Correct (trust * road 2's correction), Brake (disagreement between the roads) - spec 9.1."""
    if ctx.merge is None:
        raise ValueError("the merged variant needs a merge fitted on dev (CellContext.merge)")
    cfg = ctx.cfg
    f1, fb, f2 = (forecast_event(ctx, event_id, v) for v in ("road1", "both", "road2"))
    if f1 is None or fb is None or f2 is None:
        return None
    rule = ctx.merge_rule or ctx.merge.rule
    features = trust_inputs(f1, fb, ctx.merge.feature_names)
    increments = merged_increments(ctx, f1.inc_a, fb.inc_b, f2.inc_b, features, rule)

    brake = ctx.brake or str(cfg.get("framework.brake.variant"))
    if brake not in BRAKES:
        raise ValueError(f"unknown brake {brake!r}; choose one of {BRAKES}")
    floor = float(cfg.get("framework.brake.disagreement_floor"))
    total_1, total_2 = f1.inc_a.sum(axis=0), f2.inc_b.sum(axis=0)
    relative = disagreement(total_1, total_2, floor)
    ambiguity = None
    if brake == "variance_add":
        ambiguity = float(cfg.get("framework.brake.variance_scale")) * pooled_variance(ctx.merge.within_variance[0], ctx.merge.within_variance[1], total_1, total_2)
    elif brake == "cautious_band":
        band = cfg.get("framework.brake.band")
        increments = cautious_min(f1.inc_a, f2.inc_b) if band["mode"] == "min" else soft_threshold(increments, np.abs(f1.inc_a - f2.inc_b), float(band["kappa"]))
    elif brake == "abstain":
        increments = np.where(abstain_mask(relative, ctx.merge.gap_quantile)[None, :], 0.0, increments)
    mean_gap = float(relative[np.abs(total_1) + np.abs(total_2) > 0].mean()) if np.any(np.abs(total_1) + np.abs(total_2) > 0) else 0.0
    confidence = ctx.confidence(ctx.readings[event_id].certainty) * (1.0 - float(cfg.get("framework.brake.confidence_weight")) * min(mean_gap, 1.0))
    trust_vol, trust_volu = ctx.dial_trust
    flags = {**f1.flags, **{k: v for k, v in fb.flags.items() if k not in f1.flags}, "merge_rule": rule, "brake": brake, "mean_gap": mean_gap}
    return EventForecast(
        event_id, f1.day0, f1.assets, increments, f1.vol_parts[0] + trust_vol * fb.vol_parts[1],
        f1.volu_parts[0] + trust_volu * fb.volu_parts[1], fb.cov_shift, confidence, flags, f1.inc_a, fb.inc_b, {**f1.aux, "persona_spread": fb.aux["persona_spread"]},
        (f1.vol_parts[0], fb.vol_parts[1]), (f1.volu_parts[0], fb.volu_parts[1]), ambiguity, relative,
    )


def build_inputs(forecasts: list[EventForecast], n_sessions: int, n_assets: int, psi: float, horizon: int):
    """Calendar panels: expected returns by offset, and per-event dials with their day 0."""
    event_mu = {}
    for f in forecasts:
        mu_assets = confidence_scale(holding_forecasts(f.increments, psi), f.confidence)
        full = np.zeros((mu_assets.shape[0], n_assets))
        full[:, f.assets] = mu_assets
        event_mu[f.event_id] = (f.day0, full)
    return mu_panel(event_mu, n_sessions, n_assets, horizon)


@dataclass
class StrategyParams:
    gamma: float
    cost_scale: float
    band: float
    round_trip_bp: float
    alpha: float
    aum_usd: float
    constraint: str = "headline"


class FrameworkStrategy:
    """Daily rule: aggregate forecasts, build Sigma and Lambda from causal prices, run D21 with the band."""

    def __init__(
        self, cfg: Config, global_assets: list[str], panel: np.ndarray, forecasts: list[EventForecast],
        universe: Callable[[pd.Timestamp], set[str]], params: StrategyParams, beta: float,
        world_shift: Callable[[pd.Timestamp], tuple[float, float]] | None = None,
    ) -> None:
        self.cfg, self.assets, self.panel, self.forecasts = cfg, global_assets, panel, forecasts
        self.universe, self.params, self.beta = universe, params, beta
        self.world_shift = world_shift  # world lane (config framework.world_lane.enabled): date -> (market-level log-volatility shift, log-volume shift); None = off
        self.index = {a: i for i, a in enumerate(global_assets)}

    def _active_dial(self, d: int, attr: str, n: int) -> np.ndarray:
        horizon = self.panel.shape[1]
        out = np.zeros(n)
        for f in self.forecasts:
            if 0 <= d - f.day0 < horizon:
                dial = np.zeros(n)
                dial[f.assets] = getattr(f, attr)
                out = np.maximum(out, dial)
        return out

    def __call__(self, date: pd.Timestamp, view: MarketView, held: pd.Series) -> pd.Series:
        cfg, p = self.cfg, self.params
        d = view.close.index.get_loc(date)
        window = int(cfg.get("framework.rolling_window_sessions"))
        members = [a for a in self.universe(date) if a in self.index and a in view.close.columns]
        returns = view.close[members].pct_change().iloc[-window:]
        usable = [a for a in members if returns[a].notna().mean() > float(cfg.get("framework.min_return_coverage"))]
        if len(usable) < 2:
            return pd.Series(dtype=float)
        idx = np.array([self.index[a] for a in usable])
        rets = returns[usable].fillna(0.0).to_numpy()
        rolling = estimate_covariance(rets, float(cfg.get("framework.covariance.eigenvalue_floor")))
        horizon = self.panel.shape[1]
        mu = self.panel[d][:, idx]
        vol = self._active_dial(d, "vol_dial", len(self.assets))[idx]
        volu = self._active_dial(d, "volume_dial", len(self.assets))[idx]
        if self.world_shift is not None:  # the same market-level shift on every name: it scales the covariance and enters the cost multiplier exp(vol - alpha volu)
            vol_shift, volu_shift = self.world_shift(date)
            vol, volu = vol + vol_shift, volu + volu_shift
        shift = None
        for f in self.forecasts:
            if f.cov_shift is not None and 0 <= d - f.day0 < horizon:
                local = np.zeros((len(self.assets), len(self.assets)))
                local[np.ix_(f.assets, f.assets)] = as_dense(f.cov_shift)
                shift = local[np.ix_(idx, idx)] if shift is None else shift + local[np.ix_(idx, idx)]
        ambiguity = None
        for f in self.forecasts:
            if f.ambiguity_var is not None and 0 <= d - f.day0 < horizon:
                local = np.zeros(len(self.assets))
                local[f.assets] = f.ambiguity_var
                ambiguity = local[idx] if ambiguity is None else ambiguity + local[idx]
        event_cov = event_covariance(rolling, vol, shift, ambiguity)
        sigma = blend_covariance(event_cov, rolling, float(cfg.get("framework.covariance.shrink_to_rolling")),
                                 float(cfg.get("framework.covariance.eigenvalue_floor")))
        sigma_daily = np.sqrt(np.diag(rolling))
        dollar = (view.close[usable] * view.volume[usable]).iloc[-int(cfg.get("portfolio.impact.adv_window_days")):]
        adv = dollar.mean().to_numpy()
        base = lambda_base(sigma_daily, np.maximum(adv, 1.0), p.alpha, p.aum_usd, cfg)
        multiplier = np.array([lognormal_cost_multiplier(0.0, 0.0, 0.0, p.alpha) * np.exp(v - p.alpha * u) for v, u in zip(vol, volu, strict=True)])
        lam = np.diag(base * multiplier * p.cost_scale)
        inputs = HorizonInputs(mu, np.broadcast_to(sigma, (horizon, *sigma.shape)).copy(),
                               np.broadcast_to(lam, (horizon, *lam.shape)).copy())
        a_terminal = stationary_value(sigma, lam, p.gamma, self.beta, float(cfg.get("portfolio.stationary.tol")),
                                      int(cfg.get("portfolio.stationary.max_iter")))
        n_prev = held.reindex(usable).fillna(0.0).to_numpy()
        step = first_step(n_prev, inputs, p.gamma, self.beta, a_terminal, np.zeros(len(usable)))
        kappa = np.full(len(usable), one_way_cost_fraction(p.round_trip_bp, cfg))
        banded = apply_band(n_prev, step, kappa, p.band)
        return pd.Series(project(banded, cfg, p.constraint), index=usable)
