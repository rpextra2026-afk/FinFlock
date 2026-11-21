"""Our framework end to end on real tapes (spec 9): dev fits, per-event forecasts, the daily optimiser and the cost grid.

Per-event failures never stop the run: an event whose month has no game, or whose forecast raises, is skipped and counted with its
reason. Numbers only; no verdict is computed here.
"""

from __future__ import annotations

import hashlib
import math
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd

from finflock.common import ckpt
from finflock.common.progress import Heartbeat
from finflock.common.config import Config
from finflock.common.schema import STATUS, Status
from finflock.common.timing import TradingCalendar
from finflock.common.windows import BucketWindows
from finflock.fit.return_weights import fit_merge, fit_return_weights, merge_rows_from_forecasts
from finflock.framework.assemble import persona_answers
from finflock.framework.events import readings_from_tape
from finflock.framework.run import (
    MERGED,
    VARIANTS,
    CellContext,
    EventForecast,
    FrameworkStrategy,
    StrategyParams,
    build_inputs,
    forecast_event,
)
from finflock.mfg.precompute import discount_factor
from finflock.personas.skeletons import RETAIL, Skeleton
from finflock.portfolio.costs import fixed_cost
from finflock.portfolio.execute import BacktestResult, net_returns, run_backtest, scale_to_risk_target
from finflock.real.framework_fit import DevFit, fit_dev, panels_of, sheaf_test_scores
from finflock.real.game_inputs import GameInputs
from finflock.real.market_load import RealMarket
from finflock.real.appendix import appendix_tables
from finflock.real.summary import summary_row
from finflock.road1.graph import LINK_FROM, LINK_TO

CONFIDENCE_FROM_CERTAINTY, CONFIDENCE_NONE = "reader_certainty", "none"
EXAMPLES_SHOWN = 3  # event ids listed in an error message
DEV, TEST = "dev", "test"
REASON_CHARS = 160  # longest exception text kept per skipped event
GROUP_CHARS = 80  # reasons are grouped by this many leading characters
TOP_REASONS = 20


ERROR_TEXT_CHARS = 300  # how much of an exception message is kept in the diagnostics

@dataclass
class FrameworkResult:
    forecasts: dict[str, list[EventForecast]]
    backtests: dict[str, BacktestResult]
    net: dict[str, pd.DataFrame]
    summary: pd.DataFrame
    skipped: pd.DataFrame
    fits: dict
    diagnostics: dict = field(default_factory=dict)
    dev_fit: tuple | None = None  # (DevFit, source dials) of this run; saved by run_cpu.py --stop-after-dev, loaded by --load-dev
    extra: dict = field(default_factory=dict)  # the appendix tables (finflock.real.appendix), each with the metrics as rows


def confidence_rule(cfg: Config, policy: str):
    """The multiplier on an event's forecast: the clipped reader certainty, or 1 for every event (certainty not used)."""
    if policy == CONFIDENCE_NONE:
        return lambda certainty: 1.0
    if policy == CONFIDENCE_FROM_CERTAINTY:
        return lambda certainty: float(np.clip(certainty, float(cfg.get("framework.min_confidence")), 1.0))
    raise ValueError(f"unknown framework_confidence policy {policy!r}")


def _safe_forecasts(ctx: CellContext, ids: list[str], variant: str, skips: list[dict]) -> list[EventForecast]:
    out = []
    beat = Heartbeat(f"forecasts {variant}", len(ids), float(ctx.cfg.get("cpu.progress_seconds")))
    for done, event_id in enumerate(ids):
        beat.tick(done, f"({len(out)} forecast, {len(skips)} skipped so far)")
        try:
            forecast = forecast_event(ctx, event_id, variant)
        except Exception as exc:  # noqa: BLE001 - one event must never stop the run; the reason is recorded
            skips.append({"event_id": event_id, "variant": variant, "reason": f"{type(exc).__name__}: {str(exc)[:REASON_CHARS]}"})
            continue
        if forecast is None:
            skips.append({"event_id": event_id, "variant": variant, "reason": "no forecast (no road-1 output or none variant)"})
        else:
            out.append(forecast)
    return out


@dataclass
class Prepared:
    """Readings and event bookkeeping of one run, shared by the final stage and the ablation runs."""

    readings: dict
    day0_index: dict
    day0_date: dict
    kinds: dict
    split: dict
    dev_ids: list
    test_ids: list
    diagnostics: dict


def prepare_readings(cfg: Config, rm: RealMarket, events: pd.DataFrame, reader_tape: pd.DataFrame, mode: dict) -> Prepared:
    """The readings of the ok reader rows that sit on the trading grid, split into dev and test events, with the run's diagnostics."""
    ok_reader = reader_tape[reader_tape[STATUS] == Status.OK.value]
    readings = readings_from_tape(ok_reader, cfg, include_salvaged=bool(mode["include_salvaged"]))
    if mode["framework_confidence"] == CONFIDENCE_FROM_CERTAINTY:
        absent = sorted(e for e, r in readings.items() if math.isnan(r.certainty))
        if absent:
            raise ValueError(f"confidence is taken from the reader's certainty but {len(absent)} events have none (first: {absent[:EXAMPLES_SHOWN]}); "
                             f"use a reader mode whose framework_confidence is 'none', or the labelled fill mode")
    day0_index = dict(zip(events["event_id"], rm.sessions.get_indexer(pd.DatetimeIndex(events["session_day0"])), strict=True))
    on_grid = {e for e, i in day0_index.items() if i >= 1}
    readings = {e: r for e, r in readings.items() if e in on_grid}
    day0_date = {e: pd.Timestamp(d) for e, d in zip(events["event_id"], events["session_day0"], strict=True) if e in readings}
    kinds = dict(zip(events["event_id"], events["kind"], strict=True))
    split = dict(zip(events["event_id"], events["split"], strict=True))
    dev_ids = [e for e in readings if split[e] == DEV]
    test_ids = [e for e in readings if split[e] == TEST]
    diagnostics = {"reader_mode": mode, "salvaged_readings": int(sum(r.salvaged for r in readings.values())),
                   "events": int(len(events)), "reader_rows": int(len(reader_tape)), "reader_ok": int(len(ok_reader)),
                   "readings_with_pins": len(readings), "dev_events": len(dev_ids), "test_events": len(test_ids),
                   "reader_status": {str(k): int(v) for k, v in reader_tape[STATUS].value_counts().items()}}
    return Prepared(readings, day0_index, day0_date, kinds, split, dev_ids, test_ids, diagnostics)


def build_context(cfg: Config, rm: RealMarket, game: GameInputs, prepared: Prepared, dev: DevFit, dials: dict, persona_tape: pd.DataFrame | None,
                  skeletons: dict[str, Skeleton], mode: dict) -> CellContext:
    """The callables and fitted objects the forecasts read: road 1 from ``dev``, the persona answers, the day-0 moves, the configured weights."""
    persona_type = {pid: s.persona_type for pid, s in skeletons.items()}
    retail_ids = [pid for pid, s in skeletons.items() if s.persona_type == RETAIL]
    answers = (persona_answers(persona_tape[persona_tape[STATUS] == Status.OK.value], persona_type, prepared.day0_index, cfg)
               if persona_tape is not None and len(persona_tape) else (lambda event_id: {}))

    def day0_move(event_id: str, nodes: list[str]) -> dict[str, float]:
        d = prepared.day0_index[event_id]
        return {n: float(np.log(rm.close.iloc[d][n] / rm.close.iloc[d - 1][n])) for n in nodes
                if n in rm.close.columns and np.isfinite(rm.close.iloc[d][n]) and np.isfinite(rm.close.iloc[d - 1][n])}

    weights = cfg.get("framework.dial_weights")
    kappas = cfg.get("framework.vol_b")
    return CellContext(
        cfg=cfg, global_assets=list(rm.close.columns), readings=prepared.readings, day0_index=prepared.day0_index, day0_date=prepared.day0_date,
        source_dials=lambda e: dials[e], road1=dev.road1, month_context=game.month_context, answers=answers,
        retail_ids=retail_ids, state_z=lambda e: {}, trait_b={}, day0_move=day0_move,
        confidence=confidence_rule(cfg, str(mode["framework_confidence"])),
        dial_weights=(float(weights["w_a"]), float(weights["w_b"])),
        vol_kappas=(float(kappas["kappa_dispersion"]), float(kappas["kappa_fear"])),
    )


def fit_combination(ctx: CellContext, cfg: Config, rm: RealMarket, factors: pd.DataFrame, calendar: TradingCalendar, dev_ids: list[str],
                    variants: tuple[str, ...], skips: list[dict]) -> tuple[tuple[str, ...], dict]:
    """The dev-only combination fits: the legacy return weights of the three road variants and, when asked, the merge (Lead, Correct, Brake).

    Sets ``ctx.return_weights`` and ``ctx.merge``; returns the variants that can still be run and the fit records."""
    market = factors[["mkt_rf", "rf"]].sum(axis=1).reindex(rm.sessions)
    days = int(cfg.get("framework.return_weight_days"))
    dev_forecasts = _safe_forecasts(ctx, dev_ids, "both", skips)
    fitted, fits = {}, {}
    for variant in ("road1", "road2", "both"):
        if variant in variants and dev_forecasts:
            wa, wb, info = fit_return_weights(dev_forecasts, rm.ret, market, days, variant)
            fitted[variant] = (wa, wb, info)
            ctx.return_weights[variant] = (wa, wb)
    fits["return_weights"] = {v: {"w_a": w[0], "w_b": w[1], **w[2]} for v, w in fitted.items()}  # the legacy OLS rule, shown beside the merge
    if MERGED in variants:  # Lead, Correct, Brake: the trust is fitted on dev rows only
        names = list(cfg.get("framework.merge.trust_features"))
        triples = []
        for event_id in dev_ids:
            try:
                triple = tuple(forecast_event(ctx, event_id, v) for v in ("road1", "both", "road2"))
            except Exception as exc:  # noqa: BLE001 - one event must never stop the run; the reason is recorded
                skips.append({"event_id": event_id, "variant": MERGED, "reason": f"{type(exc).__name__}: {str(exc)[:REASON_CHARS]}"})
                continue
            if all(f is not None for f in triple):
                triples.append(triple)
        if triples:
            rows = merge_rows_from_forecasts([t[0] for t in triples], [t[1] for t in triples], [t[2] for t in triples], rm.ret, market, cfg, names)
            ctx.merge = fit_merge(rows, names, calendar, cfg)
            fits["merge"] = {"rule": ctx.merge.rule, "kept_features": ctx.merge.kept_features, "cv_loss": ctx.merge.cv_loss,
                             "within_variance": ctx.merge.within_variance, "gap_quantile": ctx.merge.gap_quantile, **ctx.merge.notes}
        else:
            variants = tuple(v for v in variants if v != MERGED)
            fits["merge"] = {"stopped": "no dev event with all three variants"}
    return variants, fits


def strategy_params(cfg: Config) -> StrategyParams:
    return StrategyParams(gamma=float(cfg.get("portfolio.default.gamma")), cost_scale=float(cfg.get("portfolio.default.cost_scale")),
                          band=float(cfg.get("portfolio.default.band")), round_trip_bp=float(cfg.get("portfolio.scoring.headline_round_trip_bp")),
                          alpha=float(cfg.get("portfolio.impact.alpha_headline")), aum_usd=float(cfg.get("portfolio.aum.headline_usd")))


def cost_grid(cfg: Config) -> dict:
    return {f"{int(bp)}bp": (lambda trade, d, bp=bp: fixed_cost(trade.to_numpy(), bp, 1.0, cfg)) for bp in cfg.get("portfolio.scoring.round_trip_bp_grid")}


def backtest_forecasts(cfg: Config, rm: RealMarket, ctx: CellContext, windows: BucketWindows, forecasts: list[EventForecast], scope: tuple, key: str,
                       delist_return: float | None, constraint: str | None = None, world_shift=None, only_days=None) -> tuple[BacktestResult, pd.DataFrame, str]:
    """One backtest of a list of forecasts (an empty list is the no-signal floor), taken from the checkpoint when it is there.

    Returns the result, the net returns by cost, and where they came from ("ran in N s" or "taken from the checkpoint")."""
    stored = ckpt.get("backtest", scope, key)
    if stored is not None:
        return stored[0], stored[1], "taken from the checkpoint"
    psi = cfg.get("framework.overnight_share") or float(cfg.get("portfolio.overnight_share_default"))
    panel = build_inputs(forecasts, len(rm.sessions), len(ctx.global_assets), float(psi), int(cfg.get("portfolio.horizon_sessions")))
    params = strategy_params(cfg)
    if constraint is not None and constraint != params.constraint:  # the second headline set (config portfolio.constraints.<name>); None = the old default, unchanged
        params = replace(params, constraint=constraint)
    strategy = FrameworkStrategy(cfg, ctx.global_assets, panel, forecasts, rm.universe(), params, discount_factor(cfg), world_shift=world_shift)
    decision_dates = [d for d in rm.sessions if windows.test_start <= d <= windows.test_end and (only_days is None or d in only_days)]  # only_days: a smoke test's short span
    began = time.time()
    result = scale_to_risk_target(run_backtest(rm.market, decision_dates, strategy, delist_return=delist_return), float(cfg.get("portfolio.vol_match.target_annual_vol")),
                                  int(cfg.get("portfolio.annualisation_days")))
    stored = (result, net_returns(result, cost_grid(cfg)))
    ckpt.put("backtest", stored, scope, key)
    return stored[0], stored[1], f"ran in {time.time() - began:.0f} s"


def run_framework_cell(
    cfg: Config, rm: RealMarket, factors: pd.DataFrame, events: pd.DataFrame, reader_tape: pd.DataFrame,
    persona_tape: pd.DataFrame | None, skeletons: dict[str, Skeleton], game: GameInputs, links: pd.DataFrame,
    windows: BucketWindows, calendar: TradingCalendar, arm: str, variants: tuple[str, ...] = (*VARIANTS, MERGED), workers: int = 1,
    reader_mode: dict | None = None, dev_fit: tuple | None = None, stop_after_dev: bool = False, after_dev: Callable | None = None,
) -> FrameworkResult:
    """``events``: finalized events with ``event_id, kind, split, session_day0``. Tapes are the finished reader and persona tapes.

    ``reader_mode`` is the record of ``finflock.reader.salvage.config_for_mode``: whether salvaged readings are read and how the
    confidence multiplier is formed (``reader_certainty``: the model's certainty, which must exist on every event; ``none``: 1 for every
    event, so no certainty is used anywhere). The default is the original design.

    ``dev_fit``: the (DevFit, source dials) of an earlier run, used instead of fitting the dev events again. ``stop_after_dev``: return right after
    the dev fit (the sheaf and the dial maps) with that fit in ``FrameworkResult.dev_fit``; ``game`` may then be None.
    """
    links = links.assign(**{c: pd.to_datetime(links[c]) for c in (LINK_FROM, LINK_TO)})  # parsed once, not on every graph build
    mode = reader_mode or {"mode": "original", "include_salvaged": False, "framework_confidence": CONFIDENCE_FROM_CERTAINTY}
    prepared = prepare_readings(cfg, rm, events, reader_tape, mode)
    diagnostics, test_ids = prepared.diagnostics, prepared.test_ids

    dev, dials = dev_fit if dev_fit is not None else fit_dev(cfg, panels_of(rm, factors, cfg), rm.snapshots, prepared.readings, prepared.kinds, prepared.day0_date,
                                                             prepared.dev_ids, links, calendar, arm, workers)
    fits = {"dial_maps": sorted(f"{k[0]}/{k[1]}" for k in dev.maps), "n_graphs": dev.n_graphs, **dev.notes}
    if dev.road1 is None or stop_after_dev:
        extra: dict = {}
        if stop_after_dev and dev.road1 is not None and dev_fit is None:
            if after_dev is not None:
                after_dev(dev, dials)  # the dev fit is stored first: nothing below can lose it
            try:  # the road-1 arm's out-of-sample score on the test events, same table as the graph and matrices (finflock.real.framework_fit.sheaf_test_scores)
                table = sheaf_test_scores(cfg, panels_of(rm, factors, cfg), prepared.readings, prepared.day0_date, prepared.dev_ids, test_ids, dev, dials, links)
                extra["appendix_sheaf_test_scores"] = table
                print("[sheaf test score]\n" + table.drop(columns=["test_graphs", "dev_graphs"]).to_string(index=False), flush=True)
            except Exception as exc:  # noqa: BLE001 - reported, never a lost fit
                fits["sheaf_test_error"] = f"{type(exc).__name__}: {str(exc)[:ERROR_TEXT_CHARS]}"
                print(f"[sheaf test score] FAILED, the dev fit is stored anyway: {fits['sheaf_test_error']}", flush=True)
        return FrameworkResult({}, {}, {}, pd.DataFrame(), pd.DataFrame(), fits, diagnostics, dev_fit=(dev, dials), extra=extra)

    ctx = build_context(cfg, rm, game, prepared, dev, dials, persona_tape, skeletons, mode)
    skips: list[dict] = []
    variants, combination = fit_combination(ctx, cfg, rm, factors, calendar, prepared.dev_ids, variants, skips)
    fits.update(combination)

    delist = float(cfg.get("portfolio.delisting.default_return"))
    forecasts, backtests, net, rows = {}, {}, {}, []
    scope = (cfg.hash, hashlib.sha256("|".join(test_ids).encode("utf-8")).hexdigest(), str(mode.get("mode")))
    for variant in variants:
        began = time.time()
        stored = ckpt.get("forecasts", scope, variant)
        if stored is None:
            before = len(skips)
            fs = _safe_forecasts(ctx, test_ids, variant, skips)
            ckpt.put("forecasts", (fs, skips[before:]), scope, variant)
            print(f"[optimiser] forecasts {variant}: {len(fs)} events forecast in {time.time() - began:.0f} s", flush=True)
        else:
            fs = stored[0]
            skips.extend(stored[1])
            print(f"[optimiser] forecasts {variant}: {len(fs)} events taken from the checkpoint", flush=True)
        forecasts[variant] = fs
        diagnostics[f"forecasts_{variant}"] = len(fs)
        if not fs:
            continue
        for treatment, value in (("delist_0", None), ("delist_default", delist)):
            key = f"{variant}/{treatment}"
            backtests[key], net[key], origin = backtest_forecasts(cfg, rm, ctx, windows, fs, scope, key, value)
            rows.append(summary_row({"variant": variant, "delisting": treatment}, backtests[key], net[key], cfg, factors))
            print(f"[optimiser] backtest {key} {origin}: " + ", ".join(f"{k}={v:.4f}" for k, v in rows[-1].items() if isinstance(v, float)), flush=True)
    skipped = pd.DataFrame(skips, columns=["event_id", "variant", "reason"])
    diagnostics["skip_reasons"] = dict(Counter(skipped["reason"].str.slice(0, GROUP_CHARS)).most_common(TOP_REASONS)) if len(skipped) else {}
    diagnostics["months_without_game"] = {str(k): v for k, v in game.skipped.items()}
    extra: dict = {}
    try:
        extra = appendix_tables(cfg, rm, forecasts, backtests, net)
        print(f"[appendix] built {len(extra)} appendix table(s): {sorted(extra)}", flush=True)
    except Exception as exc:  # noqa: BLE001 - the headline results are already computed; a failing appendix table is reported, never fatal
        diagnostics["appendix_error"] = f"{type(exc).__name__}: {str(exc)[:REASON_CHARS]}"
        print(f"[appendix] WARNING: appendix tables not built ({diagnostics['appendix_error']})", flush=True)
    return FrameworkResult(forecasts, backtests, net, pd.DataFrame(rows), skipped, fits, diagnostics, extra=extra)
