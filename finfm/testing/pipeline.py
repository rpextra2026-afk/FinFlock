"""The whole experiment on a synthetic world with a mock LLM, on CPU (spec 12.2 #17).

Runs the real worker, bundles, reader, personas, fits, game, optimiser and backtest. All numbers it
produces carry ``synthetic=True`` and must never reach a report built from real data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from finfm.common.config import Config
from finfm.common.schema import AVAILABLE_DATE, FAMILY, KIND, PINS_JSON, STATUS, TEXT, TITLE, UNIT_ID, Status
from finfm.common.windows import bucket_windows
from finfm.compute.bundle import make_job_id, read_consolidated, scan_bundles
from finfm.compute.engine import MockEngine
from finfm.compute.worker import run_shard
from finfm.data.buckets.contract import finalize_events
from finfm.data.labels import build_labels
from finfm.fit.road1_fit import training_graphs
from finfm.framework.assemble import persona_answers
from finfm.framework.events import pin_frame, readings_from_tape
from finfm.framework.road1_stage import Road1Predictor
from finfm.framework.road2_stage import PersonaEventAnswer
from finfm.framework.run import (
    CellContext,
    EventForecast,
    FrameworkStrategy,
    MonthContext,
    StrategyParams,
    build_inputs,
    forecast_event,
)
from finfm.mfg.precompute import amihud_liquidity, build_month_game, estimate_covariance
from finfm.personas.bios import template_bio
from finfm.personas.careers import load_career_grid
from finfm.personas.mock import persona_responder
from finfm.personas.run import EventPayload, PersonaProcessor, PersonaState, unit_id
from finfm.personas.skeletons import Skeleton, retail_skeletons
from finfm.personas.state import portfolio_state, retail_history, window_return
from finfm.portfolio.execute import net_returns, run_backtest
from finfm.portfolio.costs import fixed_cost
from finfm.portfolio.market import Market
from finfm.reader.dials import fit_source_dial_maps, source_dials
from finfm.reader.state_features import add_state_features
from finfm.reader.masking import MaskerCache
from finfm.reader.mock import reader_responder
from finfm.reader.prompt import WhitespaceTokenizer
from finfm.reader.run import ReaderProcessor
from finfm.road1.fit import dial_scale, select_and_fit, standardise
from finfm.testing.synthetic import World

JOB_READER = "reader"
JOB_PERSONAS = "personas"
SECTOR_DOMAIN = {"Energy": "energy", "Information Technology": "technology", "Financials": "banking"}


@dataclass
class PipelineResult:
    synthetic: bool
    events: pd.DataFrame
    reader_tape: pd.DataFrame
    persona_tape: pd.DataFrame
    forecasts: dict[str, list[EventForecast]]
    backtests: dict[str, object]
    net: dict[str, pd.DataFrame]
    context: CellContext
    market: Market
    notes: dict = field(default_factory=dict)


def _run_job(cfg: Config, job: str, frame: pd.DataFrame, processor, out_dir: Path, clock) -> pd.DataFrame:
    result = run_shard(cfg=cfg, job_id=make_job_id(job, "tessera", "llm2"), shard_id="s0000", planned=frame,
                       processor=processor, input_dirs=[], out_dir=out_dir, meta={"synthetic": True}, clock=clock)
    assert result.n_remaining == 0, f"{job} did not finish: {result.outcome}"
    scan = scan_bundles([out_dir], make_job_id(job, "tessera", "llm2"))
    return read_consolidated(scan.valid)


def persona_skeletons(world: World, cfg: Config) -> list[Skeleton]:
    from finfm.personas.skeletons import sector_tilt_of

    skeletons: list[Skeleton] = []
    for fund in world.funds:
        weights = pd.Series(fund.weights, index=world.securities)
        weights = weights / weights.sum()
        tilt = sector_tilt_of(weights, world.sector)
        skeletons.append(Skeleton(
            fund.name, fund.group, "2019Q4", "medium", fund.book, fund.turnover_pctile, "medium", float(len(weights)),
            tilt, dict(weights), None, "high", "general", float(cfg.get(f"game.types.{fund.group}.mandate")), 0.0,
            "medium", "New York metro", "13f"))
    grid = load_career_grid(str(Path(__file__).resolve().parents[2] / "config" / "career_grid.csv"))
    universe = pd.DataFrame({"security_id": world.securities, "sector": world.sector.to_numpy(),
                             "market_cap": 1.0e10, "uncovered_share": world.uncovered.to_numpy()})
    retail = retail_skeletons(cfg, grid, universe, SECTOR_DOMAIN, "2019Q4", int(cfg.get("runtime.base_seed")))
    return skeletons + retail[:4]


def run_pipeline(world: World, work_dir: Path, clock, variants=("road1", "road2", "both", "none"), arm="wgraph") -> PipelineResult:
    cfg = world.cfg
    work_dir.mkdir(parents=True, exist_ok=True)
    windows = bucket_windows(cfg, "tessera", world.calendar)
    events, _ = finalize_events(world.events, world.calendar, windows, cfg)
    events = events.merge(world.events[["event_id", "title", "text", "pins_json", "n_pins", "day0_index"]],
                          on="event_id", suffixes=("_x", ""))
    events["day0_index"] = world.sessions.get_indexer(pd.DatetimeIndex(events["session_day0"]))
    day0_date = dict(zip(events["event_id"], pd.to_datetime(events["session_day0"]), strict=True))
    day0_index = dict(zip(events["event_id"], events["day0_index"], strict=True))

    # --- reader job (real worker, mock LLM) -----------------------------------------------------
    engine = MockEngine()
    engine.register("reader", reader_responder)
    maskers = MaskerCache(world.aliases, label_prefix=str(cfg.get("reader.masked_company_prefix")),
                          short_ticker_max_len=int(cfg.get("reader.short_ticker_max_len")))
    batch = pd.DataFrame({
        UNIT_ID: events["event_id"], KIND: events["kind"], FAMILY: events["family"],
        AVAILABLE_DATE: pd.to_datetime(events["session_day0"]).dt.strftime("%Y-%m-%d"),
        TITLE: events["title"], TEXT: events["text"], PINS_JSON: events["pins_json"],
    })
    reader_tape = _run_job(cfg, JOB_READER, batch, ReaderProcessor(cfg, engine, maskers, WhitespaceTokenizer()),
                           work_dir / "reader", clock)

    # --- persona job -------------------------------------------------------------------------------
    skeletons = persona_skeletons(world, cfg)
    by_id = {s.persona_id: s for s in skeletons}
    bios = {s.bio_id: template_bio(cfg, s) for s in skeletons}
    descriptors = {s: {"node_type": "company", "sector": world.sector[s], "layer": "mid"} for s in world.securities}
    from finfm.personas.feed import NewsItem
    payloads, states = {}, {}
    market_ret = world.ret.mean(axis=1)
    for row in events.itertuples():
        stamp = pd.Timestamp(row.available_utc)
        item = NewsItem(f"item:{row.event_id}", row.kind, stamp, row.title, row.text, ())
        pins = tuple(__import__("json").loads(row.pins_json))
        payloads[row.event_id] = EventPayload(row.event_id, row.kind, stamp, (item,), (), pins, (), (), descriptors,
                                              frozenset(world.securities))
        end = world.sessions[max(row.day0_index - 1, 0)]
        wr = window_return(world.ret, end, int(cfg.get("personas.state_window_sessions")))
        market = float(wr.mean())
        for s in skeletons:
            st = portfolio_state(pd.Series(s.holdings), wr, market)
            states[(row.event_id, s.persona_id)] = PersonaState(st["state"], st["state_rel"], market,
                                                               retail_history(cfg, "low")[0], {})
    engine2 = MockEngine()
    engine2.register("persona", persona_responder)
    units = pd.DataFrame({UNIT_ID: [unit_id(e, s.persona_id) for e in events["event_id"] for s in skeletons]})
    persona_tape = _run_job(cfg, JOB_PERSONAS, units,
                            PersonaProcessor(cfg, engine2, by_id, bios, payloads, states, WhitespaceTokenizer()),
                            work_dir / "personas", clock)

    # --- fits on dev only (labels enter here, as data) ----------------------------------------------
    readings = readings_from_tape(reader_tape[reader_tape[STATUS] == Status.OK.value], cfg)
    kinds = dict(zip(events["event_id"], events["kind"], strict=True))
    node_class = {s: "mid" for s in world.securities}
    pins = pin_frame(readings, node_class, kinds, cfg)
    pins["node_class"] = "mid"
    pins["day0"] = pins["event_id"].map(day0_date)
    pins = add_state_features(pins, world.ret, world.close * world.volume, cfg)
    split = dict(zip(events["event_id"], events["split"], strict=True))
    prices = {s: pd.DataFrame({"ret": world.ret[s], "open": world.open_[s], "high": world.high[s], "low": world.low[s],
                               "close_raw": world.close[s], "volume_raw": world.volume[s]}) for s in world.securities}
    sector_peers = {sec: [s for s in world.securities if world.sector[s] == sec] for sec in set(world.sector)}
    dev_ids = [e for e in readings if split[e] == "dev"]
    label_events = pd.DataFrame([
        {"event_id": e, "security_id": s, "day0": day0_date[e]}
        for e in dev_ids for s in sector_peers[world.sector[readings[e].pins[0]["node_id"]]]
    ])
    labels = build_labels(label_events, prices, world.factors, cfg)
    pin_labels = labels[labels["window"] == "d01"].rename(columns={"security_id": "node_id"})[
        ["event_id", "node_id", "log_vol_ratio", "log_dv_ratio"]]
    fit_rows = pins.merge(pin_labels, on=["event_id", "node_id"], how="left")
    fit_rows["day0"] = fit_rows["event_id"].map(day0_date)
    fit_rows = fit_rows[fit_rows["event_id"].isin(dev_ids)]
    maps = fit_source_dial_maps(fit_rows, world.calendar, cfg)
    dials_by_event = {e: source_dials(pins[pins["event_id"] == e], maps, cfg) for e in readings}
    graphs = training_graphs({e: readings[e] for e in dev_ids}, dials_by_event, world.links, day0_date, labels, cfg)
    scale = dial_scale(graphs, float(cfg.get("road1.dial_scale_floor")))
    dev_dates = [day0_date[g.event_id] for g in graphs]
    model, best, _ = select_and_fit(arm, standardise(graphs, scale), dev_dates, world.calendar, cfg, seed_parts=("synthetic",))
    road1 = Road1Predictor(arm, model, best.grounding, scale, world.links, cfg)

    # --- months, game inputs, answers -------------------------------------------------------------------
    months: dict[pd.Period, MonthContext] = {}
    sleeve_panel = pd.concat([world.ret, world.sleeve_ret.rename("VXF")], axis=1)

    def month_context(month: pd.Period) -> MonthContext:
        if month in months:
            return months[month]
        start = month.start_time
        past = sleeve_panel.loc[: start - pd.Timedelta(days=1)].iloc[-int(cfg.get("game.covariance_window_days")):]
        sigma_plus = estimate_covariance(past.to_numpy(), float(cfg.get("game.covariance_ridge")))
        window = int(cfg.get("game.amihud.window_days"))
        abs_ret = world.ret.loc[: start - pd.Timedelta(days=1)].iloc[-window:].abs().to_numpy()
        dv = (world.close * world.volume).loc[: start - pd.Timedelta(days=1)].iloc[-window:].to_numpy()
        ell = amihud_liquidity(abs_ret, dv, float(cfg.get("game.liquidity_scale")),
                               float(cfg.get("game.amihud.winsor_low")), float(cfg.get("game.amihud.winsor_high")))
        game = build_month_game(cfg, world.securities, sigma_plus, ell, world.funds)
        months[month] = MonthContext(game, list(world.securities), world.uncovered.to_numpy(), dv.mean(axis=0))
        return months[month]

    persona_type = {s.persona_id: s.persona_type for s in skeletons}
    session_of = day0_index
    answers_fn = persona_answers(persona_tape[persona_tape[STATUS] == Status.OK.value], persona_type, session_of, cfg)
    retail_ids = [s.persona_id for s in skeletons if s.persona_type == "retail"]

    def day0_move(event_id: str, nodes: list[str]) -> dict[str, float]:
        d = day0_index[event_id]
        return {n: float(np.log(world.close.iloc[d][n] / world.close.iloc[d - 1][n])) for n in nodes}

    ctx = CellContext(
        cfg=cfg, global_assets=world.securities, readings=readings, day0_index=day0_index, day0_date=day0_date,
        source_dials=lambda e: dials_by_event[e], road1=road1, month_context=month_context, answers=answers_fn,
        retail_ids=retail_ids, state_z=lambda e: {}, trait_b={}, day0_move=day0_move,
        confidence=lambda certainty: float(np.clip(certainty, 0.0, 1.0)),
    )

    # --- dev fit of the return weights (w_A, w_B) ------------------------------------------------------------
    from finfm.fit.return_weights import fit_return_weights

    dev_forecasts = [f for f in (forecast_event(ctx, e, "both") for e in dev_ids) if f is not None]
    fitted = {v: fit_return_weights(dev_forecasts, world.ret, world.ret.mean(axis=1), 5, v) for v in ("road1", "road2", "both")}
    ctx.return_weights = {**ctx.return_weights, **{v: (w[0], w[1]) for v, w in fitted.items()}}

    if "merged" in variants:  # Lead, Correct, Brake: trust fitted on dev only (spec 9.1)
        from finfm.fit.return_weights import fit_merge, merge_rows_from_forecasts

        names = list(cfg.get("framework.merge.trust_features"))
        triples = [(forecast_event(ctx, e, "road1"), forecast_event(ctx, e, "both"), forecast_event(ctx, e, "road2")) for e in dev_ids]
        triples = [t for t in triples if all(f is not None for f in t)]
        rows = merge_rows_from_forecasts([t[0] for t in triples], [t[1] for t in triples], [t[2] for t in triples],
                                         world.ret, world.ret.mean(axis=1), cfg, names)
        ctx.merge = fit_merge(rows, names, world.calendar, cfg)

    # --- forecasts, optimiser, backtest ----------------------------------------------------------------------
    market = Market(world.close, world.volume, world.oo_ret)
    test_ids = [e for e in readings if split[e] == "test"]
    sessions = world.sessions
    test_start, test_end = windows.test_start, windows.test_end
    decision_dates = [d for d in sessions if test_start <= d <= test_end]
    params = StrategyParams(gamma=float(cfg.get("portfolio.default.gamma")), cost_scale=float(cfg.get("portfolio.default.cost_scale")),
                            band=float(cfg.get("portfolio.default.band")),
                            round_trip_bp=float(cfg.get("portfolio.scoring.headline_round_trip_bp")),
                            alpha=float(cfg.get("portfolio.impact.alpha_headline")),
                            aum_usd=float(cfg.get("portfolio.aum.headline_usd")))
    from finfm.mfg.precompute import discount_factor
    psi = cfg.get("framework.overnight_share") or float(cfg.get("portfolio.overnight_share_default"))
    forecasts, backtests, net = {}, {}, {}
    cost_models = {f"{int(bp)}bp": (lambda trade, d, bp=bp: fixed_cost(trade.to_numpy(), bp, 1.0, cfg))
                   for bp in cfg.get("portfolio.scoring.round_trip_bp_grid")}
    for variant in variants:
        fs = [f for f in (forecast_event(ctx, e, variant) for e in test_ids) if f is not None]
        forecasts[variant] = fs
        if not fs:
            continue
        panel = build_inputs(fs, len(sessions), len(world.securities), float(psi), int(cfg.get("portfolio.horizon_sessions")))
        strategy = FrameworkStrategy(cfg, world.securities, panel, fs, lambda d: set(world.securities), params, discount_factor(cfg))
        backtests[variant] = run_backtest(market, decision_dates, strategy, delist_return=None)
        net[variant] = net_returns(backtests[variant], cost_models)
    return PipelineResult(True, events, reader_tape, persona_tape, forecasts, backtests, net, ctx, market,
                          {"road1_arm": arm, "n_dev_graphs": len(graphs), "n_test_events": len(test_ids),
                           "return_weights": dict(ctx.return_weights),
                           "merge": None if ctx.merge is None else {"rule": ctx.merge.rule, "kept_features": ctx.merge.kept_features,
                                                                    "cv_loss": ctx.merge.cv_loss},
                           "return_weight_fit": {v: w[2] for v, w in fitted.items()}})
