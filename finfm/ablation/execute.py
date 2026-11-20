"""One ablation run from the world to its folder: inputs changed by the switch, what must be refitted refitted on dev, the test forecasts, Tier A, Tier B.

Everything a run writes goes to ``<out>/runs/<run id>/``: ``events.parquet`` (Tier A, one row per event), ``summary.csv`` (Tier B, one row), ``net.parquet``
(daily net returns by cost and the gross), ``lineage.json`` (what the run was built from, seconds per step, checks) and ``done.json``. A run that has
``done.json`` is not run again. Numbers only; no verdict is computed here.
"""

from __future__ import annotations

import copy
import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from finfm.ablation import tier_a, transforms
from finfm.ablation.plan import Plan, RunSpec
from finfm.common import ckpt
from finfm.common.config import Config
from finfm.common.hashing import stable_hash
from finfm.framework.road1_stage import Road1Predictor
from finfm.framework.run import MERGED, CellContext
from finfm.mfg.precompute import FundInput
from finfm.real.framework_cell import (
    Prepared,
    _safe_forecasts,
    backtest_forecasts,
    build_context,
    fit_combination,
    prepare_readings,
)
from finfm.real.framework_fit import DevFit, fit_dev, panels_of, source_dials_of
from finfm.real.game_inputs import GameInputs, make_game_inputs
from finfm.real.summary import summary_row
from finfm.real.world import World
from finfm.road1.graph import LINK_FROM, LINK_TO

DONE_FILE, LINEAGE_FILE, EVENTS_FILE, SUMMARY_FILE, NET_FILE = "done.json", "lineage.json", "events.parquet", "summary.csv", "net.parquet"
NONE_VARIANT = "none"
REASON_CHARS = 80  # skip reasons are grouped by their first characters
TOP_REASONS = 10  # and the most common are kept
SCHEMA = 1


@dataclass
class AblationWorld:
    """The world of one notebook: loaded once, with the Full pieces (dev fit, combination, linked sets) cached for the runs that keep them."""

    world: World
    plan: Plan
    dev_full: tuple[DevFit, dict]
    workers: int = 1
    prepared: Prepared | None = None
    panels: object = None
    game_full: GameInputs | None = None
    ctx_full: CellContext | None = None
    combination: tuple[tuple, dict, object, dict] | None = None  # (variants, fits, merge, return_weights)
    linked: dict[str, set[int]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        w = self.world
        w.links = w.links.assign(**{c: pd.to_datetime(w.links[c]) for c in (LINK_FROM, LINK_TO)})
        self.prepared = prepare_readings(w.cfg, w.rm, w.events, w.reader_tape, w.mode)
        self.panels = panels_of(w.rm, w.factors, w.cfg)
        self.game_full = make_game_inputs(w.cfg, w.rm, w.skeletons, w.managers)

    @property
    def market(self) -> pd.Series:
        return self.world.factors[["mkt_rf", "rf"]].sum(axis=1).reindex(self.world.rm.sessions)

    def full_context(self) -> CellContext:
        if self.ctx_full is None:
            w = self.world
            self.ctx_full = build_context(w.cfg, w.rm, self.game_full, self.prepared, self.dev_full[0], self.dev_full[1], w.persona_tape, w.skeletons, w.mode)
        return self.ctx_full

    def full_combination(self) -> tuple[tuple, dict, object, dict]:
        """Return weights and merge of Full, fitted on dev once per notebook (checkpointed): the 'fitted scalars of Full' that fixed-mode runs keep."""
        if self.combination is None:
            ctx, w = self.full_context(), self.world
            scope = (w.cfg.hash, stable_hash(sorted(self.prepared.dev_ids)), w.mode_name)
            stored = ckpt.get("combination", scope)
            if stored is None:
                began = time.time()
                variants, fits = fit_combination(ctx, w.cfg, w.rm, w.factors, w.calendar, self.prepared.dev_ids, (*ctx_variants(), MERGED), [])
                stored = (variants, fits, ctx.merge, dict(ctx.return_weights))
                ckpt.put("combination", stored, scope)
                print(f"[ablation] Full's return weights and merge fitted on dev in {time.time() - began:.0f} s", flush=True)
            else:
                print("[ablation] Full's return weights and merge taken from the checkpoint", flush=True)
            self.combination = stored
        return self.combination

    def linked_sets(self) -> dict[str, set[int]]:
        """Per test event, the assets in its Full subgraph that are not pinned (independent of the run, so the linked-stock fit compares like with like)."""
        if not self.linked:
            ctx, w = self.full_context(), self.world
            index = {a: i for i, a in enumerate(ctx.global_assets)}
            for event_id in self.prepared.test_ids:
                reading = ctx.readings[event_id]
                graph = ctx.road1.graph_for(reading, ctx.source_dials(event_id), ctx.day0_date[event_id])
                self.linked[event_id] = {index[n] for n in graph.nodes[graph.n_pinned:] if n in index}
        return self.linked


def ctx_variants() -> tuple[str, ...]:
    return ("road1", "road2", "both")


def _hash_ids(ids: list[str]) -> str:
    return hashlib.sha256("|".join(ids).encode("utf-8")).hexdigest()


def _kept_skeletons(effects: dict, spec: RunSpec, world: World, cfg: Config) -> dict:
    skeletons = world.skeletons
    if "drop_type" in effects:
        skeletons = transforms.drop_type(skeletons, effects["drop_type"])
    if spec.fraction is not None:
        skeletons = transforms.keep_fraction(skeletons, spec.fraction, (str(cfg.get("ablation.persona_count.seed_label")), world.bucket, world.llm))
    return skeletons


def execute_run(aw: AblationWorld, spec: RunSpec, out_root: Path) -> dict:
    """Runs ``spec`` and writes its folder; returns its lineage record. Raises only on a bug: a failing event is skipped and counted inside."""
    folder = out_root / str(aw.world.cfg.get("ablation.outputs.runs_subfolder")) / spec.run_id
    if (folder / DONE_FILE).exists():
        print(f"[ablation] {spec.run_id}: already done (done.json found), skipped", flush=True)
        return json.loads((folder / LINEAGE_FILE).read_text(encoding="utf-8"))
    folder.mkdir(parents=True, exist_ok=True)
    began, steps = time.time(), {}
    w, cfg_full = aw.world, aw.world.cfg
    effects = copy.deepcopy(spec.effects)
    overrides = dict(effects.get("config", {}))
    cfg = cfg_full.with_overrides(overrides) if overrides else cfg_full
    record = {"schema": SCHEMA, "run_id": spec.run_id, "key": spec.key, "name": spec.name, "kind": spec.kind, "tier": spec.tier, "mode": spec.mode,
              "refit": list(spec.refit), "effects": effects, "draw": spec.draw, "config_hash": cfg.hash, "full_config_hash": cfg_full.hash,
              "bucket": w.bucket, "llm": w.llm, "reader_mode": w.mode_name, "label": str(cfg_full.get("ablation.labels.in_training"))}
    print(f"[ablation] ===== {spec.run_id}  ({spec.name}; tier {spec.tier}; mode {spec.mode}; refit {list(spec.refit) or 'none'}) =====", flush=True)

    ablated = {k: v for k, v in effects.items() if k not in ("config", "variant", "check")}
    if spec.kind == "switch" and not ablated and cfg.hash == cfg_full.hash and "variant" not in effects:
        record.update({"identical_to_full": True, "note": str(cfg_full.get("ablation.labels.identical"))})
        _write(folder, record, None, None, None, began)
        print(f"[ablation] {spec.run_id}: the switch leaves the configuration unchanged ({record['note']}); nothing to run", flush=True)
        return record

    # ---- inputs changed by the switch ----------------------------------------------------------------------------------------------
    prepared, links = aw.prepared, w.links
    readings = prepared.readings
    stamp = time.time()
    notes: dict = {}
    if effects.get("links") == "sp500_only":
        readings, links, removed = transforms.sp500_only(readings, links, set(w.rm.close.columns))
        notes["events_removed_no_sp_pin"] = removed
    if effects.get("readings") == "blind":
        readings = transforms.text_blind(readings, prepared.kinds, [e for e in prepared.dev_ids if e in readings], cfg)
    elif effects.get("readings") == "shuffle":
        readings, shuffle_notes = transforms.text_shuffled(readings, prepared.kinds, prepared.day0_date, cfg, (*spec.identity, w.bucket, w.llm))
        notes.update(shuffle_notes)
    if effects.get("links") == "shuffle":
        links = transforms.links_shuffled(links, (*spec.identity, w.bucket, w.llm, "links"))
    dev_ids = [e for e in prepared.dev_ids if e in readings]
    test_ids = [e for e in prepared.test_ids if e in readings]
    day0_date = {e: d for e, d in prepared.day0_date.items() if e in readings}
    run_prepared = Prepared(readings, prepared.day0_index, day0_date, prepared.kinds, prepared.split, dev_ids, test_ids, prepared.diagnostics)

    # ---- road 1 -------------------------------------------------------------------------------------------------------------------
    dev_full, dials_full = aw.dev_full
    arm = spec.arm or str(cfg.get("framework.road1_arm_legacy"))
    if "road1" in spec.refit:
        fixed = None
        if str(cfg_full.get("ablation.refit.road1")) == "fixed_config" and arm == dev_full.road1.arm:
            fixed = {"grounding": dev_full.notes["grounding"], "penalty_mu": dev_full.notes["penalty_mu"]}
        seed = (spec.key,) if spec.kind == "noise" else ()
        dev, dials = fit_dev(cfg, aw.panels, w.rm.snapshots, readings, prepared.kinds, day0_date, dev_ids, links, w.calendar, arm, aw.workers, fixed, seed)
        if dev.road1 is None:
            raise RuntimeError(f"{spec.run_id}: the road-1 refit produced no model: {dev.notes.get('stopped')}")
        notes["road1_refit"] = {k: dev.notes[k] for k in ("grounding", "penalty_mu", "dev_cv_loss", "best_steps", "ridge", "matrix_rung") if k in dev.notes}
    else:
        dev = dev_full
        dials = dials_full if readings is prepared.readings else source_dials_of(cfg, aw.panels, w.rm.snapshots, readings, prepared.kinds, day0_date, dev_full.maps)
        if links is not w.links:
            old = dev_full.road1
            dev = DevFit(dev_full.maps, Road1Predictor(old.arm, old.model, old.grounding, old.scale, links, cfg), dev_full.n_dev_events, dev_full.n_graphs,
                         dev_full.notes)
    steps["inputs_and_road1_s"] = round(time.time() - stamp, 1)

    # ---- game and persona answers --------------------------------------------------------------------------------------------------
    skeletons = _kept_skeletons(effects, spec, w, cfg)
    game = aw.game_full
    if overrides.keys() & {k for k in overrides if k.startswith("game.")} or "game" in effects or len(skeletons) != len(w.skeletons):
        game = make_game_inputs(cfg, w.rm, skeletons, w.managers)
        game.fund_transform = transforms.equal_books if effects.get("game") == "equal_books" else None
    ctx = build_context(cfg, w.rm, game, run_prepared, dev, dials, w.persona_tape, w.skeletons, w.mode)
    if len(skeletons) != len(w.skeletons):
        kept = set(skeletons)
        ctx.answers = transforms.drop_answers(ctx.answers, kept)
        ctx.retail_ids = [p for p in ctx.retail_ids if p in kept]
    answers = effects.get("answers")
    pinned_of = lambda e: [p["node_id"] for p in ctx.readings[e].pins]  # noqa: E731
    if answers == "common_belief":
        ctx.answers = transforms.common_belief(ctx.answers)
    elif answers == "day0_move":
        ctx.answers = transforms.day0_values(ctx.answers, ctx.day0_move, pinned_of)
    elif answers == "shuffle_cards":
        ctx.answers = transforms.cards_shuffled(ctx.answers, (*spec.identity, w.bucket, w.llm, "cards"))
    if effects.get("consensus") == "day0_move":
        ctx.consensus_override = ctx.day0_move
    if effects.get("ripple") == "off":
        ctx.road1 = transforms.RippleOff(ctx.road1, tuple(cfg.get("road1.dial_names")))
    ctx.merge_rule, ctx.brake = effects.get("merge_rule"), effects.get("brake")
    variant = effects.get("variant", str(cfg.get("ablation.full.variant")))

    # ---- what is fitted on dev -----------------------------------------------------------------------------------------------------
    stamp = time.time()
    base_variants, base_fits, base_merge, base_weights = aw.full_combination()
    refit_merge = "merge" in spec.refit or spec.mode == "refit"
    fits = {}
    if refit_merge:
        needed = (MERGED,) if variant == MERGED else (variant,)
        _, fits = fit_combination(ctx, cfg, w.rm, w.factors, w.calendar, dev_ids, (*ctx_variants(), *needed) if variant == MERGED else needed, [])
    else:
        ctx.merge, ctx.return_weights = base_merge, dict(base_weights)
        fits = {"kept_from_full": True}
    steps["combination_s"] = round(time.time() - stamp, 1)

    # ---- test forecasts ------------------------------------------------------------------------------------------------------------
    stamp = time.time()
    skips: list[dict] = []
    scope = (cfg.hash, spec.run_id, _hash_ids(test_ids), w.mode_name, SCHEMA)
    stored = ckpt.get("ablation_forecasts", scope, variant)
    if stored is None:
        forecasts = [] if variant == NONE_VARIANT else _safe_forecasts(ctx, test_ids, variant, skips)
        ckpt.put("ablation_forecasts", (forecasts, skips), scope, variant)
    else:
        forecasts, skips = stored
        print(f"[ablation] {spec.run_id}: {len(forecasts)} forecasts taken from the checkpoint", flush=True)
    steps["forecasts_s"] = round(time.time() - stamp, 1)
    print(f"[ablation] {spec.run_id}: {len(forecasts)} test events forecast, {len(skips)} skipped in {steps['forecasts_s']} s", flush=True)

    # ---- Tier A --------------------------------------------------------------------------------------------------------------------
    events_frame = pd.DataFrame()
    if forecasts:
        stamp = time.time()
        table = tier_a.event_table(forecasts, ctx.readings, prepared.kinds, w.rm.ret, aw.market, aw.panels.cap_proxy, aw.linked_sets(), cfg)
        events_frame = tier_a.per_event(table, cfg)
        steps["tier_a_s"] = round(time.time() - stamp, 1)
        print(f"[ablation] {spec.run_id}: tier A " + ", ".join(f"{k}={v:.4f}" for k, v in tier_a.summarise(events_frame, cfg).items() if k.startswith(("rank_ic", "linked_r2"))), flush=True)
    if effects.get("check") == "road2_zero" and forecasts:
        record["check_road2_zero_max_abs"] = float(max(np.abs(f.inc_b).max() for f in forecasts if f.inc_b is not None))
        print(f"[ablation] {spec.run_id}: largest absolute road-2 increment = {record['check_road2_zero_max_abs']:.3e} (an engineering check; it should be zero up to numerical error)", flush=True)

    # ---- Tier B --------------------------------------------------------------------------------------------------------------------
    summary, net = None, None
    if spec.tier == "B":
        stamp = time.time()
        treatment = str(cfg_full.get("ablation.treatment"))
        key = f"{variant}/{treatment}"
        delist = None if treatment == "delist_0" else float(cfg.get("portfolio.delisting.default_return"))
        result, net_frame, origin = backtest_forecasts(cfg, w.rm, ctx, w.windows, forecasts, scope, key, delist)
        summary = summary_row({"run_id": spec.run_id, "variant": variant, "delisting": treatment}, result, net_frame, cfg, w.factors)
        net = net_frame.assign(gross=result.gross)
        steps["backtest_s"] = round(time.time() - stamp, 1)
        print(f"[ablation] {spec.run_id}: backtest {origin}: " + ", ".join(f"{k}={v:.4f}" for k, v in summary.items() if k in ("sharpe_gross", "sharpe_net_headline", "cum_net_return")), flush=True)
    record.update({"notes": notes, "fits": _plain(fits), "n_test_events": len(test_ids), "n_forecasts": len(forecasts), "n_skipped": len(skips),
                   "skip_reasons": pd.Series([s["reason"][:REASON_CHARS] for s in skips]).value_counts().head(TOP_REASONS).to_dict() if skips else {}, "seconds_by_step": steps})
    _write(folder, record, events_frame, summary, net, began)
    return record


def _plain(value):
    """JSON-safe copy of a fit record (numpy arrays and scalars become lists and numbers; objects are named)."""
    if isinstance(value, dict):
        return {str(k): _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    return value if isinstance(value, (str, int, float, bool)) or value is None else type(value).__name__


def _write(folder: Path, record: dict, events: pd.DataFrame | None, summary: dict | None, net: pd.DataFrame | None, began: float) -> None:
    if events is not None and len(events):
        events.to_parquet(folder / EVENTS_FILE, index=False)
    if summary is not None:
        pd.DataFrame([summary]).to_csv(folder / SUMMARY_FILE, index=False)
    if net is not None:
        net.to_parquet(folder / NET_FILE)
    record["seconds"] = round(time.time() - began, 1)
    (folder / LINEAGE_FILE).write_text(json.dumps(record, indent=1, default=str), encoding="utf-8")
    (folder / DONE_FILE).write_text(json.dumps({"run_id": record["run_id"], "seconds": record["seconds"]}), encoding="utf-8")
    print(f"[ablation] {record['run_id']}: finished in {record['seconds']} s -> {folder}", flush=True)
