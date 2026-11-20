"""The framework final (road 1 + road 2 + merge + optimiser + backtest) on the NEW cleaned tapes and the NEW pooled sheaf.

One run folder per tape. Stages, each logged and resumable through the checkpoint store (``finfm.common.ckpt``):

  load      cleaned tape -> events, readings, source dials, world-lane pins; market, links, 13F managers, persona tape
  dev_fit   the build's fitted pooled sheaf -> ``dev_fit.pkl`` (keyed by the reader mode ``reader2_clean``) with the dev-only pin maps
  freeze    FREEZE_FRAMEWORK.json must match (clean tape, config, code, persona tapes, dev_fit)
  combine   the dev-only merge fit (Lead, Correct, Brake) and the legacy return weights
  forecast  every variant on the test events
  backtest  every variant x headline set (long-only, dollar-neutral) x delisting rule, each put on the same risk target
  metrics   summary rows, paired Sharpe differences, Holm, deflated Sharpe, PBO, CPCV, rank IC, cost grid, stress, capacity, checklist, delisting trace

Numbers and intervals only. No verdict words. The run never opens the sealed window of the cleaned tape.
"""

from __future__ import annotations

import concurrent.futures as cf
import json
import multiprocessing
import os
import pickle
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pandas as pd

from finfm.build.run import build_config
from finfm.common import ckpt
from finfm.common.config import Config
from finfm.common.hashing import sha256_file, sha256_hex
from finfm.common.progress import Heartbeat
from finfm.common.schema import STATUS, Status
from finfm.common.timing import TradingCalendar
from finfm.common.windows import BucketWindows
from finfm.compute.payloads import load_skeletons
from finfm.framework import clean_adapter as CA
from finfm.framework import final_metrics as FM
from finfm.framework import freeze as FZ
from finfm.framework import calibration as CAL
from finfm.framework import road1_pooled as RP
from finfm.framework import sheaf_internals as SI
from finfm.framework import sealed as SEALED
from finfm.framework import world_lane as WL
from finfm.framework.delist_trace import trace as delist_trace
from finfm.framework import rival_adapter as RA
from finfm.framework.run import MERGED, VARIANTS
from finfm.portfolio.execute import BacktestResult
from finfm.real.appendix import appendix_tables
from finfm.real.devfit_io import DEV_FILE, DEV_FIT_IGNORED_SECTIONS, read_dev_fit
from finfm.real.framework_cell import (
    Prepared,
    _safe_forecasts,
    backtest_forecasts,
    build_context,
    fit_combination,
)
from finfm.real.framework_fit import LABEL_WINDOW, panels_of
from finfm.real.game_inputs import make_game_inputs
from finfm.real.market_load import RealMarket, load_real_market
from finfm.real.summary import summary_row
from finfm.real.tapes import TapeError, load_tape
from finfm.reader.salvage import config_for_mode
from finfm.road1.graph import LINK_FROM, LINK_TO
from finfm.stage import road1_real as rr
from finfm.stage.reader_stage import PRESTAGED

MODE = "reader2_clean"
FIT_FILE_KEY = "build.runner.fit_cache"
LOG_SECONDS_KEY = "cpu.progress_seconds"
Log = Callable[[str], None]
DECIMALS = 4  # decimals of the markdown tables
REASON_CHARS = 80  # skip reasons are grouped on this many characters
TOP_REASONS = 8
ERROR_CHARS = 160
WORLD_PARTS = 3  # a backtest key with more than variant/set/rule parts is the no-world-lane probe
HORIZON_FIVE = 5
TABLE_ROWS = 60
SHORT_HASH = 12  # characters of a hash shown in messages
SMOKE_CALENDAR_DAYS = 30  # decision days of a smoke test: this many calendar days from the first tradable day

_G: dict[str, Any] = {}  # shared with forked backtest workers (read-only)
FIGURE_TABLES: dict[str, pd.DataFrame] = {}  # filled by build_tables, written to <results>/figure_data/


@dataclass
class CleanWorld:
    cfg: Config
    base: Config
    run: str
    bucket: str
    llm: str
    run_cfg: dict[str, Any]
    rm: RealMarket
    factors: pd.DataFrame
    links: pd.DataFrame
    managers: pd.DataFrame
    skeletons: dict
    persona_tape: pd.DataFrame | None
    persona_used: list[Path]
    calendar: TradingCalendar
    windows: BucketWindows
    clean: pd.DataFrame
    ce: CA.CleanEvents
    root: Path
    rival_cutoff: pd.Timestamp | None = None  # rival slopes are fitted on scores before this day (the original hold-out start, also in the sealed stage)


def load_clean_world(root: Path, run: str, base: Config, with_personas: bool = True, sealed_clean: Path | None = None, sealed_spec: dict | None = None) -> CleanWorld:
    """Everything the framework reads for one cleaned tape. Nothing here opens the sealed rows, except in the sealed-window stage (``sealed_clean`` = the cleaned file of the longer window,
    ``sealed_spec`` = framework.sealed): there the test window is the sealed one, while the dev events keep the embargo of the original hold-out start."""
    cfg = build_config(base)
    run_cfg = dict(base.get(f"clean.runs.{run}"))
    fit_run_cfg = dict(run_cfg)   # the original windows: the embargo of the dev events is always measured from the original hold-out start
    if sealed_spec is not None:
        run_cfg = SEALED.window_cfg(run_cfg, sealed_spec)
    bucket, llm = run_cfg["bucket"], run.split("-")[1]
    market_dir = root / str(cfg.get("build.paths.market_dir"))
    calendar = TradingCalendar.from_config(cfg)
    clean = pd.read_parquet(sealed_clean if sealed_clean is not None else root / str(base.get("clean.output_dir")) / f"clean_{run}.parquet")
    cut = calendar.shift(str(fit_run_cfg["holdout_start"]), -int(cfg.get("windows.calendar_embargo_trading_days")))
    ce = CA.from_clean(clean, run_cfg, cfg, embargo_cut=cut, drop_zero_pins=bool(cfg.get("build.sheaf.drop_zero_pins")), test_from=pd.Timestamp(sealed_spec["start"]) if sealed_spec else None)
    sleeve_px, sleeve_actions = market_dir / "sleeve_px.parquet", market_dir / "sleeve_corp_action.parquet"
    if not sleeve_px.exists():
        raise SystemExit(f"{sleeve_px} is missing: the game's sleeve proxy ({cfg.get('game.sleeve_proxy')}) has no prices. Run scripts/fetch_sleeve.py --separate once (internet), then rebuild the pack")
    rm = load_real_market(market_dir, cfg, pd.read_parquet(sleeve_px), pd.read_parquet(sleeve_actions) if sleeve_actions.exists() else None)
    links = pd.read_parquet(root / run_cfg["links"])
    links = links.assign(**{c: pd.to_datetime(links[c]) for c in (LINK_FROM, LINK_TO)})
    try:
        persona_tape, used = load_tape(cfg, root, "personas", bucket, llm) if with_personas else (None, [])
    except TapeError as exc:
        print(f"no persona tape ({exc}); the game runs on the reader's consensus only", flush=True)
        persona_tape, used = None, []
    windows = BucketWindows(bucket, calendar.snap_forward(run_cfg["fit_start"]), calendar.snap_backward(run_cfg["fit_end"]), calendar.snap_forward(run_cfg["holdout_start"]),
                            calendar.snap_backward(run_cfg["holdout_end"]), int(cfg.get("windows.calendar_embargo_trading_days")))
    return CleanWorld(cfg, base, run, bucket, llm, run_cfg, rm, pd.read_parquet(root / str(cfg.get("build.paths.factors_file"))), links, pd.read_parquet(market_dir / "f13_managers.parquet"),
                      load_skeletons(market_dir / PRESTAGED / f"personas-{bucket}"), persona_tape, used, calendar, windows, clean, ce, root,
                      pd.Timestamp(fit_run_cfg["holdout_start"]))


def prepare_clean(world: CleanWorld) -> Prepared:
    """Readings split into dev and test, day 0 index and date, and the first tradable day of every event (the session after day 0: decisions are taken at the day-0 close)."""
    ce, rm = world.ce, world.rm
    events = ce.events
    day0_index = dict(zip(events["event_id"], rm.sessions.get_indexer(pd.DatetimeIndex(events["session_day0"])), strict=True))
    on_grid = {e for e, i in day0_index.items() if i >= 1}
    readings = {e: r for e, r in ce.readings.items() if e in on_grid}
    day0_date = {e: pd.Timestamp(d) for e, d in zip(events["event_id"], events["session_day0"], strict=True) if e in readings}
    kinds = dict(zip(events["event_id"], events["kind"], strict=True))
    split = dict(zip(events["event_id"], events["split"], strict=True))
    dev_ids = [e for e in readings if split[e] == CA.DEV]
    test_ids = [e for e in readings if split[e] == CA.TEST]
    diagnostics = {"reader_mode": MODE, "events": int(len(events)), "readings_with_pins": len(readings), "dev_events": len(dev_ids), "test_events": len(test_ids), **ce.counts,
                   "session_types_test": {str(k): int(v) for k, v in events[events["split"] == CA.TEST]["session_type"].value_counts().items()}}
    return Prepared(readings, day0_index, day0_date, kinds, split, dev_ids, test_ids, diagnostics)


def restrict_persona_tape(world: CleanWorld, prep: Prepared) -> None:
    """The persona tape was run on the raw units; only the events that survived the cleaning and have company pins can be read. Counts go to the diagnostics."""
    if world.persona_tape is None:
        return
    keep = world.persona_tape["event_id"].isin(set(prep.day0_index))
    prep.diagnostics.update({"persona_rows": int(len(world.persona_tape)), "persona_rows_on_clean_events": int(keep.sum()), "persona_events_on_clean_events": int(world.persona_tape.loc[keep, "event_id"].nunique()),
                             "persona_events_on_clean_test_events": int(world.persona_tape.loc[world.persona_tape["event_id"].isin(set(prep.test_ids)), "event_id"].nunique())})
    world.persona_tape = world.persona_tape[keep]


def events_table(world: CleanWorld, prep: Prepared) -> pd.DataFrame:
    """The test events with their day 0 and first tradable day (the same table every variant and every rival is run on)."""
    rm = world.rm
    rows = [{"event_id": e, "session_day0": prep.day0_date[e], "first_tradable_day": rm.sessions[prep.day0_index[e] + 1] if prep.day0_index[e] + 1 < len(rm.sessions) else pd.NaT,
             "kind": prep.kinds[e], "n_pins": len(prep.readings[e].pins)} for e in prep.test_ids]
    return pd.DataFrame(rows).sort_values("event_id").reset_index(drop=True)


def decision_days(world: CleanWorld) -> pd.DatetimeIndex:
    w = world.windows
    return pd.DatetimeIndex([d for d in world.rm.sessions if w.test_start <= d <= w.test_end])


def write_equal_terms(out: Path, events: pd.DataFrame, days: pd.DatetimeIndex, cfg: Config) -> dict[str, str]:
    """events_test.parquet and days_test.parquet with the sha256 of the files and of the id lists, so that rivals and variants can be checked to share them."""
    ev_path, day_path = out / str(cfg.get("framework.final.events_file")), out / str(cfg.get("framework.final.days_file"))
    events.to_parquet(ev_path, index=False)
    pd.DataFrame({"decision_date": days}).to_parquet(day_path, index=False)
    info = {"events_file": ev_path.name, "events_file_sha256": sha256_file(ev_path), "events_ids_sha256": sha256_hex("\n".join(events["event_id"])), "n_events": int(len(events)),
            "days_file": day_path.name, "days_file_sha256": sha256_file(day_path), "days_ids_sha256": sha256_hex("\n".join(str(d.date()) for d in days)), "n_days": int(len(days))}
    (out / "equal_terms.json").write_text(json.dumps(info, indent=1))
    return info


# ---- dev fit --------------------------------------------------------------------------------------------------------------------------

def dev_meta(world: CleanWorld) -> dict[str, Any]:
    return {"bucket": world.bucket, "llm": world.llm, "reader_mode": MODE, "arm": RP.ARM, "config_hash": world.cfg.hash,
            "fit_config_hash": world.cfg.hash_without(DEV_FIT_IGNORED_SECTIONS)}


def make_or_load_dev_fit(world: CleanWorld, prep: Prepared, fit_path: Path, dev_path: Path, log: Log):
    """``dev_fit.pkl`` of this run (read back when it exists and matches; refused when it is keyed by another mode, arm, bucket, LLM or configuration), else made from the
    build's fitted sheaf (``Fitted`` in the fit cache) plus the dev-only pin maps."""
    meta = dev_meta(world)
    if dev_path.exists():
        dev, dials = read_dev_fit(dev_path, meta)
        log(f"dev fit read from {dev_path} (keyed by reader mode {MODE})")
        return dev, dials
    if not fit_path.exists():
        raise SystemExit(f"{fit_path} is missing: the build's fit stage must finish before the framework stage")
    fitted = pickle.loads(fit_path.read_bytes())["fitted"]
    cfg, rm = world.cfg, world.rm
    panels = panels_of(rm, world.factors, cfg)
    dev_readings = {e: prep.readings[e] for e in prep.dev_ids}
    nodes = {e: tuple(p["node_id"] for p in r.pins) for e, r in dev_readings.items()}
    labels = rr.label_table(nodes, prep.day0_date, panels, cfg)
    labels = labels[labels["window"] == LABEL_WINDOW].rename(columns={"security_id": "node_id"})[["event_id", "node_id", "log_vol_ratio", "log_dv_ratio"]]
    pin_maps = RP.fit_pin_maps(world.ce.dials, labels, prep.dev_ids, cfg)
    dev = RP.make_dev_fit(fitted, world.links, cfg, pin_maps, len(prep.dev_ids), int(fitted.n_dev_graphs), {"n_pin_label_rows": int(len(labels))})
    dials = {e: world.ce.dials[e] for e in prep.readings}
    with dev_path.open("wb") as handle:
        pickle.dump({"meta": meta, "dev": dev, "dials": dials}, handle, protocol=pickle.HIGHEST_PROTOCOL)
    log(f"dev fit written to {dev_path}; pin maps {pin_maps}")
    return dev, dials


# ---- backtests ------------------------------------------------------------------------------------------------------------------------

def _backtest_task(task: tuple[str, str, str]):
    variant, which, rule = task
    g = _G
    value = None if rule == "delist_0" else float(g["cfg"].get("portfolio.delisting.default_return"))
    key = f"{variant}/{which}/{rule}" + ("" if g["world_on"] else "/no_world")
    result, net, origin = backtest_forecasts(g["cfg"], g["rm"], g["ctx"], g["windows"], g["forecasts"][variant], g["scope"], key, value, constraint=which,
                                             world_shift=g["world_shift"] if g["world_on"] else None, only_days=g.get("only_days"))
    return key, result, net, origin


def add_rivals(world: CleanWorld, forecasts: dict, days: pd.DatetimeIndex, out: Path, cfg: Config, log: Log) -> dict:
    """Rival score files found in <out>/<rivals_dir> become forecasts on the same days (equal terms). Returns what was found and what was run."""
    folder = out / str(cfg.get("framework.final.rivals_dir"))
    note: dict[str, Any] = {"folder": str(folder), "found": [], "run": [], "not_run": {}}
    if not folder.exists():
        note["not_run"]["all"] = "no rivals folder in the run folder"
        return note
    for path in sorted(folder.glob("*.parquet")):
        name = path.stem
        note["found"].append(name)
        scores = pd.read_parquet(path)
        slope, n = RA.fit_slope(scores, world.rm.oo_ret, world.rival_cutoff or world.windows.test_start)
        if not np.isfinite(slope):
            note["not_run"][name] = f"slope not fitted: only {n} scored pairs before the test window"
            continue
        forecasts[f"rival:{name}"] = RA.forecasts_from_scores(name, scores, days, world.rm.sessions, list(world.rm.close.columns), slope, int(cfg.get("portfolio.horizon_sessions")))
        note["run"].append({"name": name, "slope": slope, "dev_points": n, "forecast_days": len(forecasts[f"rival:{name}"])})
        log(f"rival {name}: slope {slope:.4g} from {n} dev points, {len(forecasts[f'rival:{name}'])} forecast days")
    return note


def result_line(key: str, net: pd.DataFrame, cfg: Config) -> str:
    """One RESULT line for the status block: the net Sharpe at the headline cost with a block-bootstrap interval."""
    from finfm.common.rng import make_rng
    from finfm.eval.stats import block_length, bootstrap_ci

    headline = f"{int(cfg.get('portfolio.scoring.headline_round_trip_bp'))}bp"
    periods = int(cfg.get("eval.annualisation_days"))
    values = net[headline].to_numpy()
    point, low, high = bootstrap_ci(values, lambda x: float(x.mean() / x.std() * np.sqrt(periods)) if x.std() > 0 else float("nan"), block_length(len(values)),
                                    int(cfg.get("eval.bootstrap.draws")), float(cfg.get("eval.bootstrap.ci_level")), make_rng("final", "result_line", key))
    return f"RESULT {key} net Sharpe at {headline} = {point:.3f} (95% block-bootstrap interval {low:.3f} to {high:.3f}, {len(values)} days)"


def run_backtests(tasks: list[tuple[str, str, str]], workers: int, log: Log, beat_seconds: float) -> dict[str, tuple[BacktestResult, pd.DataFrame]]:
    done: dict[str, tuple[BacktestResult, pd.DataFrame]] = {}
    beat = Heartbeat("backtests", len(tasks), beat_seconds)
    if workers <= 1 or len(tasks) == 1:
        for i, task in enumerate(tasks):
            began = time.time()
            key, result, net, origin = _backtest_task(task)
            done[key] = (result, net)
            log(f"backtest {key} {origin} ({time.time() - began:.0f} s)")
            log(result_line(key, net, _G["cfg"]))
            beat.tick(i + 1)
        return done
    context = multiprocessing.get_context("fork")
    with cf.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        futures = {pool.submit(_backtest_task, t): t for t in tasks}
        for i, future in enumerate(cf.as_completed(futures)):
            key, result, net, origin = future.result()
            done[key] = (result, net)
            log(f"backtest {key} {origin}")
            log(result_line(key, net, _G["cfg"]))
            beat.tick(i + 1)
    return done


def markdown_table(frame: pd.DataFrame, decimals: int = DECIMALS) -> str:
    if frame.empty:
        return "(no rows)"
    shown = frame.copy()
    for c in shown.columns:
        if pd.api.types.is_float_dtype(shown[c]):
            shown[c] = shown[c].map(lambda v: "" if pd.isna(v) else f"{v:.{decimals}f}")
    head = "| " + " | ".join(map(str, shown.columns)) + " |"
    rule = "|" + "|".join("---" for _ in shown.columns) + "|"
    body = ["| " + " | ".join(str(v) for v in row) + " |" for row in shown.itertuples(index=False)]
    return "\n".join([head, rule, *body])


def run_final(root: Path, run: str, base: Config, out: Path, log: Log, workers: int = 1, only_variants: tuple[str, ...] | None = None, stop_after: str | None = None,
              dev_fit_path: Path | None = None, task_shard: tuple[int, int] | None = None, smoke: int = 0, sealed: bool = False, calibrate_volu: bool | None = None) -> Path:
    """The whole framework stage for one tape; returns the results folder. ``smoke`` = N: a mechanics test on the first N test events and the decision days that follow them (no freeze check, labelled, never a result)."""
    began = time.time()
    seconds: dict[str, float] = {}
    out = Path(out)
    spec = SEALED.sealed_settings(base, run) if sealed else None
    folder = out / str(spec["results_dir"] if sealed else base.get("framework.final.results_dir"))
    folder.mkdir(parents=True, exist_ok=True)
    code_root = Path(__file__).resolve().parents[2]
    beat_seconds = float(base.get(LOG_SECONDS_KEY))
    sealed_file = None
    if sealed:   # the freeze must match BEFORE anything of the sealed window is opened; then the lock; then the cleaning of the longer window
        if smoke:
            raise SystemExit("the sealed window is never run as a smoke test")
        freeze_path = root / str(base.get("clean.output_dir")) / str(base.get("framework.final.freeze_file"))
        note = FZ.check_static(freeze_path, base, root, code_root, run)   # config, code, tapes, persona tapes (the dev fit is checked after it is rebuilt, below)
        log("SEALED STAGE: freeze matches; taking the open-once lock")
        SEALED.take_lock(folder, spec, run, note, FZ.git_head(code_root))
        sealed_file = SEALED.sealed_clean(root, run, base, folder, spec, log)

    t0 = time.time()
    log("STAGE load start")
    world = load_clean_world(root, run, base, sealed_clean=sealed_file, sealed_spec=spec)
    prep = prepare_clean(world)
    cfg = world.cfg
    restrict_persona_tape(world, prep)
    if smoke:
        prep.test_ids = sorted(prep.test_ids, key=lambda e: (prep.day0_date[e], e))[:smoke]
        prep.diagnostics["SMOKE"] = f"mechanics test on the first {smoke} test events; not a result"
    seconds["load"] = time.time() - t0
    log(f"STAGE load done in {seconds['load']:.0f} s: {prep.diagnostics}")

    t0 = time.time()
    log("STAGE dev_fit start")
    fit_path = out / str(cfg.get(FIT_FILE_KEY))
    dev_path = Path(dev_fit_path) if dev_fit_path else folder / str(cfg.get("framework.final.dev_fit_file"))
    dev, dials = make_or_load_dev_fit(world, prep, fit_path, dev_path, log)
    seconds["dev_fit"] = time.time() - t0
    cal_spec = dict(cfg.get("framework.calibration"))
    cal_on = bool(cal_spec["enabled"]) if calibrate_volu is None else bool(calibrate_volu)
    cal_path = root / str(cal_spec["dir"]) / f"volu_calibration_{run}.json"
    if stop_after == "calibrate":   # the dev-only fitting stage that runs BEFORE the freeze: nothing of the test or sealed window is read
        return fit_calibration_stage(world, prep, dev, dials, cfg, run, folder, cal_path, log)

    if smoke:
        freeze_note = {"freeze": "skipped (smoke test)", "freeze_file": None, "created_utc": None}
    else:
        freeze_note = FZ.check(root / str(base.get("clean.output_dir")) / str(cfg.get("framework.final.freeze_file")), base, root, code_root, run, dev_path, folder / FZ.RUNTIME_NAME)
    log(f"STAGE freeze: {freeze_note['freeze']} ({freeze_note['freeze_file']})")
    if sealed:   # the dev fit of the sealed stage must be the main run's dev fit, byte for byte, when the main run's pin is at hand
        main_pin = out / str(base.get("framework.final.results_dir")) / FZ.RUNTIME_NAME
        mine = sha256_file(dev_path)
        if main_pin.exists() and json.loads(main_pin.read_text()).get("dev_fit_sha256") not in (None, mine):
            raise SystemExit(f"the sealed stage's dev fit ({mine[:SHORT_HASH]}) is not the main run's pinned dev fit ({json.loads(main_pin.read_text())['dev_fit_sha256'][:SHORT_HASH]}): the stage stops")
        log(f"sealed stage dev fit sha256 {mine[:SHORT_HASH]} ({'equal to the main run pin' if main_pin.exists() else 'no main run pin attached to compare with'})")
    if stop_after == "freeze":
        return folder

    events = events_table(world, prep)
    days = decision_days(world)
    if smoke:
        first = events["first_tradable_day"].min()
        days = days[(days >= first) & (days <= first + pd.Timedelta(days=SMOKE_CALENDAR_DAYS))]
    equal = write_equal_terms(folder, events, days, cfg)
    log(f"equal terms written: {equal}")

    t0 = time.time()
    log("STAGE combine start")
    game = make_game_inputs(cfg, world.rm, world.skeletons, world.managers)
    mode_record = config_for_mode(base, MODE)[1]
    ctx = build_context(cfg, world.rm, game, prep, dev, dials, world.persona_tape, world.skeletons, mode_record)
    ctx.day0_index = prep.day0_index
    calibration_note: dict[str, Any] = {"switch": "on" if cal_on else "off"}
    if cal_on:
        if not cal_path.exists():
            raise SystemExit(f"{cal_path} is missing: fit it first (scripts/fit_calibration.py {run}), or switch it off (framework.calibration.enabled / --no-volu-calibration)")
        record = CAL.read(cal_path)
        ctx.volu_calibration = CAL.setting_from_record(record, True)
        calibration_note.update({"file_sha256": CAL.calibration_sha256(record), "used": bool(record["used"]), "scale": record["scale"], "shift": record["shift"], "reason": record["reason"]})
    log(f"road-2 volume dial calibration: {calibration_note}")
    skips: list[dict] = []
    wanted = tuple(only_variants) if only_variants else (*VARIANTS, MERGED)
    scope = (cfg.hash, sha256_hex("|".join(prep.test_ids)), MODE, "final", "volu_calibration_" + ("on" if cal_on else "off"))
    stored = ckpt.get("combination", scope, wanted)
    if stored is None:
        variants, combination = fit_combination(ctx, cfg, world.rm, world.factors, world.calendar, prep.dev_ids, wanted, skips)
        ckpt.put("combination", (variants, combination, dict(ctx.return_weights), ctx.merge, skips), scope, wanted)
    else:
        variants, combination, ctx.return_weights, ctx.merge, skips = stored
        log("combination fit taken from the checkpoint")
    seconds["combine"] = time.time() - t0
    log(f"STAGE combine done in {seconds['combine']:.0f} s")

    t0 = time.time()
    log("STAGE forecast start")
    forecasts: dict[str, list] = {}
    for variant in variants:
        began_v = time.time()
        cached = ckpt.get("forecasts", scope, variant)
        if cached is None:
            before = len(skips)
            fs = _safe_forecasts(ctx, prep.test_ids, variant, skips)
            ckpt.put("forecasts", (fs, skips[before:]), scope, variant)
        else:
            fs = cached[0]
            skips.extend(cached[1])
        forecasts[variant] = fs
        log(f"forecasts {variant}: {len(fs)} events in {time.time() - began_v:.0f} s")
    seconds["forecast"] = time.time() - t0
    log(f"STAGE forecast done in {seconds['forecast']:.0f} s")
    if stop_after == "forecast":
        return folder
    forecasts["none"] = []  # the no-signal floor: no forecasts, so no positions and no return; it is a row of zeros and is not backtested
    rivals_note = add_rivals(world, forecasts, days, out, cfg, log)

    # world lane: fitted on the fit window only; shift for every test decision day
    shift = None
    if WL.enabled(cfg):
        prices = pd.read_parquet(root / world.run_cfg["prices"], columns=["security_id", "session_date", "close_raw", "volume_raw"])
        shift = WL.fit_world_shift(world.ce.world_pins, prices, world.run_cfg, list(base.get("clean.touch.themes")), cfg, days)
        log(f"world lane: {shift.notes}")

    t0 = time.time()
    log("STAGE backtest start")
    sets = list(cfg.get("framework.final.headline_sets"))
    rules = list(cfg.get("framework.final.delisting_rules"))
    _G.update({"cfg": cfg, "rm": world.rm, "ctx": ctx, "windows": world.windows, "forecasts": forecasts, "scope": scope, "world_shift": shift, "world_on": WL.enabled(cfg), "only_days": set(days) if smoke else None})
    rival_names = [k for k in forecasts if k.startswith("rival:")]
    tasks = [(v, s, r) for v in [*variants, *rival_names] for s in sets for r in rules if forecasts.get(v)]
    if task_shard:  # one slice of the backtests in this process (separate processes share the checkpoint folder); the metrics are left to the unsharded run
        tasks = tasks[task_shard[0] :: task_shard[1]]
    done = run_backtests(tasks, workers, log, beat_seconds)
    if task_shard:
        log(f"task shard {task_shard[0]}/{task_shard[1]} finished: {len(tasks)} backtests stored in the checkpoint")
        return folder
    if WL.enabled(cfg):  # one comparison row without the world lane: the merged variant, headline set, 0 rule
        _G["world_on"] = False
        probe = MERGED if forecasts.get(MERGED) else next((v for v in variants if forecasts.get(v)), None)
        if probe:
            done.update(run_backtests([(probe, sets[0], rules[0])], 1, log, beat_seconds))
    seconds["backtest"] = time.time() - t0
    log(f"STAGE backtest done in {seconds['backtest']:.0f} s")

    t0 = time.time()
    log("STAGE metrics start")
    tables = build_tables(world, prep, forecasts, done, sets, rules, shift, skips, combination, base, root, out, log, sealed)
    seconds["metrics"] = time.time() - t0
    log(f"STAGE metrics done in {seconds['metrics']:.0f} s")
    figure_tables = dict(FIGURE_TABLES)
    try:   # the sheaf's fitted internals (fit-window rows only); the build's graphs.pkl is optional: without it the ripple table is left out
        graphs_file = out / "graphs.pkl"
        figure_tables.update(SI.sheaf_internal_tables(out / str(cfg.get(FIT_FILE_KEY)), graphs_file if graphs_file.exists() else None, cfg))
    except Exception as exc:  # noqa: BLE001 - figure data never stops a run
        log(f"sheaf internals not written: {type(exc).__name__}: {str(exc)[:ERROR_CHARS]}")
    (folder / "figure_data").mkdir(exist_ok=True)
    for name, frame in figure_tables.items():   # the numbers behind the paper's figures (daily series, dial pairs, position sizes, per-event forecasts)
        frame.to_parquet(folder / "figure_data" / f"{name}.parquet", index=False)
    for name, frame in tables.items():
        frame.to_csv(folder / f"{name}.csv", index=False)
    diagnostics = {**prep.diagnostics, "sealed_window": [str(spec["start"]), str(spec["end"])] if sealed else None, "volu_calibration": calibration_note, "stage_seconds": {k: round(v, 1) for k, v in seconds.items()}, "total_seconds": round(time.time() - began, 1), "equal_terms": equal, "freeze": freeze_note,
                   "persona_events_with_answers_test": int(sum(bool(ctx.answers(e)) for e in prep.test_ids)), "world_lane": getattr(shift, "notes", None), "rivals": rivals_note,
                   "skip_reasons": dict(Counter(s["reason"][:REASON_CHARS] for s in skips).most_common(TOP_REASONS)), "fits": {k: v for k, v in combination.items() if k != "merge"}}
    (folder / "diagnostics.json").write_text(json.dumps(diagnostics, indent=1, default=str))
    (folder / "FINAL_REPORT.md").write_text(report_text(run, tables, diagnostics, int(cfg.get("eval.report.decimals")), dict(cfg.get("framework.final"))))
    log(f"WROTE {folder} in {time.time() - began:.0f} s")
    return folder


def fit_calibration_stage(world: CleanWorld, prep: Prepared, dev, dials, cfg: Config, run: str, folder: Path, cal_path: Path, log: Log) -> Path:
    """Fit the road-2 volume-dial calibration on the fit-window (dev) events only and write ``volu_calibration_<run>.json`` (the file whose hash goes into the freeze)."""
    spec = dict(cfg.get("framework.calibration"))
    game = make_game_inputs(cfg, world.rm, world.skeletons, world.managers)
    ctx = build_context(cfg, world.rm, game, prep, dev, dials, world.persona_tape, world.skeletons, config_for_mode(world.base, MODE)[1])
    ctx.day0_index = prep.day0_index
    log(f"calibration stage: road-2 forecasts of {len(prep.dev_ids)} fit-window events (no test or sealed event is touched)")
    skips: list[dict] = []
    dev_forecasts = _safe_forecasts(ctx, prep.dev_ids, "both", skips)
    columns = list(world.rm.close.columns)
    nodes = CAL.sampled_nodes(dev_forecasts, columns, int(spec["assets_per_event"]), run)
    labels = rr.label_table(nodes, prep.day0_date, panels_of(world.rm, world.factors, cfg), cfg)
    labels = labels[labels["window"] == LABEL_WINDOW]
    pairs = CAL.fit_window_pairs(dev_forecasts, labels, columns, prep.day0_date, int(spec["assets_per_event"]), run)
    record = {"run": run, **CAL.fit_volu_calibration(pairs, cfg), "fit_window": [str(world.run_cfg["fit_start"]), str(world.run_cfg["fit_end"])], "dev_events_forecast": len(dev_forecasts), "skipped": len(skips)}
    digest = CAL.write(cal_path, record)
    pd.DataFrame(record["blocks"]).to_csv(folder / f"volu_calibration_blocks_{run}.csv", index=False)
    log(f"calibration written to {cal_path} (sha256 {digest[:SHORT_HASH]}): used={record['used']} scale={record['scale']:.4f} shift={record['shift']:.4f}; {record['reason']}")
    for b in record["blocks"]:
        log(f"  block {b['block']} {b['first_day']}..{b['last_day']}: raw RMSE {b['raw_rmse']:.4f} -> calibrated RMSE {b['calibrated_rmse']:.4f} (scale {b['scale_fitted_on_other_blocks']:.3f}, shift {b['shift_fitted_on_other_blocks']:.3f}, {b['n_pairs']} pairs)")
    return folder


def build_tables(world, prep, forecasts, done, sets, rules, shift, skips, combination, base, root, out, log, sealed: bool = False) -> dict[str, pd.DataFrame]:
    cfg, rm = world.cfg, world.rm
    rows = []
    for key, (result, net) in done.items():
        parts = key.split("/")
        variant, which, rule = parts[0], parts[1], parts[2]
        rows.append(summary_row({"variant": variant, "headline_set": which, "delisting": rule, "world_lane": "off" if len(parts) > WORLD_PARTS else ("on" if WL.enabled(cfg) else "off")}, result, net, cfg, world.factors))
    summary = pd.DataFrame(rows)
    tables: dict[str, pd.DataFrame] = {"summary": summary}
    headline = f"{int(cfg.get('portfolio.scoring.headline_round_trip_bp'))}bp"
    main_rule, on = str(cfg.get("eval.appendix_treatment")), "/no_world"
    # one series set per headline set and delisting rule (world lane as configured)
    for which in sets:
        for rule in rules:
            series = {k.split("/")[0]: net[headline] for k, (_, net) in done.items() if k.endswith(f"/{which}/{rule}")}
            if len(series) > 1:
                tables[f"paired_sharpe_variants_{which}_{rule}"] = FM.paired_sharpe_table(series, cfg, f"net_sharpe_{which}_{rule}")
            rivals = [k for k in series if k.startswith("rival:")]
            if rivals:
                pairs = [(v, r) for v in ("merged", "both", "road1", "road2") for r in rivals if v in series]
                tables[f"paired_sharpe_vs_rivals_{which}_{rule}"] = FM.paired_sharpe_table(series, cfg, f"rivals_{which}_{rule}", pairs)
            for name, frame in FM.deflated_pbo_cpcv(series, cfg).items():
                tables[f"{name}_{which}_{rule}"] = frame
    for which in sets:
        subset_bt = {k.rsplit("/", 2)[0] + "/" + k.rsplit("/", 1)[1]: v[0] for k, v in done.items() if f"/{which}/" in k and on not in k}
        subset_net = {k.rsplit("/", 2)[0] + "/" + k.rsplit("/", 1)[1]: v[1] for k, v in done.items() if f"/{which}/" in k and on not in k}
        try:
            for name, frame in appendix_tables(cfg, rm, {v: f for v, f in forecasts.items() if f}, subset_bt, subset_net).items():
                tables[f"{name}_{which}"] = frame
        except Exception as exc:  # noqa: BLE001 - reported, the headline rows are already written
            log(f"appendix tables for {which} not built: {type(exc).__name__}: {str(exc)[:ERROR_CHARS]}")
        tables[f"cost_grid_{which}"] = FM.cost_grid_table({k: v[0] for k, v in done.items() if f"/{which}/" in k and on not in k}, cfg)
    ranks = FM.event_rank_ic_horizons({v: f for v, f in forecasts.items() if f}, rm, cfg)
    if len(ranks):
        tables["rank_ic_horizons"] = ranks
    metrics_path = out / f"ALL_METRICS_{world.run}.parquet"
    if metrics_path.exists() and not sealed:   # the build's ALL_METRICS is the hold-out's; the sheaf is not scored on the sealed window
        tables["r2_os_per_dial"] = FM.sheaf_r2_per_dial(pd.read_parquet(metrics_path))
    # the sheaf's job in the optimiser: realised volatility against the risk target; accuracy of the vol and volume dials as the optimiser reads them
    figure_tables: dict[str, pd.DataFrame] = {
        "daily_series": pd.concat([FM.daily_series_table(k, r, n) for k, (r, n) in done.items()], ignore_index=True),
        "position_sizes": pd.concat([FM.position_size_table(k, r) for k, (r, _) in done.items()], ignore_index=True),
        "forecast_events": pd.DataFrame([{"variant": v, "event_id": f.event_id, "day0": f.day0, "n_assets": len(f.assets), "abs_increment_sum": float(np.abs(f.increments).sum()), "confidence": f.confidence,
                                          "mean_vol_dial": float(np.mean(f.vol_dial)), "mean_volu_dial": float(np.mean(f.volume_dial)), "has_cov_shift": f.cov_shift is not None}
                                         for v, fs in forecasts.items() for f in fs])}
    tables["realised_vol_vs_target"] = FM.realised_vol_table({k: v[0] for k, v in done.items()}, {k: v[1] for k, v in done.items()}, cfg)
    try:
        test_nodes = {f.event_id: tuple(rm.close.columns[i] for i in f.assets) for fs in forecasts.values() for f in fs}
        labels = rr.label_table(test_nodes, prep.day0_date, panels_of(rm, world.factors, cfg), cfg)
        labels = labels[labels["window"] == LABEL_WINDOW]
        pairs: dict = {}
        tables["dial_accuracy_vol_volu"] = FM.dial_accuracy_table({v: f for v, f in forecasts.items() if f}, labels, list(rm.close.columns), cfg, pairs)
        keep = int(cfg.get("framework.figure_data.pairs_per_event"))
        figure_tables["dial_pairs"] = pd.concat([f.groupby("event_id").head(keep).assign(variant=v) for v, f in pairs.items()], ignore_index=True) if pairs else pd.DataFrame()
    except Exception as exc:  # noqa: BLE001 - reported; the checklist shows the table as missing
        log(f"dial accuracy not built: {type(exc).__name__}: {str(exc)[:ERROR_CHARS]}")
    FIGURE_TABLES.clear()
    FIGURE_TABLES.update(figure_tables)   # written to <results>/figure_data/ (parquet) by run_final, not to the csv tables
    # delisting trace: the same strategy under the two rules, for every variant and headline set that ran both
    px = pd.read_parquet(root / world.run_cfg["prices"], columns=["security_id", "session_date", "close_raw", "volume_raw"])
    actions = pd.read_parquet(root / str(cfg.get("build.paths.market_dir")) / "corp_action.parquet")
    traces, hits = [], []
    default = float(cfg.get("portfolio.delisting.default_return"))
    for key, (result, _) in done.items():
        if key.endswith("/delist_default"):
            mate = done.get(key.replace("/delist_default", "/delist_0"))
            if mate is not None:
                t, h = delist_trace(mate[0], result, rm, px, actions, default, key.rsplit("/", 1)[0])
                traces.append(t)
                hits.append(h)
    if any(len(t) for t in traces):
        tables["delisting_trace_by_name"] = pd.concat([t for t in traces if len(t)], ignore_index=True)
        tables["delisting_trace_hits"] = pd.concat([h for h in hits if len(h)], ignore_index=True)
    # the checklist
    tmp = {name: frame for name, frame in tables.items()}
    present = {"paired_sharpe_vs_variants": any(n.startswith("paired_sharpe_variants") for n in tmp), "sharpe_interval_block_bootstrap": any(n.startswith("appendix_sharpe_interval") for n in tmp),
               "holm_within_family": any("p_holm" in f.columns for f in tmp.values()), "deflated_sharpe": any(n.startswith("deflated_sharpe") for n in tmp), "pbo": any(n.startswith("pbo") for n in tmp),
               "cpcv": any(n.startswith("cpcv") for n in tmp), "rank_ic_1d": "rank_ic_horizons" in tmp and bool((tmp["rank_ic_horizons"]["horizon_days"] == 1).any()),
               "rank_ic_5d": "rank_ic_horizons" in tmp and bool((tmp["rank_ic_horizons"]["horizon_days"] == HORIZON_FIVE).any()), "icir": "rank_ic_horizons" in tmp,
               "r2_os_ret": "r2_os_per_dial" in tmp and bool((tmp["r2_os_per_dial"]["dial"] == "ret").any()), "r2_os_vol": "r2_os_per_dial" in tmp and bool((tmp["r2_os_per_dial"]["dial"] == "vol").any()),
               "r2_os_volu": "r2_os_per_dial" in tmp and bool((tmp["r2_os_per_dial"]["dial"] == "volu").any()), "stress_windows": any(n.startswith("appendix_stress_windows") for n in tmp),
               "cost_grid_0_40bp": any(n.startswith("cost_grid") for n in tmp), "capacity_aum_levels": any(n.startswith("appendix_capacity") for n in tmp),
               "paired_sharpe_vs_rivals": any(n.startswith("paired_sharpe_vs_rivals") for n in tmp),
               "realised_vol_vs_target": "realised_vol_vs_target" in tmp and len(tmp["realised_vol_vs_target"]) > 0, "dial_accuracy_vol_volu": "dial_accuracy_vol_volu" in tmp and len(tmp["dial_accuracy_vol_volu"]) > 0}
    flat = {k: (pd.DataFrame({"x": [1]}) if present.get(k) else pd.DataFrame()) for k in FM.REQUIRED["Inference"] + FM.REQUIRED["Forecast level"] + FM.REQUIRED["Windows, cost, capacity"] + FM.REQUIRED["The sheaf in the optimiser"]}
    added = ["holm_within_family", "deflated_sharpe", "pbo", "cpcv", "rank_ic_5d", "icir", "cost_grid_0_40bp", "paired_sharpe_vs_variants", "r2_os_ret", "r2_os_vol", "r2_os_volu", "realised_vol_vs_target", "dial_accuracy_vol_volu"]
    tables["checklist"] = FM.checklist(summary, flat, added, {"paired_sharpe_vs_rivals": "no rival score files for this tape were found in the run folder (rivals are run on events_test.parquet and days_test.parquet)", **({m: "not produced in the sealed window (the sheaf is not scored there)" for m in ("r2_os_ret", "r2_os_vol", "r2_os_volu")} if sealed else {})})
    return tables


def report_text(run: str, tables: dict[str, pd.DataFrame], diagnostics: dict, decimals: int, cfg_final: dict) -> str:
    lines = [f"# Framework final, tape {run}" + (" -- SEALED WINDOW (opened once)" if diagnostics.get("sealed_window") else ""), "", "Numbers and intervals only. Both headline sets are put on the same risk target (portfolio.vol_match).", ""]
    summary = tables["summary"]
    cols = ["variant", "headline_set", "delisting", "world_lane", "sharpe_gross", "sharpe_net_5bp", "sharpe_net_10bp", "sharpe_net_20bp", "annual_return_net", "annual_vol_net", "max_drawdown_net_headline",
            "turnover_annual", "annual_cost", "break_even_round_trip_bp", "alpha_annual", "t_alpha", "cum_net_return", "delist_hits", "risk_scale"]
    cols = [c for c in cols if c in summary.columns]
    first, second = str(cfg_final["headline_set"]), str(cfg_final["second_set"])
    for label, which in (("HEADLINE (market-neutral, long-short)", first), ("SECOND TABLE (long-only, 5% cap)", second)):
        part = summary[summary["headline_set"] == which][cols].sort_values(["delisting", "variant"])
        lines += [f"## {label}: set '{which}'", "", markdown_table(part, decimals), ""]
    for name in ("checklist", "realised_vol_vs_target", "dial_accuracy_vol_volu", "delisting_trace_by_name", "rank_ic_horizons", "r2_os_per_dial"):
        if name in tables:
            lines += [f"## {name}", "", markdown_table(tables[name].head(TABLE_ROWS), decimals), ""]
    lines += ["## Stage seconds", "", json.dumps(diagnostics.get("stage_seconds"), indent=1), ""]
    return "\n".join(lines)
