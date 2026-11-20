"""CPU scoring of the MASS rival: bundle -> parsed votes -> type-distribution annealing -> disagreement signal -> published portfolio rule -> common metrics.

Same machinery and output conventions as ``score_basic`` / ``score_finripple`` (their helpers are imported, not copied; their numbers are untouched).
Realised returns are read in scoring code only. Every setting is in ``config/cpu.yaml`` (section ``score_mass``) or the shared config.

What is MASS's own and what is ours:
* signal: vote share per investor type, then ``alpha * mean - (1 - alpha) * std`` over the types weighted by the type distribution (``rivals.mass.signals``);
* type distribution: simulated annealing on past dates (``rivals.mass.optimizer``), online and causal: at decision date t only dates whose label is
  fully realised by the close of t are used. Past test returns enter this online fit exactly as they would in a live run; they never decide alpha or the top share;
* portfolio: top share of the pool by signal, equal weight, long only, rebalanced weekly (``rivals.mass.portfolio``), through the common executor, fills and costs;
* tuning: ``alpha`` x ``top_fraction`` (the tunables that can change after the run) are chosen on the dev window only, by net Sharpe at the headline cost.
Nothing is imputed; numbers only, no verdict wording.
"""

from __future__ import annotations

import json
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from finfm.common.config import Config
from finfm.common.hashing import sha256_file
from finfm.common.schema import DETAIL, PARSE_OK, STATUS, TOKENS_IN, TOKENS_OUT, UNIT_ID, Status
from finfm.common.timing import TradingCalendar
from finfm.common.windows import BucketWindows, bucket_windows, clean_start
from finfm.compute.bundle import consolidate_rows, part_sources, scan_bundles, verify_bundle
from finfm.data.sp500 import members_on
from finfm.eval.dsr_pbo import deflated_sharpe
from finfm.eval.portfolio_metrics import bridge, break_even_round_trip_bp, capacity, factor_alpha, sharpe, window_metrics
from finfm.eval.score_basic import (
    Rows, Run, cost_models, headline_cost_name, md_table, series_rows, small_flag, verdict_words, window_spans, write_parquet,
)
from finfm.eval.score_market import FactorFileMissing, ScoreMarket, abnormal_forward, fill_session, load_market
from finfm.eval.signal_metrics import ic_report
from finfm.portfolio.execute import net_returns, run_backtest, scale_to_risk_target, turnover
from finfm.rivals.llt.repro import leg_returns, yearly_sharpe
from finfm.rivals.mass.optimizer import anneal
from finfm.rivals.mass.portfolio import mass_strategy, weekly_rebalance_dates
from finfm.rivals.mass.repro import ic_table, scaling_check, summary
from finfm.rivals.mass.run import AGENT, CHOSEN, DATE, SHOWN, STYLE, TYPE
from finfm.rivals.mass.signals import VoteBook

__all__ = ["FactorFileMissing", "MassInputs", "load_bundle", "load_real_inputs", "run_scoring", "validate_meta"]

KEY = "cpu.score_mass"
BASIC_KEY = "cpu.score_basic"
SEPARATOR = "|"
SIGNAL_COLUMNS = ("Signal", "Signal_mean", "Signal_std")
STYLE_STAGE = "style"
ANNEALED, UNIFORM = "annealed", "uniform"
TUNING_KEYS = ("alpha", "top_fraction")
POOL_FILE, TYPES_FILE, MACRO_FILE, NEWS_FILE, UNITS_FILE = "mass_pool.parquet", "mass_types.json", "mass_macro.json", "mass_news.json", "units.parquet"
MODALITY_PREFIX, MODALITY_SUFFIX = "mass_modality_", ".parquet"
NEWS_MODALITY = "news"
GRID_PREFIX = "grid"


def sm(cfg: Config, path: str):
    return cfg.get(f"{KEY}.{path}")


def sb(cfg: Config, path: str):
    return cfg.get(f"{BASIC_KEY}.{path}")


# --- bundle ------------------------------------------------------------------------------------------------------


def load_bundle(path: Path) -> tuple[dict, pd.DataFrame, pd.DataFrame, dict]:
    """``(manifest, consolidated tape, every stored row, run state)``. The stored rows keep duplicates so they can be counted."""
    check = verify_bundle(path)
    if not check.ok:
        raise ValueError(f"bundle failed verification: {check.problems}")
    with tempfile.TemporaryDirectory() as folder:
        shutil.copy(path, Path(folder) / f"x{path.suffix}")
        refs = scan_bundles([folder], check.manifest["job_id"]).valid
        parts = [part.read() for part in part_sources(refs)]
        raw = pd.concat([p for p in parts if len(p)], ignore_index=True) if any(len(p) for p in parts) else pd.DataFrame({UNIT_ID: [], STATUS: []})
        tape = consolidate_rows(parts)
    with zipfile.ZipFile(path) as archive:
        state = json.loads(archive.read("state/state.json")) if "state/state.json" in archive.namelist() else {}
    return check.manifest, tape, raw, state


def validate_meta(meta: dict, cfg: Config) -> str | None:
    """A message when the bundle is not a MASS bundle of the LLM the rivals run on, else ``None``."""
    if meta.get("job") != sm(cfg, "job"):
        return f"this scorer reads job '{sm(cfg, 'job')}', the bundle is job '{meta.get('job')}'"
    if meta.get("llm") != sm(cfg, "llm"):
        return f"the rival runs use LLM '{sm(cfg, 'llm')}' only (experiment configuration); the bundle is LLM '{meta.get('llm')}'"
    return None


# --- inputs ------------------------------------------------------------------------------------------------------


@dataclass
class MassInputs:
    cfg: Config
    bucket: str
    llm: str
    calendar: TradingCalendar
    windows: BucketWindows
    clean: pd.Timestamp
    units: pd.DataFrame            # unit_id, est_seconds (the staged MASS unit table)
    pool: pd.DataFrame             # Date, Stock, Industry, FREE_MV
    types: dict[str, list[str]]    # investor type -> modalities
    modality_files: list[str]      # modalities present in the staged inputs (file names without prefix/suffix)
    news_days: int                 # days with news titles
    macro_days: int                # days with a macro text
    input_hashes: dict[str, str]
    snapshots: pd.DataFrame        # snapshot_date, security_id
    market: ScoreMarket


def load_real_inputs(root: Path, cfg: Config, bucket: str, llm: str) -> MassInputs:
    upload = root / str(cfg.get("cpu.folders.upload"))
    folder = upload / "prestaged" / f"{sm(cfg, 'job')}-{bucket}"
    calendar = TradingCalendar.from_config(cfg)
    files = {name: folder / name for name in (UNITS_FILE, POOL_FILE, TYPES_FILE, MACRO_FILE, NEWS_FILE)}
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        raise FileNotFoundError(f"staged MASS inputs missing: {missing}")
    return MassInputs(
        cfg, bucket, llm, calendar, bucket_windows(cfg, bucket, calendar), clean_start(str(cfg.get(f"llms.models.{llm}.stated_cutoff")), calendar),
        pd.read_parquet(files[UNITS_FILE]), pd.read_parquet(files[POOL_FILE]), json.loads(files[TYPES_FILE].read_text(encoding="utf-8")),
        sorted(p.name[len(MODALITY_PREFIX):-len(MODALITY_SUFFIX)] for p in folder.glob(f"{MODALITY_PREFIX}*{MODALITY_SUFFIX}")),
        len(json.loads(files[NEWS_FILE].read_text(encoding="utf-8"))), len(json.loads(files[MACRO_FILE].read_text(encoding="utf-8"))),
        {name: sha256_file(p) for name, p in files.items()}, pd.read_parquet(upload / "sp500_snapshot.parquet"), load_market(root, cfg, calendar))


# --- coverage ------------------------------------------------------------------------------------------------------


def unit_date(unit_id: str) -> str:
    return str(unit_id).split(SEPARATOR)[0]


def coverage(tape: pd.DataFrame, raw: pd.DataFrame, inp: MassInputs, rows: Rows) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Count failed, missing, unparsed and duplicate units against the staged unit table. Returns ``(usable rows, per-date coverage)``."""
    expected = set(inp.units[UNIT_ID])
    stored = raw[UNIT_ID].astype(str) if len(raw) else pd.Series(dtype=str)
    in_table = tape[UNIT_ID].isin(expected)
    rows.add("coverage", "units_expected", len(expected), n=len(expected))
    rows.add("coverage", "rows_stored", len(raw), n=len(raw), note="every row of every part of the bundle, before keeping one per unit")
    rows.add("coverage", "duplicate_rows", int(stored.duplicated().sum()), n=len(raw), note="same unit stored again (reruns, checkpoints); one row per unit is kept: an answered, parsed row replaces an unanswered or unparsed one")
    rows.add("coverage", "units_in_bundle", len(tape), n=len(expected))
    rows.add("coverage", "units_missing", len(expected - set(tape[UNIT_ID])), n=len(expected), note="in the staged table, not in the bundle")
    rows.add("coverage", "units_not_in_table", int((~in_table).sum()), n=len(tape), note="in the bundle, not in the staged table; ignored")
    tape = tape[in_table].copy()
    status = tape[STATUS].astype(str)
    for label in (Status.OK, Status.SKIPPED, Status.ERROR, Status.FAILED):
        rows.add("coverage", f"units_status_{label.value}", int((status == label.value).sum()), n=len(tape))
    rows.add("coverage", "units_failed_or_error", int(status.isin([Status.FAILED.value, Status.ERROR.value]).sum()), n=len(tape), note="no answer at all (attempts exhausted or exception)")
    ok = tape[status == Status.OK.value]
    parsed = ok[PARSE_OK].fillna(False).astype(bool)
    style_failed = ok[DETAIL].astype(str).eq(STYLE_STAGE) & ~parsed
    rows.add("coverage", "units_unparsed", int((~parsed).sum()), n=len(ok), note="answered but not usable")
    rows.add("coverage", "units_unparsed_style", int(style_failed.sum()), n=len(ok), note="the investing-style answer did not parse, so no decision was asked")
    rows.add("coverage", "units_unparsed_decision", int((~parsed & ~style_failed).sum()), n=len(ok), note="the decision answer did not parse or named a stock that was not shown")
    rows.add("coverage", "parse_rate", float(parsed.mean()) if len(ok) else np.nan, n=len(ok))
    usable = ok[parsed].copy()
    rows.add("coverage", "units_usable", len(usable), n=len(expected), note="status ok and parsed: these vote")
    empty = usable[CHOSEN].map(lambda text: not json.loads(text))
    rows.add("coverage", "units_empty_answer", int(empty.sum()), n=len(usable),
             note="parsed with an empty list of stocks (the repository counts them as agents that chose nothing; in this build the investor types that read only news have no stock table to choose from). They vote, so they are counted here and not presented as selections")
    expected_per_date = inp.units[UNIT_ID].map(unit_date).value_counts()
    usable_per_date = usable[UNIT_ID].map(unit_date).value_counts()
    types_per_date = usable.groupby(usable[UNIT_ID].map(unit_date))[TYPE].nunique()
    table = pd.DataFrame({"date": pd.to_datetime(expected_per_date.index), "expected": expected_per_date.to_numpy()})
    table["usable"] = usable_per_date.reindex(expected_per_date.index).fillna(0).astype(int).to_numpy()
    table["types_covered"] = types_per_date.reindex(expected_per_date.index).fillna(0).astype(int).to_numpy()
    table["share"] = table["usable"] / table["expected"]
    table = table.sort_values("date").reset_index(drop=True)
    rows.add("coverage", "dates_expected", len(table), n=len(table))
    rows.add("coverage", "dates_with_votes", int((table["usable"] > 0).sum()), n=len(table))
    rows.add("coverage", "dates_without_signal", int((table["types_covered"] < int(sm(inp.cfg, "min_types_per_date"))).sum()), n=len(table),
             note=f"fewer than {sm(inp.cfg, 'min_types_per_date')} investor types voted; no signal, no trade on that date")
    return usable, table


def run_usage(tape: pd.DataFrame, usable: pd.DataFrame, rows: Rows) -> None:
    """LLM calls and tokens as far as the tape shows them. Style prompts are shared by units with the same text, so their count is a bound."""
    styled = tape[tape[STYLE].astype(str).ne("{}") & tape[STYLE].notna()] if STYLE in tape else tape.iloc[0:0]
    pairs = tape[[DATE, TYPE]].drop_duplicates() if DATE in tape and TYPE in tape else tape.iloc[0:0]
    rows.add("usage", "decision_calls", len(styled), n=len(tape), note="units whose style parsed, so a decision was asked (one call each)")
    rows.add("usage", "style_calls_upper_bound", len(pairs), n=len(tape), note="distinct (date, investor type) pairs; units with the same prompt text share one call, so the true count can be lower")
    rows.add("usage", "calls_upper_bound", len(styled) + len(pairs), n=len(tape))
    for column in (TOKENS_IN, TOKENS_OUT):
        if column in tape:
            rows.add("usage", f"sum_{column}", int(tape[column].fillna(0).sum()), n=len(tape), note="decision calls only; style calls are not stored per unit")
    rows.add("usage", "usable_units", len(usable), n=len(tape))


def modality_report(inp: MassInputs, rows: Rows) -> None:
    """Which MASS modalities the investor types could see in this run and which the free data could not supply."""
    cfg = inp.cfg
    wanted = list(cfg.get("rivals.mass.modalities"))
    present = set(inp.modality_files) | ({NEWS_MODALITY} if inp.news_days else set())
    used = sorted({m for mods in inp.types.values() for m in mods})
    rows.add("modalities", "configured", ", ".join(wanted))
    rows.add("modalities", "available_in_staged_inputs", ", ".join(sorted(present)))
    rows.add("modalities", "unavailable", ", ".join(m for m in wanted if m not in present) or "none", note=str(sm(cfg, "labels.modalities")))
    rows.add("modalities", "seen_by_investor_types", ", ".join(used))
    rows.add("modalities", "macro_text_days", inp.macro_days, note="" if inp.macro_days else f"no macro text in the staged inputs (rivals.mass.use_macro_data is {cfg.get('rivals.mass.use_macro_data')})")
    rows.add("modalities", "news_days", inp.news_days)
    rows.add("modalities", "investor_types", len(inp.types))


# --- votes and signals ------------------------------------------------------------------------------------------------


@dataclass
class DateVotes:
    """Vote shares of one decision date: ``values[t, s]`` = share of type ``types[t]``'s agents that were shown stock ``stocks[s]`` and chose it."""

    date: pd.Timestamp
    types: list[str]
    stocks: list[str]
    values: np.ndarray


def date_votes(usable: pd.DataFrame, pool: pd.DataFrame, min_types: int, max_agent: int | None = None) -> dict[pd.Timestamp, DateVotes]:
    """Vote shares per decision date from the usable rows (optionally only agents with index below ``max_agent``). Dates with too few types have none."""
    book = VoteBook()
    frame = usable if max_agent is None else usable[usable[AGENT].astype(int) < max_agent]
    for date, itype, shown, chosen in zip(frame[DATE], frame[TYPE], frame[SHOWN], frame[CHOSEN], strict=True):
        book.record(date, itype, json.loads(shown), json.loads(chosen))
    stocks_by_date = {str(pd.Timestamp(d).date()): sorted(g["Stock"]) for d, g in pool.groupby("Date")}
    out = {}
    for date in book.dates():
        types, stocks = book.types(date), stocks_by_date.get(str(date))
        if len(types) < min_types or not stocks:
            continue
        values = np.array([[(cell["score"] / cell["num_investors"]) if (cell := book.data[date][t].get(s)) and cell["num_investors"] else 0.0
                            for s in stocks] for t in types])
        out[pd.Timestamp(date)] = DateVotes(pd.Timestamp(date), types, stocks, values)
    return out


def disagreement_frame(votes: DateVotes, distribution: dict[str, float], alpha: float) -> pd.DataFrame:
    """``Stock, Signal, Signal_mean, Signal_std`` for one date: ``mean = sum w v / sum w``, ``std = sqrt(sum w (v - mean)^2 / sum w)``, ``Signal = alpha mean - (1 - alpha) std``."""
    weights = np.array([distribution[t] for t in votes.types], dtype=float)
    mean = weights @ votes.values / weights.sum()
    std = np.sqrt(weights @ (votes.values - mean) ** 2 / weights.sum())
    return pd.DataFrame({"Stock": votes.stocks, "Signal": alpha * mean - (1.0 - alpha) * std, "Signal_mean": mean, "Signal_std": -std})


def uniform_distribution(types: list[str], weight: float) -> dict[str, float]:
    return {t: weight for t in types}


def realised_labels(panel: pd.DataFrame, votes: dict[pd.Timestamp, DateVotes], sessions: pd.DatetimeIndex, horizon: int) -> tuple[dict, dict]:
    """``(label by decision date, position of the session whose close realises the label)``. Label = abnormal return from the fill open to the close ``horizon`` sessions on."""
    labels, ready = {}, {}
    for date in votes:
        fill = fill_session(sessions, date)
        if fill is None or fill not in panel.index:
            continue
        position = sessions.get_loc(fill) + horizon - 1
        if position >= len(sessions):
            continue
        labels[date], ready[date] = panel.loc[fill].dropna(), position
    return labels, ready


def anneal_signals(votes: dict[pd.Timestamp, DateVotes], labels: dict, ready: dict, alpha: float, inp: MassInputs, log=lambda m: None) -> tuple[dict, pd.DataFrame]:
    """Signal frame per date and the type weights used on each date, online: the window of date t holds the latest dates whose labels are realised by the close of t."""
    cfg = inp.cfg
    sessions, look_back = inp.calendar.sessions, int(cfg.get("rivals.mass.annealing.look_back_window"))
    seed = int(cfg.get("runtime.base_seed"))
    types = sorted({t for v in votes.values() for t in v.types})
    current = uniform_distribution(types, float(sm(cfg, "initial_type_weight")))
    dates, frames, weights = sorted(votes), {}, []
    for number, date in enumerate(dates, start=1):
        today = sessions.get_loc(date)
        eligible = [d for d in dates[:number - 1] if d in labels and ready[d] <= today]
        window = eligible[-look_back:]
        if len(window) >= look_back:
            current = anneal(current, window, lambda dist, ds: {d: disagreement_frame(votes[d], dist, alpha) for d in ds}, labels, cfg,
                             (seed, "mass_anneal", f"{alpha:g}", str(date.date())))
        frames[date] = disagreement_frame(votes[date], current, alpha)
        weights.append(pd.DataFrame({"date": date, "alpha": alpha, "investor_type": list(current), "weight": np.array(list(current.values())) / sum(current.values()),
                                     "optimised": len(window) >= look_back}))
        if number % int(sm(cfg, "progress_every_dates")) == 0:
            log(f"annealing alpha {alpha}: {number}/{len(dates)} dates")
    return frames, pd.concat(weights, ignore_index=True)


def uniform_signals(votes: dict[pd.Timestamp, DateVotes], alpha: float, inp: MassInputs) -> dict:
    types = sorted({t for v in votes.values() for t in v.types})
    uniform = uniform_distribution(types, float(sm(inp.cfg, "initial_type_weight")))
    return {d: disagreement_frame(v, uniform, alpha) for d, v in votes.items()}


def signal_series(frames: dict, column: str = SIGNAL_COLUMNS[0]) -> dict[pd.Timestamp, pd.Series]:
    """Per date a Series by stock, sorted by stock id so that ties (many stocks nobody chose) break the same way every time, not by pool order."""
    return {d: f.set_index("Stock")[column].sort_index() for d, f in frames.items()}


# --- portfolio -------------------------------------------------------------------------------------------------------


def run_mass(signals: dict, span, top_fraction: float, fill: str, delist: bool, inp: MassInputs, cache: dict, models: dict | None = None) -> Run:
    cfg = inp.cfg
    market, panel = cache[fill] if fill in cache else inp.market.execution_market(fill, cfg)
    cache[fill] = (market, panel)
    dates = [d for d in inp.calendar.sessions if span[0] <= d <= span[1]]
    omitted = [0]
    in_window = {d: s for d, s in signals.items() if span[0] <= d <= span[1]}
    result = run_backtest(market, dates, mass_strategy(in_window, weekly_rebalance_dates(inp.calendar.sessions), cfg, top_fraction),
                          delist_return=float(cfg.get("portfolio.delisting.default_return")) if delist else None, risk_free_daily=None)
    result = scale_to_risk_target(result, float(cfg.get("portfolio.vol_match.target_annual_vol")), int(cfg.get("portfolio.annualisation_days")))
    net = net_returns(result, models if models is not None else cost_models(cfg, inp.market, omitted))
    return Run(result, net, leg_returns(result, panel), turnover(result), result.positions.abs().sum(axis=1) > 0, omitted[0])


def grid_models(cfg: Config, market: ScoreMarket) -> dict:
    """Only gross and the headline cost: the dev grid needs nothing else."""
    models = cost_models(cfg, market, [0])
    return {k: models[k] for k in ("gross", headline_cost_name(cfg))}


def tuning_grid(cfg: Config) -> list[tuple[float, float]]:
    spec = cfg.get("rivals.mass.tuning")
    return [(float(a), float(t)) for a in spec[TUNING_KEYS[0]] for t in spec[TUNING_KEYS[1]]]


def grid_name(alpha: float, top: float) -> str:
    return f"{GRID_PREFIX}_a{alpha:g}_t{top:g}"


def choose_on_dev(grid_sharpes: dict[tuple[float, float], float], cfg: Config) -> tuple[float, float]:
    """The (alpha, top share) with the highest dev net Sharpe; ties go to the first in the configured order. Takes dev numbers only."""
    order = tuning_grid(cfg)
    finite = {k: v for k, v in grid_sharpes.items() if np.isfinite(v)}
    if not finite:
        return float(cfg.get("rivals.mass.alpha")), float(cfg.get("rivals.mass.top_fraction"))
    return max(order, key=lambda k: (finite.get(k, -np.inf), -order.index(k)))


# --- IC ---------------------------------------------------------------------------------------------------------------


def forward_by_date(panel: pd.DataFrame, dates, sessions: pd.DatetimeIndex) -> dict[pd.Timestamp, pd.Series]:
    out = {}
    for date in dates:
        fill = fill_session(sessions, date)
        if fill is not None and fill in panel.index:
            out[date] = panel.loc[fill]
    return out


def ic_block(rows: Rows, name: str, window: str, signals: dict, forward: dict[int, dict], span, cfg: Config) -> None:
    chosen = {d: s for d, s in signals.items() if span[0] <= d <= span[1]}
    rows.add("signal", "dates_scored", len(chosen), set=name, window=window, n=len(chosen), n_dates=len(chosen))
    if not chosen:
        return
    for horizon, fwd in sorted(forward.items()):
        table = ic_table(chosen, fwd)
        tag = f"h{horizon}"
        if table.empty:
            rows.add("signal", "rank_ic", None, set=name, window=window, fill=tag, note="no date with returns")
            continue
        s = summary(table)
        enough = s["n_dates"] >= int(sb(cfg, "min_ic_dates"))
        note = "" if enough else f"{s['n_dates']} dates, fewer than {sb(cfg, 'min_ic_dates')}: no ICIR or t-statistic"
        rows.add("signal", "ic", s["ic"], set=name, window=window, fill=tag, n=s["n_dates"], n_dates=s["n_dates"], note="cross-section: the pool of the date")
        rows.add("signal", "rank_ic", s["rank_ic"], set=name, window=window, fill=tag, n=s["n_dates"], n_dates=s["n_dates"], note=note)
        rows.add("signal", "icir", s["icir"] if enough else None, set=name, window=window, fill=tag, n=s["n_dates"], n_dates=s["n_dates"], note=note)
        rows.add("signal", "rank_icir", s["rank_icir"] if enough else None, set=name, window=window, fill=tag, n=s["n_dates"], n_dates=s["n_dates"], note=note)
        if enough:
            report = ic_report(table["rank_ic"].dropna(), int(cfg.get("eval.hac_lag")))
            rows.add("signal", "rank_ic_t_newey_west", report["t_newey_west"], set=name, window=window, fill=tag, n=s["n_dates"], n_dates=s["n_dates"], note=f"HAC lag {cfg.get('eval.hac_lag')}")


# --- scoring ----------------------------------------------------------------------------------------------------------


def schedule_note(dates: list[pd.Timestamp], sessions: pd.DatetimeIndex, cfg: Config) -> tuple[str, float]:
    gaps = np.diff(sessions.get_indexer(pd.DatetimeIndex(sorted(dates)))) if len(dates) > 1 else np.array([])
    median = float(np.median(gaps)) if len(gaps) else float("nan")
    weekly = np.isfinite(median) and median >= float(sm(cfg, "schedule.weekly_min_median_gap_sessions"))
    return (str(sm(cfg, "labels.schedule_weekly")) if weekly else "decision dates are denser than weekly; the portfolio still rebalances weekly (published rule)"), median


def window_runs(name: str, signals: dict, top: float, spans: dict, inp: MassInputs, cache: dict, rows: Rows, log, store: dict, pnl: dict, weights: list,
                full: bool, fills: tuple[str, ...]) -> None:
    """Backtests of one variant in every window, with the full metric set when ``full``."""
    cfg = inp.cfg
    periods, headline_cost = int(cfg.get("eval.annualisation_days")), headline_cost_name(cfg)
    for window, span in spans.items():
        if span is None:
            continue
        sub = {d: s for d, s in signals.items() if span[0] <= d <= span[1]}
        for fill in fills:
            delists = (True, False) if (fill == fills[0] and name == str(sm(cfg, "headline_variant"))) else (True,)
            for delist in delists:
                label = name + (str(sb(cfg, "no_delist_suffix")) if not delist else "")
                base = dict(set="pool", window=window, variant=label, fill=fill)
                if not sub:
                    rows.add("portfolio", "sharpe", None, note="no signals in this window", **base, cost=headline_cost)
                    continue
                log(f"backtest {label} {window} {fill}")
                run = run_mass(signals, span, top, fill, delist, inp, cache)
                store[(name, window, fill, delist)] = run
                for cost in run.net.columns:
                    series_rows(rows, run.net[cost], run, base, cfg, cost)
                    pnl.setdefault((label, cost), []).append(pd.DataFrame({"window": window, "fill": fill, "net": run.net[cost], "turnover": run.turnover,
                                                                            "long_leg": run.legs["long"]}).rename_axis("fill_date").reset_index())
                rows.add("portfolio", "one_way_turnover_per_invested_day", float(run.turnover[run.invested].mean()) if run.invested.any() else np.nan,
                         n=int(run.invested.sum()), **base, cost="")
                rows.add("portfolio", "delisting_hits", int(run.result.delist_hits.sum()), n=len(run.legs), **base, cost="")
                rows.add("portfolio", "missing_return_hits", int(run.result.missing_return_hits.sum()), n=len(run.legs), **base, cost="")
                rows.add("portfolio", "impact_cost_omitted_trades", run.omitted_impact, **base, cost="", note="trades without ADV or volatility history: no impact cost charged, counted here")
                if fill == fills[0] and delist:
                    pos = run.result.positions
                    long_form = pos.stack().rename("weight").reset_index().set_axis(["fill_date", "security_id", "weight"], axis=1)
                    weights.append(long_form[long_form["weight"] != 0].assign(window=window, variant=name))
                    if full and run.invested.any():
                        extras(rows, run, base, inp, periods, headline_cost, name)


def extras(rows: Rows, run: Run, base: dict, inp: MassInputs, periods: int, headline_cost: str, name: str) -> None:
    cfg = inp.cfg
    cost_series = run.net["gross"] - run.net[headline_cost]
    flag = small_flag(int(run.invested.sum()), periods)
    for key, value in bridge(run.net["gross"], cost_series, run.result.trades, periods).items():
        rows.add("bridge", key, value, n=len(run.net), note=flag, **base, cost=headline_cost)
    rows.add("bridge", "break_even_round_trip_bp", break_even_round_trip_bp(run.net["gross"], run.result.trades, float(cfg.get("portfolio.scoring.bp_to_fraction"))),
             n=len(run.net), note=flag, **base, cost="")
    for model, cols in sb(cfg, "factor_models").items():
        if run.net[headline_cost].std() > 0:
            out = factor_alpha(run.net[headline_cost], inp.market.factors[list(cols)], int(cfg.get("eval.hac_lag")), periods)
            rows.add("alpha", f"alpha_annual_{model}", out["alpha_annual"], n=out["n"], note=f"HAC t = {out['t_alpha']:.3f}; {flag}; " + str(sb(cfg, "labels.alpha_approximation")), **base, cost=headline_cost)
    adv, sigma = inp.market.adv_and_sigma(cfg)
    if len(run.result.trades):
        for r in capacity(run.result.trades, adv, run.net["gross"], sigma, cfg).itertuples():
            rows.add("capacity", f"max_adv_share_aum_{r.aum_usd:g}", float(r.max_adv_share), n=len(run.net), **base, cost="")
            rows.add("capacity", f"annual_return_after_impact_aum_{r.aum_usd:g}", float(r.annual_return_after_impact), n=len(run.net), **base, cost="")
    if name == str(sm(cfg, "headline_variant")) and base["fill"] == str(sb(cfg, "fills.headline")):
        series = run.net[headline_cost]
        stress = {k: (v["start"], v["end"]) for k, v in sb(cfg, "stress").items()}
        done = set()
        for r in window_metrics(series, stress, periods, float(cfg.get("eval.cvar_level"))).itertuples():
            done.add(r.window)
            inside = int(run.invested.loc[stress[r.window][0]:stress[r.window][1]].sum())
            note = "; ".join(x for x in (r.window, str(sb(cfg, f"stress.{r.window}").get("status", "")), f"invested days in this window {inside}" + ("" if inside else ": no position, the zeros are not a result")) if x)
            rows.add("stress", "total_return", r.total_return, n=r.n_days, **base, cost=headline_cost, note=note)
            rows.add("stress", "max_drawdown", r.max_drawdown, n=r.n_days, **base, cost=headline_cost, note=r.window)
        for k in stress:
            if k not in done:
                rows.add("stress", "total_return", None, **base, cost=headline_cost, note=f"{k}: {sb(cfg, 'labels.not_computed')} (outside this window's sample)")
        for year, value in yearly_sharpe(series, periods).items():
            in_year = series[series.index.year == year]
            rows.add("robust_year", "sharpe", value, n=len(in_year), **base, cost=headline_cost, note=f"{year}; invested days {int(run.invested[in_year.index].sum())}")


def excess_rows(rows: Rows, runs: dict, bench: dict, inp: MassInputs) -> None:
    """MASS's own yardstick: cumulative excess return over the equal-weight pool (index analogue), same executor, same dates."""
    cfg = inp.cfg
    periods, headline_cost = int(cfg.get("eval.annualisation_days")), headline_cost_name(cfg)
    for (name, window, fill, delist), run in runs.items():
        if delist is False or (window, fill) not in bench:
            continue
        reference = bench[(window, fill)].net["gross"]
        for column, label in (("gross", "gross"), (headline_cost, headline_cost)):
            excess = (run.net[column] - reference).dropna()
            if excess.empty:
                continue
            base = dict(set="pool", window=window, variant=name, fill=fill, cost=label)
            rows.add("excess", "cumulative_excess_return", float(np.expm1(np.log1p(run.net[column].reindex(excess.index)).sum() - np.log1p(reference.reindex(excess.index)).sum())),
                     n=len(excess), note="compounded strategy return over compounded equal-weight pool return, same dates", **base)
            rows.add("excess", "annual_excess_return", float(excess.mean() * periods), n=len(excess), **base)
            rows.add("excess", "information_ratio", sharpe(excess, periods), n=len(excess), note="annualised mean over std of the daily excess", **base)
        rows.add("excess", "benchmark_annual_return", float(reference.mean() * periods), n=len(reference), set="pool", window=window, variant=name, fill=fill, cost="gross")


def run_scoring(inp: MassInputs, tape: pd.DataFrame, raw: pd.DataFrame, manifest: dict, state: dict, out_dir: Path, log=lambda message: None) -> dict:
    """Score the tape end to end, write every output under ``out_dir``, return the long results table and the text summary."""
    cfg = inp.cfg
    rows = Rows(cfg)
    periods, headline_cost = int(cfg.get("eval.annualisation_days")), headline_cost_name(cfg)
    open_fill, close_fill = str(sb(cfg, "fills.headline")), str(sb(cfg, "fills.sensitivity"))
    horizon = int(sm(cfg, "label_horizon"))
    if horizon not in [int(h) for h in cfg.get("eval.ic_horizons")]:
        raise ValueError(f"label_horizon {horizon} is not one of eval.ic_horizons {cfg.get('eval.ic_horizons')}")
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = tuning_grid(cfg)
    count_text = f"{len(grid)} (alpha, top share) configurations tuned on {sm(cfg, 'tuning.selection_window')} of budget {cfg.get('rivals.tuning_configurations')}; shown_stocks and temperature are generation settings, not tuned here"
    meta = {"config_hash": cfg.hash, "cpu_settings_hash": cfg.unhashed_hash, "configuration_count": count_text, "cost_setting": sb(cfg, "labels.cost_setting"),
            "bundle_id": manifest.get("bundle_id"), "bucket": inp.bucket, "llm": inp.llm, "code_version": manifest.get("meta", {}).get("code_version"),
            "bundle_config_hash": manifest.get("meta", {}).get("config_hash"), **{f"input_sha256_{k}": v for k, v in inp.input_hashes.items()}}

    rows.add("run", "units_total", len(inp.units), n=len(inp.units))
    rows.add("run", "loop_seconds", state.get("elapsed_seconds"), note="worker loop only; engine load time is not in the bundle")
    rows.add("run", "code_version", meta["code_version"])
    rows.add("run", "bundle_config_hash", meta["bundle_config_hash"], note="hash the GPU session recorded (a different, longer hash format)")
    rows.add("run", "agents_per_type", int(cfg.get("rivals.mass.agents_per_type")))
    log("coverage")
    usable, per_date = coverage(tape, raw, inp, rows)
    run_usage(tape, usable, rows)
    modality_report(inp, rows)
    if usable.empty:
        raise ValueError("no usable MASS rows in the bundle: nothing to score")

    log("votes")
    min_types = int(sm(cfg, "min_types_per_date"))
    votes = date_votes(usable, inp.pool, min_types)
    note, median_gap = schedule_note(list(votes), inp.calendar.sessions, cfg)
    rows.add("honesty", "schedule", note, n=len(votes), note=f"median gap between decision dates {median_gap:.1f} sessions")
    rows.add("honesty", "annealing", str(sm(cfg, "labels.annealing")))
    rows.add("honesty", "label", f"abnormal return from the open after the decision to the close {horizon} sessions on; a date's label enters the annealing window only once that close is in the past")
    rows.add("honesty", "tie_break", "stocks with equal signal (many that nobody chose) are ordered by stock id before the top share is cut")
    rows.add("honesty", "cost_setting", str(sb(cfg, "labels.cost_setting")))
    rows.add("honesty", "configuration_count", count_text)
    rows.add("honesty", "market_adjustment", str(sb(cfg, "labels.market_approximation")))
    rows.add("honesty", "alpha_dating", str(sb(cfg, "labels.alpha_approximation")))
    rows.add("honesty", "holding_period", str(sb(cfg, "labels.holding")))
    members = {d: set(members_on(inp.snapshots, d).security_ids) for d in votes}
    panels = {int(h): abnormal_forward(inp.market, list(votes), members, int(h), cfg) for h in cfg.get("eval.ic_horizons")}
    forward = {h: forward_by_date(p, votes, inp.calendar.sessions) for h, p in panels.items()}
    labels, ready = realised_labels(panels[horizon], votes, inp.calendar.sessions, horizon)
    spans = window_spans(inp)
    for name, span in spans.items():
        rows.add("coverage", "window_span" if span else "window_empty", f"{span[0].date()} to {span[1].date()}" if span else None, window=name,
                 note="" if span else f"{sb(cfg, 'labels.not_computed')}: no sessions of this window lie in the LLM's data range")

    alphas = sorted({a for a, _ in grid} | {float(cfg.get("rivals.mass.alpha"))})
    annealed, weights_used = {}, []
    for alpha in alphas:
        log(f"annealing alpha {alpha} over {len(votes)} dates")
        frames, used = anneal_signals(votes, labels, ready, alpha, inp, log)
        annealed[alpha] = frames
        weights_used.append(used)
    uniform = uniform_signals(votes, float(cfg.get("rivals.mass.alpha")), inp)

    log("signal metrics")
    signal_sets = {f"annealed_alpha{a:g}": signal_series(annealed[a]) for a in alphas} | {"uniform_types": signal_series(uniform)}
    for name, signals in signal_sets.items():
        for window, span in spans.items():
            if span is not None:
                ic_block(rows, name, window, signals, forward, span, cfg)
    scaling = {}
    for n_agents in sm(cfg, "scaling_agents_per_type"):
        sub_signals = signal_series(uniform_signals(date_votes(usable, inp.pool, min_types, int(n_agents)), float(cfg.get("rivals.mass.alpha")), inp))
        for window, span in spans.items():
            chosen = {d: s for d, s in sub_signals.items() if span and span[0] <= d <= span[1]}
            table = ic_table(chosen, forward[horizon]) if chosen else pd.DataFrame()
            if not table.empty:
                scaling.setdefault(window, {})[int(n_agents) * len(inp.types)] = summary(table)
    for window, found in scaling.items():
        if len(found) >= 2:
            for agents, r in scaling_check(found).iterrows():
                rows.add("repro", "rank_ic_by_agent_count", r["rank_ic"], set="uniform_types", window=window, n=int(agents), n_dates=int(r["n_dates"]),
                         note=f"{int(agents)} agents = subset of this bundle's agents; the pre-registered 64/128/256 check needs separate runs; rises with agents: {bool(r['rank_ic_rises'])}")

    # dev tuning: the selection reads the dev window only
    cache: dict = {}
    selection = str(sm(cfg, "tuning.selection_window"))
    dev_span = spans.get(selection)
    models = grid_models(cfg, inp.market)
    trial_returns: dict[tuple, dict] = {}
    by_alpha = {a: signal_series(annealed[a]) for a in alphas}
    for window, span in spans.items():
        if span is None:
            continue
        for alpha, top in grid:
            if any(span[0] <= d <= span[1] for d in by_alpha[alpha]):
                log(f"grid {grid_name(alpha, top)} {window}")
                trial_returns.setdefault(window, {})[(alpha, top)] = run_mass(by_alpha[alpha], span, top, open_fill, True, inp, cache, models).net[headline_cost]
    dev_sharpes = {k: sharpe(r, periods) for k, r in trial_returns.get(selection, {}).items()} if dev_span else {}
    chosen_alpha, chosen_top = choose_on_dev(dev_sharpes, cfg)
    for (alpha, top), value in dev_sharpes.items():
        rows.add("tuning", "dev_net_sharpe", value, set="pool", window=selection, variant=grid_name(alpha, top), cost=headline_cost, note="selection input: dev window only")
    rows.add("tuning", "chosen_alpha", chosen_alpha, window=selection, note="highest dev net Sharpe at the headline cost" + ("" if dev_sharpes else "; no dev backtest, the published values are kept"))
    rows.add("tuning", "chosen_top_fraction", chosen_top, window=selection)

    variants = {}
    published = (float(cfg.get("rivals.mass.alpha")), float(cfg.get("rivals.mass.top_fraction")))
    for name, spec in sm(cfg, "variants").items():
        alpha, top = (published if spec["alpha_key"] else (chosen_alpha, chosen_top))
        variants[name] = (signal_series(uniform) if spec["distribution"] == UNIFORM else by_alpha[alpha], top, alpha)
    pnl: dict = {}
    weights_out: list = []
    runs: dict = {}
    headline = str(sm(cfg, "headline_variant"))
    for name, (signals, top, alpha) in variants.items():
        rows.add("tuning", "variant_parameters", f"alpha {alpha:g}, top share {top:g}, {sm(cfg, f'variants.{name}.distribution')} type weights", variant=name)
        fills = (open_fill, close_fill) if name == headline else (open_fill,)
        window_runs(name, signals, top, spans, inp, cache, rows, log, runs, pnl, weights_out, True, fills)

    log("benchmark and excess return")
    bench_signals = variants[headline][0]
    bench = {}
    for window, span in spans.items():
        if span is not None and any(d >= span[0] and d <= span[1] for d in bench_signals):
            for fill in (open_fill, close_fill):
                bench[(window, fill)] = run_mass(bench_signals, span, float(sm(cfg, "benchmark_top_fraction")), fill, True, inp, cache, models)
    excess_rows(rows, runs, bench, inp)

    for window, found in trial_returns.items():
        key = (headline, window, open_fill, True)
        if key not in runs:
            continue
        series = runs[key].net[headline_cost]
        per_period = np.array([r.mean() / r.std() for r in found.values() if r.std() > 0])
        if series.std() > 0 and len(per_period) == len(grid):
            dsr = deflated_sharpe(series.to_numpy(), per_period, float(cfg.get("eval.dsr.euler_gamma")))
            rows.add("honesty", "deflated_sharpe_probability", dsr["dsr"], set="pool", window=window, variant=headline, cost=headline_cost, n=len(series),
                     note=f"trials {dsr['trials']} (the dev grid, run in this window); per-period expected best from skill-free trials {dsr['sr0']:.4f}")
        else:
            rows.add("honesty", "deflated_sharpe_probability", None, set="pool", window=window, variant=headline, cost=headline_cost, n=len(series), note="a grid configuration has no variation in this window")

    table = rows.frame()
    table.insert(0, "config_hash", cfg.hash[: int(sb(cfg, "hash_chars"))])
    table["configuration_count"] = count_text
    table["cost_setting"] = str(sb(cfg, "labels.cost_setting"))
    outputs = sm(cfg, "output")
    signal_long = pd.concat([f.assign(date=d, signal_set=f"annealed_alpha{a:g}") for a in alphas for d, f in annealed[a].items()]
                            + [f.assign(date=d, signal_set="uniform_types") for d, f in uniform.items()], ignore_index=True).rename(
        columns={"Stock": "security_id", "Signal": "signal", "Signal_mean": "signal_mean", "Signal_std": "signal_neg_std"})
    write_parquet(out_dir / outputs["signals"], signal_long, meta)
    write_parquet(out_dir / outputs["type_weights"], pd.concat(weights_used, ignore_index=True), meta)
    write_parquet(out_dir / outputs["coverage"], per_date, meta)
    if weights_out:
        everything = pd.concat(weights_out, ignore_index=True)
        for name in variants:
            write_parquet(out_dir / outputs["weights"].format(variant=name), everything[everything["variant"] == name].drop(columns="variant"), meta)
    for (variant, cost), parts in pnl.items():
        write_parquet(out_dir / outputs["pnl"].format(variant=variant, cost=cost), pd.concat(parts, ignore_index=True), meta)
    table.to_csv(out_dir / outputs["summary_csv"], index=False)
    (out_dir / outputs["metrics"]).write_text(json.dumps({"meta": meta, "rows": json.loads(table.to_json(orient="records"))}, indent=1, default=str), encoding="utf-8")
    (out_dir / outputs["hash_file"]).write_text("\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n", encoding="utf-8")
    text = summary_markdown(table, meta, cfg)
    (out_dir / outputs["summary_md"]).write_text(text, encoding="utf-8")
    return {"table": table, "text": text, "meta": meta, "runs": runs, "signals": annealed, "uniform": uniform, "chosen": (chosen_alpha, chosen_top), "coverage": per_date}


# --- summary ---------------------------------------------------------------------------------------------------------------


def summary_markdown(table: pd.DataFrame, meta: dict, cfg: Config) -> str:
    d = int(cfg.get("eval.report.decimals"))
    headline, headline_cost, open_fill = str(sm(cfg, "headline_variant")), headline_cost_name(cfg), str(sb(cfg, "fills.headline"))
    parts = [f"# MASS scored on {meta['bucket']} / {meta['llm']}\n",
             f"- config hash: `{meta['config_hash']}`  |  cpu settings hash: `{meta['cpu_settings_hash']}`",
             f"- configurations: {meta['configuration_count']}",
             f"- cost grid: {meta['cost_setting']}",
             f"- bundle: `{meta['bundle_id']}`",
             f"- every row below: {sb(cfg, 'labels.in_sample')}; dev rows: {sb(cfg, 'labels.exploratory')}\n"]
    for title, section in (("Coverage, parsing and LLM calls", ["run", "coverage", "usage"]), ("Modalities", ["modalities"]), ("Honesty notes", ["honesty"]), ("Tuning (dev only)", ["tuning"])):
        sub = table[table["section"].isin(section)][["section", "set", "window", "variant", "metric", "value", "n", "note"]]
        parts += [f"## {title}\n", md_table(sub, d)]
    sig = table[(table["section"] == "signal") & table["metric"].isin(["dates_scored", "ic", "icir", "rank_ic", "rank_icir"])]
    parts += ["## Signal (IC per decision date over the pool; h1 and h5 sessions)\n", md_table(sig[["set", "window", "fill", "metric", "value", "n_dates", "note"]].rename(columns={"fill": "horizon"}), d)]
    pf = table[(table["section"] == "portfolio") & (table["cost"] == headline_cost) & table["metric"].isin(["sharpe", "annual_return", "invested_days"]) & (table["fill"] == open_fill)]
    parts += [f"## Portfolio at {headline_cost}, fill {open_fill}\n", md_table(pf[["window", "variant", "metric", "value", "n", "note"]], d)]
    grid = table[(table["section"] == "portfolio") & (table["variant"] == headline) & (table["metric"] == "sharpe") & (table["fill"] == open_fill)]
    parts += [f"## Headline variant across the cost grid ({headline})\n", md_table(grid[["window", "cost", "value", "n", "note"]], d)]
    for title, section in (("Excess over the equal-weight pool", "excess"), ("Gross-to-net bridge", "bridge"), ("Factor alpha", "alpha"), ("Stress windows", "stress"),
                           ("Capacity", "capacity"), ("Robustness: year by year", "robust_year"), ("Scaling check", "repro")):
        sub = table[(table["section"] == section) & ((table["variant"] == headline) | (table["variant"] == ""))]
        parts += [f"## {title}\n", md_table(sub[["window", "fill", "metric", "value", "n", "note"]], d)]
    text = "\n".join(parts)
    bad = verdict_words(text, cfg)
    if bad:
        raise ValueError(f"verdict words in the summary: {bad}")
    return text
