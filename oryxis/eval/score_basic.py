"""CPU scoring of the Basic rival (Lopez-Lira and Tang): bundle -> signals -> published portfolio rule -> common metrics.

Realised returns are read here and in ``score_market`` only (scoring code). Positions come from the existing LLT rule
through ``portfolio.execute.run_backtest``, which sees prices only through ``Market.asof``. Every setting is in
``config/cpu.yaml`` (section ``score_basic``) or the shared config. Nothing is imputed: whatever cannot be computed is
written as the configured "not computed" label with the reason. Numbers only, no verdict wording.
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import statsmodels.api as sm
from scipy.stats import norm

from oryxis.common.config import Config
from oryxis.common.schema import PARSE_OK, STATUS, UNIT_ID, Status
from oryxis.common.timing import TradingCalendar
from oryxis.common.windows import BucketWindows, bucket_windows, clean_start, cutoff_part
from oryxis.compute.bundle import read_consolidated, scan_bundles, verify_bundle
from oryxis.data.buckets.events import time_known
from oryxis.data.sp500 import members_on
from oryxis.eval.dsr_pbo import deflated_sharpe
from oryxis.eval.portfolio_metrics import (
    bridge, break_even_round_trip_bp, capacity, cvar, factor_alpha, max_drawdown, sharpe, window_metrics,
)
from oryxis.eval.score_market import (
    FactorFileMissing, ScoreMarket, abnormal_forward, fill_session, load_market,
)
from oryxis.eval.signal_metrics import daily_rank_ic, ic_report
from oryxis.eval.stats import sharpe_difference_se, smallest_detectable_gap
from oryxis.portfolio.costs import fixed_cost, sqrt_impact_cost
from oryxis.portfolio.execute import BacktestResult, net_returns, run_backtest, turnover
from oryxis.portfolio.market import MarketView
from oryxis.rivals.common import daily_scores, dedupe_headlines, liquid_only, make_strategy, partial_step, sign_legs
from oryxis.rivals.llt.portfolio import llt_scores, llt_strategy
from oryxis.rivals.llt.repro import leg_returns, yearly_sharpe
from oryxis.common.rng import make_rng

__all__ = ["FactorFileMissing", "ScoringInputs", "load_bundle", "load_real_inputs", "run_scoring"]

KEY = "cpu.score_basic"
IN_TITLE, IN_BODY, TICKER_ONLY, NEITHER = "name_in_title", "name_in_body_only", "ticker_in_title_only", "neither"
GROUPS = (IN_TITLE, IN_BODY, TICKER_ONLY, NEITHER)
ANSWER, SCORE = "answer", "score"
SEPARATOR = "|"
HALF = 0.5


def sb(cfg: Config, path: str):
    return cfg.get(f"{KEY}.{path}")


# --- rows ------------------------------------------------------------------------------------------------------


class Rows:
    """Long-format results. A missing value is written as the configured "not computed" label with its reason."""

    COLUMNS = ["section", "set", "window", "variant", "cost", "fill", "metric", "value", "ci_low", "ci_high", "n", "n_dates", "label", "note"]

    def __init__(self, cfg: Config) -> None:
        self.cfg, self.rows = cfg, []
        self.not_computed = str(sb(cfg, "labels.not_computed"))

    def label(self, window: str) -> str:
        exploratory = window in sb(self.cfg, "exploratory_windows")
        return str(sb(self.cfg, "labels.exploratory" if exploratory else "labels.in_sample"))

    def add(self, section, metric, value, *, set="", window="", variant="", cost="", fill="", ci=(None, None), n=None, n_dates=None, note="") -> None:
        missing = value is None or (isinstance(value, (float, np.floating)) and not np.isfinite(value))
        self.rows.append({
            "section": section, "set": set, "window": window, "variant": variant, "cost": cost, "fill": fill, "metric": metric,
            "value": self.not_computed if missing else value, "ci_low": ci[0], "ci_high": ci[1], "n": n, "n_dates": n_dates,
            "label": self.label(window) if window else str(sb(self.cfg, "labels.in_sample")),
            "note": note if (note or not missing) else self.not_computed,
        })

    def frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.rows, columns=self.COLUMNS)


# --- inputs ----------------------------------------------------------------------------------------------------


@dataclass
class ScoringInputs:
    cfg: Config
    bucket: str
    llm: str
    calendar: TradingCalendar
    windows: BucketWindows
    clean: pd.Timestamp
    units: pd.DataFrame        # unit_id, security_id, company_name, headline, decision_date (prestaged)
    events: pd.DataFrame       # unit_id, available_utc, text
    aliases: pd.DataFrame      # node_id, alias, alias_type
    snapshots: pd.DataFrame    # snapshot_date, security_id
    market: ScoreMarket


def load_bundle(path: Path) -> tuple[dict, pd.DataFrame, dict]:
    check = verify_bundle(path)
    if not check.ok:
        raise ValueError(f"bundle failed verification: {check.problems}")
    with tempfile.TemporaryDirectory() as folder:
        shutil.copy(path, Path(folder) / f"x{path.suffix}")
        tape = read_consolidated(scan_bundles([folder], check.manifest["job_id"]).valid)
    with zipfile.ZipFile(path) as archive:
        state = json.loads(archive.read("state/state.json")) if "state/state.json" in archive.namelist() else {}
    return check.manifest, tape, state


def load_real_inputs(root: Path, cfg: Config, bucket: str, llm: str) -> ScoringInputs:
    upload = root / str(cfg.get("cpu.folders.upload"))
    calendar = TradingCalendar.from_config(cfg)
    windows = bucket_windows(cfg, bucket, calendar)
    clean = clean_start(str(cfg.get(f"llms.models.{llm}.stated_cutoff")), calendar)
    units = pd.read_parquet(upload / "prestaged" / f"{sb(cfg, 'job')}-{bucket}" / "units.parquet")
    events = pd.read_parquet(upload / "prestaged" / f"reader-{bucket}" / "units.parquet", columns=[UNIT_ID, "available_utc", "text"])
    return ScoringInputs(cfg, bucket, llm, calendar, windows, clean, units, events, pd.read_parquet(upload / "aliases.parquet"),
                         pd.read_parquet(upload / "sp500_snapshot.parquet"), load_market(root, cfg, calendar))


# --- units: company mention, dates, universe ---------------------------------------------------------------------


def _pattern(words) -> re.Pattern | None:
    words = sorted({w for w in words if str(w).strip()}, key=len, reverse=True)
    return re.compile("(?<![A-Za-z0-9])(" + SEPARATOR.join(re.escape(w) for w in words) + ")(?![A-Za-z0-9])", re.IGNORECASE) if words else None


def classify_units(frame: pd.DataFrame, aliases: pd.DataFrame, cfg: Config) -> pd.Series:
    """Where the company is named: title (any name alias), ticker in the title only, name in the body only, or neither."""
    types = sb(cfg, "alias_types")
    by_type = {t: aliases[aliases["alias_type"] == t].groupby("node_id")["alias"].apply(list) for t in types.values()}
    out = []
    for row in frame.itertuples():
        name_re, tick_re = _pattern(by_type[types["name"]].get(row.security_id, [])), _pattern(by_type[types["ticker"]].get(row.security_id, []))
        title, body = str(row.headline), str(row.text) if isinstance(row.text, str) else ""
        if name_re and name_re.search(title):
            out.append(IN_TITLE)
        elif tick_re and tick_re.search(title):
            out.append(TICKER_ONLY)
        elif name_re and name_re.search(body):
            out.append(IN_BODY)
        else:
            out.append(NEITHER)
    return pd.Series(out, index=frame.index)


def decision_dates(available_utc: pd.Series, calendar: TradingCalendar, cfg: Config) -> pd.DataFrame:
    """Decision date of each headline: after the close of this session, the fill is the open of the next one.

    Items with a time of day use the project rule (``text_decision_date``). Date-only items count as available at the
    end of their date: the decision is the last session on or before that date (so a weekend date fills at Monday's open).
    """
    stamps = pd.to_datetime(available_utc, utc=True)
    known = time_known(stamps, cfg).to_numpy()
    sessions = calendar.sessions
    last_close = calendar.closes_utc[-1]
    out = pd.Series(pd.NaT, index=available_utc.index, dtype="datetime64[ns]")
    usable = known & (stamps <= last_close).to_numpy()
    if usable.any():
        out[usable] = calendar.text_decision_date(stamps[usable]).to_numpy()
    dates = stamps[~known].dt.tz_localize(None).dt.normalize()
    if len(dates):
        position = sessions.searchsorted(pd.DatetimeIndex(dates), side="right") - 1
        out[~known] = sessions[np.maximum(position, 0)].to_numpy()
    return pd.DataFrame({"decision_date": out, "time_known": known})


def size_layer(security: str, date: pd.Timestamp, members: set, caps: pd.DataFrame, names: list[str]) -> str:
    ids = [s for s in members if s in caps.columns]
    if security not in ids:
        return "no_price"
    proxy = caps.loc[:date].iloc[-1][ids]
    rank = proxy.rank(pct=True).get(security)
    return names[min(int(np.ceil(rank * len(names))) - 1, len(names) - 1)] if pd.notna(rank) else "no_price"


def build_units(tape: pd.DataFrame, inp: ScoringInputs, rows: Rows) -> tuple[pd.DataFrame, dict]:
    """One row per scored unit (group, answer, dates, universe flags, layer) and the members of each decision date. Every drop is counted in ``rows``."""
    cfg, sm_ = inp.cfg, inp.market
    frame = inp.units.merge(inp.events, on=UNIT_ID, how="left").merge(
        tape[[UNIT_ID, STATUS, PARSE_OK, ANSWER, SCORE]], on=UNIT_ID, how="inner")
    rows.add("coverage", "units_in_bundle", len(tape), n=len(tape))
    rows.add("coverage", "units_joined_to_events", len(frame), n=len(tape))
    frame["group"] = classify_units(frame, inp.aliases, cfg)
    frame["decision_date_prestaged"] = pd.to_datetime(frame["decision_date"])
    dates = decision_dates(frame["available_utc"], inp.calendar, cfg)
    frame["decision_date"], frame["time_known"] = dates["decision_date"], dates["time_known"]
    rows.add("timing", "date_only_items", int((~frame["time_known"]).sum()), n=len(frame), note="treated as available at the end of their date")
    rows.add("timing", "no_decision_date", int(frame["decision_date"].isna().sum()), n=len(frame), note="available after the last session; dropped")
    rows.add("timing", "decision_date_differs_from_prestaged", int((frame["decision_date"] != frame["decision_date_prestaged"]).sum()), n=len(frame),
             note="the scorer's own date (project timing rule) is used")
    frame = frame[frame["decision_date"].notna()].copy()
    frame["fill_date"] = [fill_session(inp.calendar.sessions, d) for d in frame["decision_date"]]
    split = inp.windows.label(pd.DatetimeIndex(frame["decision_date"])).to_numpy()
    frame["split"] = split
    frame["part"] = cutoff_part(pd.DatetimeIndex(frame["decision_date"]), inp.windows, inp.clean).to_numpy()
    for label in ("embargo", "outside"):
        rows.add("coverage", f"dropped_{label}", int((frame["split"] == label).sum()), n=len(frame))
    frame = frame[frame["split"].isin(["dev", "test"])].copy()
    members = {d: set(members_on(inp.snapshots, d).security_ids) for d in frame["decision_date"].unique()}
    frame["member"] = [s in members[d] for s, d in zip(frame["security_id"], frame["decision_date"], strict=True)]
    rows.add("coverage", "dropped_not_sp500_member", int((~frame["member"]).sum()), n=len(frame), note="members as of the decision date")
    frame = frame[frame["member"]].copy()
    priced = frame["security_id"].isin(sm_.close.columns)
    rows.add("coverage", "dropped_no_price_series", int((~priced).sum()), n=len(frame))
    frame = frame[priced].copy()
    before = len(frame)
    frame = dedupe_headlines(frame.reset_index(drop=True), float(cfg.get("rivals.llt.osa_similarity_threshold")))
    rows.add("coverage", "dropped_duplicate_headline", before - len(frame), n=before, note="near-identical headline of the same company and decision date")
    caps = (sm_.close * sm_.volume).rolling(int(cfg.get("personas.panel.cap_proxy_window_days")), min_periods=1).mean()
    layer_names = list(cfg.get("cpu.road1_eval.size_layers"))
    frame["layer"] = [size_layer(s, d, members[d], caps, layer_names) for s, d in zip(frame["security_id"], frame["decision_date"], strict=True)]
    return frame.reset_index(drop=True), members


def attach_abnormal(frame: pd.DataFrame, members: dict, inp: ScoringInputs) -> dict[int, pd.DataFrame]:
    """Abnormal forward returns per horizon (rows = fill sessions), attached to the units as ``ar<h>``."""
    panels = {}
    for horizon in inp.cfg.get("eval.ic_horizons"):
        panel = abnormal_forward(inp.market, list(frame["decision_date"]), members, int(horizon), inp.cfg)
        r, c = panel.index.get_indexer(frame["fill_date"]), panel.columns.get_indexer(frame["security_id"])
        found = (r >= 0) & (c >= 0)
        values = np.full(len(frame), np.nan)
        values[found] = panel.to_numpy()[r[found], c[found]]
        frame[f"ar{horizon}"] = values
        panels[int(horizon)] = panel
    return panels


# --- windows -----------------------------------------------------------------------------------------------------


def window_spans(inp: ScoringInputs) -> dict[str, tuple[pd.Timestamp, pd.Timestamp] | None]:
    w, cal = inp.windows, inp.calendar
    spans: dict[str, tuple[pd.Timestamp, pd.Timestamp] | None] = {"dev": (w.dev_start, w.dev_end), "test_full": (w.test_start, w.test_end)}
    spans["post_cutoff"] = (max(inp.clean, w.test_start), w.test_end) if inp.clean <= w.test_end else None
    spans["in_cutoff"] = (w.test_start, min(cal.shift(inp.clean, -1), w.test_end)) if inp.clean > w.test_start else None
    return {name: spans[name] for name in sb(inp.cfg, "windows")}


def in_span(dates: pd.Series, span) -> pd.Series:
    return (dates >= span[0]) & (dates <= span[1])


# --- signal metrics ----------------------------------------------------------------------------------------------


def boot_ratio(hits: pd.Series, dates: pd.Series, cfg: Config, parts: tuple) -> tuple[float, float, float]:
    frame = pd.DataFrame({"h": hits.astype(float).to_numpy(), "d": dates.to_numpy()}).groupby("d")["h"].agg(["sum", "count"])
    share = float(frame["sum"].sum() / frame["count"].sum())
    if len(frame) < 2:
        return share, np.nan, np.nan
    level, draws = float(cfg.get("eval.bootstrap.ci_level")), int(cfg.get("eval.bootstrap.draws"))
    idx = make_rng(*parts).integers(0, len(frame), size=(draws, len(frame)))
    ratios = frame["sum"].to_numpy()[idx].sum(axis=1) / frame["count"].to_numpy()[idx].sum(axis=1)
    return share, float(np.quantile(ratios, (1 - level) / 2)), float(np.quantile(ratios, 1 - (1 - level) / 2))


def clustered_regression(y: pd.Series, x: pd.Series, dates: pd.Series, min_clusters: int) -> dict[str, float]:
    data = pd.DataFrame({"y": y.to_numpy(), "x": x.to_numpy(), "d": dates.to_numpy()}).dropna()
    if data["d"].nunique() < min_clusters or data["x"].nunique() < 2:
        return {"slope": np.nan, "se": np.nan, "t": np.nan, "n": len(data), "clusters": data["d"].nunique()}
    fit = sm.OLS(data["y"], sm.add_constant(data["x"])).fit(cov_type="cluster", cov_kwds={"groups": pd.factorize(data["d"])[0]})
    return {"slope": float(fit.params["x"]), "se": float(fit.bse["x"]), "t": float(fit.tvalues["x"]), "n": int(fit.nobs), "clusters": int(data["d"].nunique())}


def cluster_mean(values: pd.Series, dates: pd.Series) -> tuple[float, float, int]:
    v = values.dropna()
    n = len(v)
    if n < 2:
        return (float(v.mean()) if n else np.nan), np.nan, n
    sums = (v - v.mean()).groupby(dates[v.index]).sum()
    return float(v.mean()), float(np.sqrt((sums**2).sum()) / n), n


def signal_block(rows: Rows, name: str, window: str, units: pd.DataFrame, panels: dict[int, pd.DataFrame], inp: ScoringInputs) -> None:
    cfg = inp.cfg
    n_dates = int(units["decision_date"].nunique())
    rows.add("signal", "units_scored", len(units), set=name, window=window, n=len(units), n_dates=n_dates)
    if units.empty:
        return
    level = float(cfg.get("eval.bootstrap.ci_level"))
    z = norm.ppf(1 - (1 - level) / 2)
    firm_day = llt_scores_frame(units, str(sb(cfg, "aggregation")))
    rows.add("signal", "firm_days_scored", len(firm_day), set=name, window=window, n=len(firm_day), n_dates=int(firm_day["decision_date"].nunique()))
    for horizon in sorted(panels):
        col, tag = f"ar{horizon}", f"h{horizon}"
        valid = units[units[col].notna()]
        rows.add("signal", "units_with_return", len(valid), set=name, window=window, fill=tag, n=len(units), n_dates=int(valid["decision_date"].nunique()))
        for answer, positive in (("YES", True), ("NO", False)):
            part = valid[valid[ANSWER] == answer]
            hits = part[col] > 0 if positive else part[col] < 0
            share, lo, hi = boot_ratio(hits, part["decision_date"], cfg, (cfg.get("runtime.base_seed"), "hit", name, window, answer, horizon)) if len(part) else (np.nan, None, None)
            rows.add("signal", f"hit_rate_{answer}", share, set=name, window=window, fill=tag, ci=(lo, hi), n=len(part), n_dates=int(part["decision_date"].nunique()),
                     note="YES: return > 0, NO: return < 0 (abnormal); CI resamples decision dates")
        for answer in ("YES", "NO", "UNKNOWN"):
            part = valid[valid[ANSWER] == answer]
            mean, se, n = cluster_mean(part[col], part["decision_date"])
            rows.add("signal", f"mean_abnormal_return_{answer}", mean, set=name, window=window, fill=tag,
                     ci=(mean - z * se, mean + z * se) if np.isfinite(se) else (None, None), n=n, n_dates=int(part["decision_date"].nunique()),
                     note=f"date-clustered SE {se:.6f}" if np.isfinite(se) else "")
        reg = clustered_regression(valid[col], valid[SCORE], valid["decision_date"], int(sb(cfg, "min_clusters")))
        rows.add("signal", "score_regression_slope", reg["slope"], set=name, window=window, fill=tag,
                 ci=(reg["slope"] - z * reg["se"], reg["slope"] + z * reg["se"]) if np.isfinite(reg["se"]) else (None, None), n=reg["n"], n_dates=reg["clusters"],
                 note=f"t = {reg['t']:.3f}, date-clustered" if np.isfinite(reg["t"]) else f"fewer than {sb(cfg, 'min_clusters')} decision dates")
        fd = firm_day.merge(valid[["decision_date", "security_id", col]].drop_duplicates(["decision_date", "security_id"]), on=["decision_date", "security_id"], how="left")
        forward = panels[horizon]
        ic, n_days, months = pooled_ic(firm_day, forward, cfg)
        note = f"{n_days} days scored on their own, {len(ic) - n_days} months pooled (fewer than {cfg.get('eval.ic_min_stocks')} stocks a day), {months} months with signals"
        enough = len(ic) >= int(sb(cfg, "min_ic_dates"))
        few = "" if enough else f"; {len(ic)} observations, fewer than {sb(cfg, 'min_ic_dates')}: no ICIR or t-statistics"
        rows.add("signal", "rank_ic", float(ic.mean()) if len(ic) else np.nan, set=name, window=window, fill=tag, n=len(fd), n_dates=len(ic), note=note + few)
        if enough:
            rep = ic_report(ic, int(cfg.get("eval.hac_lag")))
            for key in ("icir", "t_naive", "t_newey_west"):
                rows.add("signal", f"rank_{key}", rep[key], set=name, window=window, fill=tag, n=len(fd), n_dates=rep["n_dates"], note=note)
        else:
            rows.add("signal", "rank_icir", None, set=name, window=window, fill=tag, n=len(fd), n_dates=len(ic), note=note + few)


def pooled_ic(firm_day: pd.DataFrame, forward: pd.DataFrame, cfg: Config) -> tuple[pd.Series, int, int]:
    """Rank IC series (days with enough stocks on their own, other days pooled to their month) and the counts behind it.

    Returns the series, how many entries are single days, and how many calendar months carry any signal.
    """
    minimum = int(cfg.get("eval.ic_min_stocks"))
    ic = daily_rank_ic(firm_day[["decision_date", "security_id", SCORE]], forward, minimum)
    per_day = firm_day.groupby("decision_date").size()
    enough = set(per_day[per_day >= minimum].index)
    return ic, int(sum(d in enough for d in ic.index)), int(firm_day["decision_date"].dt.to_period("M").nunique())


def llt_scores_frame(units: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    return daily_scores(units[["decision_date", "security_id", SCORE]].astype({SCORE: float}), aggregation)


# --- portfolio -----------------------------------------------------------------------------------------------------


def short_only_strategy(scores_by_date, cfg: Config, *, rebalance: float, min_firms: int, screen: bool):
    """The short leg of the LLT rule alone, scaled to the full gross exposure (the mirror of ``long_only``)."""
    spec, liquid = cfg.get("rivals.llt"), cfg.get("market.liquid_only")
    window = int(cfg.get("market.etf.dv_window_sessions"))

    def rule(scores: pd.Series, date: pd.Timestamp, view: MarketView, held: pd.Series) -> pd.Series:
        target = sign_legs(scores, float(spec["leg_weight"]), min_firms)
        if screen and len(target):
            target = liquid_only(target, view, float(liquid["min_price"]), float(liquid["min_median_dv_usd"]), window)
        shorts = target[target < 0]
        target = shorts / shorts.abs().sum() * (2.0 * float(spec["leg_weight"])) if len(shorts) else shorts
        return partial_step(held, target, rebalance) if rebalance < 1.0 else target

    return make_strategy(rule, scores_by_date)


def build_strategy(variant: str, scores_by_date, cfg: Config):
    spec = sb(cfg, f"variants.{variant}")
    llt = cfg.get("rivals.llt")
    kwargs = {"rebalance": float(llt[spec["rebalance_key"]]), "min_firms": int(llt["min_firms_per_leg"]), "screen": bool(spec["screen"])}
    if spec["legs"] == "short":
        return short_only_strategy(scores_by_date, cfg, **kwargs)
    return llt_strategy(scores_by_date, cfg, long_leg_only=spec["legs"] == "long", **kwargs)


def cost_models(cfg: Config, sm_: ScoreMarket, omitted: list[int]) -> dict:
    scoring = cfg.get("portfolio.scoring")
    headline_bp = float(scoring["headline_round_trip_bp"])
    models = {"gross": lambda trades, _d: 0.0}
    for bp in scoring["round_trip_bp_grid"]:
        models[f"rt{bp:g}bp"] = lambda trades, _d, bp=bp: fixed_cost(trades.to_numpy(), float(bp), 1.0, cfg)
    for multiple in sb(cfg, "cost_multiples_of_headline"):
        models[f"x{multiple:g}"] = lambda trades, _d, m=multiple: fixed_cost(trades.to_numpy(), headline_bp, float(m), cfg)
    adv, sigma = sm_.adv_and_sigma(cfg)
    aum = float(cfg.get("portfolio.aum.headline_usd"))

    def impact(trades: pd.Series, date: pd.Timestamp) -> float:
        traded = trades[trades != 0]
        a, s = adv.loc[date].reindex(traded.index), sigma.loc[date].reindex(traded.index)
        ok = a.notna() & s.notna() & (a > 0)
        omitted[0] += int((~ok).sum())
        return sqrt_impact_cost(traded[ok].to_numpy(), s[ok].to_numpy(), a[ok].to_numpy(), aum, 1.0, cfg) if ok.any() else 0.0

    models[f"impact_{aum:g}"] = impact
    return models


def headline_cost_name(cfg: Config) -> str:
    return f"rt{float(cfg.get('portfolio.scoring.headline_round_trip_bp')):g}bp"


def small_flag(invested: int, periods: int) -> str:
    """Words that travel with any statistic computed on fewer invested days than a year."""
    return f"invested days {invested}" + ("; too small to interpret: fewer invested days than a year" if invested < periods else "")


@dataclass
class Run:
    result: BacktestResult
    net: pd.DataFrame
    legs: pd.DataFrame
    turnover: pd.Series
    invested: pd.Series
    omitted_impact: int


def run_variant(units: pd.DataFrame, span, variant: str, fill: str, delist: bool, inp: ScoringInputs, cache: dict) -> Run:
    cfg = inp.cfg
    market, panel = inp.market.execution_market(fill, cfg) if fill not in cache else cache[fill]
    cache[fill] = (market, panel)
    scores = llt_scores(units[["decision_date", "security_id", SCORE]].astype({SCORE: float}), str(sb(cfg, "aggregation")))
    dates = [d for d in inp.calendar.sessions if span[0] <= d <= span[1]]
    omitted = [0]
    result = run_backtest(market, dates, build_strategy(variant, scores, cfg),
                          delist_return=float(cfg.get("portfolio.delisting.default_return")) if delist else None, risk_free_daily=None)
    net = net_returns(result, cost_models(cfg, inp.market, omitted))
    borrow = (float(cfg.get("portfolio.scoring.short_borrow_stress_bp_per_year")) * float(cfg.get("portfolio.scoring.bp_to_fraction"))
              / float(cfg.get("eval.annualisation_days")) * result.positions.clip(upper=0.0).abs().sum(axis=1))
    net[f"{headline_cost_name(cfg)}_borrow"] = net[headline_cost_name(cfg)] - borrow
    return Run(result, net, leg_returns(result, panel), turnover(result), result.positions.abs().sum(axis=1) > 0, omitted[0])


def series_rows(rows: Rows, series: pd.Series, run: Run, base: dict, cfg: Config, cost: str, section: str = "portfolio") -> None:
    periods = int(cfg.get("eval.annualisation_days"))
    n, invested = len(series), int(run.invested.sum())
    sd = series.std()
    sr = sharpe(series, periods)
    se = float(np.sqrt(periods) * np.sqrt((1 + HALF * (series.mean() / sd) ** 2) / n)) if sd and sd > 0 else np.nan
    small = "too small to interpret: fewer invested days than a year" if invested < periods else ""
    common = dict(base, cost=cost)
    rows.add(section, "sessions", n, n=n, **common)
    rows.add(section, "invested_days", invested, n=n, **common)
    rows.add(section, "mean_daily", float(series.mean()), n=n, **common)
    rows.add(section, "annual_return", float(series.mean() * periods), n=n, **common)
    rows.add(section, "annual_vol", float(sd * np.sqrt(periods)) if sd else np.nan, n=n, **common)
    rows.add(section, "sharpe", sr, n=n, note="; ".join(x for x in (f"iid SE {se:.2f}" if np.isfinite(se) else "", small) if x), **common)
    rows.add(section, "max_drawdown", max_drawdown(series) if sd else np.nan, n=n, **common)
    rows.add(section, "cvar", cvar(series, float(cfg.get("eval.cvar_level"))) if sd else np.nan, n=n, **common)


def write_parquet(path: Path, frame: pd.DataFrame, meta: dict) -> None:
    table = pa.Table.from_pandas(frame)
    table = table.replace_schema_metadata({**(table.schema.metadata or {}), **{k.encode(): str(v).encode() for k, v in meta.items()}})
    pq.write_table(table, path)


def run_scoring(inp: ScoringInputs, tape: pd.DataFrame, manifest: dict, state: dict, out_dir: Path, log=lambda message: None) -> dict:
    """Score the tape end to end, write every output under ``out_dir``, return the long results table and the text summary."""
    cfg = inp.cfg
    rows = Rows(cfg)
    periods = int(cfg.get("eval.annualisation_days"))
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = sb(cfg, "variants")
    tuning = cfg.get("rivals.llt.tuning")
    grid_size = int(np.prod([len(v) for v in tuning.values()]))
    count_text = f"{len(grid)} variants run, 0 tuned (grid {grid_size}, budget {cfg.get('rivals.tuning_configurations')})"
    meta = {"config_hash": cfg.hash, "cpu_settings_hash": cfg.unhashed_hash, "configuration_count": count_text,
            "cost_setting": sb(cfg, "labels.cost_setting"), "bundle_id": manifest.get("bundle_id"), "bucket": inp.bucket, "llm": inp.llm}

    # run health
    parsed = tape[PARSE_OK].astype(bool)
    rows.add("run", "units_total", len(inp.units), n=len(inp.units))
    rows.add("run", "units_ok", int((tape[STATUS] == Status.OK.value).sum()), n=len(tape))
    rows.add("run", "parse_rate", float(parsed.mean()), n=len(tape))
    rows.add("run", "unparsed_or_refused", int((~parsed).sum()), n=len(tape), note="score 0, counted; refusals are not told apart from other unparseable text")
    for answer in ("YES", "NO", "UNKNOWN"):
        rows.add("run", "answer_share", float((tape[ANSWER] == answer).mean()), set=answer, n=len(tape))
    rows.add("run", "mean_score", float(tape.loc[parsed, SCORE].mean()), n=int(parsed.sum()))
    rows.add("run", "loop_seconds", state.get("elapsed_seconds"), note="worker loop only; engine load time is not in the bundle")
    rows.add("run", "code_version", manifest.get("meta", {}).get("code_version"))
    rows.add("run", "bundle_config_hash", manifest.get("meta", {}).get("config_hash"), note="hash the GPU session recorded (a different, longer hash format)")

    log("units: company mention, dates, universe")
    units, members = build_units(tape, inp, rows)
    for group in GROUPS:
        rows.add("coverage", "units_in_group", int((units["group"] == group).sum()), set=group, n=len(units))
    log("abnormal forward returns (betas)")
    panels = attach_abnormal(units, members, inp)
    spans = window_spans(inp)
    sets = {name: units[units["group"].isin(groups)] for name, groups in sb(cfg, "sets").items()}
    headline_set = str(sb(cfg, "headline_set"))
    for name, span in spans.items():
        if span is None:
            rows.add("coverage", "window_empty", None, window=name, note=f"{sb(cfg, 'labels.not_computed')}: no sessions of this window lie in the LLM's data range")
        else:
            rows.add("coverage", "window_span", f"{span[0].date()} to {span[1].date()}", window=name)
    approximation = str(sb(cfg, "labels.market_approximation"))
    rows.add("honesty", "market_adjustment", approximation)
    rows.add("honesty", "holding_period", str(sb(cfg, "labels.holding")))
    rows.add("honesty", "alpha_dating", str(sb(cfg, "labels.alpha_approximation")))
    rows.add("honesty", "cost_setting", str(sb(cfg, "labels.cost_setting")))
    rows.add("honesty", "configuration_count", count_text)
    rows.add("honesty", "liquid_only_screen", "price and median dollar volume from the market config; the spec's market-cap percentile needs shares outstanding, which are not on disk")

    log("signal metrics")
    # signal metrics: sets, then the groups shown separately
    for name, part in {**sets, **{g: units[units["group"] == g] for g in sb(cfg, "separate_groups")}}.items():
        for window, span in spans.items():
            if span is None:
                continue
            signal_block(rows, name, window, part[in_span(part["decision_date"], span)], panels, inp)

    # portfolio
    pnl: dict[tuple[str, str], list[pd.DataFrame]] = {}
    runs: dict[tuple[str, str, str], Run] = {}
    cache: dict = {}
    headline_variant, headline_cost = str(sb(cfg, "headline_variant")), headline_cost_name(cfg)
    layer_names = list(cfg.get("cpu.road1_eval.size_layers"))
    jobs = []
    for set_name, part in sets.items():
        for window, span in spans.items():
            if span is None:
                continue
            sub = part[in_span(part["decision_date"], span)]
            for variant in grid:
                jobs.append((set_name, window, span, variant, str(sb(cfg, "fills.headline")), True, sub))
            for variant in sb(cfg, "close_fill_variants"):
                jobs.append((set_name, window, span, variant, str(sb(cfg, "fills.sensitivity")), True, sub))
            jobs.append((set_name, window, span, headline_variant + str(sb(cfg, "no_delist_suffix")), str(sb(cfg, "fills.headline")), False, sub))
    for number, (set_name, window, span, variant, fill, delist, sub) in enumerate(jobs, start=1):
        log(f"backtest {number}/{len(jobs)}: {set_name} {window} {variant} {fill}")
        label = variant.removesuffix(str(sb(cfg, "no_delist_suffix")))
        base = dict(set=set_name, window=window, variant=variant, fill=fill)
        if sub.empty:
            rows.add("portfolio", "sharpe", None, note="no signals in this window", **base, cost=headline_cost)
            continue
        run = run_variant(sub, span, label, fill, delist, inp, cache)
        runs[(set_name, window, variant + SEPARATOR + fill)] = run
        for cost in run.net.columns:
            series_rows(rows, run.net[cost], run, base, cfg, cost)
            part = pd.DataFrame({"set": set_name, "window": window, "fill": fill, "net": run.net[cost], "turnover": run.turnover,
                                 "long_leg": run.legs["long"], "short_leg": run.legs["short"]}).rename_axis("fill_date").reset_index()
            pnl.setdefault((variant, cost), []).append(part)
        rows.add("portfolio", "one_way_turnover_per_invested_day", float(run.turnover[run.invested].mean()) if run.invested.any() else np.nan,
                 n=int(run.invested.sum()), **base, cost="")
        rows.add("portfolio", "long_leg_mean_daily", float(run.legs["long"].mean()), n=len(run.legs), **base, cost="")
        rows.add("portfolio", "short_leg_mean_daily", float(run.legs["short"].mean()), n=len(run.legs), **base, cost="")
        rows.add("portfolio", "delisting_hits", int(run.result.delist_hits.sum()), n=len(run.legs), **base, cost="")
        rows.add("portfolio", "missing_return_hits", int(run.result.missing_return_hits.sum()), n=len(run.legs), **base, cost="")
        rows.add("portfolio", "impact_cost_omitted_trades", run.omitted_impact, **base, cost="", note="trades without ADV or volatility history: no impact cost charged, counted here")
        if variant in grid and fill == str(sb(cfg, "fills.headline")) and delist and run.invested.any():
            cost_series = run.net["gross"] - run.net[headline_cost]
            flag = small_flag(int(run.invested.sum()), periods)
            for key, value in bridge(run.net["gross"], cost_series, run.result.trades, periods).items():
                rows.add("bridge", key, value, n=len(run.net), note=flag, **base, cost=headline_cost)
            rows.add("bridge", "break_even_round_trip_bp", break_even_round_trip_bp(run.net["gross"], run.result.trades, float(cfg.get("portfolio.scoring.bp_to_fraction"))),
                     n=len(run.net), note=flag, **base, cost="")
            factors = inp.market.factors
            for model, cols in sb(cfg, "factor_models").items():
                if run.net[headline_cost].std() > 0:
                    out = factor_alpha(run.net[headline_cost], factors[list(cols)], int(cfg.get("eval.hac_lag")), periods)
                    rows.add("alpha", f"alpha_annual_{model}", out["alpha_annual"], n=out["n"], note=f"HAC t = {out['t_alpha']:.3f}, lag {cfg.get('eval.hac_lag')}; {flag}; " + str(sb(cfg, "labels.alpha_approximation")), **base, cost=headline_cost)
            adv, sigma = inp.market.adv_and_sigma(cfg)
            if len(run.result.trades):
                cap = capacity(run.result.trades, adv, run.net["gross"], sigma, cfg)
                for r in cap.itertuples():
                    rows.add("capacity", f"max_adv_share_aum_{r.aum_usd:g}", float(r.max_adv_share), n=len(run.net), **base, cost="")
                    rows.add("capacity", f"annual_return_after_impact_aum_{r.aum_usd:g}", float(r.annual_return_after_impact), n=len(run.net), **base, cost="")
            series = run.net[headline_cost]
            if variant == headline_variant:
                stress = {k: (v["start"], v["end"]) for k, v in sb(cfg, "stress").items()}
                done = set()
                for r in window_metrics(series, stress, periods, float(cfg.get("eval.cvar_level"))).itertuples():
                    done.add(r.window)
                    status = str(sb(cfg, f"stress.{r.window}").get("status", ""))
                    inside = int(run.invested.loc[stress[r.window][0]:stress[r.window][1]].sum())
                    status = "; ".join(x for x in (status, f"invested days in this stress window {inside}" + ("" if inside else ": no position, the zeros are not a result")) if x)
                    rows.add("stress", "total_return", r.total_return, n=r.n_days, **base, cost=headline_cost, note="; ".join(x for x in (r.window, status) if x))
                    rows.add("stress", "max_drawdown", r.max_drawdown, n=r.n_days, **base, cost=headline_cost, note=r.window)
                    rows.add("stress", "cvar", r.cvar, n=r.n_days, **base, cost=headline_cost, note=r.window)
                for k in stress:
                    if k not in done:
                        rows.add("stress", "total_return", None, **base, cost=headline_cost, note=f"{k}: {sb(cfg, 'labels.not_computed')} (outside this window's sample)")
                for year, value in yearly_sharpe(series, periods).items():
                    in_year = series[series.index.year == year]
                    rows.add("robust_year", "sharpe", value, n=len(in_year), **base, cost=headline_cost, note=f"{year}; invested days {int(run.invested[in_year.index].sum())}")

    log("size layers, honesty, reproduction")
    # size layers and groups, headline variant, open fill
    for set_name in (headline_set,):
        for window, span in spans.items():
            if span is None:
                continue
            for layer in layer_names + ["no_price"]:
                sub = sets[set_name][in_span(sets[set_name]["decision_date"], span) & (sets[set_name]["layer"] == layer)]
                base = dict(set=f"{set_name}/{layer}", window=window, variant=headline_variant, fill=str(sb(cfg, "fills.headline")))
                if sub.empty:
                    rows.add("robust_layer", "sharpe", None, **base, cost=headline_cost, note="no signals")
                    continue
                run = run_variant(sub, span, headline_variant, str(sb(cfg, "fills.headline")), True, inp, cache)
                series_rows(rows, run.net[headline_cost], run, base, cfg, headline_cost, section="robust_layer")

    # honesty: smallest detectable Sharpe gap, deflated Sharpe, reproduction pattern
    for window, span in spans.items():
        key = (headline_set, window, headline_variant + SEPARATOR + str(sb(cfg, "fills.headline")))
        if key not in runs:
            continue
        run = runs[key]
        series = run.net[headline_cost]
        sr = sharpe(series, periods)
        years = len(series) / periods
        for rho in cfg.get("eval.power.correlations"):
            if np.isfinite(sr):
                gap = smallest_detectable_gap(sharpe_difference_se(sr, sr, float(rho), years, periods), float(cfg.get("eval.bootstrap.ci_level")))
            else:
                gap = np.nan
            rows.add("honesty", "smallest_detectable_sharpe_gap", gap, set=headline_set, window=window, n=len(series), cost=headline_cost,
                     note=f"correlation {rho}; both Sharpe ratios set to Basic's own net value; {years:.2f} years")
        trial_keys = [(headline_set, window, v + SEPARATOR + str(sb(cfg, "fills.headline"))) for v in grid]
        trial_returns = [runs[k].net[headline_cost] for k in trial_keys if k in runs]
        per_period = np.array([r.mean() / r.std() for r in trial_returns if r.std() > 0])
        if series.std() > 0 and len(per_period) == len(grid):
            dsr = deflated_sharpe(series.to_numpy(), per_period, float(cfg.get("eval.dsr.euler_gamma")))
            rows.add("honesty", "deflated_sharpe_probability", dsr["dsr"], set=headline_set, window=window, n=len(series), cost=headline_cost,
                     note=f"trials {dsr['trials']} (the variants run); per-period expected best from skill-free trials {dsr['sr0']:.4f}")
        else:
            rows.add("honesty", "deflated_sharpe_probability", None, set=headline_set, window=window, n=len(series), cost=headline_cost, note="a variant has no variation in this window")
        for set_name in sets:
            k = (set_name, window, headline_variant + SEPARATOR + str(sb(cfg, "fills.headline")))
            if k not in runs:
                continue
            r = runs[k]
            sharpes = {c: sharpe(r.net[c], periods) for c in ["gross"] + [f"rt{bp:g}bp" for bp in cfg.get("portfolio.scoring.round_trip_bp_grid")]}
            values = list(sharpes.values())
            a = np.isfinite(values[0]) and values[0] > 0
            b = all(np.isfinite(v) for v in values) and all(values[i] > values[i + 1] for i in range(len(values) - 1))
            c = r.legs["short"].mean() > r.legs["long"].mean()
            small = "; too small to interpret" if int(r.invested.sum()) < periods else ""
            text = ", ".join(f"{k_}: {v:.3f}" for k_, v in sharpes.items())
            for metric, ok, detail in (("gross_sharpe_positive", a, text), ("sharpe_falls_with_cost", b, text),
                                       ("short_leg_stronger_than_long", c, f"short {r.legs['short'].mean():.6f}, long {r.legs['long'].mean():.6f}")):
                rows.add("repro", metric, "matches" if ok else "deviation", set=set_name, window=window, variant=headline_variant, n=len(r.net), note=detail + small)
            rows.add("repro", "overall", "matches" if (a and b and c) else "deviation", set=set_name, window=window, variant=headline_variant, n=len(r.net),
                     note="published pattern only; absolute numbers are never compared with the paper" + small)

    table = rows.frame()
    table.insert(0, "config_hash", cfg.hash[: int(sb(cfg, "hash_chars"))])
    table["configuration_count"] = count_text
    table["cost_setting"] = str(sb(cfg, "labels.cost_setting"))
    outputs = sb(cfg, "output")
    write_parquet(out_dir / outputs["units"], units.drop(columns=[c for c in ("text",) if c in units]), meta)
    firm_days = {name: llt_scores_frame(part, str(sb(cfg, "aggregation"))).assign(set=name) for name, part in sets.items()}
    write_parquet(out_dir / outputs["signals"], pd.concat(firm_days.values(), ignore_index=True), meta)
    for (variant, cost), parts in pnl.items():
        write_parquet(out_dir / outputs["pnl"].format(variant=variant, cost=cost), pd.concat(parts, ignore_index=True), meta)
    table.to_csv(out_dir / outputs["summary_csv"], index=False)
    (out_dir / outputs["metrics"]).write_text(json.dumps({"meta": meta, "rows": json.loads(table.to_json(orient="records"))}, indent=1, default=str), encoding="utf-8")
    (out_dir / outputs["hash_file"]).write_text("\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n", encoding="utf-8")
    text = summary_markdown(table, meta, cfg)
    (out_dir / outputs["summary_md"]).write_text(text, encoding="utf-8")
    return {"table": table, "text": text, "meta": meta, "units": units, "runs": runs}


# --- summary ---------------------------------------------------------------------------------------------------------


def verdict_words(text: str, cfg: Config) -> list[str]:
    words = {w.lower() for w in cfg.get("eval.report.forbidden_words")}
    return sorted(w for w in words if re.search(rf"\b{re.escape(w)}\b", text, re.IGNORECASE))


def _fmt(value, decimals: int) -> str:
    return f"{value:.{decimals}f}" if isinstance(value, (float, np.floating)) else ("" if value is None or (isinstance(value, float) and np.isnan(value)) else str(value))


def md_table(frame: pd.DataFrame, decimals: int) -> str:
    if frame.empty:
        return "(none)\n"
    shown = frame.copy()
    for column in ("n", "n_dates"):
        if column in shown:
            shown[column] = shown[column].map(lambda v: "" if pd.isna(v) else str(int(v)))
    shown = shown.map(lambda v: _fmt(v, decimals))
    lines = ["| " + " | ".join(shown.columns) + " |", "|" + "|".join("---" for _ in shown.columns) + "|"]
    lines += ["| " + " | ".join(str(v).replace("|", "/") for v in r) + " |" for r in shown.itertuples(index=False)]
    return "\n".join(lines) + "\n"


def summary_markdown(table: pd.DataFrame, meta: dict, cfg: Config) -> str:
    d = int(cfg.get("eval.report.decimals"))
    headline_set, headline_variant, headline_cost = str(sb(cfg, "headline_set")), str(sb(cfg, "headline_variant")), headline_cost_name(cfg)
    open_fill = str(sb(cfg, "fills.headline"))
    parts = [f"# Basic (Lopez-Lira and Tang) scored on {meta['bucket']} / {meta['llm']}\n",
             f"- config hash: `{meta['config_hash']}`  |  cpu settings hash: `{meta['cpu_settings_hash']}`",
             f"- configurations: {meta['configuration_count']}",
             f"- cost grid: {meta['cost_setting']}",
             f"- bundle: `{meta['bundle_id']}`",
             f"- every row below: {sb(cfg, 'labels.in_sample')}; dev rows: {sb(cfg, 'labels.exploratory')}\n"]
    for title, section in (("Run health and coverage", ["run", "coverage", "timing"]), ("Honesty notes", ["honesty"])):
        sub = table[table["section"].isin(section)][["section", "set", "window", "metric", "value", "n", "note"]]
        parts += [f"## {title}\n", md_table(sub, d)]
    sig = table[(table["section"] == "signal") & table["metric"].isin(["units_scored", "hit_rate_YES", "hit_rate_NO", "rank_ic", "rank_icir", "score_regression_slope"])]
    parts += ["## Signal (abnormal returns; h1 = open of day +1 to its close, h5 = to the close of day +5)\n",
              md_table(sig[["set", "window", "fill", "metric", "value", "ci_low", "ci_high", "n", "n_dates", "note"]].rename(columns={"fill": "horizon"}), d)]
    pf = table[(table["section"] == "portfolio") & (table["cost"] == headline_cost) & (table["metric"].isin(["sharpe", "annual_return", "invested_days"]))
               & (table["fill"] == open_fill) & (table["window"] != "dev")]
    parts += [f"## Portfolio at {headline_cost}, fill {open_fill}\n", md_table(pf[["set", "window", "variant", "metric", "value", "n", "note"]], d)]
    grid = table[(table["section"] == "portfolio") & (table["variant"] == headline_variant) & (table["metric"] == "sharpe") & (table["fill"] == open_fill) & (table["set"] == headline_set)]
    parts += [f"## Headline variant across the cost grid ({headline_set})\n", md_table(grid[["window", "cost", "value", "n", "note"]], d)]
    for title, section in (("Gross-to-net bridge", "bridge"), ("Factor alpha", "alpha"), ("Stress windows", "stress"), ("Capacity", "capacity"),
                           ("Robustness: year by year", "robust_year"), ("Robustness: size layer", "robust_layer"), ("Reproduction pattern", "repro")):
        sub = table[(table["section"] == section) & (table["set"].str.startswith(headline_set) | (table["set"] == ""))]
        parts += [f"## {title}\n", md_table(sub[["set", "window", "fill", "metric", "value", "n", "note"]], d)]
    text = "\n".join(parts)
    bad = verdict_words(text, cfg)
    if bad:
        raise ValueError(f"verdict words in the summary: {bad}")
    return text
