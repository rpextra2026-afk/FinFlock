"""CPU scoring of the FinRipple-lite rival: bundle -> mapped signals -> published rule -> common metrics + the paper's own yardsticks.

Same machinery and output format as ``score_basic`` (shared helpers are imported from it; Basic's numbers are untouched). Realised returns are
read in scoring code only. A signal is one (event, affected company, impact score): the model's company name is mapped back to an S&P security
(ticker in parentheses first, then the normalised name; ambiguous or unknown names are dropped and counted). Every setting is in
``config/cpu.yaml`` (sections ``score_finripple`` and the shared ``score_basic``). Nothing is imputed; numbers only, no verdict wording.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import f_oneway, norm

from finflock.common.config import Config
from finflock.common.schema import PARSE_OK, STATUS, TOKENS_OUT, UNIT_ID, Status
from finflock.common.windows import bucket_windows, clean_start, cutoff_part
from finflock.data.f13 import normalise_name
from finflock.data.sp500 import members_on
from finflock.common.timing import TradingCalendar
from finflock.eval.dsr_pbo import deflated_sharpe
from finflock.eval.portfolio_metrics import (
    bridge, break_even_round_trip_bp, capacity, factor_alpha, sharpe, window_metrics,
)
from finflock.eval.score_basic import (
    SEPARATOR, Rows, Run, ScoringInputs, attach_abnormal, boot_ratio, clustered_regression, cluster_mean, cost_models, decision_dates,
    headline_cost_name, in_span, load_bundle, md_table, pooled_ic, sb, series_rows, size_layer, small_flag, verdict_words, window_spans,
    write_parquet,
)
from finflock.eval.score_market import FactorFileMissing, fill_session, load_market
from finflock.eval.signal_metrics import ic_report
from finflock.eval.stats import sharpe_difference_se, smallest_detectable_gap
from finflock.portfolio.execute import net_returns, run_backtest, turnover
from finflock.portfolio.market import MarketView
from finflock.rivals.common import daily_scores, make_strategy, sign_legs, top_bottom_legs
from finflock.rivals.finripple_lite.masked import MASK_MAP, SOURCE, label_in
from finflock.rivals.finripple_lite.repro import impact_regression
from finflock.rivals.finripple_lite.run import CUT, QUOTE_STYLE, recover_answer
from finflock.rivals.llt.repro import leg_returns, yearly_sharpe

__all__ = ["FactorFileMissing", "NameMapper", "load_bundle", "load_real_inputs", "run_scoring"]

KEY = "cpu.score_finripple"
NAMED, RIPPLE = "named", "ripple"
AFFECTED, RAW = "affected_json", "raw_output"
SCORE = "score"
HALF = 0.5
MAPPED, AMBIGUOUS, UNMAPPED = "mapped", "ambiguous", "unmapped"


def fr(cfg: Config, path: str):
    return cfg.get(f"{KEY}.{path}")


# --- company name -> security --------------------------------------------------------------------------------------


class NameMapper:
    """Maps the company a model names back to an S&P security: ticker in parentheses, then normalised name, then a bare ticker."""

    def __init__(self, aliases: pd.DataFrame, cfg: Config) -> None:
        self.prefix = str(cfg.get("reader.masked_company_prefix"))
        self.suffixes = set(cfg.get("f13.name_match.suffixes"))
        self.paren = re.compile(str(fr(cfg, "mapping.parentheses")))
        self.split = re.compile(str(fr(cfg, "mapping.split_tickers")))
        self.bare_max = int(fr(cfg, "mapping.bare_ticker_max_len"))
        tickers = aliases[aliases["alias_type"] == "ticker"]
        self.tickers = {str(a).strip().upper(): n for n, a in zip(tickers["node_id"], tickers["alias"], strict=True)}
        self.tickers.update({str(k).upper(): v for k, v in dict(fr(cfg, "mapping.extra_tickers") or {}).items()})
        owners: dict[str, set] = defaultdict(set)
        for node, alias in zip(*[aliases.loc[aliases["alias_type"] == "name", c] for c in ("node_id", "alias")], strict=True):
            owners[normalise_name(alias, self.suffixes)].add(node)
        owners.pop("", None)
        self.names = {k: next(iter(v)) for k, v in owners.items() if len(v) == 1}
        extra = {normalise_name(k, self.suffixes): v for k, v in dict(fr(cfg, "mapping.extra_names") or {}).items()}
        self.names.update(extra)
        self.ambiguous = {k for k, v in owners.items() if len(v) > 1} - set(extra)

    def map(self, text: str) -> tuple[str | None, str]:
        text = str(text).strip()
        inside = [piece for group in self.paren.findall(text) for piece in self.split.split(group.upper()) if piece]
        hits = {self.tickers[piece] for piece in inside if piece in self.tickers}
        if len(hits) == 1:
            return next(iter(hits)), MAPPED
        bare = self.paren.sub(" ", text)
        key = normalise_name(bare, self.suffixes)
        if key in self.names:
            return self.names[key], MAPPED
        if key in self.ambiguous or len(hits) > 1:
            return None, AMBIGUOUS
        if re.fullmatch(r"[A-Z][A-Z0-9.\-]*", bare.strip()) and len(bare.strip()) <= self.bare_max and bare.strip() in self.tickers:
            return self.tickers[bare.strip()], MAPPED
        return None, UNMAPPED


# --- inputs --------------------------------------------------------------------------------------------------------


EVENT_COLUMNS = [UNIT_ID, "available_utc", "title", "text", "pins_json"]


def load_real_inputs(root: Path, cfg: Config, bucket: str, llm: str, units_dir: Path | None = None) -> ScoringInputs:
    """``units_dir``: a folder holding ``prestaged/finripple-<bucket>/units.parquet`` other than the upload folder (the masked-names run)."""
    upload = root / str(cfg.get("cpu.folders.upload"))
    calendar = TradingCalendar.from_config(cfg)
    windows = bucket_windows(cfg, bucket, calendar)
    clean = clean_start(str(cfg.get(f"llms.models.{llm}.stated_cutoff")), calendar)
    units = pd.read_parquet((units_dir or upload) / "prestaged" / f"{fr(cfg, 'job')}-{bucket}" / "units.parquet")
    units = units[[c for c in (UNIT_ID, "asof_date", SOURCE, MASK_MAP) if c in units]]
    events = pd.read_parquet(upload / "prestaged" / f"reader-{bucket}" / "units.parquet", columns=EVENT_COLUMNS)
    events = events[events[UNIT_ID].isin(units[SOURCE] if SOURCE in units else units[UNIT_ID])]
    return ScoringInputs(cfg, bucket, llm, calendar, windows, clean, units, events, pd.read_parquet(upload / "aliases.parquet"),
                         pd.read_parquet(upload / "sp500_snapshot.parquet"), load_market(root, cfg, calendar))


# --- tape -> signals -------------------------------------------------------------------------------------------------


def recover_answers(tape: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict[str, int]]:
    """Answers that did not parse as a whole keep the complete company entries they hold (parse_ok -> True): the entries before a cut-off, and, with
    ``salvage_quote_style``, the entries of answers written in a quote style that is neither JSON nor a Python literal. Returns the tape and the counts per
    mechanism (``cut``, ``quote_style``)."""
    counts = {CUT: 0, QUOTE_STYLE: 0}
    if not fr(cfg, "salvage_cut_answers") or RAW not in tape:
        return tape, counts
    tape = tape.copy()
    for index in tape.index[~tape[PARSE_OK].fillna(False).astype(bool)]:
        items, how = recover_answer(str(tape.at[index, RAW]), cfg)
        if items:
            tape.at[index, AFFECTED], tape.at[index, PARSE_OK] = json.dumps(items, sort_keys=True), True
            counts[how] += 1
    return tape, counts


def recover_cut_answers(tape: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, int]:
    """``recover_answers`` with one count (all mechanisms)."""
    tape, counts = recover_answers(tape, cfg)
    return tape, sum(counts.values())


def named_companies(pins_json: str) -> set:
    return {p["node_id"] for p in json.loads(pins_json) if p.get("node_type") == "company"} if isinstance(pins_json, str) else set()


def explode_answers(tape: pd.DataFrame, events: pd.DataFrame, mapper: NameMapper, rows: Rows, units: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (event, named company in the answer) with the mapping outcome; the counts go to ``rows``.

    Masked-names run (``units`` has the source id and the label table): the answer names ``COMPANY_C``; the stored table maps it back. A real
    company name in such an answer is a leak of the masking: it is counted and not used.
    """
    pins = dict(zip(events[UNIT_ID], events["pins_json"], strict=True))
    masked = units is not None and SOURCE in units and MASK_MAP in units
    source = dict(zip(units[UNIT_ID], units[SOURCE], strict=True)) if masked else {}
    tables = {u: json.loads(m) for u, m in zip(units[UNIT_ID], units[MASK_MAP], strict=True)} if masked else {}
    prefix = str(mapper.prefix) if masked else ""
    out, items, parsed_events, real_names = [], 0, 0, 0
    for record in tape[tape[PARSE_OK].fillna(False).astype(bool)].itertuples():
        unit = getattr(record, UNIT_ID)
        event = source.get(unit, unit)
        parsed_events += 1
        named = named_companies(pins.get(event))
        for item in json.loads(getattr(record, AFFECTED)):
            items += 1
            if masked:
                label = label_in(item["name"], prefix)
                security, outcome = (tables[unit].get(label), MAPPED) if label in tables[unit] else (None, UNMAPPED)
                real_names += int(label is None and mapper.map(item["name"])[0] is not None)
            else:
                security, outcome = mapper.map(item["name"])
            out.append({UNIT_ID: event, "security_id": security, "outcome": outcome, "score": float(item[SCORE]),
                        "impact_type": item.get("impact_type", ""), "answer_name": item["name"],
                        "relation": NAMED if security in named else RIPPLE})
    frame = pd.DataFrame(out, columns=[UNIT_ID, "security_id", "outcome", SCORE, "impact_type", "answer_name", "relation"])
    rows.add("mapping", "answers_parsed", parsed_events, n=len(tape))
    rows.add("mapping", "companies_named_by_the_model", items, n=parsed_events)
    rows.add("mapping", "names_per_answer", items / parsed_events if parsed_events else np.nan, n=parsed_events)
    for outcome in (MAPPED, AMBIGUOUS, UNMAPPED):
        count = int((frame["outcome"] == outcome).sum())
        rows.add("mapping", f"names_{outcome}", count, n=items)
        rows.add("mapping", f"share_{outcome}", count / items if items else np.nan, n=items)
    if masked:
        rows.add("mapping", "real_company_names_in_masked_answers", real_names, n=items, note="names that match a company although the input was masked: not used as signals")
    return frame


def build_signals(tape: pd.DataFrame, inp: ScoringInputs, mapper: NameMapper, rows: Rows) -> tuple[pd.DataFrame, dict]:
    """Signal rows inside the dev/test windows, S&P members with prices, with dates, split, part and size layer. Every drop is counted."""
    cfg, sm_ = inp.cfg, inp.market
    answers = explode_answers(tape, inp.events, mapper, rows, inp.units)
    answers = answers[answers["outcome"] == MAPPED]
    rows.add("coverage", "mapped_signals", len(answers), n=len(answers))
    dates = decision_dates(inp.events["available_utc"], inp.calendar, cfg)
    events = inp.events[[UNIT_ID]].assign(decision_date=dates["decision_date"].to_numpy(), time_known=dates["time_known"].to_numpy())
    rows.add("timing", "date_only_events", int((~events["time_known"]).sum()), n=len(events), note="treated as available at the end of their date")
    rows.add("timing", "no_decision_date", int(events["decision_date"].isna().sum()), n=len(events), note="available after the last session; dropped")
    frame = answers.merge(events, on=UNIT_ID, how="left")
    frame = frame[frame["decision_date"].notna()].copy()
    frame["fill_date"] = [fill_session(inp.calendar.sessions, d) for d in frame["decision_date"]]
    frame["split"] = inp.windows.label(pd.DatetimeIndex(frame["decision_date"])).to_numpy()
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
    caps = (sm_.close * sm_.volume).rolling(int(cfg.get("personas.panel.cap_proxy_window_days")), min_periods=1).mean()
    layer_names = list(cfg.get("cpu.road1_eval.size_layers"))
    frame["layer"] = [size_layer(s, d, members[d], caps, layer_names) for s, d in zip(frame["security_id"], frame["decision_date"], strict=True)]
    for relation in (NAMED, RIPPLE):
        rows.add("coverage", f"signals_{relation}", int((frame["relation"] == relation).sum()), n=len(frame))
    return frame.reset_index(drop=True), members


# --- signal metrics ----------------------------------------------------------------------------------------------------


def firm_days(signals: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    return daily_scores(signals[["decision_date", "security_id", SCORE]].astype({SCORE: float}), aggregation)


def anova(score: pd.Series, residual: pd.Series, minimum: int) -> dict[str, float]:
    """One-way ANOVA of the standardised residual across predicted-score groups: F, p and eta squared (the paper's second yardstick)."""
    data = pd.DataFrame({"s": score.to_numpy(), "e": residual.to_numpy()}).dropna()
    groups = [g["e"].to_numpy() for _, g in data.groupby("s") if len(g) >= minimum]
    if len(groups) < 2:
        return {"f": np.nan, "p": np.nan, "eta2": np.nan, "groups": len(groups)}
    f, p = f_oneway(*groups)
    pooled = np.concatenate(groups)
    ss_total = float(((pooled - pooled.mean()) ** 2).sum())
    ss_between = float(sum(len(g) * (g.mean() - pooled.mean()) ** 2 for g in groups))
    return {"f": float(f), "p": float(p), "eta2": ss_between / ss_total if ss_total > 0 else np.nan, "groups": len(groups)}


def signal_block(rows: Rows, name: str, window: str, signals: pd.DataFrame, panels: dict[int, pd.DataFrame], inp: ScoringInputs) -> None:
    cfg = inp.cfg
    n_dates = int(signals["decision_date"].nunique())
    rows.add("signal", "signals_scored", len(signals), set=name, window=window, n=len(signals), n_dates=n_dates)
    if signals.empty:
        return
    level = float(cfg.get("eval.bootstrap.ci_level"))
    z = norm.ppf(1 - (1 - level) / 2)
    aggregation = str(sb(cfg, "aggregation"))
    fd = firm_days(signals, aggregation)
    rows.add("signal", "firm_days_scored", len(fd), set=name, window=window, n=len(fd), n_dates=int(fd["decision_date"].nunique()))
    for horizon in sorted(panels):
        col, tag = f"ar{horizon}", f"h{horizon}"
        valid = signals[signals[col].notna() & (signals[SCORE] != 0)]
        rows.add("signal", "signals_with_return", int(signals[col].notna().sum()), set=name, window=window, fill=tag, n=len(signals),
                 n_dates=int(signals.loc[signals[col].notna(), "decision_date"].nunique()))
        for label, positive in (("positive", True), ("negative", False)):
            part = valid[valid[SCORE] > 0] if positive else valid[valid[SCORE] < 0]
            hits = part[col] > 0 if positive else part[col] < 0
            share, lo, hi = (boot_ratio(hits, part["decision_date"], cfg, (cfg.get("runtime.base_seed"), "fr_hit", name, window, label, horizon))
                             if len(part) else (np.nan, None, None))
            rows.add("signal", f"hit_rate_{label}", share, set=name, window=window, fill=tag, ci=(lo, hi), n=len(part),
                     n_dates=int(part["decision_date"].nunique()), note="positive score: return > 0, negative score: return < 0 (abnormal); CI resamples decision dates")
        for label, part in (("positive", valid[valid[SCORE] > 0]), ("negative", valid[valid[SCORE] < 0])):
            mean, se, n = cluster_mean(part[col], part["decision_date"])
            rows.add("signal", f"mean_abnormal_return_{label}", mean, set=name, window=window, fill=tag,
                     ci=(mean - z * se, mean + z * se) if np.isfinite(se) else (None, None), n=n, n_dates=int(part["decision_date"].nunique()),
                     note=f"date-clustered SE {se:.6f}" if np.isfinite(se) else "")
        reg = clustered_regression(signals[col], signals[SCORE], signals["decision_date"], int(sb(cfg, "min_clusters")))
        rows.add("signal", "score_regression_slope", reg["slope"], set=name, window=window, fill=tag,
                 ci=(reg["slope"] - z * reg["se"], reg["slope"] + z * reg["se"]) if np.isfinite(reg["se"]) else (None, None), n=reg["n"], n_dates=reg["clusters"],
                 note=f"t = {reg['t']:.3f}, date-clustered" if np.isfinite(reg["t"]) else f"fewer than {sb(cfg, 'min_clusters')} decision dates")
        ic, n_days, months = pooled_ic(fd, panels[horizon], cfg)
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
    own_yardsticks(rows, name, window, signals, inp)


def own_yardsticks(rows: Rows, name: str, window: str, signals: pd.DataFrame, inp: ScoringInputs) -> None:
    """The paper's yardsticks on the next-day residual: regression of the standardised residual on the predicted shock (slope, t, R2) and ANOVA."""
    col = "ar1"
    data = signals[[col, SCORE]].dropna()
    if data.empty:
        return
    tag = "h1"
    reg = impact_regression(data[SCORE].to_numpy(), data[col].to_numpy())
    note = "CAPM-type residual (stock minus beta x market, open-to-close approximation), standardised; plain OLS as in the paper, not date-clustered"
    rows.add("paper", "capm_slope", reg["slope"], set=name, window=window, fill=tag, n=reg["n"], note=note)
    rows.add("paper", "capm_t", reg["t"], set=name, window=window, fill=tag, n=reg["n"], note=note)
    rows.add("paper", "capm_r2", reg["r2"], set=name, window=window, fill=tag, n=reg["n"], note=note)
    standardised = (data[col] - data[col].mean()) / data[col].std() if data[col].std() > 0 else data[col] * np.nan
    result = anova(data[SCORE], standardised, int(fr(inp.cfg, "anova_min_group")))
    for metric, key in (("anova_f", "f"), ("anova_p", "p"), ("eta_squared", "eta2")):
        rows.add("paper", metric, result[key], set=name, window=window, fill=tag, n=len(data), note=f"{result['groups']} score groups with at least {fr(inp.cfg, 'anova_min_group')} observations")


# --- portfolio -----------------------------------------------------------------------------------------------------------


def build_strategy(variant: str, scores_by_date, cfg: Config):
    spec = fr(cfg, f"variants.{variant}")
    leg = float(cfg.get("rivals.finripple.leg_weight"))

    def rule(scores: pd.Series, date: pd.Timestamp, view: MarketView, held: pd.Series) -> pd.Series:
        target = sign_legs(scores, leg, int(spec["min_firms"])) if spec["rule"] == "sign" else top_bottom_legs(scores, float(spec["fraction"]), leg)
        if spec["legs"] == "both" or not len(target):
            return target
        side = target[target > 0] if spec["legs"] == "long" else target[target < 0]
        return side / side.abs().sum() * (2.0 * leg) if len(side) else side

    return make_strategy(rule, scores_by_date)


def variant_scores(signals: pd.DataFrame, variant: str, cfg: Config) -> dict[pd.Timestamp, pd.Series]:
    table = firm_days(signals, str(fr(cfg, f"variants.{variant}.aggregation")))
    return {d: part.set_index("security_id")[SCORE] for d, part in table.groupby("decision_date")}


def run_variant(signals: pd.DataFrame, span, variant: str, fill: str, delist: bool, inp: ScoringInputs, cache: dict) -> Run:
    cfg = inp.cfg
    market, panel = inp.market.execution_market(fill, cfg) if fill not in cache else cache[fill]
    cache[fill] = (market, panel)
    dates = [d for d in inp.calendar.sessions if span[0] <= d <= span[1]]
    omitted = [0]
    result = run_backtest(market, dates, build_strategy(variant, variant_scores(signals, variant, cfg), cfg),
                          delist_return=float(cfg.get("portfolio.delisting.default_return")) if delist else None, risk_free_daily=None)
    net = net_returns(result, cost_models(cfg, inp.market, omitted))
    borrow = (float(cfg.get("portfolio.scoring.short_borrow_stress_bp_per_year")) * float(cfg.get("portfolio.scoring.bp_to_fraction"))
              / float(cfg.get("eval.annualisation_days")) * result.positions.clip(upper=0.0).abs().sum(axis=1))
    net[f"{headline_cost_name(cfg)}_borrow"] = net[headline_cost_name(cfg)] - borrow
    return Run(result, net, leg_returns(result, panel), turnover(result), result.positions.abs().sum(axis=1) > 0, omitted[0])


# --- the whole run -----------------------------------------------------------------------------------------------------------


def run_scoring(inp: ScoringInputs, tape: pd.DataFrame, manifest: dict, state: dict, out_dir: Path, log=lambda message: None, max_tokens: int | None = None) -> dict:
    """Score the tape end to end, write every output under ``out_dir``, return the long results table and the text summary."""
    cfg = inp.cfg
    rows = Rows(cfg)
    periods = int(cfg.get("eval.annualisation_days"))
    out_dir.mkdir(parents=True, exist_ok=True)
    grid = fr(cfg, "variants")
    tuning = cfg.get("rivals.finripple.tuning")
    grid_size = int(np.prod([len(v) for v in tuning.values()]))
    count_text = f"{len(grid)} variants run, 0 tuned (grid {grid_size}, budget {cfg.get('rivals.tuning_configurations')})"
    meta = {"config_hash": cfg.hash, "cpu_settings_hash": cfg.unhashed_hash, "configuration_count": count_text,
            "cost_setting": sb(cfg, "labels.cost_setting"), "bundle_id": manifest.get("bundle_id"), "bucket": inp.bucket, "llm": inp.llm}

    # run health
    strict = tape[PARSE_OK].fillna(False).astype(bool)
    tape, recovered = recover_answers(tape, cfg)
    parsed = tape[PARSE_OK].fillna(False).astype(bool)
    max_tokens = int(max_tokens or cfg.get("rivals.finripple.max_tokens"))
    cut = int((tape[TOKENS_OUT] >= max_tokens).sum()) if TOKENS_OUT in tape else None
    rows.add("run", "units_total", len(inp.units), n=len(inp.units))
    rows.add("run", "units_ok", int((tape[STATUS] == Status.OK.value).sum()), n=len(tape))
    rows.add("run", "strict_parse_rate", float(strict.mean()), n=len(tape), note="the answer parses as a whole")
    rows.add("run", "refusal_rate_strict", float(1.0 - strict.mean()), n=len(tape), note="the paper's refusal-to-answer rate: answers that could not be post-processed into valid output")
    rows.add("run", "answers_recovered_from_cut_output", recovered[CUT], n=len(tape), note="did not parse as a whole; the complete company entries before the cut-off are kept")
    rows.add("run", "answers_recovered_quote_style", recovered[QUOTE_STYLE], n=len(tape), note="did not parse as a whole (single-quoted, apostrophes inside strings); the complete name / impact_type / impact_score entries are read as written")
    rows.add("run", "parse_rate", float(parsed.mean()), n=len(tape), note="strict plus recovered (both kinds); the signals use these")
    rows.add("run", "refusal_rate", float(1.0 - parsed.mean()), n=len(tape), note="answers with nothing usable")
    rows.add("run", "cut_at_max_tokens", cut, n=len(tape), note=f"answers as long as the {max_tokens}-token limit of this run (cut off)")
    rows.add("run", "mean_tokens_out", float(tape[TOKENS_OUT].mean()) if TOKENS_OUT in tape else None, n=len(tape))
    rows.add("run", "loop_seconds", state.get("elapsed_seconds"), note="worker loop only; engine load time is not in the bundle")
    rows.add("run", "code_version", manifest.get("meta", {}).get("code_version"))
    rows.add("run", "bundle_config_hash", manifest.get("meta", {}).get("config_hash"), note="hash the GPU session recorded (a different, longer hash format)")

    log("mapping names and building signals")
    mapper = NameMapper(inp.aliases, cfg)
    signals, members = build_signals(tape, inp, mapper, rows)
    log("abnormal forward returns (betas)")
    panels = attach_abnormal(signals, members, inp)
    spans = window_spans(inp)
    sets = {name: signals[signals["relation"].isin(groups)] for name, groups in fr(cfg, "sets").items()}
    headline_set = str(fr(cfg, "headline_set"))
    for name, span in spans.items():
        rows.add("coverage", "window_span" if span else "window_empty", f"{span[0].date()} to {span[1].date()}" if span else None, window=name,
                 note="" if span else f"{sb(cfg, 'labels.not_computed')}: no sessions of this window lie in the LLM's data range")
    rows.add("honesty", "market_adjustment", str(sb(cfg, "labels.market_approximation")))
    rows.add("honesty", "holding_period", str(sb(cfg, "labels.holding")))
    rows.add("honesty", "alpha_dating", str(sb(cfg, "labels.alpha_approximation")))
    rows.add("honesty", "cost_setting", str(sb(cfg, "labels.cost_setting")))
    rows.add("honesty", "configuration_count", count_text)
    rows.add("honesty", "graph_text", "our rendering from industry, shared-owner and return-correlation links; the paper's graphs are patent, supply-chain, mutual-fund and leadership graphs")
    rows.add("honesty", "residual_models", "CAPM-type residual only; FF3 and FF5 residual regressions are not computed here")

    log("signal metrics")
    for name, part in sets.items():
        for window, span in spans.items():
            if span is not None:
                signal_block(rows, name, window, part[in_span(part["decision_date"], span)], panels, inp)

    # portfolio
    pnl: dict[tuple[str, str], list[pd.DataFrame]] = {}
    runs: dict[tuple[str, str, str], Run] = {}
    cache: dict = {}
    headline_variant, headline_cost = str(fr(cfg, "headline_variant")), headline_cost_name(cfg)
    open_fill, close_fill = str(sb(cfg, "fills.headline")), str(sb(cfg, "fills.sensitivity"))
    layer_names = list(cfg.get("cpu.road1_eval.size_layers"))
    jobs = []
    for set_name, part in sets.items():
        for window, span in spans.items():
            if span is None:
                continue
            sub = part[in_span(part["decision_date"], span)]
            variants = list(grid) if set_name in fr(cfg, "full_grid_sets") else [headline_variant]
            for variant in variants:
                jobs.append((set_name, window, span, variant, open_fill, True, sub))
            if set_name in fr(cfg, "full_grid_sets"):
                for variant in fr(cfg, "close_fill_variants"):
                    jobs.append((set_name, window, span, variant, close_fill, True, sub))
                jobs.append((set_name, window, span, headline_variant + str(sb(cfg, "no_delist_suffix")), open_fill, False, sub))
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
        if variant in grid and fill == open_fill and delist and run.invested.any() and set_name in fr(cfg, "full_grid_sets"):
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
    for window, span in spans.items():
        if span is None:
            continue
        for layer in layer_names + ["no_price"]:
            sub = sets[headline_set][in_span(sets[headline_set]["decision_date"], span) & (sets[headline_set]["layer"] == layer)]
            base = dict(set=f"{headline_set}/{layer}", window=window, variant=headline_variant, fill=open_fill)
            if sub.empty:
                rows.add("robust_layer", "sharpe", None, **base, cost=headline_cost, note="no signals")
                continue
            run = run_variant(sub, span, headline_variant, open_fill, True, inp, cache)
            series_rows(rows, run.net[headline_cost], run, base, cfg, headline_cost, section="robust_layer")

    for window, span in spans.items():
        key = (headline_set, window, headline_variant + SEPARATOR + open_fill)
        if key not in runs:
            continue
        run = runs[key]
        series = run.net[headline_cost]
        sr = sharpe(series, periods)
        years = len(series) / periods
        for rho in cfg.get("eval.power.correlations"):
            gap = smallest_detectable_gap(sharpe_difference_se(sr, sr, float(rho), years, periods), float(cfg.get("eval.bootstrap.ci_level"))) if np.isfinite(sr) else np.nan
            rows.add("honesty", "smallest_detectable_sharpe_gap", gap, set=headline_set, window=window, n=len(series), cost=headline_cost,
                     note=f"correlation {rho}; both Sharpe ratios set to FinRipple-lite's own net value; {years:.2f} years")
        trial_keys = [(headline_set, window, v + SEPARATOR + open_fill) for v in grid]
        per_period = np.array([runs[k].net[headline_cost].mean() / runs[k].net[headline_cost].std() for k in trial_keys if k in runs and runs[k].net[headline_cost].std() > 0])
        if series.std() > 0 and len(per_period) == len(grid):
            dsr = deflated_sharpe(series.to_numpy(), per_period, float(cfg.get("eval.dsr.euler_gamma")))
            rows.add("honesty", "deflated_sharpe_probability", dsr["dsr"], set=headline_set, window=window, n=len(series), cost=headline_cost,
                     note=f"trials {dsr['trials']} (the variants run); per-period expected best from skill-free trials {dsr['sr0']:.4f}")
        else:
            rows.add("honesty", "deflated_sharpe_probability", None, set=headline_set, window=window, n=len(series), cost=headline_cost, note="a variant has no variation in this window")
        for set_name in sets:
            k = (set_name, window, headline_variant + SEPARATOR + open_fill)
            if k not in runs:
                continue
            r = runs[k]
            sharpes = {c: sharpe(r.net[c], periods) for c in ["gross"] + [f"rt{bp:g}bp" for bp in cfg.get("portfolio.scoring.round_trip_bp_grid")]}
            small = "; too small to interpret" if int(r.invested.sum()) < periods else ""
            capm = table_value(rows, "paper", "capm_r2", set_name, window)
            slope = table_value(rows, "paper", "capm_slope", set_name, window)
            low, high = fr(cfg, "repro.published_r2_range")
            ceiling = float(fr(cfg, "repro.r2_small_below"))
            shown = lambda v: "not computed" if v is None else f"{float(v):.4f}"
            rows.add("repro", "r2_positive_but_small", "matches" if r2_pattern(capm, slope, ceiling) else "deviation", set=set_name, window=window, variant=headline_variant,
                     n=len(r.net), note=f"measured R2 {shown(capm)}, slope {shown(slope)}; pattern: slope above zero and R2 above zero but below {ceiling} (published R2 {low}-{high}); "
                                        f"absolute numbers are never compared with the paper{small}")
            rows.add("repro", "gross_sharpe_positive", "matches" if np.isfinite(sharpes["gross"]) and sharpes["gross"] > 0 else "deviation", set=set_name, window=window,
                     variant=headline_variant, n=len(r.net), note=", ".join(f"{k_}: {v:.3f}" for k_, v in sharpes.items()) + small)

    table = rows.frame()
    table.insert(0, "config_hash", cfg.hash[: int(sb(cfg, "hash_chars"))])
    table["configuration_count"] = count_text
    table["cost_setting"] = str(sb(cfg, "labels.cost_setting"))
    outputs = fr(cfg, "output")
    write_parquet(out_dir / outputs["signals"], signals, meta)
    answers = tape[[c for c in (UNIT_ID, STATUS, PARSE_OK, AFFECTED, TOKENS_OUT, RAW) if c in tape]]
    write_parquet(out_dir / outputs["answers"], answers, meta)
    for (variant, cost), parts in pnl.items():
        write_parquet(out_dir / outputs["pnl"].format(variant=variant, cost=cost), pd.concat(parts, ignore_index=True), meta)
    table.to_csv(out_dir / outputs["summary_csv"], index=False)
    (out_dir / outputs["metrics"]).write_text(json.dumps({"meta": meta, "rows": json.loads(table.to_json(orient="records"))}, indent=1, default=str), encoding="utf-8")
    (out_dir / outputs["hash_file"]).write_text("\n".join(f"{k}: {v}" for k, v in meta.items()) + "\n", encoding="utf-8")
    text = summary_markdown(table, meta, cfg)
    (out_dir / outputs["summary_md"]).write_text(text, encoding="utf-8")
    return {"table": table, "text": text, "meta": meta, "signals": signals, "runs": runs}


def r2_pattern(r2, slope, ceiling: float) -> bool:
    """The published pattern for zero-shot-with-graph: the predicted shock explains the residual in the right direction, but only a little.

    R2 alone cannot show this (it is never negative); the slope must be positive too.
    """
    return all(v is not None and np.isfinite(v) for v in (r2, slope)) and slope > 0.0 and 0.0 < r2 < ceiling


def table_value(rows: Rows, section: str, metric: str, set_name: str, window: str):
    for r in rows.rows:
        if r["section"] == section and r["metric"] == metric and r["set"] == set_name and r["window"] == window:
            return r["value"] if isinstance(r["value"], (int, float, np.floating)) else None
    return None


# --- summary ---------------------------------------------------------------------------------------------------------------


def summary_markdown(table: pd.DataFrame, meta: dict, cfg: Config) -> str:
    d = int(cfg.get("eval.report.decimals"))
    headline_set, headline_variant, headline_cost = str(fr(cfg, "headline_set")), str(fr(cfg, "headline_variant")), headline_cost_name(cfg)
    open_fill = str(sb(cfg, "fills.headline"))
    parts = [f"# FinRipple-lite scored on {meta['bucket']} / {meta['llm']}\n",
             f"- config hash: `{meta['config_hash']}`  |  cpu settings hash: `{meta['cpu_settings_hash']}`",
             f"- configurations: {meta['configuration_count']}",
             f"- cost grid: {meta['cost_setting']}",
             f"- bundle: `{meta['bundle_id']}`",
             f"- every row below: {sb(cfg, 'labels.in_sample')}; dev rows: {sb(cfg, 'labels.exploratory')}\n"]
    for title, section in (("Run health and coverage", ["run", "coverage", "timing", "mapping"]), ("Honesty notes", ["honesty"])):
        sub = table[table["section"].isin(section)][["section", "set", "window", "metric", "value", "n", "note"]]
        parts += [f"## {title}\n", md_table(sub, d)]
    sig = table[(table["section"] == "signal") & table["metric"].isin(["signals_scored", "hit_rate_positive", "hit_rate_negative", "rank_ic", "rank_icir", "score_regression_slope"])]
    parts += ["## Signal (abnormal returns; h1 = open of day +1 to its close, h5 = to the close of day +5)\n",
              md_table(sig[["set", "window", "fill", "metric", "value", "ci_low", "ci_high", "n", "n_dates", "note"]].rename(columns={"fill": "horizon"}), d)]
    parts += ["## The paper's own yardsticks (next-day residual)\n", md_table(table[table["section"] == "paper"][["set", "window", "metric", "value", "n", "note"]], d)]
    pf = table[(table["section"] == "portfolio") & (table["cost"] == headline_cost) & (table["metric"].isin(["sharpe", "annual_return", "invested_days"]))
               & (table["fill"] == open_fill) & (table["window"] != "dev")]
    parts += [f"## Portfolio at {headline_cost}, fill {open_fill}\n", md_table(pf[["set", "window", "variant", "metric", "value", "n", "note"]], d)]
    grid = table[(table["section"] == "portfolio") & (table["variant"] == headline_variant) & (table["metric"] == "sharpe") & (table["fill"] == open_fill) & (table["set"] == headline_set)]
    parts += [f"## Headline variant across the cost grid ({headline_set})\n", md_table(grid[["window", "cost", "value", "n", "note"]], d)]
    for title, section in (("Gross-to-net bridge", "bridge"), ("Factor alpha", "alpha"), ("Stress windows", "stress"), ("Capacity", "capacity"),
                           ("Robustness: year by year", "robust_year"), ("Robustness: size layer", "robust_layer"), ("Reproduction pattern", "repro")):
        sub = table[(table["section"] == section) & (table["set"].str.startswith(headline_set) | (table["set"] == ""))]
        parts += [f"## {title}\n", md_table(sub[["set", "window", "variant", "fill", "metric", "value", "n", "note"]], d)]
    text = "\n".join(parts)
    bad = verdict_words(text, cfg)
    if bad:
        raise ValueError(f"verdict words in the summary: {bad}")
    return text
