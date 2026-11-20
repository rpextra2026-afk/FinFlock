"""Cleaning steps for one reader tape. Every step is a pure function: frame in, frame out, plus a count record.

Every number comes from ``config/clean.yaml``. A step never edits the reader tape; it works on a copy.
"""

from __future__ import annotations

import html
import json
import re
from collections.abc import Callable, Mapping
from typing import Any

import numpy as np
import pandas as pd

from oryxis.common.config import Config
from oryxis.common.schema import UNIT_ID
from oryxis.common.timing import TradingCalendar

ROOT = "clean"
LANE_COMPANY = "company_event"
LANE_WORLD = "world_event"
LANE_ATTENTION = "attention_only"
LANE_JUNK = "junk"
NODE_COMPANY = "company"
NODE_THEME = "theme"
NODE_MARKET = "market"
NODE_SECTOR = "sector"
MARKET_NODE_ID = "market"
URL_PATTERN = re.compile(r"https?://\S+")
CASHTAG_PATTERN = re.compile(r"\$([A-Za-z]{1,5})\b")
SPACE_PATTERN = re.compile(r"\s+")
SESSION_BEFORE_OPEN = "before_open"
SESSION_INTRADAY = "intraday"
SESSION_AFTER_CLOSE = "after_close"
SESSION_NON_TRADING_DAY = "non_trading_day"
SESSION_DATE_ONLY = "date_only"


def _c(cfg: Config, path: str) -> Any:
    return cfg.get(f"{ROOT}.{path}")


class Counter:
    """Rows in and out after every step, with the reason for each drop."""

    def __init__(self) -> None:
        self.steps: list[dict[str, Any]] = []

    def record(self, step: str, rows_in: int, rows_out: int, **reasons: Any) -> None:
        self.steps.append({"step": step, "rows_in": int(rows_in), "rows_out": int(rows_out), "dropped": int(rows_in - rows_out), **reasons})


def normalise_text(value: object) -> str:
    """Lower case, no links, no cashtags, one space between words."""
    text = html.unescape(str(value))
    text = URL_PATTERN.sub(" ", text)
    text = CASHTAG_PATTERN.sub(" ", text)
    return SPACE_PATTERN.sub(" ", text).lower().strip()


def parse_pins(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, str):
        return []
    try:
        loaded = json.loads(value)
    except ValueError:
        return []
    return loaded if isinstance(loaded, list) else []


def load_items(run_cfg: Mapping[str, Any], root: str, cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reader tape joined to the unit text. The tape is read only."""
    tape = pd.read_parquet(f"{root}/{run_cfg['tape']}")
    units = pd.read_parquet(f"{root}/{run_cfg['units']}", columns=[UNIT_ID, "kind", "family", "available_date", "available_utc", "title", "text"])
    units = units.rename(columns={"family": "unit_family", "kind": "unit_kind"})
    items = tape.merge(units, on=UNIT_ID, how="inner", suffixes=("", "_unit"))
    note = {"tape_rows": int(len(tape)), "unit_rows": int(len(units)), "joined_rows": int(len(items)), "tape_rows_without_text": int(len(tape) - len(items))}
    share = note["tape_rows_without_text"] / max(len(tape), 1)
    if share > float(_c(cfg, "load.max_unmatched_share")):
        raise ValueError(f"{share:.1%} of the tape rows have no text in the units file {run_cfg['units']}")
    items["kind"] = items["unit_kind"]
    items["family"] = items["unit_family"]
    items["text_clean"] = (items["title"].fillna("") + " . " + items["text"].fillna("")).map(normalise_text)
    return items.reset_index(drop=True), note


def split_windows(items: pd.DataFrame, run_cfg: Mapping[str, Any], counter: Counter) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rows inside fit + hold-out stay; rows after the hold-out go to the sealed frame; rows before the fit start are dropped."""
    dates = pd.to_datetime(items["available_date"])
    lo, hi = pd.Timestamp(run_cfg["fit_start"]), pd.Timestamp(run_cfg["holdout_end"])
    inside = (dates >= lo) & (dates <= hi)
    sealed = dates > hi
    counter.record("windows", len(items), int(inside.sum()), before_fit_start=int((dates < lo).sum()), sealed_after_holdout=int(sealed.sum()))
    kept = items[inside].copy()
    kept["split"] = np.where(pd.to_datetime(kept["available_date"]) <= pd.Timestamp(run_cfg["fit_end"]), "fit", "holdout")
    return kept.reset_index(drop=True), items[sealed].copy().reset_index(drop=True)


def gate(items: pd.DataFrame, themes: Mapping[str, re.Pattern[str]], cfg: Config, counter: Counter) -> pd.DataFrame:
    """Analyst items, not_market_news and skipped items leave (each switch in config); counts are kept."""
    analyst = items["event_type_final"].eq(_c(cfg, "gate.analyst_value"))
    not_market = items["event_type_final"].eq(_c(cfg, "gate.not_market_value"))
    skipped = items["status"].ne(_c(cfg, "gate.ok_value"))
    drop = pd.Series(False, index=items.index)
    reasons: dict[str, Any] = {}
    for name, mask, switch in (("analyst_opinion", analyst, "drop_analyst"), ("not_market_news", not_market & ~analyst, "drop_not_market_news"), ("skipped", skipped & ~analyst & ~not_market, "drop_skipped")):
        applies = mask & ~drop if _c(cfg, f"gate.{switch}") else mask & False
        reasons[name] = int(applies.sum())
        reasons[f"{name}_matching_a_theme"] = int((applies & items["text_clean"].map(lambda t: any(p.search(t) for p in themes.values()))).sum())
        drop = drop | applies
    counter.record("gate", len(items), int((~drop).sum()), **reasons)
    return items[~drop].reset_index(drop=True)


def entity_check(items: pd.DataFrame, nodes: set[str], cfg: Config, counter: Counter) -> pd.DataFrame:
    """Keep only pins whose node is a known company node; unmatched pins are counted and logged."""
    kept_pins: list[list[dict[str, Any]]] = []
    unmatched: dict[str, int] = {}
    total = 0
    for value in items["reading_pins_json"]:
        pins = parse_pins(value)
        good = []
        for pin in pins:
            total += 1
            if pin.get("node_id") in nodes:
                good.append(pin)
            else:
                unmatched[str(pin.get("node_id"))] = unmatched.get(str(pin.get("node_id")), 0) + 1
        kept_pins.append(good)
    out = items.copy()
    out["pins"] = kept_pins
    counter.record("entity_check", len(items), len(items), pins_seen=total, pins_unmatched=int(sum(unmatched.values())), distinct_unmatched_nodes=len(unmatched), unmatched_examples=sorted(unmatched, key=unmatched.get, reverse=True)[: int(_c(cfg, 'report.examples'))])
    return out


def session_timing(items: pd.DataFrame, cal: TradingCalendar, cfg: Config, counter: Counter) -> pd.DataFrame:
    """First session that can react: timed items use the close rule; date-only items use their own date (next session if closed)."""
    out = items.copy()
    stamps = pd.to_datetime(out["available_utc"], utc=True)
    hour = _c(cfg, "timing.date_only_hour_utc")
    date_only = (stamps.dt.hour == hour) & (stamps.dt.minute == 0) & (stamps.dt.second == 0)
    session = pd.Series(pd.NaT, index=out.index, dtype="datetime64[ns]")
    kinds = pd.Series(SESSION_DATE_ONLY, index=out.index, dtype=object)
    if date_only.any():
        session[date_only] = cal.day0_date_only(pd.to_datetime(out.loc[date_only, "available_date"])).values
    timed = ~date_only
    if timed.any():
        ts = pd.DatetimeIndex(stamps[timed])
        pos = cal.closes_utc.searchsorted(ts, side="right")
        ok = pos < len(cal.sessions)
        got = np.full(len(ts), np.datetime64("NaT"), dtype="datetime64[ns]")
        got[ok] = cal.sessions[pos[ok]].values
        session[timed] = got
        kind = np.full(len(ts), SESSION_NON_TRADING_DAY, dtype=object)
        local_day = ts[ok].tz_convert(str(cfg.get("calendar.display_timezone"))).tz_localize(None).normalize()
        day0 = pd.DatetimeIndex(got[ok])
        started = ts[ok] >= cal.opens_utc[pos[ok]]
        on_session_day = local_day.isin(cal.sessions)
        kind[ok] = np.where(started, SESSION_INTRADAY, np.where(local_day == day0, SESSION_BEFORE_OPEN, np.where(on_session_day, SESSION_AFTER_CLOSE, SESSION_NON_TRADING_DAY)))
        kinds[timed] = kind
    out["session_date"] = session
    out["session_type"] = kinds
    out["ts_utc"] = stamps
    missing = int(out["session_date"].isna().sum())
    counter.record("session_timing", len(items), len(items) - missing, date_only_items=int(date_only.sum()), no_session_available=missing, **{f"session_{k}": int(v) for k, v in kinds.value_counts().items()})
    return out[out["session_date"].notna()].reset_index(drop=True)


def dedupe(items: pd.DataFrame, cal: TradingCalendar, cfg: Config, counter: Counter) -> pd.DataFrame:
    """Same story (same kind, same text prefix) inside the window becomes one item with ``repeat_count``."""
    key_chars = int(_c(cfg, "dedupe.key_chars"))
    window = int(_c(cfg, "dedupe.window_sessions"))
    min_chars = int(_c(cfg, "dedupe.min_key_chars"))
    out = items.copy()
    out["dedupe_key"] = out["text_clean"].str.slice(0, key_chars)
    out["session_pos"] = cal.sessions.get_indexer(pd.DatetimeIndex(out["session_date"]))
    out = out.sort_values(["kind", "dedupe_key", "ts_utc"]).reset_index(drop=True)
    group = np.zeros(len(out), dtype=np.int64)
    start_pos = -1
    current = -1
    prev_key: tuple[Any, Any] | None = None
    for i, (kind, key, pos) in enumerate(zip(out["kind"], out["dedupe_key"], out["session_pos"], strict=True)):
        same = prev_key == (kind, key) and len(key) >= min_chars and pos - start_pos <= window
        if not same:
            current += 1
            start_pos = pos
        group[i] = current
        prev_key = (kind, key)
    out["dedupe_group"] = group
    counts = out.groupby("dedupe_group")[UNIT_ID].transform("size")
    out["repeat_count"] = counts.astype(int)
    first = out.groupby("dedupe_group").cumcount().eq(0)
    merged = out[first].drop(columns=["dedupe_key", "session_pos"]).reset_index(drop=True)
    counter.record("dedupe", len(items), len(merged), items_merged_away=int(len(items) - len(merged)), groups_with_repeats=int((merged["repeat_count"] > 1).sum()))
    return merged


def cause_groups(items: pd.DataFrame, cfg: Config, group_of: Mapping[str, str], counter: Counter) -> pd.DataFrame:
    """Cause 'other' goes to the no_cause group; every other cause keeps the group of the causes list."""
    out = items.copy()
    other = out["cause_top"].eq(_c(cfg, "causes.other_value")) | out["cause_top"].isna()
    no_cause = _c(cfg, "causes.no_cause_group")
    out["cause_group"] = [no_cause if o else group_of.get(c, no_cause) for o, c in zip(other, out["cause_top"], strict=True)]
    counter.record("cause_groups", len(out), len(out), cause_other_to_no_cause=int(other.sum()), **{f"group_{k}": int(v) for k, v in out["cause_group"].value_counts().items()})
    return out


def compile_themes(cfg: Config) -> dict[str, re.Pattern[str]]:
    return {name: re.compile(pattern) for name, pattern in _c(cfg, "touch.themes").items()}


def assign_lanes(items: pd.DataFrame, themes: Mapping[str, re.Pattern[str]], cfg: Config, counter: Counter) -> pd.DataFrame:
    """company_event: at least one valid pin. world_event: no pin, at least one theme. attention_only: neither."""
    out = items.copy()
    out["n_pins"] = out["pins"].map(len)
    hits = out["text_clean"].map(lambda t: [name for name, p in themes.items() if p.search(t)])
    out["themes"] = hits
    lane = np.where(out["n_pins"] > 0, LANE_COMPANY, np.where(hits.map(len) > 0, LANE_WORLD, LANE_ATTENTION))
    out["lane"] = lane
    counter.record("lanes", len(out), len(out), **{f"lane_{k}": int(v) for k, v in pd.Series(lane).value_counts().items()})
    return out


def returns_frame(prices: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Market-adjusted close-to-close return in percent, sessions by stocks."""
    px = prices.sort_values(["security_id", "session_date"]).copy()
    px["ret"] = px.groupby("security_id")["close_raw"].pct_change()
    wide = px.pivot(index="session_date", columns="security_id", values="ret")
    return wide.sub(wide.mean(axis=1), axis=0) * float(_c(cfg, "returns.scale"))


def learn_trust(items: pd.DataFrame, rets: pd.DataFrame, run_cfg: Mapping[str, Any], cfg: Config, counter: Counter) -> dict[str, float]:
    """One weight per news kind from fit-window company pins: own-stock absolute move at day 0, relative to the overall mean."""
    fit = items[(items["lane"] == LANE_COMPANY) & items["split"].eq("fit")]
    moves: dict[str, list[float]] = {}
    for kind, session, pins in zip(fit["kind"], fit["session_date"], fit["pins"], strict=True):
        if session not in rets.index:
            continue
        for pin in pins:
            node = pin.get("node_id")
            if node in rets.columns and np.isfinite(rets.at[session, node]):
                moves.setdefault(kind, []).append(abs(float(rets.at[session, node])))
    every = [v for values in moves.values() for v in values]
    overall = float(np.mean(every)) if every else 0.0
    lo, hi, floor, default = (float(_c(cfg, f"trust.{k}")) for k in ("clip_low", "clip_high", "min_pins", "default"))
    weights = {k: float(np.clip(np.mean(v) / overall, lo, hi)) if len(v) >= floor and overall > 0 else default for k, v in moves.items()}
    counter.record("trust", len(items), len(items), trust_weights={k: round(v, int(_c(cfg, 'report.decimals'))) for k, v in weights.items()}, fit_pins_per_kind={k: len(v) for k, v in moves.items()})
    return weights


def fit_sector_model(items: pd.DataFrame, rets: pd.DataFrame, nodes: list[str], cfg: Config, counter: Counter) -> Callable[[pd.Series], tuple[np.ndarray, np.ndarray]] | None:
    """Data-driven sector clusters from fit-window returns, and a text classifier trained on fit-window company pins."""
    if not _c(cfg, "sector.enabled"):
        return None
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression

    fit_sessions = pd.DatetimeIndex(items.loc[items["split"].eq("fit"), "session_date"].unique())
    block = rets.loc[rets.index.isin(fit_sessions)].reindex(columns=nodes)
    block = block.loc[:, block.notna().sum() >= int(_c(cfg, "sector.min_returns_days"))].fillna(0.0)
    cols = list(block.columns)
    values = block.values
    values = (values - values.mean(0)) / (values.std(0) + float(np.finfo(float).eps))
    _, s_values, vt = np.linalg.svd(values, full_matrices=False)
    comps = int(_c(cfg, "sector.svd_components"))
    emb = vt[:comps].T * s_values[:comps]
    clusters = KMeans(int(_c(cfg, "sector.clusters")), n_init=int(_c(cfg, "sector.kmeans_n_init")), random_state=int(_c(cfg, "sector.seed"))).fit_predict(emb)
    cluster_of = dict(zip(cols, clusters.tolist(), strict=True))
    train = items[(items["lane"] == LANE_COMPANY) & items["split"].eq("fit")]
    texts, labels = [], []
    for text, pins in zip(train["text_clean"], train["pins"], strict=True):
        first = pins[0].get("node_id") if pins else None
        if first in cluster_of:
            texts.append(text)
            labels.append(cluster_of[first])
    tf = {k: _c(cfg, f"sector.tfidf.{k}") for k in ("ngram_min", "ngram_max", "min_df", "max_features")}
    vec = TfidfVectorizer(ngram_range=(tf["ngram_min"], tf["ngram_max"]), min_df=tf["min_df"], max_features=tf["max_features"], sublinear_tf=True)
    clf = LogisticRegression(C=float(_c(cfg, "sector.logistic.c")), max_iter=int(_c(cfg, "sector.logistic.max_iter")), class_weight="balanced").fit(vec.fit_transform(texts), labels)
    counter.record("sector_model", len(items), len(items), cluster_sizes=np.bincount(clusters).tolist(), classifier_training_pins=len(texts))

    def predict(series: pd.Series) -> tuple[np.ndarray, np.ndarray]:
        proba = clf.predict_proba(vec.transform(series))
        return clf.classes_[proba.argmax(1)], proba.max(1)

    return predict


def touch_points(items: pd.DataFrame, trust: Mapping[str, float], sector_predict: Callable[[pd.Series], tuple[np.ndarray, np.ndarray]] | None, cfg: Config, counter: Counter) -> pd.DataFrame:
    """Long table: one row per (item, touch point). Items with no touch point keep one row with an empty node."""
    base = float(_c(cfg, "repeat_weight.base"))
    dir_map = {k: float(v) for k, v in _c(cfg, "direction").items()}
    market_themes = set(_c(cfg, "touch.market_themes"))
    themes_on_company = bool(_c(cfg, "touch.themes_on_company_events"))
    cut = float(_c(cfg, "sector.confidence_cut"))
    sector_id: dict[Any, tuple[Any, float]] = {}
    if sector_predict is not None:
        world = items[items["lane"] != LANE_COMPANY]
        if len(world):
            labels, conf = sector_predict(world["text_clean"])
            sector_id = {i: (lab, c) for i, lab, c in zip(world.index, labels, conf, strict=True)}
    rows: list[dict[str, Any]] = []
    for idx, r in items.iterrows():
        tw = float(trust.get(r["kind"], float(_c(cfg, "trust.default"))))
        rw = base + float(np.log(r["repeat_count"]))
        common = {UNIT_ID: r[UNIT_ID], "run_split": r["split"], "lane": r["lane"], "kind": r["kind"], "family": r["family"], "event_type_final": r["event_type_final"], "cause_top": r["cause_top"], "cause_group": r["cause_group"], "ts_utc": r["ts_utc"], "session_date": r["session_date"], "session_type": r["session_type"], "repeat_count": r["repeat_count"], "trust_weight": tw, "repeat_weight": rw, "weight": tw * rw}
        emitted = False
        for pin in r["pins"]:
            direction = dir_map.get(str(pin.get("direction")), 0.0)
            rows.append({**common, "node_type": NODE_COMPANY, "node_id": pin.get("node_id"), "pin_role": pin.get("role"), "direction": direction, "size_pct": pin.get("value_effect_pct"), "magnitude": pin.get("magnitude"), "signed_weight": tw * rw * direction})
            emitted = True
        if r["lane"] != LANE_COMPANY or themes_on_company:
            for theme in r["themes"]:
                rows.append({**common, "node_type": NODE_THEME, "node_id": theme, "direction": 0.0, "signed_weight": 0.0})
                emitted = True
            if market_themes.intersection(r["themes"]):
                rows.append({**common, "node_type": NODE_MARKET, "node_id": MARKET_NODE_ID, "direction": 0.0, "signed_weight": 0.0})
                emitted = True
            if idx in sector_id and sector_id[idx][1] >= cut:
                rows.append({**common, "node_type": NODE_SECTOR, "node_id": f"sector_{sector_id[idx][0]}", "direction": 0.0, "signed_weight": 0.0})
                emitted = True
        if not emitted:
            rows.append({**common, "node_type": None, "node_id": None, "direction": 0.0, "signed_weight": 0.0})
    long = pd.DataFrame(rows)
    counter.record("touch_points", len(items), len(items), rows=int(len(long)), **{f"touch_{k}": int(v) for k, v in long["node_type"].fillna("none").value_counts().items()})
    return long
