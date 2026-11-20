"""Experiment E on real tapes: dev fits of the dial maps and the road-1 methods, test scoring (spec 8).

Reads a reader tape, the prestaged event table, links, prices and the factor table. Labels enter only
as data for the fits on dev and for scoring on test; nothing here feeds a label back into a forecast.
Output is numbers and intervals; no verdict wording.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np
import pandas as pd

from oryxis.common.binning import rank_bins
from oryxis.common.config import Config
from oryxis.common.cv import time_blocked_folds
from oryxis.common.rng import make_rng
from oryxis.common.schema import STATUS, Status
from oryxis.common.timing import TradingCalendar
from oryxis.common.windows import BucketWindows
from oryxis.data.buckets.contract import finalize_events
from oryxis.data.labels import build_labels
from oryxis.data.market import total_returns, wide
from oryxis.data.sp500 import members_on
from oryxis.fit.road1_fit import label_matrix, training_graphs
from oryxis.framework.events import EventReading, pin_frame, readings_from_tape
from oryxis.reader.dials import fit_source_dial_maps, source_dials
from oryxis.road1.arms import MatrixModel
from oryxis.road1.fit import dial_scale, fit_matrix, oos_r2, predict_graph, select_and_fit, standardise, usable
from oryxis.road1.graph import EventGraph, build_event_graph
from oryxis.road1.sheaf_test import d10_regression
from oryxis.road1.strata import POOLED

PRICE_COLUMNS = ("open", "high", "low", "close_raw", "volume_raw")
RET = "ret"


@dataclass
class Panels:
    sessions: pd.DatetimeIndex
    close: pd.DataFrame
    ret: pd.DataFrame
    prices: dict[str, pd.DataFrame]
    cap_proxy: pd.DataFrame
    factors: pd.DataFrame


def load_panels(root: Path, cfg: Config, calendar: TradingCalendar) -> Panels:
    """Session-indexed panels from ``px_daily``/``corp_action`` and the factor table."""
    market = root / str(cfg.get("cpu.folders.market"))
    px = pd.read_parquet(market / "px_daily.parquet")
    actions = pd.read_parquet(market / "corp_action.parquet")
    sessions = calendar.sessions
    wides = {c: wide(px, c, sessions) for c in ("close_raw", "open", "high", "low", "volume_raw")}
    ret, _ = total_returns(wides["close_raw"], wides["open"], actions)
    window = int(cfg.get("personas.panel.cap_proxy_window_days"))
    cap_proxy = (wides["close_raw"] * wides["volume_raw"]).rolling(window, min_periods=1).mean()
    prices = {}
    for security in wides["close_raw"].columns:
        frame = pd.DataFrame({RET: ret[security], **{c: wides[c][security] for c in ("open", "high", "low")},
                              "close_raw": wides["close_raw"][security], "volume_raw": wides["volume_raw"][security]})
        if frame["close_raw"].notna().any():
            prices[security] = frame
    factors = pd.read_parquet(root / str(cfg.get("cpu.folders.factors"))).reindex(sessions)
    return Panels(sessions, wides["close_raw"], ret, prices, cap_proxy, factors)


def reader_events(units: pd.DataFrame, calendar: TradingCalendar, windows: BucketWindows, cfg: Config) -> pd.DataFrame:
    """Events of a prestaged reader table with day 0 and the dev/test split (embargo and outside are dropped)."""
    frame = units.rename(columns={"unit_id": "event_id"})[["event_id", "kind", "family", "available_utc", "pins_json"]]
    kept, _ = finalize_events(frame, calendar, windows, cfg)
    return kept


def node_classes(
    readings: dict[str, EventReading], day0: dict[str, pd.Timestamp], snapshots: pd.DataFrame, panels: Panels, cfg: Config,
) -> dict[tuple[str, str], str]:
    """Size layer of each pinned node at its event's day 0 (S&P members only; others use the outside class)."""
    layers = list(cfg.get("cpu.road1_eval.size_layers"))
    outside = str(cfg.get("cpu.road1_eval.outside_layer"))
    out: dict[tuple[str, str], str] = {}
    cache: dict[pd.Timestamp, pd.Series] = {}
    for event_id, reading in readings.items():
        date = day0[event_id]
        if date not in cache:
            members = sorted(members_on(snapshots, date).security_ids)
            before = panels.cap_proxy.loc[: date - pd.Timedelta(days=1)]
            size = before.iloc[-1].reindex(members).dropna() if len(before) else pd.Series(dtype=float)
            bins = rank_bins(size, len(layers)).map(dict(enumerate(layers))) if len(size) else pd.Series(dtype=object)
            cache[date] = bins
        for pin in reading.pins:
            out[(event_id, pin["node_id"])] = str(cache[date].get(pin["node_id"], outside))
    return out


def graph_nodes(readings: dict[str, EventReading], links: pd.DataFrame, day0: dict[str, pd.Timestamp], cfg: Config) -> dict[str, tuple[str, ...]]:
    """Nodes of every event graph (dials do not change the node set)."""
    nodes = {}
    for event_id, reading in readings.items():
        pinned = [p["node_id"] for p in reading.pins]
        graph = build_event_graph(cfg, event_id, pinned, np.zeros((len(pinned), len(cfg.get("road1.dial_names")))), links,
                                  day0[event_id], reading.group_probs)
        nodes[event_id] = graph.nodes
    return nodes


def label_table(nodes: dict[str, tuple[str, ...]], day0: dict[str, pd.Timestamp], panels: Panels, cfg: Config) -> pd.DataFrame:
    rows = [{"event_id": e, "security_id": n, "day0": day0[e]} for e, ns in nodes.items() for n in ns if n in panels.prices]
    labels = build_labels(pd.DataFrame(rows), panels.prices, panels.factors, cfg)
    values = [c for c in labels.columns if c.startswith("log_") or c == "abn_ret"]
    labels[values] = labels[values].replace([np.inf, -np.inf], np.nan)  # a log of zero volume is a missing label, never imputed
    return labels


# ---- fitting -------------------------------------------------------------------------------------------------

def matrix_cv_loss(graphs: list[EventGraph], dates, rung: int, ridge: float, n_types: int, calendar, cfg: Config) -> float:
    """Mean squared error over labelled free-node dials in time-blocked folds (standardised units)."""
    cv = cfg.get("road1.cv")
    losses = []
    for train_idx, val_idx in time_blocked_folds(dates, int(cv["folds"]), int(cv["embargo_sessions"]), calendar):
        train, val = [graphs[i] for i in train_idx], [graphs[i] for i in val_idx]
        if not usable(train) or not usable(val):
            continue
        model = fit_matrix(train, rung, ridge, n_types)
        err = [(g.y - model.predict(g)) ** 2 for g in usable(val)]
        losses.append(float(np.nanmean(np.concatenate(err))))
    if not losses:
        raise ValueError("no fold with labelled graphs on both sides")
    return float(np.mean(losses))


def fit_matrices(graphs, dates, calendar, cfg: Config) -> dict[int, tuple[MatrixModel, float, float]]:
    """For each rung: ridge chosen by dev CV, refit on all dev graphs. ``{rung: (model, ridge, cv_loss)}``."""
    n_types = len(cfg.get("road1.link_types"))
    out = {}
    for rung in cfg.get("cpu.road1_eval.matrix_rungs"):
        scored = [(matrix_cv_loss(graphs, dates, int(rung), float(r), n_types, calendar, cfg), float(r)) for r in cfg.get("road1.ridge_grid")]
        loss, ridge = min(scored)
        out[int(rung)] = (fit_matrix(usable(graphs), int(rung), ridge, n_types), ridge, loss)
    return out


@dataclass
class Fitted:
    scale: np.ndarray
    maps: dict
    wgraph: tuple
    sheaf: tuple | None  # None when the sheaf is not fitted (run_road1.py --no-sheaf)
    matrices: dict[int, tuple[MatrixModel, float, float]]
    n_dev_graphs: int
    notes: dict = field(default_factory=dict)


def n_params(model) -> int:
    return int(model.n_parameters()) if hasattr(model, "n_parameters") else int(sum(p.numel() for p in model.parameters()))


# ---- scoring -------------------------------------------------------------------------------------------------

def forecasts_for(fitted: Fitted, graph: EventGraph, cfg: Config) -> dict[str, np.ndarray]:
    """Free-node forecasts (original units) of every method for one graph."""
    out = {"wgraph": predict_graph(fitted.wgraph[0], graph, fitted.wgraph[1].grounding, fitted.scale, cfg)}
    if fitted.sheaf is not None:
        out["sheaf"] = predict_graph(fitted.sheaf[0], graph, fitted.sheaf[1].grounding, fitted.scale, cfg)
    standardised = standardise([graph], fitted.scale)[0]
    for rung, (model, _, _) in fitted.matrices.items():
        out[f"wmatrix_r{rung}"] = model.predict(standardised) * fitted.scale
    return out


def long_scores(fitted: Fitted, graphs: list[EventGraph], dates: dict[str, pd.Timestamp], cfg: Config) -> pd.DataFrame:
    """One row per (event, free node, dial) with the label and every method's forecast."""
    dials = list(cfg.get("road1.dial_names"))
    rows = []
    for graph in graphs:
        if graph.y is None:
            continue
        preds = forecasts_for(fitted, graph, cfg)
        for j, node in enumerate(graph.nodes[graph.n_pinned:]):
            for k, dial in enumerate(dials):
                if np.isfinite(graph.y[j, k]):
                    rows.append({"event_id": graph.event_id, "date": dates[graph.event_id], "node": node, "dial": dial,
                                 "y": graph.y[j, k], **{m: p[j, k] for m, p in preds.items()}})
    return pd.DataFrame(rows)


def cluster_r2(y: np.ndarray, pred: np.ndarray, base: float, cluster: np.ndarray, draws: int, level: float, rng) -> dict:
    """Out-of-sample R2 against a fixed baseline forecast and its date-clustered bootstrap interval."""
    labels, inverse = np.unique(cluster, return_inverse=True)
    sse = np.bincount(inverse, weights=(y - pred) ** 2)
    sst = np.bincount(inverse, weights=(y - base) ** 2)
    counts = rng.multinomial(len(labels), np.full(len(labels), 1.0 / len(labels)), size=draws)
    boot = 1.0 - (counts @ sse) / (counts @ sst)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(boot, [tail, 1.0 - tail])
    return {"r2": float(1.0 - sse.sum() / sst.sum()), "ci_low": float(low), "ci_high": float(high),
            "rmse": float(np.sqrt(((y - pred) ** 2).mean())), "n": int(len(y)), "n_dates": int(len(labels))}


def score_table(scores: pd.DataFrame, train_mean: dict[str, float], cfg: Config, seed_parts: tuple) -> pd.DataFrame:
    draws, level = int(cfg.get("road1.bootstrap.draws")), float(cfg.get("road1.confidence_level"))
    rng = make_rng(*seed_parts, "score")
    methods = [c for c in scores.columns if c not in ("event_id", "date", "node", "dial", "y")]
    rows = []
    for dial, part in scores.groupby("dial"):
        cluster = part["date"].astype(str).to_numpy()
        for method in methods:
            rows.append({"dial": dial, "method": method,
                         **cluster_r2(part["y"].to_numpy(), part[method].to_numpy(), train_mean[dial], cluster, draws, level, rng)})
        rows.append({"dial": dial, "method": "no_ripple_zero",
                     **cluster_r2(part["y"].to_numpy(), np.zeros(len(part)), train_mean[dial], cluster, draws, level, rng)})
    return pd.DataFrame(rows)


def d10_table(scores: pd.DataFrame, table_method: str, cfg: Config, seed_parts: tuple) -> pd.DataFrame:
    draws, level = int(cfg.get("road1.bootstrap.draws")), float(cfg.get("road1.confidence_level"))
    rng = make_rng(*seed_parts, "d10")
    rows = []
    for dial, part in scores.groupby("dial"):
        if part["date"].nunique() < 2:
            continue
        result = d10_regression(part["y"].to_numpy(), part[table_method].to_numpy(), part["sheaf"].to_numpy(),
                                part["date"].astype(str).to_numpy(), rng, draws, level)
        rows.append({"dial": dial, "table": table_method, **result.__dict__})
    return pd.DataFrame(rows)


def score_table_strata(scores: pd.DataFrame, strata: pd.DataFrame, train_mean: dict[str, float], cfg: Config, seed_parts: tuple) -> pd.DataFrame:
    """``score_table`` for all events and for each pin-count stratum (single pin, several pins), one block per stratum."""
    tagged = scores.merge(strata[["event_id", "pin_stratum"]], on="event_id", how="left")
    blocks = [(POOLED, tagged)] + [(name, part) for name, part in tagged.groupby("pin_stratum")]
    frames = []
    for name, part in blocks:
        if part.empty:
            continue
        table = score_table(part[scores.columns], train_mean, cfg, (*seed_parts, name))
        table.insert(0, "stratum", name)
        table.insert(1, "n_events", int(part["event_id"].nunique()))
        frames.append(table)
    return pd.concat(frames, ignore_index=True)


def oracle_inputs(readings: dict[str, EventReading], labels: pd.DataFrame, cfg: Config) -> tuple[dict[str, EventReading], dict[str, pd.DataFrame]]:
    """Readings and source dials for the ORACLE fit (spec 8.2): each pin's source values are the stock's realised day 0-1 dials, not the reader's.
    A pin without a finite realised label for all three dials (no prices) is left out of the pinned set; an event with no such pin is left out."""
    known = label_matrix(labels)
    dials = list(cfg.get("road1.dial_names"))
    kept, source = {}, {}
    for event_id, reading in readings.items():
        pins = [p for p in reading.pins if np.isfinite(known.get((event_id, p["node_id"]), np.full(len(dials), np.nan))).all()]
        if pins:
            kept[event_id] = replace(reading, pins=tuple(pins))
            source[event_id] = pd.DataFrame([known[(event_id, p["node_id"])] for p in pins], columns=dials)
    return kept, source


def pinned_reference_table(
    readings: dict[str, EventReading], dials: dict[str, pd.DataFrame], labels: pd.DataFrame, split: dict[str, str], day0: dict[str, pd.Timestamp],
    cfg: Config, seed_parts: tuple,
) -> pd.DataFrame:
    """The reader's own-firm reference (spec 8.2): the source dial of each pinned stock against its realised dial, no graph, on the test window.
    R2 is against the training-window mean of the same realised dials."""
    known = label_matrix(labels)
    dial_names = list(cfg.get("road1.dial_names"))
    rows = []
    for event_id, reading in readings.items():
        if event_id not in dials or event_id not in split:
            continue
        for k, pin in enumerate(reading.pins):
            y = known.get((event_id, pin["node_id"]))
            if y is None:
                continue
            for j, dial in enumerate(dial_names):
                if np.isfinite(y[j]):
                    rows.append({"event_id": event_id, "split": split[event_id], "date": day0[event_id], "dial": dial, "y": float(y[j]),
                                 "reader_own_firm": float(dials[event_id].iloc[k][dial])})
    frame = pd.DataFrame(rows, columns=["event_id", "split", "date", "dial", "y", "reader_own_firm"])
    draws, level = int(cfg.get("road1.bootstrap.draws")), float(cfg.get("road1.confidence_level"))
    rng = make_rng(*seed_parts, "pinned_reference")
    out = []
    for dial in dial_names:
        dev, test = frame[(frame["dial"] == dial) & (frame["split"] == "dev")], frame[(frame["dial"] == dial) & (frame["split"] == "test")]
        if dev.empty or test.empty:
            continue
        out.append({"dial": dial, "method": "reader_own_firm", **cluster_r2(test["y"].to_numpy(), test["reader_own_firm"].to_numpy(), float(dev["y"].mean()),
                                                                            test["date"].astype(str).to_numpy(), draws, level, rng)})
    return pd.DataFrame(out)


def pin_types_of(readings: dict[str, EventReading], pins_json: dict[str, str]) -> dict[str, tuple[str, ...]]:
    """Node types (company, country, commodity) of each event's pinned nodes, from the unit table's rule-based pins."""
    out = {}
    for event_id, reading in readings.items():
        types = {pin["node_id"]: pin["node_type"] for pin in json.loads(pins_json[event_id])}
        out[event_id] = tuple(types.get(pin["node_id"], "unknown") for pin in reading.pins)
    return out


def split_graphs(graphs: list[EventGraph], split: dict[str, str], which: str) -> list[EventGraph]:
    return [g for g in graphs if split[g.event_id] == which]


def clean_rows(tape: pd.DataFrame) -> pd.DataFrame:
    """Rows of a reader tape the road-1 stage can use: status ok and a parsed reading."""
    return tape[tape[STATUS] == Status.OK.value]


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=1, default=str), encoding="utf-8")
