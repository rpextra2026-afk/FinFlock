"""Dev-only fits of the framework on real data: dial maps and the road-1 model.

Built on the functions of the road-1 evaluation (``finflock.stage.road1_real``, run by ``scripts/run_road1.py``) so that the framework and
Experiment E share one implementation of node classes, graph nodes, labels, dial maps and fitting. Labels enter only here, as data,
through ``road1_real`` (a script-side layer, like ``fit``). Nothing fitted here sees a test-window event.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from finflock.common.config import Config
from finflock.common.timing import TradingCalendar
from finflock.fit.road1_fit import training_graphs
from finflock.framework.events import EventReading, pin_frame
from finflock.framework.road1_stage import Road1Predictor
from finflock.reader.dials import DialMap, fit_source_dial_maps, source_dials
from finflock.reader.state_features import add_state_features, dollar_volume_from_prices
from finflock.real.market_load import RealMarket
from finflock.road1.arms import MatrixModel
from finflock.road1.fit import dial_scale, predict_graph, select_and_fit, standardise, usable
from finflock.stage import road1_real as rr

LABEL_WINDOW = "d01"


@dataclass
class DevFit:
    maps: dict[tuple[str, str], DialMap]
    road1: Road1Predictor | None
    n_dev_events: int
    n_graphs: int
    notes: dict = field(default_factory=dict)


def panels_of(rm: RealMarket, factors: pd.DataFrame, cfg: Config) -> rr.Panels:
    """The price panels of ``road1_real`` from the market loaded for the framework (same tables, same returns)."""
    prices = {s: pd.DataFrame({rr.RET: rm.ret[s], "open": rm.open_[s], "high": rm.high[s], "low": rm.low[s],
                               "close_raw": rm.close[s], "volume_raw": rm.volume[s]})
              for s in rm.close.columns if rm.close[s].notna().any()}
    window = int(cfg.get("personas.panel.cap_proxy_window_days"))
    cap_proxy = (rm.close * rm.volume).rolling(window, min_periods=1).mean()
    return rr.Panels(rm.sessions, rm.close, rm.ret, prices, cap_proxy, factors.reindex(rm.sessions))


WMATRIX = "wmatrix"  # the weighted-matrix arm of the ladder; ``matrix_rung`` of the config key ``ablation.ladder`` picks the rung


def pins_with_state(cfg: Config, panels: rr.Panels, snapshots: pd.DataFrame, readings: dict[str, EventReading], kinds: dict[str, str],
                    day0_date: dict[str, pd.Timestamp]) -> pd.DataFrame:
    """Long frame of every pin reading with the state features (known before the event) and its node class."""
    classes = rr.node_classes(readings, day0_date, snapshots, panels, cfg)
    pins = pin_frame(readings, {}, kinds, cfg)
    pins["day0"] = pins["event_id"].map(day0_date)
    pins = add_state_features(pins, panels.ret, dollar_volume_from_prices(panels.prices, panels.sessions), cfg)
    pins["node_class"] = [classes[(e, n)] for e, n in zip(pins["event_id"], pins["node_id"], strict=True)]
    return pins


def source_dials_of(cfg: Config, panels: rr.Panels, snapshots: pd.DataFrame, readings: dict[str, EventReading], kinds: dict[str, str],
                    day0_date: dict[str, pd.Timestamp], maps: dict) -> dict[str, pd.DataFrame]:
    """The source dials of every event of ``readings`` under already fitted dial maps (no refit; used when only the readings change)."""
    pins = pins_with_state(cfg, panels, snapshots, readings, kinds, day0_date)
    return {e: source_dials(part, maps, cfg) for e, part in pins.groupby("event_id")}


def fit_dev(
    cfg: Config, panels: rr.Panels, snapshots: pd.DataFrame, readings: dict[str, EventReading], kinds: dict[str, str],
    day0_date: dict[str, pd.Timestamp], dev_ids: list[str], links: pd.DataFrame, calendar: TradingCalendar, arm: str,
    workers: int = 1, fixed: dict | None = None, seed_extra: tuple = (),
) -> tuple[DevFit, dict[str, pd.DataFrame]]:
    """Dial maps and the road-1 model from dev events; also returns the source dials of every event (dev and test).

    ``fixed``: ``{"grounding": g, "penalty_mu": p}``, the grid point to keep (an ablation refit that does not search the grid again). ``seed_extra``: more
    identity for the fit seeds (the noise-floor refits of Full differ only in this)."""
    dev_readings = {e: readings[e] for e in dev_ids if e in readings}
    pins = pins_with_state(cfg, panels, snapshots, readings, kinds, day0_date)
    nodes = rr.graph_nodes(dev_readings, links, day0_date, cfg)
    labels = rr.label_table(nodes, day0_date, panels, cfg)
    notes = {"n_label_rows": int(len(labels))}
    if labels.empty:
        return DevFit({}, None, len(dev_ids), 0, {**notes, "stopped": "no labels"}), {}
    pin_labels = labels[labels["window"] == LABEL_WINDOW].rename(columns={"security_id": "node_id"})[
        ["event_id", "node_id", "log_vol_ratio", "log_dv_ratio"]]
    dev_pins = pins[pins["event_id"].isin(set(dev_readings))].merge(pin_labels, on=["event_id", "node_id"], how="left")
    maps = fit_source_dial_maps(dev_pins, calendar, cfg)
    dials = {e: source_dials(part, maps, cfg) for e, part in pins.groupby("event_id")}
    graphs = training_graphs(dev_readings, dials, links, day0_date, labels, cfg)
    notes["n_graphs"] = len(graphs)
    if not graphs:
        return DevFit(maps, None, len(dev_ids), 0, {**notes, "stopped": "no training graphs"}), dials
    scale = dial_scale(graphs, float(cfg.get("road1.dial_scale_floor")))
    dev_dates = [day0_date[g.event_id] for g in graphs]
    standardised = standardise(graphs, scale)
    try:
        if arm == WMATRIX:  # closed-form ridge per link type; the rung is the configured one
            rung = int(cfg.get("ablation.ladder.L4.rung"))
            model, ridge, loss = rr.fit_matrices(standardised, dev_dates, calendar, cfg.with_overrides({"cpu.road1_eval.matrix_rungs": [rung]}))[rung]
            notes.update({"dial_scale": scale.tolist(), "dev_cv_loss": float(loss), "ridge": ridge, "matrix_rung": rung})
            return DevFit(maps, Road1Predictor(f"{arm}_r{rung}", model, float(cfg.get("road1.grounding_lambda_default")), scale, links, cfg),
                          len(dev_ids), len(graphs), notes), dials
        grid = None if fixed is None else [(fixed["grounding"], fixed["penalty_mu"])]
        model, best, _ = select_and_fit(arm, standardised, dev_dates, calendar, cfg, seed_parts=("real", arm, *seed_extra), workers=workers, grid=grid)
    except ValueError as exc:  # too few labelled graphs for the time-blocked folds: reported, never a crash
        return DevFit(maps, None, len(dev_ids), len(graphs), {**notes, "stopped": f"road-1 fit: {exc}"}), dials
    notes.update({"dial_scale": scale.tolist(), "grounding": best.grounding, "dev_cv_loss": float(best.val_loss), "penalty_mu": best.penalty_mu,
                  "best_steps": [int(x) for x in best.best_steps]})
    return DevFit(maps, Road1Predictor(arm, model, best.grounding, scale, links, cfg), len(dev_ids), len(graphs), notes), dials


def score_graphs(predictor: Road1Predictor, dev_graphs: list, test_graphs: list, dates: dict[str, pd.Timestamp], cfg: Config, seed_parts: tuple) -> pd.DataFrame:
    """Out-of-sample score of one fitted road-1 method on labelled test graphs, with the same table and formulas as the weighted graph and the weighted
    matrices get in ``run_road1.py`` (``road1_real.score_table``): R2 against the dev mean with a date-clustered bootstrap interval, RMSE, n, n_dates,
    for each dial, plus the no-ripple baseline. Nothing here is fitted: the predictor and the dev mean come from the dev events only."""
    if isinstance(predictor.model, MatrixModel):
        raise ValueError("score_graphs scores the graph and the sheaf; a matrix arm is scored by run_road1.py")
    dials = list(cfg.get("road1.dial_names"))
    rows = []
    for graph in test_graphs:
        if graph.y is None:
            continue
        forecast = predict_graph(predictor.model, graph, predictor.grounding, predictor.scale, cfg)
        for j, node in enumerate(graph.nodes[graph.n_pinned:]):
            for k, dial in enumerate(dials):
                if np.isfinite(graph.y[j, k]):
                    rows.append({"event_id": graph.event_id, "date": dates[graph.event_id], "node": node, "dial": dial, "y": graph.y[j, k], predictor.arm: forecast[j, k]})
    if not rows:
        raise ValueError("no labelled free node on the test side")
    train_mean = dict(zip(dials, np.nanmean(np.vstack([g.y for g in usable(dev_graphs)]), axis=0), strict=True))
    return rr.score_table(pd.DataFrame(rows), train_mean, cfg, seed_parts)


def sheaf_test_scores(
    cfg: Config, panels: rr.Panels, readings: dict[str, EventReading], day0_date: dict[str, pd.Timestamp], dev_ids: list[str], test_ids: list[str],
    dev: DevFit, dials: dict[str, pd.DataFrame], links: pd.DataFrame,
) -> pd.DataFrame:
    """The fitted road-1 arm of ``dev`` (the sheaf in the headline configuration) scored on the test events. The test graphs and labels are built exactly as
    the dev ones in ``fit_dev`` (same node, label and graph code, the source dials of every event from ``dials``); test labels enter only here, as data to
    score against, after the fit is stored. Rows: one per dial and method (the arm and ``no_ripple_zero``)."""
    if dev.road1 is None:
        raise ValueError("no fitted road-1 model to score")

    def graphs_of(ids: list[str]) -> list:
        part = {e: readings[e] for e in ids if e in readings}
        labels = rr.label_table(rr.graph_nodes(part, links, day0_date, cfg), day0_date, panels, cfg)
        return training_graphs(part, dials, links, day0_date, labels, cfg) if not labels.empty else []

    dev_graphs, test_graphs = graphs_of(dev_ids), graphs_of(test_ids)
    if not usable(dev_graphs) or not test_graphs:
        raise ValueError(f"not enough labelled graphs (dev {len(dev_graphs)}, test {len(test_graphs)})")
    table = score_graphs(dev.road1, dev_graphs, test_graphs, day0_date, cfg, ("real", dev.road1.arm, "test_score"))
    return table.assign(test_graphs=len(test_graphs), dev_graphs=len(dev_graphs))
