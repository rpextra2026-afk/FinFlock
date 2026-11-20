"""Road 1 from the final build's pooled sheaf (one rule per link type, pooled across causes), with the same three dials: ret, vol, volu.

The sheaf is the one ``finfm.build.run`` fitted (``Fitted.sheaf``); nothing is refitted here. Pins reach it in the units it was trained on (``s_*`` columns of
``finfm.framework.clean_adapter``). Its output for free nodes is in label units (return in percent, vol and volume as log ratios); the return is converted to a fraction here. A pinned node has no label-side output, so its vol and volume dials come from a
one-variable map fitted on dev pins only (``fit_pin_maps``); its return dial is the pin's own cleaned value effect.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from finfm.common.config import Config
from finfm.framework.events import EventReading
from finfm.framework.road1_stage import DIALS, Road1Predictor
from finfm.real.framework_fit import DevFit
from finfm.road1.fit import predict_graph
from finfm.road1.graph import EventGraph, build_event_graph

ARM = "pooled_sheaf"
POOLED_GROUP = np.ones(1)
S_COLUMNS = ("s_ret", "s_vol", "s_volu")


class PooledRoad1Predictor(Road1Predictor):
    """The pooled sheaf as a road-1 predictor; the group vector is always the single pooled group."""

    def __init__(self, model, grounding: float, scale: np.ndarray, links: pd.DataFrame, cfg: Config, pin_maps: Mapping[str, tuple[float, float]]) -> None:
        super().__init__(ARM, model, grounding, scale, links, cfg)
        self.pin_maps = dict(pin_maps)

    def graph_for(self, reading: EventReading, source_dials: pd.DataFrame, asof: pd.Timestamp) -> EventGraph:
        nodes = [p["node_id"] for p in reading.pins]
        return build_event_graph(self.cfg, reading.event_id, nodes, source_dials.loc[:, list(S_COLUMNS)].to_numpy(), self.links, asof, POOLED_GROUP)

    def predict(self, reading: EventReading, source_dials: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
        graph = self.graph_for(reading, source_dials, asof)
        free = predict_graph(self.model, graph, self.grounding, self.scale, self.cfg) if graph.n_free else np.zeros((0, len(DIALS)))
        free = free * np.array([float(self.cfg.get("framework.pooled.label_ret_to_fraction")), 1.0, 1.0])  # return label in percent -> fraction; vol and volume are log ratios
        pinned = np.column_stack([source_dials["ret"].to_numpy(dtype=float)] + [self.pin_maps.get(d, (0.0, 0.0))[0] + self.pin_maps.get(d, (0.0, 0.0))[1] * source_dials[f"s_{d}"].to_numpy(dtype=float)
                                                                                  for d in DIALS[1:]])
        values = np.vstack([pinned, free]) if graph.n_free else pinned
        frame = pd.DataFrame(values, columns=list(DIALS), index=pd.Index(graph.nodes, name="node_id"))
        frame["role"] = ["pinned"] * graph.n_pinned + ["ripple"] * graph.n_free
        frame["truncated"] = graph.truncated
        return frame


def fit_pin_maps(source: Mapping[str, pd.DataFrame], labels: pd.DataFrame, dev_ids: list[str], cfg: Config) -> dict[str, tuple[float, float]]:
    """Intercept and slope of the label (log volatility ratio, log dollar-volume ratio) of a pinned node on its pin's ``s_vol`` / ``s_volu``; dev events only.

    ``labels``: ``event_id, node_id, log_vol_ratio, log_dv_ratio``. With fewer rows than ``framework.pooled.pin_map_min_rows`` the map is zero (counted by the caller)."""
    need, ridge = int(cfg.get("framework.pooled.pin_map_min_rows")), float(cfg.get("framework.pooled.pin_map_ridge"))
    dev = set(dev_ids)
    rows = pd.concat([frame[["s_vol", "s_volu"]].assign(event_id=event, node_id=frame.index) for event, frame in source.items() if event in dev], ignore_index=True) if dev else pd.DataFrame()
    out: dict[str, tuple[float, float]] = {}
    joined = rows.merge(labels, on=["event_id", "node_id"], how="inner") if len(rows) else rows
    for dial, x_col, y_col in (("vol", "s_vol", "log_vol_ratio"), ("volu", "s_volu", "log_dv_ratio")):
        part = joined[[x_col, y_col]].replace([np.inf, -np.inf], np.nan).dropna() if len(joined) else pd.DataFrame()
        if len(part) < need:
            out[dial] = (0.0, 0.0)
            continue
        x, y = part[x_col].to_numpy(dtype=float), part[y_col].to_numpy(dtype=float)
        xc, yc = x - x.mean(), y - y.mean()
        slope = float((xc @ yc) / (xc @ xc + ridge * len(x)))
        out[dial] = (float(y.mean() - slope * x.mean()), slope)
    return out


def check_label_units(scale: np.ndarray, cfg: Config) -> None:
    """The build trains on pins and labels in percent; the framework converts with ``label_ret_to_fraction``. Stop when the fitted return scale says the labels are not in percent."""
    limit = float(cfg.get("framework.pooled.label_ret_scale_min"))
    if float(scale[0]) < limit:
        raise SystemExit(f"the fitted scale of the return label is {float(scale[0]):.4f} (< {limit}): the sheaf labels are not in percent, so the conversion to fractions would be wrong; stopping")


def make_dev_fit(fitted, links: pd.DataFrame, cfg: Config, pin_maps: Mapping[str, tuple[float, float]], n_dev_events: int, n_graphs: int, notes: dict | None = None) -> DevFit:
    """A ``DevFit`` for the framework cell from the build's fitted sheaf (``fitted.sheaf = (model, cv)``)."""
    if fitted.sheaf is None:
        raise ValueError("the build's fit holds no sheaf")
    check_label_units(fitted.scale, cfg)
    model, cv = fitted.sheaf
    predictor = PooledRoad1Predictor(model, float(cv.grounding), np.asarray(fitted.scale, dtype=float), links, cfg, pin_maps)
    info = {"dial_scale": [float(x) for x in fitted.scale], "grounding": float(cv.grounding), "penalty_mu": float(cv.penalty_mu), "pin_maps": {k: list(v) for k, v in pin_maps.items()},
            "road1_arm": ARM, **(notes or {})}
    return DevFit({}, predictor, n_dev_events, n_graphs, info)
