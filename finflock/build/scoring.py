"""Scored rows of the sheaf: one row per (event, free node, dial) with link features, plus the day-by-stock panel for the all-rows view.

The forecasts come from the existing road-1 code (``finflock.stage.road1_real.forecasts_for``); this module adds the features the slices need and
sums the event forecasts into the panel. No label is used to make a forecast.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from finflock.common.config import Config
from finflock.road1.graph import EventGraph
from finflock.stage import road1_real as rr

ROOT = "build"
DIALS = ("ret", "vol", "volu")
NON_OWNER = ("industry", "corr")
OWNER = "owner"


def link_features(graph: EventGraph, types: Sequence[str]) -> pd.DataFrame:
    """For each free node: which link types join it to a pinned node, the strongest weight per type, and how many pinned neighbours it has."""
    p, n_free = graph.n_pinned, graph.n_free
    has = np.zeros((n_free, len(types)), dtype=bool)
    weight = np.zeros((n_free, len(types)))
    neighbours: list[set[int]] = [set() for _ in range(n_free)]
    for a, b, t, w in zip(graph.src, graph.dst, graph.link_type, graph.weight, strict=True):
        for pinned, free in ((a, b), (b, a)):
            if pinned < p <= free:
                j = free - p
                has[j, t] = True
                weight[j, t] = max(weight[j, t], w)
                neighbours[j].add(int(pinned))
    cols: dict[str, Any] = {f"has_{t}": has[:, i] for i, t in enumerate(types)}
    cols.update({f"w_{t}": weight[:, i] for i, t in enumerate(types)})
    cols["n_pinned_neighbours"] = [len(s) for s in neighbours]
    return pd.DataFrame(cols)


def scored_rows(fitted: rr.Fitted, graphs: Sequence[EventGraph], day0: Mapping[str, pd.Timestamp], event_meta: pd.DataFrame, types: Sequence[str], cfg: Config) -> pd.DataFrame:
    """One row per (event, free node, dial) with the label and every method's forecast."""
    dials = list(cfg.get("road1.dial_names"))
    meta = event_meta.set_index("event_id")
    parts = []
    for graph in graphs:
        if graph.y is None or not graph.n_free:
            continue
        preds = rr.forecasts_for(fitted, graph, cfg)
        feats = link_features(graph, types)
        nodes = graph.nodes[graph.n_pinned :]
        for k, dial in enumerate(dials):
            frame = pd.DataFrame({"event_id": graph.event_id, "day": day0[graph.event_id], "node": nodes, "dial": dial, "y": graph.y[:, k], **{f"pred_{m}": p[:, k] for m, p in preds.items()}})
            frame["n_pinned"] = graph.n_pinned
            parts.append(pd.concat([frame, feats], axis=1))
    rows = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    rows = rows[np.isfinite(rows["y"])].reset_index(drop=True)
    for column in meta.columns:
        rows[column] = rows["event_id"].map(meta[column])
    return rows


def panel_labels(days: Sequence[pd.Timestamp], nodes: Sequence[str], panels: rr.Panels, cfg: Config) -> pd.DataFrame:
    """Labels (window d01) of every (day, stock) of the given days, built with the same label code as the events."""
    cap = int(cfg.get("build.panel.max_days"))
    days = list(days)[:cap] if cap > 0 else list(days)
    ids, node_of, day_of = [], {}, {}
    for day in days:
        for node in nodes:
            if node in panels.prices:
                key = f"{day:%Y%m%d}|{node}"
                ids.append(key)
                node_of[key] = (node,)
                day_of[key] = day
    labels = rr.label_table(node_of, day_of, panels, cfg)
    labels = labels[labels["window"] == str(cfg.get("build.horizon_window"))]
    out = labels.rename(columns={"abn_ret": "ret", "log_vol_ratio": "vol", "log_dv_ratio": "volu", "security_id": "node"})[["event_id", "node", "ret", "vol", "volu"]]
    out["day"] = out["event_id"].str.split("|").str[0].map(lambda s: pd.Timestamp(s))
    return out.drop(columns="event_id")


def panel_with_forecasts(panel: pd.DataFrame, rows: pd.DataFrame, methods: Sequence[str]) -> pd.DataFrame:
    """Long panel (day, node, dial) with the label, the summed event forecasts of each method (0 where no event reached the node) and the linked flag."""
    long = panel.melt(id_vars=["day", "node"], value_vars=list(DIALS), var_name="dial", value_name="y").dropna(subset=["y"])
    pred_cols = [f"pred_{m}" for m in methods]
    sums = rows.groupby(["day", "node", "dial"], sort=False)[pred_cols].sum().reset_index()
    linked = rows[rows["has_industry"] | rows["has_corr"]].groupby(["day", "node"], sort=False).size().rename("linked_events").reset_index()
    long = long.merge(sums, on=["day", "node", "dial"], how="left").merge(linked, on=["day", "node"], how="left")
    long[pred_cols] = long[pred_cols].fillna(0.0)
    long["linked_today"] = long["linked_events"].fillna(0).gt(0)
    return long.drop(columns="linked_events")
