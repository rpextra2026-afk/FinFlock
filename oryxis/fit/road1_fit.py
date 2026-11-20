"""Dev-only assembly of road-1 training graphs: reader readings + links + label table (labels as data)."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oryxis.common.config import Config
from oryxis.framework.events import EventReading
from oryxis.road1.graph import EventGraph, build_event_graph

LABEL_WINDOW = "d01"
N_DIALS = 3  # return, volatility, volume


def label_matrix(labels: pd.DataFrame) -> dict[tuple[str, str], np.ndarray]:
    """``(event_id, security_id) -> [abn_ret(d01), log_vol_ratio, log_dv_ratio]`` from the label table."""
    window = labels[labels["window"] == LABEL_WINDOW]
    out = {}
    for r in window.itertuples():
        values = np.array([r.abn_ret, r.log_vol_ratio, r.log_dv_ratio], dtype=float)
        out[(r.event_id, r.security_id)] = np.where(np.isfinite(values), values, np.nan)  # a log ratio against a zero baseline is missing, not infinite
    return out


def training_graphs(
    readings: dict[str, EventReading], source_dials: dict[str, pd.DataFrame], links: pd.DataFrame,
    day0: dict[str, pd.Timestamp], labels: pd.DataFrame, cfg: Config, node_to_security: dict[str, str] | None = None,
) -> list[EventGraph]:
    """Event graphs with free-node labels (NaN where a node has no label, e.g. no prices)."""
    known = label_matrix(labels)
    out = []
    for event_id, reading in readings.items():
        if event_id not in source_dials or event_id not in day0:
            continue
        nodes = [p["node_id"] for p in reading.pins]
        graph = build_event_graph(
            cfg, event_id, nodes, source_dials[event_id][["ret", "vol", "volu"]].to_numpy(), links, day0[event_id],
            reading.group_probs,
        )
        if not graph.n_free:
            continue
        lookup = node_to_security or {}
        y = np.vstack([known.get((event_id, lookup.get(n, n)), np.full(N_DIALS, np.nan)) for n in graph.nodes[graph.n_pinned:]])
        out.append(graph.with_labels(y))
    return out
