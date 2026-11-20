"""The sheaf's fitted internals, per tape, as small tables for figures (``figure_data/sheaf_*.parquet``). Fit-window (dev) rows only: no test row is read.

  sheaf_maps       one row per link type x cause group x side (u, v): the fitted Delta (d x d entries), its Frobenius norm and the singular values of I + Delta
  sheaf_cv_grid    every grounding x penalty candidate of the build's dev cross-validation with its validation loss and the chosen one flagged
  sheaf_chosen     the chosen grounding and penalty, the dev CV loss and the number of parameters of each arm
  sheaf_ripple_dev per link type: size of the sheaf's predicted ripple on the dev graphs' free nodes (mean absolute, median, 95th percentile, per dial, in label units)

Memory: the dev graphs are read from ``graphs.pkl`` one at a time (the test graphs of that file are dropped at once and never used).
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from finfm.common.config import Config
from finfm.road1.fit import predict_graph

DIAL_NAMES = ("ret", "vol", "volu")
QUANTILES = (0.5, 0.95)
PERCENT = 100


def maps_table(model, link_types: list[str]) -> pd.DataFrame:
    """Fitted Delta and the singular values of I + Delta, per link type, cause group and side."""
    delta = {"u": model.delta_u.detach().cpu().numpy(), "v": model.delta_v.detach().cpu().numpy()}
    symmetric = model.symmetric.cpu().numpy().astype(bool)
    d = int(model.d)
    rows = []
    for side, tensor in delta.items():
        for t in range(tensor.shape[0]):
            for g in range(tensor.shape[1]):
                dm = tensor[t, g]
                sv = np.linalg.svd(np.eye(d) + dm, compute_uv=False)
                rows.append({"link_type": link_types[t] if t < len(link_types) else str(t), "cause_group": g, "side": side, "symmetric_type": bool(symmetric[t]), "tied_to_u": bool(symmetric[t]) and side == "v",
                             "delta_frobenius": float(np.linalg.norm(dm)), **{f"delta_{i}{j}": float(dm[i, j]) for i in range(d) for j in range(d)}, **{f"singular_value_{k + 1}": float(v) for k, v in enumerate(sv)}})
    return pd.DataFrame(rows)


def chosen_tables(payload: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid = payload["configs"].copy()
    grid["best_steps"] = grid["best_steps"].map(lambda x: ",".join(str(i) for i in x))
    fits = payload["fits"].copy()
    sheaf = fits[fits["method"] == "sheaf"].iloc[0]
    grid["chosen"] = [(r.arm == "sheaf" and r.grounding == sheaf["grounding"] and r.penalty_mu == sheaf["penalty"]) or (r.arm == "wgraph" and r.grounding == fits[fits["method"] == "wgraph"].iloc[0]["grounding"]) for r in grid.itertuples()]
    return grid, fits


def ripple_by_link_type(model, grounding: float, scale: np.ndarray, graphs_path: Path, link_types: list[str], cfg: Config) -> pd.DataFrame:
    """Absolute ripple on the free nodes of the dev graphs, grouped by the link type of each node's strongest incident edge."""
    with Path(graphs_path).open("rb") as handle:
        graphs = pickle.load(handle)
    dev = list(graphs["dev"])
    del graphs   # the test graphs of this file are not read
    stride = max(1, len(dev) // int(cfg.get("framework.figure_data.max_dev_graphs")))
    values: dict[int, list[np.ndarray]] = {}
    with torch.no_grad():
        for graph in dev[::stride]:
            if not graph.n_free:
                continue
            ripple = np.abs(np.asarray(predict_graph(model, graph, grounding, scale, cfg)))
            src, dst, kind, weight = (np.asarray(x) for x in (graph.src, graph.dst, graph.link_type, graph.weight, ))
            best_w = np.full(graph.n_nodes, -np.inf)
            best_t = np.full(graph.n_nodes, -1)
            for node_ids in (src, dst):
                for e in np.argsort(weight):   # ascending: the strongest edge of a node is written last
                    n = node_ids[e]
                    if weight[e] >= best_w[n]:
                        best_w[n], best_t[n] = weight[e], kind[e]
            free_types = best_t[graph.n_pinned :]
            for t in np.unique(free_types):
                values.setdefault(int(t), []).append(ripple[free_types == t])
    rows = []
    for t, parts in sorted(values.items()):
        stack = np.vstack(parts)
        for k, name in enumerate(DIAL_NAMES):
            rows.append({"link_type": link_types[t] if 0 <= t < len(link_types) else "none", "dial": name, "n_free_node_rows": int(len(stack)), "abs_ripple_mean": float(stack[:, k].mean()),
                         **{f"abs_ripple_p{int(q * PERCENT)}": float(np.quantile(stack[:, k], q)) for q in QUANTILES}, "n_dev_graphs": int(len(parts))})
    return pd.DataFrame(rows)


def sheaf_internal_tables(fitted_path: Path, graphs_path: Path | None, cfg: Config) -> dict[str, pd.DataFrame]:
    with Path(fitted_path).open("rb") as handle:
        payload = pickle.load(handle)
    fitted = payload["fitted"]
    model, chosen = fitted.sheaf
    link_types = list(cfg.get("road1.link_types"))
    grid, fits = chosen_tables(payload)
    out = {"sheaf_maps": maps_table(model, link_types), "sheaf_cv_grid": grid, "sheaf_chosen": fits}
    if graphs_path is not None and Path(graphs_path).exists():
        out["sheaf_ripple_dev"] = ripple_by_link_type(model, float(chosen.grounding), np.asarray(fitted.scale), graphs_path, link_types, cfg)
    return out
