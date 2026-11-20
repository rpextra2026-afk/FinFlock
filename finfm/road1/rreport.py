"""The sheaf regime ``r`` as a reported distribution (spec 8.3, D10 Prop. 10.2).

``r = rho(Delta^-1/2 W Delta^-1/2)`` for ``L_FF + lambda I = Delta - W`` measures how strongly free nodes interact relative to their own
grounding; ``r / (1 - r)`` bounds the amplification. It is computed for the fitted weighted graph (identity maps, fitted ``beta_t``) and for the
fitted sheaf (the maps of the event's dominant cause group), on a hashed sample of event subgraphs, and reported as median / percentile /
share of subgraphs with ``r >= 1`` per stratum. It is a measurement: the config threshold only labels which side of it the median falls on.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
import torch

from finfm.common.config import Config
from finfm.common.hashing import sha256_hex
from finfm.road1.arms import GraphModel, SheafModel, assemble_laplacian, identity_maps
from finfm.road1.graph import EventGraph
from finfm.road1.rkill import free_system, r_bound, r_value

TENSOR_DTYPE = torch.float64
COLUMNS = ["model", "event_id", "stratum", "n_pinned", "n_free", "r", "bound", "positive_definite"]


def sample_events(graphs: Sequence[EventGraph], limit: int | None, seed: str) -> list[EventGraph]:
    """The graphs with free nodes, in a fixed hashed order, cut to ``limit`` (``None`` = all)."""
    usable = sorted((g for g in graphs if g.n_free > 0), key=lambda g: sha256_hex(f"{seed}|{g.event_id}"))
    return usable if limit is None else usable[:limit]


def laplacian_of(model: GraphModel | SheafModel, graph: EventGraph) -> torch.Tensor:
    """Full Laplacian (pinned nodes first) of a fitted model on one event graph, in the standardised units the model was fitted in."""
    if isinstance(model, SheafModel):
        group = int(np.argmax(graph.group_probs))
        return model.laplacian(graph, group)
    types = torch.as_tensor(graph.link_type)
    weight = torch.as_tensor(graph.weight, dtype=TENSOR_DTYPE) * model.beta[types]
    eye = identity_maps(len(graph.src), model.d)
    return assemble_laplacian(graph.n_nodes, model.d, torch.as_tensor(graph.src), torch.as_tensor(graph.dst), weight, eye, eye)


def r_table(models: dict[str, tuple[GraphModel | SheafModel, float]], graphs: Sequence[EventGraph], strata: dict[str, str], cfg: Config) -> pd.DataFrame:
    """One row per (model, event). ``models``: name -> (fitted model, grounding)."""
    spec = cfg.get("cpu.road1_eval.r_report")
    chosen = sample_events(graphs, spec["max_events"], str(spec["seed_label"]))
    d = len(cfg.get("road1.dial_names"))
    rows = []
    with torch.no_grad():
        for name, (model, grounding) in models.items():
            for graph in chosen:
                stats = r_value(free_system(laplacian_of(model, graph).numpy(), graph.n_pinned, d, grounding), d)
                rows.append({"model": name, "event_id": graph.event_id, "stratum": strata.get(graph.event_id, ""), "n_pinned": graph.n_pinned,
                             "n_free": graph.n_free, "r": stats.r, "bound": r_bound(stats.r), "positive_definite": stats.positive_definite})
    return pd.DataFrame(rows, columns=COLUMNS)


def r_summary(table: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Median, percentile and spread of ``r`` per model and stratum (all strata pooled as ``all``), with the configured threshold as a label."""
    spec = cfg.get("cpu.road1_eval.r_report")
    pooled = table.assign(stratum="all")
    rows = []
    for (model, stratum), part in pd.concat([pooled, table]).groupby(["model", "stratum"]):
        below = part[part["r"] < 1.0]
        rows.append({"model": model, "stratum": stratum, "n_events": len(part), "median_r": float(part["r"].median()),
                     "percentile_r": float(part["r"].quantile(float(spec["percentile"]))), "max_r": float(part["r"].max()),
                     "share_r_at_least_1": float((part["r"] >= 1.0).mean()), "median_bound": float(below["bound"].median()) if len(below) else float("nan"),
                     "threshold": float(spec["kill_threshold"]), "median_r_below_threshold": bool(part["r"].median() < float(spec["kill_threshold"])),
                     "percentile": float(spec["percentile"])})
    return pd.DataFrame(rows)
