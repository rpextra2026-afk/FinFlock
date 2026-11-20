"""Per-event subgraph for road 1 (spec 8.1): pinned nodes, their strongest neighbours, typed edges."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from finfm.common.config import Config

LINK_SRC = "src_node"
LINK_DST = "dst_node"
LINK_TYPE = "link_type"
LINK_WEIGHT = "weight"
LINK_FROM = "available_from"
LINK_TO = "valid_to"


@dataclass(frozen=True)
class EventGraph:
    """One event's subgraph. Node order: pinned nodes first (in the order given), then free nodes."""

    event_id: str
    nodes: tuple[str, ...]
    n_pinned: int
    src: np.ndarray  # (E,) node index
    dst: np.ndarray  # (E,)
    link_type: np.ndarray  # (E,) index into config road1.link_types
    weight: np.ndarray  # (E,) in (0, 1]
    s: np.ndarray  # (n_pinned, d) pinned dial values
    group_probs: np.ndarray  # (G,) cause-group probabilities
    truncated: bool = False
    y: np.ndarray | None = None  # (n_free, d) free-node dial labels (dev only)
    meta: dict = field(default_factory=dict)

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def n_free(self) -> int:
        return self.n_nodes - self.n_pinned

    def with_labels(self, y: np.ndarray) -> EventGraph:
        if y.shape != (self.n_free, self.s.shape[1]):
            raise ValueError("labels must have one row per free node")
        return EventGraph(
            self.event_id, self.nodes, self.n_pinned, self.src, self.dst, self.link_type,
            self.weight, self.s, self.group_probs, self.truncated, y, self.meta,
        )


def available_links(links: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
    """Links known at ``asof``: ``available_from <= asof < valid_to`` (null valid_to = open)."""
    started = pd.to_datetime(links[LINK_FROM]) <= asof
    open_end = links[LINK_TO].isna() | (pd.to_datetime(links[LINK_TO]) > asof)
    return links[started & open_end]


def build_event_graph(
    cfg: Config,
    event_id: str,
    pinned_nodes: list[str],
    pinned_dials: np.ndarray,
    links: pd.DataFrame,
    asof: pd.Timestamp,
    group_probs: np.ndarray,
) -> EventGraph:
    """Select the subgraph around the pinned nodes and index it.

    Candidates: the pinned nodes, the ``neighbours_per_type`` strongest neighbours of each pinned
    node per link type, and any node linked to two or more pinned nodes (shielding paths). The
    result is capped at ``max_nodes`` by strongest link to a pinned node; truncation is flagged.
    All links between the selected nodes that are available at ``asof`` become edges.
    """
    types = list(cfg.get("road1.link_types"))
    per_type = int(cfg.get("road1.neighbours_per_type"))
    max_nodes = int(cfg.get("road1.max_nodes"))
    known = available_links(links, asof)
    known = known[known[LINK_TYPE].isin(types)]
    pinned_set = set(pinned_nodes)

    both = pd.concat(
        [
            known.rename(columns={LINK_SRC: "pin", LINK_DST: "other"})[["pin", "other", LINK_TYPE, LINK_WEIGHT]],
            known.rename(columns={LINK_DST: "pin", LINK_SRC: "other"})[["pin", "other", LINK_TYPE, LINK_WEIGHT]],
        ],
        ignore_index=True,
    )
    touching = both[both["pin"].isin(pinned_set) & ~both["other"].isin(pinned_set)]
    ranked = touching.sort_values(LINK_WEIGHT, ascending=False, kind="mergesort")
    top = ranked.groupby(["pin", LINK_TYPE], sort=False).head(per_type)
    shared = touching.groupby("other")["pin"].nunique()
    candidates = set(top["other"]) | set(shared[shared >= 2].index)

    strength = touching.groupby("other")[LINK_WEIGHT].max()
    ordered = sorted(candidates, key=lambda n: (-float(strength.get(n, 0.0)), n))
    truncated = len(ordered) + len(pinned_nodes) > max_nodes
    free_nodes = ordered[: max(max_nodes - len(pinned_nodes), 0)]

    nodes = list(pinned_nodes) + free_nodes
    index = {n: i for i, n in enumerate(nodes)}
    inside = known[known[LINK_SRC].isin(index) & known[LINK_DST].isin(index)]
    inside = inside.sort_values(LINK_WEIGHT, ascending=False, kind="mergesort").drop_duplicates(
        [LINK_SRC, LINK_DST, LINK_TYPE]
    )
    type_index = {t: i for i, t in enumerate(types)}
    return EventGraph(
        event_id=event_id,
        nodes=tuple(nodes),
        n_pinned=len(pinned_nodes),
        src=inside[LINK_SRC].map(index).to_numpy(dtype=int),
        dst=inside[LINK_DST].map(index).to_numpy(dtype=int),
        link_type=inside[LINK_TYPE].map(type_index).to_numpy(dtype=int),
        weight=inside[LINK_WEIGHT].to_numpy(dtype=float),
        s=np.asarray(pinned_dials, dtype=float),
        group_probs=np.asarray(group_probs, dtype=float),
        truncated=truncated,
    )


def group_probabilities(cause_probs: dict[str, float], cfg: Config) -> np.ndarray:
    """Sum the reader's cause probabilities into the six cause groups (spec 8.1)."""
    groups = cfg.get("causes.groups")
    return np.array([sum(cause_probs.get(c, 0.0) for c in members) for members in groups.values()])
