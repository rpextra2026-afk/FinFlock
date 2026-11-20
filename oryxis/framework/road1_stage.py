"""Road 1 at prediction time: pinned source dials in, dials of every node of the subgraph out."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oryxis.common.config import Config
from oryxis.framework.events import EventReading
from oryxis.road1.arms import MatrixModel
from oryxis.road1.fit import Model, predict_graph
from oryxis.road1.graph import EventGraph, build_event_graph

DIALS = ("ret", "vol", "volu")


class Road1Predictor:
    """Wraps one fitted method (``wgraph``, ``sheaf`` or a matrix rung) and the dial scale."""

    def __init__(
        self, arm: str, model: Model | MatrixModel, grounding: float, scale: np.ndarray, links: pd.DataFrame, cfg: Config
    ) -> None:
        self.arm, self.model, self.grounding, self.scale, self.links, self.cfg = arm, model, grounding, scale, links, cfg

    def graph_for(self, reading: EventReading, source_dials: pd.DataFrame, asof: pd.Timestamp) -> EventGraph:
        nodes = [p["node_id"] for p in reading.pins]
        return build_event_graph(
            self.cfg, reading.event_id, nodes, source_dials.loc[:, list(DIALS)].to_numpy(), self.links, asof,
            reading.group_probs,
        )

    def predict(self, reading: EventReading, source_dials: pd.DataFrame, asof: pd.Timestamp) -> pd.DataFrame:
        """Dials of all nodes: pinned nodes keep their source dials, free nodes get the ripple."""
        graph = self.graph_for(reading, source_dials, asof)
        if isinstance(self.model, MatrixModel):
            standardised = EventGraph(
                graph.event_id, graph.nodes, graph.n_pinned, graph.src, graph.dst, graph.link_type, graph.weight,
                graph.s / self.scale, graph.group_probs,
            )
            free = self.model.predict(standardised) * self.scale
        else:
            free = predict_graph(self.model, graph, self.grounding, self.scale, self.cfg)
        values = np.vstack([graph.s, free]) if graph.n_free else graph.s
        frame = pd.DataFrame(values, columns=list(DIALS), index=pd.Index(graph.nodes, name="node_id"))
        frame["role"] = ["pinned"] * graph.n_pinned + ["ripple"] * graph.n_free
        frame["truncated"] = graph.truncated
        return frame
