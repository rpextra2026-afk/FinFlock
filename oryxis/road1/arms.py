"""The three road-1 methods (spec 8.2): weighted graph, weighted matrix ladder, sheaf.

All three read the same event subgraph and the same pinned dial values and predict the dials of the
free nodes. Dials are standardised by their dev standard deviation before fitting, so every method
works in dimensionless units.

* ``wgraph`` (A1): one scalar weight ``beta_t >= 0`` per link type, trivial sheaf, each dial diffuses
  on its own; ``x_F = -(L_FF + lambda I)^-1 L_FP s`` (a joint solve over all pinned nodes).
* ``wmatrix`` (A2): one-hop table ``x_j = sum_i sum_t w_ij B_t s_i`` with three rungs: scalar ``b_t I``,
  free ``3x3`` per link type, and free ``3x3`` per link type and cause group mixed by ``pi_g``.
* ``sheaf`` (A3): restriction maps ``F = I + Delta`` per link type and cause group, coboundary
  ``F_v x_v - F_u x_u`` on every edge, Laplacian ``sum_e w_e B_e' B_e``, same ripple formula.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from oryxis.road1.graph import EventGraph

DTYPE = torch.float64
NODE_DIAL_AXES = (0, 2, 1, 3)  # (n, n, d, d) -> (n, d, n, d)
BATCH_NODE_DIAL_AXES = (0, 1, 3, 2, 4)  # (B, n, n, d, d) -> (B, n, d, n, d)


def _to_tensor(array: np.ndarray) -> torch.Tensor:
    return torch.as_tensor(np.asarray(array), dtype=DTYPE)


# --- Laplacian and ripple ------------------------------------------------------------------


def assemble_laplacian(
    n: int, d: int, src: torch.Tensor, dst: torch.Tensor, weight: torch.Tensor,
    Fu: torch.Tensor, Fv: torch.Tensor,
) -> torch.Tensor:
    """Sheaf Laplacian ``sum_e w_e B_e' B_e`` as a ``(n d) x (n d)`` matrix, index ``node * d + dial``.

    ``(B_e x) = F_v x_v - F_u x_u`` with edge maps ``Fu, Fv`` of shape ``(E, d, d)``.
    """
    w = weight[:, None, None]
    L_uu = w * Fu.transpose(1, 2) @ Fu
    L_vv = w * Fv.transpose(1, 2) @ Fv
    L_uv = -w * Fu.transpose(1, 2) @ Fv
    blocks = torch.zeros(n, n, d, d, dtype=DTYPE)
    blocks = blocks.index_put((src, src), L_uu, accumulate=True)
    blocks = blocks.index_put((dst, dst), L_vv, accumulate=True)
    blocks = blocks.index_put((src, dst), L_uv, accumulate=True)
    blocks = blocks.index_put((dst, src), L_uv.transpose(1, 2), accumulate=True)
    return blocks.permute(*NODE_DIAL_AXES).reshape(n * d, n * d)


def ripple(
    L: torch.Tensor, n_pinned: int, s: torch.Tensor, grounding: float
) -> torch.Tensor:
    """Free-node values ``x_F = -(L_FF + lambda I)^-1 L_FP s`` (D5 (ii)); ``s`` is ``(n_pinned, d)``."""
    d = s.shape[1]
    p = n_pinned * d
    L_FF, L_FP = L[p:, p:], L[p:, :p]
    if L_FF.shape[0] == 0:
        return torch.zeros(0, d, dtype=DTYPE)
    system = L_FF + grounding * torch.eye(L_FF.shape[0], dtype=DTYPE)
    x = torch.linalg.solve(system, -L_FP @ s.reshape(-1))
    return x.reshape(-1, d)


def identity_maps(n_edges: int, d: int) -> torch.Tensor:
    return torch.eye(d, dtype=DTYPE).expand(n_edges, d, d)


# --- batching ------------------------------------------------------------------------------


@dataclass
class Batch:
    """Several events padded to common sizes so a whole minibatch is solved in one call.

    Layout per event: ``P`` pinned slots then ``F`` free slots. Padding slots are isolated nodes
    (no edges): they add nothing to the Laplacian and their predictions are masked from the loss.
    """

    n_nodes: int
    n_pinned: int
    n_free: int
    src: torch.Tensor  # (B, E)
    dst: torch.Tensor  # (B, E)
    link_type: torch.Tensor  # (B, E)
    weight: torch.Tensor  # (B, E); zero for padded edges
    s: torch.Tensor  # (B, P, d)
    y: torch.Tensor | None  # (B, F, d)
    mask: torch.Tensor  # (B, F) real free nodes
    probs: torch.Tensor  # (B, G)
    n_real_free: torch.Tensor  # (B,)


def collate(graphs: Sequence[EventGraph]) -> Batch:
    """Pad a list of event graphs to a batch."""
    d = graphs[0].s.shape[1]
    P = max(g.n_pinned for g in graphs)
    F = max(g.n_free for g in graphs)
    E = max(len(g.src) for g in graphs)
    B = len(graphs)
    src = np.zeros((B, E), dtype=np.int64)
    dst = np.zeros((B, E), dtype=np.int64)
    typ = np.zeros((B, E), dtype=np.int64)
    weight = np.zeros((B, E))
    s = np.zeros((B, P, d))
    y = np.zeros((B, F, d))
    mask = np.zeros((B, F), dtype=bool)
    has_labels = all(g.y is not None for g in graphs)
    for b, g in enumerate(graphs):
        def slot(i: np.ndarray, g=g) -> np.ndarray:
            return np.where(i < g.n_pinned, i, P + (i - g.n_pinned))

        k = len(g.src)
        src[b, :k], dst[b, :k], typ[b, :k], weight[b, :k] = slot(g.src), slot(g.dst), g.link_type, g.weight
        s[b, : g.n_pinned] = g.s
        mask[b, : g.n_free] = True
        if has_labels:
            finite = np.isfinite(g.y).all(axis=1)  # nodes without a label (e.g. no prices) are masked out
            mask[b, : g.n_free] = finite
            y[b, : g.n_free] = np.where(finite[:, None], g.y, 0.0)
    return Batch(
        P + F, P, F, torch.as_tensor(src), torch.as_tensor(dst), torch.as_tensor(typ),
        _to_tensor(weight), _to_tensor(s), _to_tensor(y) if has_labels else None,
        torch.as_tensor(mask), _to_tensor(np.stack([g.group_probs for g in graphs])),
        torch.as_tensor(mask.sum(axis=1), dtype=DTYPE),
    )


def batch_laplacian(
    n: int, d: int, src: torch.Tensor, dst: torch.Tensor, weight: torch.Tensor,
    Fu: torch.Tensor, Fv: torch.Tensor,
) -> torch.Tensor:
    """Batched ``assemble_laplacian``: inputs ``(B, E, ...)``, output ``(B, n d, n d)``."""
    B = src.shape[0]
    w = weight[..., None, None]
    L_uu = w * Fu.transpose(-1, -2) @ Fu
    L_vv = w * Fv.transpose(-1, -2) @ Fv
    L_uv = -w * Fu.transpose(-1, -2) @ Fv
    b = torch.arange(B)[:, None].expand_as(src)
    blocks = torch.zeros(B, n, n, d, d, dtype=DTYPE)
    blocks = blocks.index_put((b, src, src), L_uu, accumulate=True)
    blocks = blocks.index_put((b, dst, dst), L_vv, accumulate=True)
    blocks = blocks.index_put((b, src, dst), L_uv, accumulate=True)
    blocks = blocks.index_put((b, dst, src), L_uv.transpose(-1, -2), accumulate=True)
    return blocks.permute(*BATCH_NODE_DIAL_AXES).reshape(B, n * d, n * d)


def batch_ripple(L: torch.Tensor, n_pinned: int, s: torch.Tensor, grounding: float) -> torch.Tensor:
    """Batched ripple: ``L`` is ``(B, n d, n d)``, ``s`` is ``(B, P, d)``; returns ``(B, F, d)``."""
    B, _, d = s.shape
    p = n_pinned * d
    L_FF, L_FP = L[:, p:, p:], L[:, p:, :p]
    system = L_FF + grounding * torch.eye(L_FF.shape[1], dtype=DTYPE)
    x = torch.linalg.solve(system, -(L_FP @ s.reshape(B, -1, 1)))
    return x.reshape(B, -1, d)


# --- A1: weighted graph --------------------------------------------------------------------


class GraphModel(torch.nn.Module):
    """One non-negative weight per link type; identity restriction maps."""

    def __init__(self, n_types: int, d: int):
        super().__init__()
        self.d = d
        self.raw_beta = torch.nn.Parameter(torch.zeros(n_types, dtype=DTYPE))  # softplus(0) ~ 0.69

    @property
    def beta(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.raw_beta)

    def predict(self, graph: EventGraph, grounding: float, min_group_prob: float = 0.0) -> torch.Tensor:
        src, dst = torch.as_tensor(graph.src), torch.as_tensor(graph.dst)
        types = torch.as_tensor(graph.link_type)
        weight = _to_tensor(graph.weight) * self.beta[types]
        eye = identity_maps(len(graph.src), self.d)
        L = assemble_laplacian(graph.n_nodes, self.d, src, dst, weight, eye, eye)
        return ripple(L, graph.n_pinned, _to_tensor(graph.s), grounding)

    def predict_batch(self, batch: Batch, grounding: float, min_group_prob: float = 0.0) -> torch.Tensor:
        weight = batch.weight * self.beta[batch.link_type]
        eye = torch.eye(self.d, dtype=DTYPE).expand(*batch.src.shape, self.d, self.d)
        L = batch_laplacian(batch.n_nodes, self.d, batch.src, batch.dst, weight, eye, eye)
        return batch_ripple(L, batch.n_pinned, batch.s, grounding)

    def penalty(self) -> torch.Tensor:
        return torch.zeros((), dtype=DTYPE)


# --- A3: sheaf -----------------------------------------------------------------------------


class SheafModel(torch.nn.Module):
    """Restriction maps ``F = I + Delta`` per link type and cause group (tied for symmetric types)."""

    def __init__(self, n_types: int, n_groups: int, d: int, symmetric: Sequence[bool]):
        super().__init__()
        self.d = d
        self.delta_u = torch.nn.Parameter(torch.zeros(n_types, n_groups, d, d, dtype=DTYPE))
        self.delta_v = torch.nn.Parameter(torch.zeros(n_types, n_groups, d, d, dtype=DTYPE))
        self.register_buffer("symmetric", torch.as_tensor(list(symmetric), dtype=torch.bool))

    def n_parameters(self) -> int:
        sym = self.symmetric
        per_map = self.delta_u.shape[1] * self.d * self.d
        return int(per_map * (sym.sum() + 2 * (~sym).sum()))

    def penalty(self) -> torch.Tensor:
        """Squared norm of ``Delta`` over the free parameters (the tied half of symmetric types is unused)."""
        keep = (~self.symmetric).to(DTYPE)[:, None, None, None]
        return (self.delta_u**2).sum() + ((self.delta_v * keep) ** 2).sum()

    def edge_maps(self, graph: EventGraph, group: int) -> tuple[torch.Tensor, torch.Tensor]:
        types = torch.as_tensor(graph.link_type)
        eye = torch.eye(self.d, dtype=DTYPE)
        Fu = eye + self.delta_u[types, group]
        Fv_free = eye + self.delta_v[types, group]
        Fv = torch.where(self.symmetric[types][:, None, None], Fu, Fv_free)
        return Fu, Fv

    def laplacian(self, graph: EventGraph, group: int) -> torch.Tensor:
        Fu, Fv = self.edge_maps(graph, group)
        return assemble_laplacian(
            graph.n_nodes, self.d, torch.as_tensor(graph.src), torch.as_tensor(graph.dst),
            _to_tensor(graph.weight), Fu, Fv,
        )

    def predict(self, graph: EventGraph, grounding: float, min_group_prob: float = 0.0) -> torch.Tensor:
        """Ripple under each cause group, averaged with the reader's group probabilities (rule A)."""
        s = _to_tensor(graph.s)
        total = torch.zeros(graph.n_free, self.d, dtype=DTYPE)
        for group, prob in enumerate(graph.group_probs):
            if prob <= min_group_prob:
                continue
            total = total + float(prob) * ripple(self.laplacian(graph, group), graph.n_pinned, s, grounding)
        return total

    def predict_batch(self, batch: Batch, grounding: float, min_group_prob: float = 0.0) -> torch.Tensor:
        """Batched ``predict``: ripple per cause group on padded events, mixed with ``pi_g``."""
        eye = torch.eye(self.d, dtype=DTYPE)
        symmetric = self.symmetric[batch.link_type][..., None, None]
        total = torch.zeros(batch.src.shape[0], batch.n_free, self.d, dtype=DTYPE)
        for group in range(batch.probs.shape[1]):
            prob = batch.probs[:, group]
            if not bool((prob > min_group_prob).any()):
                continue
            Fu = eye + self.delta_u[batch.link_type, group]
            Fv = torch.where(symmetric, Fu, eye + self.delta_v[batch.link_type, group])
            L = batch_laplacian(batch.n_nodes, self.d, batch.src, batch.dst, batch.weight, Fu, Fv)
            total = total + prob[:, None, None] * batch_ripple(L, batch.n_pinned, batch.s, grounding)
        return total


# --- A2: weighted matrix (closed-form ridge) -----------------------------------------------


def matrix_features(graph: EventGraph, n_types: int) -> np.ndarray:
    """``Phi[j, t, b] = sum over edges of type t between a pinned node i and free j of w * s_i[b]``."""
    d = graph.s.shape[1]
    phi = np.zeros((graph.n_free, n_types, d))
    p = graph.n_pinned
    for a, b, t, w in zip(graph.src, graph.dst, graph.link_type, graph.weight, strict=True):
        for pinned, free in ((a, b), (b, a)):
            if pinned < p <= free:
                phi[free - p, t] += w * graph.s[pinned]
    return phi


def matrix_design(graph: EventGraph, rung: int, n_types: int) -> np.ndarray:
    """Features of the free nodes for the rung's regression.

    * rung 1: ``Phi`` itself, shape ``(n_free, T, d)``; one scalar ``b_t`` per type is shared by the
      dials, so output dial ``a`` uses ``Phi[:, :, a]``.
    * rung 2: ``Phi`` flattened to ``(n_free, T d)``; each output dial has its own ``B_t[a, :]``.
    * rung 3: ``pi_g * Phi`` flattened and concatenated over cause groups, ``(n_free, G T d)``.
    """
    phi = matrix_features(graph, n_types)
    if rung == 1:
        return phi
    flat = phi.reshape(graph.n_free, -1)
    if rung == 2:
        return flat
    return np.concatenate([prob * flat for prob in graph.group_probs], axis=1)


@dataclass
class MatrixModel:
    """Fitted one-hop table in standardised dial units.

    ``theta``: rung 1 ``(T,)``; rungs 2 and 3 ``(d, n_features)`` with one row per output dial.
    """

    rung: int
    theta: np.ndarray
    n_types: int

    def predict(self, graph: EventGraph) -> np.ndarray:
        design = matrix_design(graph, self.rung, self.n_types)
        if self.rung == 1:
            return np.einsum("t,jta->ja", self.theta, design)
        return design @ self.theta.T

    def n_parameters(self) -> int:
        return int(self.theta.size)
