"""Memory-bounded fitting: micro-batches give the same loss and gradients, and the worker count follows free memory (finflock/common/resources.py)."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from finflock.common import resources
from finflock.road1.arms import SheafModel
from finflock.road1.fit import BYTES_PER_VALUE, accumulate_gradients, mean_loss, micro_batches
from finflock.road1.graph import EventGraph

D = 3
TYPES = 9
GROUPS = 6


def graph(i: int, n_pinned: int, n_free: int, rng) -> EventGraph:
    n = n_pinned + n_free
    pairs = [(a, b) for a in range(n_pinned) for b in range(n_pinned, n)]
    src, dst = np.array([a for a, _ in pairs]), np.array([b for _, b in pairs])
    probs = rng.dirichlet(np.ones(GROUPS))
    return EventGraph(f"e{i}", tuple(f"n{k}" for k in range(n)), n_pinned, src, dst, rng.integers(0, TYPES, size=len(pairs)),
                      rng.uniform(0.2, 1.0, size=len(pairs)), rng.normal(size=(n_pinned, D)), probs, False, rng.normal(size=(n_free, D)))


@pytest.fixture
def graphs():
    rng = np.random.default_rng(0)
    sizes = [(1, 3), (2, 5), (1, 9), (3, 12), (1, 4), (2, 17), (1, 25), (2, 6), (1, 11), (1, 30)]
    return [graph(i, p, f, rng) for i, (p, f) in enumerate(sizes)]


def model_with_weights(seed: int = 0) -> SheafModel:
    torch.manual_seed(seed)
    m = SheafModel(TYPES, GROUPS, D, [True] * 4 + [False] * 5)
    with torch.no_grad():
        m.delta_u.normal_(0.0, 0.2)
        m.delta_v.normal_(0.0, 0.2)
    return m


def gradients(m: SheafModel, graphs, cap):
    m.zero_grad()
    value = accumulate_gradients(m, graphs, 0.05, 0.0, cap)
    return value, [p.grad.clone() for p in m.parameters()]


def test_micro_batches_respect_the_cap_and_keep_every_event_once(graphs):
    cap = 4 * 12**2 * D * D * BYTES_PER_VALUE
    parts = micro_batches(graphs, cap)
    assert sorted(g.event_id for part in parts for g in part) == sorted(g.event_id for g in graphs)
    assert len(parts) > len(micro_batches(graphs, None))
    for part in parts:
        nodes = max(g.n_pinned for g in part) + max(g.n_free for g in part)
        assert len(part) == 1 or len(part) * nodes**2 * D * D * BYTES_PER_VALUE <= cap


def test_the_gradient_and_loss_do_not_depend_on_the_micro_batch_size(graphs):
    whole_value, whole = gradients(model_with_weights(), graphs, None)
    tiny = 2 * 8**2 * D * D * BYTES_PER_VALUE
    assert len(micro_batches(graphs, tiny)) >= len(graphs) // 2
    small_value, small = gradients(model_with_weights(), graphs, tiny)
    assert small_value == pytest.approx(whole_value, rel=1e-10)
    for a, b in zip(whole, small, strict=True):
        assert torch.allclose(a, b, rtol=1e-8, atol=1e-12)
    with torch.no_grad():
        assert float(mean_loss(model_with_weights(), graphs, 0.05, 0.0, tiny)) == pytest.approx(float(mean_loss(model_with_weights(), graphs, 0.05, 0.0)), rel=1e-10)


def test_worker_plan_follows_free_memory_and_never_exceeds_the_request(cfg, monkeypatch):
    monkeypatch.setattr(resources, "usable_cpus", lambda: 16)
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: 8 * resources.BYTES_PER_GB)
    plan = resources.plan_workers(cfg)
    spec = cfg.get("cpu.resources")
    per_worker = int(float(spec["worker_base_gb"]) * resources.BYTES_PER_GB) + int(float(spec["peak_factor"]) * int(spec["max_batch_bytes"]))
    assert plan["workers"] == min(int(8 * resources.BYTES_PER_GB * float(spec["memory_fraction"]) // per_worker), int(spec["max_workers"]), 16)
    assert plan["workers"] < 16  # memory, not the core count, is what limits it
    assert resources.plan_workers(cfg, requested=1)["workers"] == 1
    monkeypatch.setattr(resources, "available_memory_bytes", lambda: 1)
    assert resources.plan_workers(cfg)["workers"] == 1  # always at least one


def test_resource_log_records_stage_time_and_memory(tmp_path):
    log = resources.ResourceLog(tmp_path / "r.json")
    with log.stage("allocate"):
        block = np.ones(2_000_000)  # a few MB, enough for the sampler to see something
        assert block.sum() > 0
    assert log.stages[0]["stage"] == "allocate" and log.stages[0]["peak_rss_gb"] > 0
    assert (tmp_path / "r.json").exists()
