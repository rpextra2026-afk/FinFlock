"""Road 1: subgraph, Laplacian, the three methods, r, tension, the sheaf test, CV (spec 12.2 #10)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
import torch
from scipy.optimize import minimize

from finflock.common.cv import time_blocked_folds
from finflock.common.timing import TradingCalendar
from finflock.road1.arms import (
    DTYPE,
    GraphModel,
    SheafModel,
    assemble_laplacian,
    collate,
    identity_maps,
    matrix_design,
    ripple,
)
from finflock.road1.fit import (
    dial_scale,
    fit_matrix,
    new_model,
    oos_r2,
    predict_graph,
    select_and_fit,
    standardise,
    train_model,
)
from finflock.road1.graph import (
    EventGraph,
    build_event_graph,
    group_probabilities,
)
from finflock.road1.rkill import r_bound, r_value
from finflock.road1.sheaf_test import d10_regression
from finflock.road1.tension import tension, tension_matrix

D = 2  # dials in the synthetic tests (the code is generic in d; production uses 3)


def random_graph(rng, n_pinned=2, n_free=6, n_types=2, n_groups=1, with_labels=None, event_id="e"):
    n = n_pinned + n_free
    pairs = [(a, b) for a in range(n) for b in range(a + 1, n) if rng.random() < 0.5]
    # every free node touches the graph so the problem is connected enough to be informative
    for j in range(n_pinned, n):
        pairs.append((int(rng.integers(0, j)), j))
    pairs = sorted(set(pairs))
    src, dst = np.array([p[0] for p in pairs]), np.array([p[1] for p in pairs])
    probs = rng.dirichlet(np.ones(n_groups)) if n_groups > 1 else np.ones(1)
    return EventGraph(
        event_id, tuple(f"n{i}" for i in range(n)), n_pinned, src, dst,
        rng.integers(0, n_types, len(pairs)), rng.uniform(0.2, 1.0, len(pairs)),
        rng.normal(size=(n_pinned, D)), probs,
    )


def laplacian_np(graph, model=None, group=0):
    if model is None:
        eye = identity_maps(len(graph.src), D)
        return assemble_laplacian(graph.n_nodes, D, torch.as_tensor(graph.src), torch.as_tensor(graph.dst),
                                  torch.as_tensor(graph.weight), eye, eye).numpy()
    return model.laplacian(graph, group).detach().numpy()


# --- Laplacian and ripple -----------------------------------------------------------------


def test_laplacian_is_symmetric_psd_and_identity_maps_give_the_graph_laplacian(cfg):
    rng = np.random.default_rng(0)
    g = random_graph(rng)
    L = laplacian_np(g)
    assert np.allclose(L, L.T) and np.linalg.eigvalsh(L).min() > -1e-10
    plain = np.zeros((g.n_nodes, g.n_nodes))
    for a, b, w in zip(g.src, g.dst, g.weight):
        plain[a, a] += w
        plain[b, b] += w
        plain[a, b] -= w
        plain[b, a] -= w
    assert np.allclose(L, np.kron(plain, np.eye(D)))  # trivial sheaf = graph Laplacian x identity
    model = SheafModel(2, 1, D, [True, False])
    with torch.no_grad():
        model.delta_u.normal_(0, 0.3)
        model.delta_v.normal_(0, 0.3)
    Ls = laplacian_np(g, model)
    assert np.allclose(Ls, Ls.T) and np.linalg.eigvalsh(Ls).min() > -1e-10


def test_batched_prediction_equals_one_graph_at_a_time_for_both_models():
    rng = np.random.default_rng(14)
    graphs = [random_graph(rng, n_pinned=int(rng.integers(1, 4)), n_free=int(rng.integers(2, 8)),
                           n_groups=3, event_id=f"e{i}") for i in range(7)]
    sheaf = SheafModel(2, 3, D, [True, False])
    with torch.no_grad():
        sheaf.delta_u.normal_(0, 0.3)
        sheaf.delta_v.normal_(0, 0.3)
    graph_model = GraphModel(2, D)
    with torch.no_grad():
        graph_model.raw_beta.normal_(0, 1)
    batch = collate(graphs)
    for model in (sheaf, graph_model):
        batched = model.predict_batch(batch, 0.05).detach().numpy()
        for b, g in enumerate(graphs):
            single = model.predict(g, 0.05).detach().numpy()
            assert np.allclose(batched[b, : g.n_free], single, atol=1e-10)
            assert np.allclose(batched[b, g.n_free :], 0)  # padding slots predict nothing


def test_ripple_is_the_minimiser_of_the_dirichlet_energy():
    rng = np.random.default_rng(1)
    g = random_graph(rng, n_pinned=2, n_free=5)
    L, lam = laplacian_np(g), 0.05
    p = g.n_pinned * D
    x_free = ripple(torch.as_tensor(L), g.n_pinned, torch.as_tensor(g.s), lam).numpy().reshape(-1)
    s = g.s.reshape(-1)

    def energy(xf):
        x = np.concatenate([s, xf])
        return 0.5 * x @ L @ x + 0.5 * lam * xf @ xf

    best = minimize(energy, np.zeros_like(x_free), method="BFGS", options={"gtol": 1e-12})
    assert np.allclose(best.x, x_free, atol=1e-6)
    assert L.shape[0] == g.n_nodes * D and p == 4


def test_tension_equals_the_minimal_energy_and_is_zero_for_consistent_pins():
    rng = np.random.default_rng(2)
    g = random_graph(rng, n_pinned=3, n_free=5)
    L, lam = laplacian_np(g), 0.05
    p = g.n_pinned * D
    x_free = ripple(torch.as_tensor(L), g.n_pinned, torch.as_tensor(g.s), lam).numpy().reshape(-1)
    x = np.concatenate([g.s.reshape(-1), x_free])
    minimal = 0.5 * x @ L @ x + 0.5 * lam * x_free @ x_free
    assert tension(L, g.s, lam) == pytest.approx(minimal, rel=1e-9)
    S = tension_matrix(L, g.n_pinned, D, lam)
    assert np.allclose(S, S.T) and np.linalg.eigvalsh(S).min() > -1e-10
    assert p == S.shape[0]
    zero = np.zeros_like(g.s)
    assert tension(L, zero, lam) == 0.0


def test_r_on_a_hand_built_system_equals_the_analytic_value():
    a, b = 3.0, 1.2
    system = np.array([[a, -b], [-b, a]])  # d = 1: Delta = a I, W = b sigma_x, r = b / a
    stats = r_value(system, d=1)
    assert stats.r == pytest.approx(b / a) and stats.positive_definite
    blocks = np.kron(system, np.eye(3))  # d = 3 with identical blocks gives the same r
    assert r_value(blocks, d=3).r == pytest.approx(b / a)
    assert r_bound(0.5) == pytest.approx(1.0) and math.isinf(r_bound(1.0))


# --- the three methods -----------------------------------------------------------------------


def test_sheaf_with_identity_maps_equals_the_weighted_graph(cfg):
    rng = np.random.default_rng(3)
    g = random_graph(rng, n_groups=3)
    sheaf = SheafModel(2, 3, D, [True, True])  # Delta = 0: trivial sheaf
    graph_model = GraphModel(2, D)
    with torch.no_grad():
        graph_model.raw_beta.fill_(math.log(math.e - 1.0))  # softplus = 1
    a = sheaf.predict(g, 0.05).detach().numpy()
    b = graph_model.predict(g, 0.05).detach().numpy()
    assert np.allclose(a, b, atol=1e-12)


def test_sheaf_recovers_the_laplacian_blocks_of_a_planted_sheaf(cfg):
    """Noise-free labels from a planted sheaf: the fitted maps reproduce its Laplacian blocks (<5%).

    Restriction maps are identified only up to an orthogonal change of the edge space, so the
    comparison is on the invariants F'F that make up the Laplacian."""
    rng = np.random.default_rng(4)
    symmetric = [True, False]
    truth = SheafModel(2, 1, D, symmetric)
    with torch.no_grad():
        truth.delta_u.normal_(0, 0.35)
        truth.delta_v.normal_(0, 0.35)
    graphs = []
    for i in range(150):
        g = random_graph(rng, n_pinned=2, n_free=5, n_groups=1, event_id=f"e{i}")
        graphs.append(g.with_labels(truth.predict(g, 0.05).detach().numpy()))
    fast = cfg.with_overrides({"road1.optimiser": {"lr": 0.03, "max_steps": 4000, "patience": 4000,
                                                   "batch_events": 150, "eval_every": 50}})
    fitted = train_model(SheafModel(2, 1, D, symmetric), graphs, None, grounding=0.05, penalty_mu=0.0,
                         cfg=fast, seed_parts=("recover",), fixed_steps=4000).model
    for t in range(2):
        Fu_t = torch.eye(D, dtype=DTYPE) + truth.delta_u[t, 0]
        Fv_t = Fu_t if symmetric[t] else torch.eye(D, dtype=DTYPE) + truth.delta_v[t, 0]
        Fu_f = torch.eye(D, dtype=DTYPE) + fitted.delta_u[t, 0]
        Fv_f = Fu_f if symmetric[t] else torch.eye(D, dtype=DTYPE) + fitted.delta_v[t, 0]
        for true_block, fit_block in ((Fu_t.T @ Fu_t, Fu_f.T @ Fu_f), (Fv_t.T @ Fv_t, Fv_f.T @ Fv_f),
                                      (Fu_t.T @ Fv_t, Fu_f.T @ Fv_f)):
            rel = (true_block - fit_block).norm() / true_block.norm()
            assert float(rel.detach()) < 0.05


def test_weighted_graph_recovers_planted_type_weights(cfg):
    rng = np.random.default_rng(5)
    truth = GraphModel(2, D)
    with torch.no_grad():
        truth.raw_beta.copy_(torch.tensor([1.5, -1.0], dtype=DTYPE))
    graphs = []
    for i in range(120):
        g = random_graph(rng, event_id=f"e{i}")
        graphs.append(g.with_labels(truth.predict(g, 0.05).detach().numpy()))
    fast = cfg.with_overrides({"road1.optimiser": {"lr": 0.05, "max_steps": 1500, "patience": 1500,
                                                   "batch_events": 120, "eval_every": 50}})
    fitted = train_model(GraphModel(2, D), graphs, None, grounding=0.05, penalty_mu=0.0, cfg=fast,
                         seed_parts=("graph",), fixed_steps=1500).model
    assert np.allclose(fitted.beta.detach().numpy(), truth.beta.detach().numpy(), rtol=0.05)


def test_matrix_ladder_fits_better_with_more_freedom_and_rung2_recovers_a_planted_table():
    rng = np.random.default_rng(6)
    n_types = 2
    B = rng.normal(size=(n_types, D, D))  # planted rung-2 table
    graphs = []
    for i in range(200):
        g = random_graph(rng, n_pinned=2, n_free=5, event_id=f"e{i}")
        pred = np.zeros((g.n_free, D))
        for a, b, t, w in zip(g.src, g.dst, g.link_type, g.weight):
            for pin, free in ((a, b), (b, a)):
                if pin < g.n_pinned <= free:
                    pred[free - g.n_pinned] += w * B[t] @ g.s[pin]
        graphs.append(g.with_labels(pred))
    sse = {}
    for rung in (1, 2, 3):
        model = fit_matrix(graphs, rung, ridge=1e-8, n_types=n_types)
        sse[rung] = sum(((model.predict(g) - g.y) ** 2).sum() for g in graphs)
    assert sse[2] < 1e-9  # planted table is exactly representable
    assert sse[1] > 1e-3 and sse[2] <= sse[1] and sse[3] <= sse[2] + 1e-9
    assert fit_matrix(graphs, 2, 1e-8, n_types).n_parameters() == D * n_types * D


def test_one_hop_table_ignores_pinned_to_pinned_and_second_hop_edges():
    g = EventGraph("e", ("p", "f1", "f2"), 1, np.array([0, 1]), np.array([1, 2]), np.array([0, 0]),
                   np.array([1.0, 1.0]), np.array([[1.0, 2.0]]), np.ones(1))
    design = matrix_design(g, 2, n_types=1)
    assert np.allclose(design[0], [1.0, 2.0]) and np.allclose(design[1], 0)  # f2 is two hops away


# --- sheaf test (D10) ----------------------------------------------------------------------


def simulate_d10(rng, n_clusters, per_cluster, signal, noise=0.5):
    cluster = np.repeat(np.arange(n_clusters), per_cluster)
    table = rng.normal(size=len(cluster))
    extra = rng.normal(size=len(cluster))
    y = 0.2 + 1.0 * table + signal * extra + rng.normal(0, noise, len(cluster))
    return y, table, table + extra, cluster


def test_d10_detects_a_real_sheaf_effect_and_is_flat_when_there_is_none():
    rng = np.random.default_rng(7)
    y, table, sheaf, cluster = simulate_d10(rng, 60, 8, signal=0.4)
    real = d10_regression(y, table, sheaf, cluster, rng, draws=1000, level=0.95)
    assert real.ci_low > 0 and real.beta2 == pytest.approx(0.4, abs=0.15) and real.n_clusters == 60
    rejections = 0
    for _ in range(80):
        y, table, sheaf, cluster = simulate_d10(rng, 40, 6, signal=0.0)
        null = d10_regression(y, table, sheaf, cluster, rng, draws=400, level=0.95)
        rejections += int(null.ci_low > 0 or null.ci_high < 0)
    assert rejections / 80 < 0.15  # near the nominal 5%


def test_oos_r2_is_one_for_perfect_forecasts_and_zero_for_the_training_mean():
    y = np.random.default_rng(8).normal(size=(50, 3))
    assert np.allclose(oos_r2(y, y, y.mean(axis=0)), 1.0)
    assert np.allclose(oos_r2(y, np.tile(y.mean(axis=0), (50, 1)), y.mean(axis=0)), 0.0)


# --- subgraph selection --------------------------------------------------------------------------


def links_frame(rows):
    return pd.DataFrame(rows, columns=["src_node", "dst_node", "link_type", "weight", "available_from", "valid_to"])


def test_subgraph_takes_top_neighbours_per_type_shared_nodes_and_respects_availability(cfg):
    small = cfg.with_overrides({"road1.neighbours_per_type": 2})
    asof = pd.Timestamp("2021-06-01")
    rows = [
        ("P", "a", "industry", 0.9, "2020-01-01", None),
        ("P", "b", "industry", 0.8, "2020-01-01", None),
        ("P", "c", "industry", 0.1, "2020-01-01", None),  # beyond top-2 for this type
        ("Q", "c", "owner", 0.5, "2020-01-01", None),  # c touches two pinned nodes -> kept
        ("P", "d", "industry", 0.99, "2022-01-01", None),  # not yet available
        ("P", "e", "industry", 0.99, "2019-01-01", "2020-06-01"),  # expired
        ("a", "b", "industry", 0.7, "2020-01-01", None),  # neighbour-neighbour edge is kept
        ("P", "Q", "owner", 0.4, "2020-01-01", None),  # pinned-pinned edge is kept
    ]
    g = build_event_graph(small, "e1", ["P", "Q"], np.zeros((2, 3)), links_frame(rows), asof, np.ones(6) / 6)
    assert set(g.nodes[2:]) == {"a", "b", "c"} and g.nodes[:2] == ("P", "Q")
    assert not g.truncated and g.n_free == 3
    index = {n: i for i, n in enumerate(g.nodes)}
    pairs = {(g.nodes[s], g.nodes[d]) for s, d in zip(g.src, g.dst)}
    assert ("a", "b") in pairs and ("P", "Q") in pairs and ("P", "d") not in pairs
    assert index["P"] == 0


def test_subgraph_is_capped_and_flags_truncation(cfg):
    capped = cfg.with_overrides({"road1.max_nodes": 5, "road1.neighbours_per_type": 50})
    rows = [("P", f"n{i:02d}", "industry", 1.0 - i / 100, "2020-01-01", None) for i in range(20)]
    g = build_event_graph(capped, "e", ["P"], np.zeros((1, 3)), links_frame(rows), pd.Timestamp("2021-01-01"), np.ones(6) / 6)
    assert g.truncated and g.n_nodes == 5 and g.nodes[1] == "n00"  # strongest neighbours survive


def test_group_probabilities_sum_causes_into_the_six_groups(cfg):
    causes = {c: 0.0 for c in cfg.get("causes.causes")}
    causes["earnings_beat"], causes["bankruptcy"], causes["other"] = 0.5, 0.3, 0.2
    probs = group_probabilities(causes, cfg)
    assert probs.shape == (6,) and probs.sum() == pytest.approx(1.0)
    assert probs[0] == pytest.approx(0.5) and probs[2] == pytest.approx(0.3) and probs[5] == pytest.approx(0.2)


# --- cross-validation and the fitting driver ---------------------------------------------------


def test_time_blocked_folds_never_overlap_and_respect_the_embargo():
    cal = TradingCalendar("XNYS", "2018-01-01", "2020-12-31")
    dates = pd.Series(cal.sessions[(cal.sessions >= "2019-01-02") & (cal.sessions <= "2019-12-31")])
    folds = time_blocked_folds(dates, 4, 5, cal)
    all_val = np.concatenate([v for _, v in folds])
    assert sorted(all_val) == list(range(len(dates)))  # validation blocks tile the sample
    for train, val in folds:
        assert not set(train) & set(val)
        gap_before = [d for d in dates.iloc[train] if d < dates.iloc[val].min()]
        gap_after = [d for d in dates.iloc[train] if d > dates.iloc[val].max()]
        if gap_before:
            assert cal.count_between(max(gap_before), dates.iloc[val].min()) >= 5
        if gap_after:
            assert cal.count_between(dates.iloc[val].max(), min(gap_after)) >= 5


def test_select_and_fit_counts_every_configuration_and_beats_the_mean_forecast(cfg):
    rng = np.random.default_rng(9)
    cal = TradingCalendar("XNYS", "2018-01-01", "2020-12-31")
    truth = GraphModel(len(cfg.get("road1.link_types")), 3)
    with torch.no_grad():
        truth.raw_beta.fill_(1.0)
    graphs, dates = [], []
    sessions = cal.sessions[(cal.sessions >= "2019-01-02") & (cal.sessions <= "2019-12-31")]
    for i in range(80):
        n_types = len(cfg.get("road1.link_types"))
        g = random_graph(rng, n_types=n_types, n_groups=6, event_id=f"e{i}")
        g = EventGraph(g.event_id, g.nodes, g.n_pinned, g.src, g.dst, g.link_type, g.weight,
                       np.repeat(g.s[:, :1], 3, axis=1) * np.array([1.0, 0.8, 0.5]), g.group_probs)
        graphs.append(g.with_labels(truth.predict(g, 0.05).detach().numpy() * 1.0))
        dates.append(sessions[i * 3])
    scale = dial_scale(graphs, float(cfg.get("road1.dial_scale_floor")))
    scaled = standardise(graphs, scale)
    quick = cfg.with_overrides({"road1.optimiser": {"lr": 0.05, "max_steps": 60, "patience": 60,
                                                    "batch_events": 16, "eval_every": 10}})
    seen = []
    model, best, results = select_and_fit("wgraph", scaled, dates, cal, quick, seed_parts=("t",),
                                          on_config=seen.append)
    assert len(results) == len(seen) == len(quick.get("road1.grounding_lambda_grid"))
    sheaf_seen = []
    select_and_fit("sheaf", scaled[:30], dates[:30], cal, quick, seed_parts=("t",), on_config=sheaf_seen.append)
    assert len(sheaf_seen) == 3 * 3  # grounding grid x penalty grid
    pred = np.concatenate([predict_graph(model, g, best.grounding, scale, quick) for g in graphs])
    y = np.concatenate([g.y for g in graphs])
    assert (oos_r2(y, pred, y.mean(axis=0)) > 0).all()
    assert new_model("sheaf", cfg).n_parameters() > new_model("wgraph", cfg).raw_beta.numel()


def test_unlabelled_free_nodes_are_masked_not_fatal(cfg):
    rng = np.random.default_rng(15)
    truth = GraphModel(2, D)
    graphs = []
    for i in range(30):
        g = random_graph(rng, event_id=f"e{i}")
        y = truth.predict(g, 0.05).detach().numpy().copy()
        y[0] = np.nan  # a context node without prices has no label
        graphs.append(g.with_labels(y))
    batch = collate(graphs)
    assert int(batch.mask.sum()) == sum(g.n_free - 1 for g in graphs)
    from finflock.road1.fit import mean_loss
    assert torch.isfinite(mean_loss(GraphModel(2, D), graphs, 0.05, 0.0))
    assert np.isfinite(fit_matrix(graphs, 2, 1.0, 2).theta).all()
