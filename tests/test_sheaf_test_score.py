"""The road-1 arm's out-of-sample score (finfm.real.framework_fit.score_graphs): same table as the graph and matrices get in run_road1.py."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch

from finfm.common.timing import TradingCalendar
from finfm.real.framework_fit import score_graphs
from finfm.road1.arms import GraphModel, MatrixModel
from finfm.road1.fit import dial_scale, select_and_fit, standardise
from finfm.road1.graph import EventGraph
from tests.test_road1 import random_graph


def planted(cfg, n: int, seed: int):
    rng = np.random.default_rng(seed)
    n_types = len(cfg.get("road1.link_types"))
    truth = GraphModel(n_types, 3)
    with torch.no_grad():
        truth.raw_beta.fill_(1.0)
    cal = TradingCalendar("XNYS", "2018-01-01", "2020-12-31")
    sessions = cal.sessions[(cal.sessions >= "2019-01-02") & (cal.sessions <= "2019-12-31")]
    graphs, dates = [], {}
    for i in range(n):
        g = random_graph(rng, n_types=n_types, n_groups=6, event_id=f"e{seed}-{i}")
        g = EventGraph(g.event_id, g.nodes, g.n_pinned, g.src, g.dst, g.link_type, g.weight, np.repeat(g.s[:, :1], 3, axis=1) * np.array([1.0, 0.8, 0.5]), g.group_probs)
        graphs.append(g.with_labels(truth.predict(g, 0.05).detach().numpy()))
        dates[g.event_id] = sessions[(i * 3) % len(sessions)]
    return graphs, dates, cal


def test_score_table_has_one_row_per_dial_for_the_arm_and_the_baseline_and_the_fit_beats_the_baseline(cfg):
    dev, dev_dates, cal = planted(cfg, 80, 9)
    test, test_dates, _ = planted(cfg, 30, 10)
    scale = dial_scale(dev, float(cfg.get("road1.dial_scale_floor")))
    quick = cfg.with_overrides({"road1.optimiser": {"lr": 0.05, "max_steps": 60, "patience": 60, "batch_events": 16, "eval_every": 10}})
    model, best, _ = select_and_fit("wgraph", standardise(dev, scale), [dev_dates[g.event_id] for g in dev], cal, quick, seed_parts=("t",))
    predictor = SimpleNamespace(arm="wgraph", model=model, grounding=best.grounding, scale=scale)
    table = score_graphs(predictor, dev, test, test_dates, quick, ("t", "score"))
    dials = list(cfg.get("road1.dial_names"))
    assert sorted(table["dial"].unique()) == sorted(dials)
    assert set(table["method"]) == {"wgraph", "no_ripple_zero"}
    assert len(table) == 2 * len(dials)
    assert {"r2", "ci_low", "ci_high", "rmse", "n", "n_dates"} <= set(table.columns)
    by = table.pivot(index="dial", columns="method", values="r2")
    assert (by["wgraph"] > by["no_ripple_zero"]).all() and (by["wgraph"] > 0).all()
    assert (table["ci_low"] <= table["r2"]).all() and (table["r2"] <= table["ci_high"]).all()


def test_a_matrix_arm_is_refused_and_a_test_side_without_labels_is_an_error(cfg):
    dev, _, _ = planted(cfg, 10, 1)
    matrix_arm = SimpleNamespace(arm="wmatrix_r1", model=MatrixModel.__new__(MatrixModel), grounding=0.1, scale=np.ones(3))
    with pytest.raises(ValueError, match="matrix arm"):
        score_graphs(matrix_arm, dev, dev, {g.event_id: pd.Timestamp("2019-02-01") for g in dev}, cfg, ("t",))
    unlabelled = [EventGraph(g.event_id, g.nodes, g.n_pinned, g.src, g.dst, g.link_type, g.weight, g.s, g.group_probs) for g in dev]
    predictor = SimpleNamespace(arm="wgraph", model=GraphModel(len(cfg.get("road1.link_types")), 3), grounding=0.1, scale=np.ones(3))
    with pytest.raises(ValueError, match="no labelled free node"):
        score_graphs(predictor, dev, unlabelled, {g.event_id: pd.Timestamp("2019-02-01") for g in dev}, cfg, ("t",))
