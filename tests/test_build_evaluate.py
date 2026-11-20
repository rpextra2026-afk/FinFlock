"""The evaluation layer on a synthetic scored table with a known signal (final-build spec, sections 6 and 7)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oryxis.build import evaluate as E
from oryxis.build import metrics as M
from oryxis.common.config import load_config

from tests.conftest import CONFIG_DIR


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_DIR).with_overrides({"build.metrics.placebo_shuffles": 8, "build.metrics.bootstrap_draws": 60, "build.metrics.slice_bootstrap_draws": 30, "build.metrics.min_rows_per_slice": 20})


def synthetic(signal=True, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2022-01-03", periods=40)
    nodes = [f"N{i}" for i in range(30)]
    rows, panel = [], []
    eid = 0
    for d in days:
        truth = {n: rng.normal() for n in nodes}
        for dial in E.DIALS:
            for n in nodes:
                panel.append({"day": d, "node": n, "dial": dial, "y": truth[n] + 0.5 * rng.normal(), "pred_sheaf": 0.0, "pred_wgraph": 0.0, "linked_today": False})
        for _ in range(2):
            eid += 1
            chosen = rng.choice(nodes, size=6, replace=False)
            for n in chosen:
                for dial in E.DIALS:
                    y = truth[n] + 0.5 * rng.normal()
                    rows.append({"event_id": f"e{eid}", "day": d, "node": n, "dial": dial, "y": y, "pred_sheaf": (truth[n] if signal else rng.normal()), "pred_wgraph": 0.0, "has_industry": True, "has_corr": False, "has_owner": False, "w_industry": 0.5, "w_corr": 0.0, "w_owner": 0.0,
                                 "n_pinned_neighbours": 1, "n_pinned": 2, "cause_group": "no_cause", "kind": "finance", "family": "f", "session_type": "intraday", "repeat_count": 1})
    return pd.DataFrame(rows), pd.DataFrame(panel), days


def run(cfg, signal=True):
    rows, panel, days = synthetic(signal)
    keys = ["day", "node", "dial"]
    sums = rows.groupby(keys)[["pred_sheaf", "pred_wgraph"]].sum()
    panel = panel.drop(columns=["pred_sheaf", "pred_wgraph"]).merge(sums, on=keys, how="left").fillna({"pred_sheaf": 0.0, "pred_wgraph": 0.0})
    link = rows.groupby(["day", "node"]).size().rename("n").reset_index()
    panel = panel.merge(link, on=["day", "node"], how="left"); panel["linked_today"] = panel["n"].notna(); panel = panel.drop(columns="n")
    col = M.Collector("tapeX", "holdout", "d01")
    methods = ["sheaf", "wgraph"]
    E.evaluate_tape(col, rows, panel, methods, "sheaf", {d: 0.0 for d in E.DIALS}, {d: 1.0 for d in E.DIALS}, {m: {d: 1.0 for d in E.DIALS} for m in methods}, {"weight_edges": np.array([0.25, 0.5, 0.75])}, cfg, np.random.default_rng(0), list(days))
    return col.frame()


def test_a_real_signal_shows_a_gain_above_every_placebo(cfg):
    f = run(cfg, signal=True)
    real = f[(f.metric == "gain") & (f.slice == "placebo") & (f.slice_value == "real") & (f.dial == "ret")].value.iloc[0]
    assert real > 0.3
    for name in ("shuffle_days", "shuffle_events_nodes", "shuffle_links", "random_node_same_day"):
        q = f[(f.slice == "placebo") & (f.slice_value == name) & (f.metric == "gain_quantile") & (f.dial == "ret")].value.iloc[0]
        assert q < real
        share = f[(f.slice == "placebo") & (f.slice_value == name) & (f.metric == "share_matching_or_beating_real") & (f.dial == "ret")].value.iloc[0]
        assert share == 0.0


def test_a_random_signal_has_no_gain_and_the_table_has_every_view_method_and_slice(cfg):
    f = run(cfg, signal=False)
    r2 = f[(f.metric == "r2_os") & (f.slice == "view") & (f.view == "linked") & (f.method == "sheaf") & (f.dial == "ret")].value.iloc[0]
    assert r2 < 0
    assert set(f[f.slice == "view"].view) == {"linked", "all", "unlinked"} and set(f.method) >= {"sheaf", "wgraph"}
    assert {"weekday", "link_type", "cause_group", "kind", "session_type"} <= set(f.slice)
    assert not f.duplicated(["dial", "method", "view", "slice", "slice_value", "metric"]).any()


def test_multiple_comparison_rows_are_written(cfg):
    f = run(cfg, signal=True)
    mc = f[f.slice == "multiple_comparison"]
    assert {"p_value", "p_adjusted"} <= set(mc.metric) and (mc.value.dropna() <= 1.0).all()
