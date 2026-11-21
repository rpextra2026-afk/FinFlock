"""The fit-and-score core of the road-1 run, the oracle inputs and the reader's own-firm reference, on small synthetic graphs."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finflock.common.resources import ResourceLog
from finflock.common.timing import TradingCalendar
from finflock.framework.events import EventReading
from finflock.road1.rreport import r_summary, r_table
from finflock.stage import road1_real as rr
from finflock.stage.road1_run import NotEnoughGraphs, fit_and_score
from tests.test_road1_memory import D, GROUPS, graph

SMALL = {"road1.optimiser": {"lr": 0.05, "max_steps": 12, "patience": 5, "batch_events": 8, "eval_every": 3},
         "road1.grounding_lambda_grid": [0.05], "road1.sheaf_penalty_grid": [1.0], "road1.ridge_grid": [1.0], "road1.bootstrap.draws": 60,
         "road1.cv": {"folds": 2, "embargo_sessions": 5}, "cpu.road1_eval.sheaf_test.underpowered_below_events": 5,
         "cpu.road1_eval.sheaf_test.min_clusters": 3, "cpu.road1_eval.r_report.max_events": 12, "cpu.road1_eval.matrix_rungs": [1, 2, 3]}


@pytest.fixture
def setup(cfg):
    small = cfg.with_overrides(SMALL)
    calendar = TradingCalendar.from_config(small)
    sessions = calendar.sessions
    rng = np.random.default_rng(1)
    dev_days = sessions[(sessions >= "2014-01-02") & (sessions <= "2018-06-29")]
    test_days = sessions[(sessions >= "2019-02-01") & (sessions <= "2020-06-30")]
    graphs, day0 = [], {}
    for i in range(60):
        n_pinned = 1 + int(i % 3 == 0)
        g = graph(i, n_pinned, int(rng.integers(3, 14)), rng)
        day0[g.event_id] = pd.Timestamp((dev_days if i < 36 else test_days)[(i * 7) % 30 + (0 if i < 36 else 0)])
        graphs.append(g)
    dev, test = graphs[:36], graphs[36:]
    types = {g.event_id: ("company",) * g.n_pinned for g in graphs}
    return small, calendar, dev, test, day0, types


def test_fit_and_score_returns_every_table_and_the_declared_tests(setup):
    cfg, calendar, dev, test, day0, types = setup
    log = ResourceLog()
    out = fit_and_score(dev, test, day0, types, calendar, cfg, ("t",), 1, log.stage)
    expected = {"fits", "configs_evaluated", "road1_scores", "road1_scores_by_stratum", "strata_counts", "rung_selection", "d10_sheaf_by_stratum", "r_events", "r_summary"}
    assert expected <= set(out.tables)
    fits = out.tables["fits"].set_index("method")
    assert {"wgraph", "sheaf", "wmatrix_r1", "wmatrix_r2", "wmatrix_r3"} <= set(fits.index)
    assert fits.loc["sheaf", "parameters"] == 756  # 3 symmetric x 54 + ... as in the plan: 4 symmetric + 5 directed link types
    d10 = out.tables["d10_sheaf_by_stratum"]
    assert {"single_pin", "multi_pin", "all"} <= set(d10["stratum"]) and d10["tested"].astype(bool).any()
    assert set(d10.loc[d10["role"] == "primary_table", "table"]) == {"wmatrix_r3"}
    assert {"stage": "fit sheaf"}.items() <= {k: v for k, v in log.stages[1].items() if k == "stage"}.items()
    rs = out.tables["r_summary"]
    assert {"wgraph", "sheaf"} == set(rs["model"]) and (rs["median_r"] >= 0).all() and rs["n_events"].max() <= 12


def test_fewer_labelled_graphs_than_needed_stops_cleanly(setup):
    cfg, calendar, dev, test, day0, types = setup
    with pytest.raises(NotEnoughGraphs):
        fit_and_score([], test, day0, types, calendar, cfg, ("t",), 1, ResourceLog().stage)


def test_r_is_zero_without_edges_and_grows_with_coupling(setup):
    cfg, _, dev, _, _, _ = setup
    from finflock.road1.arms import GraphModel

    model = GraphModel(9, D)
    isolated = graph(900, 1, 4, np.random.default_rng(3))
    isolated = type(isolated)(isolated.event_id, isolated.nodes, isolated.n_pinned, isolated.src[:0], isolated.dst[:0], isolated.link_type[:0], isolated.weight[:0],
                              isolated.s, isolated.group_probs, False, isolated.y)
    table = r_table({"wgraph": (model, 0.05)}, [isolated], {isolated.event_id: "single_pin"}, cfg)
    assert table["r"].iloc[0] == pytest.approx(0.0, abs=1e-12)
    base = dev[0]
    free = list(range(base.n_pinned, base.n_nodes))  # chain the free nodes together: now they interact, so r is positive
    chain_src, chain_dst = np.array(free[:-1]), np.array(free[1:])
    chained = type(base)(base.event_id, base.nodes, base.n_pinned, np.concatenate([base.src, chain_src]), np.concatenate([base.dst, chain_dst]),
                         np.concatenate([base.link_type, np.zeros(len(chain_src), dtype=int)]), np.concatenate([base.weight, np.full(len(chain_src), 0.9)]),
                         base.s, base.group_probs, False, base.y)
    coupled = r_table({"wgraph": (model, 0.05)}, [chained, dev[1]], {}, cfg)
    assert coupled.set_index("event_id").loc[chained.event_id, "r"] > 0.0 and r_summary(coupled, cfg)["n_events"].max() == 2


def make_readings(n_pins):
    pins = tuple({"node_id": f"S{i}", "direction": "up", "value_effect_pct": 2.0, "magnitude": 1, "role": "source"} for i in range(n_pins))
    return EventReading("e1", pins, {"other": 1.0}, np.ones(GROUPS) / GROUPS, 0.5)


def test_oracle_inputs_use_realised_values_and_drop_unlabelled_pins(cfg):
    labels = pd.DataFrame({"event_id": ["e1", "e1"], "security_id": ["S0", "S1"], "window": ["d01", "d01"], "abn_ret": [0.5, np.nan], "log_vol_ratio": [0.1, 0.2],
                           "log_dv_ratio": [0.3, 0.4]})
    readings, dials = rr.oracle_inputs({"e1": make_readings(3)}, labels, cfg)
    assert [p["node_id"] for p in readings["e1"].pins] == ["S0"]  # S1 has a NaN return and S2 no label
    assert dials["e1"].iloc[0].tolist() == [0.5, 0.1, 0.3] and list(dials["e1"].columns) == ["ret", "vol", "volu"]
    assert rr.oracle_inputs({"e1": make_readings(1)}, labels.iloc[0:0], cfg) == ({}, {})


def test_pinned_reference_scores_the_readers_own_firm_values_on_test_only(cfg):
    small = cfg.with_overrides({"road1.bootstrap.draws": 50})
    rng = np.random.default_rng(0)
    readings, dials, rows, split, day0 = {}, {}, [], {}, {}
    for i in range(40):
        e = f"e{i}"
        readings[e] = EventReading(e, ({"node_id": "S0", "direction": "up", "value_effect_pct": 1.0, "magnitude": 1, "role": "source"},), {"other": 1.0},
                                   np.ones(GROUPS) / GROUPS, 0.5)
        y = rng.normal(size=3)
        dials[e] = pd.DataFrame([y + 0.1 * rng.normal(size=3)], columns=["ret", "vol", "volu"])  # a good own-firm prediction
        rows.append({"event_id": e, "security_id": "S0", "window": "d01", "abn_ret": y[0], "log_vol_ratio": y[1], "log_dv_ratio": y[2]})
        split[e], day0[e] = ("dev" if i < 20 else "test"), pd.Timestamp("2019-01-10") + pd.Timedelta(days=i)
    table = rr.pinned_reference_table(readings, dials, pd.DataFrame(rows), split, day0, small, ("t",))
    assert set(table["dial"]) == {"ret", "vol", "volu"} and (table["r2"] > 0.8).all() and (table["n"] == 20).all()


def test_without_the_sheaf_only_the_graph_and_the_matrices_are_fitted_and_scored(setup):
    cfg, calendar, dev, test, day0, types = setup
    log = ResourceLog()
    out = fit_and_score(dev, test, day0, types, calendar, cfg, ("t",), 1, log.stage, with_sheaf=False)
    assert set(out.tables["fits"]["method"]) == {"wgraph", "wmatrix_r1", "wmatrix_r2", "wmatrix_r3"}
    assert "d10_sheaf_by_stratum" not in out.tables
    assert out.fitted.sheaf is None and "sheaf" not in out.scores.columns
    assert set(out.tables["road1_scores"]["method"]) >= {"wgraph", "wmatrix_r1", "wmatrix_r2", "wmatrix_r3", "no_ripple_zero"}
    assert set(out.tables["r_summary"]["model"]) == {"wgraph"}
    assert "fit sheaf" not in {s["stage"] for s in log.stages}


def test_graph_and_matrix_scores_do_not_depend_on_whether_the_sheaf_is_fitted(setup):
    cfg, calendar, dev, test, day0, types = setup
    with_sheaf = fit_and_score(dev, test, day0, types, calendar, cfg, ("t",), 1, ResourceLog().stage, with_r_report=False)
    without = fit_and_score(dev, test, day0, types, calendar, cfg, ("t",), 1, ResourceLog().stage, with_r_report=False, with_sheaf=False)
    for method in ("wgraph", "wmatrix_r1", "wmatrix_r2", "wmatrix_r3"):
        a = with_sheaf.scores[method].to_numpy()
        b = without.scores[method].to_numpy()
        assert np.allclose(a, b, atol=1e-6), method
