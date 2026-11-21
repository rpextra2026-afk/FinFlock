"""Fund types from persona skeletons, S&P shares from the manager table, and node classes by cap tercile."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finflock.personas.skeletons import Skeleton
from finflock.portfolio.market import Market
from finflock.real.framework_fit import panels_of
from finflock.real.game_inputs import fund_inputs, panels_from_skeletons, sp_cover_table
from finflock.real.market_load import RealMarket
from finflock.stage import road1_real as rr

QUARTER = "2021Q2"
NAMES = [f"S{i}" for i in range(8)]


def toy(seed=0, n=120):
    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2022-01-03", periods=n)
    ret = pd.DataFrame(rng.normal(0, 0.01, (n, len(NAMES))), index=index, columns=NAMES)
    close = 50 * (1 + ret).cumprod()
    volume = pd.DataFrame(1e6, index=index, columns=NAMES)
    oo = close.pct_change().shift(-1)
    snaps = pd.DataFrame({"snapshot_date": index[0], "security_id": NAMES, "source": "t"})
    return RealMarket(close, close, close, close, volume, ret, oo, snaps, Market(close, volume, oo, close.apply(lambda c: c.last_valid_index())))


def skeleton(pid, kind, holdings, turnover):
    return Skeleton(pid, kind, QUARTER, "large", 1.0e9, turnover, "slow", float(len(holdings)), {}, holdings, None, "high", "general",
                    0.5, 0.0, "slow", "New York metro", "13f")


def test_sp_cover_is_book_weighted_over_active_filers_of_a_type(cfg):
    managers = pd.DataFrame({"period": [pd.Timestamp("2021-06-30")] * 4, "type": ["hedge_fund"] * 3 + ["mutual_pension"],
                             "sp_cover": [0.2, 0.6, 0.9, 0.5], "book_usd": [1.0, 3.0, 1.0, 5.0],
                             "passive_flag": [False, False, True, False]})
    cover = sp_cover_table(managers, cfg)
    assert cover[(QUARTER, "hedge_fund")] == pytest.approx((0.2 * 1 + 0.6 * 3) / 4) and cover[(QUARTER, "mutual_pension")] == pytest.approx(0.5)


def test_fund_inputs_scale_holdings_to_the_cover_and_skip_types_without_cover():
    panel = [skeleton("hedge_fund_00@" + QUARTER, "hedge_fund", {"S0": 0.5, "S1": 0.25, "ZZ": 0.25}, 0.1),
             skeleton("hedge_fund_01@" + QUARTER, "hedge_fund", {"S2": 1.0}, 0.9),
             skeleton("mutual_pension_00@" + QUARTER, "mutual_pension", {"S0": 1.0}, 0.1),
             skeleton("retail_00@" + QUARTER, "retail", {"S0": 1.0}, 1.0)]
    funds = fund_inputs(panel, NAMES, {(QUARTER, "hedge_fund"): 0.4})
    assert [f.name for f in funds] == ["hedge_fund_00@" + QUARTER, "hedge_fund_01@" + QUARTER]  # no cover for pensions, retail is never a fund
    first = funds[0]
    assert first.weights.sum() == pytest.approx(0.4) and first.weights[NAMES.index("S0")] == pytest.approx(0.4 * 0.5 / 0.75)
    assert [f.turnover_pctile for f in funds] == [0.5, 1.0]
    assert set(panels_from_skeletons({s.persona_id: s for s in panel})) == {QUARTER}


def test_node_classes_come_from_the_road1_evaluation_on_panels_built_from_the_framework_market(cfg):
    rm = toy(n=130)
    rm.volume.iloc[:, :] = np.arange(1, len(NAMES) + 1)[None, :] * 1e6  # S0 smallest ... S7 largest
    small_window = cfg.with_overrides({"personas.panel.cap_proxy_window_days": 20})
    panels = panels_of(rm, pd.DataFrame(0.0, index=rm.sessions, columns=["mkt_rf", "rf"]), small_window)
    assert set(panels.prices) == set(NAMES) and panels.cap_proxy.shape == rm.close.shape
    day = rm.sessions[100]

    class Reading:
        pins = ({"node_id": "S0"}, {"node_id": "S7"}, {"node_id": "COMPANY_X"})

    classes = rr.node_classes({"e": Reading()}, {"e": day}, rm.snapshots, panels, small_window)
    assert classes[("e", "S0")] == "small" and classes[("e", "S7")] == "large" and classes[("e", "COMPANY_X")] == "borrowed"
