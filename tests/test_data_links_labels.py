"""Link builders (point-in-time) and labels (abnormal returns, vol and volume ratios)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from oryxis.data.labels import (
    build_labels,
    event_window_mask,
    fit_residual_model,
    garman_klass,
    label_one,
)
from oryxis.data.links import (
    correlation_links,
    industry_links,
    owner_links,
    rival_links,
    stack_snapshots,
    table_links,
)

SNAP = pd.Timestamp("2021-06-01")


# --- links -------------------------------------------------------------------------------------


def test_industry_links_stay_inside_the_sic_group_and_pick_the_closest_in_size(cfg):
    small = cfg.with_overrides({"links.industry.neighbours_per_node": 1})
    sic = pd.Series({"a": "3571", "b": "3572", "c": "3579", "d": "2834", "e": "2836"})
    mcap = pd.Series({"a": 1e9, "b": 1.1e9, "c": 9e9, "d": 5e8, "e": 6e8})
    links = industry_links(sic, mcap, SNAP, {"a", "b"}, small)
    pairs = {(r.src_node, r.dst_node) for r in links.itertuples()}
    assert ("a", "b") in pairs and ("d", "e") in pairs
    assert not any({x, y} & {"d", "e"} and {x, y} & {"a", "b", "c"} for x, y in pairs)  # no cross-group
    row = links[(links.src_node == "a") & (links.dst_node == "b")].iloc[0]
    assert row["src_traded"] and row["dst_traded"] and row["weight"] == 1.0 and row["available_from"] == SNAP


def test_rival_links_after_the_last_file_are_flagged_carried_forward(cfg):
    sim = pd.DataFrame({"src": ["a"], "dst": ["b"], "score": [0.4]})
    inside = rival_links(sim, pd.Timestamp("2022-06-01"), pd.Timestamp("2022-12-31"), set(), cfg)
    carried = rival_links(sim, pd.Timestamp("2024-06-01"), pd.Timestamp("2022-12-31"), set(), cfg)
    assert not inside["carried_forward"].iloc[0] and carried["carried_forward"].iloc[0]


def correlated_returns(rng, n=300):
    market = rng.normal(0, 0.01, n)
    sector = rng.normal(0, 0.01, n)
    idx = pd.bdate_range("2020-01-01", periods=n)
    frame = pd.DataFrame({
        "a": market + sector + rng.normal(0, 0.002, n), "b": market + sector + rng.normal(0, 0.002, n),
        "c": market + rng.normal(0, 0.01, n), "d": rng.normal(0, 0.01, n), "e": rng.normal(0, 0.01, n),
    }, index=idx)
    return frame


def test_correlation_links_find_the_planted_pair_and_use_only_the_past(cfg):
    rng = np.random.default_rng(0)
    returns = correlated_returns(rng)
    snapshot = returns.index[-20]
    small = cfg.with_overrides({"links.corr.top_k": 1})
    links = correlation_links(returns, snapshot, {"a"}, small)
    assert any((r.src_node, r.dst_node) == ("a", "b") for r in links.itertuples())
    row = links[(links.src_node == "a") & (links.dst_node == "b")].iloc[0]
    assert 0.9 < row["weight"] <= 1.0 and row["src_traded"] and not row["dst_traded"]
    future = returns.copy()
    future.loc[future.index >= snapshot] *= -3.0  # change everything from the snapshot on
    again = correlation_links(future, snapshot, {"a"}, small)
    pd.testing.assert_frame_equal(links, again)


def test_owner_links_follow_shared_ownership_weighted_by_book(cfg):
    weights = pd.DataFrame({"a": [0.5, 0.5, 0.0], "b": [0.5, 0.5, 0.0], "c": [0.0, 0.0, 1.0]}, index=["m1", "m2", "m3"])
    books = pd.Series({"m1": 1.0, "m2": 1.0, "m3": 5.0})
    links = owner_links(weights, books, SNAP, {"a", "b", "c"}, cfg)
    assert {(r.src_node, r.dst_node) for r in links.itertuples()} == {("a", "b")}
    assert links["weight"].between(0, 1).all()


def test_table_links_use_default_weights_and_stacked_snapshots_chain_validity(cfg):
    table = pd.DataFrame({"src_node": ["c:US"], "dst_node": ["s1"], "weight": [np.nan]})
    one = table_links(table, "country", pd.Timestamp("2021-01-01"), {"s1"}, cfg)
    two = table_links(table, "country", pd.Timestamp("2021-02-01"), {"s1"}, cfg)
    stacked = stack_snapshots([two, one])
    assert one["weight"].iloc[0] == 1.0
    first = stacked.sort_values("available_from").iloc[0]
    assert first["valid_to"] == pd.Timestamp("2021-02-01") and pd.isna(stacked.sort_values("available_from").iloc[1]["valid_to"])


# --- labels ------------------------------------------------------------------------------------


def make_world(rng, n=400, jump=0.0, vol_boost=1.0, volume_boost=1.0, day0=300):
    sessions = pd.bdate_range("2020-01-01", periods=n)
    factors = pd.DataFrame(rng.normal(0, 0.01, (n, 6)), index=sessions,
                           columns=["mkt_rf", "smb", "hml", "rmw", "cma", "mom"])
    factors["rf"] = 0.0
    beta = np.array([1.2, 0.3, -0.2, 0.0, 0.1, 0.0])
    noise = rng.normal(0, 0.01, n)
    ret = factors.to_numpy()[:, :6] @ beta + noise
    ret[day0] += jump
    ret[day0 + 1 : day0 + 6] *= vol_boost
    close = 100 * np.exp(np.cumsum(ret))
    spread = 0.01 * np.ones(n)
    spread[day0 + 1 : day0 + 6] *= vol_boost
    px = pd.DataFrame({
        "ret": ret, "open": close * 0.998, "high": close * (1 + spread), "low": close * (1 - spread),
        "close_raw": close, "volume_raw": 1e6,
    }, index=sessions)
    px.loc[sessions[day0 : day0 + 6], "volume_raw"] *= volume_boost
    return sessions, factors, px, beta, noise


def test_residual_model_recovers_betas_and_ignores_excluded_rows():
    rng = np.random.default_rng(1)
    sessions, factors, px, beta, noise = make_world(rng)
    f = factors.to_numpy()[:, :6]
    usable = np.zeros(len(sessions), dtype=bool)
    usable[40:290] = True
    beta_hat, sigma = fit_residual_model(px["ret"].to_numpy(), f, usable, 120)
    assert np.allclose(beta_hat[1:], beta, atol=0.2) and sigma == pytest.approx(0.01, rel=0.2)
    tainted = px["ret"].to_numpy().copy()
    tainted[100:140] += 0.5  # an unrelated event in the window
    bad, _ = fit_residual_model(tainted, f, usable, 120)
    good, _ = fit_residual_model(tainted, f, usable & ~np.isin(np.arange(len(usable)), np.arange(100, 140)), 120)
    assert abs(good[1] - beta[0]) < abs(bad[1] - beta[0])
    assert fit_residual_model(px["ret"].to_numpy(), f, np.zeros(len(usable), dtype=bool), 120) is None


def test_label_windows_standardise_a_planted_jump_and_flat_news_gives_none(cfg):
    rng = np.random.default_rng(2)
    sessions, factors, px, beta, noise = make_world(rng, jump=0.06)
    kwargs = dict(
        ret=px["ret"].to_numpy(), factors=factors.to_numpy()[:, :6], rf=factors["rf"].to_numpy(),
        open_=px["open"].to_numpy(), high=px["high"].to_numpy(), low=px["low"].to_numpy(),
        close=px["close_raw"].to_numpy(), dollar_volume=(px["close_raw"] * px["volume_raw"]).to_numpy(),
        day0=300, excluded=np.zeros(len(sessions), dtype=bool), cfg=cfg,
    )
    out = label_one(**kwargs)
    assert out["d0"] > 4.0 and out["d01"] > 3.0  # a 6% jump on 1% daily vol is about 6 sigma
    assert abs(out["d12"]) < 3.0 and abs(out["d310"]) < 3.0
    short = dict(kwargs, day0=20)
    assert label_one(**short) is None  # no estimation window yet
    late = dict(kwargs, day0=len(sessions) - 3)
    assert label_one(**late) is None  # the d310 window runs past the data


def test_the_estimation_window_ends_ten_sessions_before_day_zero(cfg):
    rng = np.random.default_rng(3)
    sessions, factors, px, *_ = make_world(rng)
    base = dict(
        factors=factors.to_numpy()[:, :6], rf=factors["rf"].to_numpy(), open_=px["open"].to_numpy(),
        high=px["high"].to_numpy(), low=px["low"].to_numpy(), close=px["close_raw"].to_numpy(),
        dollar_volume=(px["close_raw"] * px["volume_raw"]).to_numpy(), day0=300,
        excluded=np.zeros(len(sessions), dtype=bool), cfg=cfg,
    )
    first = label_one(ret=px["ret"].to_numpy(), **base)
    altered = px["ret"].to_numpy().copy()
    altered[291:300] += 0.2  # the 9 sessions before day 0 are NOT used to estimate betas or sigma
    second = label_one(ret=altered, **base)
    assert first["d0"] == pytest.approx(second["d0"])


def test_volatility_and_volume_ratios_rise_with_event_activity(cfg):
    rng = np.random.default_rng(4)
    for boost, expect in ((3.0, "up"), (1.0, "flat")):
        sessions, factors, px, *_ = make_world(rng, vol_boost=boost, volume_boost=boost)
        out = label_one(
            px["ret"].to_numpy(), factors.to_numpy()[:, :6], factors["rf"].to_numpy(), px["open"].to_numpy(),
            px["high"].to_numpy(), px["low"].to_numpy(), px["close_raw"].to_numpy(),
            (px["close_raw"] * px["volume_raw"]).to_numpy(), 300, np.zeros(len(sessions), dtype=bool), cfg,
        )
        if expect == "up":
            assert out["log_vol_ratio"] > 0.5 and out["log_dv_ratio"] > 0.5
        else:
            assert abs(out["log_vol_ratio"]) < 0.3 and abs(out["log_dv_ratio"]) < 0.3
    assert event_window_mask(20, np.array([5]), 0, 3).sum() == 4


def test_garman_klass_is_zero_for_a_flat_day_and_positive_for_range():
    flat = garman_klass(np.array([10.0]), np.array([10.0]), np.array([10.0]), np.array([10.0]), 0.386)
    wide = garman_klass(np.array([10.0]), np.array([11.0]), np.array([9.0]), np.array([10.0]), 0.386)
    assert flat[0] == 0.0 and wide[0] > 0


def test_build_labels_skips_assets_without_history_and_returns_every_window(cfg):
    rng = np.random.default_rng(5)
    sessions, factors, px, *_ = make_world(rng, jump=0.05)
    events = pd.DataFrame({"event_id": ["e1", "e2", "e3"], "security_id": ["A", "A", "Z"],
                           "day0": [sessions[300], sessions[10], sessions[300]]})
    out = build_labels(events, {"A": px}, factors, cfg)
    assert set(out["event_id"]) == {"e1"}  # e2 has no estimation window, Z has no prices
    assert set(out["window"]) == {"d0", "d01", "d12", "d15", "d310"} and out["abn_ret"].notna().all()


def test_unpriced_context_companies_get_no_correlation_and_no_owner_links(cfg):
    """Spec 3.5 (R-9): no prices means no correlation links; 13F holdings in non-S&P names are dropped, so no owner links."""
    rng = np.random.default_rng(1)
    returns = correlated_returns(rng)  # priced columns only; "ctx" has no price series
    snapshot = returns.index[-20]
    corr = correlation_links(returns, snapshot, {"a"}, cfg)
    weights = pd.DataFrame({"a": [0.5, 0.5], "b": [0.5, 0.5]}, index=["m1", "m2"])  # holdings columns are S&P names only
    owner = owner_links(weights, pd.Series({"m1": 1.0, "m2": 1.0}), SNAP, {"a", "b"}, cfg)
    for links in (corr, owner):
        assert "ctx" not in set(links["src_node"]) | set(links["dst_node"])
