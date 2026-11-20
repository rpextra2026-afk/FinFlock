"""The clean-tape framework run: the reader mode, the dev-fit key, the pooled predictor, the world lane switch, the delisting trace, the metrics and the freeze."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import torch

from oryxis.common.config import load_config
from oryxis.framework import final_metrics as FM
from oryxis.framework import freeze as FZ
from oryxis.framework import road1_pooled as RP
from oryxis.framework import world_lane as WL
from oryxis.framework.delist_trace import hit_table, trace
from oryxis.framework.events import EventReading
from oryxis.real.devfit_io import dev_fit_problems
from oryxis.real.framework_cell import confidence_rule
from oryxis.reader.salvage import config_for_mode, mode_settings
from oryxis.road1.arms import SheafModel
from tests.conftest import CONFIG_DIR
from tests.test_build_world import RUN, THEMES, synthetic


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_DIR).with_overrides({"build.metrics.slice_bootstrap_draws": 40, "build.world.min_days": 20})


# ---- the reader mode ---------------------------------------------------------------------------------------------------------

def test_reader2_clean_mode_uses_no_certainty_and_no_scheduled(cfg):
    settings = mode_settings(cfg, "reader2_clean")
    assert settings["framework_confidence"] == "none" and settings["include_salvaged"] is False and settings["clean_tape"] is True
    _, record = config_for_mode(cfg, "reader2_clean")
    assert {"certainty", "scheduled"} <= set(record["dial_features_dropped"])
    assert confidence_rule(cfg, "none")(float("nan")) == 1.0  # framework_confidence = 1 for every event


def test_old_modes_are_untouched_by_the_new_one(cfg):
    for name in ("original", "salvaged_no_certainty"):
        assert not mode_settings(cfg, name).get("clean_tape")
    assert str(cfg.get("framework.road1_arm_legacy")) == "sheaf" and str(cfg.get("framework.road1_arm")) == "pooled_sheaf"


def test_dev_fit_of_another_mode_is_refused():
    now = {"bucket": "db3", "llm": "llm2", "reader_mode": "reader2_clean", "arm": "pooled_sheaf", "config_hash": "h", "fit_config_hash": "f"}
    old = {"bucket": "db3", "llm": "llm2", "reader_mode": "salvaged_no_certainty", "arm": "sheaf", "config_hash": "h", "fit_config_hash": "f"}
    problems, _ = dev_fit_problems(old, now)
    assert any("reader_mode" in p for p in problems) and any("arm" in p for p in problems)
    assert dev_fit_problems(now, now) == ([], [])


# ---- the pooled predictor -------------------------------------------------------------------------------------------------

def links_frame():
    rows = [("A", "B", 0, 0.9, "2000-01-01", None), ("A", "C", 1, 0.5, "2000-01-01", None), ("B", "D", 2, 0.4, "2000-01-01", None)]
    frame = pd.DataFrame(rows, columns=["src_node", "dst_node", "link_type", "weight", "available_from", "valid_to"])
    frame["link_type"] = ["industry", "corr", "owner"]
    frame["available_from"] = pd.to_datetime(frame["available_from"])
    frame["valid_to"] = pd.to_datetime(frame["valid_to"])
    return frame


def test_pooled_predictor_pins_keep_label_unit_dials_and_free_nodes_are_predicted(cfg):
    from oryxis.build.run import build_config

    pooled = build_config(cfg)
    torch.manual_seed(0)
    model = SheafModel(3, 1, 3, [True, True, True])
    predictor = RP.PooledRoad1Predictor(model, 0.1, np.array([0.02, 0.5, 0.5]), links_frame(), pooled, {"vol": (0.1, 0.01), "volu": (0.0, 0.02)})
    reading = EventReading("e1", ({"node_id": "A", "direction": 1.0, "value_effect_pct": 4.0, "magnitude": 2.0, "role": "source"},), {}, np.zeros(6), float("nan"))
    dials = pd.DataFrame({"ret": [0.032], "vol": [0.032], "volu": [0.032], "s_ret": [3.2], "s_vol": [3.2], "s_volu": [3.2]}, index=["A"])
    out = predictor.predict(reading, dials, pd.Timestamp("2020-01-02"))
    assert out.loc["A", "role"] == "pinned" and out.loc["A", "ret"] == pytest.approx(0.032)
    assert out.loc["A", "vol"] == pytest.approx(0.1 + 0.01 * 3.2) and out.loc["A", "volu"] == pytest.approx(0.02 * 3.2)
    assert (out["role"].iloc[1:] == "ripple").all() and len(out) > 1
    with torch.no_grad():
        model.delta_u += 0.3
        model.delta_v += 0.3
    scaled = predictor.predict(reading, dials, pd.Timestamp("2020-01-02"))
    raw = RP.predict_graph(model, predictor.graph_for(reading, dials, pd.Timestamp("2020-01-02")), 0.1, predictor.scale, pooled)
    assert scaled["ret"].iloc[1:].to_numpy() == pytest.approx(raw[:, 0] * 0.01)  # free-node return: percent -> fraction
    assert scaled["vol"].iloc[1:].to_numpy() == pytest.approx(raw[:, 1])
    with pytest.raises(SystemExit):
        RP.check_label_units(np.array([0.02, 0.5, 0.5]), pooled)
    assert list(out.columns[:3]) == ["ret", "vol", "volu"]


def test_all_zero_group_vector_does_not_silence_the_pooled_sheaf(cfg):
    """The pooled sheaf has one group; an 'other' / no_cause reading (all-zero cause vector) still gets the pooled rule."""
    from oryxis.build.run import build_config

    pooled = build_config(cfg)
    torch.manual_seed(1)
    model = SheafModel(3, 1, 3, [True, True, True])
    with torch.no_grad():
        model.delta_u += 0.3
        model.delta_v += 0.3
    predictor = RP.PooledRoad1Predictor(model, 0.1, np.array([0.02, 0.5, 0.5]), links_frame(), pooled, {})
    dials = pd.DataFrame({"ret": [0.03], "vol": [0.03], "volu": [0.03], "s_ret": [3.0], "s_vol": [3.0], "s_volu": [3.0]}, index=["A"])
    pins = ({"node_id": "A", "direction": 1.0, "value_effect_pct": 3.0, "magnitude": 2.0, "role": "source"},)
    zero = predictor.predict(EventReading("z", pins, {}, np.zeros(6), float("nan")), dials, pd.Timestamp("2020-01-02"))
    one = predictor.predict(EventReading("o", pins, {}, np.eye(6)[0], float("nan")), dials, pd.Timestamp("2020-01-02"))
    pd.testing.assert_frame_equal(zero, one)


def test_pin_maps_fit_on_dev_only_and_are_zero_when_too_few_rows(cfg):
    rng = np.random.default_rng(0)
    source = {f"e{i}": pd.DataFrame({"s_vol": [x], "s_volu": [x]}, index=["A"]) for i, x in enumerate(rng.uniform(0, 5, 400))}
    labels = pd.DataFrame({"event_id": list(source), "node_id": "A", "log_vol_ratio": [0.1 + 0.2 * f["s_vol"].iloc[0] for f in source.values()], "log_dv_ratio": 0.0})
    maps = RP.fit_pin_maps(source, labels, list(source)[:300], cfg)
    assert maps["vol"][1] == pytest.approx(0.2, abs=1e-6) and maps["vol"][0] == pytest.approx(0.1, abs=1e-6)
    assert RP.fit_pin_maps(source, labels, list(source)[:5], cfg)["vol"] == (0.0, 0.0)


# ---- the world lane switch ----------------------------------------------------------------------------------------------------

def test_world_lane_shift_is_higher_on_news_days_and_the_switch_turns_it_off(cfg):
    clean, prices = synthetic(effect=True)
    days = pd.bdate_range("2022-01-03", "2023-12-29")
    on = cfg.with_overrides({"framework.world_lane.enabled": True})
    shift = WL.fit_world_shift(clean, prices, RUN, THEMES, on, days)
    assert shift.vol.abs().max() > 0 and shift.vol.abs().max() <= float(on.get("framework.world_lane.clip_log_vol")) + 1e-12
    news_days = pd.DatetimeIndex(sorted(set(clean["session_date"]))).intersection(days)
    assert shift.vol.reindex(news_days).mean() > shift.vol.drop(news_days).mean()
    assert WL.shift_for(on, shift) is shift
    off = cfg.with_overrides({"framework.world_lane.enabled": False})
    assert WL.shift_for(off, shift) is None and not WL.enabled(off)


# ---- the delisting trace ----------------------------------------------------------------------------------------------------------

def test_delisting_trace_finds_the_names_and_their_price_rows(cfg):
    from oryxis.portfolio.execute import run_backtest
    from oryxis.portfolio.market import Market

    days = pd.bdate_range("2021-01-04", periods=40)
    close = pd.DataFrame({"GOOD": 100.0 + np.arange(40) * 0.1, "DEAD": 50.0 + np.arange(40) * 0.0}, index=days)
    close.loc[days[25:], "DEAD"] = np.nan
    oo = close.pct_change(fill_method=None).shift(-1)
    oo.loc[days[-1]] = np.nan
    market = Market(close, close * 0 + 1e6, oo, close.apply(lambda c: c.last_valid_index()))

    class Rm:
        pass

    rm = Rm()
    rm.market = market

    def target(date, view, held):
        return pd.Series({"GOOD": 0.5, "DEAD": 0.5})

    def go(rule):
        return run_backtest(market, list(days[10:35]), target, delist_return=rule)

    zero, default = go(None), go(-0.3)
    assert not np.allclose(zero.gross.to_numpy(), default.gross.to_numpy())
    px = close.stack().rename("close_raw").reset_index()
    px.columns = ["session_date", "security_id", "close_raw"]
    px["volume_raw"] = 1e6
    table, hits = trace(zero, default, rm, px, pd.DataFrame(columns=["security_id", "ex_date", "action_type"]), -0.3, "demo")
    assert list(table["security_id"]) == ["DEAD"] and table["hit_dates"].iloc[0] >= 1
    assert table["contribution_gross"].iloc[0] == pytest.approx(hits["weight"].sum() * -0.3)
    assert "data_check" in table.columns and table["px_rows_after_last_close"].iloc[0] == 0
    assert len(hit_table(default, rm)) == len(hits)


# ---- the metrics --------------------------------------------------------------------------------------------------------------------

def make_series(n=400, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2022-01-03", periods=n)
    return {name: pd.Series(rng.normal(mu, 0.01, n), index=days) for name, mu in (("a", 0.0006), ("b", 0.0003), ("c", 0.0))}


def test_paired_sharpe_table_has_holm_adjusted_p_values(cfg):
    small = cfg.with_overrides({"eval.bootstrap.draws": 199})
    table = FM.paired_sharpe_table(make_series(), small, "net_sharpe")
    assert len(table) == 3 and (table["p_holm"] >= table["p_value"] - 1e-12).all() and (table["p_holm"] <= 1.0).all()


def test_deflated_pbo_cpcv_tables(cfg):
    out = FM.deflated_pbo_cpcv(make_series(n=512), cfg)
    assert {"deflated_sharpe", "pbo", "cpcv"} <= set(out)
    assert 0.0 <= float(out["pbo"]["pbo"].iloc[0]) <= 1.0 and len(out["deflated_sharpe"]) == 3


def test_checklist_marks_present_added_and_missing(cfg):
    summary = pd.DataFrame({"variant": ["road1", "road2", "both", "merged"] * 2, "headline_set": ["headline"] * 4 + ["long_short"] * 4, "cum_net_return": 0.1, "sharpe_gross": 1.0})
    table = FM.checklist(summary, {"cpcv": pd.DataFrame({"x": [1]})}, ["cpcv"], {"pbo": "needs more days"})
    state = dict(zip(table["metric"], table["state"]))
    assert state["cum_net_return"] == "present" and state["cpcv"] == "added" and state["pbo"] == "missing" and state["alpha_annual"] == "missing"
    assert state["merged"] == "present" and state["long_short"] == "present"


# ---- the freeze -----------------------------------------------------------------------------------------------------------------------

def test_freeze_refuses_a_changed_hash_and_pins_the_dev_fit():
    frozen = {"config_hash": "a", "config_unhashed_hash": "b", "code_sha256": "c", "framework_scripts_sha256": {}, "clean_content_sha256": {"r": "x"}, "clean_file_sha256": {"r": "y"},
              "persona_tape_sha256": {"r": {"p.bundle": "z"}}, "dev_fit_sha256": {"r": None}}
    now = json.loads(json.dumps(frozen))
    assert FZ.problems(frozen, now, ["r"]) == []
    now["code_sha256"] = "changed"
    now["persona_tape_sha256"]["r"]["p.bundle"] = "other"
    bad = FZ.problems(frozen, now, ["r"])
    assert len(bad) == 2 and any("code_sha256" in b for b in bad) and any("persona_tape_sha256" in b for b in bad)


# ---- rivals through the same optimiser -------------------------------------------------------------------------------------------------

def test_rival_scores_become_one_day_forecasts_with_a_slope_fitted_before_the_test_window():
    from oryxis.framework import rival_adapter as RA

    days = pd.bdate_range("2021-01-04", periods=120)
    rng = np.random.default_rng(0)
    oo = pd.DataFrame(rng.normal(0, 0.01, (120, 4)), index=days, columns=list("ABCD"))
    rows = []
    for i, d in enumerate(days[:-1]):
        for s in "ABCD":
            score = rng.normal()
            oo.loc[days[i + 1], s] += 0.002 * score  # the next session's return carries the score
            rows.append({"decision_date": d, "security_id": s, "score": score})
    scores = pd.DataFrame(rows)
    slope, n = RA.fit_slope(scores, oo, days[80])
    assert slope == pytest.approx(0.002, abs=0.0015) and n == 80 * 4
    assert np.isnan(RA.fit_slope(scores.head(40), oo, days[80])[0])  # too few dev points: not fitted
    fs = RA.forecasts_from_scores("basic", scores, list(days[80:100]), days, list("ABCD"), slope, 10)
    assert len(fs) == 20 and fs[0].increments.shape == (10, 4) and not fs[0].increments[1:].any() and fs[0].cov_shift is None and fs[0].confidence == 1.0
    assert fs[0].day0 == 80


# ---- the sheaf's job in the optimiser: realised volatility against the target, accuracy of the vol and volume dials ------------------------------------------

def test_realised_vol_table_reports_the_unscaled_volatility_and_the_net_volatility_against_the_target(cfg):
    from types import SimpleNamespace

    rng = np.random.default_rng(0)
    days = pd.bdate_range("2023-01-02", periods=300)
    headline = f"{int(cfg.get('portfolio.scoring.headline_round_trip_bp'))}bp"
    raw = pd.Series(rng.normal(0, 0.02, len(days)), index=days)
    scale = 0.10 / float(raw.std() * np.sqrt(252))
    result = SimpleNamespace(gross=raw * scale, risk_scale=scale)
    table = FM.realised_vol_table({"merged/long_short/delist_0": result}, {"merged/long_short/delist_0": pd.DataFrame({headline: raw * scale}, index=days)}, cfg.with_overrides({"eval.bootstrap.draws": 60}))
    row = table.iloc[0]
    assert abs(row["realised_annual_vol_gross"] - 0.10) < 1e-9 and abs(row["unscaled_annual_vol"] - 0.10 / scale) < 1e-9
    assert row["gross_ci_low"] <= row["realised_annual_vol_gross"] <= row["gross_ci_high"] and row["n_days"] == 300


def test_dial_accuracy_is_exact_for_a_perfect_dial_and_has_event_clustered_intervals(cfg):
    from types import SimpleNamespace

    rng = np.random.default_rng(1)
    columns = [f"S{i}" for i in range(6)]
    forecasts, rows = [], []
    for e in range(40):
        assets = np.array([0, 1, 2])
        vol, volu = rng.normal(0, 0.3, 3), rng.normal(0, 0.3, 3)
        forecasts.append(SimpleNamespace(event_id=f"e{e}", assets=assets, vol_dial=vol, volume_dial=volu))
        rows += [{"event_id": f"e{e}", "security_id": columns[a], "log_vol_ratio": vol[j], "log_dv_ratio": volu[j]} for j, a in enumerate(assets)]
    table = FM.dial_accuracy_table({"merged": forecasts}, pd.DataFrame(rows), columns, cfg.with_overrides({"eval.bootstrap.draws": 50}))
    assert set(table["dial"]) == {"vol", "volu", "cost_input"} and (table["n_events"] == 40).all()
    stats = table.set_index(["dial", "statistic"])["value"]
    assert abs(stats[("vol", "bias")]) < 1e-12 and abs(stats[("vol", "correlation")] - 1) < 1e-9 and abs(stats[("cost_input", "r2_os_vs_zero")] - 1) < 1e-9


def test_checklist_requires_the_two_portfolio_side_metrics(cfg):
    table = FM.checklist(pd.DataFrame({"variant": ["merged"], "headline_set": ["long_short"]}), {}, [], None)
    assert {"realised_vol_vs_target", "dial_accuracy_vol_volu"} <= set(table["metric"])
    assert list(FM.REQUIRED["Headline sets"]) == ["long_short", "headline"]


def test_freeze_records_the_headline_set_and_refuses_a_change_of_it(cfg):


    final = cfg.get("framework.final")
    assert final["headline_set"] == "long_short" and final["second_set"] == "headline" and list(final["headline_sets"]) == ["long_short", "headline"]
    frozen = {"settings": {"headline_set": "long_short"}, "config_hash": "a", "config_unhashed_hash": "b", "code_sha256": "c", "framework_scripts_sha256": {}, "clean_content_sha256": {}, "clean_file_sha256": {},
              "persona_tape_sha256": {}, "dev_fit_sha256": {}}
    now = {**frozen, "settings": {"headline_set": "headline"}}
    assert any(b.startswith("settings") for b in FZ.problems(frozen, now, ["r"]))


# ---- the sealed-window stage ---------------------------------------------------------------------------------------------------------------------------

def test_sealed_stage_takes_its_lock_once_and_only_for_the_tessera_tapes(cfg, tmp_path):
    from oryxis.framework import sealed as SL

    spec = SL.sealed_settings(cfg, "tessera-llm2")
    assert spec["start"] and spec["end"] and spec["results_dir"] == "framework-sealed"
    with pytest.raises(SystemExit):
        SL.sealed_settings(cfg, "db3-llm2")
    SL.take_lock(tmp_path, spec, "tessera-llm2", {"freeze": "matched"}, "abc")
    assert json.loads((tmp_path / spec["lock_file"]).read_text())["run"] == "tessera-llm2"
    with pytest.raises(SystemExit, match="already opened"):
        SL.take_lock(tmp_path, spec, "tessera-llm2", {"freeze": "matched"}, "abc")
    window = SL.window_cfg({"fit_end": "2021-12-31", "holdout_start": "2022-01-03", "holdout_end": "2023-12-31"}, spec)
    assert window["fit_end"] == "2021-12-31" and window["holdout_start"] == str(spec["start"])


# ---- the road-2 volume-dial calibration ----------------------------------------------------------------------------------------------------------------

def make_pairs(n_events=60, per=40, scale=0.0, seed=0):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2016-01-04", periods=n_events * 3)[:: 3][:n_events]
    rows = []
    for e, d in enumerate(days):
        x = rng.normal(4.0, 2.0, per)
        y = -0.1 + scale * (x - 4.0) + rng.normal(0, 0.3, per)
        rows += [(f"e{e}", d, xi, yi) for xi, yi in zip(x, y)]
    return pd.DataFrame(rows, columns=["event_id", "day0", "x", "y"])


def test_calibration_scale_is_shrink_only_and_the_shift_matches_the_means(cfg):
    from oryxis.framework import calibration as CAL

    a, b = CAL.fit_scale_shift(np.array([0.0, 1.0, 2.0, 3.0]), np.array([0.0, 2.0, 4.0, 6.0]), 1.0)
    assert a == 1.0 and abs(b - (3.0 - 1.5)) < 1e-12   # slope 2 is clipped to 1; shift = mean(y) - a mean(x)
    a, b = CAL.fit_scale_shift(np.array([0.0, 1.0, 2.0]), np.array([3.0, 2.0, 1.0]), 1.0)
    assert a == 0.0 and abs(b - 2.0) < 1e-12           # a negative slope is clipped to 0


def test_calibration_is_used_when_it_helps_in_every_time_block_and_the_blocks_are_reported(cfg):
    from oryxis.framework import calibration as CAL

    record = CAL.fit_volu_calibration(make_pairs(scale=0.0), cfg)
    assert record["used"] is True and len(record["blocks"]) == int(cfg.get("framework.calibration.folds"))
    assert all(r["calibrated_rmse"] < r["raw_rmse"] for r in record["blocks"]) and 0.0 <= record["scale"] < 0.05 and abs(record["shift"] + 0.1) < 0.2
    assert abs(CAL.apply(np.array([5.0]), (record["scale"], record["shift"]))[0] - (record["shift"] + 5.0 * record["scale"])) < 1e-12
    assert CAL.apply(np.array([5.0]), None)[0] == 5.0   # off switch / not stable: the raw dial


def test_calibration_stays_raw_when_it_does_not_beat_the_raw_dial_in_every_block(cfg):
    from oryxis.framework import calibration as CAL

    pairs = make_pairs(scale=0.0)
    pairs["y"] = pairs["x"]   # the raw dial is exact: no calibration can beat it
    record = CAL.fit_volu_calibration(pairs, cfg)
    assert record["used"] is False and CAL.setting_from_record(record, True) is None
    assert CAL.setting_from_record({"used": True, "scale": 0.5, "shift": 0.1}, False) is None   # the off switch


def test_calibration_hash_changes_with_the_values():
    from oryxis.framework import calibration as CAL

    assert CAL.calibration_sha256({"scale": 0.1, "shift": 0.2}) != CAL.calibration_sha256({"scale": 0.1, "shift": 0.3})
    assert CAL.calibration_sha256({"a": 1, "b": 2}) == CAL.calibration_sha256({"b": 2, "a": 1})


def test_figure_data_tables_have_one_row_per_day_and_the_position_quantiles(cfg):
    from types import SimpleNamespace

    days = pd.bdate_range("2023-01-02", periods=10)
    pos = pd.DataFrame(np.random.default_rng(0).normal(0, 0.01, (10, 5)), index=days, columns=list("abcde"))
    result = SimpleNamespace(gross=pd.Series(0.001, index=days), cash_weight=pd.Series(0.1, index=days), positions=pos, trades=pos.diff().fillna(0.0), delist_hits=pd.Series(0, index=days))
    net = pd.DataFrame({"10bp": result.gross - 0.0001}, index=days)
    daily = FM.daily_series_table("merged/long_short/delist_0", result, net)
    assert len(daily) == 10 and {"fill_date", "gross", "net_10bp", "gross_exposure", "one_way_turnover", "n_positions"} <= set(daily.columns)
    sizes = FM.position_size_table("merged/long_short/delist_0", result)
    assert list(sizes["quantile"])[0] < list(sizes["quantile"])[-1] and (sizes["abs_weight"].diff().dropna() >= 0).all()


def test_sheaf_maps_table_has_the_fitted_delta_and_the_singular_values_of_i_plus_delta():
    from oryxis.framework import sheaf_internals as SI
    from oryxis.road1.arms import SheafModel

    model = SheafModel(2, 1, 3, [True, False])
    with torch.no_grad():
        model.delta_u[0, 0] = -torch.eye(3) + torch.diag(torch.tensor([1.0, 0.0, 0.0], dtype=model.delta_u.dtype))
    table = SI.maps_table(model, ["industry", "rival"])
    row = table[(table["link_type"] == "industry") & (table["side"] == "u")].iloc[0]
    assert abs(row["singular_value_1"] - 1.0) < 1e-9 and row["singular_value_2"] < 1e-9 and len(table) == 4   # 2 link types x 1 group x 2 sides
    ident = table[(table["link_type"] == "rival") & (table["side"] == "u")].iloc[0]
    assert all(abs(ident[f"singular_value_{k}"] - 1.0) < 1e-9 for k in (1, 2, 3)) and ident["delta_frobenius"] == 0.0


def test_freeze_settings_do_not_depend_on_which_runs_are_checked(cfg, tmp_path):
    from tests.conftest import CONFIG_DIR as CONFIG

    root = tmp_path
    clean = root / str(cfg.get("clean.output_dir"))
    clean.mkdir(parents=True)
    for run in ["db3-llm2", "tessera-llm2"]:
        (clean / f"clean_{run}_report.json").write_text(json.dumps({"content_sha256": "synthetic-content"}))
        (clean / f"clean_{run}.parquet").write_bytes(b"synthetic-file-hash-fixture")
    a = FZ.current_hashes(cfg, root, CONFIG.parent, ["db3-llm2"], None)
    b = FZ.current_hashes(cfg, root, CONFIG.parent, ["tessera-llm2"], None)
    assert a["settings"] == b["settings"] and len(a["settings"]["calibration"]["values"]) == len(cfg.get("clean.runs"))   # a single-run check compares like with like
