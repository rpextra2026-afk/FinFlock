"""Metrics on synthetic data with a known answer (final-build spec, section 6)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finflock.build import metrics as m
from finflock.common.config import load_config

from tests.conftest import CONFIG_DIR


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_DIR)


def synthetic(n_days=60, n_nodes=40, slope=2.0, noise=1.0, seed=0):
    rng = np.random.default_rng(seed)
    day = np.repeat(np.arange(n_days), n_nodes)
    node = np.tile(np.arange(n_nodes), n_days)
    pred = rng.normal(size=len(day))
    y = slope * pred + noise * rng.normal(size=len(day))
    return pd.DataFrame({"y": y, "pred": slope * pred, "base": np.zeros(len(day)), "day": pd.to_datetime("2023-01-02") + pd.to_timedelta(day, unit="D"), "node": node})


def test_error_metrics_perfect_forecast(cfg):
    y = np.array([1.0, -2.0, 3.0, 0.5])
    out = m.error_metrics(y, y.copy(), np.zeros(4), 1.0, cfg)
    assert out["r2_os"] == pytest.approx(1.0) and out["mse"] == 0.0 and out["mae"] == 0.0 and out["gain_pct"] == pytest.approx(100.0)


def test_gain_against_baseline_is_zero_when_forecast_equals_baseline(cfg):
    y = np.array([1.0, -2.0, 3.0, 0.5])
    assert m.error_metrics(y, np.zeros(4), np.zeros(4), 1.0, cfg)["r2_os"] == pytest.approx(0.0)


def test_holm_and_benjamini_hochberg_known_values():
    holm, bh = m.multiple_comparison([0.01, 0.04, 0.03])
    assert holm.tolist() == pytest.approx([0.03, 0.06, 0.06]) and bh.tolist() == pytest.approx([0.03, 0.04, 0.04])


def test_daily_ic_is_one_for_a_perfect_signal_and_nan_on_thin_days():
    y = np.tile(np.arange(30.0), 3)
    days = np.repeat(np.arange(3), 30)
    ic, ric = m.daily_ic(y, y.copy(), days, 3, 20)
    assert ic.tolist() == pytest.approx([1.0, 1.0, 1.0]) and ric.tolist() == pytest.approx([1.0, 1.0, 1.0])
    ic2, _ = m.daily_ic(y[:10], y[:10], np.zeros(10, dtype=int), 1, 20)
    assert np.isnan(ic2[0])


def test_bin_spread_is_monotone_for_a_monotone_relation():
    x = np.arange(100.0)
    out = m.bin_spread(x * 2, x, 5)
    assert out["monotonicity_5"] == pytest.approx(1.0) and out["spread_5"] > 0


def test_classification_metrics_perfect_and_calibrated():
    label = np.array([1, 1, 1, 0, 0, 0], dtype=float)
    score = np.array([2.0, 1.0, 0.5, -0.5, -1.0, -2.0])
    out = m.classification_metrics(label, score, np.array([0.9, 0.9, 0.9, 0.1, 0.1, 0.1]), 10)
    assert out["accuracy"] == 1.0 and out["auc"] == pytest.approx(1.0) and out["mcc"] == pytest.approx(1.0) and out["brier"] == pytest.approx(0.01)


def test_interval_coverage_matches_the_level_for_gaussian_noise():
    rng = np.random.default_rng(1)
    y = rng.normal(size=20000)
    out = m.interval_metrics(y, np.zeros(len(y)), 1.0, [0.5, 0.9], [0.05, 0.5], 10)
    assert out["coverage_50"] == pytest.approx(0.5, abs=0.02) and out["coverage_90"] == pytest.approx(0.9, abs=0.02)
    assert sum(out[f"pit_bin_{b:02d}"] for b in range(10)) == pytest.approx(1.0)


def test_slope_test_recovers_the_slope(cfg):
    f = synthetic(slope=2.0)
    days, codes = np.unique(f["day"].to_numpy(), return_inverse=True)
    out = m.slope_tests(f["y"].to_numpy(), f["pred"].to_numpy(), codes, len(days), 5, f["base"].to_numpy())
    assert out["slope"] == pytest.approx(1.0, abs=0.1) and out["slope_t_clustered"] > 5  # pred already carries the factor 2, so the slope of y on pred is 1


def test_diebold_mariano_and_gain_interval_favour_a_better_model(cfg):
    f = synthetic()
    rng = np.random.default_rng(0)
    out = m.all_metrics(f, "ret", 1.0, 1.0, 1 / (1 + np.exp(-f["pred"].to_numpy())), cfg, rng, draws=200)
    assert out["r2_os"][0] > 0.5 and out["dm_p"][0] < 0.001 and out["cw_p"][0] < 0.001
    lo, hi = out["gain_ci_block"][1], out["gain_ci_block"][2]
    assert lo < out["gain_ci_block"][0] < hi and lo > 0
    assert out["ic_mean"][0] > 0.5 and out["dir_auc"][0] > 0.7 and out["ls_mean"][0] > 0


def test_a_useless_forecast_has_no_gain(cfg):
    f = synthetic(slope=0.0)
    f["pred"] = np.random.default_rng(5).normal(size=len(f))
    out = m.all_metrics(f, "ret", 1.0, 1.0, np.full(len(f), 0.5), cfg, np.random.default_rng(0), draws=200)
    assert out["r2_os"][0] < 0 and abs(out["ic_mean"][0]) < 0.1


def test_collector_writes_the_long_format(cfg):
    c = m.Collector("tapeX", "hold-out", "d01")
    m.write_metrics(c, "ret", "all", "all", {"r2_os": (0.1, 0.0, 0.2)}, 10)
    frame = c.frame()
    assert list(frame.columns) == m.COLUMNS and frame.loc[0, "metric"] == "r2_os" and frame.loc[0, "n"] == 10
