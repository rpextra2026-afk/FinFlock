"""World lane on synthetic prices and news with a known effect: news on a theme raises the next day's volatility."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finflock.build import metrics as M
from finflock.build import world as W
from finflock.common.config import load_config

from tests.conftest import CONFIG_DIR

THEMES = ["rates_fed", "oil_energy"]


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_DIR).with_overrides({"build.metrics.slice_bootstrap_draws": 40, "build.world.min_days": 20})


def synthetic(effect=True, seed=3):
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-01", "2023-12-29")
    news = rng.random(len(days)) < 0.25
    stocks = [f"S{i}" for i in range(25)]
    rows = []
    price = {s: 100.0 for s in stocks}
    for i, d in enumerate(days):
        scale = 0.01 * (3.0 if (effect and news[i]) else 1.0)
        for s in stocks:
            price[s] *= 1 + scale * rng.normal()
            rows.append({"security_id": s, "session_date": d, "close_raw": price[s], "volume_raw": 1e6 * (1 + 0.1 * rng.random())})
    prices = pd.DataFrame(rows)
    items = []
    for i, d in enumerate(days):
        if news[i]:
            items.append({"unit_id": f"u{i}", "lane": "world_event", "node_type": "theme", "node_id": THEMES[i % 2], "session_date": d, "weight": 1.0})
    return pd.DataFrame(items), prices


RUN = {"fit_start": "2020-01-01", "fit_end": "2021-12-31", "holdout_start": "2022-01-03", "holdout_end": "2023-12-29"}


def metric(frame, name, tkind, vname, pname, metric_name, dial="vol"):
    sel = frame[(frame.dial == dial) & (frame.slice == f"world_{tkind}_{vname}") & (frame.slice_value == pname) & (frame.metric == metric_name)]
    return float(sel.value.iloc[0])


def test_a_news_effect_on_volatility_is_found_every_day_and_beats_the_shifted_placebo(cfg):
    clean, prices = synthetic(effect=True)
    col = M.Collector("t", "holdout", "d01")
    W.evaluate_world(col, clean, prices, RUN, THEMES, cfg, np.random.default_rng(0))
    f = col.frame()
    real = metric(f, "vol", "raw", "every_day", "real", "r2_os")
    assert real > 0.0 and real > metric(f, "vol", "raw", "every_day", "shift_41_sessions", "r2_os") + 0.001


def test_day_demeaned_target_leaves_nothing_for_market_level_news(cfg):
    clean, prices = synthetic(effect=True)
    col = M.Collector("t", "holdout", "d01")
    W.evaluate_world(col, clean, prices, RUN, THEMES, cfg, np.random.default_rng(0))
    f = col.frame()
    assert abs(metric(f, "vol", "demeaned", "every_day", "real", "r2_os")) < 0.01


def test_no_effect_means_no_gain_and_lag_profile_and_theme_share_rows_exist(cfg):
    clean, prices = synthetic(effect=False)
    col = M.Collector("t", "holdout", "d01")
    coefs = W.evaluate_world(col, clean, prices, RUN, THEMES, cfg, np.random.default_rng(0))
    f = col.frame()
    assert metric(f, "vol", "raw", "every_day", "real", "r2_os") < 0.01
    assert (f.slice == "world_lag_profile_raw").any() and (f.slice == "world_theme_share_by_year").any() and len(coefs) > 0
