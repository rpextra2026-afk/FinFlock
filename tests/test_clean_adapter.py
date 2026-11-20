"""The cleaned tape as framework inputs: other / no_cause carry no group weight, non-company pins go to the world lane, cleaned weights are the pin weights."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finfm.common.config import load_config
from finfm.framework import clean_adapter as CA

from tests.conftest import CONFIG_DIR


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_DIR)


def row(unit, node_type, node, cause_top, cause_group, size, weight, split="fit", day="2020-03-02", lane="company_event"):
    return {"unit_id": unit, "run_split": split, "lane": lane, "kind": "finance", "family": "f", "event_type_final": "asset_event", "cause_top": cause_top, "cause_group": cause_group,
            "ts_utc": pd.Timestamp(day, tz="UTC"), "session_date": pd.Timestamp(day), "session_type": "before_open", "repeat_count": 1, "trust_weight": 1.0, "repeat_weight": 1.0,
            "weight": weight, "node_type": node_type, "node_id": node, "pin_role": "source", "direction": float(np.sign(size)), "size_pct": size, "magnitude": 2.0, "signed_weight": weight * np.sign(size)}


@pytest.fixture(scope="module")
def clean():
    return pd.DataFrame([
        row("a", "company", "AAA", "guidance_up", "earnings_guidance", 4.0, 0.8),
        row("a", "company", "BBB", "guidance_up", "earnings_guidance", -2.0, 0.8),
        row("b", "company", "CCC", "other", "no_cause", 3.0, 1.0),
        row("c", "company", "DDD", "other", "geo_supply_other", 3.0, 1.0),
        row("d", "company", "EEE", "mna_target", "corporate_actions", 5.0, 1.5, split="holdout", day="2021-03-02"),
        row("w", "theme", "oil_energy", "supply_disruption", "geo_supply_other", 3.0, 1.0, lane="world_event"),
        row("a", "sector", "tech", "guidance_up", "earnings_guidance", 1.0, 0.8),  # a non-company pin inside a company item
        row("z", "company", "FFF", "guidance_up", "earnings_guidance", 0.0, 1.0),  # no size: not pinned
    ])


def test_other_and_no_cause_have_all_zero_group_vector(cfg, clean):
    out = CA.from_clean(clean, {}, cfg)
    for unit in ("b", "c"):
        assert not out.readings[unit].group_probs.any()
        assert sum(out.readings[unit].cause_probs.values()) == 0.0
    assert out.readings["a"].group_probs.sum() == 1.0
    assert out.counts["events_with_all_zero_group_vector"] == 2


def test_other_inside_geo_supply_other_cannot_push_that_group(cfg, clean):
    groups = list(cfg.get("causes.groups"))
    out = CA.from_clean(clean, {}, cfg)
    assert out.readings["c"].group_probs[groups.index("geo_supply_other")] == 0.0


def test_non_company_pins_go_to_world_lane_not_company_sheaf(cfg, clean):
    out = CA.from_clean(clean, {}, cfg)
    assert set(out.world_pins["node_type"]) == {"theme", "sector"}
    assert (out.world_pins["lane"] == "world_event").all()
    assert "tech" not in {p["node_id"] for r in out.readings.values() for p in r.pins}
    assert "w" not in out.readings


def test_pin_weights_are_cleaned_weights_and_units(cfg, clean):
    out = CA.from_clean(clean, {}, cfg)
    frame = out.dials["a"]
    assert frame.loc["AAA", "s_ret"] == pytest.approx(4.0 * 0.8)
    assert frame.loc["BBB", "s_ret"] == pytest.approx(-2.0 * 0.8)
    assert frame.loc["BBB", "s_vol"] == pytest.approx(2.0 * 0.8)
    scale = float(cfg.get("framework.pooled.sheaf_pin_scale"))
    assert frame.loc["AAA", "ret"] == pytest.approx(4.0 * 0.8 / scale)
    assert out.dials["d"].loc["EEE", "s_ret"] == pytest.approx(5.0 * 1.5)


def test_zero_size_pins_dropped_and_split(cfg, clean):
    out = CA.from_clean(clean, {}, cfg)
    assert "z" not in out.readings
    split = dict(zip(out.events["event_id"], out.events["split"]))
    assert split["a"] == "dev" and split["d"] == "test"


def test_session_date_is_day0_and_embargo(cfg, clean):
    out = CA.from_clean(clean, {}, cfg, embargo_cut=pd.Timestamp("2020-01-01"))
    assert "a" not in out.readings and out.counts["fit_events_embargoed"] >= 1  # fit events after the cut are dropped
    assert "d" in out.readings
    day0 = dict(zip(out.events["event_id"], out.events["session_day0"]))
    assert day0["d"] == pd.Timestamp("2021-03-02")


def test_test_from_leaves_out_test_events_before_the_sealed_start_and_keeps_the_dev_events(cfg, clean):
    out = CA.from_clean(clean, {}, cfg, test_from=pd.Timestamp("2022-01-01"))
    assert "d" not in out.readings and out.counts["test_events_before_test_from"] >= 1
    assert "a" in out.readings  # fit events are untouched
