"""Cleaning steps on small synthetic frames with known answers (final-build spec, section 2)."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest

from oryxis.clean import placeholders, steps
from oryxis.common.config import load_config
from oryxis.common.timing import TradingCalendar

from tests.conftest import CONFIG_DIR


@pytest.fixture(scope="module")
def cfg():
    return load_config(CONFIG_DIR)


@pytest.fixture(scope="module")
def cal(cfg):
    return TradingCalendar.from_config(cfg)


def frame(**cols):
    return pd.DataFrame(cols)


def test_normalise_text_strips_links_and_cashtags():
    assert steps.normalise_text("Buy $AAPL now https://x.co/a   &amp; Sell") == "buy now & sell"


def test_gate_drops_each_class_once_and_counts_theme_matches(cfg):
    themes = steps.compile_themes(cfg)
    items = frame(unit_id=list("abcd"), event_type_final=["analyst_opinion", "not_market_news", "asset_event", "asset_event"], status=["ok", "ok", "skipped", "ok"], text_clean=["fed rate hike", "tariffs on steel", "tariffs", "plain"])
    counter = steps.Counter()
    out = steps.gate(items, themes, cfg, counter)
    assert out.unit_id.tolist() == ["d"]
    record = counter.steps[0]
    assert (record["analyst_opinion"], record["not_market_news"], record["skipped"]) == (1, 1, 1)
    assert record["not_market_news_matching_a_theme"] == 1 and record["skipped_matching_a_theme"] == 1


def test_entity_check_keeps_only_known_nodes_and_logs_the_rest(cfg):
    pins = json.dumps([{"node_id": "AAA"}, {"node_id": "c:CN"}])
    counter = steps.Counter()
    out = steps.entity_check(frame(reading_pins_json=[pins, None]), {"AAA"}, cfg, counter)
    assert [len(p) for p in out.pins] == [1, 0]
    assert counter.steps[0]["pins_unmatched"] == 1 and counter.steps[0]["unmatched_examples"] == ["c:CN"]


def test_session_timing_before_open_intraday_after_close_weekend_and_date_only(cfg, cal):
    stamps = ["2023-03-06 13:00", "2023-03-06 15:00", "2023-03-06 22:00", "2023-03-04 15:00", "2023-03-04 00:00"]
    items = frame(available_utc=pd.to_datetime(stamps, utc=True), available_date=["2023-03-06", "2023-03-06", "2023-03-06", "2023-03-04", "2023-03-04"])
    out = steps.session_timing(items, cal, cfg, steps.Counter())
    assert out.session_type.tolist() == ["before_open", "intraday", "after_close", "non_trading_day", "date_only"]
    assert [d.strftime("%Y-%m-%d") for d in out.session_date] == ["2023-03-06", "2023-03-06", "2023-03-07", "2023-03-06", "2023-03-06"]


def test_dedupe_merges_same_story_inside_the_window_only(cfg, cal):
    story = "company x announces a very long headline about the same thing today"
    stamps = pd.to_datetime(["2023-03-06 15:00", "2023-03-06 16:00", "2023-03-08 15:00", "2023-03-20 15:00"], utc=True)
    items = frame(unit_id=list("abcd"), kind=["finance"] * 4, text_clean=[story] * 4, ts_utc=stamps, session_date=pd.to_datetime(["2023-03-06", "2023-03-06", "2023-03-08", "2023-03-20"]))
    out = steps.dedupe(items, cal, cfg, steps.Counter())
    assert out.unit_id.tolist() == ["a", "d"] and out.repeat_count.tolist() == [3, 1]


def test_dedupe_never_merges_short_texts(cfg, cal):
    items = frame(unit_id=list("ab"), kind=["social"] * 2, text_clean=["up", "up"], ts_utc=pd.to_datetime(["2023-03-06 15:00"] * 2, utc=True), session_date=pd.to_datetime(["2023-03-06"] * 2))
    assert len(steps.dedupe(items, cal, cfg, steps.Counter())) == 2


def test_cause_other_goes_to_no_cause_group(cfg):
    groups = {c: g for g, cs in cfg.get("causes.groups").items() for c in cs}
    out = steps.cause_groups(frame(cause_top=["earnings_beat", "other", None]), cfg, groups, steps.Counter())
    assert out.cause_group.tolist() == ["earnings_guidance", "no_cause", "no_cause"]


def test_lanes_and_touch_points_weights(cfg):
    themes = steps.compile_themes(cfg)
    items = frame(unit_id=list("abc"), kind=["finance", "social", "geo"], family=["f"] * 3, event_type_final=["asset_event"] * 3, cause_top=["earnings_beat", "other", "other"], cause_group=["earnings_guidance", "no_cause", "no_cause"], text_clean=["quarter results", "the fed raises interest rate", "a recipe for soup"], ts_utc=pd.to_datetime(["2023-03-06 15:00"] * 3, utc=True), session_date=pd.to_datetime(["2023-03-06"] * 3), session_type=["intraday"] * 3, split=["fit"] * 3, repeat_count=[1, 3, 1], pins=[[{"node_id": "AAA", "direction": "down", "value_effect_pct": -2.0, "role": "source", "magnitude": 2}], [], []])
    items = steps.assign_lanes(items, themes, cfg, steps.Counter())
    assert items.lane.tolist() == ["company_event", "world_event", "attention_only"]
    long = steps.touch_points(items, {"finance": 1.5, "social": 0.5}, None, cfg, steps.Counter())
    company = long[long.node_type == "company"].iloc[0]
    assert company.direction == -1.0 and company.weight == pytest.approx(1.5) and company.signed_weight == pytest.approx(-1.5)
    social = long[long.unit_id == "b"]
    assert set(social.node_type) == {"theme", "market"} and social.weight.iloc[0] == pytest.approx(0.5 * (1.0 + np.log(3)))
    assert long[long.unit_id == "c"].node_type.isna().all()


def test_returns_frame_is_market_adjusted_in_percent(cfg):
    px = frame(security_id=["A", "A", "B", "B"], session_date=pd.to_datetime(["2023-03-06", "2023-03-07"] * 2), close_raw=[100.0, 110.0, 100.0, 100.0])
    out = steps.returns_frame(px, cfg)
    assert out.loc["2023-03-07", "A"] == pytest.approx(5.0) and out.loc["2023-03-07", "B"] == pytest.approx(-5.0)


def test_placeholders_off_by_default_and_a_missing_cache_never_fails(cfg, tmp_path):
    long = frame(unit_id=["a"], node_type=["company"], direction=[1.0], weight=[1.0], signed_weight=[1.0])
    out, note = placeholders.direction_finbert(long, cfg, str(tmp_path))
    assert out.equals(long) and note["ran"] is False
    on = cfg.with_overrides({"clean.placeholders.use_direction_finbert": True, "clean.placeholders.use_cause_repair_embed": True, "clean.placeholders.use_world_themes_embed": True})
    out, note = placeholders.direction_finbert(long, on, str(tmp_path))
    assert out.equals(long) and note["ran"] is False and note["switch"] is True
    items = frame(unit_id=["a"], n_pins=[0], split=["fit"], event_type_final=["asset_event"], cause_top=["other"])
    assert placeholders.cause_repair_embed(items, on, str(tmp_path))[1]["ran"] is False
    assert placeholders.world_themes_embed(items, on, str(tmp_path))[1]["ran"] is False
