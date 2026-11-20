"""Building reader units from raw news inside a session: names, items, pins, events, samples, shards."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from finfm.common.config import load_config
from finfm.common.timing import TradingCalendar
from finfm.compute.session import RunSpec, SessionError
from finfm.data.names import build_aliases, drop_generic_aliases, shortened
from finfm.stage.reader_build import build_items, build_reader_units, cap_candidates
from finfm.stage.reader_stage import sources_needed
from finfm.stage.run_stage import find_market_dir, make_stager
from tests.conftest import CONFIG_DIR

CFG = load_config(CONFIG_DIR)
SMALL = CFG.with_overrides({"sample.tiers.S.db1.reader_events": 50, "sample.tiers.S.db1.framework_dev": 10,
                            "sample.tiers.S.db1.framework_test_cap": 10})



@pytest.fixture(autouse=True)
def no_fixed_shard_plans(tmp_path, monkeypatch):
    """These tests stage small synthetic tables under real run ids: the real rival shard plans (made for the real tables) must not apply."""
    import finfm.compute.shard as shard_module
    monkeypatch.setattr(shard_module, "PLAN_DIR", tmp_path / "no_plans")

def corpus(rows):
    return pd.DataFrame({"date": [r[0] for r in rows], "text": [r[1] for r in rows],
                         "extra_fields": [json.dumps({"stocks": str(r[2])}) for r in rows]})


def test_names_are_shortened_and_one_word_ordinary_words_are_dropped():
    assert shortened("Apple Inc.", CFG) == "Apple" and shortened("Jpmorgan Chase & Co", CFG) == "Jpmorgan Chase"
    sec = pd.DataFrame({"ticker": ["AAPL", "XYZ"], "title": ["Apple Inc.", "Block, Inc."]})
    aliases = build_aliases(sec, ["AAPL", "XYZ"], CFG)
    assert {"AAPL", "Apple Inc.", "Apple"} <= set(aliases.loc[aliases.node_id == "AAPL", "alias"])
    texts = pd.Series(["we block the road and block traffic"] * 30 + ["Block Inc reported"] * 5 + ["Apple unveiled"] * 30)
    kept = drop_generic_aliases(aliases, texts, CFG)
    assert "Block" not in set(kept["alias"]) and "Apple" in set(kept["alias"])


def test_ambiguous_one_word_names_are_dropped_but_full_names_remain():
    aliases = pd.DataFrame([
        ("NWS", "News", "name", None, None),
        ("NWS", "News Corp", "name", None, None),
        ("SO", "Southern", "name", None, None),
        ("SO", "Southern Co", "name", None, None),
        ("CPRI", "Capri", "name", None, None),
        ("CPRI", "Capri Holdings Ltd", "name", None, None),
        ("AAPL", "Apple", "name", None, None),
    ], columns=["node_id", "alias", "alias_type", "valid_from", "valid_to"])
    kept = drop_generic_aliases(aliases, pd.Series(["News and Southern skies; Apple reports earnings"]), CFG)
    assert set(kept["alias"]) == {"News Corp", "News Corporation", "Southern Co", "Southern Company",
                                  "Capri Holdings Ltd", "Capri Holdings", "Apple"}


def test_raw_news_becomes_dated_reader_units_with_company_pins():
    calendar = TradingCalendar.from_config(CFG)
    aliases = pd.DataFrame({"node_id": ["AAPL"] * 2, "alias": ["AAPL", "Apple"], "alias_type": ["ticker", "name"], "valid_from": None, "valid_to": None})
    rows = [(f"2018-03-{d:02d}T14:{m:02d}:00Z", f"Apple raises forecast number {d}-{m}\n\nApple said revenue would rise.", ["AAPL"])
            for d in range(1, 20) for m in (5, 40)]
    raw = {"yahoo_finance_felixdrinkall": corpus(rows)}
    items = cap_candidates(build_items(raw, "db1", SMALL), calendar, "db1", SMALL)
    units, used = build_reader_units(items, aliases, lambda node, date: True, pd.DataFrame({"iso": ["US"], "alias": ["United States"]}),
                                     pd.DataFrame({"tag": ["oil"], "keyword": ["crude oil"]}), calendar, "db1", SMALL)
    assert len(units) > 0 and set(units["kind"]) == {"finance"} and units["unit_id"].is_unique
    assert all("AAPL" in p for p in units["pins_json"]) and (units["available_date"] >= "2018-03-01").all()
    again, _ = build_reader_units(items, aliases, lambda node, date: True, pd.DataFrame({"iso": ["US"], "alias": ["United States"]}),
                                  pd.DataFrame({"tag": ["oil"], "keyword": ["crude oil"]}), calendar, "db1", SMALL)
    assert units.equals(again)  # the same inputs always give the same units, so every session sees the same shards


def test_db3_also_needs_the_db1_sources_for_cross_set_deduplication():
    assert set(sources_needed(CFG, "db1")) < set(sources_needed(CFG, "db3"))


def test_market_dataset_must_be_attached_and_only_readers_can_self_stage(tmp_path):
    with pytest.raises(SessionError, match="is not attached"):
        find_market_dir(tmp_path)
    (tmp_path / "x").mkdir()
    (tmp_path / "x" / "aliases.parquet").write_text("")
    assert find_market_dir(tmp_path) == tmp_path / "x"
    with pytest.raises(SessionError, match="not built yet"):
        make_stager(RunSpec("other-db1-llm2", "other", "db1", "llm2"), CONFIG_DIR, tmp_path)


def test_prebuilt_units_in_the_market_dataset_are_cut_into_shards_without_any_download(tmp_path):
    from finfm.compute.session import pick_shard
    from finfm.stage.reader_stage import save_prestaged, stage_job

    units = pd.DataFrame({"unit_id": [f"ev:{i}" for i in range(7)], "kind": "finance", "family": "ext_company_news", "available_date": "2018-03-01",
                          "title": "t", "text": "x", "pins_json": "[]", "available_utc": "2018-03-01 15:00:00+00:00", "est_seconds": 0.5})
    aliases = pd.DataFrame({"node_id": ["AAPL"], "alias": ["Apple"], "alias_type": ["name"], "valid_from": None, "valid_to": None})
    market = tmp_path / "market"
    market.mkdir()
    (market / "aliases.parquet").write_text("")
    save_prestaged(market, "reader", "db1", units, aliases)
    shards = stage_job("reader", "db1", "llm2", CFG, CONFIG_DIR, market, None, tmp_path / "shards")
    assert len(shards) == 1
    folder, manifest = pick_shard(tmp_path / "none", RunSpec("reader-db1-llm2", "reader", "db1", "llm2"), None, tmp_path / "shards")
    assert manifest["llm"] == "llm2" and len(manifest["unit_ids"]) == 7


def test_finripple_units_get_graph_text_from_known_links_and_are_cut_to_the_token_budget(tmp_path):
    from finfm.reader.prompt import WhitespaceTokenizer
    from finfm.stage.rival_stage import finripple_units, stage_finripple_units

    links = pd.DataFrame({"src_node": ["AAPL", "AAPL"], "dst_node": ["MSFT", "ORCL"], "link_type": "industry",
                          "available_from": pd.Timestamp("2018-01-01"), "valid_to": pd.NaT, "weight": [1.0, 0.5], "source": "sic",
                          "src_traded": True, "dst_traded": True, "carried_forward": False})
    framework = pd.DataFrame({"event_id": ["ev:1", "ev:2"], "kind": ["finance", "geo"], "title": ["Apple up", "Oil"], "text": ["body", "x"],
                              "available_utc": [pd.Timestamp("2018-03-01 15:00", tz="UTC")] * 2,
                              "pins_json": ['[{"node_id": "AAPL", "node_type": "company"}]'] * 2})
    names = {"AAPL": "Apple Inc", "MSFT": "Microsoft", "ORCL": "Oracle"}
    units = finripple_units(framework, links, names, CFG)
    assert list(units["unit_id"]) == ["ev:1"] and "Apple Inc (AAPL)" in units["kg_lines_json"].iloc[0]
    folders = stage_finripple_units("db1", "llm2", units, WhitespaceTokenizer(), CFG.with_overrides({"rivals.finripple.kg_tokens": 4}), tmp_path)
    staged = pd.read_parquet(folders[0] / "units.parquet")
    assert len(staged["kg_text"].iloc[0].split()) <= 4 and "news" in staged.columns
