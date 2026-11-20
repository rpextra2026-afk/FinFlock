"""Event layer: kinds, timing, splits, clustering, adapters, pins, sampling, index events, coverage."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finfm.common.timing import TradingCalendar
from finfm.common.windows import bucket_windows
from finfm.data.buckets.adapters import (
    drop_cross_set_duplicates,
    era_of,
    items_to_events,
    normalise_source,
    normalised_title_hash,
)
from finfm.data.buckets.contract import (
    check_columns,
    finalize_events,
    load_contract,
    read_text,
)
from finfm.data.buckets.events import assign_day0, cluster_ids, hash_bucket, kind_of, time_known
from finfm.data.buckets.pins import event_pins_json, pin_name_only_items
from finfm.data.buckets.sampling import (
    build_samples,
    coverage_matrix,
    drop_index_events,
    ensure_multi_pin,
    hashed_order,
    index_event_dates,
    kinds_missing_by_year,
    largest_remainder,
    stratified_sample,
)
from finfm.data.schema import SchemaError
from finfm.reader.masking import MaskerCache
from tests.conftest import CONFIG_DIR

CAL = TradingCalendar("XNYS", "2012-01-01", "2025-12-31")


# --- kinds, timing, hash buckets --------------------------------------------------------------------


def test_kind_mapping_uses_exact_names_and_prefix_patterns(cfg):
    mapping = dict(cfg.get("events.family_kind"))
    assert kind_of("sec_8k", mapping) == "finance" and kind_of("gdelt_spike", mapping) == "geo"
    assert kind_of("ext_social", mapping) == "social" and kind_of("unknown_family", mapping) is None


def test_hash_bucket_is_stable_and_in_range():
    assert hash_bucket("ev:1") == hash_bucket("ev:1") and 0 <= hash_bucket("ev:2") < 100
    assert len({hash_bucket(f"e{i}") for i in range(500)}) > 80  # spreads over the buckets


def test_midnight_timestamps_are_date_only_and_use_that_dates_session(cfg):
    stamps = pd.Series(pd.to_datetime(["2024-03-12 00:00:00", "2024-03-12 14:30:00", "2024-03-16 00:00:00"], utc=True))
    known = time_known(stamps, cfg)
    assert list(known) == [False, True, False]
    events = pd.DataFrame({"available_utc": stamps})
    day0 = assign_day0(events, CAL, cfg)
    assert [str(d.date()) for d in day0] == ["2024-03-12", "2024-03-12", "2024-03-18"]


def test_eras_follow_what_data_exists(cfg):
    assert era_of(pd.Timestamp("2014-06-01"), cfg) == "E1"
    assert era_of(pd.Timestamp("2015-01-15"), cfg) == "E2" and era_of(pd.Timestamp("2020-01-01"), cfg) == "E3"


def raw_events(n=6):
    base = pd.Timestamp("2022-03-01 15:00:00", tz="UTC")
    return pd.DataFrame({
        "event_id": [f"e{i}" for i in range(n)],
        "available_utc": [base + pd.Timedelta(days=30 * i) for i in range(n)],
        "family": ["sec_8k"] * n, "kind": ["finance"] * n,
    })


def test_finalize_assigns_split_from_the_buckets_own_windows_and_drops_embargo(cfg):
    w1 = bucket_windows(cfg, "db1", CAL)
    events = raw_events(30)
    events.loc[0, "available_utc"] = pd.Timestamp("2018-06-01 15:00", tz="UTC")  # dev
    events.loc[1, "available_utc"] = pd.Timestamp("2022-01-04 15:00", tz="UTC")  # embargo (between dev and 2022-01-10)
    events.loc[2, "available_utc"] = pd.Timestamp("2016-01-04 15:00", tz="UTC")  # before db1's dev start
    kept, dropped = finalize_events(events, CAL, w1, cfg)
    by = kept.set_index("event_id")
    assert by.loc["e0", "split"] == "dev"
    assert "e1" not in by.index and "e2" not in by.index
    assert dropped["embargo"] >= 1 and dropped["outside"] >= 1
    assert set(kept["split"]) <= {"dev", "test"} and kept["hash_bucket"].between(0, 99).all()


def test_contract_loading_checks_columns_and_reads_local_text(tmp_path, cfg):
    events = raw_events(3).assign(
        member_item_ids=[["a"]] * 3, first_seen_utc=pd.Timestamp("2022-03-01", tz="UTC"), time_known=True, subtype="x",
        entities=[["cik:1"]] * 3, title="t", n_items=1, n_sources=1, best_tier="B", era="E3", flags=[[]] * 3,
        source_name="sec", trust_tier="high", text_path="e0.txt", session_day0=pd.Timestamp("2022-03-01"), split="dev",
    )
    items = pd.DataFrame({"item_id": ["a"], "source_name": ["sec"], "kind": ["finance"],
                          "available_utc": [pd.Timestamp("2022-03-01", tz="UTC")], "headline": ["h"], "family": ["sec_8k"]})
    pins = pd.DataFrame({"item_id": ["a"], "pin_type": ["company"], "pin_id": ["1"], "role": ["issuer"],
                         "confidence": ["high"], "sp500_member_on_date": [True]})
    for name, frame in (("events", events), ("items", items), ("pins", pins)):
        frame.to_parquet(tmp_path / f"{name}.parquet")
    loaded_events, _, _ = load_contract(tmp_path)
    assert len(loaded_events) == 3
    (tmp_path / "e0.txt").write_text("body")
    assert read_text("e0.txt", tmp_path) == "body" and read_text("missing.txt", tmp_path) == ""
    events.drop(columns=["family"]).to_parquet(tmp_path / "events.parquet")
    with pytest.raises(SchemaError):
        load_contract(tmp_path)
    with pytest.raises(SchemaError):
        check_columns(items, ["nope"], "items")


# --- clustering -----------------------------------------------------------------------------------


def stamp(minutes):
    return pd.Timestamp("2022-03-01 10:00:00", tz="UTC") + pd.Timedelta(minutes=minutes)


def test_finance_items_within_an_hour_of_the_first_join_one_cluster(cfg):
    frame = pd.DataFrame({"entity": ["A"] * 4 + ["B"], "kind": ["finance"] * 5,
                          "t": [stamp(0), stamp(30), stamp(59), stamp(61), stamp(10)]})
    ids = cluster_ids(frame, "entity", "kind", "t", cfg)
    assert ids[0] == ids[1] == ids[2] and ids[3] != ids[0] and ids[4] != ids[0]


def test_geo_window_is_a_day_and_social_clusters_by_entity_and_day(cfg):
    frame = pd.DataFrame({"entity": ["US"] * 3 + ["A"] * 3, "kind": ["geo"] * 3 + ["social"] * 3,
                          "t": [stamp(0), stamp(1300), stamp(1500), stamp(0), stamp(600), stamp(1500)]})
    ids = cluster_ids(frame, "entity", "kind", "t", cfg)
    assert ids[0] == ids[1] and ids[2] != ids[0]
    assert ids[3] == ids[4] and ids[5] != ids[3]


# --- adapters -------------------------------------------------------------------------------------


def corpus(dates, titles, bodies, stocks):
    """The 3-column Hugging Face corpus format: ``date``, ``text`` (title, blank line, body), ``extra_fields`` (JSON)."""
    import json
    texts = [t + ("\n\n" + b if b else "") for t, b in zip(titles, bodies, strict=True)]
    return pd.DataFrame({"date": dates, "text": texts, "extra_fields": [json.dumps({"stocks": str(st)}) for st in stocks]})


def test_each_source_is_normalised_to_utc_items_and_bad_records_are_dropped(cfg):
    reddit = corpus(["2022-03-01T10:20:00Z", None, "2022-03-01T11:20:00Z"], ["Stock jumps on earnings", "ok title", "x"], ["body", "", ""],
                    [[], [], []])
    out = normalise_source(reddit, "reddit_finance_sp500", cfg)
    assert len(out) == 1 and out["title"].iloc[0] == "Stock jumps on earnings"  # row 2 has no time, row 3 a too-short title
    assert out["available_utc"].iloc[0] == pd.Timestamp("2022-03-01 10:20:00", tz="UTC") and out["text"].iloc[0] == "body"
    assert out["kind"].iloc[0] == "social" and out["time_precision"].iloc[0] == "second"
    benzinga = corpus(["2022-03-01T09:00:00Z", "2022-03-01T09:00:00Z"], ["Headline one", "Headline one"], ["", ""], [["AAPL"], ["MSFT"]])
    b = normalise_source(benzinga, "benzinga_6000stocks", cfg)
    assert len(b) == 1 and b["tickers"].iloc[0] == ["AAPL"] and b["text"].iloc[0] == "Headline one"  # same record twice kept once


def test_a_date_only_record_is_marked_as_such(cfg):
    nyt = corpus(["2018-05-01T00:00:00Z"], ["Trade talks resume"], ["a"], [[]])
    out = normalise_source(nyt, "nyt_headlines", cfg)
    assert out["time_precision"].iloc[0] == "date"


def test_guardian_files_keep_their_own_columns(cfg):
    guardian = pd.DataFrame({"id": ["g1"], "webPublicationDate": ["2019-04-02T08:30:00Z"], "webTitle": ["Markets steady"], "bodyText": ["text"]})
    out = normalise_source(guardian, "guardian", cfg)
    assert out["kind"].iloc[0] == "geo" and out["item_id"].iloc[0] == "guardian:g1"


def test_cross_set_duplicates_stay_in_the_lowest_set_number(cfg):
    one = pd.DataFrame({"title": ["Fed holds rates steady", "Oil rallies"]})
    three = pd.DataFrame({"title": ["FED HOLDS rates steady!", "Gold slips"]})
    out = drop_cross_set_duplicates({"db3": three, "db1": one}, cfg)
    assert list(out["db1"]["title"]) == ["Fed holds rates steady", "Oil rallies"]
    assert list(out["db3"]["title"]) == ["Gold slips"]
    assert normalised_title_hash("A  b!") == normalised_title_hash("a b")


def test_items_cluster_into_events_with_the_earliest_trustworthy_source(cfg):
    items = pd.DataFrame({
        "item_id": ["i1", "i2", "i3"], "source_name": ["wire", "blog", "wire"], "kind": ["finance"] * 3,
        "family": ["ext_company_news"] * 3, "tier": ["X2", "X1", "X2"], "trust_tier": ["low", "high", "medium"],
        "available_utc": [stamp(0), stamp(20), stamp(200)], "title": ["first", "second", "third"], "text": ["t1", "t2", "t3"],
    })
    entity = pd.Series(["A", "A", "A"], index=items.index)
    events = items_to_events(items, entity, cfg)
    assert len(events) == 2
    first = events.sort_values("available_utc").iloc[0]
    assert first["n_items"] == 2 and first["n_sources"] == 2 and first["best_tier"] == "X1"
    assert first["source_name"] == "blog" and first["trust_tier"] == "high" and first["title"] == "first"


# --- pins ---------------------------------------------------------------------------------------------


def test_name_only_items_are_pinned_by_aliases_countries_and_commodities(cfg):
    aliases = pd.DataFrame([("n1", "Acme Corp", "name", None, None), ("n1", "ACME", "ticker", None, None)],
                           columns=["node_id", "alias", "alias_type", "valid_from", "valid_to"])
    maskers = MaskerCache(aliases, label_prefix="C_", short_ticker_max_len=2)
    items = pd.DataFrame({"item_id": ["i1", "i2"], "available_date": ["2022-03-01", "2022-03-01"],
                          "title": ["Acme Corp cuts oil price forecast", "Quiet day"], "text": ["Exports to China fell; copper up.", "nothing"]})
    countries = pd.read_csv(CONFIG_DIR / "countries.csv")
    commodities = pd.read_csv(CONFIG_DIR / "commodity_keywords.csv")
    pins = pin_name_only_items(items, maskers, countries, commodities, lambda node, date: node == "n1", cfg)
    got = {(r.item_id, r.pin_type, r.pin_id) for r in pins.itertuples()}
    assert ("i1", "company", "n1") in got and ("i1", "country", "CN") in got and ("i1", "commodity", "copper") in got
    assert not any(r.item_id == "i2" for r in pins.itertuples())
    company = pins[pins["pin_type"] == "company"].iloc[0]
    assert company["sp500_member_on_date"] is True or bool(company["sp500_member_on_date"])
    assert pins[pins["pin_type"] == "country"]["sp500_member_on_date"].isna().all()
    assert set(pins["method"]) == {"rule"}


def test_parallel_pinning_gives_exactly_the_serial_result(cfg):
    aliases = pd.DataFrame([("n1", "Acme Corp", "name", None, None), ("n1", "ACME", "ticker", None, None), ("n2", "Borealis Inc", "name", None, None)],
                           columns=["node_id", "alias", "alias_type", "valid_from", "valid_to"])
    maskers = MaskerCache(aliases, label_prefix="C_", short_ticker_max_len=2)
    titles = ["Acme Corp and China", "Borealis Inc copper deal", "Quiet day", "Oil falls as $ACME rises", "Germany and Borealis Inc talk"]
    n = 60
    items = pd.DataFrame({"item_id": [f"i{k:03d}" for k in range(n)], "available_date": ["2022-03-01"] * n,
                          "title": [titles[k % len(titles)] for k in range(n)], "text": ["body text"] * n})
    countries, commodities = pd.read_csv(CONFIG_DIR / "countries.csv"), pd.read_csv(CONFIG_DIR / "commodity_keywords.csv")
    member = lambda node, date: node == "n1"  # noqa: E731
    serial = pin_name_only_items(items, maskers, countries, commodities, member, cfg.with_overrides({"events.pins.workers": 1}))
    parallel = pin_name_only_items(items, maskers, countries, commodities, member,
                                   cfg.with_overrides({"events.pins.workers": 3, "events.pins.min_items_per_worker": 1}))
    assert len(serial) > n and serial.equals(parallel)


def test_event_pins_prefer_issuers_map_ciks_to_securities_and_dedupe(cfg):
    pins = pd.DataFrame({"pin_type": ["company", "country", "company"], "pin_id": ["123", "US", "999"],
                         "role": ["issuer", "other", "other"], "confidence": ["high", "low", "medium"]})
    descriptors = {"sec1": {"sector": "Energy", "layer": "large"}}
    import json
    out = json.loads(event_pins_json(pins, {"123": "sec1"}, descriptors, cfg))
    assert [p["node_id"] for p in out] == ["sec1", "cik:999", "c:US"]
    assert out[0]["sector"] == "Energy" and out[0]["layer"] == "large" and out[1]["layer"] == "non_sp"


# --- sampling ------------------------------------------------------------------------------------------


def pool(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    families = ["sec_8k"] * 5 + ["fr_trade", "ofac_sanctions", "ext_social"]
    frame = pd.DataFrame({
        "event_id": [f"e{i}" for i in range(n)], "family": rng.choice(families, n),
        "era": "E3", "split": rng.choice(["dev", "test"], n, p=[0.8, 0.2]), "n_pins": rng.choice([1, 2, 3], n, p=[0.8, 0.15, 0.05]),
    })
    return frame


def test_largest_remainder_allocates_exactly_and_respects_sizes():
    sizes = pd.Series({"a": 50, "b": 30, "c": 20, "d": 2})
    alloc = largest_remainder(sizes, 40)
    assert alloc.sum() == 40 and (alloc <= sizes).all() and alloc["a"] >= alloc["b"] >= alloc["c"]
    assert largest_remainder(sizes, 1000).sum() == 102  # cannot take more than exists


def test_stratified_sample_is_reproducible_and_independent_of_row_order(cfg):
    frame = pool()
    a = stratified_sample(frame, 300, "seed", 50)
    b = stratified_sample(frame.sample(frac=1.0, random_state=3), 300, "seed", 50)
    assert set(a["event_id"]) == set(b["event_id"]) and len(a) == 300
    assert set(a["event_id"]) != set(stratified_sample(frame, 300, "other", 50)["event_id"])
    per_family = a.groupby("family").size()
    assert (per_family >= 50).all()  # the per-family minimum is honoured


def test_multi_pin_share_is_raised_to_the_minimum_when_the_pool_allows(cfg):
    frame = pool()
    chosen = stratified_sample(frame, 400, "s", 10)
    chosen = chosen[chosen["n_pins"] < 2].head(300).pipe(lambda c: pd.concat([c, chosen[chosen["n_pins"] >= 2].head(20)]))
    out = ensure_multi_pin(chosen, frame, 0.30, "s")
    assert len(out) == len(chosen) and (out["n_pins"] >= 2).mean() >= 0.30 - 1e-9
    empty_pool = frame[frame["n_pins"] < 2]
    unchanged = ensure_multi_pin(chosen[chosen["n_pins"] < 2], empty_pool, 0.30, "s")
    assert (unchanged["n_pins"] >= 2).sum() == 0  # nothing to swap in


def test_framework_sample_is_a_subset_of_the_reader_sample_with_dev_sampled_and_test_capped(cfg):
    frame = pool(3000)
    reader, framework = build_samples(frame, 1500, 200, 150, cfg, "seed")
    assert len(reader) == 1500 and set(framework["event_id"]) <= set(reader["event_id"])
    assert (framework["split"] == "dev").sum() == 200 and (framework["split"] == "test").sum() <= 150
    assert (framework["n_pins"] >= 2).mean() >= 0.30 - 1e-9
    assert not (frame[frame["split"] == "embargo"]).shape[0]
    again = build_samples(frame.sample(frac=1.0, random_state=1), 1500, 200, 150, cfg, "seed")
    assert set(again[1]["event_id"]) == set(framework["event_id"])


def test_hashed_order_depends_on_seed_and_id_only():
    ids = pd.Series(["a", "b", "c"])
    assert list(hashed_order(ids, "s")) == list(hashed_order(ids.iloc[::-1].reset_index(drop=True), "s")[::-1])
    assert list(hashed_order(ids, "s")) != list(hashed_order(ids, "t"))


# --- index events ----------------------------------------------------------------------------------------


def test_index_changes_are_read_from_consecutive_snapshots_and_events_near_them_are_dropped(cfg):
    snaps = pd.DataFrame([(pd.Timestamp("2022-01-31"), m, "s") for m in "AB"] + [(pd.Timestamp("2022-06-30"), m, "s") for m in "BC"],
                         columns=["snapshot_date", "security_id", "source"])
    changes = index_event_dates(snaps)
    assert set(map(tuple, changes[["security_id", "change"]].values)) == {("C", "addition"), ("A", "deletion")}
    events = pd.DataFrame({"event_id": ["near", "far", "other"],
                           "session_day0": [pd.Timestamp("2022-06-28"), pd.Timestamp("2022-03-01"), pd.Timestamp("2022-06-28")]})
    pinned = {"near": {"C"}, "far": {"C"}, "other": {"B"}}
    kept, by_year = drop_index_events(events, pinned, changes, CAL, cfg)
    assert list(kept["event_id"]) == ["far", "other"] and by_year.to_dict() == {2022: 1}


# --- coverage ----------------------------------------------------------------------------------------------


def test_coverage_matrix_and_declared_holes(cfg):
    events = pd.DataFrame({
        "kind": ["finance", "finance", "geo", "social"], "source_name": ["a", "a", "b", "c"],
        "session_day0": pd.to_datetime(["2020-01-02", "2021-01-04", "2020-02-03", "2021-02-01"]),
    })
    matrix = coverage_matrix(events)
    assert matrix["events"].sum() == 4
    missing = kinds_missing_by_year(matrix, ["finance", "geo", "social"], [2020, 2021])
    assert missing == {2020: ["social"], 2021: ["geo"]}
