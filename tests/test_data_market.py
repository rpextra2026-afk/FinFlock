"""Prices to returns, volume, market cap, S&P 500 point-in-time membership, layers, schemas."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finfm.common.timing import TradingCalendar
from finfm.data.market import (
    action_panel,
    dollar_volume,
    future_split_factor,
    last_trading_dates,
    market_cap,
    shares_panel,
    suspect_returns,
    total_returns,
    volume_as_traded,
    wide,
)
from finfm.data.schema import PX_DAILY, SchemaError, validate
from finfm.data.sp500 import (
    DataQualityError,
    enforce_mismatch_limit,
    members_on,
    membership_intervals,
    mismatch_report,
    universe_asof,
)
from finfm.data.store import Store
from finfm.data.universe import (
    assign_layers,
    build_universe_table,
    eligible_etfs,
    liquid_flag,
    monthly_rebalance_dates,
)

SESSIONS = pd.bdate_range("2024-01-02", periods=8)


def panel(values, columns=("A",)):
    return pd.DataFrame(values, index=SESSIONS[: len(values)], columns=list(columns), dtype=float)


def actions(rows):
    return pd.DataFrame(rows, columns=["security_id", "ex_date", "action_type", "ratio", "amount", "announce_date", "source"])


# --- returns --------------------------------------------------------------------------------


def test_plain_returns_without_actions():
    close = panel([[100.0], [110.0], [99.0]])
    open_ = panel([[100.0], [105.0], [100.0]])
    c, o = total_returns(close, open_, actions([]))
    assert c.iloc[1, 0] == pytest.approx(0.10) and c.iloc[2, 0] == pytest.approx(-0.10)
    assert o.iloc[0, 0] == pytest.approx(0.05) and o.iloc[1, 0] == pytest.approx(100 / 105 - 1)
    assert np.isnan(c.iloc[0, 0]) and np.isnan(o.iloc[2, 0])


def test_split_is_value_neutral_and_dividend_is_added_on_the_ex_date():
    close = panel([[100.0], [50.0], [51.0]])  # 2-for-1 split on the second session
    open_ = panel([[100.0], [50.0], [50.0]])
    acts = actions([("A", SESSIONS[1], "split", 2.0, None, None, "t")])
    c, o = total_returns(close, open_, acts)
    assert c.iloc[1, 0] == pytest.approx(0.0)  # a split is not a return
    assert o.iloc[0, 0] == pytest.approx(0.0)  # holding from open 0 through the ex-date open
    div = actions([("A", SESSIONS[2], "dividend", None, 2.0, None, "t")])
    c2, o2 = total_returns(close, open_, div)
    assert c2.iloc[2, 0] == pytest.approx((51 + 2) / 50 - 1)
    assert o2.iloc[1, 0] == pytest.approx((50 + 2) / 50 - 1)  # holder at open 1 receives the ex-date 2 dividend


def test_gaps_are_not_bridged():
    close = panel([[100.0], [np.nan], [110.0], [121.0]])
    c, _ = total_returns(close, close, actions([]))
    assert np.isnan(c.iloc[1, 0]) and np.isnan(c.iloc[2, 0]) and c.iloc[3, 0] == pytest.approx(0.10)


def test_suspect_returns_are_flagged_not_clipped():
    flags = suspect_returns(panel([[0.5], [1.4], [-1.2]]), 1.0)
    assert list(flags.iloc[:, 0]) == [False, True, True]


# --- volume and dollar volume ---------------------------------------------------------------------


def test_vendor_split_adjusted_volume_is_rebuilt_to_as_traded():
    acts = actions([("A", SESSIONS[3], "split", 4.0, None, None, "t")])
    vendor = panel([[400.0], [400.0], [400.0], [100.0], [100.0]])  # old days inflated 4x by the vendor
    traded = volume_as_traded(vendor, acts, vendor_adjusts_volume=True)
    assert list(traded.iloc[:, 0]) == [100.0, 100.0, 100.0, 100.0, 100.0]
    assert volume_as_traded(vendor, acts, vendor_adjusts_volume=False).equals(vendor)
    assert list(future_split_factor(acts, vendor.index, vendor.columns).iloc[:, 0]) == [4, 4, 4, 1, 1]


def test_dollar_volume_uses_as_traded_close_and_volume_of_the_same_date():
    close, vol = panel([[50.0], [51.0]]), panel([[1000.0], [2000.0]])
    assert list(dollar_volume(close, vol).iloc[:, 0]) == [50000.0, 102000.0]


# --- shares and market cap ---------------------------------------------------------------------------


def test_market_cap_uses_filed_shares_known_by_t_minus_1_and_adjusts_for_later_splits():
    sessions = SESSIONS
    acts = actions([("A", sessions[5], "split", 2.0, None, None, "t")])
    shares_out = pd.DataFrame({
        "security_id": ["A"], "cover_date": [sessions[0] - pd.Timedelta(days=30)],
        "filed_utc": [pd.Timestamp(sessions[2], tz="UTC")], "shares": [1000.0], "accession": ["x"], "concept": ["c"],
    })
    shares = shares_panel(shares_out, acts, sessions, pd.Index(["A"]))
    assert shares.iloc[:3, 0].isna().all()  # not known until the session after filing
    assert shares.iloc[3, 0] == 1000.0 and shares.iloc[4, 0] == 1000.0
    assert shares.iloc[5, 0] == 2000.0  # split-adjusted from the cover date
    close = panel([[10.0]] * 8)
    cap = market_cap(close, shares)
    assert not np.isnan(cap.iloc[3, 0]) and cap.iloc[3, 0] == pytest.approx(10 * 1000)
    assert cap.iloc[4, 0] == pytest.approx(10 * 1000)


def test_a_later_filing_replaces_the_earlier_count_only_after_it_is_filed():
    shares_out = pd.DataFrame({
        "security_id": ["A", "A"], "cover_date": [SESSIONS[0], SESSIONS[0]],
        "filed_utc": [pd.Timestamp(SESSIONS[0], tz="UTC"), pd.Timestamp(SESSIONS[4], tz="UTC")],
        "shares": [100.0, 150.0], "accession": ["x", "y"], "concept": ["c", "c"],
    })
    shares = shares_panel(shares_out, actions([]), SESSIONS, pd.Index(["A"]))
    assert list(shares.iloc[1:5, 0]) == [100.0, 100.0, 100.0, 100.0]
    assert shares.iloc[5, 0] == 150.0


def test_last_trading_date_is_the_delisting_date():
    close = pd.DataFrame({"A": [1, 2, np.nan, np.nan], "B": [1, 2, 3, 4]}, index=SESSIONS[:4])
    last = last_trading_dates(close)
    assert last["A"] == SESSIONS[1] and last["B"] == SESSIONS[3]


def test_wide_panel_reindexes_to_sessions():
    long = pd.DataFrame({"security_id": ["A", "A", "B"], "session_date": [SESSIONS[0], SESSIONS[2], SESSIONS[0]],
                         "close_raw": [1.0, 2.0, 3.0]})
    out = wide(long, "close_raw", SESSIONS[:4])
    assert out.shape == (4, 2) and out.loc[SESSIONS[1]].isna().all() and out.loc[SESSIONS[2], "A"] == 2.0
    assert action_panel(actions([]), "split", "ratio", SESSIONS[:2], pd.Index(["A"]), 1.0).eq(1.0).all().all()


# --- S&P 500 point-in-time membership ------------------------------------------------------------


def snapshots():
    rows = []
    for stamp, members in [("2020-01-31", "ABC"), ("2020-02-28", "ABD"), ("2020-03-31", "BDE")]:
        rows += [(pd.Timestamp(stamp), m, "s") for m in members]
    return pd.DataFrame(rows, columns=["snapshot_date", "security_id", "source"])


def test_members_on_use_the_latest_snapshot_on_or_before_the_date():
    snaps = snapshots()
    assert members_on(snaps, pd.Timestamp("2020-02-15")).security_ids == frozenset("ABC")
    assert members_on(snaps, pd.Timestamp("2020-02-28")).security_ids == frozenset("ABD")
    before = members_on(snaps, pd.Timestamp("2019-12-31"))
    assert before.security_ids == frozenset() and before.snapshot_date is None
    assert members_on(snaps, pd.Timestamp("2020-02-15")).stale_days == 15


def test_a_removed_name_is_absent_from_the_removal_on_and_present_before():
    snaps = snapshots()
    assert "C" in members_on(snaps, pd.Timestamp("2020-02-27")).security_ids
    assert "C" not in members_on(snaps, pd.Timestamp("2020-02-28")).security_ids


def test_snapshots_dated_after_a_date_change_nothing_for_that_date():
    """Mutation test (spec 12.2 #20): tamper with every later snapshot, earlier answers stay identical."""
    snaps = snapshots()
    day = pd.Timestamp("2020-02-15")
    tampered = snaps.copy()
    later = pd.to_datetime(tampered["snapshot_date"]) > day
    tampered.loc[later, "security_id"] = "Z" + tampered.loc[later, "security_id"]
    extra = pd.DataFrame([(pd.Timestamp("2021-01-01"), "Q", "s")], columns=snaps.columns)
    assert members_on(pd.concat([tampered, extra]), day) == members_on(snaps, day)


def test_universe_is_applied_as_of_the_previous_session():
    cal = TradingCalendar("XNYS", "2020-01-01", "2020-12-31")
    snaps = snapshots()
    # 2020-03-02 is a Monday; the previous session (Friday 2020-02-28) already carries the new snapshot
    assert universe_asof(snaps, pd.Timestamp("2020-03-02"), cal).security_ids == frozenset("ABD")
    assert universe_asof(snaps, pd.Timestamp("2020-02-28"), cal).security_ids == frozenset("ABC")


def test_membership_intervals_view_matches_the_lookup():
    view = membership_intervals(snapshots(), pd.Timestamp("2020-12-31"))
    c = view[view["security_id"] == "C"].iloc[0]
    assert c["start_date"] == pd.Timestamp("2020-01-31") and c["end_date"] == pd.Timestamp("2020-02-27")
    e = view[view["security_id"] == "E"].iloc[0]
    assert e["start_date"] == pd.Timestamp("2020-03-31") and e["end_date"] == pd.Timestamp("2020-12-31")


def test_mismatch_report_and_the_ten_percent_stop():
    snaps = snapshots()
    reference = snaps.copy()
    report = mismatch_report(snaps, reference, matched_ids=set("ABCDE"))
    enforce_mismatch_limit(report, 0.10)  # perfect agreement: passes
    bad = mismatch_report(snaps, reference, matched_ids=set("AB"))
    with pytest.raises(DataQualityError):
        enforce_mismatch_limit(bad, 0.10)


# --- layers and universe table -------------------------------------------------------------------------


def test_layers_are_terciles_of_market_cap_and_unknown_caps_get_none():
    cap = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0, "d": 4.0, "e": 5.0, "f": 6.0, "g": np.nan})
    layer = assign_layers(cap, ["small", "mid", "large"])
    assert list(layer[["a", "b"]]) == ["small", "small"] and list(layer[["e", "f"]]) == ["large", "large"]
    assert pd.isna(layer["g"])


def test_monthly_rebalance_dates_are_first_sessions_of_the_month():
    sessions = pd.bdate_range("2024-01-01", "2024-03-31")
    assert [str(d.date()) for d in monthly_rebalance_dates(sessions)] == ["2024-01-01", "2024-02-01", "2024-03-01"]


def test_universe_table_has_layers_and_marks_unknown_market_cap(cfg):
    sessions = pd.bdate_range("2024-01-01", periods=90)
    ids = list("abcdef")
    rng = np.random.default_rng(0)
    close = pd.DataFrame(rng.uniform(10, 100, (90, 6)), index=sessions, columns=ids)
    dv = close * 1e6
    mcap = pd.DataFrame(np.tile(np.arange(1.0, 7.0), (90, 1)) * 1e9, index=sessions, columns=ids)
    mcap.loc[:, "f"] = np.nan
    date = sessions[70]
    table = build_universe_table({date: frozenset(ids)}, mcap, dv, close, cfg)
    assert set(table["security_id"]) == set(ids)
    row_f = table[table["security_id"] == "f"].iloc[0]
    assert not row_f["layer_known"] and pd.isna(row_f["layer"])
    assert table[table["layer_known"]]["layer"].nunique() >= 2


def test_liquid_flag_and_etf_eligibility(cfg):
    close = pd.Series({"a": 4.0, "b": 20.0, "c": 20.0})
    median_dv = pd.Series({"a": 5e6, "b": 5e5, "c": 5e6})
    assert list(liquid_flag(close, median_dv, cfg)) == [False, False, True]
    sessions = pd.bdate_range("2022-01-03", periods=400)
    dv = pd.DataFrame({"SPY": 5e8, "ETFa": 1e6, "ULTRA": 5e8, "NEW": 5e8}, index=sessions)
    names = pd.Series({"SPY": "Broad Market", "ETFa": "Thin Fund", "ULTRA": "UltraPro 3X Fund", "NEW": "New Fund"})
    first = pd.Series({"SPY": sessions[0], "ETFa": sessions[0], "ULTRA": sessions[0], "NEW": sessions[300]})
    keep = eligible_etfs(names, first, dv, sessions[399], sessions, cfg)
    assert keep == ["SPY"]  # thin, leveraged and too-new funds are out


# --- schemas and store --------------------------------------------------------------------------------


def test_schema_validation_catches_missing_columns_null_and_duplicate_keys():
    good = pd.DataFrame({c: [1] for c in PX_DAILY.columns})
    validate(good, PX_DAILY)
    with pytest.raises(SchemaError):
        validate(good.drop(columns=["close_raw"]), PX_DAILY)
    with pytest.raises(SchemaError):
        validate(pd.concat([good, good]), PX_DAILY)
    null_key = good.copy()
    null_key["security_id"] = None
    with pytest.raises(SchemaError):
        validate(null_key, PX_DAILY)


def test_store_roundtrip_validates_before_writing(tmp_path, cfg):
    store = Store(tmp_path, "zstd")
    frame = pd.DataFrame({c: [1] for c in PX_DAILY.columns})
    store.write("px_daily", frame, part="2024")
    assert store.exists("px_daily", "2024") and store.read("px_daily", "2024").equals(frame)
    with pytest.raises(SchemaError):
        store.write("px_daily", frame.drop(columns=["open"]))
    assert not list(tmp_path.rglob("*.tmp"))
