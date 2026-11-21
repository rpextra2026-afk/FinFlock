"""13F parsing rules, VALUE units, availability, weights, uncovered share, CUSIP matching, manager types."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from finflock.common.timing import TradingCalendar
from finflock.data.f13 import (
    choose_unit_factor,
    holdings_weights,
    latest_filing_per_manager,
    match_cusips,
    normalise_name,
    parse_filings,
    parse_infotable,
    similarity,
    uncovered_share,
)
from finflock.data.managers import (
    classify_managers,
    exclude_low_cover,
    passive_flags,
    quarterly_turnover,
)

CAL = TradingCalendar("XNYS", "2013-01-01", "2025-12-31")
FMT = "%d-%b-%Y"


def submission(rows):
    return pd.DataFrame(rows, columns=["ACCESSION_NUMBER", "FILING_DATE", "SUBMISSIONTYPE", "CIK", "PERIODOFREPORT"])


# --- filings ------------------------------------------------------------------------------------


def test_only_original_13f_hr_are_kept_and_the_rest_is_counted(cfg):
    sub = submission([
        ("a1", "14-FEB-2020", "13F-HR", 1, "31-DEC-2019"),
        ("a2", "20-FEB-2020", "13F-HR/A", 1, "31-DEC-2019"),
        ("a3", "14-FEB-2020", "13F-NT", 2, "31-DEC-2019"),
    ])
    kept, summary = parse_filings(sub, cfg, CAL)
    assert list(kept["accession"]) == ["a1"]
    assert (summary.kept, summary.skipped_amendments, summary.skipped_other_types) == (1, 1, 1)


def test_holdings_become_available_the_session_after_the_filing_date(cfg):
    sub = submission([
        ("fri", "14-FEB-2020", "13F-HR", 1, "31-DEC-2019"),  # Friday
        ("mon", "17-FEB-2020", "13F-HR", 2, "31-DEC-2019"),  # Presidents' Day, market closed
        ("tue", "11-FEB-2020", "13F-HR", 3, "31-DEC-2019"),
    ])
    kept, _ = parse_filings(sub, cfg, CAL)
    kept = kept.set_index("accession")
    assert kept.loc["fri", "available_from_session"] == pd.Timestamp("2020-02-18")
    assert kept.loc["mon", "available_from_session"] == pd.Timestamp("2020-02-18")
    assert kept.loc["tue", "available_from_session"] == pd.Timestamp("2020-02-12")


def test_latest_filing_never_uses_one_that_is_not_yet_available(cfg):
    sub = submission([
        ("old", "14-FEB-2020", "13F-HR", 1, "31-DEC-2019"),
        ("new", "14-MAY-2020", "13F-HR", 1, "31-MAR-2020"),
    ])
    kept, _ = parse_filings(sub, cfg, CAL)
    before = latest_filing_per_manager(kept, pd.Timestamp("2020-05-14"))  # filed that day: not usable yet
    after = latest_filing_per_manager(kept, pd.Timestamp("2020-05-15"))
    assert list(before["accession"]) == ["old"] and list(after["accession"]) == ["new"]
    assert latest_filing_per_manager(kept, pd.Timestamp("2020-01-02")).empty


# --- VALUE unit ----------------------------------------------------------------------------------


def test_unit_factor_is_chosen_per_row_from_the_implied_price():
    # the same stock (close 100) in an old filing (thousands) and a new one (dollars)
    value = np.array([100.0, 100_000.0, 5.0])
    shares = np.array([1000.0, 1000.0, 50.0])
    close = np.array([100.0, 100.0, np.nan])
    factor = choose_unit_factor(value, shares, close, [1.0, 1000.0])
    assert list(factor[:2]) == [1000.0, 1.0] and np.isnan(factor[2])


def infotable(rows):
    cols = ["ACCESSION_NUMBER", "INFOTABLE_SK", "NAMEOFISSUER", "CUSIP", "VALUE", "SSHPRNAMT", "SSHPRNAMTTYPE", "PUTCALL", "INVESTMENTDISCRETION"]
    return pd.DataFrame(rows, columns=cols)


def test_infotable_drops_options_and_non_shares_fills_unit_from_filing_majority(cfg):
    filings, _ = parse_filings(submission([("x", "14-FEB-2020", "13F-HR", 1, "31-DEC-2019")]), cfg, CAL)
    period = pd.Timestamp("2019-12-31")
    closes = pd.Series({("C1", period): 100.0, ("C2", period): 50.0}).rename_axis(["cusip9", "period"])
    closes.index = pd.MultiIndex.from_tuples(closes.index)
    info = infotable([
        ("x", 1, "ACME", "C1", 100.0, 1000.0, "SH", "", "SOLE"),  # thousands
        ("x", 2, "BETA", "C2", 50.0, 1000.0, "SH", "", "SOLE"),  # thousands
        ("x", 3, "GAMMA", "C3", 80.0, 800.0, "SH", "", "SOLE"),  # no price: gets the filing majority (1000)
        ("x", 4, "ACME", "C1", 100.0, 1000.0, "SH", "Call", "SOLE"),  # option row
        ("x", 5, "NOTE", "C4", 10.0, 10.0, "PRN", "", "SOLE"),  # not shares
        ("zz", 6, "OTHER", "C1", 1.0, 1.0, "SH", "", "SOLE"),  # filing not kept
    ])
    out, counts = parse_infotable(info, filings, closes, cfg)
    assert sorted(out["infotable_sk"]) == [1, 2, 3]
    assert list(out.sort_values("infotable_sk")["unit_factor"]) == [1000.0, 1000.0, 1000.0]
    assert out.set_index("infotable_sk").loc[3, "value_usd"] == 80_000.0
    assert counts["dropped_putcall"] == 1 and counts["dropped_non_sh"] == 1


def test_a_filing_with_no_priced_row_is_dropped_not_guessed_by_date(cfg):
    filings, _ = parse_filings(submission([("x", "14-FEB-2020", "13F-HR", 1, "31-DEC-2019")]), cfg, CAL)
    closes = pd.Series(dtype=float, index=pd.MultiIndex.from_tuples([], names=["cusip9", "period"]))
    out, counts = parse_infotable(infotable([("x", 1, "A", "C9", 10.0, 10.0, "SH", "", "SOLE")]), filings, closes, cfg)
    assert out.empty and counts["dropped_no_unit_factor"] == 1


# --- weights and uncovered share ---------------------------------------------------------------------


def test_weights_sum_to_one_inside_the_universe_and_sp_cover_counts_the_rest(cfg):
    holdings = pd.DataFrame({
        "accession": ["x"] * 3, "cusip9": ["C1", "C2", "C3"], "shares": [10.0, 20.0, 5.0],
        "close": [100.0, 50.0, 10.0], "value_usd": [1000.0, 1000.0, 50.0],
    })
    out = holdings_weights(holdings, pd.Series(dtype=float), pd.Series({"C1": "s1", "C2": "s2"}))
    assert out["weight"].sum() == pytest.approx(1.0) and set(out["security_id"]) == {"s1", "s2"}
    assert out["sp_cover"].iloc[0] == pytest.approx(2000 / 2050)
    assert out.set_index("security_id").loc["s1", "weight"] == pytest.approx(0.5)


def test_uncovered_share_is_clipped_and_flags_odd_ratios(cfg):
    held = pd.Series({"a": 50.0, "b": 1.0, "c": 200.0})
    out = uncovered_share(held, pd.Series({"a": 100.0, "b": 100.0, "c": 100.0, "d": 100.0}), cfg)
    assert out.loc["a", "uncovered_share"] == pytest.approx(0.5)
    assert out.loc["b", "uncovered_share"] == pytest.approx(0.95)  # capped
    assert out.loc["c", "uncovered_share"] == 0.0 and out.loc["c", "flagged"]  # held 200% of outstanding
    assert out.loc["d", "held_ratio"] == 0.0 and out.loc["d", "flagged"]  # nobody holds it: below 5%


# --- CUSIP matching -----------------------------------------------------------------------------------


def test_names_are_normalised_and_similar_names_score_high(cfg):
    suffixes = set(cfg.get("f13.name_match.suffixes"))
    assert normalise_name("Acme Holdings, Inc. Class A", suffixes) == "ACME A"
    a, b = normalise_name("ACME CORP", suffixes), normalise_name("Acme Corporation Inc.", suffixes)
    assert a == b and similarity(a, b) == 1.0
    assert similarity("ACME", "ZEBRA") < 0.2


def test_cusip_matching_accepts_close_names_and_sends_the_rest_to_review(cfg):
    issuers = pd.DataFrame({"cusip9": ["C1", "C2", "C3"], "issuer_name": ["ACME CORP", "Borealis Energy Inc", "Totally Different Ltd"]})
    targets = pd.DataFrame({"security_id": ["s1", "s2", "s2", "s9"],
                            "name": ["Acme Corporation", "Borealis Energy", "Former Borealis Oil", "Zebra Stripes"]})
    accepted, review = match_cusips(issuers, targets, cfg)
    assert dict(zip(accepted["cusip9"], accepted["security_id"], strict=True)) == {"C1": "s1", "C2": "s2"}
    assert list(review["cusip9"]) == ["C3"]


# --- managers ---------------------------------------------------------------------------------------------


def test_manager_types_follow_the_rules_in_order_and_overrides_win(cfg):
    managers = pd.DataFrame({
        "name": ["First National Bank", "State Teachers Retirement System", "Vanguard Group", "Vanguard Small Fund",
                 "Quant Partners", "Steady Capital", "Churn Capital", "Override Me Bank"],
        "n_positions": [500, 800, 3000, 40, 60, 300, 300, 50],
        "turnover": [0.2, 0.05, 0.03, 0.6, 0.9, 0.1, 0.8, 0.4],
    }, index=list("abcdefgh"))
    types = classify_managers(managers, cfg, overrides={"h": "hedge_fund"})
    assert types["a"] == "bank_dealer" and types["b"] == "mutual_pension"
    assert types["c"] == "mutual_pension"  # asset manager keyword with many positions
    assert types["d"] == "hedge_fund"  # same keyword but only 40 positions -> falls through to few positions
    assert types["e"] == "hedge_fund" and types["g"] == "hedge_fund"  # few positions / top turnover
    assert types["f"] == "mutual_pension" and types["h"] == "hedge_fund"


def test_passive_flags_and_cover_exclusion():
    flags = passive_flags(pd.Series(["Vanguard Group Inc", "Tiger Global", "State Street Corp"]), ["VANGUARD", "STATE STREET"])
    assert list(flags) == [True, False, True]
    assert list(exclude_low_cover(pd.Series([0.1, 0.2, 0.9]), 0.2)) == [False, True, True]


def test_turnover_does_not_count_a_split_as_trading():
    previous = pd.Series({"a": 100.0, "b": 50.0})
    current = pd.Series({"a": 200.0, "b": 70.0})  # a split 2-for-1; b bought 20 more
    prices = pd.Series({"a": 10.0, "b": 10.0})
    split = pd.Series({"a": 2.0})
    book = 200 * 10 + 70 * 10
    assert quarterly_turnover(previous, current, prices, split, book) == pytest.approx(20 * 10 / (2 * book))
    assert np.isnan(quarterly_turnover(previous, current, prices, split, 0.0))
