"""Stricter candidate assets for name-only news."""
from __future__ import annotations

import json

import pandas as pd
import pytest

from finflock.data.pin_cleanup import clean_pins, restore_distinctive_names, ticker_in_brackets
from finflock.reader.masking import MaskerCache

ALIASES = pd.DataFrame([
    ("TGT", "TGT", "ticker", None, None), ("TGT", "Target Corporation", "name", None, None), ("TGT", "Target", "name", None, None),
    ("TSLA", "TSLA", "ticker", None, None), ("TSLA", "Tesla", "name", None, None),
    ("MRK", "MRK", "ticker", None, None), ("MRK", "Merck & Co", "name", None, None),
    ("GS", "GS", "ticker", None, None), ("GS", "Goldman Sachs", "name", None, None),
    ("V", "V", "ticker", None, None), ("V", "Visa Inc", "name", None, None), ("V", "Visa", "name", None, None),
], columns=["node_id", "alias", "alias_type", "valid_from", "valid_to"])
PREFIX = {"country": "c:", "commodity": "m:", "etf": "e:", "company": "cik:"}
COUNTRIES = pd.DataFrame([("US", "United States"), ("US", "U.S."), ("RU", "Russia")], columns=["iso", "alias"])
COMMODITIES = pd.DataFrame([("gold", "gold"), ("oil", "crude oil")], columns=["tag", "keyword"])
# the generic-name heuristic has already dropped Target and Visa; Tesla was dropped by mistake and is restored
PROMPT = ALIASES[~ALIASES["alias"].isin(["Target", "Visa", "Tesla"])].reset_index(drop=True)


def pin(node, kind):
    return {"node_id": node, "node_type": kind, "layer": "sp500", "priority": 5.0, "sector": None}


def run(title, text, pins, family="ext_world_news"):
    row = {"unit_id": "u1", "family": family, "title": title, "text": text, "available_date": "2019-01-02",
           "pins_json": json.dumps(pins)}
    maskers = MaskerCache(restore_distinctive_names(ALIASES, PROMPT), label_prefix="COMPANY_", short_ticker_max_len=2)
    cleaned, removed = clean_pins(pd.DataFrame([row]), maskers, COUNTRIES, COMMODITIES, PREFIX)
    return [p["node_id"] for p in json.loads(cleaned.iloc[0]["pins_json"])], dict(zip(removed["node_id"], removed["reason"]))


def test_ordinary_word_does_not_make_a_company_candidate():
    kept, gone = run("Federal officers targeted in protests", "They were targeted with Target-style signs.", [pin("TGT", "company")])
    assert kept == [] and gone == {"TGT": "generic_name_only"}


def test_full_name_and_ticker_keep_the_company():
    assert run("Target Corporation raises dividend", "x", [pin("TGT", "company")])[0] == ["TGT"]
    assert run("Retailer news", "Shares of the retailer (TGT) rose.", [pin("TGT", "company")])[0] == ["TGT"]


def test_distinctive_name_flagged_by_the_heuristic_is_restored():
    assert run("Tesla recalls Model 3", "x", [pin("TSLA", "company")])[0] == ["TSLA"]


def test_ticker_in_brackets_rescues_one_letter_ticker():
    assert ticker_in_brackets("Will Visa (V) Q4 Earnings Buoy Up?", "V")
    assert ticker_in_brackets("Visa (NYSE: V) rose", "V")
    assert not ticker_in_brackets("a work visa program", "V")
    assert run("Will Visa (V) earnings rise?", "x", [pin("V", "company")])[0] == ["V"]
    assert run("The federal government changed work visa rules", "x", [pin("V", "company")])[0] == []


def test_rating_issuer_is_dropped_but_the_rated_company_is_kept():
    kept, gone = run("Goldman Sachs Upgrades Merck & Co to Buy", "x", [pin("GS", "company"), pin("MRK", "company")],
                     family="ext_company_news")
    assert kept == ["MRK"] and gone == {"GS": "rating_issuer_only"}


def test_a_company_that_is_the_subject_of_a_raise_headline_is_kept():
    assert run("Tesla Raises Prices On Model 3", "x", [pin("TSLA", "company")], family="ext_company_news")[0] == ["TSLA"]


def test_country_and_commodity_need_a_visible_mention():
    kept, gone = run("A quiet town", "Nothing relevant here.", [pin("c:RU", "country"), pin("m:gold", "commodity")])
    assert kept == [] and gone == {"c:RU": "no_visible_mention", "m:gold": "no_visible_mention"}
    kept, _ = run("Russia and gold", "x", [pin("c:RU", "country"), pin("m:gold", "commodity")])
    assert kept == ["c:RU", "m:gold"]


def test_social_rows_are_untouched():
    kept, gone = run("nothing", "nothing", [pin("TGT", "company")], family="ext_social")
    assert kept == ["TGT"] and gone == {}


def test_input_frame_is_not_modified():
    row = {"unit_id": "u1", "family": "ext_world_news", "title": "x", "text": "y", "available_date": "2019-01-02",
           "pins_json": json.dumps([pin("TGT", "company")])}
    frame = pd.DataFrame([row])
    maskers = MaskerCache(PROMPT, label_prefix="COMPANY_", short_ticker_max_len=2)
    clean_pins(frame, maskers, COUNTRIES, COMMODITIES, PREFIX)
    assert json.loads(frame.iloc[0]["pins_json"])[0]["node_id"] == "TGT"
