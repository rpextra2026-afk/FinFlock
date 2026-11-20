"""Reader: masking, prompt building, output validation and the processor (spec 12.2 #6, #7)."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from finfm.common.schema import (
    AVAILABLE_DATE,
    FAMILY,
    KIND,
    PARSE_OK,
    PINS_JSON,
    REASON,
    STATUS,
    TEXT,
    TITLE,
    UNIT_ID,
    Reason,
    Status,
)
from finfm.compute.engine import GenParams, MockEngine
from finfm.reader.masking import MaskerCache, Masker, label_for
from finfm.reader.mock import reader_responder
from finfm.reader.output import parse_reading, reader_json_schema
from finfm.reader.prompt import WhitespaceTokenizer, build_reader_prompt
from finfm.reader.run import CAUSE_PREFIX, READING_PINS, ReaderProcessor

ALIASES = pd.DataFrame(
    [
        ("n1", "Acme Corp", "name", None, None),
        ("n1", "ACME", "ticker", None, None),
        ("n1", "Acme Holdings", "former_name", None, "2019-12-31"),
        ("n2", "Borealis Energy", "name", None, None),
        ("n2", "BOR", "ticker", None, None),
        ("n3", "Zed Inc", "name", None, None),
        ("n3", "ZD", "ticker", None, None),  # two letters: cashtag only
    ],
    columns=["node_id", "alias", "alias_type", "valid_from", "valid_to"],
)
PREFIX, SHORT = "COMPANY_", 2


def masker(start="2021-01-01", end="2021-01-31") -> Masker:
    return Masker(
        ALIASES, pd.Timestamp(start), pd.Timestamp(end), label_prefix=PREFIX, short_ticker_max_len=SHORT
    )


# --- masking -------------------------------------------------------------------------------


def test_labels_run_a_to_z_then_aa():
    assert [label_for(i, "C_") for i in (0, 1, 25, 26, 27, 51, 52)] == [
        "C_A", "C_B", "C_Z", "C_AA", "C_AB", "C_AZ", "C_BA",
    ]


def test_names_tickers_and_cashtags_are_masked_in_order_of_first_appearance():
    result = masker().mask("Borealis Energy and Acme Corp (ACME) rose; $BOR up, acme corp again.")
    assert result.text == "COMPANY_A and COMPANY_B (COMPANY_B) rose; COMPANY_A up, COMPANY_B again."
    assert result.mapping == {"n2": "COMPANY_A", "n1": "COMPANY_B"}


def test_short_tickers_only_as_cashtags_and_words_are_untouched():
    result = masker().mask("ZD said the zd tool and $ZD shares; BORING and ACMEX are other words")
    assert "ZD said" in result.text and "zd tool" in result.text  # bare 2-letter ticker not masked
    assert "$ZD" not in result.text and "COMPANY_A shares" in result.text
    assert "BORING" in result.text and "ACMEX" in result.text  # word boundaries


def test_former_names_follow_point_in_time_validity():
    assert "Acme Holdings" not in masker("2019-06-01", "2019-06-30").mask("Acme Holdings fell").text
    assert "Acme Holdings" in masker("2021-01-01", "2021-01-31").mask("Acme Holdings fell").text


def test_countries_and_commodities_are_not_masked():
    text = "Acme Corp faces tariffs on China and copper imports"
    assert masker().mask(text).text == "COMPANY_A faces tariffs on China and copper imports"


def test_no_alias_survives_masking_on_fifty_generated_texts():
    templates = [
        "{n} reported results. Shares of {t} jumped.",
        "Analysts at the bank upgraded {n} after the call; ${t} gained.",
        "{n}'s CEO said guidance is unchanged ({t}).",
        "Rivals {n} and Borealis Energy both moved.",
        "Short seller targets {t} stock, says {n}.",
    ]
    pairs = [("Acme Corp", "ACME"), ("Borealis Energy", "BOR")]
    m = masker()
    for i in range(50):
        name, ticker = pairs[i % 2]
        text = templates[i % len(templates)].format(n=name, t=ticker)
        out = m.mask(text).text
        assert not any(alias in out for alias in ("Acme", "ACME", "Borealis", "BOR")), out


def test_masker_cache_builds_one_masker_per_month():
    cache = MaskerCache(ALIASES, label_prefix=PREFIX, short_ticker_max_len=SHORT)
    first = cache.for_date(pd.Timestamp("2021-01-05"))
    assert cache.for_date(pd.Timestamp("2021-01-28")) is first
    assert cache.for_date(pd.Timestamp("2021-02-01")) is not first


# --- prompt ------------------------------------------------------------------------------


def pins(*specs) -> str:
    return json.dumps(
        [
            {"node_id": n, "node_type": t, "sector": s, "layer": layer, "priority": p}
            for n, t, s, layer, p in specs
        ]
    )


def test_prompt_lists_allowed_assets_and_links_refs_to_masked_labels(cfg):
    prompt = build_reader_prompt(
        cfg=cfg, masker=masker(), tokenizer=WhitespaceTokenizer(), kind="finance", family="sec_8k",
        available_date="2021-01-12", title="Acme Corp beats estimates",
        text="Acme Corp beat estimates while Borealis Energy missed.",
        pins_json=pins(("n1", "company", "Technology", "large", 5), ("c:US", "country", None, None, 1)),
    )
    assert prompt.ref_to_node == {"A1": "n1", "A2": "c:US"}
    assert "A1: company, sector=Technology, size=large (= COMPANY_A)" in prompt.user
    assert "A2: country" in prompt.user
    assert "Acme" not in prompt.user and "Borealis" not in prompt.user
    assert "Published: 2021-01-12" in prompt.user
    assert prompt.user.count("Cause list:") == 1 and "earnings_beat" in prompt.user


def test_at_most_max_pins_are_kept_by_priority_and_counted(cfg):
    many = pins(*[(f"n{i}", "company", "X", "mid", i) for i in range(9)])
    prompt = build_reader_prompt(
        cfg=cfg, masker=masker(), tokenizer=WhitespaceTokenizer(), kind="finance", family="f",
        available_date="2021-01-12", title="t", text="body", pins_json=many,
    )
    assert len(prompt.ref_to_node) == int(cfg.get("reader.max_pins"))
    assert prompt.n_pins_dropped == 3
    assert set(prompt.ref_to_node.values()) == {f"n{i}" for i in range(8, 2, -1)}


def test_long_text_is_truncated_and_flagged(cfg):
    small = cfg.with_overrides({"reader.max_text_tokens": 5})
    prompt = build_reader_prompt(
        cfg=small, masker=masker(), tokenizer=WhitespaceTokenizer(), kind="finance", family="f",
        available_date="2021-01-12", title="t", text="one two three four five six seven",
        pins_json=pins(("n1", "company", "X", "mid", 1)),
    )
    assert prompt.text_truncated and "six" not in prompt.user


# --- output validation -----------------------------------------------------------------------


def good(cfg, **over):
    causes = list(cfg.get("causes.causes"))
    base = {
        "cause": {c: (1.0 if c == "earnings_beat" else 0.0) for c in causes},
        "pins": [
            {"ref": "A1", "direction": "up", "value_effect_pct": 4.0, "magnitude": 3, "role": "source"}
        ],
        "scheduled": "scheduled",
        "certainty": 0.7,
    }
    base.update(over)
    return json.dumps(base)


def test_valid_answer_parses_and_maps_refs_to_nodes(cfg):
    reading = parse_reading(good(cfg), {"A1": "n1"}, cfg)
    assert reading.ok and reading.pins[0]["node_id"] == "n1"
    assert reading.cause["earnings_beat"] == pytest.approx(1.0)
    assert not reading.flags.cause_renormalised


def test_probabilities_are_renormalised_and_flagged(cfg):
    causes = {c: 0.0 for c in cfg.get("causes.causes")}
    causes["earnings_beat"], causes["other"] = 3.0, 1.0
    reading = parse_reading(good(cfg, cause=causes), {"A1": "n1"}, cfg)
    assert reading.flags.cause_renormalised and sum(reading.cause.values()) == pytest.approx(1.0)
    assert reading.cause["earnings_beat"] == pytest.approx(0.75)


def test_hallucinated_refs_are_dropped_and_counted(cfg):
    pins_ = [
        {"ref": "A1", "direction": "up", "value_effect_pct": 1, "magnitude": 1, "role": "source"},
        {"ref": "A9", "direction": "up", "value_effect_pct": 1, "magnitude": 1, "role": "linked"},
        {"ref": "none", "direction": "none", "value_effect_pct": 0, "magnitude": 0, "role": "linked"},
    ]
    reading = parse_reading(good(cfg, pins=pins_), {"A1": "n1"}, cfg)
    assert [p["ref"] for p in reading.pins] == ["A1"] and reading.flags.hallucinated_refs == 1


def test_out_of_range_values_are_clamped_and_flagged(cfg):
    pins_ = [{"ref": "A1", "direction": "down", "value_effect_pct": -80, "magnitude": 9, "role": "source"}]
    reading = parse_reading(good(cfg, pins=pins_, certainty=3), {"A1": "n1"}, cfg)
    assert reading.pins[0]["value_effect_pct"] == -30 and reading.pins[0]["magnitude"] == 5
    assert reading.flags.value_clamped == 1 and reading.flags.certainty_clamped


@pytest.mark.parametrize(
    "text,reason",
    [
        ("not json", Reason.INVALID_JSON),
        ("[1, 2]", Reason.INVALID_JSON),
        (json.dumps({"cause": {}, "pins": []}), Reason.MISSING_KEY),
    ],
)
def test_unparseable_answers_become_failed_parses(cfg, text, reason):
    reading = parse_reading(text, {"A1": "n1"}, cfg)
    assert not reading.ok and reading.fail_reason is reason


def test_all_zero_cause_vector_puts_everything_on_the_fallback_cause_and_is_flagged(cfg):
    zero = {c: 0.0 for c in cfg.get("causes.causes")}
    reading = parse_reading(good(cfg, cause=zero), {"A1": "n1"}, cfg)
    fallback = str(cfg.get("reader.empty_cause_fallback"))
    assert reading.ok and reading.flags.cause_defaulted and reading.cause[fallback] == 1.0
    assert sum(reading.cause.values()) == 1.0 and len(reading.pins) == 1  # the pins of the same answer are kept
    normal = parse_reading(good(cfg), {"A1": "n1"}, cfg)
    assert normal.ok and not normal.flags.cause_defaulted


def test_schema_caps_the_pin_list_at_the_number_of_assets_that_can_be_offered(cfg):
    assert reader_json_schema(cfg)["properties"]["pins"]["maxItems"] == int(cfg.get("reader.max_pins"))


def test_the_prompt_tells_the_model_that_causes_add_up_to_one_and_other_is_the_fallback(cfg):
    system = str(cfg.get("prompts.reader.system"))
    assert "add up to 1" in system and "other" in system and "at most once" in system


def test_reparse_recovers_stored_answers_without_a_model_call_and_leaves_loops_failed(cfg):
    from finfm.common.schema import PARSE_OK, RAW_OUTPUT, STATUS, UNIT_ID
    from finfm.reader.reparse import reparse_tape

    zero = {c: 0.0 for c in cfg.get("causes.causes")}
    pins = json.dumps([{"node_id": "n1", "priority": 1.0}])
    tape = pd.DataFrame({UNIT_ID: ["a", "b", "c", "d"], STATUS: ["ok", "ok", "ok", "skipped"],
                         RAW_OUTPUT: [good(cfg, cause=zero), good(cfg), '{"cause": {"other": 1}, "pins": [{"ref": "A1"', None],
                         PARSE_OK: [False, True, False, False]})
    out = reparse_tape(tape, {u: pins for u in "abcd"}, cfg)
    assert list(out[PARSE_OK]) == [True, True, False, False]
    assert out.loc[0, "cause_other"] == 1.0 and out.loc[0, "flag_cause_defaulted"] and not out.loc[1, "flag_cause_defaulted"]
    assert out.loc[0, "reading_pins_json"] and out.loc[2, "reason"] == "invalid_json"
    assert tape.loc[0, PARSE_OK] is False or not tape.loc[0, PARSE_OK]  # the input frame is not modified


def test_json_schema_allows_made_up_refs_so_hallucinations_can_be_measured(cfg):
    schema = reader_json_schema(cfg)
    assert len(schema["properties"]["cause"]["properties"]) == 29
    pattern = schema["properties"]["pins"]["items"]["properties"]["ref"]["pattern"]
    assert "A" in pattern and "none" in pattern


# --- processor ---------------------------------------------------------------------------------


def events(*rows) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                UNIT_ID: uid, KIND: "finance", FAMILY: "sec_8k", AVAILABLE_DATE: "2021-01-12",
                TITLE: title, TEXT: text, PINS_JSON: pins_json,
            }
            for uid, title, text, pins_json in rows
        ]
    )


def processor(cfg) -> ReaderProcessor:
    engine = MockEngine()
    engine.register("reader", reader_responder)
    cache = MaskerCache(ALIASES, label_prefix=PREFIX, short_ticker_max_len=SHORT)
    return ReaderProcessor(cfg, engine, cache, WhitespaceTokenizer())


def test_processor_returns_one_row_per_event_and_skips_events_without_pins(cfg):
    one = pins(("n1", "company", "Technology", "large", 5))
    batch = events(
        ("e1", "Acme Corp beat estimates", "Acme Corp record quarter, raises outlook", one),
        ("e2", "Quiet day", "nothing happens", "[]"),
        ("e3", "Acme Corp recall", "Acme Corp announces recall and lawsuit, downgrade", one),
    )
    out = processor(cfg)(batch)
    assert list(out[UNIT_ID]) == ["e1", "e2", "e3"]
    by = out.set_index(UNIT_ID)
    assert by.loc["e2", STATUS] == Status.SKIPPED.value
    assert by.loc["e2", REASON] == Reason.NO_USABLE_PIN.value
    assert by.loc["e1", PARSE_OK] and by.loc["e3", PARSE_OK]
    first = json.loads(by.loc["e1", READING_PINS])[0]
    third = json.loads(by.loc["e3", READING_PINS])[0]
    assert first["value_effect_pct"] > 0 > third["value_effect_pct"]  # planted text dependence
    causes = [by.loc["e1", f"{CAUSE_PREFIX}{c}"] for c in cfg.get("causes.causes")]
    assert sum(causes) == pytest.approx(1.0)


def test_failed_parse_is_kept_as_a_row_with_parse_ok_false(cfg):
    engine = MockEngine()
    engine.register("reader", lambda prompt, rng: {"unexpected": 1})
    proc = ReaderProcessor(
        cfg, engine, MaskerCache(ALIASES, label_prefix=PREFIX, short_ticker_max_len=SHORT),
        WhitespaceTokenizer(),
    )
    out = proc(events(("e1", "t", "body", pins(("n1", "company", "X", "mid", 1)))))
    assert not out.loc[0, PARSE_OK] and out.loc[0, STATUS] == Status.OK.value
    assert out.loc[0, REASON] == Reason.MISSING_KEY.value


def test_processor_is_deterministic(cfg):
    batch = events(("e1", "Acme Corp beat", "Acme Corp beat", pins(("n1", "company", "X", "mid", 1))))
    first, second = processor(cfg)(batch), processor(cfg)(batch)
    assert first.equals(second)
    assert isinstance(GenParams, type)


def test_ref_map_used_for_reparsing_equals_the_one_shown_in_the_prompt(cfg):
    from finfm.reader.prompt import ref_to_node_of
    from finfm.reader.masking import MaskerCache
    from finfm.reader.prompt import WhitespaceTokenizer, build_reader_prompt

    pins = [{"node_id": f"n{i}", "node_type": "company", "layer": "non_sp", "priority": float(i)} for i in range(9)]
    masker = MaskerCache(pd.DataFrame({"node_id": ["n1"], "alias": ["N1"], "alias_type": ["ticker"], "valid_from": [None], "valid_to": [None]}),
                         label_prefix="C_", short_ticker_max_len=2).for_date(pd.Timestamp("2021-01-04"))
    prompt = build_reader_prompt(cfg=cfg, masker=masker, tokenizer=WhitespaceTokenizer(), kind="finance", family="f", available_date="2021-01-04",
                                 title="t", text="x", pins_json=json.dumps(pins))
    assert ref_to_node_of(json.dumps(pins), cfg) == prompt.ref_to_node and len(prompt.ref_to_node) == int(cfg.get("reader.max_pins"))


# --- state features and dial maps (spec 4.5): nothing from day 0 or later -------------------------------------------


def _panels(n=120, seed=3):
    import numpy as np

    rng = np.random.default_rng(seed)
    index = pd.bdate_range("2021-01-04", periods=n)
    ret = pd.DataFrame(rng.normal(0, 0.01, (n, 3)), index=index, columns=["a", "b", "SPY"])
    dv = pd.DataFrame(rng.uniform(1e6, 2e6, (n, 3)), index=index, columns=["a", "b", "SPY"])
    return ret, dv


def test_trailing_state_ignores_day_zero_and_everything_after(cfg):
    from finfm.data.market import trailing_state

    ret, dv = _panels()
    day0 = ret.index[80]
    before = trailing_state(ret, dv, ["a"], [day0], 20)
    changed_ret, changed_dv = ret.copy(), dv.copy()
    changed_ret.loc[day0:, "a"] += 0.5
    changed_dv.loc[day0:, "a"] *= 100
    assert trailing_state(changed_ret, changed_dv, ["a"], [day0], 20).equals(before)
    earlier = ret.copy()
    earlier.iloc[79, 0] += 0.5  # day -1 does count
    assert not trailing_state(earlier, dv, ["a"], [day0], 20).equals(before)


def test_market_state_terciles_use_only_earlier_history(cfg):
    from finfm.data.market import market_state_terciles

    ret, dv = _panels(200)
    day0 = ret.index[150]
    base = market_state_terciles(ret["SPY"], dv["SPY"], [day0], 20, 3, 60)
    shifted_ret, shifted_dv = ret.copy(), dv.copy()
    shifted_ret.loc[day0:, "SPY"] *= 50
    shifted_dv.loc[day0:, "SPY"] *= 50
    assert market_state_terciles(shifted_ret["SPY"], shifted_dv["SPY"], [day0], 20, 3, 60).equals(base)
    assert base.loc[0, "mkt_vol_tercile"] in (0, 1, 2)


def test_add_state_features_and_feature_matrix_have_the_configured_blocks_and_no_label_column(cfg):
    from finfm.reader.dials import feature_matrix
    from finfm.reader.state_features import STATE_COLUMNS, add_state_features

    ret, dv = _panels(200)
    causes = {f"cause_{c}": 1.0 / len(cfg.get("causes.causes")) for c in cfg.get("causes.causes")}
    pins = pd.DataFrame([{"event_id": "e1", "node_id": "a", "day0": ret.index[150], "value_effect_pct": 2.0, "magnitude": 3.0, "certainty": 0.7,
                          "kind": "finance", "scheduled": "surprise", **causes},
                         {"event_id": "e1", "node_id": "ctx", "day0": ret.index[150], "value_effect_pct": -1.0, "magnitude": 2.0, "certainty": 0.7,
                          "kind": "finance", "scheduled": "surprise", **causes}])
    with_state = add_state_features(pins, ret, dv, cfg)
    assert set(STATE_COLUMNS) <= set(with_state.columns) and not any("log_" in c for c in with_state.columns)
    assert with_state.loc[1, "own_vol_trail"] != with_state.loc[1, "own_vol_trail"]  # no prices for a context company: NaN, never invented
    X = feature_matrix(with_state, cfg)
    expected = len(cfg.get("causes.causes")) + 1 + 1 + 1 + 3 + 2 + 2 + 2  # causes, magnitude, |value|, certainty, kind, own, market, scheduled
    assert X.shape == (2, expected)
    with pytest.raises(KeyError, match="own_state"):
        feature_matrix(pins, cfg)


def test_dial_map_handles_missing_state_and_the_tree_arm_wins_only_with_a_lower_cv_loss(cfg, monkeypatch):
    import numpy as np

    from finfm.common.timing import TradingCalendar
    from finfm.reader import dials
    from finfm.reader.dials import DialMap, TreeMap, fit_dial_map

    rng = np.random.default_rng(0)
    n = 160
    X = rng.normal(size=(n, 4))
    X[::7, 3] = np.nan  # a missing state value
    y = X[:, 0] + 0.1 * rng.normal(size=n)
    dates = pd.Series(pd.bdate_range("2021-01-04", periods=n))
    calendar = TradingCalendar("XNYS", "2020-01-01", "2022-12-31")
    on = cfg.with_overrides({"reader.dials.lgbm.enabled": True})

    class FakeTree:
        def __init__(self, good):
            self.good = good

        def predict(self, X):
            return np.nan_to_num(X[:, 0]) if self.good else np.zeros(len(X))

    monkeypatch.setattr(dials, "fit_tree", lambda a, b, p, seed: TreeMap(FakeTree(True), len(b), p))
    assert isinstance(fit_dial_map(X, y, dates, calendar, on), TreeMap)  # a tree with a lower out-of-fold loss replaces the ridge
    monkeypatch.setattr(dials, "fit_tree", lambda a, b, p, seed: TreeMap(FakeTree(False), len(b), p))
    assert isinstance(fit_dial_map(X, y, dates, calendar, on), DialMap)  # a worse tree never replaces the ridge floor
    off = fit_dial_map(X, y, dates, calendar, cfg)
    assert isinstance(off, DialMap) and np.isfinite(off.predict(X)).all()  # arm off by default; NaN state values are filled
