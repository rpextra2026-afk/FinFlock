"""Persona panels per quarter from 13F tables, quarter-unique ids, payloads and states."""

from __future__ import annotations

import numpy as np
import pandas as pd

from oryxis.common.config import load_config
from oryxis.common.timing import TradingCalendar
from oryxis.stage.persona_build import (
    event_payloads,
    full_panels,
    institutional_panel,
    panel_for_date,
    persona_states,
    quarter_label,
)
from tests.conftest import CONFIG_DIR

OVERRIDES = {
    "personas.panel.min_managers_per_type": 6,
    "personas.types": {"retail": {"count": 16}, "mutual_pension": {"count": 2}, "hedge_fund": {"count": 2}, "bank_dealer": {"count": 2}}}
CFG = load_config(CONFIG_DIR).with_overrides(OVERRIDES)
SECTORS = pd.Series({"A": "Energy", "B": "Financials", "C": "Information Technology", "D": "Health Care", "E": "Energy", "F": "Financials"})
DOMAIN = {"Energy": "energy", "Financials": "banking", "Information Technology": "technology", "Health Care": "health"}
PERIODS = [pd.Timestamp("2019-09-30"), pd.Timestamp("2019-12-31")]


def tables(rng):
    managers, holdings = [], []
    for period in PERIODS:
        for t, kind in enumerate(["mutual_pension", "hedge_fund", "bank_dealer"]):
            for i in range(8):
                acc = f"{period:%Y%m}-{kind}-{i}"
                managers.append({"accession": acc, "manager_cik": f"{kind}{i}", "period": period, "type": kind, "book_usd": 1e9 * (i + 1),
                                 "turnover": 0.1 * (t + 1) + 0.01 * i, "n_positions": 50 + i, "passive_flag": False, "name": f"{kind} {i}"})
                w = rng.dirichlet(np.ones(6))
                holdings += [{"accession": acc, "security_id": s, "weight": float(x)} for s, x in zip(SECTORS.index, w, strict=True)]
    return pd.DataFrame(managers), pd.DataFrame(holdings)


def universe():
    return pd.DataFrame({"security_id": SECTORS.index, "sector": SECTORS.to_numpy(), "market_cap": 1e10, "uncovered_share": 0.5})


def build():
    managers, holdings = tables(np.random.default_rng(0))
    inst = institutional_panel(managers, holdings, SECTORS, DOMAIN, CFG)
    panels = full_panels(inst, {p: universe() for p in inst}, DOMAIN, CONFIG_DIR / "career_grid.csv", CFG)
    return inst, panels


def test_each_quarter_has_the_configured_personas_with_quarter_unique_ids():
    _, panels = build()
    assert set(panels) == {"2019Q3", "2019Q4"} and quarter_label(PERIODS[1], CFG) == "2019Q4"
    for label, panel in panels.items():
        ids = [s.persona_id for s in panel]
        assert len(ids) == len(set(ids)) == 16 + 6 and all(i.endswith(f"@{label}") for i in ids)
        assert {s.persona_type for s in panel} == {"retail", "mutual_pension", "hedge_fund", "bank_dealer"}
    assert build()[1]["2019Q4"][0].holdings == panels["2019Q4"][0].holdings  # deterministic


def test_a_quarter_is_used_only_after_its_filing_lag():
    _, panels = build()
    assert panel_for_date(pd.Timestamp("2019-11-01"), panels, CFG) is None  # Q3 filings not yet known (46 days after 30 Sep = 15 Nov)
    assert panel_for_date(pd.Timestamp("2019-11-20"), panels, CFG) == "2019Q3"
    assert panel_for_date(pd.Timestamp("2020-02-01"), panels, CFG) == "2019Q3"  # Q4 filings known from 15 Feb
    assert panel_for_date(pd.Timestamp("2020-02-20"), panels, CFG) == "2019Q4"


def test_payloads_and_states_cover_every_event_and_persona_of_the_panel_in_force():
    _, panels = build()
    calendar = TradingCalendar.from_config(CFG)
    sessions = calendar.sessions[(calendar.sessions >= "2019-06-01") & (calendar.sessions <= "2020-04-30")]
    returns = pd.DataFrame(np.random.default_rng(1).normal(0, 0.01, (len(sessions), 6)), index=sessions, columns=SECTORS.index)
    market = returns.mean(axis=1)
    events = pd.DataFrame({"event_id": ["ev:1", "ev:2"], "kind": "finance", "title": ["A rises", "B falls"], "text": ["x", "y"],
                           "available_utc": [pd.Timestamp("2019-12-02 15:00", tz="UTC"), pd.Timestamp("2020-02-03 15:00", tz="UTC")],
                           "pins_json": ['[{"node_id": "A", "node_type": "company", "sector": null, "layer": "mid", "priority": 5}]'] * 2})
    payloads = event_payloads(events, events, SECTORS, DOMAIN, {"A": ["E"]}, lambda d: list(SECTORS.index), pd.Series([3, 2, 1], index=["C", "B", "A"]), CFG)
    assert set(payloads) == {"ev:1", "ev:2"} and payloads["ev:1"].links[0]["node_id"] == "E" and not payloads["ev:2"].context  # the two events are far apart in time
    states, units = persona_states(events, panels, returns, market, calendar, CFG)
    assert len(units) == len(states) == 22 * 2 and units["unit_id"].is_unique
    assert all(k[1].endswith("@2019Q3") for k in states if k[0] == "ev:1")  # Dec 2: only Q3 filings are known
    assert all(k[1].endswith("@2019Q3") for k in states if k[0] == "ev:2")  # Feb 3: Q4 filings still unknown


def test_persona_shards_need_the_bios_bundle_and_then_carry_only_what_their_units_use(tmp_path):
    from oryxis.common.schema import UNIT_ID
    from oryxis.compute import payloads as pl
    from oryxis.compute.bundle import make_job_id
    from oryxis.compute.engine import MockEngine
    from oryxis.compute.runner import run_shard_from_folder
    from oryxis.compute.session import SessionError
    from oryxis.compute.shard import Shard
    from oryxis.compute.staging import stage_shard
    from oryxis.reader.prompt import WhitespaceTokenizer
    from oryxis.stage.persona_build import persona_states
    from oryxis.stage.persona_stage import bio_units, save_persona_inputs, stage_persona_job
    import pytest

    _, panels = build()
    skeletons = [s for panel in panels.values() for s in panel]
    calendar = TradingCalendar.from_config(CFG)
    sessions = calendar.sessions[(calendar.sessions >= "2019-06-01") & (calendar.sessions <= "2020-04-30")]
    returns = pd.DataFrame(0.001, index=sessions, columns=SECTORS.index)
    events = pd.DataFrame({"event_id": ["ev:1"], "kind": "finance", "title": ["A rises"], "text": ["x"], "available_utc": [pd.Timestamp("2019-12-02 15:00", tz="UTC")],
                           "pins_json": ['[{"node_id": "A", "node_type": "company", "sector": null, "layer": "mid", "priority": 5}]']})
    payloads = event_payloads(events, events, SECTORS, DOMAIN, {}, lambda d: list(SECTORS.index), pd.Series([1], index=["A"]), CFG)
    states, units = persona_states(events, panels, returns, returns.mean(axis=1), calendar, CFG)
    market = tmp_path / "market"
    save_persona_inputs(market / "prestaged" / "personas-db1", units.assign(est_seconds=0.15), payloads, skeletons, states)
    with pytest.raises(SessionError, match="no bios found"):
        stage_persona_job("db1", "llm2", CFG, market, tmp_path / "attached", tmp_path / "s0")
    # the bios notebook: stage and run the bios job with the mock engine, then attach its output
    bunits = bio_units(skeletons, CFG)
    bshard = Shard("s0000", make_job_id("bios", "shared", "llm2"), tuple(bunits[UNIT_ID]), 1.0, 2.0)
    folder = stage_shard(tmp_path / "bios", "bios", "shared", "llm2", bshard, bunits, lambda f: pl.save_skeletons(skeletons, f), CFG)
    engine = MockEngine()
    engine.register("bio", lambda prompt, rng: {"bio": "A fund manager in New York."})
    run_shard_from_folder(folder, tmp_path / "attached", CONFIG_DIR, engine=engine, tokenizer=WhitespaceTokenizer(),
                          overrides={**OVERRIDES, "kaggle.checkpoint_minutes": 0})
    shards = stage_persona_job("db1", "llm2", CFG, market, tmp_path / "attached", tmp_path / "s1")
    assert len(shards) == 1 and len(pl.load_bios(shards[0])) == len({s.bio_id for s in skeletons if s.persona_id.endswith("@2019Q3")})
