"""Shard staging, payload round trips and the job runner (what a Kaggle session executes)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from oryxis.common.config import load_config
from oryxis.common.schema import (
    AVAILABLE_DATE,
    FAMILY,
    KIND,
    PINS_JSON,
    STATUS,
    TEXT,
    TITLE,
    UNIT_ID,
    ShardOutcome,
    Status,
)
from oryxis.compute import payloads as pl
from oryxis.compute.bundle import make_job_id, read_consolidated, scan_bundles
from oryxis.compute.engine import MockEngine
from oryxis.compute.jobs import planned_for_shard
from oryxis.compute.runner import run_shard_from_folder
from oryxis.compute.shard import Shard
from oryxis.compute.staging import read_manifest, stage_shard
from oryxis.personas.bios import template_bio
from oryxis.personas.feed import NewsItem
from oryxis.personas.mock import persona_responder
from oryxis.personas.run import EventPayload, PersonaState, unit_id
from oryxis.reader.mock import reader_responder
from oryxis.rivals.mass.modalities import ModalityStore
from oryxis.testing.pipeline import persona_skeletons
from oryxis.testing.synthetic import build_world, synthetic_config
from tests.conftest import CONFIG_DIR, FakeClock

OVERRIDES = {"kaggle.checkpoint_minutes": 0, "runtime.batch_units": 16, "runtime.canary_units": 100000,
             "kaggle.session_cutoff_minutes": 720}


@pytest.fixture(scope="module")
def world():
    return build_world(synthetic_config(load_config(CONFIG_DIR)).with_overrides(OVERRIDES), n_events=40)


@pytest.fixture
def cfg(world):
    return world.cfg


def stage_reader(world, root: Path, n=40):
    events = world.events.head(n)
    units = pd.DataFrame({
        UNIT_ID: events["event_id"], KIND: events["kind"], FAMILY: events["family"],
        AVAILABLE_DATE: pd.to_datetime(events["available_utc"]).dt.strftime("%Y-%m-%d"),
        TITLE: events["title"], TEXT: events["text"], PINS_JSON: events["pins_json"],
    })
    job_id = make_job_id("reader", "dbX", "llmY")
    shard = Shard("s0000", job_id, tuple(units[UNIT_ID]), 1.0, 2.0)
    return stage_shard(root, "reader", "dbX", "llmY", shard, units,
                       lambda folder: world.aliases.to_parquet(folder / pl.ALIASES, index=False), world.cfg), job_id


def run(world, folder, out, inputs=(), clock=None, overrides=None):
    engine = MockEngine()
    engine.register("reader", reader_responder)
    engine.register("persona", persona_responder)
    from oryxis.reader.prompt import WhitespaceTokenizer
    return run_shard_from_folder(folder, out, CONFIG_DIR, input_dirs=list(inputs), engine=engine, tokenizer=WhitespaceTokenizer(),
                                 clock=clock or FakeClock(), overrides={**OVERRIDES, **(overrides or {}), **_world_overrides(world)})


def _world_overrides(world):
    cfg = world.cfg
    return {k: cfg.get(k) for k in ("windows.buckets.tessera", "labels.beta_window_sessions", "labels.min_beta_obs", "labels.baseline_sessions",
                                    "game.horizon_days", "game.volume_window_days", "portfolio.horizon_sessions", "road1.optimiser",
                                    "road1.grounding_lambda_grid", "road1.sheaf_penalty_grid", "road1.ridge_grid",
                                    "reader.dials.min_rows_per_class", "reader.dials.ridge_grid")}


def test_a_staged_reader_shard_runs_to_one_verified_bundle(world, tmp_path):
    folder, job_id = stage_reader(world, tmp_path / "stage")
    manifest = read_manifest(folder)
    assert manifest["job"] == "reader" and len(manifest["unit_ids"]) == 40 and manifest["config_hash"]
    result = run(world, folder, tmp_path / "out")
    assert result.outcome is ShardOutcome.COMPLETE and result.n_remaining == 0
    scan = scan_bundles([tmp_path / "out"], job_id)
    rows = read_consolidated(scan.valid)
    assert len(rows) == 40 and (rows[STATUS] == Status.OK.value).all() and rows["parse_ok"].all()
    assert scan.valid[0].manifest["meta"]["job"] == "reader" and scan.valid[0].manifest["meta"]["engine"]["engine"] == "mock"


def test_a_session_that_stops_early_resumes_from_its_bundle_to_the_same_result(world, tmp_path):
    folder, job_id = stage_reader(world, tmp_path / "stage")
    whole = run(world, folder, tmp_path / "whole")
    expected = read_consolidated(scan_bundles([tmp_path / "whole"], job_id).valid).sort_values(UNIT_ID).reset_index(drop=True)

    class Ticking(FakeClock):
        """Every read of the clock costs time, so the session limit is reached after a few batches."""

        def now(self):
            self.t += 60.0
            return self.t

    short = {"kaggle.session_cutoff_minutes": 14, "kaggle.final_write_reserve_minutes": 4}
    first = run(world, folder, tmp_path / "s1", clock=Ticking(), overrides=short)
    assert first.outcome is ShardOutcome.SESSION_LIMIT and 0 < first.n_remaining < 40
    second = run(world, folder, tmp_path / "s2", inputs=[tmp_path / "s1"])
    assert second.outcome is ShardOutcome.COMPLETE and second.n_remaining == 0
    resumed = read_consolidated(scan_bundles([tmp_path / "s2"], job_id).valid).sort_values(UNIT_ID).reset_index(drop=True)
    assert resumed.equals(expected) and whole.n_remaining == 0


def test_a_shard_staged_with_another_config_is_refused(world, tmp_path):
    folder, _ = stage_reader(world, tmp_path / "stage")
    with pytest.raises(RuntimeError, match="config hash"):
        run_shard_from_folder(folder, tmp_path / "out", CONFIG_DIR, engine=MockEngine(),
                              overrides={**OVERRIDES, **_world_overrides(world), "runtime.min_parse_rate": 0.5})


def test_planned_units_follow_the_shards_own_order():
    units = pd.DataFrame({UNIT_ID: ["a", "b", "c", "d"], "x": range(4)})
    assert list(planned_for_shard(units, ["c", "a"])[UNIT_ID]) == ["c", "a"]


def persona_inputs(world):
    cfg = world.cfg
    skeletons = persona_skeletons(world, cfg)
    stamp = pd.Timestamp("2020-03-02 15:00", tz="UTC")
    descriptors = {s: {"node_type": "company", "sector": world.sector[s], "layer": "mid"} for s in world.securities}
    payloads, states = {}, {}
    for i in range(5):
        item = NewsItem(f"item{i}", "finance", stamp, f"{world.names['s00']} beat estimates", "record results and raises guidance", ("energy",))
        payloads[f"e{i}"] = EventPayload(f"e{i}", "finance", stamp, (item,), (item,), ({"node_id": "s00", "node_type": "company"},),
                                         ({"node_id": "s03", "link_type": "industry"},), ("s05",), descriptors, frozenset(world.securities))
        for s in skeletons:
            states[(f"e{i}", s.persona_id)] = PersonaState(0.03, 0.01, 0.02, "Last quarter you made no trades.", {"Energy": 0.2})
    bios = {s.bio_id: template_bio(cfg, s) for s in skeletons}
    return skeletons, bios, payloads, states


def test_persona_payloads_round_trip_exactly(world, tmp_path):
    skeletons, bios, payloads, states = persona_inputs(world)
    pl.save_events(payloads, tmp_path)
    pl.save_skeletons(skeletons, tmp_path)
    pl.save_bios(bios, tmp_path)
    pl.save_states(states, tmp_path)
    assert pl.load_events(tmp_path) == payloads
    assert pl.load_skeletons(tmp_path) == {s.persona_id: s for s in skeletons}
    assert pl.load_bios(tmp_path) == bios and pl.load_states(tmp_path) == states


def test_a_persona_shard_runs_through_the_runner_and_matches_direct_processing(world, tmp_path):
    cfg = world.cfg
    skeletons, bios, payloads, states = persona_inputs(world)
    units = pd.DataFrame({UNIT_ID: [unit_id(e, s.persona_id) for e in payloads for s in skeletons]})

    def write(folder):
        pl.save_events(payloads, folder)
        pl.save_skeletons(skeletons, folder)
        pl.save_bios(bios, folder)
        pl.save_states(states, folder)

    job_id = make_job_id("personas", "dbX", "llmY")
    shard = Shard("s0000", job_id, tuple(units[UNIT_ID]), 1.0, 2.0)
    folder = stage_shard(tmp_path / "stage", "personas", "dbX", "llmY", shard, units, write, cfg)
    result = run(world, folder, tmp_path / "out")
    assert result.outcome is ShardOutcome.COMPLETE
    rows = read_consolidated(scan_bundles([tmp_path / "out"], job_id).valid)
    assert len(rows) == len(units) and rows["saw_focal"].any()
    from oryxis.personas.run import PersonaProcessor
    from oryxis.reader.prompt import WhitespaceTokenizer
    engine = MockEngine()
    engine.register("persona", persona_responder)
    direct = PersonaProcessor(cfg, engine, {s.persona_id: s for s in skeletons}, bios, payloads, states, WhitespaceTokenizer())(units)
    got = dict(zip(rows[UNIT_ID], rows["draws_json"], strict=True))
    want = dict(zip(direct[UNIT_ID], direct["draws_json"], strict=True))
    called = {u for u, d in want.items() if isinstance(d, str)}
    assert called and {u: got[u] for u in called} == {u: want[u] for u in called}  # identical answers for every called persona


def test_mass_world_round_trips_and_runs_as_a_job(world, tmp_path):
    cfg = world.cfg
    rng = np.random.default_rng(0)
    dates = pd.bdate_range("2022-03-01", periods=3)
    stocks = [f"S{i}" for i in range(6)]
    pool = pd.DataFrame([{"Date": d, "Stock": s, "Industry": "A" if i < 3 else "B", "FREE_MV": 10.0 - i}
                         for d in dates for i, s in enumerate(stocks)])
    frame = pd.DataFrame([{"Date": d, "Stock": s, "ret_1": float(rng.normal())} for d in dates for s in stocks])
    store = ModalityStore({"price_feature": frame})
    pl.save_mass_world(pool, store, {"t0": ["price_feature"]}, {str(d.date()): "macro text" for d in dates}, {}, tmp_path)
    loaded = pl.load_mass_world(tmp_path)
    assert list(loaded.pool(str(dates[0].date()))["Stock"]) == stocks and loaded.types == {"t0": ["price_feature"]}
    assert loaded.macro(str(dates[0].date())) == "macro text" and loaded.macro("1999-01-01") is None
    assert set(loaded.store.frames) == {"price_feature"}

    units = pd.DataFrame({UNIT_ID: [f"{d.date()}|t0|{a}" for d in dates for a in range(2)]})
    job_id = make_job_id("mass", "dbX", "llmY")
    shard = Shard("s0000", job_id, tuple(units[UNIT_ID]), 1.0, 2.0)

    def write(folder):
        pl.save_mass_world(pool, store, {"t0": ["price_feature"]}, {str(d.date()): "macro text" for d in dates}, {}, folder)

    folder = stage_shard(tmp_path / "stage", "mass", "dbX", "llmY", shard, units, write, cfg)

    class Scripted(MockEngine):
        def generate(self, requests, params):
            from oryxis.compute.engine import GenResult
            return [GenResult(r.unit_id, (json.dumps({"Outline": "x", "Details": {"StockPoolSelector": "RandomStockSelector"}})
                                          if r.task == "mass_style" else json.dumps({"Stock": ["S0"]}),), 5, 5) for r in requests]

    from oryxis.reader.prompt import WhitespaceTokenizer
    result = run_shard_from_folder(folder, tmp_path / "out", CONFIG_DIR, engine=Scripted(), tokenizer=WhitespaceTokenizer(),
                                   clock=FakeClock(), overrides={**OVERRIDES, **_world_overrides(world)})
    rows = read_consolidated(scan_bundles([tmp_path / "out"], job_id).valid)
    assert result.n_remaining == 0 and len(rows) == 6


def test_a_trial_limit_runs_only_the_first_units_and_a_full_run_continues_from_them(world, tmp_path):
    folder, job_id = stage_reader(world, tmp_path / "stage")
    from oryxis.reader.prompt import WhitespaceTokenizer
    engine = MockEngine()
    engine.register("reader", reader_responder)
    kw = dict(engine=engine, tokenizer=WhitespaceTokenizer(), clock=FakeClock(), overrides={**OVERRIDES, **_world_overrides(world)})
    trial = run_shard_from_folder(folder, tmp_path / "t", CONFIG_DIR, limit_units=7, **kw)
    assert trial.n_planned == 7 and trial.n_terminal == 7
    full = run_shard_from_folder(folder, tmp_path / "f", CONFIG_DIR, input_dirs=[tmp_path / "t"], **kw)
    assert full.n_planned == 40 and full.n_remaining == 0
    assert len(read_consolidated(scan_bundles([tmp_path / "f"], job_id).valid)) == 40


def test_identical_copies_of_the_dev_fit_count_as_one_and_different_files_do_not(tmp_path):
    import importlib.util

    spec = importlib.util.spec_from_file_location("kaggle_db1_runner", Path(__file__).resolve().parents[1] / "scripts" / "kaggle_db1_runner.py")
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    name = "framework-db3-llm2-salvaged_no_certainty-sheaf"
    one, two, other = (tmp_path / d / "results" / name / "dev_fit.pkl" for d in ("a", "b", "c"))
    for path, content in ((one, b"x"), (two, b"x"), (other, b"y")):
        path.parent.mkdir(parents=True)
        path.write_bytes(content)
    assert runner.pick_dev_fit([str(one), str(two)], "db3", "llm2", "salvaged_no_certainty") == [str(one)]
    assert len(runner.pick_dev_fit([str(one), str(other)], "db3", "llm2", "salvaged_no_certainty")) == 2
