"""The final reader (``reader2``): rules, processor protections, and whole runs with injected failures and resumes."""
from __future__ import annotations

import json
import os
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import pytest

from finflock.common.io import atomic_path  # noqa: F401  (imported so the fsync shim below patches the same module)
from finflock.common.schema import PARSE_OK, REASON, STATUS, UNIT_ID, Status
from finflock.compute.bundle import make_job_id, read_consolidated, scan_bundles
from finflock.compute.engine import Engine, GenResult
from finflock.compute.session import SessionError
from finflock.compute.shard import units_fingerprint
from finflock.reader.masking import MaskerCache
from finflock.reader.prompt import WhitespaceTokenizer
from pilot.reader_v3_prompts import examples
from reader2 import plan as P
from reader2 import settings as S
from reader2.engine import EngineSupervisor
from reader2.processor import ReaderV2Processor
from reader2.session import SMOKE_FAILED_FILE, run_reader2, signature
from tests.conftest import CONFIG_DIR

JOB = make_job_id(S.JOB, "db3", "llm2")
REF = re.compile(r"^(A\d+): ", re.M)


@pytest.fixture(autouse=True)
def no_fsync(monkeypatch):
    """Windows cannot fsync a read-only handle or a directory; the bundle logic does not depend on it."""
    import finflock.common.io as io

    monkeypatch.setattr(io.os, "fsync", lambda fd: None)
    monkeypatch.setattr(io, "_fsync_dir", lambda directory: None)


# ---------------------------------------------------------------------------- a fake model
class FakeEngine(Engine):
    """Answers every request with a valid worked-example answer that names the offered assets. Failures are injected by unit id."""

    name = "fake"

    def __init__(self, *, event_for=None, bad_first=(), always_bad=(), poison=(), die_once_on=None, calls=None):
        self.event_for = event_for or (lambda unit: "asset_event")
        self.bad_first, self.always_bad, self.poison = set(bad_first), set(always_bad), set(poison)
        self.die_once_on, self.died = die_once_on, False
        self.seen: Counter = Counter()
        self.calls = calls if calls is not None else []
        self.rebuilt = False

    def answer(self, request) -> str:
        refs = REF.findall(request.prompt)
        event = self.event_for(request.unit_id)
        template = next(a for _, _, a in examples(True) if a["event_type"] == "asset_event")
        pins = [{"ref": r, "direction": "down", "value_effect_pct": 4, "magnitude": 2, "role": "source"} for r in refs[:1]]
        answer = {**template, "event_type": event, "pins": pins}
        return json.dumps(answer, separators=(",", ":"))

    def generate(self, requests, params):
        self.calls.append([r.unit_id for r in requests])
        if self.die_once_on and not self.died and any(r.unit_id == self.die_once_on for r in requests):
            self.died = True
            raise RuntimeError("EngineCore died")
        if any(r.unit_id in self.poison for r in requests):
            raise ValueError("poison unit in batch")
        out = []
        for r in requests:
            self.seen[r.unit_id] += 1
            first = self.seen[r.unit_id] == 1
            text = "{broken" if (r.unit_id in self.always_bad or (r.unit_id in self.bad_first and first)) else self.answer(r)
            out.append(GenResult(r.unit_id, (text,), 1800, 90, "stop"))
        return out


# ---------------------------------------------------------------------------- a synthetic pack
def make_pack(root: Path, n: int = 40, n_shards: int = 1, llm: str = "llm2") -> tuple[Path, pd.DataFrame]:
    rows = []
    for i in range(n):
        kind = i % 5
        uid = f"ev:{i:04d}"
        if kind == 0:
            rows.append((uid, "finance", "ext_company_news", "Acme Corp (ACME) posts results", "Acme Corp reported a quarterly profit.",
                         [{"node_id": "ACME", "node_type": "company", "layer": "non_sp", "priority": 5.0, "sector": None}]))
        elif kind == 1:
            rows.append((uid, "geo", "ext_world_news", "Russia faces new sanctions", "Western governments announce sanctions on Russia.",
                         [{"node_id": "c:RU", "node_type": "country", "layer": "non_sp", "priority": 5.0, "sector": None}]))
        elif kind == 2:
            rows.append((uid, "social", "ext_social", "$ACME to the moon", "$ACME to the moon", [{"node_id": "ACME", "node_type": "company", "layer": "non_sp", "priority": 5.0, "sector": None}]))
        elif kind == 3:
            rows.append((uid, "finance", "earnings", "ACME - 8-K items 2.02", "Item 2.02 Results of Operations.", [{"node_id": "ACME", "node_type": "company", "layer": "non_sp", "priority": 5.0, "sector": None}]))
        else:
            rows.append((uid, "finance", "ext_company_news", "Nothing to pin here", "No candidate asset.", []))
    units = pd.DataFrame(rows, columns=["unit_id", "kind", "family", "title", "text", "pins"])
    units["pins_json"] = units.pop("pins").map(json.dumps)
    units["available_date"] = "2018-03-01"
    (root / "data/db3").mkdir(parents=True)
    units.to_parquet(root / "data/db3/units.parquet", index=False)
    aliases = pd.DataFrame([("ACME", "ACME", "ticker", None, None), ("ACME", "Acme Corp", "name", None, None)],
                           columns=["node_id", "alias", "alias_type", "valid_from", "valid_to"])
    aliases.to_parquet(root / "data/db3/aliases.parquet", index=False)
    scoped = P.scope_units(units, "db3", llm)
    plan = {"buckets": {"db3": {"units_sha256": units_fingerprint(units["unit_id"]), "n_units": n}},
            "runs": {f"db3-{llm}": {"n_units": len(scoped), "n_shards": n_shards}}}
    (root / "RUN_PLAN.json").write_text(json.dumps(plan), encoding="utf-8")
    return root, units


def run(pack, tmp_path, engine, *, out="out", inputs=None, shard=None, llm="llm2", **kw):
    return run_reader2(bucket="db3", llm=llm, pack_root=pack, config_dir=CONFIG_DIR, input_root=inputs or tmp_path / "in",
                       out_dir=tmp_path / out, shard=shard, engine=engine, tokenizer=WhitespaceTokenizer(), log=lambda *_: None, **kw)


def tape(tmp_path, *dirs) -> pd.DataFrame:
    scan = scan_bundles([tmp_path / d for d in dirs], JOB)
    return read_consolidated(scan.valid).set_index(UNIT_ID)


# ---------------------------------------------------------------------------- rules
def test_prompt_is_the_poc_prompt():
    import yaml
    from reader2.prompting import GLOSSARY, prompt_settings
    poc = Path(__file__).resolve().parents[1] / "outputs/pilot-reader/reader-v3-poc-kaggle-input/pilot/pilot_reader_v3_poc_gate_shots.yaml"
    if not poc.exists():
        pytest.skip("the POC pack is not on this machine")
    ref = yaml.safe_load(poc.read_text(encoding="utf-8"))["reader_v2"]
    assert prompt_settings()["reader_v2"]["system"] == ref["system"] and GLOSSARY == ref["glossary"]


def test_helpers_equal_the_pilot_helpers():
    import pilot.runner as pr
    from reader2 import prompting
    assert prompting.v2_prompt is pr.v2_prompt and prompting.asset_names is pr.asset_names


def test_gate_and_rating_rule(cfg):
    from finflock.reader.output_v2 import parse_reading_v2
    from reader2.rules import finalize
    refs = {"A1": "ACME"}
    text = json.dumps({**next(a for _, _, a in examples(True) if a["event_type"] == "asset_event"),
                       "pins": [{"ref": "A1", "direction": "down", "value_effect_pct": 4, "magnitude": 2, "role": "source"}]})
    def reading(event):
        return parse_reading_v2(text.replace('"asset_event"', f'"{event}"'), refs, cfg, gate=True, enforce_gate=False)
    r = reading("not_market_news")
    assert r.ok and r.cause["earnings_miss"] > 0                       # not enforced yet
    audit = finalize(r, "ext_company_news", "Dog show", cfg)
    assert r.cause["other"] == 1.0 and r.pins[0]["direction"] == "none" and r.pins[0]["value_effect_pct"] == 0.0
    assert audit["event_type_final"] == "not_market_news" and audit["gate_enforced"]
    r = reading("asset_event")
    audit = finalize(r, "ext_company_news", "Morgan Stanley Maintains Equal-Weight on Acme, Raises PT to $50", cfg)
    assert audit["rating_rule"] and r.cause["other"] == 1.0 and r.pins[0]["value_effect_pct"] == -4.0  # opinion: pins keep their effect
    r = reading("not_market_news")                                     # filings: the model's answer is recorded, never enforced
    audit = finalize(r, "earnings", "ACME - 8-K items 2.02", cfg)
    assert not audit["gate_enforced"] and audit["event_type_final"] == "asset_event" and audit["event_type_model"] == "not_market_news"
    assert r.cause["earnings_miss"] > 0 and r.pins[0]["direction"] == "down"


# ---------------------------------------------------------------------------- scoping and shards
def test_tessera_half_for_llm1_and_llm3_only():
    units = pd.DataFrame({UNIT_ID: [f"ev:{i}" for i in range(400)]})
    assert len(P.scope_units(units, "tessera", "llm2")) == 400
    for llm in ("llm1", "llm3"):
        half = P.scope_units(units, "tessera", llm)
        assert 0 < len(half) < 400 and len(half) == len(P.scope_units(units, "tessera", llm))
    assert len(P.scope_units(units, "db1", "llm1")) == 400


def test_shards_partition_the_units():
    units = pd.DataFrame({UNIT_ID: [f"ev:{i}" for i in range(300)]})
    parts = P.single_part(3)
    shards = [set(P.shard_units(units, parts, k)[UNIT_ID]) for k in range(3)]
    assert sum(len(p) for p in shards) == 300 and set().union(*shards) == set(units[UNIT_ID]) and not shards[0] & shards[1]
    with pytest.raises(ValueError):
        P.shard_units(units, parts, None)
    with pytest.raises(ValueError):
        P.shard_units(units, parts, 3)
    assert len(P.shard_units(units, P.single_part(1), None)) == 300


def test_tessera_llm2_keeps_the_shared_half_in_its_own_shards():
    from finflock.data.buckets.events import hash_bucket
    units = pd.DataFrame({UNIT_ID: [f"ev:{i}" for i in range(600)]})
    parts = [{"hash_from": 0, "hash_below": 50, "n_shards": 2}, {"hash_from": 50, "hash_below": 100, "n_shards": 3}]
    got = [set(P.shard_units(units, parts, k)[UNIT_ID]) for k in range(5)]
    assert set().union(*got) == set(units[UNIT_ID]) and sum(len(g) for g in got) == 600
    assert all(hash_bucket(u) < 50 for g in got[:2] for u in g) and all(hash_bucket(u) >= 50 for g in got[2:] for u in g)


# ---------------------------------------------------------------------------- a whole run
def test_complete_run_rows_and_rules(tmp_path):
    pack, units = make_pack(tmp_path / "pack")
    engine = FakeEngine(event_for=lambda u: "not_market_news" if u.endswith("0") else "asset_event")
    summary = run(pack, tmp_path, engine)
    assert summary["outcome"] == "complete" and summary["remaining"] == 0
    t = tape(tmp_path, "out")
    assert len(t) == len(units) and set(t[STATUS]) == {"ok", "skipped"}
    assert (t[t.family == "ext_company_news"][STATUS] == "skipped").sum() == 8 and t.loc[t[STATUS] == "skipped", REASON].eq("no_usable_pin").all()
    called = t[t[STATUS] == "ok"]
    assert called[PARSE_OK].all()
    news0 = called[called.index.str.endswith("0")]                       # gated family + not_market_news -> other, zero effect
    assert news0.loc[news0.family == "ext_company_news", "is_other"].all()
    assert all(json.loads(p)[0]["direction"] == "none" for p in news0.loc[news0.family == "ext_company_news", "pins_json"])
    filing = called[called.family == "earnings"]                         # never gated
    assert (filing.event_type_final == "asset_event").all() and not filing.gate_enforced.any()
    assert all(json.loads(p)[0]["value_effect_pct"] == -4.0 for p in filing.pins_json)  # sign follows direction
    assert called["cause_other"].between(0, 1).all() and called["certainty"].isna().all()
    assert summary["quality"]["parse_rate"] == 1.0


def test_parse_failure_is_retried_and_recorded(tmp_path):
    pack, _ = make_pack(tmp_path / "pack")
    summary = run(pack, tmp_path, FakeEngine(bad_first={"ev:0001", "ev:0002"}, always_bad={"ev:0006"}), skip_smoke=True)
    t = tape(tmp_path, "out")
    assert t.loc["ev:0001", PARSE_OK] and t.loc["ev:0001", "n_tries"] == 2        # fixed by the retry
    assert not t.loc["ev:0006", PARSE_OK] and t.loc["ev:0006", "n_tries"] == 1 + S.PARSE_RETRIES
    assert t.loc["ev:0006", STATUS] == "ok" and t.loc["ev:0006", "raw_output"] == "{broken" and t.loc["ev:0006", REASON] == "invalid_json"
    assert summary["quality"]["failure_reasons"] == {"invalid_json": 1}


def test_poison_unit_costs_one_row_not_the_batch(tmp_path):
    pack, units = make_pack(tmp_path / "pack")
    summary = run(pack, tmp_path, FakeEngine(poison={"ev:0003"}), skip_smoke=True)
    t = tape(tmp_path, "out")
    assert len(t) == len(units)
    assert t.loc["ev:0003", STATUS] == Status.FAILED.value                  # retried inside the run, then given up with its reason
    assert "poison" in str(t.loc["ev:0003", "detail"])
    good = t[(t[STATUS] == "ok")]
    assert len(good) == 31 and good[PARSE_OK].all()                          # every other unit of the same batches survived
    assert summary["remaining"] == 1 or summary["terminal"] == len(units)


def test_resume_after_a_hard_crash_runs_no_unit_twice(tmp_path):
    pack, units = make_pack(tmp_path / "pack", n=60)

    class Crash(BaseException):
        pass

    first = FakeEngine()
    real = first.generate
    state = {"n": 0}

    def crashing(requests, params):
        state["n"] += 1
        if state["n"] == 3:
            raise Crash()
        return real(requests, params)

    first.generate = crashing
    overrides = {"runtime.batch_units": 8, "kaggle.checkpoint_minutes": 0}
    with pytest.raises(Crash):
        run(pack, tmp_path, first, skip_smoke=True, overrides=overrides)
    mid = tape(tmp_path, "out")
    assert 0 < len(mid) < len(units)                                          # the checkpoints written before the crash survive
    second = FakeEngine()
    summary = run(pack, tmp_path, second, skip_smoke=True, overrides=overrides)
    assert summary["remaining"] == 0
    rerun = {u for batch in second.calls for u in batch}
    assert not rerun & set(mid[mid[STATUS].isin(["ok", "skipped"])].index)   # nothing finished is run again
    assert len(tape(tmp_path, "out")) == len(units)


def test_resume_from_attached_bundle_in_a_new_session(tmp_path):
    pack, units = make_pack(tmp_path / "pack", n=60)
    overrides = {"runtime.batch_units": 8, "kaggle.checkpoint_minutes": 0, "kaggle.session_cutoff_minutes": 3, "kaggle.final_write_reserve_minutes": 1}

    class Clock:
        t = 0.0
        def now(self):
            self.t += 20.0
            return self.t

    summary = run(pack, tmp_path, FakeEngine(), skip_smoke=True, overrides=overrides, clock=Clock())
    assert summary["outcome"] == "session_limit" and summary["remaining"] > 0
    attached = tmp_path / "attached"
    attached.mkdir()
    for b in (tmp_path / "out").glob("*.bundle"):
        (attached / b.name).write_bytes(b.read_bytes())
    second = FakeEngine()
    summary = run(pack, tmp_path, second, out="out2", inputs=attached, skip_smoke=True)
    assert summary["remaining"] == 0 and len(tape(tmp_path, "out2")) == len(units)


def test_engine_death_is_survived_by_rebuilding(tmp_path):
    pack, units = make_pack(tmp_path / "pack")
    builds = []

    def factory():
        builds.append(1)
        return FakeEngine(die_once_on="ev:0003") if len(builds) == 1 else FakeEngine()

    supervisor = EngineSupervisor(factory, log=lambda *_: None)
    summary = run(pack, tmp_path, supervisor, skip_smoke=True)
    assert len(builds) == 2 and summary["remaining"] == 0
    assert (tape(tmp_path, "out")[STATUS] != "error").all()


def test_smoke_test_stops_a_broken_setup_before_the_real_work(tmp_path):
    pack, _ = make_pack(tmp_path / "pack")
    engine = FakeEngine(always_bad={f"ev:{i:04d}" for i in range(40)})
    with pytest.raises(SessionError, match="smoke test failed"):
        run(pack, tmp_path, engine)
    assert (tmp_path / "out" / SMOKE_FAILED_FILE).exists() and not list((tmp_path / "out").glob("*.bundle"))


def test_a_bundle_of_another_prompt_is_refused(tmp_path):
    pack, _ = make_pack(tmp_path / "pack")
    run(pack, tmp_path, FakeEngine())
    plan = json.loads((pack / "RUN_PLAN.json").read_text())
    scoped_sig = signature("db3", "llm2", plan["buckets"]["db3"]["units_sha256"])
    ref = scan_bundles([tmp_path / "out"], JOB).valid[0]
    assert ref.manifest["meta"]["prompt_sha256"] == scoped_sig["prompt_sha256"]
    from reader2.session import check_resumable
    assert check_resumable([tmp_path / "out"], JOB, {**scoped_sig, "reader2_version": "other"})


def test_sharded_run_needs_a_shard_number(tmp_path):
    pack, units = make_pack(tmp_path / "pack", n_shards=2)
    with pytest.raises(SessionError, match="SHARD"):
        run(pack, tmp_path, FakeEngine())
    run(pack, tmp_path, FakeEngine(), shard=0, skip_smoke=True)
    run(pack, tmp_path, FakeEngine(), shard=1, out="out_b", skip_smoke=True)
    assert len(tape(tmp_path, "out")) + len(tape(tmp_path, "out_b")) == len(units)
    assert not set(tape(tmp_path, "out").index) & set(tape(tmp_path, "out_b").index)


def test_unit_table_must_match_the_plan(tmp_path):
    pack, _ = make_pack(tmp_path / "pack")
    plan = json.loads((pack / "RUN_PLAN.json").read_text())
    plan["buckets"]["db3"]["units_sha256"] = "0" * 64
    (pack / "RUN_PLAN.json").write_text(json.dumps(plan))
    with pytest.raises(SessionError, match="not the one"):
        run(pack, tmp_path, FakeEngine())


def test_null_text_does_not_break_the_prompt(tmp_path):
    pack, units = make_pack(tmp_path / "pack")
    units.loc[units.index[0], ["title", "text"]] = [None, None]
    units.to_parquet(pack / "data/db3/units.parquet", index=False)
    run(pack, tmp_path, FakeEngine(), skip_smoke=True)
    assert tape(tmp_path, "out").loc[units.unit_id[0], STATUS] in ("ok", "skipped")


def test_processor_isolates_a_failing_batch_by_halving(cfg):
    class Flaky(FakeEngine):
        def generate(self, requests, params):
            if len(requests) > 3:
                raise ValueError("too many at once")
            return super().generate(requests, params)

    pack = Path(os.environ.get("TMP", "."))
    units = pd.DataFrame({UNIT_ID: [f"ev:{i}" for i in range(12)], "kind": "finance", "family": "ext_company_news",
                          "title": "Acme Corp posts", "text": "Acme Corp posts profit", "available_date": "2018-03-01",
                          "pins_json": json.dumps([{"node_id": "ACME", "node_type": "company", "layer": "non_sp", "priority": 5.0, "sector": None}])})
    aliases = pd.DataFrame([("ACME", "Acme Corp", "name", None, None)], columns=["node_id", "alias", "alias_type", "valid_from", "valid_to"])
    maskers = MaskerCache(aliases, label_prefix="COMPANY_", short_ticker_max_len=2)
    processor = ReaderV2Processor(cfg, Flaky(), maskers, WhitespaceTokenizer(), log=lambda *_: None)
    out = processor(units)
    assert (out[STATUS] == "ok").all() and processor.counts["batch_splits"] >= 1


def wait_for(condition, seconds=10.0):
    import time
    end = time.time() + seconds
    while time.time() < end and not condition():
        time.sleep(0.01)


def test_watchdog_interrupts_then_exits_a_hung_run(cfg):
    import time
    from reader2.session import Watchdog

    aliases = pd.DataFrame([("ACME", "Acme Corp", "name", None, None)], columns=["node_id", "alias", "alias_type", "valid_from", "valid_to"])
    processor = ReaderV2Processor(cfg, FakeEngine(), MaskerCache(aliases, label_prefix="COMPANY_", short_ticker_max_len=2), WhitespaceTokenizer(), log=lambda *_: None)
    events = []
    processor.last_activity = processor.clock() - 3600
    with Watchdog(processor, 60, exit_fn=lambda code: events.append(("exit", code)), interrupt_fn=lambda: events.append("interrupt"),
                  log=lambda *_: None, poll=0.01, grace=0.05):
        wait_for(lambda: len(events) >= 2)
    assert events == ["interrupt", ("exit", S.HANG_EXIT_CODE)]
    processor.last_activity = processor.clock()
    events.clear()
    with Watchdog(processor, 60, exit_fn=events.append, interrupt_fn=lambda: events.append("interrupt"), log=lambda *_: None, poll=0.01, grace=0.05):
        time.sleep(0.2)
    assert events == []
    processor.last_activity = processor.clock() - 3600          # the run recovers during the grace period: no hard exit
    with Watchdog(processor, 60, exit_fn=events.append, interrupt_fn=lambda: events.append("interrupt"), log=lambda *_: None, poll=0.01, grace=30) as dog:
        wait_for(lambda: events == ["interrupt"])
        dog.stop.set()
    assert events == ["interrupt"]


# ---------------------------------------------------------------------------- the "rest" mode
def test_rest_pieces_are_disjoint_balanced_and_cover_the_unfinished_work():
    units = pd.DataFrame({UNIT_ID: [f"ev:{i}" for i in range(4000)]})
    # the unfinished work of a shard is one parity of the shard hash: the pieces must still be even
    shard0 = P.shard_units(units, P.single_part(2), 0)
    finished = set(units[UNIT_ID]) - set(shard0[UNIT_ID])
    pieces = [P.rest_units(units, finished, k, 4) for k in range(4)]
    assert sum(len(p) for p in pieces) == len(shard0) and set().union(*[set(p[UNIT_ID]) for p in pieces]) == set(shard0[UNIT_ID])
    assert max(len(p) for p in pieces) < 1.2 * min(len(p) for p in pieces)
    with pytest.raises(ValueError):
        P.rest_units(units, finished, 4, 4)


def test_rest_mode_finishes_a_run_whose_shard_never_ran(tmp_path):
    pack, units = make_pack(tmp_path / "pack", n=80, n_shards=2)
    ov = {"runtime.batch_units": 8, "kaggle.checkpoint_minutes": 0}
    run(pack, tmp_path, FakeEngine(), shard=1, out="s1", skip_smoke=True, overrides=ov)             # shard 0 is "lost"
    attached = tmp_path / "attached"
    attached.mkdir()
    for b in (tmp_path / "s1").glob("*.bundle"):
        (attached / b.name).write_bytes(b.read_bytes())
    done_before = set(tape(tmp_path, "s1").index)
    engines = [FakeEngine(), FakeEngine()]
    for k, e in enumerate(engines):
        s = run(pack, tmp_path, e, shard=f"rest:{k}/2", inputs=attached, out=f"rest{k}", skip_smoke=True, overrides=ov)
        assert s["shard"] == f"rest{k}of2" and s["remaining"] == 0
    ran = [{u for batch in e.calls for u in batch} for e in engines]
    assert not ran[0] & ran[1] and not (ran[0] | ran[1]) & done_before                                # disjoint, nothing finished is repeated
    merged = read_consolidated(scan_bundles([tmp_path / "rest0", tmp_path / "rest1", attached], JOB).valid)
    assert set(merged[UNIT_ID]) == set(units[UNIT_ID]) and merged[UNIT_ID].is_unique


def test_rest_mode_refuses_without_attached_bundles_and_bad_text(tmp_path):
    pack, _ = make_pack(tmp_path / "pack")
    with pytest.raises(SessionError, match="no bundle"):
        run(pack, tmp_path, FakeEngine(), shard="rest:0/2")
    for bad in ("rest:2/2", "rest:x", "banana", "rest:0/0"):
        with pytest.raises(SessionError, match="not valid"):
            run(pack, tmp_path, FakeEngine(), shard=bad)
