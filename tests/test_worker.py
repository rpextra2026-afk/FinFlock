"""The shard worker: cutoff, resume after stop and crash, bad units, early-abort guards."""

from __future__ import annotations

import pandas as pd
import pytest

from finfm.common.schema import (
    ATTEMPTS,
    BUNDLE_SUFFIX,
    REASON,
    STATUS,
    UNIT_ID,
    AbortReason,
    Reason,
    ShardOutcome,
    Status,
)
from finfm.compute.bundle import make_job_id, read_consolidated, scan_bundles
from finfm.compute.worker import StopFlag, run_shard
from tests.conftest import Crash, FakeClock, make_processor, make_units

JOB = make_job_id("reader", "tessera", "llm2")
MINUTE = 60.0

# Small, fast limits so the scenarios finish in milliseconds of real time.
FAST = {
    "kaggle.session_cutoff_minutes": 100,
    "kaggle.final_write_reserve_minutes": 10,
    "kaggle.checkpoint_minutes": 0,
    "runtime.batch_units": 10,
    "runtime.canary_units": 20,
    "runtime.error_window_units": 20,
}


def run(cfg, planned, processor, clock, tmp_path, *, inputs=(), out="out", **kw):
    return run_shard(
        cfg=cfg,
        job_id=JOB,
        shard_id="s0000",
        planned=planned,
        processor=processor,
        input_dirs=list(inputs),
        out_dir=tmp_path / out,
        meta={"test": True},
        clock=clock,
        **kw,
    )


def final_rows(tmp_path, *dirs):
    scan = scan_bundles([tmp_path / d for d in dirs], JOB)
    return read_consolidated(scan.valid).sort_values(UNIT_ID).reset_index(drop=True)


def test_complete_run_writes_one_verified_bundle(cfg, tmp_path):
    cfg = cfg.with_overrides(FAST)
    clock = FakeClock()
    planned = make_units(50)
    result = run(cfg, planned, make_processor(clock, 1.0), clock, tmp_path)
    assert result.outcome is ShardOutcome.COMPLETE
    assert result.n_remaining == 0
    assert len(list((tmp_path / "out").glob(f"*{BUNDLE_SUFFIX}"))) == 1
    frame = final_rows(tmp_path, "out")
    assert len(frame) == 50 and (frame[STATUS] == Status.OK.value).all()


def test_never_runs_past_the_cutoff_and_resumes_to_the_same_result(cfg, tmp_path):
    cfg = cfg.with_overrides(FAST)
    planned = make_units(200)

    full_clock = FakeClock()
    full_cfg = cfg.with_overrides({"kaggle.session_cutoff_minutes": 720})
    run(full_cfg, planned, make_processor(full_clock, 1.0), full_clock, tmp_path, out="full")
    expected = final_rows(tmp_path, "full")

    first_clock = FakeClock()
    first = run(cfg, planned, make_processor(first_clock, 60.0), first_clock, tmp_path, out="s1")
    assert first.outcome is ShardOutcome.SESSION_LIMIT
    assert first_clock.now() <= (100 - 10) * MINUTE  # stopped before cutoff minus the write reserve
    assert 0 < first.n_remaining < 200

    second_clock = FakeClock()
    second = run(
        cfg, planned, make_processor(second_clock, 1.0), second_clock, tmp_path, inputs=[tmp_path / "s1"], out="s2"
    )
    assert second.outcome is ShardOutcome.COMPLETE and second.n_remaining == 0
    resumed = final_rows(tmp_path, "s2")
    assert resumed.equals(expected)  # same result however the work was split
    assert len(list((tmp_path / "s2").glob(f"*{BUNDLE_SUFFIX}"))) == 1


def test_finished_units_are_not_recomputed_on_resume(cfg, tmp_path):
    cfg = cfg.with_overrides(FAST)
    planned = make_units(40)
    clock = FakeClock()
    run(cfg, planned, make_processor(clock, 1.0), clock, tmp_path, out="s1")
    seen: list[str] = []

    def spy(batch):
        seen.extend(batch[UNIT_ID])
        return make_processor(FakeClock(), 1.0)(batch)

    result = run(cfg, planned, spy, FakeClock(), tmp_path, inputs=[tmp_path / "s1"], out="s2")
    assert seen == [] and result.outcome is ShardOutcome.COMPLETE


def test_hard_crash_loses_only_work_after_the_last_checkpoint(cfg, tmp_path):
    cfg = cfg.with_overrides(FAST)  # checkpoint after every batch
    planned = make_units(100)
    clock = FakeClock()
    with pytest.raises(Crash):
        run(cfg, planned, make_processor(clock, 1.0, crash_after=35), clock, tmp_path, out="crashed")
    saved = final_rows(tmp_path, "crashed")
    assert 0 < len(saved) <= 35  # checkpointed batches survive the kill
    resume_clock = FakeClock()
    result = run(
        cfg, planned, make_processor(resume_clock, 1.0), resume_clock, tmp_path,
        inputs=[tmp_path / "crashed"], out="resumed",
    )
    assert result.outcome is ShardOutcome.COMPLETE
    assert len(final_rows(tmp_path, "resumed")) == 100


def test_real_cutoff_default_config_stops_before_the_9h45_limit(cfg, tmp_path):
    cfg = cfg.with_overrides({"kaggle.checkpoint_minutes": 60})
    clock = FakeClock()
    planned = make_units(20_000)
    result = run(cfg, planned, make_processor(clock, 2.0), clock, tmp_path)  # 11 h of work
    assert result.outcome is ShardOutcome.SESSION_LIMIT
    limit = (585 - float(cfg.get("kaggle.final_write_reserve_minutes"))) * MINUTE
    assert clock.now() <= limit
    assert clock.now() > limit - 64 * 2.0 * 2  # used the session, did not stop early


def test_bad_unit_becomes_an_error_row_and_the_run_continues(cfg, tmp_path):
    cfg = cfg.with_overrides(FAST)
    clock = FakeClock()
    planned = make_units(30)
    bad = planned[UNIT_ID].iloc[7]
    result = run(cfg, planned, make_processor(clock, 1.0, fail_ids={bad}), clock, tmp_path)
    assert result.outcome is ShardOutcome.COMPLETE
    frame = final_rows(tmp_path, "out").set_index(UNIT_ID)
    assert frame.loc[bad, STATUS] == Status.ERROR.value
    assert frame.loc[bad, REASON] == Reason.EXCEPTION.value
    assert (frame.drop(index=bad)[STATUS] == Status.OK.value).all()
    assert result.n_remaining == 1  # an error is retryable, not terminal


def test_unit_failing_every_session_becomes_failed_after_max_attempts(cfg, tmp_path):
    cfg = cfg.with_overrides(FAST)
    planned = make_units(30)
    bad = planned[UNIT_ID].iloc[3]
    previous: list = []
    for session in range(int(cfg.get("runtime.max_attempts")) + 1):
        clock = FakeClock()
        result = run(
            cfg, planned, make_processor(clock, 1.0, fail_ids={bad}), clock, tmp_path,
            inputs=previous, out=f"s{session}",
        )
        previous = [tmp_path / f"s{session}"]
    row = final_rows(tmp_path, f"s{session}").set_index(UNIT_ID).loc[bad]
    assert row[STATUS] == Status.FAILED.value
    assert row[ATTEMPTS] >= int(cfg.get("runtime.max_attempts"))
    assert result.n_remaining == 0  # failed is terminal, never retried again


def test_low_parse_rate_is_only_logged_by_default_and_the_run_goes_on(cfg, tmp_path):
    cfg = cfg.with_overrides(FAST)
    clock = FakeClock()
    planned = make_units(300)
    result = run(cfg, planned, make_processor(clock, 1.0, parse_ok=False), clock, tmp_path)
    assert result.outcome is ShardOutcome.COMPLETE and result.abort_reason is None and result.n_remaining == 0


def test_the_session_cutoff_is_counted_from_the_start_of_the_session(cfg, tmp_path):
    cfg = cfg.with_overrides(FAST)
    clock = FakeClock()
    planned = make_units(2000)
    spent = 80 * MINUTE  # 80 of the 90 usable minutes went on installing, downloading and loading
    result = run(cfg, planned, make_processor(clock, 1.0), clock, tmp_path, session_offset_seconds=spent)
    assert result.outcome is ShardOutcome.SESSION_LIMIT and 0 < result.n_terminal < 700


def test_low_parse_rate_stops_early_and_still_saves_when_asked_to(cfg, tmp_path):
    cfg = cfg.with_overrides({**FAST, "runtime.stop_on_parse_rate": True})
    clock = FakeClock()
    planned = make_units(500)
    result = run(cfg, planned, make_processor(clock, 1.0, parse_ok=False), clock, tmp_path)
    assert result.outcome is ShardOutcome.ABORTED
    assert result.abort_reason is AbortReason.LOW_PARSE_RATE
    assert 0 < len(final_rows(tmp_path, "out")) < 100  # stopped near the canary, not at 500


def test_high_error_rate_trips_the_circuit_breaker(cfg, tmp_path):
    cfg = cfg.with_overrides(FAST)
    clock = FakeClock()
    planned = make_units(500)
    everyone = set(planned[UNIT_ID])
    result = run(cfg, planned, make_processor(clock, 1.0, fail_ids=everyone), clock, tmp_path)
    assert result.outcome is ShardOutcome.ABORTED
    assert result.abort_reason is AbortReason.ERROR_RATE


def test_stop_flag_ends_the_run_cleanly_after_the_current_batch(cfg, tmp_path):
    cfg = cfg.with_overrides(FAST)
    clock = FakeClock()
    flag = StopFlag()
    inner = make_processor(clock, 1.0)

    def processor(batch):
        out = inner(batch)
        flag.set()  # SIGTERM arrives during the first batch
        return out

    result = run(cfg, make_units(100), processor, clock, tmp_path, stop_flag=flag)
    assert result.outcome is ShardOutcome.TERMINATED
    assert len(final_rows(tmp_path, "out")) == 10


def test_duplicate_planned_units_are_refused(cfg, tmp_path):
    planned = pd.concat([make_units(3), make_units(3)], ignore_index=True)
    with pytest.raises(ValueError):
        run(cfg, planned, make_processor(FakeClock(), 1.0), FakeClock(), tmp_path)


def test_retry_errors_runs_a_failing_unit_again_inside_the_run_until_it_is_failed(cfg, tmp_path):
    cfg = cfg.with_overrides(FAST)
    clock = FakeClock()
    planned = make_units(30)
    bad = planned[UNIT_ID].iloc[5]
    calls: list[str] = []
    inner = make_processor(clock, 1.0, fail_ids={bad})

    def spy(batch):
        calls.extend(batch[UNIT_ID])
        return inner(batch)

    result = run(cfg, planned, spy, clock, tmp_path, retry_errors=True)
    assert result.outcome is ShardOutcome.COMPLETE and result.n_remaining == 0  # failed is terminal: nothing is left open
    row = final_rows(tmp_path, "out").set_index(UNIT_ID).loc[bad]
    assert row[STATUS] == Status.FAILED.value and calls.count(bad) >= int(cfg.get("runtime.max_attempts"))
    assert (final_rows(tmp_path, "out").drop(columns=[]).set_index(UNIT_ID).drop(index=bad)[STATUS] == Status.OK.value).all()


def test_a_unit_that_fails_once_is_recovered_by_the_retry_pass(cfg, tmp_path):
    cfg = cfg.with_overrides(FAST)
    clock = FakeClock()
    planned = make_units(30)
    flaky = planned[UNIT_ID].iloc[9]
    seen = {"n": 0}
    inner = make_processor(clock, 1.0)

    def processor(batch):
        if flaky in set(batch[UNIT_ID]):
            seen["n"] += 1
            if seen["n"] == 1 or len(batch) > 1:
                raise ValueError("transient")
        return inner(batch)

    result = run(cfg, planned, processor, clock, tmp_path, retry_errors=True)
    assert result.n_remaining == 0
    frame = final_rows(tmp_path, "out").set_index(UNIT_ID)
    assert (frame[STATUS] == Status.OK.value).all()


def test_the_run_summary_and_heartbeat_reach_the_bundle(cfg, tmp_path, capsys):
    import json, zipfile

    cfg = cfg.with_overrides({**FAST, "kaggle.heartbeat_minutes": 1})
    clock = FakeClock()
    result = run(cfg, make_units(100), make_processor(clock, 1.0), clock, tmp_path)
    assert "HEARTBEAT" in capsys.readouterr().out
    with zipfile.ZipFile(result.bundle_path) as z:
        summary = json.loads(z.read("diagnostics/run_summary.json"))
        assert "HEARTBEAT" in z.read("logs/worker.log").decode()
    assert summary["units_processed_this_run"] == 100 and summary["status_counts_of_rows_written"] == {"ok": 100}


def test_a_single_run_job_is_one_shard_and_has_no_cutoff(cfg, tmp_path):
    from finfm.compute.shard import plan_shards, is_single_run

    cfg1 = cfg.with_overrides({"kaggle.single_run_jobs": ["reader__dbX__llm2"]})
    units = make_units(5000).assign(est_seconds=60.0)  # far more work than one session
    assert is_single_run(cfg1, "reader__dbX__llm2") and not is_single_run(cfg, "reader__dbX__llm2")
    shards = plan_shards("reader__dbX__llm2", units, cfg1)
    assert len(shards) == 1 and len(shards[0].unit_ids) == 5000 and not shards[0].explicit
    assert len(plan_shards("reader__dbX__llm2", units, cfg)) > 1


def test_a_failing_final_write_is_retried_and_then_falls_back_so_the_result_is_never_lost(cfg, tmp_path, monkeypatch):
    import finfm.compute.bundle as bundle_module

    cfg = cfg.with_overrides({**FAST, "kaggle.final_write_wait_seconds": 0, "kaggle.checkpoint_minutes": 100000})
    real = bundle_module.BundleWriter.checkpoint
    calls = {"n": 0}

    def flaky(self, **kw):
        calls["n"] += 1
        if self.out_dir == tmp_path / "out":
            raise OSError("disk hiccup" if calls["n"] < 3 else "disk full")
        return real(self, **kw)

    monkeypatch.setattr(bundle_module.BundleWriter, "checkpoint", flaky)
    clock = FakeClock()
    result = run(cfg, make_units(20), make_processor(clock, 1.0), clock, tmp_path)
    assert calls["n"] > int(cfg.get("kaggle.final_write_attempts")) and result.bundle_path is not None
    assert result.bundle_path.exists() and result.bundle_path.parent != tmp_path / "out" and result.n_remaining == 0
