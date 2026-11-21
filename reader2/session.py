"""One run of the final reader (one data bucket, one LLM, one shard): everything a notebook cell calls.

Safety nets, in the order they act:

1. Before any GPU time: wrong number of GPUs, missing weights, missing token for a gated model, a unit table that is not the one
   the pack was built from, a bundle from another reader version or prompt, too little free disk -> a clear stop (nothing lost).
2. A small throw-away smoke test: if fewer than half of the first answers parse, the run stops in minutes with the raw answers saved.
3. The production worker: per-unit error rows, retry of errors inside the run, rolling checkpoint bundles every few minutes
   (the old one is replaced only after the new one verifies), stop before the session cutoff, final write retried and then
   saved to a fallback folder, SIGTERM handled, circuit breaker when nearly every unit errors.
4. Inside a batch (``reader2.processor``): batch halving, parse retries, engine rebuild after a fatal engine error.
5. A restart with the newest bundle attached continues exactly where the last verified checkpoint stopped; a unit that is finished
   is never run again and never lost.
"""
from __future__ import annotations

import _thread
import contextlib
import json
import os
import re
import shutil
import signal
import tempfile
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from finflock.common.config import Config, load_config
from finflock.common.schema import FAMILY, PARSE_OK, STATUS, TERMINAL_STATUSES, UNIT_ID, Status
from finflock.compute.bundle import make_job_id, read_consolidated, scan_bundles, unit_state
from finflock.compute.factory import HFTokenizer, resolve_model_path
from finflock.compute.session import SessionError, check_machine, download_weights, ensure_runtime, link_weights
from finflock.compute.shard import units_fingerprint
from finflock.compute.worker import StopFlag, run_shard
from finflock.reader.masking import MaskerCache
from finflock.reader.prompt import WhitespaceTokenizer

from reader2 import settings as S
from reader2 import plan as P
from reader2.engine import EngineSupervisor, build_engine_v2
from reader2.processor import ReaderV2Processor
from reader2.prompting import asset_names, prompt_sha256

REST_SHARD = re.compile(r"^rest:(\d+)/(\d+)$")   # SHARD = "rest:2/4": piece 2 of 4 of everything the attached bundles have not finished
REPORT_FILE = "READER2_RUN_REPORT.json"
SMOKE_FAILED_FILE = "READER2_SMOKE_FAILED.json"
MIN_FREE_DISK_GB = 2.0
GB = 1e9
GATED_LLMS = ("llm1", "llm3")     # Hugging Face licence + token (or attached weights) needed; Olmo downloads without one
SMOKE_SHOWN = 5
RAW_SHOWN_CHARS = 1500
ROUND = 3


def base_overrides() -> dict[str, Any]:
    return {"kaggle.session_cutoff_minutes": S.SESSION_CUTOFF_MINUTES, "kaggle.checkpoint_minutes": S.CHECKPOINT_MINUTES,
            "kaggle.heartbeat_minutes": S.HEARTBEAT_MINUTES, "runtime.batch_units": S.BATCH_UNITS,
            "kaggle.final_write_reserve_minutes": S.FINAL_WRITE_RESERVE_MINUTES}


def signature(bucket: str, llm: str, units_sha: str) -> dict[str, Any]:
    return {"reader2_version": S.VERSION, "prompt_sha256": prompt_sha256(), "bucket": bucket, "llm": llm, "units_sha256": units_sha}


def check_resumable(input_dirs: list[Path], job_id: str, sig: dict[str, Any]) -> list[str]:
    """Bundles of this job that were made by another reader version, prompt, bucket or unit table must never be mixed in."""
    problems = []
    for ref in scan_bundles(input_dirs, job_id).valid:
        meta = ref.manifest.get("meta", {})
        for key, expected in sig.items():
            if meta.get(key) != expected:
                problems.append(f"{ref.path.name}: {key} is {meta.get(key)!r}, this run has {expected!r}")
    return problems


def quality(tape: pd.DataFrame) -> dict[str, Any]:
    """What a finished (or partial) tape says about the answers; for a human to read, never a reason to stop."""
    if tape.empty:
        return {"rows": 0}
    out: dict[str, Any] = {"rows": int(len(tape)), "status": {str(k): int(v) for k, v in tape[STATUS].value_counts().items()}}
    called = tape[tape[STATUS] == Status.OK.value]
    if len(called):
        parsed = called[PARSE_OK].map(lambda v: bool(v) if v == v else False)
        out["called"], out["parse_rate"] = int(len(called)), round(float(parsed.mean()), ROUND)
        ok = called[parsed]
        if len(ok):
            out["other_rate"] = round(float(ok["is_other"].astype(bool).mean()), ROUND)
            out["event_type_final"] = {str(k): int(v) for k, v in ok["event_type_final"].value_counts().items()}
            out["cause_top_top10"] = {str(k): int(v) for k, v in ok["cause_top"].value_counts().head(10).items()}
            out["pins_nonempty_rate"] = round(float(ok["pins_json"].map(lambda s: s != "[]").mean()), ROUND)
        out["tokens_out_mean"] = round(float(called["tokens_out"].mean()), 1)
        if "n_tries" in called:
            out["rows_needing_a_retry"] = int((called["n_tries"] > 1).sum())
        if FAMILY in called:
            out["by_family"] = {str(f): {"called": int(len(g)), "parse_rate": round(float(g[PARSE_OK].map(lambda v: bool(v) if v == v else False).mean()), ROUND)}
                                for f, g in called.groupby(FAMILY)}
        if "reason" in called:
            out["failure_reasons"] = {str(k): int(v) for k, v in called.loc[~called[PARSE_OK].fillna(False).astype(bool), "reason"].value_counts().items()}
    return out


def smoke_test(processor: ReaderV2Processor, planned: pd.DataFrame, done: set[str], out_dir: Path, log) -> dict[str, Any]:
    """Run a few real units and throw the answers away; stop the whole run if almost none parse."""
    todo = planned[~planned[UNIT_ID].astype(str).isin(done)]
    todo = todo[P.has_candidates(todo, processor.cfg)].head(S.SMOKE_UNITS)
    if todo.empty:
        return {"n": 0}
    started = time.monotonic()
    rows = processor(todo.reset_index(drop=True))
    called = rows[rows[STATUS] == Status.OK.value]
    rate = float(called[PARSE_OK].fillna(False).astype(bool).mean()) if len(called) else 0.0
    result = {"n": int(len(todo)), "called": int(len(called)), "parse_rate": round(rate, ROUND), "seconds": round(time.monotonic() - started, 1)}
    ok = called[called[PARSE_OK].fillna(False).astype(bool)]
    if len(ok):
        result["other_rate"] = round(float(ok["is_other"].astype(bool).mean()), ROUND)
        result["event_type_final"] = {str(k): int(v) for k, v in ok["event_type_final"].value_counts().items()}
    log(f"READER2 smoke test: {result}")
    if result.get("other_rate", 0.0) > S.SMOKE_WARN_OTHER_RATE:
        log(f"READER2 WARNING: {result['other_rate']:.0%} of the smoke answers are 'other'. That is allowed (most news is not a listed event) but "
            f"look at the first live report lines; if every family shows 'other', this model is not following the prompt.")
    if rate < S.SMOKE_MIN_PARSE_RATE:
        sample = rows.head(SMOKE_SHOWN)
        shown = [{"unit_id": r[UNIT_ID], "status": r[STATUS], "reason": r.get("reason"), "detail": r.get("detail"),
                  "raw": str(r.get("raw_output"))[:RAW_SHOWN_CHARS]} for _, r in sample.iterrows()]
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / SMOKE_FAILED_FILE).write_text(json.dumps({"result": result, "first_rows": shown}, indent=1), encoding="utf-8")
        raise SessionError(f"smoke test failed: only {rate:.0%} of {len(called)} answers parsed (limit {S.SMOKE_MIN_PARSE_RATE:.0%}). "
                           f"No GPU hours were spent on the real work. The raw answers are in {SMOKE_FAILED_FILE}: send that file.")
    return result


class Watchdog:
    """Ends a hung run: first a KeyboardInterrupt in the main thread (so the engine can shut down), then, if the process is still
    alive after a grace period, ``os._exit``. A hung engine never raises by itself; without this a notebook would sit until
    Kaggle's 12 h limit, which can cost the whole output. The newest checkpoint bundle is already on disk either way."""

    CHECK_SECONDS = 30.0

    def __init__(self, processor: ReaderV2Processor, seconds: float, *, exit_fn=os._exit, interrupt_fn=_thread.interrupt_main, log=print,
                 poll: float | None = None, grace: float = S.HANG_GRACE_SECONDS) -> None:
        self.processor, self.seconds, self.exit_fn, self.interrupt_fn, self.log, self.grace = processor, seconds, exit_fn, interrupt_fn, log, grace
        self.poll = self.CHECK_SECONDS if poll is None else poll
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop, daemon=True)

    def loop(self) -> None:
        while not self.stop.wait(self.poll):
            idle = self.processor.clock() - self.processor.last_activity
            if idle > self.seconds:
                self.log(f"READER2 WATCHDOG: no batch finished for {idle / 60:.0f} min; the engine is hung. Stopping the run. The newest "
                         f"checkpoint bundle is in the output folder: start the notebook again with it attached.")
                self.interrupt_fn()
                if not self.stop.wait(self.grace):
                    self.exit_fn(S.HANG_EXIT_CODE)
                return

    def __enter__(self) -> Watchdog:
        self.thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self.stop.set()


def run_reader2(*, bucket: str, llm: str, pack_root: str | Path, config_dir: str | Path, input_root: str | Path, out_dir: str | Path,
                shard: int | str | None = None, overrides: dict | None = None, engine=None, tokenizer=None, clock=None, gpus: int | None = None,
                session_started: float | None = None, limit_units: int | None = None, stop_flag: StopFlag | None = None,
                skip_smoke: bool = False, hang_minutes: float | None = S.HANG_MINUTES, log=print) -> dict[str, Any]:
    if bucket not in S.BUCKETS or llm not in S.LLMS:
        raise SessionError(f"unknown run {bucket}/{llm}")
    pack_root, input_root, out_dir = Path(pack_root), Path(input_root), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg: Config = load_config(config_dir, {**base_overrides(), **(overrides or {})})
    plan = P.load_plan(pack_root)
    run_plan = plan["runs"].get(f"{bucket}-{llm}")
    if run_plan is None:
        raise SessionError(f"the pack has no plan for {bucket}-{llm}")
    parts = P.run_parts(run_plan, bucket, llm)
    n_shards = P.total_shards(parts)
    rest = None
    if isinstance(shard, str):
        match = REST_SHARD.match(shard.strip())
        if match:
            rest = (int(match.group(1)), int(match.group(2)))
            if rest[1] < 1 or rest[0] >= rest[1]:
                raise SessionError(f'SHARD = "{shard}" is not valid: write rest:J/M with J from 0 to M-1')
        elif shard.strip().isdigit():
            shard = int(shard)
        else:
            raise SessionError(f'SHARD = "{shard}" is not valid: use a shard number, None, or "rest:J/M"')
    if rest is None and n_shards > 1 and shard is None:
        raise SessionError(f"{bucket}-{llm} is split into {n_shards} shards: set SHARD to a number from 0 to {n_shards - 1} "
                           f"(one per Kaggle account, or one after the other)")
    free_gb = shutil.disk_usage(out_dir).free / GB
    if free_gb < MIN_FREE_DISK_GB:
        raise SessionError(f"only {free_gb:.1f} GB of disk is free in the output folder")

    units = pd.read_parquet(pack_root / "data" / bucket / P.UNITS_FILE)
    if units[UNIT_ID].duplicated().any():
        raise SessionError("duplicate unit ids in the unit table")
    units_sha = units_fingerprint(units[UNIT_ID])
    if units_sha != plan["buckets"][bucket]["units_sha256"]:
        raise SessionError("the unit table is not the one the pack plan was made for")
    scoped = P.scope_units(units, bucket, llm)
    if len(scoped) != int(run_plan["n_units"]):
        raise SessionError(f"{len(scoped)} units in scope, the plan says {run_plan['n_units']}")
    job_id = make_job_id(S.JOB, bucket, llm)
    if rest is not None:   # everything the ATTACHED bundles have not finished (not this session's own output), cut into pieces
        attached = unit_state(scan_bundles([input_root], job_id).valid)
        finished = set(attached.loc[attached[STATUS].isin(TERMINAL_STATUSES), UNIT_ID].astype(str)) if len(attached) else set()
        if not finished:
            raise SessionError('SHARD = "rest:J/M" continues unfinished work, but no bundle of this run is attached. Attach the dataset with the earlier '
                               '.bundle files (otherwise every session would take a share of the WHOLE run). Nothing was started.')
        planned = P.rest_units(scoped, finished, rest[0], rest[1])
        shard_id = f"rest{rest[0]}of{rest[1]}"
        log(f"READER2 rest mode: {len(scoped) - len(finished & set(scoped[UNIT_ID].astype(str)))} units of the run are unfinished in the attached bundles; "
            f"this session takes piece {rest[0]} of {rest[1]}: {len(planned)} units")
    else:
        planned = P.shard_units(scoped, parts, shard)
        shard_id = f"s{int(shard):04d}" if n_shards > 1 else "all"
    if limit_units:
        planned = planned.head(int(limit_units))
    sig = signature(bucket, llm, units_sha)
    problems = check_resumable([input_root, out_dir], job_id, sig)
    if problems:
        raise SessionError("a bundle that is attached or in the output folder does not belong to this run:\n  " + "\n  ".join(problems[:SMOKE_SHOWN]))

    stop = stop_flag or StopFlag()
    try:
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
    except ValueError:  # not in the main thread (some test runners)
        pass
    own_engine = engine is None
    with tempfile.TemporaryDirectory() as scratch:
        weights = None
        if own_engine and str(cfg.get("engine.engine")) != "mock":
            check_machine(cfg, llm, gpus)
            ensure_runtime(cfg, input_root, log=log)
            weights = link_weights(cfg, llm, input_root, Path(scratch))
            if weights is None:
                weights = download_weights(cfg, llm, Path(scratch), os.environ.get(str(cfg.get("sources.hf.env_token"))))
            if weights is None and llm in GATED_LLMS:
                raise SessionError(f"{llm} is a gated model: attach its weights as a dataset, or put your Hugging Face token (with the licence "
                                   f"accepted) in the token cell / Kaggle secret HF_TOKEN. Nothing was started.")
        aliases = pd.read_parquet(pack_root / "data" / bucket / P.ALIASES_FILE)
        maskers = MaskerCache(aliases, label_prefix=str(cfg.get("reader.masked_company_prefix")),
                              short_ticker_max_len=int(cfg.get("reader.short_ticker_max_len")))
        config_path = Path(config_dir)
        names = asset_names(cfg, pd.read_csv(config_path / "countries.csv"), pd.read_csv(config_path / "commodity_keywords.csv"))
        if tokenizer is None:
            tokenizer = HFTokenizer(resolve_model_path(cfg, str(cfg.get(f"engine.models.{llm}.path"))))
        supervisor = engine if engine is not None else EngineSupervisor(lambda: build_engine_v2(cfg, llm, str(config_dir)), log=log)
        state = unit_state(scan_bundles([input_root, out_dir], job_id).valid)
        done = set(state.loc[state[STATUS].isin(TERMINAL_STATUSES), UNIT_ID].astype(str)) if len(state) else set()
        todo = int((~planned[UNIT_ID].astype(str).isin(done)).sum())
        log(f"READER2 {bucket}-{llm} shard {shard_id}: {len(planned)} units planned, {len(planned) - todo} already finished, {todo} to do")
        started = time.time()
        try:
            smoke = {} if skip_smoke else smoke_test(ReaderV2Processor(cfg, supervisor, maskers, tokenizer, names=names, log=log),
                                                      planned, done, out_dir, log)
            processor = ReaderV2Processor(cfg, supervisor, maskers, tokenizer, names=names, total_units=todo, log=log)
            describe = supervisor.describe() if hasattr(supervisor, "describe") else {}
            meta = {**sig, "job": S.JOB, "shard": shard_id, "n_shards": n_shards, "engine": describe, "gated_families": list(S.GATED_FAMILIES),
                    "weights": str(weights) if weights else None, "smoke": smoke}
            with Watchdog(processor, hang_minutes * 60.0, log=log) if hang_minutes else contextlib.nullcontext():
                result = run_shard(cfg=cfg, job_id=job_id, shard_id=shard_id, planned=planned, processor=processor, input_dirs=[input_root, out_dir],
                                   out_dir=out_dir, meta=meta, clock=clock, stop_flag=stop, retry_errors=True,
                                   session_offset_seconds=max(time.time() - session_started, 0.0) if session_started else 0.0)
        finally:
            if own_engine:
                try:
                    supervisor.close()
                except Exception as exc:  # the result is saved; a failing shutdown must not turn a finished run into an error
                    log(f"WARNING engine shutdown failed after the run: {exc!r}")
    summary: dict[str, Any] = {"run": f"{bucket}-{llm}", "shard": shard_id, "outcome": result.outcome.value,
                               "abort": result.abort_reason.value if result.abort_reason else None, "planned": result.n_planned,
                               "terminal": result.n_terminal, "remaining": result.n_remaining, "bundle": str(result.bundle_path),
                               "seconds": round(result.seconds, 1), "wall_seconds": round(time.time() - started, 1),
                               "processor": processor.report(), "rejected_inputs": [[str(p), why] for p, why in result.rejected_inputs]}
    try:  # reporting only: the bundle is already saved
        tape = read_consolidated(scan_bundles([out_dir], job_id).valid)
        summary["quality"] = quality(tape[tape[UNIT_ID].isin(set(planned[UNIT_ID].astype(str)))])
    except Exception as exc:
        summary["quality"] = {"report_failed": repr(exc)}
    try:
        (out_dir / REPORT_FILE).write_text(json.dumps(summary, indent=1, default=str), encoding="utf-8")
    except OSError as exc:
        log(f"WARNING could not write {REPORT_FILE}: {exc!r}")
    log(json.dumps(summary, indent=1, default=str))
    return summary
