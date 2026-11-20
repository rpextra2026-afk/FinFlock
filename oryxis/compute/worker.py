"""The shard worker: runs units, survives bad units, and always ends with ONE saved bundle.

Guarantees (experiment configuration sections 4.2, 6.1, 7 and 11):

* never runs past ``session_cutoff_minutes``: no batch starts that cannot finish before the
  final-write reserve, and the last bundle is written before the cutoff;
* a bad unit becomes a row with ``status=error`` and a reason, never a crash;
* resumes from every valid bundle attached as input, taking the union of completed units;
* rolling checkpoints every ``checkpoint_minutes``; the previous bundle is replaced only after the
  new one verifies;
* protects GPU time: stops early on a low parse rate, NaN logits or a high error rate.
"""

from __future__ import annotations

import tempfile
import time
from collections import Counter, deque
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from oryxis.common.config import Config
from oryxis.common.schema import (
    ATTEMPTS,
    DETAIL,
    NAN_DETECTED,
    PARSE_OK,
    REASON,
    STATUS,
    TERMINAL_STATUSES,
    UNIT_ID,
    AbortReason,
    Reason,
    ShardOutcome,
    Status,
)
from oryxis.common.units import SECONDS_PER_MINUTE
from oryxis.compute.bundle import (
    BundleWriter,
    ScanResult,
    part_sources,
    scan_bundles,
    unit_state,
)

DETAIL_MAX_CHARS = 500

Processor = Callable[[pd.DataFrame], pd.DataFrame]
HealthCheck = Callable[[pd.DataFrame, "WorkerSettings"], AbortReason | None]


class Clock(Protocol):
    def now(self) -> float:
        """Seconds on a monotonic clock."""


class MonotonicClock:
    def now(self) -> float:
        return time.monotonic()


class StopFlag:
    """Set from a signal handler (SIGTERM) to stop cleanly after the current batch."""

    def __init__(self) -> None:
        self._set = False

    def set(self) -> None:
        self._set = True

    def is_set(self) -> bool:
        return self._set


@dataclass(frozen=True)
class WorkerSettings:
    cutoff_seconds: float
    reserve_seconds: float
    checkpoint_seconds: float
    heartbeat_seconds: float
    final_write_attempts: int
    final_write_wait_seconds: float
    batch_units: int
    canary_units: int
    progress_units: int
    stop_on_parse_rate: bool
    min_parse_rate: float
    max_error_rate: float
    error_window_units: int
    max_attempts: int
    compression: str
    compress_level: int

    @classmethod
    def from_config(cls, cfg: Config) -> WorkerSettings:
        return cls(
            cutoff_seconds=float(cfg.get("kaggle.session_cutoff_minutes")) * SECONDS_PER_MINUTE,
            reserve_seconds=float(cfg.get("kaggle.final_write_reserve_minutes"))
            * SECONDS_PER_MINUTE,
            checkpoint_seconds=float(cfg.get("kaggle.checkpoint_minutes")) * SECONDS_PER_MINUTE,
            heartbeat_seconds=float(cfg.get("kaggle.heartbeat_minutes")) * SECONDS_PER_MINUTE,
            final_write_attempts=int(cfg.get("kaggle.final_write_attempts")),
            final_write_wait_seconds=float(cfg.get("kaggle.final_write_wait_seconds")),
            batch_units=int(cfg.get("runtime.batch_units")),
            canary_units=int(cfg.get("runtime.canary_units")),
            progress_units=int(cfg.get("runtime.progress_units")),
            stop_on_parse_rate=bool(cfg.get("runtime.stop_on_parse_rate")),
            min_parse_rate=float(cfg.get("runtime.min_parse_rate")),
            max_error_rate=float(cfg.get("runtime.max_error_rate")),
            error_window_units=int(cfg.get("runtime.error_window_units")),
            max_attempts=int(cfg.get("runtime.max_attempts")),
            compression=str(cfg.get("runtime.parquet_compression")),
            compress_level=int(cfg.get("runtime.bundle_compression_level")),
        )


@dataclass
class ShardResult:
    outcome: ShardOutcome
    abort_reason: AbortReason | None
    n_planned: int
    n_terminal: int
    n_remaining: int
    bundle_path: Path | None
    seconds: float
    rejected_inputs: list[tuple[Path, str]] = field(default_factory=list)


def parse_rate_check(rows: pd.DataFrame, settings: WorkerSettings) -> AbortReason | None:
    """Stop when the share of parsed answers falls below ``min_parse_rate`` (after the canary)."""
    if PARSE_OK not in rows.columns or not settings.stop_on_parse_rate:
        return None
    observed = rows[PARSE_OK].dropna()
    if len(observed) and float(observed.astype(bool).mean()) < settings.min_parse_rate:
        return AbortReason.LOW_PARSE_RATE
    return None


def nan_check(rows: pd.DataFrame, settings: WorkerSettings) -> AbortReason | None:
    """Stop when any unit reports NaN or inf logits."""
    if NAN_DETECTED in rows.columns and bool(rows[NAN_DETECTED].eq(True).any()):
        return AbortReason.NAN_DETECTED
    return None


DEFAULT_CANARY_CHECKS: tuple[HealthCheck, ...] = (parse_rate_check, nan_check)


def _error_row(unit_id: str, exc: BaseException) -> dict[str, Any]:
    return {
        UNIT_ID: unit_id,
        STATUS: Status.ERROR.value,
        REASON: Reason.EXCEPTION.value,
        DETAIL: f"{type(exc).__name__}: {exc}"[:DETAIL_MAX_CHARS],
    }


def process_batch(processor: Processor, batch: pd.DataFrame) -> pd.DataFrame:
    """Run the processor; on an exception retry unit by unit so one bad unit costs one row."""
    try:
        out = processor(batch)
    except Exception:
        pieces = []
        for _, row in batch.iterrows():
            single = row.to_frame().T.reset_index(drop=True)
            try:
                pieces.append(processor(single))
            except Exception as exc:
                pieces.append(pd.DataFrame([_error_row(str(row[UNIT_ID]), exc)]))
        out = pd.concat(pieces, ignore_index=True)
    out = out.copy()
    missing = set(batch[UNIT_ID]) - set(out[UNIT_ID])
    if missing:
        filler = pd.DataFrame(
            [
                {
                    UNIT_ID: u,
                    STATUS: Status.ERROR.value,
                    REASON: Reason.NO_OUTPUT.value,
                    DETAIL: "processor returned no row",
                }
                for u in sorted(missing)
            ]
        )
        out = pd.concat([out, filler], ignore_index=True)
    return out[out[UNIT_ID].isin(set(batch[UNIT_ID]))].reset_index(drop=True)


def _prepare_remaining(
    planned: Sequence[str], state: pd.DataFrame, settings: WorkerSettings
) -> tuple[list[str], dict[str, int], pd.DataFrame]:
    """Units still to run, their previous attempt counts, and rows that exhaust their attempts."""
    previous = {
        str(u): (str(s), int(a))
        for u, s, a in zip(state[UNIT_ID], state[STATUS], state[ATTEMPTS], strict=True)
    }
    remaining: list[str] = []
    attempts: dict[str, int] = {}
    exhausted: list[dict[str, Any]] = []
    for unit in planned:
        status, tries = previous.get(unit, (None, 0))
        if status in TERMINAL_STATUSES:
            continue
        if status == Status.ERROR.value and tries >= settings.max_attempts:
            exhausted.append(
                {
                    UNIT_ID: unit,
                    STATUS: Status.FAILED.value,
                    REASON: Reason.ATTEMPTS_EXHAUSTED.value,
                    ATTEMPTS: tries,
                }
            )
            continue
        remaining.append(unit)
        attempts[unit] = tries
    return remaining, attempts, pd.DataFrame(exhausted)


def run_shard(
    *,
    cfg: Config,
    job_id: str,
    shard_id: str,
    planned: pd.DataFrame,
    processor: Processor,
    input_dirs: Sequence[str | Path],
    out_dir: str | Path,
    meta: Mapping[str, Any],
    clock: Clock | None = None,
    stop_flag: StopFlag | None = None,
    checks: Sequence[HealthCheck] = DEFAULT_CANARY_CHECKS,
    session_offset_seconds: float = 0.0,
    retry_errors: bool = False,
) -> ShardResult:
    """Run one shard to completion, to the session limit, or until a guard stops it.

    With ``retry_errors`` the units that ended in an error are run again inside this same run (up to ``runtime.max_attempts`` tries each)
    instead of waiting for the next session: used by single-run jobs that have no next session.

    ``planned`` holds ``unit_id`` plus whatever payload the processor needs, in the fixed order of
    the shard. ``processor`` receives a batch frame and returns one row per unit with at least
    ``unit_id`` and ``status``.
    """
    if planned[UNIT_ID].duplicated().any():
        raise ValueError("planned units contain duplicate unit_id values")
    settings = WorkerSettings.from_config(cfg)
    clock = clock or MonotonicClock()
    stop_flag = stop_flag or StopFlag()
    started = clock.now()
    # the cutoff is counted from the start of the Kaggle session, so time already spent installing, downloading and loading is deducted
    stop_at = settings.cutoff_seconds - settings.reserve_seconds - float(session_offset_seconds)

    scan: ScanResult = scan_bundles(input_dirs, job_id)
    state = unit_state(scan.valid)
    planned_ids = [str(u) for u in planned[UNIT_ID]]
    remaining, attempts, exhausted = _prepare_remaining(planned_ids, state, settings)

    writer = BundleWriter(
        out_dir,
        job_id,
        shard_id,
        carried=part_sources(scan.valid),
        parents=[ref.bundle_id for ref in scan.valid],
        seq=scan.next_seq,
        meta={**meta, "config_hash": cfg.hash},
        compression=settings.compression,
        compress_level=settings.compress_level,
    )
    if len(exhausted):
        writer.add_rows(exhausted)

    payload = planned.set_index(UNIT_ID, drop=False)
    recent_status: deque[str] = deque(maxlen=settings.error_window_units)
    canary_rows: list[pd.DataFrame] = []  # kept only until the canary check, then dropped (a run of 250,000 units must not hold every row)
    status_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    parse_seen = parse_good = 0
    last_status: dict[str, str] = {}
    heartbeats: list[dict[str, Any]] = []
    retry_round = 0
    processed = 0
    canary_done = False
    next_progress = settings.progress_units
    slowest_batch = 0.0
    last_checkpoint = started
    next_heartbeat = started + settings.heartbeat_seconds
    outcome = ShardOutcome.COMPLETE
    abort: AbortReason | None = None
    log: list[str] = []
    queue = remaining
    cursor = 0

    def parse_rate() -> float | None:
        return parse_good / parse_seen if parse_seen else None

    def summary(n_left: int) -> dict[str, Any]:
        elapsed = clock.now() - started
        return {
            "units_planned": len(planned_ids), "units_processed_this_run": processed, "units_left_in_queue": n_left,
            "retry_round": retry_round, "status_counts_of_rows_written": dict(status_counts),
            "reason_counts_of_rows_written": dict(reason_counts), "parse_ok_of_observed": parse_good, "parse_observed": parse_seen,
            "parse_rate": parse_rate(), "elapsed_seconds": elapsed,
            "units_per_minute": processed / (elapsed / SECONDS_PER_MINUTE) if elapsed > 0 else None,
            "units_still_error": sum(v == Status.ERROR.value for v in last_status.values()), "heartbeats": heartbeats,
        }

    def snapshot(n_left: int) -> dict[str, Any]:
        return {
            "outcome": outcome.value,
            "abort_reason": abort.value if abort else None,
            "processed_this_session": processed,
            "remaining": n_left,
            "elapsed_seconds": clock.now() - started,
            "slowest_batch_seconds": slowest_batch,
            "rejected_input_bundles": [[str(p), why] for p, why in scan.rejected],
        }

    def save(n_left: int) -> Path:
        return writer.checkpoint(state=snapshot(n_left), diagnostics={"run_summary.json": summary(n_left)}, logs={"worker.log": "\n".join(log)})

    def heartbeat(n_left: int) -> None:
        minutes = (clock.now() - started) / SECONDS_PER_MINUTE
        done_pass = len(queue) - n_left
        rate = processed / minutes if minutes > 0 else 0.0
        rate_part = f"{rate:.1f} units/min, about {n_left / rate / SECONDS_PER_MINUTE:.1f} h left in this pass at that speed" if rate > 0 else "speed not known yet"
        shown = f"{parse_rate():.3f}" if parse_seen else "n/a"
        line = (f"HEARTBEAT {minutes:.0f} min | pass {retry_round + 1}: {done_pass}/{len(queue)} units | rows written: "
                f"{dict(status_counts)} | parse rate {shown} | units still in error {summary(n_left)['units_still_error']} | {rate_part}")
        print(line, flush=True)
        log.append(line)
        heartbeats.append({"minutes": round(minutes, 1), "processed": processed, "status_counts": dict(status_counts), "parse_rate": parse_rate()})

    while True:
        while cursor < len(queue):
            if stop_flag.is_set():
                outcome = ShardOutcome.TERMINATED
                break
            elapsed = clock.now() - started
            if elapsed >= stop_at or elapsed + slowest_batch > stop_at:
                outcome = ShardOutcome.SESSION_LIMIT
                break
            ids = queue[cursor : cursor + settings.batch_units]
            batch = payload.loc[ids].reset_index(drop=True)
            t0 = clock.now()
            out = process_batch(processor, batch)
            slowest_batch = max(slowest_batch, clock.now() - t0)
            cursor += len(ids)

            out[ATTEMPTS] = out[UNIT_ID].map(lambda u: attempts.get(str(u), 0) + 1).astype(int)
            is_error = out[STATUS] == Status.ERROR.value
            spent = is_error & (out[ATTEMPTS] >= settings.max_attempts)
            out.loc[spent, STATUS] = Status.FAILED.value
            attempts.update({str(u): int(a) for u, a in zip(out[UNIT_ID], out[ATTEMPTS], strict=True)})
            last_status.update({str(u): str(s) for u, s in zip(out[UNIT_ID], out[STATUS], strict=True)})
            writer.add_rows(out)
            processed += len(out)
            recent_status.extend(out[STATUS].tolist())
            status_counts.update(out[STATUS].astype(str).tolist())
            if REASON in out.columns:
                reason_counts.update(out[REASON].dropna().astype(str).tolist())
            if PARSE_OK in out.columns:
                flags = out[PARSE_OK].dropna().astype(bool)
                parse_seen += len(flags)
                parse_good += int(flags.sum())
            if not canary_done:
                canary_rows.append(out)

            window_full = len(recent_status) >= settings.error_window_units
            if window_full:
                error_share = sum(
                    s in (Status.ERROR.value, Status.FAILED.value) for s in recent_status
                ) / len(recent_status)
                if error_share > settings.max_error_rate:
                    outcome, abort = ShardOutcome.ABORTED, AbortReason.ERROR_RATE
                    break
            if processed >= next_progress:  # live line for whoever watches the log: running parse rate of the units answered so far
                next_progress += settings.progress_units
                shown = f"{parse_rate():.3f}" if parse_seen else "n/a"
                print(f"PROGRESS {processed}/{len(queue)} units | parse rate so far {shown} (warn below {settings.min_parse_rate}) | "
                      f"{(clock.now() - started) / SECONDS_PER_MINUTE:.0f} min", flush=True)
            if not canary_done and processed >= settings.canary_units:
                canary_done = True
                everything = pd.concat(canary_rows, ignore_index=True)
                canary_rows.clear()
                if PARSE_OK in everything.columns:
                    rate = float(everything[PARSE_OK].dropna().astype(bool).mean()) if everything[PARSE_OK].notna().any() else 1.0
                    if rate < settings.min_parse_rate:
                        log.append(f"WARNING parse rate {rate:.3f} is below {settings.min_parse_rate} after {processed} units"
                                   f" ({'stopping' if settings.stop_on_parse_rate else 'continuing, as configured'})")
                for check in checks:
                    reason = check(everything, settings)
                    if reason is not None:
                        outcome, abort = ShardOutcome.ABORTED, reason
                        break
                if abort is not None:
                    break
            if clock.now() >= next_heartbeat:
                next_heartbeat = clock.now() + settings.heartbeat_seconds
                heartbeat(len(queue) - cursor)
            if clock.now() - last_checkpoint >= settings.checkpoint_seconds:
                try:
                    save(len(queue) - cursor)
                except Exception as exc:  # a failed rolling checkpoint (disk hiccup) must not end a long run; the next one tries again
                    msg = f"WARNING checkpoint failed ({type(exc).__name__}: {exc}); continuing, the rows stay buffered"
                    print(msg, flush=True)
                    log.append(msg)
                last_checkpoint = clock.now()
        if outcome is not ShardOutcome.COMPLETE or not retry_errors:
            break
        again = [u for u, st in last_status.items() if st == Status.ERROR.value]
        if not again:
            break
        retry_round += 1
        queue, cursor = sorted(again), 0
        msg = f"RETRY pass {retry_round + 1}: {len(queue)} units ended in an error and are run again (each gets up to {settings.max_attempts} tries)"
        print(msg, flush=True)
        log.append(msg)

    n_left = len(queue) - cursor
    log.append(f"outcome={outcome.value} abort={abort.value if abort else None} left={n_left}")
    heartbeat(n_left)
    path = None
    for attempt in range(settings.final_write_attempts):  # the finished result must reach a file: retry, then a fallback folder
        try:
            path = save(n_left)
            break
        except Exception as exc:
            msg = f"WARNING final bundle write failed (try {attempt + 1}/{settings.final_write_attempts}): {type(exc).__name__}: {exc}"
            print(msg, flush=True)
            log.append(msg)
            time.sleep(settings.final_write_wait_seconds)
    if path is None:
        writer.out_dir = Path(tempfile.mkdtemp(prefix="oryxis_fallback_"))
        path = save(n_left)
        print(f"WARNING the output folder could not be written; the bundle was saved to {path}: copy it from there", flush=True)
    try:
        final_state = unit_state(scan_bundles([Path(out_dir), path.parent], job_id).valid)
        mine = final_state[final_state[UNIT_ID].isin(set(planned_ids))]
        planned_terminal = int(mine[STATUS].isin(TERMINAL_STATUSES).sum())
    except Exception as exc:  # only the count printed at the end: the bundle itself is already written
        log.append(f"count of finished units could not be read back: {exc!r}")
        planned_terminal = sum(st in TERMINAL_STATUSES for st in last_status.values()) + len(planned_ids) - len(remaining)
    return ShardResult(
        outcome=outcome,
        abort_reason=abort,
        n_planned=len(planned_ids),
        n_terminal=planned_terminal,
        n_remaining=len(planned_ids) - planned_terminal,
        bundle_path=path,
        seconds=clock.now() - started,
        rejected_inputs=scan.rejected,
    )
