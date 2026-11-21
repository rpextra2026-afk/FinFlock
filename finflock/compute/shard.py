"""Shard planning and account assignment, driven entirely by ``config/kaggle.yaml``.

A shard is a contiguous list of units sized so that model loading, the work itself and the final
bundle write all finish before ``session_cutoff_minutes`` (Kaggle keeps no output from a session that
overruns). Assignment is static longest-processing-time-first onto accounts, with a weekly cap.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from finflock.common.config import Config
from finflock.common.schema import SHARD_ID_ALL, UNIT_ID
from finflock.data.buckets.events import HASH_BUCKETS, hash_bucket
from finflock.common.units import MINUTES_PER_HOUR, SECONDS_PER_HOUR

SHARD_ID_WIDTH = 4
ERROR_SAMPLE_SHARDS = 3
PLAN_PROBE_IDS = 50  # how many finished ids are looked up to tell "same table, changed" from "unrelated table"
EST_SECONDS = "est_seconds"
EVENT_SEPARATOR = "|"  # a persona unit id is <event id>|<persona id>; a reader unit id is the event id


class PlanningError(ValueError):
    """Raised when the configured limits cannot fit the requested work."""


@dataclass(frozen=True)
class SessionBudget:
    """How a single Kaggle session is divided (all in hours unless stated)."""

    cutoff_minutes: float
    load_minutes: float
    reserve_minutes: float
    safety_minutes: float
    work_hours: float
    overhead_hours: float
    min_shard_hours: float

    @property
    def session_hours_max(self) -> float:
        """Wall time one session can take: the cutoff (a longer shard is stopped there and resumed)."""
        return self.cutoff_minutes / MINUTES_PER_HOUR


def session_budget(cfg: Config) -> SessionBudget:
    """Work time available per session: cutoff minus model load, final write and safety."""
    cutoff = float(cfg.get("kaggle.session_cutoff_minutes"))
    platform_cap = float(cfg.get("kaggle.session_platform_cap_minutes"))
    load = float(cfg.get("kaggle.model_load_minutes"))
    reserve = float(cfg.get("kaggle.final_write_reserve_minutes"))
    safety = float(cfg.get("kaggle.shard_safety_minutes"))
    min_hours = float(cfg.get("kaggle.shard_min_hours"))
    if cutoff > platform_cap:
        raise PlanningError("session_cutoff_minutes exceeds the platform cap")
    work_minutes = float(cfg.get("kaggle.shard_work_minutes"))
    if work_minutes / MINUTES_PER_HOUR < min_hours:
        raise PlanningError(f"only {work_minutes:.0f} work minutes are allowed per shard, below shard_min_hours")
    return SessionBudget(
        cutoff_minutes=cutoff,
        load_minutes=load,
        reserve_minutes=reserve,
        safety_minutes=safety,
        work_hours=work_minutes / MINUTES_PER_HOUR,
        overhead_hours=(load + reserve + safety) / MINUTES_PER_HOUR,
        min_shard_hours=min_hours,
    )


@dataclass(frozen=True)
class Shard:
    shard_id: str
    job_id: str
    unit_ids: tuple[str, ...]
    work_hours: float
    session_hours: float
    explicit: bool = False  # True for a fixed plan: the account must name its shard, "first unfinished" is not allowed


PLAN_DIR = Path(__file__).resolve().parent / "shard_plans"


def shard_sort_key(shard_id: str) -> tuple[int, int, str]:
    """Numeric shard ids sort as numbers (2 before 10); anything else sorts as text after them."""
    return (0, int(shard_id), "") if str(shard_id).isdigit() else (1, 0, str(shard_id))


def load_fixed_plan(job_id: str) -> dict | None:
    """The fixed shard plan of a job (``compute/shard_plans/<job_id>.json``), or None for the usual time-based cutting.

    A plan names the shard numbers (``first_shard`` .. ``first_shard + n_shards - 1``) and the units that earlier runs
    already finished (``done_unit_ids``); the other units are cut into ``n_shards`` equal contiguous parts in unit-id order.
    """
    path = PLAN_DIR / f"{job_id}.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def units_fingerprint(unit_ids) -> str:
    """sha256 of the sorted unit ids: identifies a unit table, so a plan made for another table is never applied silently."""
    return hashlib.sha256("\n".join(sorted(str(u) for u in unit_ids)).encode()).hexdigest()


def _plan_fits(plan: dict, ordered: pd.DataFrame) -> bool:
    """True when the plan was made for this unit table. A plan with a fingerprint is for exactly one table: any other table of that job is an error.
    Older plans (no fingerprint): an unrelated table (a test, another sample) ignores the plan; the same table with a different size is an error,
    because shard numbers would silently mean other units."""
    if "units_sha256" in plan:
        if units_fingerprint(ordered[UNIT_ID]) != plan["units_sha256"]:
            raise PlanningError(f"{plan['job_id']}: the unit table is not the one the shard plan was made for (the tables were rebuilt?); "
                                "rebuild the plan with scripts/make_shard_plan.py before any account starts")
        return True
    if len(ordered) == int(plan["n_units_total"]):
        return True
    present = set(ordered[UNIT_ID].astype(str))
    if any(u in present for u in plan["done_unit_ids"][:PLAN_PROBE_IDS]):
        raise PlanningError(f"{plan['job_id']}: the unit table has {len(ordered)} units but the shard plan was made for "
                            f"{plan['n_units_total']}; rebuild the plan with scripts/make_shard_plan.py before any account starts")
    return False


def _cut_equal(job_id: str, todo: pd.DataFrame, count: int, first: int, budget: SessionBudget) -> list[Shard]:
    """``count`` equal contiguous parts of ``todo``, numbered from ``first``."""
    if len(todo) < count:
        raise PlanningError(f"{job_id}: only {len(todo)} unfinished units for {count} shards")
    base, extra = divmod(len(todo), count)
    shards, start = [], 0
    for i in range(count):
        size = base + (1 if i < extra else 0)
        part = todo.iloc[start:start + size]
        start += size
        hours = float(part[EST_SECONDS].sum()) / SECONDS_PER_HOUR
        shards.append(Shard(str(first + i), job_id, tuple(part[UNIT_ID].astype(str)), hours, min(hours + budget.overhead_hours, budget.session_hours_max), True))
    return shards


def event_hash_bucket(unit_ids: pd.Series) -> pd.Series:
    """Hash bucket 0-99 of each unit's event (the same bucket the event table carries), so reader and persona units of one event agree."""
    return unit_ids.astype(str).str.split(EVENT_SEPARATOR, n=1).str[0].map(hash_bucket)


def _fixed_shards(job_id: str, ordered: pd.DataFrame, plan: dict, budget: SessionBudget) -> list[Shard]:
    done = set(plan["done_unit_ids"])
    todo = ordered[~ordered[UNIT_ID].astype(str).isin(done)].reset_index(drop=True)
    first = int(plan["first_shard"])
    if "parts" in plan:  # consecutive blocks of shards, each over the units whose event hash bucket lies in [hash_from, hash_below)
        buckets = event_hash_bucket(todo[UNIT_ID])
        shards = []
        for part in plan["parts"]:
            chosen = todo[(buckets >= int(part.get("hash_from", 0))) & (buckets < int(part.get("hash_below", HASH_BUCKETS)))].reset_index(drop=True)
            shards += _cut_equal(job_id, chosen, int(part["n_shards"]), first + len(shards), budget)
    else:
        shards = _cut_equal(job_id, todo, int(plan["n_shards"]), first, budget)
    retry = [u for u in plan.get("retry_unit_ids", []) if u in set(ordered[UNIT_ID].astype(str))]
    if retry:  # units whose first answer did not parse: one extra numbered shard, run with NO earlier bundle attached
        part = ordered[ordered[UNIT_ID].astype(str).isin(set(retry))]
        hours = float(part[EST_SECONDS].sum()) / SECONDS_PER_HOUR
        shards.append(Shard(str(plan["retry_shard"]), job_id, tuple(part[UNIT_ID].astype(str)), hours, min(hours + budget.overhead_hours, budget.session_hours_max), True))
    return shards


def select_shards(shards: list[Shard], wanted: str | int | None) -> list[Shard]:
    """For a fixed plan keep only the shard this account asked for; other plans are returned unchanged."""
    if not any(s.explicit for s in shards):
        return shards
    ids = [s.shard_id for s in shards]
    if wanted is None:
        raise PlanningError(f"this run is split over several accounts: set SHARD to this account's shard number, one of {ids}")
    chosen = [s for s in shards if s.shard_id == str(wanted)]
    if not chosen:
        raise PlanningError(f"this run has no shard {wanted!r}; its shards are {ids}")
    return chosen


def is_single_run(cfg: Config, job_id: str) -> bool:
    """True for a job listed in ``kaggle.single_run_jobs``: one shard holding every unit, run without a session cutoff."""
    return job_id in set(cfg.get("kaggle.single_run_jobs") or [])


def plan_shards(
    job_id: str,
    units: pd.DataFrame,
    cfg: Config,
    *,
    sort_cols: Sequence[str] = (UNIT_ID,),
) -> list[Shard]:
    """Cut units into shards of at most ``work_hours``. Order is fixed by ``sort_cols``.

    ``units`` needs ``unit_id`` and ``est_seconds`` (projected GPU seconds per unit).
    """
    ordered = units.sort_values(list(sort_cols), kind="mergesort").reset_index(drop=True)
    if is_single_run(cfg, job_id):
        hours = float(ordered[EST_SECONDS].sum()) / SECONDS_PER_HOUR
        return [Shard(SHARD_ID_ALL, job_id, tuple(ordered[UNIT_ID].astype(str)), hours, hours)]
    budget = session_budget(cfg)
    plan = load_fixed_plan(job_id)
    if plan is not None and _plan_fits(plan, ordered):
        return _fixed_shards(job_id, ordered, plan, budget)
    limit = budget.work_hours * SECONDS_PER_HOUR
    too_big = ordered[ordered[EST_SECONDS] > limit]
    if len(too_big):
        raise PlanningError(
            f"{len(too_big)} unit(s) need more than one session of work, first: "
            f"{too_big[UNIT_ID].iloc[0]}"
        )
    shards: list[Shard] = []
    current: list[str] = []
    seconds = 0.0

    def close() -> None:
        hours = seconds / SECONDS_PER_HOUR
        shards.append(
            Shard(
                shard_id=f"s{len(shards):0{SHARD_ID_WIDTH}d}",
                job_id=job_id,
                unit_ids=tuple(current),
                work_hours=hours,
                session_hours=min(hours + budget.overhead_hours, budget.session_hours_max),
            )
        )

    for unit_id, est in zip(ordered[UNIT_ID], ordered[EST_SECONDS], strict=True):
        if current and seconds + est > limit:
            close()
            current, seconds = [], 0.0
        current.append(str(unit_id))
        seconds += float(est)
    if current:
        close()
    return shards


def account_ids(cfg: Config) -> list[str]:
    """Account labels: the owner's list, or generated ``acct01 ... acctNN``."""
    explicit = cfg.get("kaggle.account_ids")
    count = int(cfg.get("kaggle.accounts"))
    if explicit:
        if len(explicit) != count:
            raise PlanningError("kaggle.account_ids length differs from kaggle.accounts")
        return [str(a) for a in explicit]
    width = len(str(count))
    return [f"acct{i + 1:0{width}d}" for i in range(count)]


def weekly_cap_hours(cfg: Config) -> float:
    """Usable GPU hours per account per week: quota times the safety factor."""
    return float(cfg.get("kaggle.week_quota_hours")) * float(cfg.get("kaggle.quota_safety"))


def assign_shards(shards: Sequence[Shard], cfg: Config) -> pd.DataFrame:
    """Static assignment: longest shard first, onto the earliest (wave, account) with room.

    A wave is one weekly quota period. No account exceeds ``weekly_cap_hours`` in a wave.
    """
    accounts = account_ids(cfg)
    cap = weekly_cap_hours(cfg)
    oversized = [s.shard_id for s in shards if s.session_hours > cap]
    if oversized:
        raise PlanningError(f"shards longer than the weekly cap per account: {oversized[:ERROR_SAMPLE_SHARDS]}")
    load: dict[tuple[int, str], float] = {}
    rows = []
    for shard in sorted(shards, key=lambda s: (-s.session_hours, s.shard_id)):
        wave = 0
        while True:
            options = [
                (load.get((wave, acct), 0.0), idx, acct)
                for idx, acct in enumerate(accounts)
                if load.get((wave, acct), 0.0) + shard.session_hours <= cap
            ]
            if options:
                _, _, acct = min(options)
                load[(wave, acct)] = load.get((wave, acct), 0.0) + shard.session_hours
                rows.append(
                    {
                        "shard_id": shard.shard_id,
                        "job_id": shard.job_id,
                        "account": acct,
                        "wave": wave,
                        "work_hours": shard.work_hours,
                        "session_hours": shard.session_hours,
                        "n_units": len(shard.unit_ids),
                    }
                )
                break
            wave += 1
    return pd.DataFrame(rows).sort_values(["wave", "account", "shard_id"]).reset_index(drop=True)
