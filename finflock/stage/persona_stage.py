"""Persona and bio jobs: prebuilt tables in the market dataset -> shard folders inside the session.

Prebuilt per bucket (``prestaged/personas-<bucket>/``): the persona unit table, the event payloads, every quarter's skeletons and the
states. Prebuilt once for all buckets (``prestaged/bios-shared/``): the bio unit table and the skeletons it covers. The bios themselves
are written by the LLM in the bios notebook; the persona notebooks read them from that notebook's output bundle.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from finflock.common.config import Config
from finflock.common.schema import ATTEMPTS, STATUS, UNIT_ID, Status
from finflock.common.units import SECONDS_PER_HOUR
from finflock.compute import payloads as pl
from finflock.compute.bundle import make_job_id, read_consolidated, scan_bundles, unit_state
from finflock.compute.session import SessionError
from finflock.compute.shard import EST_SECONDS, Shard, plan_shards, select_shards
from finflock.compute.staging import stage_shard
from finflock.personas.bios import BIO_TEXT
from finflock.personas.run import UNIT_SEPARATOR, EventPayload, PersonaState
from finflock.personas.skeletons import Skeleton

PRESTAGED = "prestaged"
UNITS_FILE = "units.parquet"
BIOS_JOB, SHARED = "bios", "shared"


def save_persona_inputs(folder: Path, units: pd.DataFrame, payloads: dict[str, EventPayload], skeletons: list[Skeleton],
                        states: dict[tuple[str, str], PersonaState]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    units.to_parquet(folder / UNITS_FILE, index=False)
    pl.save_events(payloads, folder)
    pl.save_skeletons(skeletons, folder)
    pl.save_states(states, folder)


def bio_units(skeletons: list[Skeleton], cfg: Config) -> pd.DataFrame:
    """One bio per persona and quarter: unit ``<persona>|<quarter>``."""
    ids = [f"{s.persona_id}{UNIT_SEPARATOR}{s.asof_quarter}" for s in skeletons]
    return pd.DataFrame({UNIT_ID: ids, EST_SECONDS: float(cfg.get("runs.est_seconds_per_unit.bios"))})


def bios_from_bundles(input_root: Path, llm: str, skeletons: dict[str, Skeleton]) -> dict[str, str]:
    """``bio_id -> text`` from the bios notebook's output bundle(s) attached to the session."""
    refs = scan_bundles([input_root], make_job_id(BIOS_JOB, SHARED, llm)).valid
    if not refs:
        raise SessionError(f"no bios found: run the notebook {BIOS_JOB}-{SHARED}-{llm} first and attach its output dataset to this notebook")
    rows = read_consolidated(refs)
    text = dict(zip(rows[UNIT_ID], rows[BIO_TEXT], strict=True))
    out, missing = {}, 0
    for s in skeletons.values():
        written = text.get(f"{s.persona_id}{UNIT_SEPARATOR}{s.asof_quarter}")
        if isinstance(written, str) and written:
            out[s.bio_id] = written
        else:
            missing += 1
    if missing:
        raise SessionError(f"{missing} personas have no bio in the attached bios bundle; finish the bios notebook first")
    return out


def _stage_persona_shard(shard: Shard, units: pd.DataFrame, skeletons: dict, payloads: dict, states: dict, bios: dict,
                         bucket: str, llm: str, cfg: Config, out_root: Path) -> Path:
    """One persona shard folder: only the events, states, skeletons and bios its units use."""
    pairs = [u.split(UNIT_SEPARATOR, 1) for u in shard.unit_ids]
    events_used, personas_used = {e for e, _ in pairs}, {p for _, p in pairs}

    def write(folder: Path) -> None:
        chosen = [s for s in skeletons.values() if s.persona_id in personas_used]
        pl.save_events({e: payloads[e] for e in events_used}, folder)
        pl.save_skeletons(chosen, folder)
        pl.save_bios({s.bio_id: bios[s.bio_id] for s in chosen}, folder)
        pl.save_states({k: v for k, v in states.items() if k[0] in events_used and k[1] in personas_used}, folder)

    return stage_shard(out_root, "personas", bucket, llm, shard, units, write, cfg)


def _persona_inputs(bucket: str, llm: str, cfg: Config, market_dir: Path, input_root: Path):
    pre = market_dir / PRESTAGED / f"personas-{bucket}"
    if not (pre / UNITS_FILE).exists():
        raise SessionError(f"the market dataset has no prebuilt persona inputs for {bucket} ({pre})")
    units = pd.read_parquet(pre / UNITS_FILE).assign(**{EST_SECONDS: float(cfg.get("runs.est_seconds_per_unit.personas"))})  # current config speed
    skeletons, payloads, states = pl.load_skeletons(pre), pl.load_events(pre), pl.load_states(pre)
    bios = bios_from_bundles(input_root, str(cfg.get("llms.primary")), skeletons)  # one shared bio set, written once by the primary LLM, for every LLM
    return units, skeletons, payloads, states, bios


def stage_persona_job(bucket: str, llm: str, cfg: Config, market_dir: Path, input_root: Path, out_root: Path,
                      log: Callable[[str], None] = print, only_shard: str | int | None = None) -> list[Path]:
    """Shard folders of the persona job; each carries only the events, states and skeletons its units use."""
    units, skeletons, payloads, states, bios = _persona_inputs(bucket, llm, cfg, market_dir, input_root)
    shards = select_shards(plan_shards(make_job_id("personas", bucket, llm), units, cfg), only_shard)
    log(f"{bucket}: {len(units)} persona units, staging {len(shards)} shard(s)")
    return [_stage_persona_shard(shard, units, skeletons, payloads, states, bios, bucket, llm, cfg, out_root) for shard in shards]


RETRY_SHARD = "retry"


def error_unit_ids(state: pd.DataFrame, max_attempts: int) -> list[str]:
    """Units whose latest row is an error that still has attempts left (the ones a worker would run again), in unit-id order."""
    open_errors = state[(state[STATUS] == Status.ERROR.value) & (state[ATTEMPTS] < max_attempts)]
    return sorted(open_errors[UNIT_ID].astype(str))


def stage_persona_retry(bucket: str, llm: str, cfg: Config, market_dir: Path, input_root: Path, out_root: Path,
                        log: Callable[[str], None] = print, part: int = 0, parts: int = 1) -> list[Path]:
    """ONE shard (``retry``) holding the units that the attached bundles of this job still have as a retryable error, whatever shard they came from.

    With ``parts`` > 1 the failed events are dealt out in turn (event i goes to part i mod parts) so several accounts each take ``part`` and no event is split.
    The session runs only its units, and its single output bundle also carries every attached bundle's rows.
    """
    job_id = make_job_id("personas", bucket, llm)
    refs = scan_bundles([input_root], job_id).valid
    if not refs:
        raise SessionError(f"no bundle of {job_id} is attached: attach the output bundles of the persona shards that have errors")
    ids = error_unit_ids(unit_state(refs), int(cfg.get("runtime.max_attempts")))
    if not 0 <= part < parts:
        raise SessionError(f"PART must be from 0 to {parts - 1} (PARTS = {parts}), got {part}")
    events = sorted({u.split(UNIT_SEPARATOR, 1)[0] for u in ids})
    mine = {e for i, e in enumerate(events) if i % parts == part}
    ids = [u for u in ids if u.split(UNIT_SEPARATOR, 1)[0] in mine]
    if not ids:
        raise SessionError(f"the {len(refs)} attached bundles of {job_id} hold no retryable error units; nothing to do")
    units, skeletons, payloads, states, bios = _persona_inputs(bucket, llm, cfg, market_dir, input_root)
    seconds = float(cfg.get("runs.est_seconds_per_unit.personas"))
    hours = len(ids) * seconds / SECONDS_PER_HOUR
    shard = Shard(RETRY_SHARD if parts == 1 else f"{RETRY_SHARD}{part}", job_id, tuple(ids), hours, hours, explicit=True)
    log(f"{bucket}: {len(refs)} bundles attached, {len(ids)} error units to retry in part {part + 1} of {parts} ({len(events)} failed events in all)")
    return [_stage_persona_shard(shard, units, skeletons, payloads, states, bios, bucket, llm, cfg, out_root)]


def stage_bios_job(llm: str, cfg: Config, market_dir: Path, out_root: Path, log: Callable[[str], None] = print) -> list[Path]:
    pre = market_dir / PRESTAGED / f"{BIOS_JOB}-{SHARED}"
    if not (pre / UNITS_FILE).exists():
        raise SessionError(f"the market dataset has no prebuilt bio inputs ({pre})")
    units, skeletons = pd.read_parquet(pre / UNITS_FILE), pl.load_skeletons(pre)
    shards = plan_shards(make_job_id(BIOS_JOB, SHARED, llm), units, cfg)
    log(f"{len(units)} bio units in {len(shards)} shard(s)")
    return [stage_shard(out_root, BIOS_JOB, SHARED, llm, shard, units, lambda folder: pl.save_skeletons(list(skeletons.values()), folder), cfg)
            for shard in shards]
