"""GPU budget calculator (spec 2.6): hours per job from measured throughput and the sample sizes.

``hours = (N prompts x uncached input tokens / prefill_tps + N prompts x draws x output tokens / decode_tps) / 3600``
with the throughput that was measured for the LLM and job type. Capacity per week is accounts x quota x safety.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import pandas as pd

from oryxis.common.config import Config
from oryxis.common.units import SECONDS_PER_HOUR
from oryxis.compute.shard import session_budget, weekly_cap_hours


@dataclass(frozen=True)
class JobSize:
    name: str
    bucket: str
    llm: str
    n_prompts: int
    uncached_input_tokens: float
    output_tokens: float
    draws: int = 1


def job_hours(size: JobSize, throughput: Mapping[str, float]) -> float:
    """Projected GPU hours of one job from the measured prefill and decode speeds."""
    prefill = size.n_prompts * size.uncached_input_tokens / throughput["prefill_tps_uncached"]
    decode = size.n_prompts * size.draws * size.output_tokens / throughput["decode_tps"]
    return (prefill + decode) / SECONDS_PER_HOUR


def budget_table(sizes: list[JobSize], throughput_by_llm: Mapping[str, Mapping[str, float]], cfg: Config) -> pd.DataFrame:
    """One row per job: hours, shards of one session, and sessions including model loading."""
    session = session_budget(cfg)
    rows = []
    for size in sizes:
        hours = job_hours(size, throughput_by_llm[size.llm])
        shards = max(int(-(-hours // session.work_hours)), 1)
        rows.append({"job": size.name, "bucket": size.bucket, "llm": size.llm, "work_hours": hours, "shards": shards,
                     "session_hours": hours + shards * session.overhead_hours})
    return pd.DataFrame(rows)


def weeks_needed(table: pd.DataFrame, cfg: Config) -> dict[str, float]:
    """Total session hours against the weekly capacity of all accounts."""
    capacity = int(cfg.get("kaggle.accounts")) * weekly_cap_hours(cfg)
    total = float(table["session_hours"].sum())
    return {"total_session_hours": total, "weekly_capacity_hours": capacity, "weeks": total / capacity}
