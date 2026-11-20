"""Every tunable value of the final reader in one place (the repository's ``config/`` is left untouched on purpose:
changing it would change the config hash that earlier runs and bundles refer to)."""
from __future__ import annotations

from dataclasses import dataclass, field

JOB = "reader2"                      # job name in bundles: reader2__<bucket>__<llm> (the earlier reader is reader__...)
VERSION = "reader-v2-final-1"        # stamped into every bundle and every row; resuming across versions is refused
PROMPT_ARM = "gate_shots"            # the arm that won the final POC: worked examples + the event_type gate

# Items whose text is free news or chatter: the gate (market event or not) and the analyst-headline rule apply to them.
# Everything else (SEC filings, Federal Register, OFAC, GDELT spikes ...) is an event by construction: the model still
# answers event_type (the prompt and its cached prefix stay identical) but the answer is only recorded, never enforced.
GATED_FAMILIES = ("ext_company_news", "ext_world_news", "ext_social", "wikinews_news")
RATING_FAMILIES = ("ext_company_news", "ext_world_news")

# Units per run call, checkpoints and time. The values below are the owner's session rules; see kaggle.yaml.
BATCH_UNITS = 64
CHECKPOINT_MINUTES = 5
SESSION_CUTOFF_MINUTES = 570         # 9 h 30 min after the notebook starts (owner's choice for these runs); the worker stops starting batches FINAL_WRITE_RESERVE_MINUTES earlier
FINAL_WRITE_RESERVE_MINUTES = 15
HEARTBEAT_MINUTES = 10

# Model answer budget: the POC value (answers were 70-180 tokens).
MAX_TOKENS = 600

# Parse failures are retried before a unit is given up: temperature 0 would repeat the same bad answer.
PARSE_RETRIES = 2
RETRY_TEMPERATURES = (0.3, 0.7)

# Start-up smoke test: units run (and thrown away) before the real work, so a broken setup costs minutes, not hours.
SMOKE_UNITS = 24
SMOKE_MIN_PARSE_RATE = 0.5
SMOKE_WARN_OTHER_RATE = 0.9

# Engine supervision
ENGINE_REBUILDS = 3                  # whole-engine rebuilds in a row for one call after a fatal engine error
REBUILD_TOTAL_FACTOR = 10            # ... and at most this many times ENGINE_REBUILDS in one session
FATAL_ENGINE_MARKERS = ("EngineCore", "EngineDead", "replica", "CUDA error", "illegal memory access", "NCCL")
MAX_SPLIT_DEPTH = 6                  # a failing batch is halved up to this many times to isolate a bad unit

# A frozen engine raises nothing: if no batch finishes for this long the process exits, so the last checkpoint bundle survives
# (a notebook that hangs until Kaggle's 12 h limit would lose its output).
HANG_MINUTES = 45
HANG_EXIT_CODE = 4
HANG_GRACE_SECONDS = 120.0

# Live report: one status line every this many units (in addition to the worker's own heartbeat).
REPORT_EVERY_UNITS = 512

# Estimates used to plan shards (not enforced): measured on 2 x T4 with Olmo-3-7B in the final POC.
MEASURED_SECONDS_PER_UNIT_LLM2 = 2.18      # gate_shots arm, 283 news items, batch 8, short articles
PREFILL_TOKENS_PER_SECOND = 2000.0         # assumption (not measured): new prompt tokens processed per second
LLM_SPEED_FACTOR = {"llm2": 1.0, "llm3": 1.2, "llm1": 1.8}   # relative to Olmo; llm1/llm3 are ASSUMPTIONS (never run)
HOPED_BATCH_DECODE_SPEEDUP = 2.5           # ASSUMPTION for the 'hoped' column only: batches of 64 decode this much faster than the measured batches of 8
SESSION_OVERHEAD_HOURS = 0.5               # pip install + weights download + model load, per session
# GPU work that fits in one session: cutoff - final-write reserve - overhead = 8.75 h. Shards are cut to this (on the SAFE estimate).
TARGET_SHARD_HOURS = (SESSION_CUTOFF_MINUTES - FINAL_WRITE_RESERVE_MINUTES) / 60 - SESSION_OVERHEAD_HOURS
CHARS_PER_TOKEN = 4.0
MAX_TEXT_TOKENS = 1500                     # reader.max_text_tokens
PROMPT_FIXED_TOKENS = 150                  # per-unit wrapper (asset lines, glossary) beyond the title and text

# Which units each (bucket, llm) run covers. tessera: the owner's plan gives llm1 and llm3 the "shared half" only.
HASH_BUCKETS = 100
TESSERA_SHARED_HASH_BELOW = 50


@dataclass(frozen=True)
class RunScope:
    bucket: str
    llm: str
    hash_from: int = 0
    hash_below: int = HASH_BUCKETS


def scope_of(bucket: str, llm: str) -> RunScope:
    if bucket == "tessera" and llm in ("llm1", "llm3"):
        return RunScope(bucket, llm, 0, TESSERA_SHARED_HASH_BELOW)
    return RunScope(bucket, llm)


BUCKETS = ("db1", "tessera", "db3")
LLMS = ("llm1", "llm2", "llm3")


@dataclass
class Counters:
    """Running numbers for the live report (kept in the processor)."""
    units: int = 0
    called: int = 0
    skipped: int = 0
    errors: int = 0
    parsed: int = 0
    retried: int = 0
    other: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    events: dict = field(default_factory=dict)
    families: dict = field(default_factory=dict)
