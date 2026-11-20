"""Which units a run covers, how they are cut into shards, and how long that takes (an estimate, with its assumptions named)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd

from finfm.common.schema import UNIT_ID
from finfm.compute.shard import event_hash_bucket
from finfm.reader.prompt import ref_to_node_of

from reader2 import settings as S

PLAN_FILE = "RUN_PLAN.json"
UNITS_FILE = "units.parquet"
ALIASES_FILE = "aliases.parquet"
SHARD_HASH_CHARS = 8
SECONDS_PER_HOUR = 3600.0
POC_USER_TOKENS = 140          # tokens of the POC articles beyond the shared prompt prefix (mean tokens_in 1843 - prefix about 1700)


def shard_of(unit_id: str, n_shards: int) -> int:
    """Stable shard number 0..n-1 of a unit (interleaved by hash, so every shard has the same mix of families and dates)."""
    return int(hashlib.sha256(str(unit_id).encode()).hexdigest()[:SHARD_HASH_CHARS], 16) % n_shards


def piece_of(unit_id: str, n_pieces: int) -> int:
    """Piece number 0..n-1 of a unit for the 'rest' mode. A different hash from ``shard_of``: the unfinished work of a shard is all of one
    parity of ``shard_of``, so reusing it would pile a whole shard onto half of the pieces."""
    return int(hashlib.sha256(f"rest|{unit_id}".encode()).hexdigest()[:SHARD_HASH_CHARS], 16) % n_pieces


def rest_units(scoped: pd.DataFrame, finished: set[str], piece: int, n_pieces: int) -> pd.DataFrame:
    """Units of the run that no attached bundle has finished, cut into ``n_pieces`` disjoint pieces; the units of piece ``piece``."""
    if n_pieces < 1 or not 0 <= piece < n_pieces:
        raise ValueError(f"piece must be 0 .. {n_pieces - 1} of {n_pieces}")
    left = scoped[~scoped[UNIT_ID].astype(str).isin(finished)]
    mine = left[left[UNIT_ID].map(lambda u: piece_of(u, n_pieces) == piece)]
    return mine.sort_values(UNIT_ID, kind="mergesort").reset_index(drop=True)


def scope_units(units: pd.DataFrame, bucket: str, llm: str) -> pd.DataFrame:
    """The units of one (bucket, llm) run: all of them, or for tessera on llm1/llm3 the owner's shared half (event hash bucket below 50)."""
    scope = S.scope_of(bucket, llm)
    buckets = event_hash_bucket(units[UNIT_ID])
    return units[(buckets >= scope.hash_from) & (buckets < scope.hash_below)].reset_index(drop=True)


def single_part(n_shards: int, scope: S.RunScope | None = None) -> list[dict]:
    scope = scope or S.RunScope("", "")
    return [{"hash_from": scope.hash_from, "hash_below": scope.hash_below, "n_shards": int(n_shards)}]


def run_parts(run_plan: dict, bucket: str, llm: str) -> list[dict]:
    """The shard blocks of a run: consecutive blocks of shards, each over the units whose event hash bucket lies in [hash_from, hash_below)."""
    return run_plan.get("parts") or single_part(run_plan["n_shards"], S.scope_of(bucket, llm))


def total_shards(parts: list[dict]) -> int:
    return sum(int(p["n_shards"]) for p in parts)


def shard_units(scoped: pd.DataFrame, parts: list[dict], shard: int | None) -> pd.DataFrame:
    """The units of one shard in a fixed order (unit id). One block with one shard is the whole run. Inside a block the units are
    dealt to its shards by hash, so every shard has the same mix of families and dates; blocks keep the owner's hash halves apart."""
    total = total_shards(parts)
    if total == 1:
        return scoped.sort_values(UNIT_ID, kind="mergesort").reset_index(drop=True)
    if shard is None or not 0 <= int(shard) < total:
        raise ValueError(f"this run has {total} shards: set SHARD to a number from 0 to {total - 1}")
    k = int(shard)
    for part in parts:
        n = int(part["n_shards"])
        if k < n:
            buckets = event_hash_bucket(scoped[UNIT_ID])
            chosen = scoped[(buckets >= int(part["hash_from"])) & (buckets < int(part["hash_below"]))]
            mine = chosen if n == 1 else chosen[chosen[UNIT_ID].map(lambda u: shard_of(u, n) == k)]
            return mine.sort_values(UNIT_ID, kind="mergesort").reset_index(drop=True)
        k -= n
    raise ValueError("shard number out of range")  # unreachable: total was checked above


def has_candidates(units: pd.DataFrame, cfg) -> pd.Series:
    return units["pins_json"].map(lambda s: bool(ref_to_node_of(s, cfg)))


def unit_seconds(units: pd.DataFrame, llm: str, decode_speedup: float = 1.0) -> pd.Series:
    """Estimated GPU seconds per unit (Olmo seconds scaled by the LLM factor). Assumptions are in ``settings``; only the Olmo
    base figure was measured, on short news items with batch 8, so this is deliberately on the slow side. ``decode_speedup`` > 1 gives
    the hoped-for figure when batches of 64 decode faster than the measured batches of 8 (an assumption, not a measurement)."""
    tokens = (units["text"].fillna("").str.len() / S.CHARS_PER_TOKEN).clip(upper=S.MAX_TEXT_TOKENS) + units["title"].fillna("").str.len() / S.CHARS_PER_TOKEN
    extra = (tokens + S.PROMPT_FIXED_TOKENS - POC_USER_TOKENS).clip(lower=0)
    return (S.MEASURED_SECONDS_PER_UNIT_LLM2 / decode_speedup + extra / S.PREFILL_TOKENS_PER_SECOND) * S.LLM_SPEED_FACTOR[llm]


def shard_count(gpu_hours: float) -> int:
    return max(1, int(-(-gpu_hours // S.TARGET_SHARD_HOURS)))


def load_plan(pack_root: str | Path) -> dict:
    return json.loads((Path(pack_root) / PLAN_FILE).read_text(encoding="utf-8"))
