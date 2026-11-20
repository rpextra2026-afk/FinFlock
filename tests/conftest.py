"""Shared test fixtures: the real config, a fake clock and a configurable test processor."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Collection
from pathlib import Path

import pandas as pd
import pytest

from oryxis.common.config import Config, load_config
from oryxis.common.schema import PARSE_OK, STATUS, UNIT_ID, Status

CONFIG_DIR = Path(__file__).resolve().parents[1] / "config"
HASH_HEX_CHARS = 8
HASH_RANGE = 16**HASH_HEX_CHARS


@pytest.fixture
def cfg() -> Config:
    return load_config(CONFIG_DIR)


class FakeClock:
    """Manually advanced monotonic clock."""

    def __init__(self) -> None:
        self.t = 0.0

    def now(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


class Crash(BaseException):
    """Simulates a hard process kill (not an ``Exception``, so the worker cannot absorb it)."""


def unit_value(unit_id: str) -> float:
    """Deterministic pseudo output of a unit, independent of how the work is sharded."""
    return int(hashlib.sha256(unit_id.encode()).hexdigest()[:HASH_HEX_CHARS], 16) / HASH_RANGE


def make_units(n: int, prefix: str = "u") -> pd.DataFrame:
    width = len(str(n))
    return pd.DataFrame({UNIT_ID: [f"{prefix}{i:0{width}d}" for i in range(n)]})


def make_processor(
    clock: FakeClock,
    seconds_per_unit: float,
    *,
    fail_ids: Collection[str] = (),
    parse_ok: bool = True,
    crash_after: int | None = None,
) -> Callable[[pd.DataFrame], pd.DataFrame]:
    state = {"done": 0}

    def processor(batch: pd.DataFrame) -> pd.DataFrame:
        rows = []
        for unit in batch[UNIT_ID]:
            clock.advance(seconds_per_unit)
            if unit in fail_ids:
                raise ValueError(f"bad unit {unit}")
            state["done"] += 1
            if crash_after is not None and state["done"] > crash_after:
                raise Crash()
            rows.append(
                {UNIT_ID: unit, STATUS: Status.OK.value, "value": unit_value(unit), PARSE_OK: parse_ok}
            )
        return pd.DataFrame(rows)

    return processor
