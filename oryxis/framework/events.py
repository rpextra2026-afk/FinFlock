"""Reader readings arranged per event: pins, group probabilities, certainty, source dials."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from oryxis.common.config import Config
from oryxis.common.schema import PARSE_OK, UNIT_ID
from oryxis.reader.run import CAUSE_PREFIX, CERTAINTY, READING_PINS, SCHEDULED
from oryxis.reader.salvage import (
    CERTAINTY_SOURCE,
    SALVAGE_STATUS,
    SALVAGED,
    SCHEDULED_SOURCE,
    SOURCE_MISSING,
    SOURCE_MODEL,
)

UNKNOWN_SCHEDULED = "unknown"


@dataclass(frozen=True)
class EventReading:
    event_id: str
    pins: tuple[dict, ...]  # node_id, direction, value_effect_pct, magnitude, role
    cause_probs: dict[str, float]
    group_probs: np.ndarray
    certainty: float  # NaN when the model never wrote it (salvaged rows); nothing downstream may substitute a value silently
    scheduled: str = UNKNOWN_SCHEDULED  # the reader's scheduled / surprise / unknown flag; "unknown" with source "missing" when never written
    certainty_source: str = SOURCE_MODEL  # model | filled (labelled sensitivity fill) | missing
    scheduled_source: str = SOURCE_MODEL
    salvaged: bool = False


def _source(frame_value, value_is_missing: bool) -> str:
    if value_is_missing:
        return SOURCE_MISSING
    return str(frame_value) if isinstance(frame_value, str) and frame_value else SOURCE_MODEL


def readings_from_tape(tape: pd.DataFrame, cfg: Config, include_salvaged: bool = False) -> dict[str, EventReading]:
    """One ``EventReading`` per event with a parsed reading and at least one pin.

    ``include_salvaged`` additionally reads rows the salvage step closed (``salvage_status == salvaged``); their certainty (and scheduled)
    come back as missing, never as a number. Salvaged rows need their node ids: read the tape through ``reparse_with_salvage``.
    """
    causes = list(cfg.get("causes.causes"))
    groups = cfg.get("causes.groups")
    usable = tape[PARSE_OK].fillna(False).astype(bool)
    if include_salvaged and SALVAGE_STATUS in tape.columns:
        usable = usable | (tape[SALVAGE_STATUS] == SALVAGED)
    out = {}
    for row in tape[usable].itertuples():
        pins = json.loads(getattr(row, READING_PINS) or "[]")
        if not pins:
            continue
        if any(pin.get("node_id") is None for pin in pins):
            raise ValueError(f"{getattr(row, UNIT_ID)}: a salvaged reading has no node ids; read the tape with reparse_with_salvage and the run's unit table")
        probs = {c: float(getattr(row, f"{CAUSE_PREFIX}{c}")) for c in causes}
        group = np.array([sum(probs[c] for c in members) for members in groups.values()])
        certainty = getattr(row, CERTAINTY)
        certainty_missing = certainty is None or (isinstance(certainty, float) and math.isnan(certainty))
        scheduled = getattr(row, SCHEDULED, None)
        scheduled_missing = not isinstance(scheduled, str) or not scheduled
        out[str(getattr(row, UNIT_ID))] = EventReading(
            str(getattr(row, UNIT_ID)), tuple(pins), probs, group, float("nan") if certainty_missing else float(certainty),
            UNKNOWN_SCHEDULED if scheduled_missing else scheduled,
            _source(getattr(row, CERTAINTY_SOURCE, None), certainty_missing), _source(getattr(row, SCHEDULED_SOURCE, None), scheduled_missing),
            bool(getattr(row, SALVAGE_STATUS, None) == SALVAGED),
        )
    return out


def pin_frame(readings: dict[str, EventReading], node_class: dict[str, str], kinds: dict[str, str],
              cfg: Config) -> pd.DataFrame:
    """Long frame of pin readings with the columns the source-dial maps use."""
    rows = []
    for event_id, reading in readings.items():
        for pin in reading.pins:
            row = {"event_id": event_id, "node_id": pin["node_id"], "value_effect_pct": pin["value_effect_pct"],
                   "magnitude": pin["magnitude"], "certainty": reading.certainty, "kind": kinds[event_id],
                   "scheduled": reading.scheduled, "certainty_source": reading.certainty_source, "scheduled_source": reading.scheduled_source,
                   "role": pin["role"], "node_class": node_class.get(pin["node_id"], "borrowed")}
            row.update({f"{CAUSE_PREFIX}{c}": p for c, p in reading.cause_probs.items()})
            rows.append(row)
    return pd.DataFrame(rows)
