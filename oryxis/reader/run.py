"""The reader job: one LLM call per event, frozen as the reader tape (spec Part 4)."""

from __future__ import annotations

import json
from typing import Any

import pandas as pd

from oryxis.common.config import Config
from oryxis.common.schema import (
    AVAILABLE_DATE,
    DETAIL,
    FAMILY,
    KIND,
    NAN_DETECTED,
    PARSE_OK,
    PINS_JSON,
    RAW_OUTPUT,
    REASON,
    STATUS,
    TEXT,
    TITLE,
    TOKENS_IN,
    TOKENS_OUT,
    UNIT_ID,
    Reason,
    Status,
)
from oryxis.compute.engine import Engine, GenParams, GenRequest
from oryxis.reader.masking import MaskerCache
from oryxis.reader.output import Reading, parse_reading, reader_json_schema
from oryxis.reader.prompt import BuiltPrompt, Tokenizer, build_reader_prompt

TASK = "reader"
CAUSE_PREFIX = "cause_"
CERTAINTY = "certainty"
SCHEDULED = "scheduled"
READING_PINS = "reading_pins_json"
CAUSE_TOP = "cause_top"
TEXT_TRUNCATED = "text_truncated"
N_PINS_GIVEN = "n_pins_given"
N_PINS_DROPPED = "n_pins_dropped"
N_MASKED = "n_masked"
TEMPLATE_HASH = "template_hash"
FLAG_PREFIX = "flag_"


class ReaderProcessor:
    """Worker processor: batch of events in, one row per event out."""

    def __init__(
        self, cfg: Config, engine: Engine, maskers: MaskerCache, tokenizer: Tokenizer
    ) -> None:
        self.cfg = cfg
        self.engine = engine
        self.maskers = maskers
        self.tokenizer = tokenizer
        self.params = GenParams(
            temperature=float(cfg.get("reader.temperature")),
            top_p=float(cfg.get("reader.top_p")),
            top_k=cfg.get("reader.top_k"),
            max_tokens=int(cfg.get("reader.max_tokens")),
            json_schema=reader_json_schema(cfg),
            seed=int(cfg.get("runtime.base_seed")),
        )

    def _build(self, row: pd.Series) -> BuiltPrompt:
        masker = self.maskers.for_date(pd.Timestamp(row[AVAILABLE_DATE]))
        return build_reader_prompt(
            cfg=self.cfg,
            masker=masker,
            tokenizer=self.tokenizer,
            kind=str(row[KIND]),
            family=str(row[FAMILY]),
            available_date=str(row[AVAILABLE_DATE]),
            title=str(row[TITLE]),
            text=str(row[TEXT]),
            pins_json=str(row[PINS_JSON]),
        )

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        rows: dict[str, dict[str, Any]] = {}
        built: dict[str, BuiltPrompt] = {}
        requests: list[GenRequest] = []
        for _, event in batch.iterrows():
            unit = str(event[UNIT_ID])
            prompt = self._build(event)
            if not prompt.ref_to_node:  # nothing the reader may name: do not spend a call
                rows[unit] = {
                    UNIT_ID: unit,
                    STATUS: Status.SKIPPED.value,
                    REASON: Reason.NO_USABLE_PIN.value,
                }
                continue
            built[unit] = prompt
            requests.append(
                GenRequest(unit, prompt.user, n=1, task=TASK, system=prompt.system)
            )
        results = self.engine.generate(requests, self.params) if requests else []
        for result in results:
            unit, prompt = result.unit_id, built[result.unit_id]
            reading = parse_reading(result.texts[0], prompt.ref_to_node, self.cfg)
            rows[unit] = self._row(unit, prompt, result.texts[0], reading, result)
        return pd.DataFrame([rows[str(u)] for u in batch[UNIT_ID]])

    def _row(self, unit: str, prompt: BuiltPrompt, raw: str, reading: Reading, result) -> dict:
        row: dict[str, Any] = {
            UNIT_ID: unit,
            STATUS: Status.OK.value,
            DETAIL: None,
            RAW_OUTPUT: raw,
            TOKENS_IN: result.tokens_in,
            TOKENS_OUT: result.tokens_out,
            NAN_DETECTED: result.nan_suspect,
            TEXT_TRUNCATED: prompt.text_truncated,
            N_PINS_GIVEN: prompt.n_pins_given,
            N_PINS_DROPPED: prompt.n_pins_dropped,
            N_MASKED: prompt.mask.n_replaced,
            TEMPLATE_HASH: prompt.template_hash,
        }
        row.update(reading_columns(reading, self.cfg))
        return row


def reading_columns(reading: Reading, cfg: Config) -> dict[str, Any]:
    """The parsed-answer columns of a tape row (parse flags, cause probabilities, certainty, pins); NaN/None when the parse failed."""
    row: dict[str, Any] = {PARSE_OK: reading.ok, REASON: None if reading.ok else reading.fail_reason.value}
    for name, value in reading.flags.as_dict().items():
        row[f"{FLAG_PREFIX}{name}"] = value
    causes = list(cfg.get("causes.causes"))
    if reading.ok:
        for name in causes:
            row[f"{CAUSE_PREFIX}{name}"] = reading.cause[name]
        row[CAUSE_TOP] = max(reading.cause, key=reading.cause.get)
        row[CERTAINTY] = reading.certainty
        row[SCHEDULED] = reading.scheduled
        row[READING_PINS] = json.dumps(reading.pins, sort_keys=True)
    else:
        for name in causes:
            row[f"{CAUSE_PREFIX}{name}"] = float("nan")
        row[CAUSE_TOP] = None
        row[CERTAINTY] = float("nan")
        row[SCHEDULED] = None
        row[READING_PINS] = None
    return row
