"""The worker processor of the final reader: a batch of units in, one row per unit out, whatever goes wrong inside.

Protections (each one keeps information instead of losing it):

* a unit whose prompt cannot be built becomes an ``error`` row with the exception text (the worker retries it);
* a unit with no usable candidate asset becomes a ``skipped`` row (``no_usable_pin``), exactly as in the earlier reader;
* if the engine raises for a batch, the batch is halved again and again until the failing unit is alone: one bad unit costs
  one row, never the batch;
* an answer that does not parse is generated again (a different seed and a higher temperature, because temperature 0 would
  repeat the same text) up to ``PARSE_RETRIES`` times; every attempt's outcome is recorded (``n_tries``) and the last raw
  output is always kept;
* the gate and analyst rules run in code (``reader2.rules``), never on trust.
"""
from __future__ import annotations

import hashlib
import json
import time
from collections import Counter
from dataclasses import replace
from typing import Any

import pandas as pd

from finfm.common.schema import (
    AVAILABLE_DATE, DETAIL, FAMILY, KIND, PARSE_OK, PINS_JSON, RAW_OUTPUT, REASON, STATUS, TEXT, TITLE, TOKENS_IN, TOKENS_OUT, UNIT_ID,
    Reason, Status,
)
from finfm.compute.engine import GenParams, GenRequest
from finfm.compute.vllm_engine import PROMPT_TOO_LONG
from finfm.reader.output_v2 import FLAG_NAMES, LINK_TYPES, parse_reading_v2, reader_v2_json_schema
from finfm.reader.run import ReaderProcessor

from reader2 import settings as S
from reader2.prompting import prompt_settings, v2_prompt
from reader2.rules import finalize

TASK = "reader_v2"
FINISH = "finish_reason"
FAIL = "fail_reason"
DETAIL_MAX = 500
SEED_HEX_CHARS = 15


def retry_seed(unit_id: str, attempt: int) -> int:
    return int(hashlib.sha256(f"{unit_id}|{attempt}".encode()).hexdigest()[:SEED_HEX_CHARS], 16)


def error_row(unit: str, exc: BaseException | str, reason: str = Reason.EXCEPTION.value) -> dict[str, Any]:
    text = exc if isinstance(exc, str) else f"{type(exc).__name__}: {exc}"
    return {UNIT_ID: unit, STATUS: Status.ERROR.value, REASON: reason, DETAIL: text[:DETAIL_MAX]}


class ReaderV2Processor:
    def __init__(self, cfg, engine, maskers, tokenizer, *, names=None, total_units: int | None = None, log=print,
                 clock=time.monotonic) -> None:
        self.cfg, self.engine, self.log, self.clock = cfg, engine, log, clock
        self.names = names
        self.prompt_settings = prompt_settings()
        self.base = ReaderProcessor(cfg, engine, maskers, tokenizer)  # only its prompt builder is used
        self.params = GenParams(temperature=0.0, top_p=1.0, top_k=None, max_tokens=S.MAX_TOKENS,
                                json_schema=reader_v2_json_schema(cfg, True), seed=int(cfg.get("runtime.base_seed")))
        self.causes = list(cfg.get("causes.causes"))
        self.groups = dict(cfg.get("causes.groups"))
        self.total_units = total_units
        self.started = clock()
        self.counts: Counter = Counter()
        self.events: Counter = Counter()
        self.last_report = 0
        self.last_activity = clock()   # read by the hang watchdog in reader2.session

    # ------------------------------------------------------------------ generation with isolation
    def _generate(self, requests: list[GenRequest], params: GenParams, depth: int = 0) -> dict[str, Any]:
        """``unit_id -> GenResult`` or the exception that unit ended with. Never raises."""
        if not requests:
            return {}
        try:
            results = self.engine.generate(requests, params)
        except Exception as exc:
            if len(requests) == 1 or depth >= S.MAX_SPLIT_DEPTH:
                return {r.unit_id: exc for r in requests}
            self.counts["batch_splits"] += 1
            self.log(f"READER2 batch of {len(requests)} failed ({type(exc).__name__}: {str(exc)[:160]}); splitting")
            mid = len(requests) // 2
            return {**self._generate(requests[:mid], params, depth + 1), **self._generate(requests[mid:], params, depth + 1)}
        got = {r.unit_id: r for r in results}
        return {r.unit_id: got.get(r.unit_id, RuntimeError("engine returned no result for this unit")) for r in requests}

    # ------------------------------------------------------------------ one batch
    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        self.last_activity = self.clock()
        rows: dict[str, dict[str, Any]] = {}
        built, info, requests = {}, {}, []
        for _, event in batch.iterrows():
            unit = str(event[UNIT_ID])
            try:
                event = event.copy()
                for column in (TITLE, TEXT):  # a missing text must not become the word "None" in the prompt
                    event[column] = "" if pd.isna(event[column]) else str(event[column])
                prompt = v2_prompt(self.base._build(event), self.prompt_settings, self.names)
                family, title = str(event[FAMILY]), str(event[TITLE])
            except Exception as exc:
                rows[unit] = error_row(unit, exc)
                continue
            if not prompt.ref_to_node:  # nothing the reader may name: no call
                rows[unit] = {UNIT_ID: unit, STATUS: Status.SKIPPED.value, REASON: Reason.NO_USABLE_PIN.value, FAMILY: family}
                continue
            built[unit], info[unit] = prompt, (family, title)
            requests.append(GenRequest(unit, prompt.user, n=1, task=TASK, system=prompt.system))

        outcome = self._generate(requests, self.params)
        parsed = {u: self._parse(outcome[u], built[u], info[u]) for u in built}
        for attempt, temperature in enumerate(S.RETRY_TEMPERATURES[: S.PARSE_RETRIES], start=1):
            again = [u for u in built if self._needs_retry(outcome[u], parsed[u])]
            if not again:
                break
            self.counts["retried"] += len(again)
            retry_params = replace(self.params, temperature=temperature)
            retry_requests = [GenRequest(u, built[u].user, n=1, task=TASK, system=built[u].system, seed=retry_seed(u, attempt)) for u in again]
            for unit, result in self._generate(retry_requests, retry_params).items():
                if isinstance(result, BaseException):
                    continue  # keep the earlier outcome; it is recorded below
                outcome[unit] = result
                parsed[unit] = self._parse(result, built[unit], info[unit], tries=parsed[unit][1] + 1)

        for unit in built:
            result = outcome[unit]
            if isinstance(result, BaseException):
                rows[unit] = error_row(unit, result)
            else:
                reading, tries, audit = parsed[unit]
                rows[unit] = self._row(unit, info[unit][0], built[unit], result, reading, tries, audit)
        self._count(rows)
        self._maybe_report()
        self.last_activity = self.clock()
        return pd.DataFrame([rows[str(u)] for u in batch[UNIT_ID]])

    # ------------------------------------------------------------------ helpers
    def _parse(self, result, prompt, info, tries: int = 1):
        if isinstance(result, BaseException):
            return None, tries, {}
        family, title = info
        reading = parse_reading_v2(result.texts[0], prompt.ref_to_node, self.cfg, gate=True, enforce_gate=False)
        audit = finalize(reading, family, title, self.cfg) if reading.ok else {}
        return reading, tries, audit

    @staticmethod
    def _needs_retry(result, parsed) -> bool:
        if isinstance(result, BaseException) or result.finish == PROMPT_TOO_LONG:
            return False  # an engine error is retried by the worker; a prompt that cannot fit would fail again
        return not parsed[0].ok

    def _row(self, unit: str, family: str, prompt, result, reading, tries: int, audit: dict) -> dict[str, Any]:
        row: dict[str, Any] = {
            UNIT_ID: unit, STATUS: Status.OK.value, RAW_OUTPUT: result.texts[0], TOKENS_IN: result.tokens_in,
            TOKENS_OUT: result.tokens_out, FINISH: result.finish, PARSE_OK: bool(reading.ok), FAMILY: family, "n_tries": tries,
            "text_truncated": prompt.text_truncated, "n_pins_given": prompt.n_pins_given, "n_pins_dropped": prompt.n_pins_dropped,
            "n_masked": prompt.mask.n_replaced, "template_hash": prompt.template_hash,
        }
        row.update({f"flag_{name}": int(reading.flags[name]) for name in FLAG_NAMES})
        if not reading.ok:
            reason = PROMPT_TOO_LONG if result.finish == PROMPT_TOO_LONG else reading.fail_reason
            row.update({REASON: reason, FAIL: reason})
            return row
        cause_top = max(reading.cause, key=reading.cause.get)
        pins = json.dumps(reading.pins, sort_keys=True)
        row.update({
            "why": reading.why, "n_causes": reading.n_causes, "cause_top": cause_top, "is_other": cause_top == "other",
            "group_top": max(reading.groups, key=reading.groups.get),
            "model_cause_top": max(reading.model_cause, key=reading.model_cause.get) if reading.model_cause else None,
            "event_type": audit["event_type_model"], "event_type_final": audit["event_type_final"],
            "gate_enforced": audit["gate_enforced"], "rating_rule": audit["rating_rule"],
            "pins_json": pins, "reading_pins_json": pins, "hits_json": json.dumps(reading.hits, sort_keys=True),
            "certainty": float("nan"), "scheduled": None,  # not written by this reader; the legacy names exist for the adapter
        })
        row.update({f"pi_{c}": float(reading.cause.get(c, 0.0)) for c in self.causes})
        row.update({f"cause_{c}": float(reading.cause.get(c, 0.0)) for c in self.causes})
        row.update({f"gw_{g}": float(reading.groups.get(g, 0.0)) for g in self.groups})
        row.update({f"link_{t}": float(reading.link_values.get(t, 0.0)) for t in LINK_TYPES})
        return row

    def _count(self, rows: dict[str, dict]) -> None:
        for row in rows.values():
            self.counts["units"] += 1
            status = row[STATUS]
            if status == Status.ERROR.value:
                self.counts["errors"] += 1
            elif status == Status.SKIPPED.value:
                self.counts["skipped"] += 1
            elif status == Status.OK.value:
                self.counts["called"] += 1
                self.counts["tokens_in"] += int(row.get(TOKENS_IN) or 0)
                self.counts["tokens_out"] += int(row.get(TOKENS_OUT) or 0)
                if row.get(PARSE_OK):
                    self.counts["parsed"] += 1
                    self.counts["other"] += int(bool(row.get("is_other")))
                    self.events[row["event_type_final"]] += 1
                else:
                    self.counts["unparsed"] += 1

    def report(self) -> str:
        c = self.counts
        called = max(c["called"], 1)
        seconds = max(self.clock() - self.started, 1e-9)
        line = (f"READER2 units={c['units']} called={c['called']} skipped={c['skipped']} errors={c['errors']} "
                f"parsed={c['parsed'] / called:.3f} retried={c['retried']} other={c['other'] / max(c['parsed'], 1):.3f} "
                f"events={dict(self.events)} tokens_out/unit={c['tokens_out'] / called:.0f} splits={c['batch_splits']} "
                f"{c['units'] / seconds * 3600:.0f} units/h")
        if self.total_units:
            left = max(self.total_units - c["units"], 0)
            line += f" eta={left / max(c['units'] / seconds, 1e-9) / 3600:.1f}h"
        return line

    def _maybe_report(self) -> None:
        if self.counts["units"] - self.last_report >= S.REPORT_EVERY_UNITS:
            self.last_report = self.counts["units"]
            self.log(self.report())
