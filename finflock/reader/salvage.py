"""CPU-only repair of reader answers that ended in a whitespace flood.

Some answers are complete up to the closing ``]`` of ``pins`` (or up to the ``scheduled`` value) and then run on as spaces and newlines
until the token limit, so the JSON is never closed. ``complete_tail`` appends the missing keys and the closing brace to the model's own
text; it never changes a character the model wrote. The causes and pins of a repaired row are exactly the model's.

What the model did not write (``scheduled``, ``certainty``) is NOT stored as if it had. The salvaged tape keeps those cells empty and
carries flags. A row is ``salvaged`` only; it never becomes ``parse_ok``. Consumers choose explicitly: ``readings_from_tape(...,
include_salvaged=True)`` reads it with ``certainty_source = missing``, and any feature or policy that needs the missing value stops with an
error instead of filling it. ``apply_fill`` remains as a labelled opt-in for sensitivity tables; the final pipeline does not call it.
Rows that were parsed without help are never touched. Nothing here calls a model.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from finflock.common.config import Config
from finflock.common.schema import PARSE_OK, RAW_OUTPUT, REASON, STATUS, UNIT_ID, Status
from finflock.reader.output import KEY_CAUSE, KEY_CERTAINTY, KEY_PINS, KEY_SCHEDULED, PIN_NODE, PIN_REF, REQUIRED_KEYS, parse_reading
from finflock.reader.prompt import ref_to_node_of
from finflock.reader.reparse import reparse_tape
from finflock.reader.run import CERTAINTY, FLAG_PREFIX, N_PINS_GIVEN, READING_PINS, SCHEDULED, reading_columns

SALVAGE_STATUS = "salvage_status"
NATURAL, SALVAGED, UNREPAIRED, NOT_CALLED = "natural", "salvaged", "unrepaired", "not_called"
EXCLUDED_DEFAULTED = "excluded_defaulted_cause"
SALVAGE_MISSING = "salvage_missing"  # the keys the model never wrote, joined with "+"
MISSING_SEPARATOR = "+"
TAIL_WS_CHARS = "tail_whitespace_chars"
RAW_REPAIRED = "raw_repaired"  # the text that was parsed: the model's text plus the appended tail
FLAG_TAIL_COMPLETED = "flag_tail_completed"
FLAG_SCHEDULED_MISSING = "flag_scheduled_missing"
FLAG_CERTAINTY_MISSING = "flag_certainty_missing"
FLAG_PINS_TRUNCATED = "flag_pins_truncated"  # the pins list was cut at its last complete pin object (a loop or a cut inside the list)
SALVAGE_TIER = "salvage_tier"
TIER_TAIL, TIER_PINS_TRUNCATED = "tail_completed", "pins_truncated"
PIN_OBJECTS_KEPT = "pin_objects_kept"  # complete pin objects of the model's list that the repaired text keeps (repeats and unoffered refs are dropped later by the parser)
FLAG_NODE_PENDING = "flag_node_id_pending"  # pins carry refs (A1...) but no node id: the unit table of the run was not given
CERTAINTY_SOURCE = "certainty_source"
SCHEDULED_SOURCE = "scheduled_source"
SOURCE_MODEL, SOURCE_FILLED, SOURCE_MISSING = "model", "filled", "missing"
MEDIAN_PARSED = "median_parsed"
PLACEHOLDER_CERTAINTY = 0  # lets the repaired text pass the parser; the value is discarded and never stored
CLOSE = "}"
COMMA = ","
EXAMPLES_SHOWN = 3  # offending units listed in an alignment report
FLAG_CERTAINTY_CLAMPED = f"{FLAG_PREFIX}certainty_clamped"
FLAG_SCHEDULED_UNKNOWN = f"{FLAG_PREFIX}scheduled_unknown"


class AlignmentError(ValueError):
    """The unit table does not describe the same events and pins as the tape (a stale or different table)."""


@dataclass(frozen=True)
class TailRepair:
    body: str  # the model's text without its trailing whitespace (and without a dangling comma)
    text: str  # body + appended: valid JSON
    appended: str
    missing: tuple[str, ...]  # keys the model never wrote
    whitespace_chars: int  # trailing whitespace characters the model produced after its last real character
    pins_truncated: bool = False  # True: the model's pins list was cut back to its last complete object before closing
    pin_objects_kept: int | None = None


def _cut_to_last_pin(text: str) -> tuple[str, int] | None:
    """The answer cut back to the end of the last COMPLETE pin object of an unclosed ``pins`` list, plus ``]``; ``None`` when the text does not
    have the shape ``{"cause": {...}, "pins": [ {...}, ... <unfinished>``. Only characters after a complete object are removed."""
    decoder = json.JSONDecoder()
    match = re.match(r'\s*\{\s*"cause"\s*:\s*', text)
    if match is None:
        return None
    try:
        _, end = decoder.raw_decode(text, match.end())
    except json.JSONDecodeError:
        return None
    match = re.compile(r'\s*,\s*"pins"\s*:\s*\[').match(text, end)
    if match is None:
        return None
    last, kept, position = match.end(), 0, match.end()
    while True:
        position = len(text) - len(text[position:].lstrip())
        if text.startswith(COMMA, position) and kept:
            position = len(text) - len(text[position + 1:].lstrip())
        try:
            value, end = decoder.raw_decode(text, position)
        except json.JSONDecodeError:
            break
        if not isinstance(value, dict):
            return None
        last, kept, position = end, kept + 1, end
    return text[:last] + "]", kept


def complete_tail(raw: Any, cfg: Config, truncate_pins: bool = False) -> TailRepair | None:
    """Close an answer that stopped after a complete ``cause`` and ``pins`` (and possibly ``scheduled``). ``None`` when it cannot be done
    without guessing: cut inside a value, no pins key, keys out of order, or already complete.

    ``truncate_pins`` adds a second, separately flagged tier for answers whose ``pins`` list never closed (a repetition loop or a cut inside
    the list): the list is cut back to its last complete pin object. Pins after that point are lost, never written for the model."""
    if not isinstance(raw, str):
        return None
    body = raw.rstrip()
    whitespace_chars = len(raw) - len(body)
    if body.endswith(COMMA):  # a dangling comma is punctuation, not content
        body = body[: -len(COMMA)].rstrip()
    pins_truncated, kept = False, None
    try:
        partial = json.loads(body + CLOSE)  # the model's own keys, parsed as if it had stopped here
    except json.JSONDecodeError:
        cut = _cut_to_last_pin(body) if truncate_pins else None
        if cut is None:
            return None
        body, kept = cut
        pins_truncated = True
        try:
            partial = json.loads(body + CLOSE)
        except json.JSONDecodeError:
            return None
    if not isinstance(partial, dict):
        return None
    present = list(partial)
    if len(present) >= len(REQUIRED_KEYS) or present != list(REQUIRED_KEYS[: len(present)]):
        return None
    if KEY_PINS not in partial or not isinstance(partial[KEY_CAUSE], dict) or not isinstance(partial[KEY_PINS], list):
        return None
    missing = tuple(REQUIRED_KEYS[len(present):])
    fills = {KEY_SCHEDULED: str(cfg.get("cpu.reader_salvage.scheduled_fill")), KEY_CERTAINTY: PLACEHOLDER_CERTAINTY}
    appended = "".join(f", {json.dumps(key)}: {json.dumps(fills[key])}" for key in missing) + CLOSE
    text = body + appended
    try:
        whole = json.loads(text)
    except json.JSONDecodeError:
        return None
    if any(whole[key] != partial[key] for key in present):  # the model's content must come back unchanged
        return None
    return TailRepair(body, text, appended, missing, whitespace_chars, pins_truncated, kept)


def _placeholder_refs(n_pins: int, cfg: Config) -> dict[str, str]:
    """``A1 -> A1`` .. ``An -> An``: the refs the model was offered, for use when the unit table is not available (node ids stay empty)."""
    prefix = str(cfg.get("reader.ref_prefix"))
    return {f"{prefix}{index}": f"{prefix}{index}" for index in range(1, n_pins + 1)}


def _is_true(value: Any) -> bool:
    return bool(value) if value is not None and value == value else False


def verify_alignment(tape: pd.DataFrame, pins_json: Mapping[str, str], cfg: Config, expected_units: int | None = None) -> dict[str, Any]:
    """Does the unit table describe the tape? Checks unit sets, the number of offered pins of every called unit, and that every naturally
    parsed row's stored node ids equal the table's ref-to-node map. ``ok`` is true only when all hold (and the unit count matches
    ``expected_units`` when given)."""
    units = list(tape[UNIT_ID])
    in_table = [u in pins_json for u in units]
    missing_units = [u for u, ok in zip(units, in_table, strict=True) if not ok]
    mismatch_pins, mismatch_nodes, checked_pins, checked_nodes = [], [], 0, 0
    refs = {}
    for row in tape.itertuples():
        unit = getattr(row, UNIT_ID)
        if unit not in pins_json:
            continue
        refs[unit] = ref_to_node_of(pins_json[unit], cfg)
        status = getattr(row, STATUS)
        if status == Status.OK.value:
            checked_pins += 1
            if len(refs[unit]) != int(getattr(row, N_PINS_GIVEN)):
                mismatch_pins.append(unit)
        elif len(refs[unit]) != 0:  # a unit the run skipped for having no pin must have none in the table
            checked_pins += 1
            mismatch_pins.append(unit)
        stored = getattr(row, READING_PINS, None)
        if status == Status.OK.value and _is_true(getattr(row, PARSE_OK)) and isinstance(stored, str):
            for pin in json.loads(stored):
                checked_nodes += 1
                if refs[unit].get(pin[PIN_REF]) != pin[PIN_NODE]:
                    mismatch_nodes.append(unit)
                    break
    extra_in_table = len(set(pins_json) - set(units))
    result = {"tape_units": len(units), "table_units": len(pins_json), "tape_units_missing_from_table": len(missing_units),
              "table_units_missing_from_tape": extra_in_table, "units_checked_for_pin_count": checked_pins,
              "pin_count_mismatches": len(mismatch_pins), "natural_rows_checked_for_node_ids": int(checked_nodes),
              "node_id_mismatches": len(mismatch_nodes), "examples": {"missing": missing_units[:EXAMPLES_SHOWN], "pin_count": mismatch_pins[:EXAMPLES_SHOWN], "node": mismatch_nodes[:EXAMPLES_SHOWN]}}
    result["ok"] = bool(not missing_units and not mismatch_pins and not mismatch_nodes and (expected_units is None or len(units) == expected_units))
    return result


def salvage_tape(tape: pd.DataFrame, cfg: Config, pins_json: Mapping[str, str] | None = None, truncate_pins: bool | None = None) -> pd.DataFrame:
    """A copy of a reader tape where every ``invalid_json`` row that can be closed carries the model's own causes and pins.

    ``parse_ok`` and ``reason`` are left as the run recorded them, so any loader that trusts ``parse_ok`` still ignores salvaged rows.
    Salvaged rows get ``certainty`` empty (and ``scheduled`` empty when it was missing), ``salvage_status = salvaged`` and the flags.
    With ``pins_json`` (unit id -> the event's pins of the run's unit table) the pins carry their real node ids; a unit whose offered pin
    count differs from the tape raises ``AlignmentError``. Without it, node ids stay empty and ``flag_node_id_pending`` is set.
    """
    out = tape.copy()
    n = len(out)
    if truncate_pins is None:
        truncate_pins = bool(cfg.get("cpu.reader_salvage.truncate_pins"))
    out[SALVAGE_TIER] = None
    out[PIN_OBJECTS_KEPT] = np.nan
    out[SALVAGE_STATUS] = NOT_CALLED
    out[SALVAGE_MISSING] = None
    out[RAW_REPAIRED] = None
    out[TAIL_WS_CHARS] = np.nan
    for name in (FLAG_TAIL_COMPLETED, FLAG_PINS_TRUNCATED, FLAG_SCHEDULED_MISSING, FLAG_CERTAINTY_MISSING, FLAG_NODE_PENDING):
        out[name] = pd.Series([False] * n, index=out.index, dtype=bool)
    called = out[STATUS] == Status.OK.value
    parsed = out[PARSE_OK].map(_is_true)
    out.loc[called & parsed, SALVAGE_STATUS] = NATURAL
    out.loc[called & ~parsed, SALVAGE_STATUS] = UNREPAIRED  # until a repair succeeds
    for index in out.index[called & ~parsed]:
        repair = complete_tail(out.at[index, RAW_OUTPUT], cfg, truncate_pins)
        if repair is None:
            continue
        n_given = int(out.at[index, N_PINS_GIVEN])
        if pins_json is None:
            ref_map, nodes_known = _placeholder_refs(n_given, cfg), False
        else:
            unit = out.at[index, UNIT_ID]
            if unit not in pins_json:
                continue
            ref_map, nodes_known = ref_to_node_of(pins_json[unit], cfg), True
            if len(ref_map) != n_given:
                raise AlignmentError(f"unit {unit}: the unit table offers {len(ref_map)} pins, the tape says {n_given}")
        reading = parse_reading(repair.text, ref_map, cfg)
        if not reading.ok:
            continue
        columns = reading_columns(reading, cfg)
        for name, value in columns.items():
            if name in (PARSE_OK, REASON):
                continue  # the run's own verdict stays on record
            if name not in out.columns:
                out[name] = None
            out.at[index, name] = value
        scheduled_missing, certainty_missing = KEY_SCHEDULED in repair.missing, KEY_CERTAINTY in repair.missing
        out.at[index, CERTAINTY] = np.nan
        if scheduled_missing:
            out.at[index, SCHEDULED] = None
            out.at[index, FLAG_SCHEDULED_UNKNOWN] = False  # the placeholder is not an answer, so it cannot be an unknown one
        out.at[index, FLAG_CERTAINTY_CLAMPED] = False
        pins = reading.pins if nodes_known else [{**pin, PIN_NODE: None} for pin in reading.pins]
        out.at[index, READING_PINS] = json.dumps(pins, sort_keys=True)
        out.at[index, SALVAGE_STATUS] = SALVAGED
        out.at[index, SALVAGE_MISSING] = MISSING_SEPARATOR.join(repair.missing)
        out.at[index, RAW_REPAIRED] = repair.text
        out.at[index, TAIL_WS_CHARS] = repair.whitespace_chars
        out.at[index, FLAG_TAIL_COMPLETED] = True
        out.at[index, FLAG_PINS_TRUNCATED] = repair.pins_truncated
        out.at[index, SALVAGE_TIER] = TIER_PINS_TRUNCATED if repair.pins_truncated else TIER_TAIL
        out.at[index, PIN_OBJECTS_KEPT] = np.nan if repair.pin_objects_kept is None else repair.pin_objects_kept
        out.at[index, FLAG_SCHEDULED_MISSING] = scheduled_missing
        out.at[index, FLAG_CERTAINTY_MISSING] = certainty_missing
        out.at[index, FLAG_NODE_PENDING] = not nodes_known
    return out


def reparse_with_salvage(tape: pd.DataFrame, pins_json: Mapping[str, str], cfg: Config, truncate_pins: bool | None = None) -> pd.DataFrame:
    """The pipeline entry point: the current parser on every stored answer (``reparse_tape``), then tail completion of what is still
    unparsed, with node ids from the unit table. Fails loudly when the table does not match the tape."""
    report = verify_alignment(tape, pins_json, cfg)
    if report["pin_count_mismatches"] or report["node_id_mismatches"]:
        raise AlignmentError(f"unit table does not match the tape: {report}")
    return salvage_tape(reparse_tape(tape, dict(pins_json), cfg), cfg, pins_json, truncate_pins)


def resolve_certainty_fill(tape: pd.DataFrame, cfg: Config) -> float:
    """The value ``apply_fill`` uses for a missing certainty: the median of the naturally parsed rows, or the configured number."""
    setting = cfg.get("cpu.reader_salvage.certainty_fill")
    if setting != MEDIAN_PARSED:
        value = float(setting)
    else:
        natural = tape[tape[SALVAGE_STATUS] == NATURAL][CERTAINTY].astype(float).dropna()
        if natural.empty:
            raise ValueError("certainty_fill is median_parsed but the tape has no naturally parsed row")
        value = float(natural.median())
    low, high = float(cfg.get("reader.certainty_min")), float(cfg.get("reader.certainty_max"))
    if not low <= value <= high:
        raise ValueError(f"certainty fill {value} is outside [{low}, {high}]")
    return value


def apply_fill(tape: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Labelled opt-in for SENSITIVITY tables only: salvaged rows become usable (``parse_ok`` True) with every missing cell filled.

    ``certainty_source`` / ``scheduled_source`` say ``model`` or ``filled`` on every row. The final pipeline does not call this.
    """
    out = tape.copy()
    fill_certainty = resolve_certainty_fill(out, cfg)
    fill_scheduled = str(cfg.get("cpu.reader_salvage.scheduled_fill"))
    salvaged = out[SALVAGE_STATUS] == SALVAGED
    out[CERTAINTY_SOURCE] = np.where(out[STATUS] == Status.OK.value, SOURCE_MODEL, None)
    out[SCHEDULED_SOURCE] = np.where(out[STATUS] == Status.OK.value, SOURCE_MODEL, None)
    need_certainty = salvaged & out[FLAG_CERTAINTY_MISSING].astype(bool)
    need_scheduled = salvaged & out[FLAG_SCHEDULED_MISSING].astype(bool)
    out.loc[need_certainty, CERTAINTY] = fill_certainty
    out.loc[need_certainty, CERTAINTY_SOURCE] = SOURCE_FILLED
    out.loc[need_scheduled, SCHEDULED] = fill_scheduled
    out.loc[need_scheduled, SCHEDULED_SOURCE] = SOURCE_FILLED
    out.loc[salvaged, PARSE_OK] = True
    out[PARSE_OK] = out[PARSE_OK].map(_is_true)
    info = {"rows_enabled": int(salvaged.sum()), "certainty_filled": int(need_certainty.sum()), "certainty_fill_value": fill_certainty,
            "scheduled_filled": int(need_scheduled.sum()), "scheduled_fill_value": fill_scheduled}
    return out, info


# --- pipeline modes (named, pre-specified in config/cpu.yaml reader_salvage.modes) -------------------------------------------------------


def mode_settings(cfg: Config, mode: str) -> dict[str, Any]:
    modes = dict(cfg.get("cpu.reader_salvage.modes"))
    if mode not in modes:
        raise KeyError(f"unknown reader mode {mode!r}; choose one of {sorted(modes)}")
    return dict(modes[mode])


def config_for_mode(cfg: Config, mode: str) -> tuple[Config, dict[str, Any]]:
    """The configuration a CPU run uses under ``mode`` and a record of what differs from the hashed base configuration.

    Only ``reader.dials.features`` can change (the listed blocks are removed); the base config hash is untouched on disk and stays in the
    lineage next to this record.
    """
    settings = mode_settings(cfg, mode)
    drop = list(settings["drop_dial_features"])
    base = list(cfg.get("reader.dials.features"))
    kept = [f for f in base if f not in drop]
    record = {"mode": mode, "include_salvaged": bool(settings["include_salvaged"]), "fill": bool(settings["fill"]),
              "framework_confidence": str(settings["framework_confidence"]), "exclude_defaulted_cause": bool(settings.get("exclude_defaulted_cause", False)),
              "dial_features_base": base, "dial_features_used": kept, "dial_features_dropped": [f for f in base if f in drop]}
    return (cfg.with_overrides({"reader.dials.features": kept}) if drop else cfg), record


def exclude_defaulted_cause(tape: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    """Rows whose all-zero cause vector was read as ``other`` (``flag_cause_defaulted``) stop counting as readings: ``parse_ok`` False and, if the
    tape carries salvage columns, ``salvage_status = excluded_defaulted_cause``. The model's text and every other column stay as they were."""
    out = tape.copy()
    hit = out["flag_cause_defaulted"].map(_is_true) & (out[STATUS] == Status.OK.value)
    out.loc[hit, PARSE_OK] = False
    if SALVAGE_STATUS in out.columns:
        out.loc[hit, SALVAGE_STATUS] = EXCLUDED_DEFAULTED
    return out, int(hit.sum())


READER_V2_MARK = "event_type_final"  # a column only the final reader (reader2) writes


def prepare_reader_tape(tape: pd.DataFrame, pins_json: Mapping[str, str], run_cfg: Config, mode: Mapping[str, Any]) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any] | None]:
    """The reader tape as a CPU run under ``mode`` reads it: the current parser on every stored answer, plus tail completion (and the labelled
    fill, sensitivity mode only) when the mode says so. Returns the tape, the unit-table alignment report and the fill record.

    A mode that reads salvaged rows stops (``AlignmentError``) unless the unit table matches the tape; the original mode only warns, as before.
    A mode with ``exclude_defaulted_cause`` then drops every reading whose cause the parser had to default (count in ``report``).
    """
    report = verify_alignment(tape, pins_json, run_cfg)
    fill_info = None
    if READER_V2_MARK in tape.columns:  # the final reader (reader2) already parsed the model's answer: cause_*, reading_pins_json and parse_ok are final
        if not report["ok"]:
            raise AlignmentError(f"the unit table does not match the reader tape: {report}")
        out = tape.copy()
        out[PARSE_OK] = out[PARSE_OK].fillna(False).astype(bool)
        report["reader"] = "reader2: stored columns used as they are, no reparse"
        if mode.get("exclude_defaulted_cause"):
            report["excluded_defaulted_cause"] = 0  # reader2 never defaults a cause, so the mode has nothing to drop
        return out, report, None
    if mode["include_salvaged"]:
        if not report["ok"]:
            raise AlignmentError(f"the unit table does not match the reader tape: {report}")
        out = reparse_with_salvage(tape, pins_json, run_cfg)
        if mode["fill"]:
            out, fill_info = apply_fill(out, run_cfg)
    else:
        out = reparse_tape(tape, dict(pins_json), run_cfg)
    if mode.get("exclude_defaulted_cause"):
        out, report["excluded_defaulted_cause"] = exclude_defaulted_cause(out)
    return out, report, fill_info
