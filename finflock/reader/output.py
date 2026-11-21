"""Validation of the reader's JSON answer (spec 4.3, 4.5, 12.2 #6).

Nothing is silently repaired: every repair (clamping, renormalising, dropping a pin) is counted in
``flags`` so the hallucination and parse-failure rates can be reported.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from finflock.common.config import Config
from finflock.common.schema import Reason

KEY_CAUSE = "cause"
KEY_PINS = "pins"
KEY_SCHEDULED = "scheduled"
KEY_CERTAINTY = "certainty"
REQUIRED_KEYS = (KEY_CAUSE, KEY_PINS, KEY_SCHEDULED, KEY_CERTAINTY)
PIN_REF = "ref"
PIN_DIRECTION = "direction"
PIN_VALUE = "value_effect_pct"
PIN_MAGNITUDE = "magnitude"
PIN_ROLE = "role"
PIN_NODE = "node_id"
PIN_KEYS = (PIN_REF, PIN_DIRECTION, PIN_VALUE, PIN_MAGNITUDE, PIN_ROLE)
REF_PATTERN_TEMPLATE = r"^{prefix}[0-9]+$"


@dataclass
class Flags:
    cause_sum_raw: float = 0.0
    cause_renormalised: bool = False
    unknown_causes: int = 0
    value_clamped: int = 0
    magnitude_clamped: int = 0
    certainty_clamped: bool = False
    scheduled_unknown: bool = False
    hallucinated_refs: int = 0
    invalid_pins: int = 0
    duplicate_refs: int = 0
    cause_defaulted: bool = False

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


@dataclass
class Reading:
    ok: bool
    fail_reason: Reason | None = None
    cause: dict[str, float] = field(default_factory=dict)
    pins: list[dict[str, Any]] = field(default_factory=list)
    scheduled: str | None = None
    certainty: float | None = None
    flags: Flags = field(default_factory=Flags)


def _number(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if math.isfinite(value) else None


def reader_json_schema(cfg: Config) -> dict[str, Any]:
    """JSON schema for constrained decoding. Refs are a pattern, not an enum, so that a made-up
    ref is still possible and gets counted as a hallucination."""
    causes = list(cfg.get("causes.causes"))
    bound = float(cfg.get("reader.value_effect_pct_bound"))
    ref_pattern = REF_PATTERN_TEMPLATE.format(prefix=re.escape(str(cfg.get("reader.ref_prefix"))))
    none_ref = str(cfg.get("reader.none_ref"))
    return {
        "type": "object",
        "properties": {
            KEY_CAUSE: {
                "type": "object",
                "properties": {c: {"type": "number", "minimum": 0, "maximum": 1} for c in causes},
                "required": causes,
                "additionalProperties": False,
            },
            KEY_PINS: {
                "type": "array",
                "maxItems": int(cfg.get("reader.max_pins")),  # an answer cannot list more assets than were offered: stops repetition loops
                "items": {
                    "type": "object",
                    "properties": {
                        PIN_REF: {"type": "string", "pattern": f"{ref_pattern}|^{none_ref}$"},
                        PIN_DIRECTION: {"enum": list(cfg.get("reader.directions"))},
                        PIN_VALUE: {"type": "number", "minimum": -bound, "maximum": bound},
                        PIN_MAGNITUDE: {
                            "type": "integer",
                            "minimum": 0,
                            "maximum": int(cfg.get("reader.magnitude_max")),
                        },
                        PIN_ROLE: {"enum": list(cfg.get("reader.roles"))},
                    },
                    "required": list(PIN_KEYS),
                },
            },
            KEY_SCHEDULED: {"enum": list(cfg.get("reader.scheduled_values"))},
            KEY_CERTAINTY: {
                "type": "number",
                "minimum": float(cfg.get("reader.certainty_min")),
                "maximum": float(cfg.get("reader.certainty_max")),
            },
        },
        "required": list(REQUIRED_KEYS),
    }


def parse_reading(text: str, ref_to_node: Mapping[str, str], cfg: Config) -> Reading:
    """Parse and validate one answer against the allowed refs (``A1`` -> node id)."""
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return Reading(False, Reason.INVALID_JSON)
    if not isinstance(data, dict):
        return Reading(False, Reason.INVALID_JSON)
    for key in REQUIRED_KEYS:
        if key not in data:
            return Reading(False, Reason.MISSING_KEY)

    flags = Flags()
    causes = list(cfg.get("causes.causes"))
    raw_cause = data[KEY_CAUSE]
    if not isinstance(raw_cause, dict):
        return Reading(False, Reason.INVALID_JSON)
    cause: dict[str, float] = {}
    for name, value in raw_cause.items():
        number = _number(value)
        if name not in causes:
            flags.unknown_causes += 1
        elif number is not None:
            cause[name] = max(number, 0.0)  # negatives are impossible; large values keep their ratio
    total = sum(cause.values())
    flags.cause_sum_raw = total
    if total <= 0:  # the model gave no cause weight at all: the news has no listed cause, counted and flagged
        fallback = str(cfg.get("reader.empty_cause_fallback"))
        cause, total, flags.cause_defaulted = {fallback: 1.0}, 1.0, True
    tolerance = float(cfg.get("reader.cause_prob_tolerance"))
    flags.cause_renormalised = abs(total - 1.0) > tolerance
    cause = {c: cause.get(c, 0.0) / total for c in causes}

    bound = float(cfg.get("reader.value_effect_pct_bound"))
    mag_max = int(cfg.get("reader.magnitude_max"))
    directions = set(cfg.get("reader.directions"))
    roles = set(cfg.get("reader.roles"))
    none_ref = str(cfg.get("reader.none_ref"))
    pins: list[dict[str, Any]] = []
    seen: set[str] = set()
    raw_pins = data[KEY_PINS]
    for item in raw_pins if isinstance(raw_pins, Sequence) and not isinstance(raw_pins, str) else []:
        if not isinstance(item, dict) or any(k not in item for k in PIN_KEYS):
            flags.invalid_pins += 1
            continue
        ref = item[PIN_REF]
        if ref == none_ref:
            continue
        if ref not in ref_to_node:
            flags.hallucinated_refs += 1
            continue
        if ref in seen:
            flags.duplicate_refs += 1
            continue
        value, magnitude = _number(item[PIN_VALUE]), _number(item[PIN_MAGNITUDE])
        if (
            item[PIN_DIRECTION] not in directions
            or item[PIN_ROLE] not in roles
            or value is None
            or magnitude is None
        ):
            flags.invalid_pins += 1
            continue
        clipped = min(max(value, -bound), bound)
        flags.value_clamped += int(clipped != value)
        mag = min(max(int(round(magnitude)), 0), mag_max)
        flags.magnitude_clamped += int(mag != magnitude)
        seen.add(ref)
        pins.append(
            {
                PIN_REF: ref,
                PIN_NODE: ref_to_node[ref],
                PIN_DIRECTION: item[PIN_DIRECTION],
                PIN_VALUE: clipped,
                PIN_MAGNITUDE: mag,
                PIN_ROLE: item[PIN_ROLE],
            }
        )

    scheduled = data[KEY_SCHEDULED]
    if scheduled not in set(cfg.get("reader.scheduled_values")):
        flags.scheduled_unknown = True
        scheduled = "unknown"
    certainty = _number(data[KEY_CERTAINTY])
    if certainty is None:
        return Reading(False, Reason.MISSING_KEY, flags=flags)
    low, high = float(cfg.get("reader.certainty_min")), float(cfg.get("reader.certainty_max"))
    clipped_certainty = min(max(certainty, low), high)
    flags.certainty_clamped = clipped_certainty != certainty
    return Reading(True, None, cause, pins, scheduled, clipped_certainty, flags)
