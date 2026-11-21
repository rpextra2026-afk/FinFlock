"""Single source of column names, status codes and file-format identifiers.

Dataset, window and model names are NOT here: they live in ``config/``.
"""

from __future__ import annotations

from enum import Enum

# --- tape / unit columns -------------------------------------------------------------------
UNIT_ID = "unit_id"
STATUS = "status"
REASON = "reason"
DETAIL = "detail"
ATTEMPTS = "attempts"
PARSE_OK = "parse_ok"
NAN_DETECTED = "nan_detected"


# --- event / reader columns ------------------------------------------------------------------
KIND = "kind"
FAMILY = "family"
AVAILABLE_DATE = "available_date"
TITLE = "title"
TEXT = "text"
PINS_JSON = "pins_json"
RAW_OUTPUT = "raw_output"
TOKENS_IN = "tokens_in"
TOKENS_OUT = "tokens_out"


class Status(str, Enum):
    """Outcome of one unit of work."""

    OK = "ok"
    SKIPPED = "skipped"  # deterministic skip with a reason, never retried
    ERROR = "error"  # transient failure, retried until ``max_attempts``
    FAILED = "failed"  # error that exhausted its attempts, terminal


TERMINAL_STATUSES = frozenset({Status.OK.value, Status.SKIPPED.value, Status.FAILED.value})
# Lower rank wins when the same unit appears in several parts or bundles.
STATUS_RANK = {
    Status.OK.value: 0,
    Status.SKIPPED.value: 1,
    Status.FAILED.value: 2,
    Status.ERROR.value: 3,
}


class Reason(str, Enum):
    """Reason codes recorded for skipped, errored and failed units."""

    EXCEPTION = "exception"
    NO_OUTPUT = "no_output"
    NO_USABLE_PIN = "no_usable_pin"
    NO_FOCAL_ACCESS = "no_focal_access"
    MISSING_PRICES = "missing_prices"
    MISSING_SHARES = "missing_shares"
    DELISTED = "delisted"
    PARSE_FAILED = "parse_failed"
    ATTEMPTS_EXHAUSTED = "attempts_exhausted"
    INVALID_JSON = "invalid_json"
    MISSING_KEY = "missing_key"
    EMPTY_CAUSE = "empty_cause"


class ShardOutcome(str, Enum):
    """How a worker run ended."""

    COMPLETE = "complete"
    SESSION_LIMIT = "session_limit"
    TERMINATED = "terminated"
    ABORTED = "aborted"


class AbortReason(str, Enum):
    """Why a worker stopped early to protect GPU time."""

    LOW_PARSE_RATE = "low_parse_rate"
    NAN_DETECTED = "nan_detected"
    ERROR_RATE = "error_rate"


# --- bundle format -------------------------------------------------------------------------
BUNDLE_SUFFIX = ".bundle"
BUNDLE_PREFIX = "finflock"
BUNDLE_FORMAT_VERSION = 1
MANIFEST_NAME = "MANIFEST.json"
PARTS_DIR = "parts/"
STATE_DIR = "state/"
DIAGNOSTICS_DIR = "diagnostics/"
LOGS_DIR = "logs/"
STATE_NAME = "state/state.json"
PART_SUFFIX = ".parquet"
PART_PREFIX = "part-"
PART_NAME_HASH_CHARS = 16
JOB_ID_SEPARATOR = "__"
SHARD_ID_ALL = "all"
