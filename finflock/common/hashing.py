"""Hashing helpers: canonical JSON, sha256 of bytes, files and objects."""

from __future__ import annotations

import datetime as dt
import hashlib
import json
from pathlib import Path
from typing import Any

HASH_CHUNK_BYTES = 1 << 20


def _json_default(obj: Any) -> Any:
    """Make dates, paths, sets and numpy values JSON serialisable and order independent."""
    if isinstance(obj, (dt.date, dt.datetime)):
        return obj.isoformat()
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, (set, frozenset)):
        return sorted(obj)
    tolist = getattr(obj, "tolist", None)
    if callable(tolist):
        return tolist()
    raise TypeError(f"object of type {type(obj).__name__} is not canonically serialisable")


def canonical_json(obj: Any) -> str:
    """Deterministic JSON text: sorted keys, no whitespace, non-ASCII kept."""
    return json.dumps(
        obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_json_default
    )


def sha256_hex(data: bytes | str) -> str:
    """Hex sha256 of bytes or of a UTF-8 string."""
    if isinstance(data, str):
        data = data.encode("utf-8")
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Hex sha256 of a file, read in chunks."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def stable_hash(obj: Any) -> str:
    """Hex sha256 of the canonical JSON of an object."""
    return sha256_hex(canonical_json(obj))
