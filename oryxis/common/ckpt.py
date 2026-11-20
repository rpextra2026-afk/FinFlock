"""A checkpoint store for long CPU jobs: finished pieces of work are written to a folder and found again by a key.

Switched on only when the environment variable ``ORYXIS_CKPT_DIR`` names a folder (scripts/cpu_session.py sets it); without it every call is a no-op
and nothing changes. A key is a hash of everything the piece of work depends on, so a stored piece is never reused for different inputs. Files are
written atomically (temporary name, then rename), so a process killed at the cutoff never leaves a half-written piece behind.
"""

from __future__ import annotations

import hashlib
import os
import pickle
from pathlib import Path
from typing import Any

ENV = "ORYXIS_CKPT_DIR"
KEY_CHARS = 32
ERROR_CHARS = 120


def folder() -> Path | None:
    value = os.environ.get(ENV)
    return Path(value) if value else None


def key_of(*parts: Any) -> str:
    return hashlib.sha256(repr(parts).encode("utf-8")).hexdigest()[:KEY_CHARS]


def get(kind: str, *parts: Any) -> Any | None:
    """The stored piece of work, or None (also when the file is unreadable: it is then simply computed again)."""
    root = folder()
    if root is None:
        return None
    path = root / f"{kind}-{key_of(*parts)}.pkl"
    if not path.exists():
        return None
    try:
        with path.open("rb") as handle:
            return pickle.load(handle)  # files written by this project only
    except Exception:  # noqa: BLE001 - a damaged piece is recomputed, never trusted
        return None


def put(kind: str, value: Any, *parts: Any) -> None:
    root = folder()
    if root is None:
        return
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{kind}-{key_of(*parts)}.pkl"
    temporary = path.with_suffix(".tmp")
    try:
        with temporary.open("wb") as handle:
            pickle.dump(value, handle, protocol=pickle.HIGHEST_PROTOCOL)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    except Exception as exc:  # noqa: BLE001 - a piece that cannot be stored costs only its re-computation after a restart; the run goes on
        temporary.unlink(missing_ok=True)
        print(f"[ckpt] WARNING: could not store a {kind} piece ({type(exc).__name__}: {str(exc)[:ERROR_CHARS]}); the run continues without it", flush=True)


def counts() -> dict[str, int]:
    """Stored pieces by kind, for the logs."""
    root = folder()
    out: dict[str, int] = {}
    if root is not None and root.exists():
        for path in root.glob("*.pkl"):
            kind = path.name.split("-", 1)[0]
            out[kind] = out.get(kind, 0) + 1
    return out
