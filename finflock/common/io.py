"""Atomic file writes and parquet helpers.

Every output is written to a temporary file in the target directory, flushed to disk and renamed,
so a crash can never leave a half-written file under the final name.
"""

from __future__ import annotations

import contextlib
import io
import os
import tempfile
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from finflock.common.hashing import canonical_json

TMP_SUFFIX = ".tmp"


def _fsync_dir(directory: Path) -> None:
    fd = os.open(directory, os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


@contextlib.contextmanager
def atomic_path(path: str | Path) -> Iterator[Path]:
    """Yield a temporary path; on clean exit fsync it and rename it onto ``path``."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=target.name + ".", suffix=TMP_SUFFIX)
    os.close(fd)
    tmp = Path(tmp_name)
    try:
        yield tmp
        with open(tmp, "rb") as handle:
            os.fsync(handle.fileno())
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()
        raise
    _fsync_dir(target.parent)


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    """Write bytes atomically."""
    with atomic_path(path) as tmp:
        tmp.write_bytes(data)


def atomic_write_text(path: str | Path, text: str) -> None:
    """Write UTF-8 text atomically."""
    atomic_write_bytes(path, text.encode("utf-8"))


def atomic_write_json(path: str | Path, obj: Any) -> None:
    """Write canonical JSON atomically."""
    atomic_write_text(path, canonical_json(obj))


def frame_to_parquet_bytes(frame: pd.DataFrame, compression: str) -> bytes:
    """Parquet bytes of a frame without pandas metadata, for byte-stable output."""
    table = pa.Table.from_pandas(frame, preserve_index=False).replace_schema_metadata(None)
    buffer = io.BytesIO()
    pq.write_table(table, buffer, compression=compression)
    return buffer.getvalue()


def parquet_bytes_to_frame(data: bytes, columns: list[str] | None = None) -> pd.DataFrame:
    """Read parquet bytes into a frame (optionally only some columns)."""
    return pq.read_table(io.BytesIO(data), columns=columns).to_pandas()


def atomic_write_parquet(path: str | Path, frame: pd.DataFrame, compression: str) -> None:
    """Write a frame as parquet atomically."""
    atomic_write_bytes(path, frame_to_parquet_bytes(frame, compression))
