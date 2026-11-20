"""The parquet store: one directory per table, written atomically, validated against its schema."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from finfm.common.io import atomic_write_parquet
from finfm.data.schema import SPECS, validate

SUFFIX = ".parquet"


class Store:
    """Reads and writes named tables below ``root`` (``store/`` in the repository layout)."""

    def __init__(self, root: str | Path, compression: str):
        self.root = Path(root)
        self.compression = compression

    def path(self, table: str, part: str | None = None) -> Path:
        name = table if part is None else f"{table}/{part}"
        return self.root / f"{name}{SUFFIX}"

    def write(self, table: str, frame: pd.DataFrame, part: str | None = None, *, check: bool = True) -> Path:
        if check and table in SPECS:
            validate(frame, SPECS[table])
        target = self.path(table, part)
        atomic_write_parquet(target, frame.reset_index(drop=True), self.compression)
        return target

    def read(self, table: str, part: str | None = None, columns: list[str] | None = None) -> pd.DataFrame:
        return pd.read_parquet(self.path(table, part), columns=columns)

    def exists(self, table: str, part: str | None = None) -> bool:
        return self.path(table, part).exists()
