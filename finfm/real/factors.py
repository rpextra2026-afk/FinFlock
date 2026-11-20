"""Parse a Ken French daily csv (a text header, a column row, daily rows, then a footer) into a date-indexed frame of decimals."""

from __future__ import annotations

import io
import re

import pandas as pd

DATE_ROW = re.compile(r"^\s*(\d{8})\s*,")


def parse_french_csv(text: str, scale: float) -> pd.DataFrame:
    lines = text.splitlines()
    first = next(i for i, line in enumerate(lines) if DATE_ROW.match(line))
    header = [c.strip() for c in lines[first - 1].split(",")]
    rows = []
    for line in lines[first:]:
        if not DATE_ROW.match(line):
            break  # the footer (copyright, annual tables) starts where dated rows stop
        rows.append(line)
    frame = pd.read_csv(io.StringIO("\n".join(rows)), header=None, names=["date"] + header[1:])
    frame.index = pd.to_datetime(frame.pop("date").astype(str), format="%Y%m%d")
    frame.columns = [c.strip() for c in frame.columns]
    return frame.astype(float) * scale
