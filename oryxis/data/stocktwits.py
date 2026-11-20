"""StockTwits posts for db3, read from the public research bucket by streaming (spec set 3, social).

Two passes over files that are never stored whole: the ``symbols`` files say which posts name an S&P 500 ticker on which date, the ``messages``
files hold the text. Posts are thinned to a fixed number per year by a hash of the message id, so the choice is the same on every run. The
``created_at`` of a post is a date without a time of day, so every post is a date-only item and is handled that way downstream.
"""

from __future__ import annotations

import csv
import io
import re
from collections.abc import Callable, Collection, Iterable

import numpy as np
import pandas as pd

ID = "message_id"
DATE = "created_at"
SYMBOL = "symbol"
BODY = "message_body"
KEY = "key"
YEAR = "year"
SYMBOLS = "symbols"
DATE_IN_LINE = re.compile(r",(\d{4}-\d{2}-\d{2}),")
ID_AT_LINE_START = re.compile(r"^(\d+),", re.MULTILINE)
UINT64 = np.uint64
ROW_START = re.compile(rb"(?m)^(\d+),")
Fetch = Callable[[int, int], bytes]  # (start, end exclusive) -> the bytes of that range of the file
YEAR_CHARS = 4  # a date is YYYY-MM-DD


def thin_key(ids: np.ndarray, multiplier: int) -> np.ndarray:
    """Pseudo-random 64-bit key of each message id (multiplicative hash, wraps modulo 2^64). Smaller key = kept first."""
    return ids.astype(UINT64) * UINT64(multiplier)


def ticker_universe(security_ids: Iterable[str], replacements: dict[str, str]) -> set[str]:
    """Tickers as StockTwits writes them: ours plus the replaced spelling (``BRK-B`` -> ``BRK.B``)."""
    out: set[str] = set()
    for ticker in security_ids:
        out.add(ticker)
        spelled = ticker
        for old, new in replacements.items():
            spelled = spelled.replace(old, new)
        out.add(spelled)
    return out


def filter_symbol_chunk(chunk: pd.DataFrame, start: str, end: str, universe: Collection[str], multiplier: int) -> pd.DataFrame:
    """Posts of one chunk dated inside ``[start, end]`` that name a ticker of the universe: one row per post with its tickers."""
    dated = chunk[(chunk[DATE] >= start) & (chunk[DATE] <= end) & chunk[SYMBOL].isin(universe)]
    if dated.empty:
        return pd.DataFrame({ID: pd.Series(dtype="int64"), DATE: pd.Series(dtype=object), SYMBOLS: pd.Series(dtype=object), KEY: pd.Series(dtype="uint64")})
    grouped = dated.groupby(ID, sort=False).agg({DATE: "first", SYMBOL: lambda s: sorted(set(s))}).reset_index().rename(columns={SYMBOL: SYMBOLS})
    grouped[KEY] = thin_key(grouped[ID].to_numpy(), multiplier)
    return grouped


def keep_smallest_per_year(frames: Iterable[pd.DataFrame], per_year: int) -> pd.DataFrame:
    """Merge partial results and keep, for each calendar year, the ``per_year`` posts with the smallest key (a post seen twice merges its tickers)."""
    pieces = [f for f in frames if len(f)]
    if not pieces:
        return pd.DataFrame({ID: [], DATE: [], SYMBOLS: [], KEY: []})
    merged = pd.concat(pieces, ignore_index=True)
    merged = merged.groupby(ID, sort=False).agg({DATE: "first", SYMBOLS: lambda lists: sorted({t for sub in lists for t in sub}), KEY: "first"}).reset_index()
    merged[YEAR] = merged[DATE].str.slice(0, YEAR_CHARS)
    kept = merged.sort_values([YEAR, KEY]).groupby(YEAR, sort=False).head(per_year)
    return kept.drop(columns=YEAR).sort_values(ID).reset_index(drop=True)


def edge_dates(head: str, tail: str) -> tuple[str | None, str | None]:
    """Earliest date among the first rows and latest date among the last rows of a symbols file (``None`` when none is readable)."""
    first, last = DATE_IN_LINE.findall(head), DATE_IN_LINE.findall(tail)
    return (min(first) if first else None, max(last) if last else None)


def edge_ids(head: str, tail: str) -> tuple[int | None, int | None]:
    """Smallest id among the first rows and largest among the last rows of a file whose lines start with the message id."""
    first, last = ID_AT_LINE_START.findall(head), ID_AT_LINE_START.findall(tail)
    return (min(int(v) for v in first) if first else None, max(int(v) for v in last) if last else None)


def symbols_file_outside(first_date: str | None, last_date: str | None, start: str, end: str) -> bool:
    """True only when both edges are known and the whole file lies on one side of the window (an unreadable edge never skips a file)."""
    if first_date is None or last_date is None:
        return False
    return last_date < start or first_date > end


def messages_file_needed(first_id: int | None, last_id: int | None, wanted: np.ndarray) -> bool:
    """True when a message file may hold a wanted id (an unreadable edge never skips a file)."""
    if first_id is None or last_id is None or len(wanted) == 0:
        return True
    low = np.searchsorted(wanted, first_id, side="left")
    return bool(low < len(wanted) and wanted[low] <= last_id)


# ---- message files: find and read only the byte ranges that can hold a wanted id -----------------------------------------------------------------
# A message file is sorted by id inside each run, but a file can hold a second, lower-id run at its end, so its first and last ids say nothing
# about what lies between. The ids are sampled at evenly spaced offsets: a stretch between two samples whose ids rise can only hold ids between
# them; a stretch where the ids fall (or a sample that cannot be read) is read in full, as is the last stretch.


def probe_id(chunk: bytes) -> int | None:
    """Id of the first whole row start in a probe, skipping the partial line the probe begins in."""
    newline = chunk.find(b"\n")
    if newline < 0:
        return None
    match = ROW_START.search(chunk, newline + 1)
    return int(match.group(1)) if match else None


def sample_offsets(size: int, points: int) -> list[int]:
    """Evenly spaced offsets from 0 to ``size`` inclusive (``points`` stretches)."""
    return [size * k // points for k in range(points + 1)]


def plan_runs(offsets: list[int], ids: list[int | None], wanted: np.ndarray) -> list[tuple[int, int, tuple[int, int] | None]]:
    """Byte runs ``(start, end, id bounds or None)`` that may hold a wanted id. ``ids[k]`` is the sampled id at ``offsets[k]`` (the last offset is the file end)."""
    runs: list[tuple[int, int, tuple[int, int] | None]] = []
    for k in range(len(offsets) - 1):
        low, high = ids[k], ids[k + 1] if k + 2 < len(offsets) else None
        known = low is not None and high is not None and high >= low
        if known:
            left = np.searchsorted(wanted, low, side="left")
            if not (left < len(wanted) and wanted[left] <= high):
                continue
        bounds = (low, high) if known else None
        if runs and runs[-1][1] == offsets[k]:
            start, _, previous = runs.pop()
            merged = None if previous is None or bounds is None else (min(previous[0], bounds[0]), max(previous[1], bounds[1]))
            runs.append((start, offsets[k + 1], merged))
        else:
            runs.append((offsets[k], offsets[k + 1], bounds))
    return runs


def parse_run(data: bytes, run_length: int, bounds: tuple[int, int] | None, at_file_start: bool) -> pd.DataFrame:
    """Rows of a downloaded run: those whose row starts inside the run itself (the bytes after it only complete the last row)."""
    empty = pd.DataFrame({ID: pd.Series(dtype="int64"), BODY: pd.Series(dtype=object)})
    skip = 0 if at_file_start else max(data.find(b"\n") + 1, 0)  # the run begins in the middle of a row
    starts = [(m.start(), int(m.group(1))) for m in ROW_START.finditer(data, skip)]
    if bounds is not None:
        starts = [(pos, i) for pos, i in starts if bounds[0] <= i <= bounds[1]]
    inside = [pos for pos, _ in starts if pos < run_length]
    if not inside:
        return empty
    after = [pos for pos, _ in starts if pos >= run_length]
    region = data[inside[0] : after[0] if after else len(data)]
    rows = []
    for record in csv.reader(io.StringIO(region.decode(errors="replace"), newline="")):  # tolerant of a quoted field cut by the end of the region
        if len(record) >= 2 and record[0].isdigit():
            rows.append((int(record[0]), ",".join(record[1:])))
    return pd.DataFrame(rows, columns=[ID, BODY]) if rows else empty


def collect_wanted(fetch: Fetch, size: int, wanted: np.ndarray, points: int, probe_bytes: int, overlap: int, max_run_bytes: int,
                   exhaustive: bool = False) -> tuple[pd.DataFrame, dict]:
    """Messages with a wanted id from one file. Returns the rows and a small report.

    ``exhaustive`` reads the whole file. The planned mode (sampled ids) misses short blocks of low ids that sit between two samples, which the real
    files do contain (two sorted streams are interleaved in small blocks), so it is kept only for tests and for files known to be a single run.
    """
    if exhaustive:
        runs = [(0, size, None)]
    else:
        offsets = sample_offsets(size, points)
        ids = [probe_id(fetch(off, min(off + probe_bytes, size))) if off < size else None for off in offsets]
        runs = plan_runs(offsets, ids, wanted)
    pieces, streamed = [], 0
    for run_start, run_end, bounds in runs:
        for start in range(run_start, run_end, max_run_bytes):  # long runs are read in pieces so memory stays small
            end = min(start + max_run_bytes, run_end)
            data = fetch(start, min(end + overlap, size))
            streamed += len(data)
            pieces.append(parse_run(data, end - start, bounds, at_file_start=start == 0))
    found = pd.concat(pieces, ignore_index=True) if pieces else pd.DataFrame({ID: pd.Series(dtype="int64"), BODY: pd.Series(dtype=object)})
    found = found[found[ID].isin(wanted)].drop_duplicates(ID)
    return found, {"runs": len(runs), "bytes": streamed, "size": size, "found": len(found)}
