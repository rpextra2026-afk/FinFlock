"""Re-read the stored raw answers of a reader tape with the current parser.

The tape keeps every raw answer, so a parser fix (what counts as a valid answer, how an empty cause vector is treated) never needs a
GPU rerun: the parsed columns are recomputed from ``raw_output`` and the offered pins. Answers that were cut off or looped stay failed;
only a new model call can repair those.
"""

from __future__ import annotations

import pandas as pd

from oryxis.common.config import Config
from oryxis.common.schema import PARSE_OK, RAW_OUTPUT, STATUS, UNIT_ID, Status
from oryxis.reader.output import parse_reading
from oryxis.reader.prompt import ref_to_node_of
from oryxis.reader.run import reading_columns


def reparse_tape(tape: pd.DataFrame, pins_json: dict[str, str], cfg: Config) -> pd.DataFrame:
    """A copy of ``tape`` whose parsed columns come from the current parser. ``pins_json``: unit id -> the event's pins as prestaged."""
    out = tape.copy()
    rows = []
    for index, row in out.iterrows():
        if row[STATUS] != Status.OK.value or not isinstance(row[RAW_OUTPUT], str) or row[UNIT_ID] not in pins_json:
            continue
        reading = parse_reading(row[RAW_OUTPUT], ref_to_node_of(pins_json[row[UNIT_ID]], cfg), cfg)
        rows.append((index, reading_columns(reading, cfg)))
    for index, columns in rows:
        for name, value in columns.items():
            if name not in out.columns:
                out[name] = None
            out.at[index, name] = value
    out[PARSE_OK] = out[PARSE_OK].astype(bool)
    return out
