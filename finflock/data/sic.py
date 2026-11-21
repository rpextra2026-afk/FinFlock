"""SIC codes -> sector labels (longest matching prefix from ``config/sic_sector.csv``) and the security info table.

The SEC submissions API gives each company's CURRENT SIC code, so sectors and industry peers are a point-in-time
approximation (a company that changed industry keeps its current code); this is stated in the data audit.
"""

from __future__ import annotations

import pandas as pd

SIC_WIDTH = 4  # SIC codes have four digits


def sector_of_sic(sic: str | float, table: pd.DataFrame) -> str | None:
    """Sector of a SIC code by the longest prefix in ``table`` (columns ``sic_prefix``, ``sector``); None if unknown."""
    if sic is None or (isinstance(sic, float) and pd.isna(sic)):
        return None
    code = str(int(float(sic))).zfill(SIC_WIDTH)
    prefixes = table.assign(sic_prefix=table["sic_prefix"].astype(str)).sort_values("sic_prefix", key=lambda s: s.str.len(), ascending=False)
    for prefix, sector in zip(prefixes["sic_prefix"], prefixes["sector"], strict=True):
        if code.startswith(prefix):
            return str(sector)
    return None
