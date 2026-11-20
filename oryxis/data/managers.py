"""Manager types from 13F data (spec 3.4): an assumption, stated in the paper, overrides win.

Rules in order, case-insensitive on the filer name: ``bank_dealer`` keywords; ``mutual_pension``
keywords, or an asset-manager keyword with more than ``asset_manager_min_positions`` positions;
``hedge_fund`` for the remaining filers with few positions or in the top turnover tercile; anything
left is ``mutual_pension`` below the median turnover and ``hedge_fund`` otherwise.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from oryxis.common.config import Config


def contains_any(name: str, keywords: list[str]) -> bool:
    upper = str(name).upper()
    return any(k.upper() in upper for k in keywords)


def classify_managers(
    managers: pd.DataFrame, cfg: Config, overrides: dict[str, str] | None = None
) -> pd.Series:
    """Type per manager. ``managers`` has ``name, n_positions, turnover`` indexed by manager cik."""
    spec = cfg.get("managers")
    top = managers["turnover"].quantile(float(spec["top_turnover_quantile"]))
    median = managers["turnover"].median()
    types = {}
    for cik, row in managers.iterrows():
        if overrides and str(cik) in overrides:
            types[cik] = overrides[str(cik)]
        elif contains_any(row["name"], spec["bank_dealer_keywords"]):
            types[cik] = "bank_dealer"
        elif contains_any(row["name"], spec["mutual_pension_keywords"]) or (
            contains_any(row["name"], spec["asset_manager_keywords"])
            and row["n_positions"] > spec["asset_manager_min_positions"]
        ):
            types[cik] = "mutual_pension"
        elif row["n_positions"] <= spec["hedge_fund_max_positions"] or row["turnover"] >= top:
            types[cik] = "hedge_fund"
        else:
            types[cik] = "mutual_pension" if row["turnover"] < median else "hedge_fund"
    return pd.Series(types, name="type")


def passive_flags(names: pd.Series, families: list[str]) -> pd.Series:
    """True for filers whose name matches a passive family (they hold shares fixed in the game)."""
    return names.map(lambda n: contains_any(n, families))


def quarterly_turnover(
    previous: pd.Series, current: pd.Series, prices: pd.Series, split_ratio: pd.Series, book: float
) -> float:
    """``sum |delta shares x price| / (2 book)`` between consecutive quarters, shares split-adjusted.

    ``previous`` shares are multiplied by the split ratio between the two period ends so that a split
    is not counted as trading.
    """
    adjusted = previous.mul(split_ratio.reindex(previous.index).fillna(1.0), fill_value=0.0)
    names = adjusted.index.union(current.index)
    change = current.reindex(names, fill_value=0.0) - adjusted.reindex(names, fill_value=0.0)
    return float((change.abs() * prices.reindex(names).fillna(0.0)).sum() / (2.0 * book)) if book > 0 else np.nan


def exclude_low_cover(sp_cover: pd.Series, minimum: float) -> pd.Series:
    """Managers whose S&P 500 share of the book is below ``minimum`` are excluded (counted by the caller)."""
    return sp_cover >= minimum
