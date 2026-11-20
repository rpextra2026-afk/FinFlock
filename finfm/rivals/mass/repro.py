"""MASS's own yardsticks (spec 10.4): IC, ICIR, rank IC, rank ICIR and the agent-count scaling check."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr

from finfm.rivals.common import ic_summary, rank_ic


def ic_table(signals: dict[pd.Timestamp, pd.Series], forward: dict[pd.Timestamp, pd.Series]) -> pd.DataFrame:
    """Per date IC (Pearson) and rank IC between the signal and the forward return."""
    rows = []
    for date, signal in signals.items():
        returns = forward.get(date)
        if returns is None:
            continue
        both = pd.concat([signal, returns], axis=1, keys=["s", "r"]).dropna()
        ic = float(pearsonr(both["s"], both["r"])[0]) if len(both) > 2 and both["s"].nunique() > 1 and both["r"].nunique() > 1 else np.nan
        rows.append({"date": date, "ic": ic, "rank_ic": rank_ic(signal, returns)})
    return pd.DataFrame(rows).set_index("date")


def summary(table: pd.DataFrame) -> dict[str, float]:
    """IC, ICIR, rank IC and rank ICIR as the paper reports them."""
    plain, ranked = ic_summary(table["ic"]), ic_summary(table["rank_ic"])
    return {"ic": plain["ic"], "icir": plain["icir"], "rank_ic": ranked["ic"], "rank_icir": ranked["icir"], "n_dates": plain["n_dates"]}


def scaling_check(summaries: dict[int, dict[str, float]]) -> pd.DataFrame:
    """Rank IC against the number of agents (the pre-registered 64 / 128 / 256 check); monotone is the pattern to look for."""
    frame = pd.DataFrame(summaries).T.sort_index()
    frame["rank_ic_rises"] = frame["rank_ic"].diff().fillna(0.0) >= 0.0
    return frame
