"""MASS aggregation: vote shares per investor type, then the market disagreement signal (repository logic).

For each stock the score of an investor type is the share of that type's agents that were shown the stock
and chose it (zero when none saw it). Across types, with weights ``w`` (the agent distribution):
``mean = sum w v / sum w``, ``std = sqrt(sum w (v - mean)^2 / sum w)`` and
``signal = alpha mean - (1 - alpha) std``. The returned triple is ``(signal, mean, -std)``.
"""

from __future__ import annotations

from collections import defaultdict

import numpy as np
import pandas as pd


class VoteBook:
    """Accumulates agent decisions: ``score`` (times chosen) and ``num_investors`` (times shown)."""

    def __init__(self) -> None:
        self.data: dict = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {"score": 0, "num_investors": 0})))

    def record(self, date, investor_type: str, shown: list[str], chosen: list[str]) -> None:
        picked = set(chosen)
        for stock in set(shown):
            cell = self.data[date][investor_type][stock]
            cell["score"] += int(stock in picked)
            cell["num_investors"] += 1

    def dates(self) -> list:
        return sorted(self.data)

    def types(self, date) -> list[str]:
        return sorted(self.data[date])

    def disagreement(self, date, pool: list[str], distribution: dict[str, float], alpha: float) -> pd.DataFrame:
        """``Signal, Signal_mean, Signal_std`` per stock of ``pool`` on ``date``."""
        types = self.types(date)
        weights = np.array([distribution[t] for t in types], dtype=float)
        rows = []
        for stock in pool:
            values = np.array([
                (cell["score"] / cell["num_investors"]) if (cell := self.data[date][t].get(stock)) and cell["num_investors"] else 0.0
                for t in types
            ])
            mean = float((weights * values).sum() / weights.sum())
            std = float(np.sqrt((weights * (values - mean) ** 2).sum() / weights.sum()))
            rows.append({"Stock": stock, "Signal": alpha * mean - (1.0 - alpha) * std, "Signal_mean": mean, "Signal_std": -std})
        return pd.DataFrame(rows)
