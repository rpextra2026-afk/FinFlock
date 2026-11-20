"""Pieces shared by the rival frameworks: headline dedupe, long-short rules, IC statistics.

Rivals receive only the dataset and the same LLM; nothing from our reader, personas or dials. Each
produces a signal table ``(decision_date, security_id, score, n_items)`` and target weights by its own
published rule. Weights go through the same executor, fills and costs as ours (spec 9.4, 10.1).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from oryxis.data.f13 import normalise_name
from oryxis.portfolio.market import MarketView

MIN_OBSERVATIONS = 3  # a correlation needs at least this many pairs
SIGNAL_COLUMNS = ["decision_date", "security_id", "score", "n_items"]


def osa_distance(a: str, b: str) -> int:
    """Optimal string alignment distance (restricted Damerau-Levenshtein)."""
    la, lb = len(a), len(b)
    table = np.zeros((la + 1, lb + 1), dtype=int)
    table[:, 0], table[0, :] = np.arange(la + 1), np.arange(lb + 1)
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = int(a[i - 1] != b[j - 1])
            table[i, j] = min(table[i - 1, j] + 1, table[i, j - 1] + 1, table[i - 1, j - 1] + cost)
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                table[i, j] = min(table[i, j], table[i - 2, j - 2] + 1)
    return int(table[la, lb])


def osa_similarity(a: str, b: str) -> float:
    """``1 - OSA distance / longer length`` (1.0 for identical strings)."""
    longest = max(len(a), len(b))
    return 1.0 if longest == 0 else 1.0 - osa_distance(a, b) / longest


def dedupe_headlines(frame: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """Drop later near-identical headlines of the same company and decision date (similarity above threshold)."""
    keep = []
    for _, part in frame.groupby(["security_id", "decision_date"], sort=False):
        kept: list[str] = []
        for index, headline in zip(part.index, part["headline"].str.lower(), strict=True):
            if all(osa_similarity(headline, other) <= threshold for other in kept):
                kept.append(headline)
                keep.append(index)
    return frame.loc[sorted(keep)]


def name_index(names: Mapping[str, str], suffixes: set[str]) -> dict[str, str]:
    """Normalised company name -> security id, for mapping names an LLM returns back to securities."""
    return {normalise_name(name, suffixes): security for security, name in names.items()}


def daily_scores(signals: pd.DataFrame, aggregation: str) -> pd.DataFrame:
    """One score per (decision date, security): mean, sum, or the value of largest magnitude."""
    grouped = signals.groupby(["decision_date", "security_id"])["score"]
    if aggregation == "mean":
        score = grouped.mean()
    elif aggregation == "sum":
        score = grouped.sum()
    else:
        score = grouped.agg(lambda s: s.loc[s.abs().idxmax()])
    return pd.DataFrame({"score": score, "n_items": grouped.size()}).reset_index()


def sign_legs(scores: pd.Series, leg_weight: float, min_firms: int) -> pd.Series:
    """Equal-weighted long-short on the sign of the score (LLT): each leg sums to ``leg_weight``.

    A leg with fewer than ``min_firms`` firms is not entered; the other leg is entered alone.
    """
    longs, shorts = scores[scores > 0].index, scores[scores < 0].index
    weights = pd.Series(dtype=float)
    if len(longs) >= min_firms and len(longs):
        weights = pd.concat([weights, pd.Series(leg_weight / len(longs), index=longs)])
    if len(shorts) >= min_firms and len(shorts):
        weights = pd.concat([weights, pd.Series(-leg_weight / len(shorts), index=shorts)])
    return weights


def top_bottom_legs(scores: pd.Series, fraction: float, leg_weight: float) -> pd.Series:
    """Top ``fraction`` long and bottom ``fraction`` short, equal weights, each leg summing to ``leg_weight``."""
    if len(scores) < 2:
        return pd.Series(dtype=float)
    k = max(int(np.ceil(fraction * len(scores))), 1)
    ordered = scores.sort_values(kind="mergesort")
    shorts, longs = ordered.index[:k], ordered.index[-k:]
    return pd.concat([pd.Series(leg_weight / k, index=longs), pd.Series(-leg_weight / k, index=shorts)])


def partial_step(held: pd.Series, target: pd.Series, fraction: float) -> pd.Series:
    """Move a fraction of the way from the held weights to the target (``1.0`` is full rebalancing)."""
    names = held.index.union(target.index)
    mixed = (1.0 - fraction) * held.reindex(names, fill_value=0.0) + fraction * target.reindex(names, fill_value=0.0)
    return mixed[mixed.abs() > 0]


def long_only(weights: pd.Series, total: float) -> pd.Series:
    """Keep the long leg only, rescaled to sum to ``total`` (the long-only robustness of 11.4 C)."""
    longs = weights[weights > 0]
    return longs / longs.sum() * total if len(longs) else longs


def liquid_only(weights: pd.Series, view: MarketView, min_price: float, min_dv: float, window: int) -> pd.Series:
    """Drop names with a last close below ``min_price`` or a median dollar volume below ``min_dv``."""
    close = view.close[weights.index].iloc[-1]
    median_dv = (view.close[weights.index] * view.volume[weights.index]).iloc[-window:].median()
    ok = (close >= min_price) & (median_dv >= min_dv)
    return weights[ok[weights.index].fillna(False)]


def make_strategy(
    rule: Callable[[pd.Series, pd.Timestamp, MarketView, pd.Series], pd.Series],
    scores_by_date: Mapping[pd.Timestamp, pd.Series],
) -> Callable[[pd.Timestamp, MarketView, pd.Series], pd.Series]:
    """Target function for the executor: today's scores through the rival's published rule."""

    def target(date: pd.Timestamp, view: MarketView, held: pd.Series) -> pd.Series:
        scores = scores_by_date.get(date)
        if scores is None or scores.empty:
            return held[held.abs() > 0] * 0.0 if len(held) else pd.Series(dtype=float)
        return rule(scores, date, view, held)

    return target


def rank_ic(scores: pd.Series, forward: pd.Series) -> float:
    """Spearman correlation between a cross-section of scores and forward returns."""
    both = pd.concat([scores, forward], axis=1, keys=["s", "r"]).dropna()
    if len(both) < MIN_OBSERVATIONS or both["s"].nunique() < 2 or both["r"].nunique() < 2:
        return float("nan")
    return float(spearmanr(both["s"], both["r"])[0])


def ic_summary(ic: pd.Series) -> dict[str, float]:
    """Mean IC, IC information ratio and the number of dates."""
    clean = ic.dropna()
    return {"ic": float(clean.mean()), "icir": float(clean.mean() / clean.std()) if clean.std() > 0 else float("nan"),
            "n_dates": int(len(clean))}
