"""N1: Loughran-McDonald tone of a text, (pos - neg) / (pos + neg), 0 when no sentiment word (dictionary bundled with pysentiment2)."""

from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


def tone_from_counts(positive: int, negative: int) -> float:
    total = positive + negative
    return 0.0 if total == 0 else (positive - negative) / total


def lm_tone(texts: Iterable[str]) -> pd.Series:
    import pysentiment2 as ps

    lm = ps.LM()
    out = []
    for text in texts:
        score = lm.get_score(lm.tokenize(str(text)))
        out.append(tone_from_counts(int(score["Positive"]), int(score["Negative"])))
    return pd.Series(out, dtype=float)


def lm_version() -> str:
    from importlib.metadata import version

    return f"pysentiment2 {version('pysentiment2')}"
