"""Post-processing of the persona tape on CPU (spec 6.10): lean removal, draws, emotion variables.

Input is one row per (event, persona, draw) with the persona's answers; nothing here touches labels.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finfm.common.config import Config

EVENT = "event_id"
PERSONA = "persona_id"
TYPE = "persona_type"
SESSION = "session"  # integer index of the event's day-0 session in the trading calendar
FIELDS = ("fear", "anger", "calm", "conviction", "value_mean")
DRAW = "draw"


def running_lean(
    sessions: np.ndarray, values: np.ndarray, window: int, min_gap: int
) -> np.ndarray:
    """Mean of the previous ``window`` answers made at least ``min_gap`` sessions earlier.

    ``sessions`` must be sorted ascending. Entry ``i`` uses answers ``j`` with
    ``sessions[j] <= sessions[i] - min_gap``, the last ``window`` of them; NaN when there are none.
    """
    cutoff = np.searchsorted(sessions, sessions - min_gap, side="right")
    csum = np.concatenate([[0.0], np.cumsum(values)])
    start = np.maximum(cutoff - window, 0)
    count = cutoff - start
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(count > 0, (csum[cutoff] - csum[start]) / count, np.nan)


def remove_lean(answers: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Subtract each persona's running mean from each numeric field (type-level mean on cold start)."""
    window = int(cfg.get("personas.lean.window_answers"))
    gap = int(cfg.get("personas.lean.min_gap_sessions"))
    ordered = answers.sort_values([SESSION, PERSONA, DRAW], kind="mergesort").reset_index(drop=True)
    out = ordered.copy()
    for field in FIELDS:
        own = pd.Series(np.nan, index=ordered.index)
        for _, part in ordered.groupby(PERSONA, sort=False):
            own.loc[part.index] = running_lean(
                part[SESSION].to_numpy(), part[field].to_numpy(), window, gap
            )
        by_type = pd.Series(np.nan, index=ordered.index)
        for _, part in ordered.groupby(TYPE, sort=False):
            by_type.loc[part.index] = running_lean(
                part[SESSION].to_numpy(), part[field].to_numpy(), window, gap
            )
        out[f"{field}_star"] = ordered[field] - own.fillna(by_type).fillna(0.0)
    return out


def aggregate_draws(deleaned: pd.DataFrame) -> pd.DataFrame:
    """Mean over the draws of each (event, persona) and the draw standard deviation ``D_draw``."""
    stars = [f"{f}_star" for f in FIELDS]
    grouped = deleaned.groupby([EVENT, PERSONA, TYPE, SESSION], sort=False)
    mean = grouped[stars].mean()
    spread = grouped["value_mean_star"].std(ddof=0).rename("d_draw")
    return mean.join(spread).reset_index()


def emotion_variables(aggregated: pd.DataFrame, scale_max: float) -> pd.DataFrame:
    """``fa = (fear - anger) / 10`` in [-1, 1] and ``ar = (max(fear, anger) - calm) / 10`` in [-1, 1]."""
    out = aggregated.copy()
    out["fa"] = (out["fear_star"] - out["anger_star"]) / scale_max
    out["ar"] = (np.maximum(out["fear_star"], out["anger_star"]) - out["calm_star"]) / scale_max
    return out


def type_belief(per_persona: pd.DataFrame, books: pd.Series, value_col: str) -> pd.DataFrame:
    """Book-weighted mean belief and dispersion of each persona type, per event (retail: equal weights)."""
    frame = per_persona.assign(weight=per_persona[PERSONA].map(books).fillna(1.0))
    frame.loc[frame[TYPE] == "retail", "weight"] = 1.0
    rows = []
    for (event, ptype), part in frame.groupby([EVENT, TYPE]):
        w = part["weight"].to_numpy() / part["weight"].sum()
        v = part[value_col].to_numpy()
        mean = float(w @ v)
        rows.append({EVENT: event, TYPE: ptype, "belief": mean, "dispersion": float(np.sqrt(w @ (v - mean) ** 2))})
    return pd.DataFrame(rows)


def persona_dispersion(per_persona: pd.DataFrame, books: pd.Series, value_col: str) -> pd.Series:
    """Ownership-weighted standard deviation of the value change across all personas, per event."""
    result = {}
    for event, part in per_persona.groupby(EVENT):
        w = part[PERSONA].map(books).fillna(1.0).to_numpy()
        w = w / w.sum()
        v = part[value_col].to_numpy()
        result[event] = float(np.sqrt(w @ (v - w @ v) ** 2))
    return pd.Series(result, name="d_persona")
