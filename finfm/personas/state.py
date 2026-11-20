"""Persona state and the sentences that carry it into the prompt (spec 6.6, 6.7, plan D6).

State is computed, never read from the model: ``state_p(t) = sum_j w_pj R_j(t - W, t - 1)``, the
holdings-weighted cumulative return over the last ``W`` sessions closed before the event. Holdings
are the latest filed ones (buy-and-hold since filing); retail uses its synthetic basket.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import pandas as pd

from finfm.common.config import Config

PERCENT = 100.0
SYNTHETIC = "synthetic"
FILED = "13f"


def window_return(returns: pd.DataFrame, end_session: pd.Timestamp, sessions: int) -> pd.Series:
    """Cumulative total return of each security over the ``sessions`` sessions ending at ``end_session``.

    ``returns`` is a daily panel (sessions x securities); a missing return counts as 0 for that day.
    """
    window = returns.loc[:end_session].iloc[-sessions:]
    return (1.0 + window.fillna(0.0)).prod() - 1.0


def portfolio_state(weights: pd.Series, window_returns: pd.Series, market_return: float) -> dict[str, float]:
    """Holdings-weighted return, the return relative to the market, and the covered weight share."""
    covered = weights.reindex(window_returns.index).dropna()
    total = float(weights.sum())
    if covered.empty or total <= 0:
        return {"state": 0.0, "state_rel": -market_return, "covered": 0.0}
    state = float((covered * window_returns.reindex(covered.index)).sum() / covered.sum())
    return {"state": state, "state_rel": state - market_return, "covered": float(covered.sum() / total)}


def within_type_percentile(values: pd.Series, types: pd.Series) -> pd.Series:
    """Percentile rank (0-1) of each persona's state among personas of its own type."""
    return values.groupby(types).rank(pct=True, method="average")


def state_z(values: pd.Series, types: pd.Series) -> pd.Series:
    """Within-type standardised state ``z`` used by the risk-aversion knob (0 when a type has no spread)."""
    grouped = values.groupby(types)
    spread = grouped.transform("std").replace(0.0, np.nan)
    return ((values - grouped.transform("mean")) / spread).fillna(0.0)


def state_sentence(cfg: Config, state: float, market: float) -> str:
    """"Over the past three months your portfolio is up 8% (ahead of the market, which is up 3%)."""
    template = str(cfg.get("prompts.persona.state_sentence"))
    return template.format(
        direction="up" if state >= 0 else "down",
        size=abs(state) * PERCENT,
        relation="ahead of" if state >= market else "behind",
        market_direction="up" if market >= 0 else "down",
        market_size=abs(market) * PERCENT,
    )


def mandate_sentence(cfg: Config, mandate: float) -> str:
    low, high = (float(x) for x in cfg.get("personas.mandate_room_edges"))
    key = "low" if mandate < low else "medium" if mandate < high else "high"
    return str(cfg.get("prompts.persona.mandate_sentence")).format(
        room=cfg.get(f"prompts.persona.mandate_room.{key}")
    )


def exposure_sentence(
    cfg: Config, sector: str, sector_weight: float, ref: str, asset_weight: float
) -> str:
    """Own exposure in words; zero exposure is stated plainly (spec 6.6)."""
    return str(cfg.get("prompts.persona.exposure_sentence")).format(
        sector_pct=sector_weight * PERCENT, sector=sector, asset_pct=asset_weight * PERCENT, ref=ref
    )


def institution_history(
    cfg: Config, previous: Mapping[str, float], current: Mapping[str, float]
) -> tuple[str, str]:
    """Last-trade sentence from two consecutive filed sector-weight vectors, and its source tag.

    Names the sectors with the largest increase and decrease between the two filings that are
    available on the event date. Changes below ``history.min_change`` are not mentioned.
    """
    sectors = sorted(set(previous) | set(current))
    change = {s: current.get(s, 0.0) - previous.get(s, 0.0) for s in sectors}
    top = int(cfg.get("personas.history.top_sectors"))
    floor = float(cfg.get("personas.history.min_change"))
    added = [s for s, c in sorted(change.items(), key=lambda kv: (-kv[1], kv[0])) if c >= floor][:top]
    trimmed = [s for s, c in sorted(change.items(), key=lambda kv: (kv[1], kv[0])) if c <= -floor][:top]
    if not added and not trimmed:
        return str(cfg.get("prompts.persona.history_institution_none")), FILED
    template = str(cfg.get("prompts.persona.history_institution"))
    return (
        template.format(added=" and ".join(added) or "nothing", trimmed=" and ".join(trimmed) or "nothing"),
        FILED,
    )


def retail_history(cfg: Config, turnover_class: str) -> tuple[str, str]:
    """Synthetic last-trade sentence from the turnover class. No data identify retail trades."""
    key = str(cfg.get(f"personas.history.retail_rotations.{turnover_class}"))
    return str(cfg.get(f"prompts.persona.history_retail.{key}")), SYNTHETIC
