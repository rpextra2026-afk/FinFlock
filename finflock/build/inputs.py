"""Cleaned tape -> the inputs of the road-1 sheaf code (events, pinned dials, day 0, split).

The cleaning step already decided what counts as a company pin; this module only reshapes it. Nothing here looks at a label.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from finflock.clean.steps import LANE_COMPANY, NODE_COMPANY
from finflock.common.config import Config
from finflock.common.schema import UNIT_ID
from finflock.common.timing import TradingCalendar
from finflock.framework.events import EventReading

ROOT = "build"
SPLIT_FIT = "fit"
SPLIT_HOLDOUT = "holdout"
DEV = "dev"
TEST = "test"
DIALS = ("ret", "vol", "volu")


@dataclass
class SheafInputs:
    readings: dict[str, EventReading]
    dials: dict[str, pd.DataFrame]
    day0: dict[str, pd.Timestamp]
    split: dict[str, str]
    kinds: dict[str, str]
    pin_types: dict[str, tuple[str, ...]]
    counts: dict[str, Any]


def group_probabilities(cfg: Config) -> np.ndarray:
    """One pooled group, or the cause groups when the cause-split ablation is switched on."""
    if bool(cfg.get(f"{ROOT}.sheaf.cause_split")):
        raise NotImplementedError("the cause-split ablation needs the per-reading cause group; it is wired in the ablation stage")
    return np.ones(1)


def embargo_cut(calendar: TradingCalendar, holdout_start: str, sessions: int) -> pd.Timestamp:
    """Fit events whose day 0 is within ``sessions`` sessions of the hold-out start are dropped (their labels reach into the hold-out)."""
    return calendar.shift(holdout_start, -sessions)


def build_inputs(clean: pd.DataFrame, run_cfg: dict[str, Any], calendar: TradingCalendar, cfg: Config) -> SheafInputs:
    """One event per company-lane item; pins are its company touch points, weighted by trust x repeat; signed by direction for the return dial."""
    drop_zero = bool(cfg.get(f"{ROOT}.sheaf.drop_zero_pins"))
    embargo = int(cfg.get("windows.calendar_embargo_trading_days"))
    cut = embargo_cut(calendar, str(run_cfg["holdout_start"]), embargo)
    company = clean[(clean["lane"] == LANE_COMPANY) & (clean["node_type"] == NODE_COMPANY)].copy()
    company["size"] = company["size_pct"].astype(float).fillna(0.0)
    counts: dict[str, Any] = {"company_pin_rows": int(len(company))}
    if drop_zero:
        zero = company["size"].eq(0.0)
        counts["zero_size_pins_dropped"] = int(zero.sum())
        company = company[~zero]
    group = np.ones(1)
    readings: dict[str, EventReading] = {}
    dials: dict[str, pd.DataFrame] = {}
    day0: dict[str, pd.Timestamp] = {}
    split: dict[str, str] = {}
    kinds: dict[str, str] = {}
    pin_types: dict[str, tuple[str, ...]] = {}
    embargoed = 0
    for unit, part in company.groupby(UNIT_ID, sort=False):
        session = pd.Timestamp(part["session_date"].iloc[0])
        which = str(part["run_split"].iloc[0])
        if which == SPLIT_FIT and session > cut:
            embargoed += 1
            continue
        merged = part.groupby("node_id", sort=False).agg(size=("size", "sum"), weight=("weight", "first"), role=("pin_role", "first"), magnitude=("magnitude", "max"))
        pins = tuple({"node_id": node, "direction": float(np.sign(row["size"])), "value_effect_pct": float(row["size"]), "magnitude": row["magnitude"], "role": row["role"]} for node, row in merged.iterrows())
        readings[unit] = EventReading(event_id=unit, pins=pins, cause_probs={}, group_probs=group, certainty=float("nan"))
        weight = merged["weight"].to_numpy(dtype=float)
        size = merged["size"].to_numpy(dtype=float)
        dials[unit] = pd.DataFrame({"ret": size * weight, "vol": np.abs(size) * weight, "volu": np.abs(size) * weight}, index=merged.index)
        day0[unit] = session
        split[unit] = DEV if which == SPLIT_FIT else TEST
        kinds[unit] = str(part["kind"].iloc[0])
        pin_types[unit] = tuple(str(r) for r in merged["role"])
    counts.update({"events": len(readings), "events_dev": sum(v == DEV for v in split.values()), "events_test": sum(v == TEST for v in split.values()), "fit_events_embargoed": embargoed})
    return SheafInputs(readings, dials, day0, split, kinds, pin_types, counts)
