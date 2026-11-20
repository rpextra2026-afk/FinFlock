"""The cleaned tape (``clean_<run>.parquet``) as the framework's inputs: ``EventReading`` objects, source dials, an events table and the world-lane pins.

One row of the clean file is one pin of one item. This module only reshapes it; the cleaning (gate, dedupe, session timing, trust x repeat weight)
was decided before and nothing here looks at a label or a price.

Rules (spec of the final framework run):
  * a reading whose cause is ``other`` or ``no_cause`` carries NO cause-group weight: its group vector is all zeros, so it cannot push any group;
  * a pin whose node is not a company (theme, market, sector) never enters the company sheaf; it goes to the world lane;
  * the weight of a pin is the cleaned weight (trust x repeat);
  * ``session_date`` of the clean file is the first session that can react to the item; it is the event's day 0 and decides the first tradable day.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from finfm.clean.steps import LANE_COMPANY, LANE_WORLD, NODE_COMPANY
from finfm.common.config import Config
from finfm.common.schema import UNIT_ID
from finfm.framework.events import UNKNOWN_SCHEDULED, EventReading
from finfm.reader.salvage import SOURCE_MISSING

SPLIT_FIT, SPLIT_HOLDOUT = "fit", "holdout"
DEV, TEST = "dev", "test"
DIALS = ("ret", "vol", "volu")
NO_GROUP_CAUSES = ("other",)  # causes that never carry a group weight, besides the no-cause group itself


@dataclass
class CleanEvents:
    readings: dict[str, EventReading]
    dials: dict[str, pd.DataFrame]  # per event, indexed by node: ret, vol, volu as fractions x weight, plus s_ret, s_vol, s_volu in the sheaf's training units
    events: pd.DataFrame  # event_id, kind, split, session_day0, session_type, cause_group, cause_top, repeat_count
    world_pins: pd.DataFrame  # every non-company pin, lane set to world_event, for finfm.build.world
    counts: dict[str, Any] = field(default_factory=dict)


def group_vector(cause_top: str, cause_group: str, cfg: Config) -> tuple[dict[str, float], np.ndarray]:
    """Cause probabilities (one-hot on the cause) and the group vector in the order of ``causes.groups``; all zeros for ``other`` and the no-cause group."""
    causes = list(cfg.get("causes.causes"))
    groups = cfg.get("causes.groups")
    no_cause = str(cfg.get("clean.causes.no_cause_group"))
    probs = {c: 0.0 for c in causes}
    if cause_top in NO_GROUP_CAUSES or cause_group == no_cause or cause_top not in probs:
        return probs, np.zeros(len(groups))
    probs[cause_top] = 1.0
    return probs, np.array([sum(probs[c] for c in members) for members in groups.values()])


def pin_dials(part: pd.DataFrame, pin_scale: float) -> tuple[tuple[dict, ...], pd.DataFrame]:
    """Pins of one event (merged by node) and their source dials: weight x size, with the signed size for ``ret``.

    ``ret/vol/volu`` are fractions (size_pct / 100 x weight); ``s_*`` are what the build's sheaf was trained on (size_pct x weight, percent)."""
    size = part["size_pct"].astype(float).fillna(0.0)
    frame = part.assign(size=size)
    merged = frame.groupby("node_id", sort=False).agg(size=("size", "sum"), weight=("weight", "first"), role=("pin_role", "first"), magnitude=("magnitude", "max"))
    pins = tuple({"node_id": node, "direction": float(np.sign(row["size"])), "value_effect_pct": float(row["size"]), "magnitude": row["magnitude"], "role": row["role"]}
                 for node, row in merged.iterrows())
    size_v, weight = merged["size"].to_numpy(dtype=float), merged["weight"].to_numpy(dtype=float)
    out = pd.DataFrame({"ret": size_v * weight / pin_scale, "vol": np.abs(size_v) * weight / pin_scale, "volu": np.abs(size_v) * weight / pin_scale,
                        "s_ret": size_v * weight, "s_vol": np.abs(size_v) * weight, "s_volu": np.abs(size_v) * weight}, index=merged.index)
    return pins, out


def from_clean(clean: pd.DataFrame, run_cfg: dict[str, Any], cfg: Config, embargo_cut: pd.Timestamp | None = None, drop_zero_pins: bool = True, test_from: pd.Timestamp | None = None) -> CleanEvents:
    """Readings, source dials, events and world pins of one cleaned tape.

    ``embargo_cut``: fit-window events later than this day are left out (their labels reach into the hold-out), as in ``finfm.build.inputs``.
    ``test_from``: only the sealed-window stage sets it: test events whose day 0 is earlier are left out (they belong to the hold-out the main run already used)."""
    pin_scale = float(cfg.get("framework.pooled.sheaf_pin_scale"))
    is_company = clean["node_type"].eq(NODE_COMPANY)
    company = clean[is_company & clean["lane"].eq(LANE_COMPANY)]
    other = clean[~is_company & clean["node_type"].notna()]  # theme / market / sector pins, wherever they sit in the file
    world = other.assign(lane=LANE_WORLD)
    counts: dict[str, Any] = {"company_pin_rows": int(len(company)), "non_company_pin_rows_to_world_lane": int(len(world)), "pin_rows_total": int(len(clean))}
    if drop_zero_pins:
        zero = company["size_pct"].astype(float).fillna(0.0).eq(0.0)
        counts["zero_size_pins_dropped"] = int(zero.sum())
        company = company[~zero]
    readings: dict[str, EventReading] = {}
    dials: dict[str, pd.DataFrame] = {}
    rows = []
    embargoed = no_group = before_test = 0
    for unit, part in company.groupby(UNIT_ID, sort=False):
        head = part.iloc[0]
        session = pd.Timestamp(head["session_date"])
        split = DEV if str(head["run_split"]) == SPLIT_FIT else TEST
        if split == DEV and embargo_cut is not None and session > embargo_cut:
            embargoed += 1
            continue
        if split == TEST and test_from is not None and session < test_from:
            before_test += 1
            continue
        pins, frame = pin_dials(part, pin_scale)
        probs, group = group_vector(str(head["cause_top"]), str(head["cause_group"]), cfg)
        no_group += int(not group.any())
        readings[str(unit)] = EventReading(str(unit), pins, probs, group, float("nan"), UNKNOWN_SCHEDULED, SOURCE_MISSING, SOURCE_MISSING, False)
        dials[str(unit)] = frame
        rows.append({"event_id": str(unit), "kind": head["kind"], "split": split, "session_day0": session, "session_type": head["session_type"],
                     "cause_group": head["cause_group"], "cause_top": head["cause_top"], "repeat_count": int(head["repeat_count"])})
    events = pd.DataFrame(rows, columns=["event_id", "kind", "split", "session_day0", "session_type", "cause_group", "cause_top", "repeat_count"])
    counts.update({"events": len(readings), "events_dev": int((events["split"] == DEV).sum()), "events_test": int((events["split"] == TEST).sum()),
                   "fit_events_embargoed": embargoed, "test_events_before_test_from": before_test, "events_with_all_zero_group_vector": no_group})
    return CleanEvents(readings, dials, events, world.reset_index(drop=True), counts)
