"""The input transforms behind the ablation switches (ablation-spec-for-claude-code.md, section 5).

Every function takes the objects the framework cell reads and returns changed copies; none of them reads a label or a result. Randomness comes only from
``make_rng`` with a named identity (switch, bucket, llm, draw, ...), so a draw gives the identical shuffle when it is run again.
"""

from __future__ import annotations

import dataclasses
from collections import Counter
from collections.abc import Callable, Mapping

import numpy as np
import pandas as pd

from finfm.common.config import Config
from finfm.common.rng import make_rng
from finfm.framework.events import EventReading
from finfm.framework.road2_stage import PersonaEventAnswer
from finfm.mfg.precompute import FundInput
from finfm.personas.skeletons import Skeleton
from finfm.road1.graph import LINK_DST, LINK_SRC, LINK_TYPE

PIN_TEXT_FIELDS = ("direction", "value_effect_pct", "magnitude")  # what the reader wrote about each pin; node id and role are structure
DIRECTION_UP, DIRECTION_DOWN, DIRECTION_NONE = "up", "down", "none"
QUARTER_SEPARATOR = "@"
SCALE_PCT_TO_FRACTION = 100.0


def _group_probs(cause_probs: Mapping[str, float], cfg: Config) -> np.ndarray:
    return np.array([sum(cause_probs[c] for c in members) for members in cfg.get("causes.groups").values()])


def _dev_means(readings: Mapping[str, EventReading], kinds: Mapping[str, str], dev_ids: list[str]) -> dict[str, dict]:
    """Per news kind, the dev-period average of every field the language model wrote."""
    out: dict[str, dict] = {}
    for kind in sorted({kinds[e] for e in dev_ids}):
        part = [readings[e] for e in dev_ids if kinds[e] == kind]
        pins = [p for r in part for p in r.pins]
        certainty = [r.certainty for r in part if not np.isnan(r.certainty)]
        out[kind] = {
            "cause_probs": {c: float(np.mean([r.cause_probs[c] for r in part])) for c in part[0].cause_probs},
            "value_effect_pct": float(np.mean([p["value_effect_pct"] for p in pins])) if pins else 0.0,
            "magnitude": float(np.mean([p["magnitude"] for p in pins])) if pins else 0.0,
            "certainty": float(np.mean(certainty)) if certainty else float("nan"),
            "scheduled": Counter(r.scheduled for r in part).most_common(1)[0][0],
        }
    return out


def text_blind(readings: Mapping[str, EventReading], kinds: Mapping[str, str], dev_ids: list[str], cfg: Config) -> dict[str, EventReading]:
    """R1: every field the language model wrote is replaced by the dev average of the same news kind. Pins, times and prices stay."""
    means = _dev_means(readings, kinds, dev_ids)
    out = {}
    for event_id, reading in readings.items():
        mean = means.get(kinds[event_id])
        if mean is None:  # a kind that has no dev event: nothing to average, the reading is kept and the event is counted by the caller
            out[event_id] = reading
            continue
        effect = mean["value_effect_pct"]
        direction = DIRECTION_UP if effect > 0 else DIRECTION_DOWN if effect < 0 else DIRECTION_NONE
        pins = tuple({**p, "direction": direction, "value_effect_pct": effect, "magnitude": mean["magnitude"]} for p in reading.pins)
        out[event_id] = dataclasses.replace(reading, pins=pins, cause_probs=dict(mean["cause_probs"]), group_probs=_group_probs(mean["cause_probs"], cfg),
                                            certainty=mean["certainty"], scheduled=mean["scheduled"])
    return out


def text_shuffled(readings: Mapping[str, EventReading], kinds: Mapping[str, str], day0_date: Mapping[str, pd.Timestamp], cfg: Config,
                  identity: tuple) -> tuple[dict[str, EventReading], dict[str, int]]:
    """R2: the same fields are permuted across events of the same calendar day and news kind; each event keeps its own pins and prices.

    Returns the readings and ``{"groups": n, "events_moved": n, "events_alone": n}`` (an event alone in its day and kind keeps its reading)."""
    groups: dict[tuple, list[str]] = {}
    for event_id in sorted(readings):
        groups.setdefault((day0_date[event_id], kinds[event_id]), []).append(event_id)
    donor: dict[str, str] = {}
    for key, members in groups.items():
        if len(members) < 2:
            continue
        order = make_rng(*identity, str(key[0].date()), key[1]).permutation(len(members))
        for position, member in zip(order, members, strict=True):
            donor[member] = members[int(position)]
    out = dict(readings)
    for event_id, other in donor.items():
        source, own = readings[other], readings[event_id]
        pins = tuple({**p, **{f: source.pins[i % len(source.pins)][f] for f in PIN_TEXT_FIELDS}} for i, p in enumerate(own.pins))
        out[event_id] = dataclasses.replace(own, pins=pins, cause_probs=dict(source.cause_probs), group_probs=_group_probs(source.cause_probs, cfg),
                                            certainty=source.certainty, scheduled=source.scheduled)
    return out, {"groups": len(groups), "events_moved": len(donor), "events_alone": len(readings) - len(donor)}


def sp500_only(readings: Mapping[str, EventReading], links: pd.DataFrame, sp_nodes: set[str]) -> tuple[dict[str, EventReading], pd.DataFrame, int]:
    """G2: pins on nodes outside the S&P 500 are dropped (their news is not carried to S&P names) and links keep S&P ends only.

    Returns the readings (events left with no pin are removed), the links, and the number of events removed."""
    out = {}
    for event_id, reading in readings.items():
        pins = tuple(p for p in reading.pins if p["node_id"] in sp_nodes)
        if pins:
            out[event_id] = dataclasses.replace(reading, pins=pins)
    kept = links[links[LINK_SRC].isin(sp_nodes) & links[LINK_DST].isin(sp_nodes)]
    return out, kept.reset_index(drop=True), len(readings) - len(out)


def links_shuffled(links: pd.DataFrame, identity: tuple) -> pd.DataFrame:
    """G3: neighbour identities are permuted within each link type (the destination column), so every type keeps its degree distribution."""
    out = links.copy()
    for link_type, rows in links.groupby(LINK_TYPE).groups.items():
        positions = out.index.get_indexer(rows)
        order = make_rng(*identity, str(link_type)).permutation(len(positions))
        values = out[LINK_DST].to_numpy(copy=True)
        values[positions] = values[positions][order]
        out[LINK_DST] = values
    return out


# ---- road 2 ---------------------------------------------------------------------------------------------------------------------------

AnswerFn = Callable[[str], Mapping[str, PersonaEventAnswer]]


def common_belief(answers: AnswerFn) -> AnswerFn:
    """P4: every persona that saw the event gets the cross-persona mean value change for each asset; the disagreement between types is deleted."""
    def wrapped(event_id: str) -> dict[str, PersonaEventAnswer]:
        seen = {pid: a for pid, a in answers(event_id).items() if a.saw}
        nodes = sorted({n for a in seen.values() for n in a.values})
        mean = {n: float(np.mean([a.values[n] for a in seen.values() if n in a.values])) for n in nodes}
        return {pid: dataclasses.replace(a, values={n: mean[n] for n in a.values}) for pid, a in seen.items()}
    return wrapped


def day0_values(answers: AnswerFn, moves: Callable[[str, list[str]], Mapping[str, float]], pinned: Callable[[str], list[str]]) -> AnswerFn:
    """P5: every persona's value change on a pinned asset is the observed day-0 move (percent); on any other asset it is zero (the consensus there)."""
    def wrapped(event_id: str) -> dict[str, PersonaEventAnswer]:
        observed = moves(event_id, pinned(event_id))
        return {pid: dataclasses.replace(a, values={n: SCALE_PCT_TO_FRACTION * observed.get(n, 0.0) for n in set(a.values) | set(observed)})
                for pid, a in answers(event_id).items() if a.saw}
    return wrapped


def cards_shuffled(answers: AnswerFn, identity: tuple) -> AnswerFn:
    """P6: within an event and a quarter, the answers are permuted among the personas, so each persona's holdings and weights meet another persona's answer."""
    def wrapped(event_id: str) -> dict[str, PersonaEventAnswer]:
        given = dict(answers(event_id))
        by_quarter: dict[str, list[str]] = {}
        for pid in sorted(given):
            by_quarter.setdefault(pid.split(QUARTER_SEPARATOR)[-1], []).append(pid)
        out = dict(given)
        for quarter, ids in by_quarter.items():
            if len(ids) < 2:
                continue
            order = make_rng(*identity, event_id, quarter).permutation(len(ids))
            for position, pid in zip(order, ids, strict=True):
                out[pid] = given[ids[int(position)]]
        return out
    return wrapped


def drop_answers(answers: AnswerFn, kept: set[str]) -> AnswerFn:
    """P8 and P12: only the answers of the personas in ``kept`` are used."""
    return lambda event_id: {pid: a for pid, a in answers(event_id).items() if pid in kept}


def drop_type(skeletons: Mapping[str, Skeleton], persona_type: str) -> dict[str, Skeleton]:
    """P8: the personas of one type are removed from the answers and from the game."""
    return {pid: s for pid, s in skeletons.items() if s.persona_type != persona_type}


def keep_fraction(skeletons: Mapping[str, Skeleton], fraction: float, identity: tuple) -> dict[str, Skeleton]:
    """P12: a hashed share ``fraction`` of the personas of each type and quarter is kept (at least one), the rest removed."""
    kept: dict[str, Skeleton] = {}
    groups: dict[tuple, list[str]] = {}
    for pid, s in skeletons.items():
        groups.setdefault((s.persona_type, s.asof_quarter), []).append(pid)
    for key, ids in groups.items():
        ids = sorted(ids)
        order = make_rng(*identity, *key).permutation(len(ids))
        for position in order[: max(int(round(fraction * len(ids))), 1)]:
            kept[ids[int(position)]] = skeletons[ids[int(position)]]
    return kept


def equal_books(funds: list[FundInput]) -> list[FundInput]:
    """P7: the ownership-derived type weights are replaced by equal weights: every fund gets the mean book."""
    mean_book = float(np.mean([f.book for f in funds]))
    return [dataclasses.replace(f, book=mean_book) for f in funds]


class RippleOff:
    """G1: road 1 keeps its output on the pinned assets and gives the linked assets nothing (return, vol, volume set to zero)."""

    def __init__(self, road1, dial_columns: tuple[str, ...]) -> None:
        self.road1, self.columns = road1, list(dial_columns)

    def __getattr__(self, name: str):
        return getattr(self.road1, name)

    def predict(self, reading, source_dials, asof):
        frame = self.road1.predict(reading, source_dials, asof)
        frame.loc[frame["role"] == "ripple", self.columns] = 0.0
        return frame
