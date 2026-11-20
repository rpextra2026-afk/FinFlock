"""Simulated annealing over the agent-type distribution (repository logic, paper hyperparameters).

At date ``t`` the optimiser looks at a window of past dates whose forward-return labels have fully
realised (``look_back_window + interval`` sessions back), proposes a tweak that moves weight between two
random types, accepts it by the Metropolis rule and keeps the best distribution by rank IC. Labels are
passed in as data; the module never reads a label store.
"""

from __future__ import annotations

import bisect
import math
from collections.abc import Callable

import numpy as np
import pandas as pd

from finfm.common.config import Config
from finfm.common.rng import make_rng
from finfm.rivals.common import rank_ic


def adjust_start(dates: list, current, look_back: int, interval: int):
    """First date of the optimisation window, or ``None`` while there is not enough history."""
    ordered = sorted(dates)
    index = bisect.bisect_left(ordered, current)
    return None if index - look_back - interval < 0 else ordered[index - look_back - interval]


def tweak(distribution: dict[str, float], rng: np.random.Generator, step_std: float) -> dict[str, float]:
    """Move a normal draw of weight from one random type to another, keeping weights positive."""
    keys = sorted(distribution)
    if len(keys) < 2:
        raise ValueError("at least two agent types are needed")
    while True:
        step = float(rng.normal(0.0, step_std))
        a, b = rng.choice(len(keys), size=2, replace=False)
        if distribution[keys[a]] - step > 0 and distribution[keys[b]] + step > 0:
            out = dict(distribution)
            out[keys[a]] -= step
            out[keys[b]] += step
            return out


def fitness(signal_frames: dict, labels: dict, column: str) -> float:
    """Mean over dates of the rank IC between the chosen signal column and the realised forward return."""
    values = []
    for date, frame in signal_frames.items():
        label = labels.get(date)
        if label is not None:
            values.append(rank_ic(frame.set_index("Stock")[column], label))
    clean = [v for v in values if v == v]
    return float(np.mean(clean)) if clean else float("-inf")


def anneal(
    distribution: dict[str, float], window_dates: list, signals_for: Callable[[dict[str, float], list], dict],
    labels: dict, cfg: Config, seed_parts: tuple,
) -> dict[str, float]:
    """Best distribution found by simulated annealing on the window (``signals_for(dist, dates)`` -> frames)."""
    spec = cfg.get("rivals.mass.annealing")
    column = str(cfg.get("rivals.mass.fitness_column"))
    rng = make_rng(*seed_parts)
    current = dict(distribution)
    current_fit = fitness(signals_for(current, window_dates), labels, column)
    best, best_fit = dict(current), current_fit
    temperature = float(spec["init_temp"])
    for _ in range(int(spec["max_iter"])):
        proposal = tweak(current, rng, float(spec["step_std"]))
        proposal_fit = fitness(signals_for(proposal, window_dates), labels, column)
        if proposal_fit > current_fit:
            current, current_fit = proposal, proposal_fit
        else:
            gap = (current_fit - proposal_fit) * float(spec["diff_scale"])
            if math.isfinite(gap) and rng.random() < math.exp(-gap / temperature):
                current, current_fit = proposal, proposal_fit
        if current_fit > best_fit:
            best, best_fit = dict(current), current_fit
        temperature *= float(spec["cooling_rate"])
    return best
