"""Road 2 for one event: persona answers to beliefs and knobs, the anchored path, dials (spec 7.5-7.7)."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from oryxis.common.config import Config
from oryxis.mfg.path import Anchor, anchored_path, expected_path, predicted_increments, price_path
from oryxis.mfg.precompute import MonthGame
from oryxis.mfg.push import PushResult, end_point
from oryxis.mfg.speed import EventKnobs, event_knobs
from oryxis.mfg.volume import LowRankCov, covariance_factor, event_volume

VALUE_SCALE = 100.0  # persona value changes are percent
TINY = 1e-12  # guards a division by an empty ownership weight


@dataclass(frozen=True)
class PersonaEventAnswer:
    """One persona's de-leaned answer to one event (zeros and ``saw=False`` when it was not called)."""

    saw: bool
    fa: float
    ar: float
    values: Mapping[str, float]  # node id -> value change in percent
    d_draw: float = 0.0


@dataclass(frozen=True)
class Road2Result:
    path: np.ndarray  # (H+1, N) prices
    expected: np.ndarray  # (H+1, N)
    increments: np.ndarray  # (H, N) predicted tradable increments, day 1 first
    push: PushResult
    knobs: EventKnobs
    anchor: Anchor | None
    volume_dial: np.ndarray
    one_sided: np.ndarray
    cov_shift: LowRankCov  # the (H, n) factor; .dense() is the (n, n) matrix
    vol_dial: np.ndarray
    sleeve_flow: float
    beliefs: np.ndarray


def belief_vectors(
    cfg: Config, game: MonthGame, consensus: np.ndarray, answers: Mapping[str, PersonaEventAnswer],
    retail_ids: list[str], asset_index: Mapping[str, int], uncovered: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Type beliefs ``delta_k`` (K, N) and the outside belief ``delta_0`` (N,) with the retail mixture.

    A persona that saw the event reports ``dev = v / 100 - consensus`` on the assets it was shown, zero
    elsewhere; a persona that did not see it has no deviation. Retail enters the outside belief:
    ``delta_0 = (1 - u) consensus + u retail`` with ``u = rho_ret * uncovered_share`` (A10, D16).
    """
    n = game.n_assets
    deltas = np.tile(consensus, (len(game.types), 1))
    for k, fund in enumerate(game.types):
        answer = answers.get(fund.name)
        if answer is None or not answer.saw:
            continue
        for node, value in answer.values.items():
            if node in asset_index:
                j = asset_index[node]
                deltas[k, j] = value / VALUE_SCALE
    retail = consensus.copy()
    seen = [answers[r] for r in retail_ids if r in answers and answers[r].saw]
    for j_node, j in asset_index.items():
        values = [(a.values[j_node] / VALUE_SCALE) if j_node in a.values else consensus[j] for a in seen] + [
            consensus[j] for r in retail_ids if r not in answers or not answers[r].saw
        ]
        if values:
            retail[j] = float(np.mean(values))
    u = float(cfg.get("game.retail_rho")) * uncovered
    delta0 = (1.0 - u) * consensus + u * retail
    return deltas, delta0


def knob_inputs(game: MonthGame, answers: Mapping[str, PersonaEventAnswer], state_z: Mapping[str, float],
                trait_b: Mapping[str, float]) -> dict[str, np.ndarray]:
    fa = np.array([answers[f.name].fa if f.name in answers and answers[f.name].saw else 0.0 for f in game.types])
    ar = np.array([answers[f.name].ar if f.name in answers and answers[f.name].saw else 0.0 for f in game.types])
    z = np.array([state_z.get(f.name, 0.0) for f in game.types])
    b = np.array([trait_b.get(f.name, 0.0) for f in game.types])
    return {"fear_minus_anger": fa, "arousal": ar, "state_z": z, "trait_b": b}


def run_road2_event(
    cfg: Config, game: MonthGame, consensus: np.ndarray, answers: Mapping[str, PersonaEventAnswer],
    retail_ids: list[str], asset_index: Mapping[str, int], uncovered: np.ndarray, state_z: Mapping[str, float],
    trait_b: Mapping[str, float], pinned: np.ndarray, observed: np.ndarray, baseline_dv: np.ndarray,
    volume_window: int, vol_kappas: tuple[float, float],
) -> Road2Result:
    """Beliefs, knobs, anchored path, expected increments and the road-2 dials of one event."""
    deltas, delta0 = belief_vectors(cfg, game, consensus, answers, retail_ids, asset_index, uncovered)
    knobs = event_knobs(game, cfg, **knob_inputs(game, answers, state_z, trait_b))
    horizon = int(cfg.get("game.horizon_days"))
    if len(pinned):
        path, anchor, push = anchored_path(game, knobs, deltas, delta0, pinned, observed, cfg)
    else:
        push = end_point(game, knobs, deltas, delta0)
        path, anchor = price_path(push, knobs.phi, delta0, horizon), None
    expected = expected_path(path, consensus, float(cfg.get("game.varsigma")))
    volume = event_volume(game, push, knobs.phi, path, delta0, baseline_dv, volume_window)
    dispersion = np.array([
        np.std([a.values.get(node, np.nan) for a in answers.values() if a.saw and node in a.values] or [0.0])
        for node in sorted(asset_index, key=asset_index.get)
    ]) / VALUE_SCALE
    book = np.array([t.book for t in game.types])
    fa_asset = np.array([
        np.sum([book[k] * game.types[k].x_sp[j] * knob for k, knob in enumerate(knob_inputs(game, answers, state_z, trait_b)["fear_minus_anger"])])
        / max(np.sum([book[k] * game.types[k].x_sp[j] for k in range(len(game.types))]), TINY)
        for j in range(game.n_assets)
    ])
    vol_dial = vol_kappas[0] * dispersion + vol_kappas[1] * np.abs(fa_asset)
    return Road2Result(
        path=path, expected=expected, increments=predicted_increments(expected), push=push, knobs=knobs, anchor=anchor,
        volume_dial=volume["volume_dial"], one_sided=volume["one_sided_share"],
        cov_shift=covariance_factor(path, delta0, horizon), vol_dial=vol_dial, sleeve_flow=push.sleeve_flow, beliefs=deltas,
    )
