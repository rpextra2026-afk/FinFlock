"""From tapes to the framework's inputs: persona answers per event, ready for the game."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping

import pandas as pd

from oryxis.common.config import Config
from oryxis.common.schema import STATUS, UNIT_ID, Status
from oryxis.framework.road2_stage import PersonaEventAnswer
from oryxis.personas.postprocess import (
    DRAW,
    EVENT,
    PERSONA,
    SESSION,
    TYPE,
    aggregate_draws,
    emotion_variables,
    remove_lean,
)
from oryxis.personas.run import DRAWS_JSON, EVENT_ID, PERSONA_ID, SAW_FOCAL


def explode_draws(tape: pd.DataFrame, persona_type: Mapping[str, str], session_of: Mapping[str, int]) -> pd.DataFrame:
    """One row per (event, persona, draw) for the units that were called and parsed, with the mean of the
    shown assets' value changes as ``value_mean``."""
    rows = []
    for record in tape[(tape[STATUS] == Status.OK.value) & tape[SAW_FOCAL]].itertuples():
        for draw, answer in enumerate(json.loads(getattr(record, DRAWS_JSON))):
            if not answer["ok"] or not answer["values"]:
                continue
            values = answer["values"]
            rows.append({
                EVENT: getattr(record, EVENT_ID), PERSONA: getattr(record, PERSONA_ID),
                TYPE: persona_type[getattr(record, PERSONA_ID)], SESSION: session_of[getattr(record, EVENT_ID)],
                DRAW: draw, "fear": answer["fear"], "anger": answer["anger"], "calm": answer["calm"],
                "conviction": answer["conviction"], "value_mean": sum(values.values()) / len(values), "values": values,
            })
    return pd.DataFrame(rows)


def persona_answers(
    tape: pd.DataFrame, persona_type: Mapping[str, str], session_of: Mapping[str, int], cfg: Config
) -> Callable[[str], dict[str, PersonaEventAnswer]]:
    """Lean-removed answers per event: ``fa``, ``ar`` and de-leaned per-asset value changes in percent."""
    draws = explode_draws(tape, persona_type, session_of)
    if draws.empty:
        return lambda event_id: {}
    cleaned = remove_lean(draws.drop(columns="values"), cfg).merge(
        draws[[EVENT, PERSONA, DRAW, "values"]], on=[EVENT, PERSONA, DRAW]
    )
    cleaned["lean"] = cleaned["value_mean"] - cleaned["value_mean_star"]
    summary = emotion_variables(aggregate_draws(cleaned), float(cfg.get("personas.answer.scale_max")))
    summary = summary.set_index([EVENT, PERSONA])
    by_event: dict[str, dict[str, PersonaEventAnswer]] = {}
    for (event, persona), part in cleaned.groupby([EVENT, PERSONA]):
        lean = float(part["lean"].mean())
        assets: dict[str, list[float]] = {}
        for values in part["values"]:
            for node, v in values.items():
                assets.setdefault(node, []).append(v)
        row = summary.loc[(event, persona)]
        by_event.setdefault(event, {})[persona] = PersonaEventAnswer(
            True, float(row["fa"]), float(row["ar"]),
            {node: float(sum(v) / len(v)) - lean for node, v in assets.items()}, float(row["d_draw"]),
        )
    return lambda event_id: by_event.get(event_id, {})
