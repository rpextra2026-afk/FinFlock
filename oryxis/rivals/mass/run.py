"""The MASS job: per (date, investor type, agent) one style call and one decision call (spec 10.4).

Units are ``date|type|agent``. Identical style prompts (same type, same day) are generated once; a
decision depends on the agent's own random stock subset and is generated per agent. With temperature 0
retries are skipped (they would return the same text).
"""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass

import pandas as pd

from oryxis.common.config import Config
from oryxis.common.rng import make_rng
from oryxis.common.schema import DETAIL, PARSE_OK, REASON, STATUS, TOKENS_IN, TOKENS_OUT, UNIT_ID, Reason, Status
from oryxis.compute.engine import Engine, GenParams, GenRequest
from oryxis.rivals.mass.agent import (
    decision_prompt,
    parse_decision,
    parse_style,
    select_pool,
    style_prompt,
)
from oryxis.rivals.mass.modalities import ModalityStore

TASK_STYLE, TASK_DECISION = "mass_style", "mass_decision"
SEP = "|"
DATE, TYPE, AGENT = "date", "investor_type", "agent"
SHOWN, CHOSEN, STYLE = "shown_json", "chosen_json", "style_json"


def unit(date: str, investor_type: str, agent: int) -> str:
    return SEP.join([date, investor_type, str(agent)])


@dataclass
class MassWorld:
    """What the job needs from the data layer."""

    pool: Callable[[str], pd.DataFrame]  # date -> Stock, Industry, FREE_MV of the 100 largest S&P members
    store: ModalityStore
    types: Mapping[str, Sequence[str]]  # investor type -> modalities
    macro: Callable[[str], str | None]
    news: Callable[[str, list[str]], list[str] | None]


class MassProcessor:
    def __init__(self, cfg: Config, engine: Engine, world: MassWorld) -> None:
        self.cfg, self.engine, self.world = cfg, engine, world
        spec = cfg.get("rivals.mass")
        self.params = GenParams(temperature=float(spec["temperature"]), top_p=1.0, top_k=None, max_tokens=int(spec["max_tokens"]),
                                seed=int(cfg.get("runtime.base_seed")))
        self.system = str(cfg.get("mass_prompts.system"))

    def _tables(self, modalities: Sequence[str], date: str, stocks: list[str]) -> dict[str, pd.DataFrame]:
        out = {}
        for modality in modalities:
            frame = self.world.store.frames.get(modality)
            if frame is not None:
                out[modality] = frame[(frame["Date"] == pd.Timestamp(date)) & frame["Stock"].isin(stocks)]
        return out

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        spec = self.cfg.get("rivals.mass")
        parts = {u: u.split(SEP) for u in map(str, batch[UNIT_ID])}
        style_texts: dict[str, str] = {}
        for u, (date, itype, _) in parts.items():
            modalities = self.world.types[itype]
            descriptions = self.world.store.descriptions(list(modalities))
            titles = self.world.news(date, self.world.pool(date)["Stock"].tolist()) if "news" in modalities else None
            style_texts[u] = style_prompt(self.cfg, descriptions, titles, self.world.macro(date))
        unique = sorted(set(style_texts.values()))
        styles = {}
        if unique:
            requests = [GenRequest(f"style-{i}", text, task=TASK_STYLE, system=self.system) for i, text in enumerate(unique)]
            for result, text in zip(self.engine.generate(requests, self.params), unique, strict=True):
                styles[text] = result.texts[0]
        shown_by_unit, prompts, rows = {}, {}, {}
        for u, (date, itype, agent) in parts.items():
            style, selector = parse_style(styles[style_texts[u]])
            if style is None:
                rows[u] = self._failed(u, date, itype, agent, "style")
                continue
            rng = make_rng(int(self.cfg.get("runtime.base_seed")), "mass", u)
            shown = select_pool(selector, self.world.pool(date), int(spec["shown_stocks"]), rng)
            modalities = self.world.types[itype]
            tables = self._tables(modalities, date, sorted(set(shown)))
            descriptions = self.world.store.descriptions(list(modalities))
            prompts[u] = decision_prompt(self.cfg, descriptions, tables, style, int(spec["selected_stocks"]))
            shown_by_unit[u] = (sorted(set(shown)), style, date, itype, agent)
        if prompts:
            requests = [GenRequest(u, text, task=TASK_DECISION, system=self.system) for u, text in prompts.items()]
            for result in self.engine.generate(requests, self.params):
                shown, style, date, itype, agent = shown_by_unit[result.unit_id]
                chosen = parse_decision(result.texts[0], shown)
                rows[result.unit_id] = {
                    UNIT_ID: result.unit_id, STATUS: Status.OK.value, REASON: None if chosen is not None else Reason.PARSE_FAILED.value,
                    DETAIL: None, PARSE_OK: chosen is not None, DATE: date, TYPE: itype, AGENT: int(agent),
                    SHOWN: json.dumps(shown), CHOSEN: json.dumps(chosen or []), STYLE: json.dumps(style, sort_keys=True),
                    TOKENS_IN: result.tokens_in, TOKENS_OUT: result.tokens_out,
                }
        return pd.DataFrame([rows[str(u)] for u in batch[UNIT_ID]])

    def _failed(self, u: str, date: str, itype: str, agent: str, stage: str) -> dict:
        return {UNIT_ID: u, STATUS: Status.OK.value, REASON: Reason.PARSE_FAILED.value, DETAIL: stage, PARSE_OK: False,
                DATE: date, TYPE: itype, AGENT: int(agent), SHOWN: "[]", CHOSEN: "[]", STYLE: "{}", TOKENS_IN: 0, TOKENS_OUT: 0}
