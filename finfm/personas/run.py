"""The persona job: one request (three draws) per (event, persona), frozen as the persona tape.

A persona's state is exogenous to the market's reaction to the event (holdings times past returns),
so answers for different events are independent given the inputs and the tape is a pure function of
(event, persona skeleton as of that date, LLM, prompt template, draw seed). No price computation
happens on the GPU: states and history sentences are precomputed (``PersonaState``).
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

import pandas as pd

from finfm.common.config import Config
from finfm.common.hashing import sha256_hex
from finfm.common.rng import derive_seed
from finfm.common.schema import (
    DETAIL,
    NAN_DETECTED,
    PARSE_OK,
    REASON,
    STATUS,
    TOKENS_IN,
    TOKENS_OUT,
    UNIT_ID,
    Reason,
    Status,
)
from finfm.compute.engine import Engine, GenParams, GenRequest
from finfm.personas.feed import NewsItem, PersonaProfile, assemble_feed
from finfm.personas.prompt import ShownAsset, build_persona_prompt, parse_answer, select_assets
from finfm.personas.skeletons import Skeleton
from finfm.reader.prompt import Tokenizer

TASK = "persona"
UNIT_SEPARATOR = "|"
EVENT_ID = "event_id"
PERSONA_ID = "persona_id"
SAW_FOCAL = "saw_focal"
FEED_VIEW_ID = "feed_view_id"
DRAWS_JSON = "draws_json"
N_DRAWS_OK = "n_draws_ok"
BIO_ID = "bio_id"
TEMPLATE_HASH = "template_hash"
N_ASSETS = "n_assets_shown"


def unit_id(event_id: str, persona_id: str) -> str:
    return f"{event_id}{UNIT_SEPARATOR}{persona_id}"


@dataclass(frozen=True)
class EventPayload:
    """Everything about an event the personas need (built by the dataset layer, not by the GPU job)."""

    event_id: str
    kind: str
    available_utc: pd.Timestamp
    focal: tuple[NewsItem, ...]
    context: tuple[NewsItem, ...]
    pins: tuple[Mapping, ...]
    links: tuple[Mapping, ...]
    reserve_candidates: tuple[str, ...]
    descriptors: Mapping[str, Mapping]
    sp_members: frozenset[str]


@dataclass(frozen=True)
class PersonaState:
    """Precomputed per (event, persona): state, history sentence and sector weights."""

    state: float
    state_rel: float
    market_return: float
    history: str
    sector_weights: Mapping[str, float] = field(default_factory=dict)


class PersonaProcessor:
    """Worker processor: batch of ``event|persona`` units in, one row per unit out."""

    def __init__(
        self,
        cfg: Config,
        engine: Engine,
        skeletons: Mapping[str, Skeleton],
        bios: Mapping[str, str],
        events: Mapping[str, EventPayload],
        states: Mapping[tuple[str, str], PersonaState],
        tokenizer: Tokenizer,
    ) -> None:
        self.cfg, self.engine = cfg, engine
        self.skeletons, self.bios, self.events, self.states = skeletons, bios, events, states
        self.tokenizer = tokenizer
        sampling = cfg.get("personas.sampling")
        self.params = GenParams(
            temperature=float(sampling["temperature"]), top_p=float(sampling["top_p"]),
            top_k=int(sampling["top_k"]), max_tokens=int(sampling["max_tokens"]),
            seed=int(cfg.get("runtime.base_seed")),
        )
        self.draws = int(cfg.get("personas.draws"))
        self.base_seed = int(cfg.get("runtime.base_seed"))

    def _prepare(self, event_id: str, persona_id: str):
        event, skeleton = self.events[event_id], self.skeletons[persona_id]
        profile = PersonaProfile(persona_id, skeleton.persona_type, skeleton.literacy_tier, skeleton.domain)
        feed = assemble_feed(
            self.cfg, profile, event.focal, event.context, event.available_utc, event.kind,
            self.tokenizer, self.base_seed,
        )
        if not feed.saw_focal:
            return feed, None, None
        assets = select_assets(
            self.cfg, skeleton, event.pins, event.links, set(event.sp_members),
            event.reserve_candidates, event.descriptors,
        )
        if not assets:
            return feed, assets, None
        state = self.states[(event_id, persona_id)]
        prompt = build_persona_prompt(
            self.cfg, skeleton, self.bios[skeleton.bio_id], feed, assets,
            {"state": state.state}, state.market_return, state.history, state.sector_weights,
        )
        return feed, assets, prompt

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        rows: dict[str, dict] = {}
        prompts, requests = {}, []
        for _, record in batch.iterrows():
            unit = str(record[UNIT_ID])
            event_id, persona_id = unit.split(UNIT_SEPARATOR, 1)
            feed, assets, prompt = self._prepare(event_id, persona_id)
            base = {UNIT_ID: unit, EVENT_ID: event_id, PERSONA_ID: persona_id,
                    SAW_FOCAL: feed.saw_focal, FEED_VIEW_ID: feed.feed_view_id}
            if not feed.saw_focal:  # not called; downstream deviation is 0
                rows[unit] = {**base, STATUS: Status.SKIPPED.value, REASON: Reason.NO_FOCAL_ACCESS.value}
            elif prompt is None:
                rows[unit] = {**base, STATUS: Status.SKIPPED.value, REASON: Reason.NO_USABLE_PIN.value}
            else:
                prompts[unit] = (base, assets, prompt)
                requests.append(GenRequest(
                    unit, prompt.user, n=self.draws, task=TASK, system=prompt.system,
                    seed=derive_seed(self.base_seed, TASK, unit),
                ))
        for result in self.engine.generate(requests, self.params) if requests else []:
            base, assets, prompt = prompts[result.unit_id]
            rows[result.unit_id] = self._row(base, assets, prompt, result)
        return pd.DataFrame([rows[str(u)] for u in batch[UNIT_ID]])

    def _row(self, base: dict, assets: Sequence[ShownAsset], prompt, result) -> dict:
        draws = []
        for text in result.texts:
            answer = parse_answer(text, prompt.ref_to_node, self.cfg)
            draws.append({
                "ok": answer.ok, "fear": answer.fear, "anger": answer.anger, "calm": answer.calm,
                "conviction": answer.conviction, "clamped": answer.clamped,
                "values": {prompt.ref_to_node[r]: v for r, v in answer.values.items()},
            })
        n_ok = sum(d["ok"] for d in draws)
        return {
            **base, STATUS: Status.OK.value, REASON: None if n_ok else Reason.PARSE_FAILED.value, DETAIL: None,
            PARSE_OK: n_ok > 0, N_DRAWS_OK: n_ok, DRAWS_JSON: json.dumps(draws, sort_keys=True),
            N_ASSETS: len(assets), TOKENS_IN: result.tokens_in, TOKENS_OUT: result.tokens_out,
            NAN_DETECTED: result.nan_suspect,
            TEMPLATE_HASH: sha256_hex(prompt.system),
        }
