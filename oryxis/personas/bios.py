"""Persona bios: written once per persona and quarter, cached, shared by all LLMs (spec 6.8).

The bio writer sees only the skeleton facts as of the quarter: no fund names, no events after the
as-of date. A deterministic template bio (no LLM) is the fallback and an ablation arm.
"""

from __future__ import annotations

import json
from collections.abc import Mapping

import pandas as pd

from oryxis.common.config import Config
from oryxis.common.hashing import sha256_hex, stable_hash
from oryxis.common.schema import DETAIL, PARSE_OK, REASON, STATUS, TOKENS_IN, TOKENS_OUT, UNIT_ID, Status
from oryxis.compute.engine import Engine, GenParams, GenRequest
from oryxis.personas.skeletons import Skeleton
from oryxis.personas.state import mandate_sentence

BIO_TASK = "bio"
BIO_TEXT = "bio_text"
BIO_KEY = "bio_key"
TOP_SECTORS = 2


def trait_words(cfg: Config, trait_b: float) -> str:
    key = "positive" if trait_b > 0 else "negative" if trait_b < 0 else "neutral"
    return str(cfg.get(f"prompts.bio.trait_words.{key}"))


def skeleton_facts(cfg: Config, skeleton: Skeleton) -> dict[str, str]:
    """The only facts the bio writer may use, as words (no names, no dates beyond the quarter)."""
    top = sorted(skeleton.sector_tilt, key=lambda s: (-skeleton.sector_tilt[s], s))[:TOP_SECTORS]
    return {
        "type": skeleton.persona_type.replace("_", " "),
        "size": skeleton.size_stratum,
        "turnover": skeleton.turnover_class,
        "positions": str(int(round(skeleton.n_positions))),
        "sectors": " and ".join(top) if top else "a broad mix of sectors",
        "career": f"You work as {skeleton.career}." if skeleton.career else "",
        "trait": trait_words(cfg, skeleton.trait_b),
        "literacy": str(cfg.get(f"prompts.bio.literacy_words.{skeleton.literacy_tier}")),
        "mandate": mandate_sentence(cfg, skeleton.mandate),
    }


def template_bio(cfg: Config, skeleton: Skeleton) -> str:
    """Deterministic bio from the facts; used as the ``bio_template`` ablation and as fallback."""
    facts = skeleton_facts(cfg, skeleton)
    return " ".join(str(cfg.get("prompts.bio.template")).format(
        size=facts["size"], kind=facts["type"], turnover=facts["turnover"], positions=facts["positions"],
        sectors=facts["sectors"], career=facts["career"], trait=facts["trait"].capitalize(),
        literacy=facts["literacy"], mandate=facts["mandate"],
    ).split())


def bio_prompt(cfg: Config, skeleton: Skeleton) -> str:
    """The verbatim instruction followed by the facts."""
    instruction = str(cfg.get("prompts.bio.instruction")).format(asof_date=skeleton.asof_quarter)
    return instruction + "\n" + json.dumps(skeleton_facts(cfg, skeleton), sort_keys=True)


def bio_cache_key(cfg: Config, skeleton: Skeleton, writer_model: str) -> str:
    """Cache key ``(persona, quarter, skeleton hash, writer model, template hash)`` (spec 6.8)."""
    template_hash = sha256_hex(str(cfg.get("prompts.bio.instruction")))
    return stable_hash([skeleton.persona_id, skeleton.asof_quarter, skeleton.bio_id, writer_model, template_hash])


class BioProcessor:
    """Worker processor writing bios; units are ``persona|quarter`` and rows carry the cache key."""

    def __init__(self, cfg: Config, engine: Engine, skeletons: Mapping[str, Skeleton], writer_model: str):
        self.cfg, self.engine, self.skeletons, self.writer_model = cfg, engine, skeletons, writer_model
        self.params = GenParams(
            temperature=float(cfg.get("prompts.bio.temperature")), top_p=1.0, top_k=None,
            max_tokens=int(cfg.get("prompts.bio.max_tokens")), seed=int(cfg.get("runtime.base_seed")),
        )

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        requests = [
            GenRequest(str(u), bio_prompt(self.cfg, self.skeletons[str(u)]), n=1, task=BIO_TASK)
            for u in batch[UNIT_ID]
        ]
        rows = []
        for result in self.engine.generate(requests, self.params):
            skeleton = self.skeletons[result.unit_id]
            text = " ".join(result.texts[0].split())
            rows.append({
                UNIT_ID: result.unit_id, STATUS: Status.OK.value, REASON: None, DETAIL: None,
                PARSE_OK: bool(text), BIO_TEXT: text, BIO_KEY: bio_cache_key(self.cfg, skeleton, self.writer_model),
                TOKENS_IN: result.tokens_in, TOKENS_OUT: result.tokens_out,
            })
        return pd.DataFrame(rows)
