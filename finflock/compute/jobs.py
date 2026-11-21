"""Job registry: how a shard folder becomes a planned frame plus a processor (spec 2.3, 13.1).

A job builder reads only the shard folder and the config; it never touches the network. The job names
are the stages that need an LLM: ``reader``, ``bios``, ``personas``, ``basic``, ``finripple``, ``mass``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from finflock.common.config import Config
from finflock.common.schema import UNIT_ID
from finflock.compute import payloads as pl
from finflock.compute.engine import Engine
from finflock.personas.bios import BioProcessor
from finflock.personas.feed import KINDS
from finflock.personas.run import PersonaProcessor
from finflock.reader.masking import MaskerCache
from finflock.reader.prompt import Tokenizer
from finflock.reader.run import ReaderProcessor
from finflock.rivals.finripple_lite.run import FinRippleProcessor
from finflock.rivals.llt.run import LLTProcessor
from finflock.rivals.mass.run import MassProcessor


@dataclass
class JobInput:
    planned: pd.DataFrame
    processor: Callable[[pd.DataFrame], pd.DataFrame]


def _units(folder: Path) -> pd.DataFrame:
    return pd.read_parquet(folder / pl.UNITS)


def build_reader(cfg: Config, folder: Path, engine: Engine, tokenizer: Tokenizer) -> JobInput:
    kinds = _units(folder)["kind"]
    if not kinds.isin(KINDS).all():
        raise ValueError(f"{folder / pl.UNITS}: {int((~kinds.isin(KINDS)).sum())} units have a kind outside {KINDS}; fix the unit table before running")
    aliases = pd.read_parquet(folder / pl.ALIASES)
    maskers = MaskerCache(aliases, label_prefix=str(cfg.get("reader.masked_company_prefix")),
                          short_ticker_max_len=int(cfg.get("reader.short_ticker_max_len")))
    return JobInput(_units(folder), ReaderProcessor(cfg, engine, maskers, tokenizer))


def build_personas(cfg: Config, folder: Path, engine: Engine, tokenizer: Tokenizer) -> JobInput:
    processor = PersonaProcessor(cfg, engine, pl.load_skeletons(folder), pl.load_bios(folder), pl.load_events(folder),
                                 pl.load_states(folder), tokenizer)
    return JobInput(_units(folder), processor)


def build_bios(cfg: Config, folder: Path, engine: Engine, tokenizer: Tokenizer) -> JobInput:
    skeletons = pl.load_skeletons(folder)
    writer = str(cfg.get(f"llms.models.{cfg.get('llms.primary')}.name"))
    # units are ``persona|quarter``; the processor is keyed the same way
    keyed = {f"{s.persona_id}|{s.asof_quarter}": s for s in skeletons.values()}
    return JobInput(_units(folder), BioProcessor(cfg, engine, keyed, writer))


def build_basic(cfg: Config, folder: Path, engine: Engine, tokenizer: Tokenizer) -> JobInput:
    return JobInput(_units(folder), LLTProcessor(cfg, engine))


def build_finripple(cfg: Config, folder: Path, engine: Engine, tokenizer: Tokenizer) -> JobInput:
    return JobInput(_units(folder), FinRippleProcessor(cfg, engine))


def build_mass(cfg: Config, folder: Path, engine: Engine, tokenizer: Tokenizer) -> JobInput:
    return JobInput(_units(folder), MassProcessor(cfg, engine, pl.load_mass_world(folder)))


JOBS: dict[str, Callable[[Config, Path, Engine, Tokenizer], JobInput]] = {
    "reader": build_reader,
    "personas": build_personas,
    "bios": build_bios,
    "basic": build_basic,
    "finripple": build_finripple,
    "mass": build_mass,
}


def planned_for_shard(units: pd.DataFrame, unit_ids: list[str]) -> pd.DataFrame:
    """The shard's units in the shard's own fixed order."""
    order = {u: i for i, u in enumerate(unit_ids)}
    chosen = units[units[UNIT_ID].isin(order)].copy()
    chosen["_order"] = chosen[UNIT_ID].map(order)
    return chosen.sort_values("_order").drop(columns="_order").reset_index(drop=True)
