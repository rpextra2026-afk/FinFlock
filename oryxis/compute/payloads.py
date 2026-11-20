"""Serialisation of job inputs: what the orchestrator stages into a shard folder and a worker reads back.

Every job reads plain parquet / json files from one directory (``inputs/<job>/<shard>/``), so a Kaggle
session needs no database and no network. Writers and readers live side by side so a round trip is
tested. Nothing here holds a label.
"""

from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from oryxis.personas.feed import KINDS, NewsItem
from oryxis.personas.run import EventPayload, PersonaState
from oryxis.personas.skeletons import Skeleton
from oryxis.rivals.mass.modalities import ModalityStore
from oryxis.rivals.mass.run import MassWorld

UNITS = "units.parquet"
ALIASES = "aliases.parquet"
EVENTS = "events.parquet"
SKELETONS = "skeletons.json"
BIOS = "bios.parquet"
STATES = "states.parquet"
MASS_POOL = "mass_pool.parquet"
MASS_TYPES = "mass_types.json"
MASS_MACRO = "mass_macro.json"
MASS_NEWS = "mass_news.json"
MASS_MODALITY = "mass_modality_{name}.parquet"
SHOWN_BAD_EVENTS = 3  # how many offending event ids the error message names


def _write(frame: pd.DataFrame, path: Path) -> None:
    frame.to_parquet(path, index=False)


# --- personas ---------------------------------------------------------------------------------


def _item_dict(item: NewsItem) -> dict:
    return {**asdict(item), "available_utc": item.available_utc.isoformat(), "domains": list(item.domains)}


def _item(d: dict) -> NewsItem:
    return NewsItem(d["item_id"], d["kind"], pd.Timestamp(d["available_utc"]), d["title"], d["text"], tuple(d["domains"]))


def save_events(payloads: dict[str, EventPayload], folder: Path) -> None:
    rows = [{
        "event_id": e.event_id, "kind": e.kind, "available_utc": e.available_utc.isoformat(),
        "focal": json.dumps([_item_dict(i) for i in e.focal]), "context": json.dumps([_item_dict(i) for i in e.context]),
        "pins": json.dumps(list(e.pins)), "links": json.dumps(list(e.links)),
        "reserve": json.dumps(list(e.reserve_candidates)), "descriptors": json.dumps(e.descriptors),
        "sp_members": json.dumps(sorted(e.sp_members)),
    } for e in payloads.values()]
    _write(pd.DataFrame(rows), folder / EVENTS)


def check_kinds(frame: pd.DataFrame, where: str) -> None:
    """Refuse a table whose events or news items have a kind the feed cannot look up (it would fail unit by unit, deep inside a GPU session)."""
    bad = {str(r.event_id) for r in frame.itertuples() if r.kind not in KINDS}
    bad |= {str(r.event_id) for r in frame.itertuples() for col in ("focal", "context") for d in json.loads(getattr(r, col)) if d["kind"] not in KINDS}
    if bad:
        raise ValueError(f"{where}: {len(bad)} events have an event or news-item kind outside {KINDS} (first: {sorted(bad)[:SHOWN_BAD_EVENTS]}); fix the kind in the unit table before running")


def load_events(folder: Path) -> dict[str, EventPayload]:
    out = {}
    frame = pd.read_parquet(folder / EVENTS)
    check_kinds(frame, str(folder / EVENTS))
    for r in frame.itertuples():
        out[r.event_id] = EventPayload(
            r.event_id, r.kind, pd.Timestamp(r.available_utc), tuple(_item(d) for d in json.loads(r.focal)),
            tuple(_item(d) for d in json.loads(r.context)), tuple(json.loads(r.pins)), tuple(json.loads(r.links)),
            tuple(json.loads(r.reserve)), json.loads(r.descriptors), frozenset(json.loads(r.sp_members)))
    return out


def save_skeletons(skeletons: list[Skeleton], folder: Path) -> None:
    (folder / SKELETONS).write_text(json.dumps([asdict(s) for s in skeletons], sort_keys=True), encoding="utf-8")


def load_skeletons(folder: Path) -> dict[str, Skeleton]:
    out = {}
    for d in json.loads((folder / SKELETONS).read_text(encoding="utf-8")):
        out[d["persona_id"]] = Skeleton(**d)
    return out


def save_bios(bios: dict[str, str], folder: Path) -> None:
    _write(pd.DataFrame({"bio_id": list(bios), "text": list(bios.values())}), folder / BIOS)


def load_bios(folder: Path) -> dict[str, str]:
    frame = pd.read_parquet(folder / BIOS)
    return dict(zip(frame["bio_id"], frame["text"], strict=True))


def save_states(states: dict[tuple[str, str], PersonaState], folder: Path) -> None:
    rows = [{"event_id": e, "persona_id": p, "state": s.state, "state_rel": s.state_rel, "market_return": s.market_return,
             "history": s.history, "sector_weights": json.dumps(dict(s.sector_weights))} for (e, p), s in states.items()]
    _write(pd.DataFrame(rows), folder / STATES)


def load_states(folder: Path) -> dict[tuple[str, str], PersonaState]:
    return {(r.event_id, r.persona_id): PersonaState(r.state, r.state_rel, r.market_return, r.history, json.loads(r.sector_weights))
            for r in pd.read_parquet(folder / STATES).itertuples()}


# --- MASS ------------------------------------------------------------------------------------------


def save_mass_world(pool: pd.DataFrame, store: ModalityStore, types: dict[str, list[str]], macro: dict[str, str | None],
                    news: dict[str, list[str]], folder: Path) -> None:
    _write(pool, folder / MASS_POOL)
    (folder / MASS_TYPES).write_text(json.dumps(types, sort_keys=True), encoding="utf-8")
    (folder / MASS_MACRO).write_text(json.dumps(macro, sort_keys=True), encoding="utf-8")
    (folder / MASS_NEWS).write_text(json.dumps(news, sort_keys=True), encoding="utf-8")
    for name, frame in store.frames.items():
        _write(frame, folder / MASS_MODALITY.format(name=name))


def load_mass_world(folder: Path) -> MassWorld:
    pool = pd.read_parquet(folder / MASS_POOL)
    frames = {p.name[len("mass_modality_"):-len(".parquet")]: pd.read_parquet(p) for p in sorted(folder.glob("mass_modality_*.parquet"))}
    macro = json.loads((folder / MASS_MACRO).read_text(encoding="utf-8"))
    news = json.loads((folder / MASS_NEWS).read_text(encoding="utf-8"))
    types = json.loads((folder / MASS_TYPES).read_text(encoding="utf-8"))
    return MassWorld(
        pool=lambda date: pool[pool["Date"] == pd.Timestamp(date)].reset_index(drop=True), store=ModalityStore(frames), types=types,
        macro=lambda date: macro.get(date), news=lambda date, stocks: news.get(date),
    )
