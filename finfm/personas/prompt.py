"""Persona prompt (spec 6.6, 6.9): feed, assets shown by literacy, own state, question, strict output.

The model only reports emotion and valuation numbers; it never outputs a trade, size or speed.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from finfm.common.config import Config
from finfm.personas.feed import Feed
from finfm.personas.skeletons import Skeleton
from finfm.personas.state import (
    exposure_sentence,
    mandate_sentence,
    state_sentence,
)

NODE_COMPANY = "company"
RESERVED = "reserved"


@dataclass(frozen=True)
class ShownAsset:
    ref: str
    node_id: str
    node_type: str
    sector: str | None
    layer: str | None
    link_type: str | None = None
    market_cap_rank_share: float | None = None  # 0 = largest, used for the salience mapping


@dataclass(frozen=True)
class PersonaPrompt:
    system: str
    user: str
    ref_to_node: dict[str, str]


def select_assets(
    cfg: Config,
    skeleton: Skeleton,
    pins: Sequence[Mapping],
    links: Sequence[Mapping],
    sp_members: set[str],
    reserve_candidates: Sequence[str],
    descriptors: Mapping[str, Mapping],
) -> list[ShownAsset]:
    """Assets shown to a persona (spec 6.6 and 6.9 rev. final).

    Low literacy: only assets named in the item. Medium: plus country and commodity nodes. High: plus
    up to ``high_literacy_linked_max`` of the most strongly linked assets, labelled with the link
    type. When fewer than ``reserved_sp_trigger_below`` pins are S&P 500 members, up to
    ``reserved_sp_slots`` of the slots go to S&P members chosen from the link snapshot (the caller
    passes them in ``reserve_candidates``, ordered by link weight then market cap).
    """
    limit = int(cfg.get("personas.pins_shown_max"))
    tier = skeleton.literacy_tier
    base: list[tuple[str, str | None]] = []
    for pin in pins:
        if tier == "low" and str(pin["node_type"]) != NODE_COMPANY:
            continue
        base.append((str(pin["node_id"]), None))
    if tier == "high":
        for link in links[: int(cfg.get("personas.high_literacy_linked_max"))]:
            base.append((str(link["node_id"]), str(link["link_type"])))
    names = {node for node, _ in base}
    reserved: list[tuple[str, str | None]] = []
    if sum(1 for node in names if node in sp_members) < int(cfg.get("personas.reserved_sp_trigger_below")):
        wanted = [c for c in reserve_candidates if c not in names]
        reserved = [(c, RESERVED) for c in wanted[: int(cfg.get("personas.reserved_sp_slots"))]]
    ordered = base[: max(limit - len(reserved), 0)] + reserved
    assets: list[ShownAsset] = []
    seen: set[str] = set()
    for node, link_type in ordered:
        if node in seen:
            continue
        seen.add(node)
        info = descriptors.get(node, {})
        assets.append(
            ShownAsset(
                f"{cfg.get('reader.ref_prefix')}{len(assets) + 1}", node,
                str(info.get("node_type", NODE_COMPANY)), info.get("sector"), info.get("layer"),
                link_type, info.get("market_cap_rank_share"),
            )
        )
    return assets


def asset_line(cfg: Config, tier: str, asset: ShownAsset, weight: float) -> str:
    """One asset line; descriptor depth follows literacy (salience words for low literacy)."""
    sector = asset.sector or "unknown"
    if tier == "low":
        if asset.node_type != NODE_COMPANY:
            kind = asset.node_type
            return f"{asset.ref}: " + str(cfg.get("prompts.persona.asset_salience.etf")).format(kind=kind)
        well_known = (asset.market_cap_rank_share or 1.0) <= float(cfg.get("personas.well_known_top_share"))
        key = "well_known" if well_known else "obscure"
        return f"{asset.ref}: " + str(cfg.get(f"prompts.persona.asset_salience.{key}")).format(sector=sector)
    if tier == "medium" or asset.node_type != NODE_COMPANY:
        return f"{asset.ref}: {asset.node_type}, sector={sector}, size={asset.layer or 'n/a'}"
    link = f", {asset.link_type} link" if asset.link_type and asset.link_type != RESERVED else ""
    return f"{asset.ref}: {asset.node_type}, size={asset.layer}, sector={sector}{link}, your weight {weight:.1%}"


def feed_text(cfg: Config, feed: Feed) -> str:
    template = str(cfg.get("prompts.persona.feed_item"))
    lines = []
    for item in feed.items:
        excerpt = f" — {item.text}" if item.text else ""
        lines.append(template.format(kind=item.kind, tag=item.tag, hours_ago=item.hours_ago,
                                     title=item.title, excerpt=excerpt))
    return "\n".join(lines)


def build_persona_prompt(
    cfg: Config,
    skeleton: Skeleton,
    bio: str,
    feed: Feed,
    assets: Sequence[ShownAsset],
    state: Mapping[str, float],
    market_return: float,
    history: str,
    sector_weights: Mapping[str, float],
) -> PersonaPrompt:
    """Assemble the persona prompt in cache-friendly order: feed, assets, then the persona parts."""
    first = assets[0] if assets else None
    own = skeleton.holdings
    exposure = ""
    if first is not None:
        exposure = exposure_sentence(
            cfg, first.sector or "unknown", sector_weights.get(first.sector or "", 0.0),
            first.ref, own.get(first.node_id, 0.0),
        )
    career_line = (
        str(cfg.get("prompts.persona.career_line")).format(career=skeleton.career)
        if skeleton.career else ""
    )
    you_parts = [bio, career_line, mandate_sentence(cfg, skeleton.mandate), exposure,
                 state_sentence(cfg, state["state"], market_return), history]
    json_format = str(cfg.get("prompts.persona.json_format")).format()
    user = str(cfg.get("prompts.persona.user")).format(
        feed=feed_text(cfg, feed),
        assets="\n".join(asset_line(cfg, skeleton.literacy_tier, a, own.get(a.node_id, 0.0)) for a in assets),
        you="\n".join(p for p in you_parts if p),
        json_format=json_format,
    )
    return PersonaPrompt(
        str(cfg.get("prompts.persona.system")), user, {a.ref: a.node_id for a in assets}
    )


# --- output ----------------------------------------------------------------------------------

FEAR, ANGER, CALM, CONVICTION, VALUE = "fear", "anger", "calm", "conviction", "value_change_pct"


@dataclass(frozen=True)
class PersonaAnswer:
    ok: bool
    fear: float
    anger: float
    calm: float
    conviction: float
    values: dict[str, float]
    clamped: bool


def _num(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value) if value == value and abs(value) != float("inf") else None


def parse_answer(text: str, refs: Mapping[str, str], cfg: Config) -> PersonaAnswer:
    """Accept only the five numbers; clamp out-of-range values and flag it (spec 12.2 #6)."""
    scale_max = float(cfg.get("personas.answer.scale_max"))
    value_bound = float(cfg.get("personas.answer.value_bound"))
    empty = PersonaAnswer(False, 0.0, 0.0, 0.0, 0.0, {}, False)
    try:
        data = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return empty
    if not isinstance(data, dict):
        return empty
    scalars = [_num(data.get(k)) for k in (FEAR, ANGER, CALM, CONVICTION)]
    raw_values = data.get(VALUE)
    if any(v is None for v in scalars) or not isinstance(raw_values, dict):
        return empty
    clamped = False
    clipped = []
    for v in scalars:
        c = min(max(v, 0.0), scale_max)
        clamped |= c != v
        clipped.append(c)
    values: dict[str, float] = {}
    for ref, v in raw_values.items():
        number = _num(v)
        if ref not in refs or number is None:
            continue
        c = min(max(number, -value_bound), value_bound)
        clamped |= c != number
        values[ref] = c
    return PersonaAnswer(True, *clipped, values, clamped)

