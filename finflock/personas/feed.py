"""What each persona sees: access, understanding, trust, latency and feed assembly (spec Part 5).

* Access ``A[p, s]`` decides, deterministically per (persona, item), whether an item reaches the
  persona at all: ``hash(seed, persona, item) / 2^64 < A``.
* Understanding ``U = literacy value x domain match`` sets how much text the persona uses.
* Trust ``T[p, s]`` tags each item with a reliability label and orders the feed.
* Latency ``d[p, s]`` delays when the persona sees an item; later items are excluded.

Everything here is a pure function of (event, persona, config): the same inputs give the same feed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from finflock.common.config import Config
from finflock.common.hashing import stable_hash
from finflock.common.rng import derive_seed
from finflock.common.units import SECONDS_PER_HOUR
from finflock.personas.careers import literacy_value
from finflock.reader.prompt import Tokenizer

UINT64_RANGE = 2.0**64
KINDS = ("finance", "geo", "social")


@dataclass(frozen=True)
class PersonaProfile:
    """The parts of a persona skeleton that decide its information diet."""

    persona_id: str
    persona_type: str  # retail | mutual_pension | hedge_fund | bank_dealer
    literacy_tier: str  # low | medium | high
    domain: str  # energy, technology, ... or general

    @property
    def matrix_key(self) -> str:
        """Row of ``personas.source_matrices`` that applies to this persona."""
        if self.persona_type == "retail":
            return f"retail_{self.literacy_tier}"
        return self.persona_type


@dataclass(frozen=True)
class NewsItem:
    item_id: str
    kind: str
    available_utc: pd.Timestamp
    title: str
    text: str
    domains: tuple[str, ...]  # domains of the pinned sectors / commodities, may be empty


@dataclass(frozen=True)
class FeedItem:
    item_id: str
    kind: str
    title: str
    text: str
    depth: str  # headline | short | long
    tag: str
    hours_ago: float
    trust: float
    focal: bool


@dataclass(frozen=True)
class Feed:
    items: tuple[FeedItem, ...]
    feed_view_id: str
    saw_focal: bool


def matrix_value(cfg: Config, profile: PersonaProfile, table: str, kind: str) -> float:
    return float(cfg.get(f"personas.source_matrices.{profile.matrix_key}.{table}.{kind}"))


def access_granted(base_seed: int, persona_id: str, item_id: str, access: float) -> bool:
    """Deterministic draw: reaches the persona iff ``hash / 2^64 < access``."""
    return derive_seed(base_seed, persona_id, item_id) / UINT64_RANGE < access


def understanding(cfg: Config, profile: PersonaProfile, item: NewsItem) -> float:
    """``U = literacy value x domain match`` (spec 5.1)."""
    if profile.domain == str(cfg.get("personas.general_domain")):
        match = float(cfg.get("personas.domain_match.general"))
    elif profile.domain in item.domains:
        match = float(cfg.get("personas.domain_match.match"))
    else:
        match = float(cfg.get("personas.domain_match.mismatch"))
    return literacy_value(cfg, profile.literacy_tier) * match


def depth_for(cfg: Config, u: float) -> str:
    """Headline only, a short excerpt, or a long excerpt, by understanding."""
    if u < float(cfg.get("personas.depth.headline_only_below")):
        return "headline"
    if u < float(cfg.get("personas.depth.short_below")):
        return "short"
    return "long"


def reliability_tag(cfg: Config, trust: float) -> str:
    """Bin trust into the reliability labels (very high ... rumour)."""
    labels = list(cfg.get("personas.trust_labels"))
    for edge, label in zip(cfg.get("personas.trust_bins"), labels, strict=False):
        if trust >= float(edge):
            return label
    return labels[-1]


def _excerpt(cfg: Config, tokenizer: Tokenizer, item: NewsItem, depth: str) -> str:
    if depth == "headline":
        return ""
    key = "short_tokens" if depth == "short" else "long_tokens"
    text, _ = tokenizer.truncate(item.text, int(cfg.get(f"personas.depth.{key}")))
    return text


def assemble_feed(
    cfg: Config,
    profile: PersonaProfile,
    focal: Sequence[NewsItem],
    context: Sequence[NewsItem],
    event_available_utc: pd.Timestamp,
    event_kind: str,
    tokenizer: Tokenizer,
    base_seed: int,
) -> Feed:
    """Feed of one persona for one event (spec 5.3).

    The persona decides at ``event.available_utc + latency(event kind)``. A focal or context item is
    shown iff it passes access and its own latency lets the persona see it by then. If no focal item
    passes, ``saw_focal`` is false and the persona is not called.
    """
    latency = {k: matrix_value(cfg, profile, "latency_hours", k) for k in KINDS}
    decision = event_available_utc + pd.Timedelta(hours=latency[event_kind])

    def visible(item: NewsItem) -> bool:
        seen_at = item.available_utc + pd.Timedelta(hours=latency[item.kind])
        return seen_at <= decision and access_granted(
            base_seed, profile.persona_id, item.item_id, matrix_value(cfg, profile, "access", item.kind)
        )

    shown_focal = [i for i in focal if visible(i)]
    context_limit = int(cfg.get("personas.context_items_max"))
    window = pd.Timedelta(hours=float(cfg.get("personas.context_hours")))
    pool = [
        i for i in context
        if event_available_utc - window <= i.available_utc < event_available_utc and visible(i)
    ]
    trust_of = lambda i: matrix_value(cfg, profile, "trust", i.kind)  # noqa: E731
    pool.sort(key=lambda i: (-trust_of(i), -i.available_utc.value, i.item_id))
    chosen = [(i, True) for i in shown_focal] + [(i, False) for i in pool[:context_limit]]
    chosen.sort(key=lambda pair: (-trust_of(pair[0]), -pair[0].available_utc.value, pair[0].item_id))

    items = []
    for item, is_focal in chosen:
        trust = trust_of(item)
        depth = depth_for(cfg, understanding(cfg, profile, item))
        items.append(
            FeedItem(
                item.item_id, item.kind, item.title, _excerpt(cfg, tokenizer, item, depth), depth,
                reliability_tag(cfg, trust),
                (event_available_utc - item.available_utc).total_seconds() / SECONDS_PER_HOUR,
                trust, is_focal,
            )
        )
    view = stable_hash([(i.item_id, i.depth) for i in items])
    return Feed(tuple(items), view, bool(shown_focal))
