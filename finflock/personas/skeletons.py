"""Persona skeletons: every field except the bio is produced by code from data and config (spec 6.3).

Institutions come from clusters of real 13F filers (``clusters.py``); retail personas come from the
career grid with a synthetic basket (no free data identify real retail holdings; the paper states
this). The LLM only narrates a skeleton into a short bio, once, cached.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from finflock.common.binning import rank_bins
from finflock.common.config import Config
from finflock.common.hashing import stable_hash
from finflock.common.rng import make_rng
from finflock.personas.careers import Career, literacy_value, retail_career_sequence
from finflock.personas.clusters import BOOK, N_POSITIONS, PASSIVE_ADJACENT, TURNOVER

RETAIL = "retail"


@dataclass(frozen=True)
class Skeleton:
    persona_id: str
    persona_type: str
    asof_quarter: str
    size_stratum: str
    book_usd: float
    turnover: float
    turnover_class: str
    n_positions: float
    sector_tilt: Mapping[str, float]
    holdings: Mapping[str, float]  # security -> weight inside the S&P 500 sleeve, sums to 1
    career: str | None
    literacy_tier: str
    domain: str
    mandate: float
    trait_b: float
    speed_class: str
    locale: str
    history_source: str

    @property
    def bio_id(self) -> str:
        """Hash of every skeleton field the bio may depend on (the cache key of the bio)."""
        fields = asdict(self)
        fields.pop("holdings")
        return stable_hash(fields)


def sector_tilt_of(holdings: pd.Series, sector_of: pd.Series) -> dict[str, float]:
    """Weight in each sector, normalised to sum to 1."""
    by_sector = holdings.groupby(sector_of.reindex(holdings.index)).sum()
    total = by_sector.sum()
    return {str(k): float(v / total) for k, v in by_sector.items()} if total > 0 else {}


def domain_of(tilt: Mapping[str, float], sector_domain: Mapping[str, str], general: str) -> str:
    """Domain of the sector with the largest weight (``general`` if none maps)."""
    if not tilt:
        return general
    top = max(sorted(tilt), key=lambda s: tilt[s])
    return sector_domain.get(top, general)


def _terciles(values: pd.Series, labels: Sequence[str]) -> pd.Series:
    return rank_bins(values, len(labels)).map(dict(enumerate(labels)))


def institutional_skeletons(
    cfg: Config,
    persona_type: str,
    clusters: pd.DataFrame,
    holdings: pd.DataFrame,
    sector_of: pd.Series,
    sector_domain: Mapping[str, str],
    asof_quarter: str,
    trait_b: float,
) -> list[Skeleton]:
    """Skeletons of one institutional type from aggregated clusters and their holdings."""
    index_like = float(cfg.get("personas.index_like.passive_share"))
    sizes = _terciles(clusters[BOOK], cfg.get("personas.cluster.size_terciles"))
    speeds = _terciles(clusters[TURNOVER], cfg.get("personas.cluster.speed_terciles"))
    out = []
    for cluster, row in clusters.iterrows():
        weights = holdings.loc[cluster]
        weights = weights[weights > 0]
        tilt = sector_tilt_of(weights, sector_of)
        passive = float(row.get(PASSIVE_ADJACENT, 0.0))
        mandate = (
            float(cfg.get("personas.index_like.mandate"))
            if passive > index_like
            else float(cfg.get(f"game.types.{persona_type}.mandate"))
        )
        out.append(
            Skeleton(
                persona_id=f"{persona_type}_{int(cluster):02d}",
                persona_type=persona_type,
                asof_quarter=asof_quarter,
                size_stratum=str(sizes[cluster]),
                book_usd=float(row[BOOK]),
                turnover=float(row[TURNOVER]),
                turnover_class=str(speeds[cluster]),
                n_positions=float(row[N_POSITIONS]),
                sector_tilt=tilt,
                holdings={str(k): float(v / weights.sum()) for k, v in weights.items()},
                career=None,
                literacy_tier=str(cfg.get("personas.institutional_literacy")),
                domain=domain_of(tilt, sector_domain, str(cfg.get("personas.general_domain"))),
                mandate=mandate,
                trait_b=trait_b,
                speed_class=str(speeds[cluster]),
                locale=str(cfg.get("personas.locale")),
                history_source="13f",
            )
        )
    return out


def retail_basket(
    cfg: Config, universe: pd.DataFrame, domain_sectors: set[str], seed_parts: tuple
) -> pd.Series:
    """Synthetic basket: part of the weight in the persona's domain sectors, the rest broad.

    Selection probability and weight are proportional to ``uncovered share x market cap`` among
    S&P 500 members; draws are seeded by the persona and quarter, so the basket is reproducible.
    """
    size = int(cfg.get("personas.retail.basket_size"))
    domain_share = float(cfg.get("personas.retail.domain_share"))
    rng = make_rng(*seed_parts)
    score = (universe["uncovered_share"] * universe["market_cap"]).clip(lower=0.0)
    pool = universe.assign(score=score).set_index("security_id")
    in_domain = pool[pool["sector"].isin(domain_sectors)]
    if in_domain.empty:  # general persona or no member in the domain: the whole basket is broad
        domain_share, n_domain = 0.0, 0
    else:
        n_domain = min(int(round(size * domain_share)), len(in_domain))
    chosen: dict[str, float] = {}
    groups = [(in_domain, n_domain, domain_share), (pool.drop(in_domain.index), size - n_domain, 1.0 - domain_share)]
    for frame, count, share in groups:
        if count <= 0 or frame.empty:
            continue
        probs = frame["score"] / frame["score"].sum()
        picks = rng.choice(frame.index.to_numpy(), size=min(count, len(frame)), replace=False, p=probs.to_numpy())
        raw = frame.loc[picks, "score"]
        for name, value in (raw / raw.sum() * share).items():
            chosen[str(name)] = float(value)
    series = pd.Series(chosen)
    return series / series.sum()


def retail_skeletons(
    cfg: Config,
    grid: Sequence[Career],
    universe: pd.DataFrame,
    sector_domain: Mapping[str, str],
    asof_quarter: str,
    seed: int,
) -> list[Skeleton]:
    """The retail personas of a quarter: careers from the grid, a budget, a turnover and a basket."""
    careers = retail_career_sequence(list(grid), cfg)
    budgets = list(cfg.get("personas.retail.budget_buckets_usd"))
    turnovers = list(cfg.get("personas.retail.turnover_grid"))
    out = []
    for i, career in enumerate(careers):
        persona_id = f"{RETAIL}_{i:02d}"
        rng = make_rng(seed, persona_id, asof_quarter, "traits")
        domain_sectors = {s for s, d in sector_domain.items() if d == career.domain}
        basket = retail_basket(cfg, universe, domain_sectors, (seed, persona_id, asof_quarter, "basket"))
        sector_of = universe.set_index("security_id")["sector"]
        tilt = sector_tilt_of(basket, sector_of)
        turnover_class = str(turnovers[int(rng.integers(len(turnovers)))])
        out.append(
            Skeleton(
                persona_id=persona_id,
                persona_type=RETAIL,
                asof_quarter=asof_quarter,
                size_stratum="retail",
                book_usd=float(budgets[int(rng.integers(len(budgets)))]),
                turnover=float(turnovers.index(turnover_class)),
                turnover_class=turnover_class,
                n_positions=float(len(basket)),
                sector_tilt=tilt,
                holdings={str(k): float(v) for k, v in basket.items()},
                career=career.name,
                literacy_tier=career.literacy,
                domain=career.domain,
                mandate=float(cfg.get("personas.retail.mandate")),
                trait_b=float(cfg.get("personas.retail.trait_b")),
                speed_class=turnover_class,
                locale=str(cfg.get("personas.locale")),
                history_source="synthetic",
            )
        )
    return out


def default_panel(cfg: Config, skeletons: Sequence[Skeleton], seed: int) -> list[Skeleton]:
    """The first ``count`` personas of each type by a hashed order (type proportions held)."""
    panel: list[Skeleton] = []
    for persona_type, spec in cfg.get("personas.types").items():
        members = [s for s in skeletons if s.persona_type == persona_type]
        members.sort(key=lambda s: stable_hash([seed, s.persona_id]))
        panel.extend(members[: int(spec["count"])])
    return panel


def literacy_for(cfg: Config, skeleton: Skeleton) -> float:
    return literacy_value(cfg, skeleton.literacy_tier)
