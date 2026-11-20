"""Persona panels per quarter from the 13F tables, and the inputs of the persona job for a bucket's events.

Institutional personas are clusters of real 13F filers (``personas.clusters``); retail personas come from the career grid with a
synthetic basket. Everything is a deterministic function of the tables and config. Approximations that follow from free data are
named in the code: the market-cap proxy is dollar volume (no shares outstanding), retail's uncovered share is a constant, and
sectors and industry peers come from today's SIC codes.
"""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from oryxis.common.config import Config
from oryxis.personas.careers import load_career_grid
from oryxis.personas.clusters import (
    BOOK,
    N_POSITIONS,
    PASSIVE_ADJACENT,
    TURNOVER,
    aggregate_clusters,
    align_labels,
    cluster_managers,
    features,
    fit_feature_space,
)
from oryxis.personas.feed import NewsItem
from oryxis.personas.run import EventPayload, PersonaState, unit_id
from oryxis.personas.skeletons import Skeleton, default_panel, institutional_skeletons, retail_skeletons, sector_tilt_of
from oryxis.personas.state import institution_history, portfolio_state, retail_history, window_return

MONTHS_PER_QUARTER = 3
YEAR_CHARS = 4  # a label looks like 2019Q4
INSTITUTIONAL = ("mutual_pension", "bank_dealer", "hedge_fund")


def quarter_label(period: pd.Timestamp, cfg: Config) -> str:
    return str(cfg.get("personas.panel.asof_label")).format(year=period.year, quarter=(period.month - 1) // MONTHS_PER_QUARTER + 1)


def sector_domain_map(path: Path) -> dict[str, str]:
    table = pd.read_csv(path)
    return dict(zip(table["sector"], table["domain"], strict=True))


def market_cap_proxy(px: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Rolling mean dollar volume (sessions x securities): the size proxy used where shares outstanding are missing."""
    window = int(cfg.get("personas.panel.cap_proxy_window_days"))
    close = px.pivot_table(index="session_date", columns="security_id", values="close_raw", aggfunc="last").sort_index()
    volume = px.pivot_table(index="session_date", columns="security_id", values="volume_raw", aggfunc="last").sort_index()
    return (close * volume).rolling(window, min_periods=1).mean()


def institutional_panel(
    managers: pd.DataFrame, holdings: pd.DataFrame, sector_of: pd.Series, sector_domain: dict[str, str], cfg: Config,
) -> dict[pd.Timestamp, list[Skeleton]]:
    """Institutional skeletons per period: per type, k-means clusters of active filers aligned to the previous period."""
    out: dict[pd.Timestamp, list[Skeleton]] = {}
    counts = cfg.get("personas.types")
    weights = holdings.merge(managers[["accession", "manager_cik", "period"]], on="accession")
    space, previous = None, {}
    seed, n_init = int(cfg.get("runtime.base_seed")), int(cfg.get("personas.cluster.n_init"))
    for period, part in managers[~managers["passive_flag"]].groupby("period"):
        label = quarter_label(period, cfg)
        skeletons: list[Skeleton] = []
        for persona_type in INSTITUTIONAL:
            members = part[part["type"] == persona_type].drop_duplicates("manager_cik").set_index("accession")
            if len(members) < int(cfg.get("personas.panel.min_managers_per_type")):
                continue
            held = weights[weights["accession"].isin(members.index)].pivot_table(index="accession", columns="security_id", values="weight", aggfunc="sum").fillna(0.0)
            members = members.loc[held.index].assign(**{PASSIVE_ADJACENT: 0.0})
            members[TURNOVER] = members[TURNOVER].fillna(0.0)
            tilt = held.T.groupby(sector_of.reindex(held.columns).fillna("unknown").to_numpy()).sum().T
            tilt = tilt.div(tilt.sum(axis=1).replace(0.0, np.nan), axis=0).fillna(0.0)
            space = space or fit_feature_space(members, tilt, cfg)
            labels, centroids = cluster_managers(features(space, members, tilt), int(counts[persona_type]["count"]), seed, n_init)
            if persona_type in previous:
                labels, centroids = align_labels(labels, centroids, previous[persona_type])
            previous[persona_type] = centroids
            clusters, vectors = aggregate_clusters(members, held, labels)
            skeletons += institutional_skeletons(cfg, persona_type, clusters, vectors, sector_of, sector_domain, label, 0.0)
        out[pd.Timestamp(period)] = skeletons
    return out


def full_panels(
    institutional: dict[pd.Timestamp, list[Skeleton]], universe_by_period: dict[pd.Timestamp, pd.DataFrame], sector_domain: dict[str, str],
    career_grid_path: Path, cfg: Config,
) -> dict[str, list[Skeleton]]:
    """Per quarter label: the 50 personas (institutional clusters, then retail), each with a quarter-unique id ``<persona>@<quarter>``."""
    grid = load_career_grid(str(career_grid_path))
    separator = str(cfg.get("personas.panel.key_separator"))
    seed = int(cfg.get("runtime.base_seed"))
    panels: dict[str, list[Skeleton]] = {}
    for period, skeletons in institutional.items():
        label = quarter_label(period, cfg)
        retail = retail_skeletons(cfg, grid, universe_by_period[period], sector_domain, label, seed)
        chosen = default_panel(cfg, skeletons + retail, seed)
        panels[label] = [replace(s, persona_id=f"{s.persona_id}{separator}{label}") for s in chosen]
    return panels


def panel_for_date(date: pd.Timestamp, panels: dict[str, list[Skeleton]], cfg: Config) -> str | None:
    """The newest quarter label whose filings are known on ``date`` (quarter end plus the filing lag)."""
    lag = pd.Timedelta(days=int(cfg.get("personas.panel.lag_days")))
    usable = [label for label in panels if _end_of(label) + lag < pd.Timestamp(date)]
    return max(usable, key=_end_of) if usable else None


def _end_of(label: str) -> pd.Timestamp:
    year, quarter = int(label[:YEAR_CHARS]), int(label[-1])
    return pd.Timestamp(year=year, month=quarter * MONTHS_PER_QUARTER, day=1) + pd.offsets.MonthEnd(0)


def domains_of(pins: list[dict], sector_domain: dict[str, str]) -> tuple[str, ...]:
    return tuple(sorted({sector_domain[p["sector"]] for p in pins if p.get("sector") in sector_domain}))


def event_payloads(
    events: pd.DataFrame, context_pool: pd.DataFrame, sector_of: pd.Series, sector_domain: dict[str, str], peers: dict[str, list[str]],
    members_on, reserve: pd.Series, cfg: Config,
) -> dict[str, EventPayload]:
    """One payload per event: the event as the focal item, recent events as context, pins with sectors, SIC peers, reserve members."""
    hours = pd.Timedelta(hours=float(cfg.get("personas.panel.context_pool_hours")))
    context_max = int(cfg.get("personas.context_items_max"))
    per_pin = int(cfg.get("personas.panel.links_per_pin"))
    pool = context_pool.sort_values("available_utc")
    stamps = pd.to_datetime(pool["available_utc"], utc=True)
    out: dict[str, EventPayload] = {}
    for row in events.itertuples():
        pins = json.loads(row.pins_json)
        for p in pins:
            if p["node_type"] == "company":
                p["sector"] = sector_of.get(p["node_id"])
        when = pd.Timestamp(row.available_utc)
        when = when.tz_localize("UTC") if when.tzinfo is None else when
        domains = domains_of(pins, sector_domain)
        focal = NewsItem(row.event_id, row.kind, when, str(row.title), str(row.text), domains)
        recent = pool[(stamps < when) & (stamps >= when - hours) & (pool["event_id"] != row.event_id)].tail(context_max)
        context = tuple(NewsItem(r.event_id, r.kind, pd.Timestamp(r.available_utc), str(r.title), str(r.text), domains) for r in recent.itertuples())
        links = tuple({"node_id": peer, "link_type": "industry"} for p in pins if p["node_type"] == "company" for peer in peers.get(p["node_id"], [])[:per_pin])
        members = frozenset(members_on(when.tz_convert(None).normalize()))
        descriptors = {n: {"node_type": "company", "sector": sector_of.get(n), "layer": "mid"} for n in members | {p["node_id"] for p in pins} | {k["node_id"] for k in links}
                       if n in sector_of.index}
        reserve_list = tuple(s for s in reserve.index if s in members)
        out[row.event_id] = EventPayload(row.event_id, row.kind, when, (focal,), context, tuple(pins), links, reserve_list, descriptors, members)
    return out


def persona_states(
    events: pd.DataFrame, panels: dict[str, list[Skeleton]], returns: pd.DataFrame, market_return: pd.Series, calendar, cfg: Config,
) -> tuple[dict[tuple[str, str], PersonaState], pd.DataFrame]:
    """``(states, units)``: a state per (event, persona) and the persona job's unit table (event | persona)."""
    window = int(cfg.get("personas.state_window_sessions"))
    separator = str(cfg.get("personas.panel.key_separator"))
    states, units = {}, []
    labels = sorted(panels, key=_end_of)
    for row in events.itertuples():
        day = pd.Timestamp(row.available_utc).tz_convert(None) if pd.Timestamp(row.available_utc).tzinfo else pd.Timestamp(row.available_utc)
        label = panel_for_date(day, panels, cfg)
        if label is None:
            continue
        end = calendar.snap_backward(day.normalize() - pd.Timedelta(days=1))
        wr = window_return(returns, end, window)
        market = float(market_return.loc[:end].iloc[-window:].add(1.0).prod() - 1.0)
        before = labels[labels.index(label) - 1] if labels.index(label) > 0 else None
        earlier = {s.persona_id.split(separator)[0]: s for s in panels[before]} if before else {}
        for s in panels[label]:
            st = portfolio_state(pd.Series(s.holdings), wr, market)
            base = s.persona_id.split(separator)[0]
            if s.persona_type == "retail":
                history, _ = retail_history(cfg, s.turnover_class)
            elif base in earlier:
                history, _ = institution_history(cfg, earlier[base].sector_tilt, s.sector_tilt)
            else:  # no earlier filing of this persona: nothing to compare
                history = str(cfg.get("prompts.persona.history_institution_none"))
            states[(row.event_id, s.persona_id)] = PersonaState(st["state"], st["state_rel"], market, history, dict(s.sector_tilt))
            units.append(unit_id(row.event_id, s.persona_id))
    return states, pd.DataFrame({"unit_id": units})
