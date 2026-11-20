"""A small SYNTHETIC world for CPU integration tests (spec 12.2 #17). Never used for results.

Everything it returns is marked ``synthetic``. News carries a planted effect: tone words in the text move
the named stock for a few days after day 0, and a quarter of that spills to its sector peers, so the
whole pipeline has something real to find. Names, prices, funds and links are all generated.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import numpy as np
import pandas as pd

from finfm.common.config import Config
from finfm.common.timing import TradingCalendar
from finfm.mfg.precompute import FundInput

SYNTHETIC = True
POSITIVE_TEXT = {1: "beat estimates", 2: "beat estimates and raises guidance"}
NEGATIVE_TEXT = {1: "miss estimates", 2: "miss estimates and cuts outlook"}
EFFECT_PER_TONE = 0.004
SPILLOVER = 0.25
DAY0_MULTIPLE = 3.0  # an announcement jump is larger than each later day of drift
REACTION_DAYS = 3
PEER_SECTORS = ("Energy", "Information Technology", "Financials")


@dataclass
class World:
    synthetic: bool
    cfg: Config
    calendar: TradingCalendar
    sessions: pd.DatetimeIndex
    securities: list[str]
    sector: pd.Series
    names: dict[str, str]
    close: pd.DataFrame
    open_: pd.DataFrame
    high: pd.DataFrame
    low: pd.DataFrame
    volume: pd.DataFrame
    ret: pd.DataFrame
    oo_ret: pd.DataFrame
    sleeve_ret: pd.Series
    factors: pd.DataFrame
    snapshots: pd.DataFrame
    links: pd.DataFrame
    events: pd.DataFrame
    aliases: pd.DataFrame
    funds: list[FundInput]
    uncovered: pd.Series
    tone: dict[str, int]
    event_asset: dict[str, str]


def synthetic_config(cfg: Config) -> Config:
    """Test-sized windows and shorter estimation windows; everything else stays at the real defaults."""
    return cfg.with_overrides({
        "windows.buckets.tessera": {"dev_start": "2019-03-01", "dev_end": "2019-12-31",
                                "test_start": "2020-01-14", "test_end": "2020-08-31"},
        "labels.beta_window_sessions": 120, "labels.min_beta_obs": 60, "labels.baseline_sessions": 40,
        "game.horizon_days": 6, "game.volume_window_days": 4,
        "portfolio.horizon_sessions": 6,
        "road1.optimiser": {"lr": 0.05, "max_steps": 60, "patience": 60, "batch_events": 32, "eval_every": 10},
        "road1.grounding_lambda_grid": [0.05], "road1.sheaf_penalty_grid": [1.0], "road1.ridge_grid": [1.0],
        "reader.dials.min_rows_per_class": 20, "reader.dials.ridge_grid": [1.0],
    })


def build_world(cfg: Config, seed: int = 0, n_assets: int = 24, n_sessions: int = 640, n_events: int = 260) -> World:
    rng = np.random.default_rng(seed)
    calendar = TradingCalendar("XNYS", "2017-01-01", "2021-12-31")
    sessions = calendar.sessions[calendar.sessions >= "2018-01-02"][:n_sessions]
    securities = [f"s{i:02d}" for i in range(n_assets)]
    sector = pd.Series([PEER_SECTORS[i % len(PEER_SECTORS)] for i in range(n_assets)], index=securities)
    names = {s: f"Zorg{i:02d} Corp" for i, s in enumerate(securities)}

    market = rng.normal(0.0002, 0.008, n_sessions)
    sector_factor = {s: rng.normal(0, 0.005, n_sessions) for s in PEER_SECTORS}
    base = np.column_stack([market + sector_factor[sector[s]] + rng.normal(0, 0.01, n_sessions) for s in securities])

    warmup, tail = 300, 12
    day0_pool = np.arange(warmup, n_sessions - tail)
    events, tone, event_asset = [], {}, {}
    for i in range(n_events):
        asset = securities[int(rng.integers(n_assets))]
        d0 = int(rng.choice(day0_pool))
        level = int(rng.choice([1, 2]))
        sign = int(rng.choice([-1, 1]))
        words = POSITIVE_TEXT[level] if sign > 0 else NEGATIVE_TEXT[level]
        event_id = f"syn:{i:04d}"
        tone[event_id], event_asset[event_id] = sign * level, asset
        effect = EFFECT_PER_TONE * sign * level
        j = securities.index(asset)
        base[d0, j] += DAY0_MULTIPLE * effect
        base[d0 + 1 : d0 + 1 + REACTION_DAYS, j] += effect
        peers = [securities.index(p) for p in securities if sector[p] == sector[asset] and p != asset]
        base[d0 + 1 : d0 + 1 + REACTION_DAYS, peers] += SPILLOVER * effect
        stamp = pd.Timestamp(sessions[d0]).tz_localize("UTC") + pd.Timedelta(hours=15)
        events.append({
            "event_id": event_id, "available_utc": stamp, "kind": "finance", "family": "sec_8k", "subtype": "sec_8k",
            "title": f"{names[asset]} {words.split(' and ')[0]}", "text": f"{names[asset]} said it will {words}.",
            "pins_json": json.dumps([{"node_id": asset, "node_type": "company", "sector": sector[asset],
                                      "layer": "mid", "priority": 10.0}]),
            "n_pins": 1, "day0_index": d0,
        })
    ret = pd.DataFrame(base, index=sessions, columns=securities)
    close = 100.0 * np.exp(ret.cumsum())
    open_ = close.shift(1).fillna(close.iloc[0]) * np.exp(0.5 * ret)
    high = np.maximum(open_, close) * (1 + 0.004)
    low = np.minimum(open_, close) * (1 - 0.004)
    volume = pd.DataFrame(1.0e6 * np.exp(rng.normal(0, 0.2, ret.shape)), index=sessions, columns=securities)
    for e in events:
        volume.iloc[e["day0_index"] : e["day0_index"] + 5, securities.index(event_asset[e["event_id"]])] *= 2.0
    oo_ret = open_.shift(-1) / open_ - 1.0
    factors = pd.DataFrame(rng.normal(0, 0.005, (n_sessions, 6)), index=sessions,
                           columns=["mkt_rf", "smb", "hml", "rmw", "cma", "mom"])
    factors["mkt_rf"] = market
    factors["rf"] = 0.0
    sleeve = pd.Series(0.7 * market + rng.normal(0, 0.008, n_sessions), index=sessions)

    snapshots = pd.DataFrame([(pd.Timestamp("2018-01-02"), s, "synthetic") for s in securities],
                             columns=["snapshot_date", "security_id", "source"])
    rows = []
    for sec in PEER_SECTORS:
        members = [s for s in securities if sector[s] == sec]
        rows += [(a, b, "industry", pd.Timestamp("2018-01-02"), pd.NaT, 1.0, "synthetic", True, True, False)
                 for k, a in enumerate(members) for b in members[k + 1:]]
    links = pd.DataFrame(rows, columns=["src_node", "dst_node", "link_type", "available_from", "valid_to", "weight",
                                        "source", "src_traded", "dst_traded", "carried_forward"])
    aliases = pd.DataFrame([(s, names[s], "name", None, None) for s in securities],
                           columns=["node_id", "alias", "alias_type", "valid_from", "valid_to"])
    groups = ["mutual_pension", "bank_dealer", "hedge_fund"]
    funds = []
    for k in range(9):
        weights = rng.dirichlet(np.ones(n_assets) * 0.6) * rng.uniform(0.55, 0.95)
        funds.append(FundInput(f"{groups[k % 3]}_{k // 3:02d}", groups[k % 3], float(np.exp(rng.normal(22, 0.4))),
                               weights, float((k // 3 + 1) / 4)))
    return World(
        True, cfg, calendar, sessions, securities, sector, names, close, open_, high, low, volume, ret, oo_ret, sleeve,
        factors, snapshots, links, pd.DataFrame(events), aliases, funds,
        pd.Series(rng.uniform(0.3, 0.9, n_assets), index=securities), tone, event_asset,
    )
