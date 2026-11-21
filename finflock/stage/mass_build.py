"""MASS inputs from free data: weekly decision dates, the 100 largest S&P 500 members each week, price-derived modalities, dividend
yield, news titles and the investor types (random modality draws).

Not available from free data and therefore absent (the paper states this): the fundamentals modalities valuation, quality and growth
(they need SEC financial-statement panels) and the macro text. The MASS draw only uses the modalities that exist.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pandas as pd

from finflock.common.config import Config
from finflock.common.schema import UNIT_ID
from finflock.compute import payloads as pl
from finflock.compute.bundle import make_job_id
from finflock.compute.shard import EST_SECONDS, plan_shards
from finflock.compute.staging import stage_shard
from finflock.data.market import total_returns
from finflock.rivals.mass.modalities import ModalityStore, draw_modalities, price_modalities
from finflock.rivals.mass.portfolio import weekly_rebalance_dates
from finflock.rivals.mass.run import unit

DIVIDEND_WINDOW_SESSIONS = 252
NEWS_TITLES_MAX = 20


def adjusted_panels(px: pd.DataFrame, actions: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """Wide panels with open/high/low/close scaled to total-return-consistent levels, and volume."""
    wide = {c: px.pivot_table(index="session_date", columns="security_id", values=c, aggfunc="last").sort_index()
            for c in ("open", "high", "low", "close_raw", "volume_raw")}
    close_ret, _ = total_returns(wide["close_raw"], wide["open"], actions)
    level = (1.0 + close_ret.fillna(0.0)).cumprod()
    scale = level / wide["close_raw"]
    return {"open": wide["open"] * scale, "high": wide["high"] * scale, "low": wide["low"] * scale, "close": level, "volume": wide["volume_raw"]}


def dividend_yield(px: pd.DataFrame, actions: pd.DataFrame) -> pd.DataFrame:
    """Trailing 252-session dividends per as-traded close (sessions x securities)."""
    close = px.pivot_table(index="session_date", columns="security_id", values="close_raw", aggfunc="last").sort_index()
    paid = pd.DataFrame(0.0, index=close.index, columns=close.columns)
    for row in actions[actions["action_type"] == "dividend"].itertuples():
        if row.security_id in paid.columns and row.ex_date in paid.index:
            paid.at[row.ex_date, row.security_id] += row.amount
    return paid.rolling(DIVIDEND_WINDOW_SESSIONS, min_periods=1).sum() / close


def decision_dates(sessions: pd.DatetimeIndex, start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    return sorted(d for d in weekly_rebalance_dates(sessions) if start <= d <= end)


def build_world(
    px: pd.DataFrame, actions: pd.DataFrame, sector_of: pd.Series, cap_proxy: pd.DataFrame, members_on: Callable[[pd.Timestamp], list[str]],
    news_events: pd.DataFrame, dates: list[pd.Timestamp], cfg: Config,
) -> tuple[pd.DataFrame, dict[str, pd.DataFrame], dict[str, list[str]], dict[str, list[str]]]:
    """``(pool, modalities, types, news)`` for the weekly dates; only rows of pooled stocks on decision dates are kept."""
    spec = cfg.get("rivals.mass")
    panels = adjusted_panels(px, actions)
    sector = sector_of.reindex(panels["close"].columns).fillna("unknown")
    modalities = price_modalities(panels["close"], panels["high"], panels["low"], panels["open"], panels["volume"], sector)
    dy = dividend_yield(px, actions)
    modalities["dividend"] = dy.stack(future_stack=True).rename("Dividend yield").reset_index().set_axis(["Date", "Stock", "Dividend yield"], axis=1)
    pool_rows = []
    for date in dates:
        members = [m for m in members_on(date) if m in cap_proxy.columns]
        top = cap_proxy.loc[:date].iloc[-1].reindex(members).dropna().nlargest(int(spec["pool_size"]))
        pool_rows += [{"Date": date, "Stock": s, "Industry": sector.get(s, "unknown"), "FREE_MV": float(v)} for s, v in top.items()]
    pool = pd.DataFrame(pool_rows)
    keep = pd.MultiIndex.from_frame(pool[["Date", "Stock"]])
    store = {name: frame[pd.MultiIndex.from_frame(frame[["Date", "Stock"]]).isin(keep)].reset_index(drop=True) for name, frame in modalities.items()}
    available = sorted(name for name in spec["modalities"] if name in store or name == "news")
    seed = int(cfg.get("runtime.base_seed"))
    types = {f"type{i:02d}": draw_modalities(cfg, available, (seed, "mass_type", i)) for i in range(int(spec["n_types"]))}
    news: dict[str, list[str]] = {}
    for row in news_events.itertuples():
        day = str(pd.Timestamp(row.available_utc).tz_convert(None).normalize().date()) if pd.Timestamp(row.available_utc).tzinfo else str(pd.Timestamp(row.available_utc).normalize().date())
        news.setdefault(day, []).append(str(row.title))
    return pool, store, types, {d: t[:NEWS_TITLES_MAX] for d, t in news.items()}


def mass_units(dates: list[pd.Timestamp], types: dict[str, list[str]], cfg: Config) -> pd.DataFrame:
    agents = int(cfg.get("rivals.mass.agents_per_type"))
    ids = [unit(str(d.date()), t, a) for d in dates for t in types for a in range(agents)]
    return pd.DataFrame({UNIT_ID: ids, EST_SECONDS: float(cfg.get("runs.est_seconds_per_unit.mass"))})


def save_mass_inputs(folder: Path, units: pd.DataFrame, pool: pd.DataFrame, store: dict[str, pd.DataFrame], types, news) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    units.to_parquet(folder / "units.parquet", index=False)
    pl.save_mass_world(pool, ModalityStore(store), types, {}, news, folder)


def stage_mass_job(bucket: str, llm: str, cfg: Config, market_dir: Path, out_root: Path, log: Callable[[str], None] = print) -> list[Path]:
    from finflock.compute.session import SessionError

    pre = market_dir / "prestaged" / f"mass-{bucket}"
    if not (pre / "units.parquet").exists():
        raise SessionError(f"the market dataset has no prebuilt MASS inputs for {bucket} ({pre})")
    units = pd.read_parquet(pre / "units.parquet")
    world = pl.load_mass_world(pre)
    pool = pd.read_parquet(pre / pl.MASS_POOL)
    news = json.loads((pre / pl.MASS_NEWS).read_text(encoding="utf-8"))
    shards = plan_shards(make_job_id("mass", bucket, llm), units, cfg)
    log(f"{bucket}: {len(units)} mass units in {len(shards)} shard(s)")
    folders = []
    for shard in shards:
        dates = {u.split("|")[0] for u in shard.unit_ids}

        def write(folder: Path, dates=dates) -> None:
            chosen = {n: f[f["Date"].isin({pd.Timestamp(d) for d in dates})] for n, f in world.store.frames.items()}
            pl.save_mass_world(pool[pool["Date"].isin({pd.Timestamp(d) for d in dates})], ModalityStore(chosen), world.types, {},
                               {d: t for d, t in news.items() if d in dates}, folder)

        folders.append(stage_shard(out_root, "mass", bucket, llm, shard, units, write, cfg))
    return folders
