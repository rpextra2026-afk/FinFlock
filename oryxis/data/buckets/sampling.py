"""Sampling the reader sample S_R and the framework sample S_F, index events, coverage (spec 3.7).

Strata are ``family x era x split``; within a stratum events are ordered by ``sha256(seed|event_id)``
so a sample is reproducible and independent of row order. Events within the embargo are dropped
before any sampling. The rule is written to config before looking at any price.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from oryxis.common.config import Config
from oryxis.common.hashing import sha256_hex
from oryxis.common.timing import TradingCalendar

STRATA = ["family", "era", "split"]
HASH_CHARS = 16


def hashed_order(event_ids: pd.Series, seed: str) -> pd.Series:
    """Deterministic pseudo-random rank key of each event: ``sha256(seed|event_id)``."""
    return event_ids.map(lambda e: sha256_hex(f"{seed}|{e}")[:HASH_CHARS])


def largest_remainder(sizes: pd.Series, total: int) -> pd.Series:
    """Allocate ``total`` items over strata proportionally to ``sizes`` (never above a stratum's size)."""
    total = min(total, int(sizes.sum()))
    exact = sizes / sizes.sum() * total
    base = np.floor(exact).astype(int)
    short = total - int(base.sum())
    for key in (exact - base).sort_values(ascending=False, kind="mergesort").index[:short]:
        base[key] += 1
    return base.clip(upper=sizes)


def stratified_sample(pool: pd.DataFrame, n: int, seed: str, per_family_min: int) -> pd.DataFrame:
    """Hashed stratified sample of ``n`` events. First guarantee ``per_family_min`` per family where
    available, then fill proportionally to stratum size."""
    if n >= len(pool):
        return pool.copy()
    pool = pool.assign(_key=hashed_order(pool["event_id"], seed)).sort_values("_key", kind="mergesort")
    chosen = []
    for _, part in pool.groupby("family", sort=False):
        chosen.append(part.head(min(per_family_min, len(part))))
    first = pd.concat(chosen) if chosen else pool.iloc[0:0]
    if len(first) >= n:
        return first.sort_values("_key").head(n).drop(columns="_key")
    rest = pool.drop(first.index)
    sizes = rest.groupby(STRATA, sort=True).size()
    quota = largest_remainder(sizes, n - len(first))
    taken = [part.head(int(quota.get(key, 0))) for key, part in rest.groupby(STRATA, sort=True)]
    return pd.concat([first, *taken]).drop(columns="_key")


def ensure_multi_pin(selected: pd.DataFrame, pool: pd.DataFrame, min_share: float, seed: str) -> pd.DataFrame:
    """Raise the multi-pin share of ``selected`` to ``min_share`` by swapping single-pin events for
    multi-pin ones from the rest of the pool (hash order). Stops when the pool has no more multi-pin."""
    target = int(np.ceil(min_share * len(selected)))
    have = int((selected["n_pins"] >= 2).sum())
    if have >= target:
        return selected
    candidates = pool[~pool["event_id"].isin(selected["event_id"]) & (pool["n_pins"] >= 2)]
    candidates = candidates.assign(_key=hashed_order(candidates["event_id"], seed)).sort_values("_key")
    singles = selected[selected["n_pins"] < 2].assign(_key=lambda f: hashed_order(f["event_id"], seed))
    swaps = min(target - have, len(candidates), len(singles))
    drop = singles.sort_values("_key", ascending=False).head(swaps)["event_id"]
    return pd.concat([selected[~selected["event_id"].isin(drop)], candidates.head(swaps).drop(columns="_key")])


def build_samples(
    events: pd.DataFrame, reader_n: int, framework_dev: int, framework_test_cap: int, cfg: Config, seed: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``(S_R, S_F)``: S_F is a subset of S_R; dev is sampled, test is taken up to its cap."""
    spec = cfg.get("events.sampling")
    usable = events[events["split"].isin(["dev", "test"])]
    reader = stratified_sample(usable, reader_n, f"{seed}|{spec['sr_label']}", int(spec["per_family_min"]))
    sf_seed = f"{seed}|{spec['sf_label']}"
    pieces = []
    for split, size in (("dev", framework_dev), ("test", framework_test_cap)):
        pool = reader[reader["split"] == split]
        chosen = stratified_sample(pool, size, sf_seed, int(spec["per_family_min"]))
        pieces.append(ensure_multi_pin(chosen, pool, float(spec["multi_pin_share_min"]), sf_seed))
    return reader, pd.concat(pieces)


def index_event_dates(snapshots: pd.DataFrame) -> pd.DataFrame:
    """Additions and deletions seen between consecutive snapshots, dated at the later snapshot."""
    stamps = sorted(pd.to_datetime(snapshots["snapshot_date"]).unique())
    members = {s: set(snapshots.loc[pd.to_datetime(snapshots["snapshot_date"]) == s, "security_id"]) for s in stamps}
    rows = []
    for before, after in zip(stamps, stamps[1:], strict=False):
        rows += [(sec, pd.Timestamp(after), "addition") for sec in sorted(members[after] - members[before])]
        rows += [(sec, pd.Timestamp(after), "deletion") for sec in sorted(members[before] - members[after])]
    return pd.DataFrame(rows, columns=["security_id", "date", "change"])


def drop_index_events(
    events: pd.DataFrame, pinned: dict[str, set[str]], changes: pd.DataFrame, calendar: TradingCalendar, cfg: Config
) -> tuple[pd.DataFrame, pd.Series]:
    """Drop events whose window (day -1 .. day +10) comes within the pad of an index change of a pinned
    asset (the effective date is only known to within a few days). Returns kept events and drops by year."""
    spec = cfg.get("events.index_events")
    pad = int(spec["pad_sessions"])
    before, after = int(spec["window_before"]), int(spec["window_after"])
    by_security = {sec: part["date"].tolist() for sec, part in changes.groupby("security_id")}
    sessions = calendar.sessions
    dropped = []
    for event_id, day0 in zip(events["event_id"], events["session_day0"], strict=True):
        low = sessions[max(sessions.searchsorted(day0) + before, 0)]
        high = sessions[min(sessions.searchsorted(day0) + after, len(sessions) - 1)]
        hit = False
        for security in pinned.get(event_id, ()):
            for date in by_security.get(security, ()):
                position = sessions.searchsorted(date)
                near_low = sessions[max(position - pad, 0)]
                near_high = sessions[min(position + pad, len(sessions) - 1)]
                if near_low <= high and low <= near_high:
                    hit = True
        dropped.append(hit)
    mask = pd.Series(dropped, index=events.index)
    years = pd.to_datetime(events.loc[mask, "session_day0"]).dt.year
    return events[~mask], years.value_counts().sort_index()


def coverage_matrix(events: pd.DataFrame) -> pd.DataFrame:
    """Event counts by kind x year x source (declared holes show up as zeros)."""
    years = pd.to_datetime(events["session_day0"]).dt.year.rename("year")
    return events.groupby(["kind", years, "source_name"]).size().rename("events").reset_index()


def kinds_missing_by_year(matrix: pd.DataFrame, kinds: list[str], years: list[int]) -> dict[int, list[str]]:
    """Kinds with no events in each year (the persona types that need them are not tested there)."""
    present = matrix.groupby("year")["kind"].apply(set).to_dict()
    return {y: [k for k in kinds if k not in present.get(y, set())] for y in years}
