"""Free price data from yfinance, shaped like the store's tables (``px_daily``, ``corp_action``) plus a survivorship audit.

yfinance reports split-adjusted prices, volumes and dividends. The store holds prices as traded, so closes and
dividends are converted back with the later splits (``market.future_split_factor``); volume stays vendor-adjusted
and ``market.volume_is_split_adjusted: true`` tells the rest of the code to undo that. Tickers with no data (mostly
delisted or acquired names) are skipped and listed, never imputed: ``audit`` counts them by year.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence

import pandas as pd

from finflock.common.config import Config
from finflock.data.market import future_split_factor

Downloader = Callable[[Sequence[str], str, str], pd.DataFrame]
FIELDS = ("Open", "High", "Low", "Close", "Adj Close", "Volume", "Dividends", "Stock Splits")
SPLIT, DIVIDEND = "split", "dividend"


def vendor_symbol(ticker: str, cfg: Config) -> str:
    out = ticker
    for old, new in cfg.get("sources.prices.symbol_replacements").items():
        out = out.replace(old, new)
    return out


def components_to_snapshots(components: pd.DataFrame, start: str, end: str, source: str) -> pd.DataFrame:
    """Long table ``(snapshot_date, security_id, source)`` from the date -> comma-separated tickers file."""
    stamps = pd.to_datetime(components["date"])
    kept = components[(stamps >= pd.Timestamp(start)) & (stamps < pd.Timestamp(end))]
    rows = [(pd.Timestamp(d), t.strip()) for d, tickers in zip(kept["date"], kept["tickers"], strict=True) for t in str(tickers).split(",") if t.strip()]
    frame = pd.DataFrame(rows, columns=["snapshot_date", "security_id"])
    frame["source"] = source
    return frame


def yfinance_downloader(tickers: Sequence[str], start: str, end: str) -> pd.DataFrame:  # pragma: no cover - network
    import yfinance as yf

    return yf.download(list(tickers), start=start, end=end, auto_adjust=False, actions=True, group_by="ticker",
                       threads=True, progress=False)


def split_frame(raw: pd.DataFrame, tickers: Sequence[str]) -> dict[str, pd.DataFrame]:
    """One frame per ticker that has any row; a one-ticker download has flat columns, several have (ticker, field)."""
    out: dict[str, pd.DataFrame] = {}
    if raw is None or raw.empty:
        return out
    for ticker in tickers:
        part = raw[ticker] if isinstance(raw.columns, pd.MultiIndex) and ticker in raw.columns.get_level_values(0) else (
            raw if not isinstance(raw.columns, pd.MultiIndex) and len(tickers) == 1 else None)
        if part is None:
            continue
        part = part.dropna(how="all", subset=[c for c in ("Open", "Close") if c in part.columns])
        if len(part):
            out[ticker] = part
    return out


def download_all(
    tickers: Sequence[str], cfg: Config, downloader: Downloader = yfinance_downloader, sleep: Callable[[float], None] = time.sleep,
    log: Callable[[str], None] = print,
) -> tuple[dict[str, pd.DataFrame], list[str]]:
    """Batches with retries; a batch that keeps failing marks its tickers missing and the run goes on."""
    size, retries = int(cfg.get("sources.prices.batch_size")), int(cfg.get("sources.prices.retries"))
    start, end = str(cfg.get("sources.prices.start")), str(cfg.get("sources.prices.end"))
    frames: dict[str, pd.DataFrame] = {}
    failed_batches: list[str] = []
    for i in range(0, len(tickers), size):
        batch = list(tickers[i:i + size])
        for attempt in range(retries):
            try:
                frames.update(split_frame(downloader(batch, start, end), batch))
                break
            except Exception as exc:  # network or vendor error: retry, then give the batch up
                log(f"batch {i // size} attempt {attempt + 1} failed: {exc}")
                sleep(float(cfg.get("sources.prices.retry_sleep_seconds")))
        else:
            failed_batches.extend(batch)
        sleep(float(cfg.get("sources.prices.pause_seconds")))
        log(f"{min(i + size, len(tickers))}/{len(tickers)} tickers, {len(frames)} with data")
    return frames, failed_batches


def build_tables(frames: dict[str, pd.DataFrame], vendor: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """``px_daily`` (as-traded close, vendor-adjusted volume) and ``corp_action`` from per-ticker vendor frames."""
    splits_rows, div_rows = [], []
    for ticker, part in frames.items():
        s = part["Stock Splits"] if "Stock Splits" in part else pd.Series(dtype=float)
        d = part["Dividends"] if "Dividends" in part else pd.Series(dtype=float)
        splits_rows += [(ticker, pd.Timestamp(day).normalize(), SPLIT, float(v), float("nan")) for day, v in s[s > 0].items()]
        div_rows += [(ticker, pd.Timestamp(day).normalize(), DIVIDEND, float("nan"), float(v)) for day, v in d[d > 0].items()]
    actions = pd.DataFrame(splits_rows + div_rows, columns=["security_id", "ex_date", "action_type", "ratio", "amount"])
    actions["announce_date"] = pd.NaT
    actions["source"] = vendor
    rows = []
    for ticker, part in frames.items():
        idx = pd.DatetimeIndex(part.index).tz_localize(None).normalize()
        sessions = idx
        factor = future_split_factor(actions[actions["security_id"] == ticker], sessions, pd.Index([ticker]))[ticker].to_numpy()
        frame = pd.DataFrame({
            "security_id": ticker, "session_date": sessions, "open": part["Open"].to_numpy() * factor,
            "high": part["High"].to_numpy() * factor, "low": part["Low"].to_numpy() * factor,
            "close_raw": part["Close"].to_numpy() * factor, "volume_raw": part["Volume"].to_numpy(), "vendor": vendor,
        })
        rows.append(frame)
    px = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(columns=["security_id", "session_date", "open", "high", "low", "close_raw", "volume_raw", "vendor"])
    if len(actions):  # dividends were reported per post-split share; as traded they were larger by the later splits
        sessions_by = {t: pd.DatetimeIndex(sorted(pd.to_datetime(px.loc[px["security_id"] == t, "session_date"]))) for t in actions["security_id"].unique()}
        for t, sess in sessions_by.items():
            mask = (actions["security_id"] == t) & (actions["action_type"] == DIVIDEND)
            if mask.any() and len(sess):
                factors = future_split_factor(actions[(actions["security_id"] == t)], sess, pd.Index([t]))[t]
                at = factors.reindex(pd.DatetimeIndex(actions.loc[mask, "ex_date"]), method="ffill").to_numpy()
                actions.loc[mask, "amount"] = actions.loc[mask, "amount"].to_numpy() * at
    return px, actions


def coverage(px: pd.DataFrame, wanted: Sequence[str], min_rows: int) -> pd.DataFrame:
    """Per wanted ticker: rows, first and last date, and whether it counts as covered."""
    grouped = px.groupby("security_id")["session_date"].agg(["count", "min", "max"]) if len(px) else pd.DataFrame(columns=["count", "min", "max"])
    table = grouped.reindex(list(wanted)).rename(columns={"count": "rows", "min": "first_date", "max": "last_date"})
    table["rows"] = table["rows"].fillna(0).astype(int)
    table["covered"] = table["rows"] >= min_rows
    table.index.name = "security_id"
    return table.reset_index()


def audit(snapshots: pd.DataFrame, cov: pd.DataFrame) -> pd.DataFrame:
    """Survivorship audit: for each year, the share of index members (ticker-days) whose prices exist in the data."""
    covered = set(cov.loc[cov["covered"], "security_id"])
    frame = snapshots.assign(year=pd.to_datetime(snapshots["snapshot_date"]).dt.year, has_data=snapshots["security_id"].isin(covered))
    by_year = frame.groupby("year").agg(member_days=("security_id", "size"), covered_member_days=("has_data", "sum"))
    names = frame.groupby("year").apply(lambda g: pd.Series({"distinct_members": g["security_id"].nunique(),
                                                             "missing_members": g.loc[~g["has_data"], "security_id"].nunique()}),
                                        include_groups=False)
    out = by_year.join(names)
    out["covered_share"] = out["covered_member_days"] / out["member_days"]
    out["covered_member_share"] = 1.0 - out["missing_members"] / out["distinct_members"]  # by name, not weighted by snapshot count
    return out.reset_index()
