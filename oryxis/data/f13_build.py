"""13F quarterly data sets -> compact tables for the persona panels: filings, holdings of S&P 500 names, manager book and type.

Names are matched to securities by normalised issuer name (``f13.match_cusips``); the matching is approximate and the
unmatched share is reported, not hidden. Prices come from the market tables (as traded close at the period end).
"""

from __future__ import annotations

import zipfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from oryxis.common.config import Config
from oryxis.common.timing import TradingCalendar
from oryxis.data.f13 import match_cusips, parse_filings, parse_infotable
from oryxis.data.managers import classify_managers, passive_flags, quarterly_turnover

SUMMARY = "SUMMARYPAGE.tsv"
BOOK_RAW = "TABLEVALUETOTAL"
COVER_NAME = "FILINGMANAGER_NAME"


def read_zip(path: Path, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """``(submission with manager names and book, infotable)`` of one quarterly zip, all columns as text."""
    files = cfg.get("f13.files")
    with zipfile.ZipFile(path) as z:
        member = {Path(n).name: n for n in z.namelist()}  # newer files keep their tables inside a folder
        sub = pd.read_csv(z.open(member[files["submission"]]), sep="\t", dtype=str)
        cover = pd.read_csv(z.open(member[files["coverpage"]]), sep="\t", dtype=str, usecols=["ACCESSION_NUMBER", COVER_NAME])
        summary = pd.read_csv(z.open(member[SUMMARY]), sep="\t", dtype=str, usecols=["ACCESSION_NUMBER", "TABLEENTRYTOTAL", BOOK_RAW])
        info = pd.read_csv(z.open(member[files["infotable"]]), sep="\t", dtype=str)
    return sub.merge(cover, on="ACCESSION_NUMBER", how="left").merge(summary, on="ACCESSION_NUMBER", how="left"), info


def close_at_period_end(px_close: pd.DataFrame, periods: Sequence[pd.Timestamp]) -> pd.DataFrame:
    """As-traded close of every security on the last session at or before each period end (periods x securities)."""
    return px_close.reindex(pd.DatetimeIndex(periods), method="ffill")


def reduce_zip(path: Path, names: pd.DataFrame, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """One quarterly zip reduced to ``(submission with names and book, info rows of matched issuers, accepted cusip matches)``.

    Only rows whose issuer name matches a security are kept, so the memory and disk needed stay small. Matching a name
    does not depend on the other files, so doing it zip by zip gives the same result as doing it once for everything.
    """
    sub, info = read_zip(path, cfg)
    c = cfg.get("f13.columns")
    issuers = info.rename(columns={c["cusip"]: "cusip9", c["issuer"]: "issuer_name"})[["cusip9", "issuer_name"]]
    accepted, _ = match_cusips(issuers, names, cfg)
    wanted = [c[k] for k in ("accession", "infotable_sk", "cusip", "issuer", "value", "shares", "sh_prn", "putcall")]
    return sub, info.loc[info[c["cusip"]].isin(accepted["cusip9"]), wanted].copy(), accepted


def assemble_tables(
    reduced: Sequence[tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]], px_close: pd.DataFrame, splits: pd.DataFrame,
    calendar: TradingCalendar, cfg: Config, overrides: dict[str, str] | None = None, passive_families: Sequence[str] = (),
) -> dict[str, pd.DataFrame]:
    """Filings, holdings, managers and the CUSIP map from the reduced zips."""
    submission = pd.concat([r[0] for r in reduced], ignore_index=True)
    matched = pd.concat([r[1] for r in reduced], ignore_index=True)
    accepted = pd.concat([r[2] for r in reduced], ignore_index=True).drop_duplicates("cusip9")
    first = pd.Timestamp(cfg.get("sources.f13.first_period"))
    filings, _ = parse_filings(submission, cfg, calendar)
    filings = filings[filings["period"] >= first].drop_duplicates("accession").reset_index(drop=True)
    meta = submission.drop_duplicates("ACCESSION_NUMBER").set_index("ACCESSION_NUMBER")
    c = cfg.get("f13.columns")
    cusip_to_security = accepted.set_index("cusip9")["security_id"]
    px_end = close_at_period_end(px_close, sorted(filings["period"].unique()))
    close_series = pd.Series({(cusip, period): px_end.at[period, sec] for cusip, sec in cusip_to_security.items() if sec in px_end.columns
                              for period in px_end.index})
    close_series.index = pd.MultiIndex.from_tuples(close_series.index)
    holdings, counts = parse_infotable(matched, filings, close_series, cfg)
    holdings["security_id"] = holdings["cusip9"].map(cusip_to_security)
    holdings["dollars"] = holdings["shares"] * holdings["close"]
    holdings = holdings.dropna(subset=["dollars"])
    holdings["weight"] = holdings["dollars"] / holdings.groupby("accession")["dollars"].transform("sum")
    # manager book: the filing's reported total value in dollars (unit factor from its matched rows), positions from all rows
    factor = holdings.groupby("accession")["unit_factor"].first()
    book = pd.to_numeric(meta.loc[filings["accession"], BOOK_RAW], errors="coerce").to_numpy() * filings["accession"].map(factor).to_numpy()
    managers = filings.assign(name=meta.loc[filings["accession"], COVER_NAME].to_numpy(), book_usd=book,
                              n_positions=pd.to_numeric(meta.loc[filings["accession"], "TABLEENTRYTOTAL"], errors="coerce").to_numpy())
    sp_value = holdings.groupby("accession")["value_usd"].sum()
    managers["sp_cover"] = managers["accession"].map(sp_value) / managers["book_usd"]
    managers = managers.dropna(subset=["book_usd"]).query("book_usd > 0").sort_values(["manager_cik", "period"]).reset_index(drop=True)
    managers = managers[managers["sp_cover"] >= float(cfg.get("sources.f13.min_sp_cover"))].reset_index(drop=True)
    managers["turnover"] = _turnovers(managers, holdings, px_end, splits)
    typed = []
    for period, part in managers.groupby("period"):
        frame = part.set_index("manager_cik")[["name", "n_positions", "turnover"]].fillna({"turnover": 0.0})
        typed.append(classify_managers(frame, cfg, overrides).rename_axis("manager_cik").reset_index().assign(period=period))
    managers = managers.merge(pd.concat(typed, ignore_index=True), on=["manager_cik", "period"], how="left")
    managers["passive_flag"] = passive_flags(managers["name"], list(passive_families))
    keep = holdings.merge(managers[["accession"]], on="accession")[["accession", "security_id", "shares", "close", "value_usd", "weight"]]
    return {"f13_filings": filings, "f13_holdings": keep, "f13_managers": managers.drop(columns=["submission_type"]),
            "f13_cusip_map": accepted, "counts": pd.DataFrame([counts])}


def build_tables(
    zips: Sequence[Path], aliases: pd.DataFrame, px_close: pd.DataFrame, splits: pd.DataFrame, calendar: TradingCalendar, cfg: Config,
    overrides: dict[str, str] | None = None, passive_families: Sequence[str] = (),
) -> dict[str, pd.DataFrame]:
    """``reduce_zip`` on every zip, then ``assemble_tables``."""
    names = aliases[aliases["alias_type"] == "name"].rename(columns={"node_id": "security_id", "alias": "name"})[["security_id", "name"]]
    return assemble_tables([reduce_zip(path, names, cfg) for path in zips], px_close, splits, calendar, cfg, overrides, passive_families)


def _turnovers(managers: pd.DataFrame, holdings: pd.DataFrame, px_end: pd.DataFrame, splits: pd.DataFrame) -> np.ndarray:
    """Quarterly turnover per filing against the same manager's previous filing (NaN for the first)."""
    shares = {a: g.groupby("security_id")["shares"].sum() for a, g in holdings.groupby("accession")}  # several CUSIPs can map to one security
    out = np.full(len(managers), np.nan)
    previous: dict[str, tuple[pd.Timestamp, str]] = {}
    for i, row in enumerate(managers.itertuples()):
        last = previous.get(row.manager_cik)
        if last is not None and last[1] in shares and row.accession in shares:
            span = splits[(splits["ex_date"] > last[0]) & (splits["ex_date"] <= row.period)]
            ratio = span.groupby("security_id")["ratio"].prod() if len(span) else pd.Series(dtype=float)
            prices = px_end.loc[row.period] if row.period in px_end.index else pd.Series(dtype=float)
            out[i] = quarterly_turnover(shares[last[1]], shares[row.accession], prices, ratio, float(row.book_usd))
        previous[row.manager_cik] = (row.period, row.accession)
    return out
