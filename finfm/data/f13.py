"""SEC Form 13F ingestion, CUSIP matching and ownership shares (spec 3.4).

Rules (all with counts logged by the caller, nothing dropped silently):

* original 13F-HR only; amendments and notices are skipped and counted;
* PUT/CALL rows and non-SH rows are dropped;
* the VALUE unit (dollars or thousands) is chosen per row from the implied price against the as-traded
  close at period end, never from the filing date;
* holdings are usable at ``t`` only if the filing date is at most ``t - 1`` trading day;
* weights inside a universe are ``shares x as-traded close at period end``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

from finfm.common.config import Config
from finfm.common.timing import TradingCalendar

HALF = 0.5
PUT_CALL_NONE = ("", "nan", "None")


@dataclass(frozen=True)
class FilingSummary:
    """Counts of what the filing filter kept and skipped (reported with the data audit)."""

    kept: int
    skipped_amendments: int
    skipped_other_types: int


def parse_filings(submission: pd.DataFrame, cfg: Config, calendar: TradingCalendar) -> tuple[pd.DataFrame, FilingSummary]:
    """Original 13F-HR filings with their availability session (first session after the filing date)."""
    c = cfg.get("f13.columns")
    frame = pd.DataFrame({
        "accession": submission[c["accession"]],
        "manager_cik": submission[c["cik"]].astype(str),
        "period": pd.to_datetime(submission[c["period"]], format=cfg.get("f13.date_format")),
        "filing_date": pd.to_datetime(submission[c["filing_date"]], format=cfg.get("f13.date_format")),
        "submission_type": submission[c["submission_type"]].astype(str),
    })
    frame["is_amendment"] = frame["submission_type"].str.endswith("/A")
    original = frame["submission_type"].isin(list(cfg.get("f13.original_types")))
    summary = FilingSummary(
        int(original.sum()), int(frame["is_amendment"].sum()), int((~original & ~frame["is_amendment"]).sum())
    )
    kept = frame[original].copy()
    kept["available_from_session"] = [
        calendar.sessions[calendar.sessions.searchsorted(d, side="right")]
        if calendar.sessions.searchsorted(d, side="right") < len(calendar.sessions) else pd.NaT
        for d in kept["filing_date"]
    ]
    return kept.reset_index(drop=True), summary


def choose_unit_factor(
    value_raw: np.ndarray, shares: np.ndarray, close: np.ndarray, factors: list[float]
) -> np.ndarray:
    """Per row, the factor whose implied price ``value x factor / shares`` is closest (in logs) to the
    as-traded close. Rows without a usable close return NaN."""
    with np.errstate(divide="ignore", invalid="ignore"):
        implied = np.column_stack([value_raw * f / shares for f in factors])
        distance = np.abs(np.log(implied / close[:, None]))
    distance = np.where(np.isfinite(distance), distance, np.inf)
    best = distance.argmin(axis=1)
    chosen = np.asarray(factors, dtype=float)[best]
    return np.where(np.isfinite(distance.min(axis=1)), chosen, np.nan)


def parse_infotable(
    infotable: pd.DataFrame, filings: pd.DataFrame, close_at_period_end: pd.Series, cfg: Config
) -> tuple[pd.DataFrame, dict[str, int]]:
    """Holdings rows of the kept filings with ``unit_factor`` and ``value_usd``.

    ``close_at_period_end`` is indexed by ``(cusip9, period)`` with the as-traded close. Rows with no
    price get the majority factor of their filing; if the filing has none either they are dropped and
    counted (never decided by date).
    """
    c = cfg.get("f13.columns")
    counts = {"rows": len(infotable)}
    frame = infotable[infotable[c["accession"]].isin(filings["accession"])].copy()
    frame = frame.rename(columns={
        c["accession"]: "accession", c["infotable_sk"]: "infotable_sk", c["cusip"]: "cusip9", c["value"]: "value_raw",
        c["shares"]: "shares", c["sh_prn"]: "sh_prn", c["putcall"]: "putcall", c["issuer"]: "issuer_name",
    })
    frame["putcall"] = frame["putcall"].astype(str)
    is_option = ~frame["putcall"].isin(PUT_CALL_NONE)
    not_shares = frame["sh_prn"].astype(str) != str(cfg.get("f13.share_type"))
    counts["dropped_putcall"], counts["dropped_non_sh"] = int(is_option.sum()), int((not_shares & ~is_option).sum())
    frame = frame[~is_option & ~not_shares].copy()
    frame = frame.merge(filings[["accession", "manager_cik", "period"]], on="accession")
    keys = list(zip(frame["cusip9"], frame["period"], strict=True))
    frame["close"] = close_at_period_end.reindex(pd.MultiIndex.from_tuples(keys)).to_numpy()
    frame["shares"] = pd.to_numeric(frame["shares"], errors="coerce")
    frame["value_raw"] = pd.to_numeric(frame["value_raw"], errors="coerce")
    factors = [float(f) for f in cfg.get("f13.unit_factors")]
    frame["unit_factor"] = choose_unit_factor(
        frame["value_raw"].to_numpy(), frame["shares"].to_numpy(), frame["close"].to_numpy(), factors
    )
    majority = frame.groupby("accession")["unit_factor"].agg(lambda s: s.mode().iloc[0] if s.notna().any() else np.nan)
    missing = frame["unit_factor"].isna()
    frame.loc[missing, "unit_factor"] = frame.loc[missing, "accession"].map(majority)
    counts["dropped_no_unit_factor"] = int(frame["unit_factor"].isna().sum())
    frame = frame.dropna(subset=["unit_factor", "shares", "value_raw"])
    frame["value_usd"] = frame["value_raw"] * frame["unit_factor"]
    rule = str(cfg.get("f13.combination_rule"))
    if rule == "dedupe_identical":
        before = len(frame)
        frame = frame.drop_duplicates(["accession", "cusip9", "shares", "putcall"])
        counts["dropped_combination_duplicates"] = before - len(frame)
    return frame.reset_index(drop=True), counts


def latest_filing_per_manager(filings: pd.DataFrame, session: pd.Timestamp) -> pd.DataFrame:
    """For each manager, the filing with the latest period among those available on ``session``."""
    usable = filings[filings["available_from_session"] <= session]
    ordered = usable.sort_values(["manager_cik", "period", "filing_date"])
    return ordered.groupby("manager_cik").tail(1).reset_index(drop=True)


def holdings_weights(holdings: pd.DataFrame, close_at_period_end: pd.Series, universe_ids: pd.Series) -> pd.DataFrame:
    """Dollar weights ``shares x close`` within a universe and the manager's S&P cover.

    ``universe_ids`` maps ``cusip9`` to ``security_id`` for universe members only; other holdings count
    toward the manager's reported book (from ``value_usd``) but not toward the universe weight.
    """
    frame = holdings.assign(security_id=holdings["cusip9"].map(universe_ids))
    book = frame.groupby("accession")["value_usd"].sum().rename("book_usd")
    inside = frame.dropna(subset=["security_id"]).copy()
    inside["dollars"] = inside["shares"] * inside["close"]
    totals = inside.groupby("accession")["dollars"].transform("sum")
    inside["weight"] = inside["dollars"] / totals
    cover = (inside.groupby("accession")["value_usd"].sum() / book).rename("sp_cover")
    return inside.merge(book, on="accession").merge(cover, on="accession")


def uncovered_share(held_shares: pd.Series, shares_out: pd.Series, cfg: Config) -> pd.DataFrame:
    """Retail weight ``u = clip(1 - held / outstanding, 0, 0.95)`` and the raw-ratio flag (spec 3.4)."""
    ratio = held_shares.reindex(shares_out.index).fillna(0.0) / shares_out
    low, high = (float(x) for x in cfg.get("f13.uncovered_share.flag_range"))
    cap = float(cfg.get("f13.uncovered_share.cap"))
    return pd.DataFrame({
        "held_ratio": ratio,
        "uncovered_share": (1.0 - ratio).clip(lower=float(cfg.get("f13.uncovered_share.floor")), upper=cap),
        "flagged": (ratio < low) | (ratio > high),
    })


# --- CUSIP matching ------------------------------------------------------------------------------


def normalise_name(name: str, suffixes: set[str]) -> str:
    """Upper-case, strip punctuation and corporate suffixes for name matching."""
    tokens = re.sub(r"[^A-Z0-9 ]", " ", str(name).upper()).split()
    return " ".join(t for t in tokens if t not in suffixes)


def similarity(a: str, b: str) -> float:
    """Token Jaccard blended with character bigram overlap; 1.0 for identical normalised names."""
    ta, tb = set(a.split()), set(b.split())
    if not ta or not tb:
        return 0.0
    token = len(ta & tb) / len(ta | tb)
    ba = {a[i : i + 2] for i in range(len(a) - 1)}
    bb = {b[i : i + 2] for i in range(len(b) - 1)}
    bigram = len(ba & bb) / len(ba | bb) if ba and bb else 0.0
    return HALF * token + HALF * bigram


def match_cusips(
    issuers: pd.DataFrame, targets: pd.DataFrame, cfg: Config
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Map 13F issuer names to securities by normalised-name similarity.

    ``issuers``: ``cusip9, issuer_name``. ``targets``: ``security_id, name`` (current and former names,
    one row each). Returns accepted matches and the below-threshold candidates for ``QC/cusip_review``.
    Candidates are blocked by the first token to keep the search small.
    """
    suffixes = set(cfg.get("f13.name_match.suffixes"))
    threshold = float(cfg.get("f13.name_match.threshold"))
    prepared = targets.assign(norm=targets["name"].map(lambda n: normalise_name(n, suffixes)))
    by_block: dict[str, list[tuple[str, str]]] = {}
    for security, norm in zip(prepared["security_id"], prepared["norm"], strict=True):
        if norm:
            by_block.setdefault(norm.split()[0], []).append((security, norm))
    accepted, review = [], []
    for cusip, name in issuers[["cusip9", "issuer_name"]].drop_duplicates("cusip9").itertuples(index=False):
        norm = normalise_name(name, suffixes)
        if not norm:
            continue
        best = max(
            ((security, similarity(norm, cand)) for security, cand in by_block.get(norm.split()[0], [])),
            key=lambda pair: pair[1], default=(None, 0.0),
        )
        row = {"cusip9": cusip, "issuer_name": name, "security_id": best[0], "score": best[1]}
        (accepted if best[0] is not None and best[1] >= threshold else review).append(row)
    return pd.DataFrame(accepted, columns=["cusip9", "issuer_name", "security_id", "score"]), pd.DataFrame(
        review, columns=["cusip9", "issuer_name", "security_id", "score"]
    )
