"""Point-in-time S&P 500 membership from dated snapshots (spec 3.3, open item 22).

Never start from today's list and walk backwards: that hides every name that was dropped, which is
survivorship bias. Membership on a date is the latest snapshot dated on or before that date, applied
as of the previous session; later snapshots can change nothing.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from oryxis.common.timing import TradingCalendar

SNAPSHOT_DATE = "snapshot_date"
SECURITY = "security_id"


class DataQualityError(RuntimeError):
    """Raised when a data check says the build must stop (spec: more than 10% mismatch)."""


@dataclass(frozen=True)
class Members:
    date: pd.Timestamp
    snapshot_date: pd.Timestamp | None
    stale_days: int | None
    security_ids: frozenset[str]


def members_on(snapshots: pd.DataFrame, date: pd.Timestamp) -> Members:
    """Members according to the latest snapshot dated on or before ``date``."""
    stamps = pd.to_datetime(snapshots[SNAPSHOT_DATE])
    eligible = stamps[stamps <= pd.Timestamp(date)]
    if eligible.empty:
        return Members(pd.Timestamp(date), None, None, frozenset())
    latest = eligible.max()
    ids = snapshots.loc[stamps == latest, SECURITY]
    return Members(pd.Timestamp(date), latest, int((pd.Timestamp(date) - latest).days), frozenset(ids))


def universe_asof(snapshots: pd.DataFrame, session: pd.Timestamp, calendar: TradingCalendar) -> Members:
    """Members usable on ``session``: the membership at the previous session (information through t-1)."""
    previous = calendar.shift(session, -1)
    return members_on(snapshots, previous)


def membership_intervals(snapshots: pd.DataFrame, last_date: pd.Timestamp) -> pd.DataFrame:
    """Derived view ``(security_id, start_date, end_date, source, snapshot_date)`` for reporting.

    A name starts at the first snapshot that shows it and ends the day before the first later
    snapshot that does not. The view is for audit; the point-in-time lookup is ``members_on``.
    """
    stamps = sorted(pd.to_datetime(snapshots[SNAPSHOT_DATE]).unique())
    rows = []
    for security, part in snapshots.groupby(SECURITY):
        present = set(pd.to_datetime(part[SNAPSHOT_DATE]))
        start = None
        source = part["source"].iloc[0]
        for index, stamp in enumerate(stamps):
            if stamp in present and start is None:
                start = stamp
            if stamp not in present and start is not None:
                rows.append((security, start, stamp - pd.Timedelta(days=1), source, start))
                start = None
        if start is not None:
            rows.append((security, start, pd.Timestamp(last_date), source, start))
    return pd.DataFrame(rows, columns=["security_id", "start_date", "end_date", "source", "snapshot_date"])


def mismatch_report(
    snapshots: pd.DataFrame, reference: pd.DataFrame, matched_ids: set[str]
) -> pd.DataFrame:
    """Per snapshot date: members, those unmatched to the security master, and disagreement with a
    reference membership (e.g. index-fund holdings by filing date) on the same snapshot date."""
    rows = []
    for stamp, part in snapshots.groupby(SNAPSHOT_DATE):
        ours = set(part[SECURITY])
        ref = set(reference.loc[pd.to_datetime(reference[SNAPSHOT_DATE]) == pd.Timestamp(stamp), SECURITY])
        disagree = len(ours ^ ref) if ref else 0
        rows.append({
            SNAPSHOT_DATE: pd.Timestamp(stamp), "members": len(ours), "unmatched": len(ours - matched_ids),
            "reference_members": len(ref), "disagree": disagree,
        })
    return pd.DataFrame(rows)


def enforce_mismatch_limit(report: pd.DataFrame, limit: float) -> None:
    """Stop the build when too large a share of member-days is unmatched or disagrees (spec 3.3)."""
    member_days = float(report["members"].sum())
    bad = float((report["unmatched"] + report["disagree"]).sum())
    if member_days and bad / member_days > limit:
        raise DataQualityError(f"{bad / member_days:.1%} of member-days are unmatched or disagree (limit {limit:.0%})")
