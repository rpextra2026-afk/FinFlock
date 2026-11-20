"""Company links for FinRipple-lite and the road-1 graph: SIC peers, past-return correlation and shared 13F owners.

One snapshot per quarter, dated when that quarter's 13F filings are known (``personas.panel.lag_days``), so a link is never used
before it could have been built. Supplier, customer and rival links are not available from free data and are not built; this is stated
in the paper. SIC codes are today's codes (see ``finfm.data.sic``).
"""

from __future__ import annotations

import pandas as pd

from finfm.common.config import Config
from finfm.data.links import correlation_links, industry_links, owner_links, stack_snapshots


def snapshot_dates(periods: list[pd.Timestamp], cfg: Config) -> list[pd.Timestamp]:
    lag = pd.Timedelta(days=int(cfg.get("personas.panel.lag_days")))
    return [pd.Timestamp(p) + lag for p in sorted(periods)]


def build_links(
    sic: pd.Series, cap_proxy: pd.DataFrame, returns: pd.DataFrame, managers: pd.DataFrame, holdings: pd.DataFrame,
    members_on, cfg: Config,
) -> pd.DataFrame:
    """Stacked industry, correlation and owner links over all quarterly snapshots."""
    weights = holdings.merge(managers[["accession", "period", "book_usd"]], on="accession")
    industry, corr, owner = [], [], []
    for period in sorted(managers["period"].unique()):
        snapshot = pd.Timestamp(period) + pd.Timedelta(days=int(cfg.get("personas.panel.lag_days")))
        traded = set(members_on(snapshot))
        known = [s for s in traded if s in sic.index]
        cap = cap_proxy.loc[:snapshot].iloc[-1] if len(cap_proxy.loc[:snapshot]) else pd.Series(dtype=float)
        industry.append(industry_links(sic.reindex(known).dropna(), cap.reindex(known), snapshot, traded, cfg))
        corr.append(correlation_links(returns[[s for s in known if s in returns.columns]], snapshot, traded, cfg))
        part = weights[weights["period"] == period]
        if len(part):
            matrix = part.pivot_table(index="accession", columns="security_id", values="weight", aggfunc="sum").fillna(0.0)
            books = part.drop_duplicates("accession").set_index("accession")["book_usd"]
            owner.append(owner_links(matrix, books, snapshot, traded, cfg))
    return pd.concat([stack_snapshots(industry), stack_snapshots(corr), stack_snapshots(owner)], ignore_index=True)
