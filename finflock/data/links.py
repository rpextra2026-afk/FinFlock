"""Link snapshot builders for the road-1 graph (spec 3.5).

Each builder returns rows of the ``link`` table for ONE snapshot date: ``available_from`` is that
date and ``valid_to`` is filled when snapshots are stacked (the next snapshot's date). A link is never
used before ``available_from``. Node types are not stored; ``src_traded`` / ``dst_traded`` say whether
an endpoint is an S&P 500 member on the snapshot date.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from finflock.common.config import Config

SIC_WIDTH = 4  # SIC codes are four digits; shorter ones are left-padded
COLUMNS = ["src_node", "dst_node", "link_type", "available_from", "valid_to", "weight", "source",
           "src_traded", "dst_traded", "carried_forward"]


def _rows(pairs, link_type, snapshot, source, traded, carried=False) -> pd.DataFrame:
    frame = pd.DataFrame(pairs, columns=["src_node", "dst_node", "weight"])
    if frame.empty:
        return pd.DataFrame(columns=COLUMNS)
    frame["link_type"] = link_type
    frame["available_from"] = pd.Timestamp(snapshot)
    frame["valid_to"] = pd.NaT
    frame["source"] = source
    frame["src_traded"] = frame["src_node"].isin(traded)
    frame["dst_traded"] = frame["dst_node"].isin(traded)
    frame["carried_forward"] = carried
    return frame[COLUMNS]


def industry_links(sic: pd.Series, mcap: pd.Series, snapshot, traded: set[str], cfg: Config) -> pd.DataFrame:
    """Same 3-digit SIC; each node links to the ``neighbours_per_node`` closest in log market cap.

    ``sic`` is the SIC printed on the latest filing at or before the snapshot (never the SEC's current
    SIC). Nodes without a market cap fall back to a deterministic id order inside their group.
    """
    digits = int(cfg.get("links.industry.sic_digits"))
    k = int(cfg.get("links.industry.neighbours_per_node"))
    weight = float(cfg.get("links.weights.industry"))
    pairs = set()
    code = sic.astype(str).str.zfill(SIC_WIDTH).str[:digits]
    for _, members in code.groupby(code):
        ids = sorted(members.index)
        size = np.log(mcap.reindex(ids).fillna(mcap.median() if mcap.notna().any() else 1.0).clip(lower=1.0))
        for node in ids:
            nearest = (size - size[node]).abs().drop(node).sort_values(kind="mergesort").index[:k]
            pairs.update((min(node, other), max(node, other)) for other in nearest)
    return _rows([(a, b, weight) for a, b in sorted(pairs)], "industry", snapshot, "sic", traded)


def rival_links(similarity: pd.DataFrame, snapshot, last_file_date, traded: set[str], cfg: Config) -> pd.DataFrame:
    """Text-based rival similarity (``src, dst, score``); links after the last file are carried forward."""
    carried = pd.Timestamp(snapshot) > pd.Timestamp(last_file_date) and bool(cfg.get("links.rival.carry_forward_flag"))
    pairs = [(min(a, b), max(a, b), float(s)) for a, b, s in similarity[["src", "dst", "score"]].itertuples(index=False)]
    return _rows(sorted(set(pairs)), "rival", snapshot, "tnic", traded, carried)


def correlation_links(returns: pd.DataFrame, snapshot, traded: set[str], cfg: Config) -> pd.DataFrame:
    """Top-k past-correlation neighbours per stock from the last ``window_sessions`` returns before ``snapshot``."""
    window = int(cfg.get("links.corr.window_sessions"))
    k = int(cfg.get("links.corr.top_k"))
    past = returns.loc[: pd.Timestamp(snapshot) - pd.Timedelta(days=1)].iloc[-window:]
    past = past.dropna(axis=1, thresh=int(float(cfg.get("links.corr.min_coverage")) * len(past)))
    corr = past.corr().to_numpy()
    np.fill_diagonal(corr, 0.0)
    names = list(past.columns)
    pairs = set()
    for i, node in enumerate(names):
        for j in np.argsort(-np.abs(np.nan_to_num(corr[i])), kind="stable")[:k]:
            pairs.add((min(node, names[j]), max(node, names[j]), float(abs(corr[i, j]))))
    return _rows(sorted(pairs), "corr", snapshot, "returns", traded)


def owner_links(weights: pd.DataFrame, books: pd.Series, snapshot, traded: set[str], cfg: Config) -> pd.DataFrame:
    """Shared-owner links: ``w_ij = sum_k a_k x_ki x_kj / sum_k a_k x_ki`` ranked to ``[0, 1]`` per row.

    ``weights`` is managers x securities (portfolio weights), ``books`` the manager books.
    """
    k = int(cfg.get("links.owner.top_k"))
    book = books.reindex(weights.index).to_numpy()
    X = weights.fillna(0.0).to_numpy()
    co = (X * book[:, None]).T @ X
    denominator = (X * book[:, None]).sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        w = np.where(denominator[:, None] > 0, co / denominator[:, None], 0.0)
    np.fill_diagonal(w, 0.0)
    names = list(weights.columns)
    pairs = {}
    for i, node in enumerate(names):
        order = np.argsort(-w[i], kind="stable")[:k]
        order = [j for j in order if w[i, j] > 0]
        for rank, j in enumerate(order):
            key = (min(node, names[j]), max(node, names[j]))
            pairs[key] = max(pairs.get(key, 0.0), 1.0 - rank / max(len(order), 1))
    return _rows([(a, b, s) for (a, b), s in sorted(pairs.items())], "owner", snapshot, "13f", traded)


def table_links(table: pd.DataFrame, link_type: str, snapshot, traded: set[str], cfg: Config) -> pd.DataFrame:
    """Hand tables (country, commodity, ETF membership): ``src_node, dst_node, weight`` with defaults."""
    default = float(cfg.get(f"links.weights.{link_type}"))
    rows = [(a, b, float(w) if pd.notna(w) else default) for a, b, w in table[["src_node", "dst_node", "weight"]].itertuples(index=False)]
    return _rows(rows, link_type, snapshot, "table", traded)


def stack_snapshots(snapshots: list[pd.DataFrame]) -> pd.DataFrame:
    """Concatenate snapshots of one link type; each is valid until the next snapshot's date."""
    parts = sorted((s for s in snapshots if len(s)), key=lambda s: s["available_from"].iloc[0])
    for current, following in zip(parts, parts[1:], strict=False):
        current["valid_to"] = following["available_from"].iloc[0]
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=COLUMNS)
