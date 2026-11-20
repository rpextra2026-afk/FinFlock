"""Section 3 placeholders: three optional replacements, each behind a switch in ``clean.placeholders`` (default OFF).

A missing cache never fails the run: the function logs it and the switch runs off.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finfm.clean.steps import LANE_ATTENTION, LANE_COMPANY, LANE_WORLD, NODE_COMPANY, UNIT_ID
from finfm.common.config import Config

LOG = logging.getLogger(__name__)
ROOT = "clean.placeholders"


def _cache(cfg: Config, root: str, key: str) -> Path:
    return Path(root) / str(cfg.get(f"{ROOT}.cache_dir")) / str(cfg.get(f"{ROOT}.{key}"))


def _embeddings(cfg: Config, root: str) -> tuple[np.ndarray, dict[str, int]] | None:
    vec_path, id_path = _cache(cfg, root, "embeddings_file"), _cache(cfg, root, "embedding_ids_file")
    if not vec_path.exists() or not id_path.exists():
        LOG.warning("embedding cache missing (%s); the embedding placeholders run off", vec_path)
        return None
    ids = pd.read_parquet(id_path)[UNIT_ID].tolist()
    return np.load(vec_path).astype(np.float32), {u: i for i, u in enumerate(ids)}


def cause_repair_embed(items: pd.DataFrame, cfg: Config, root: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Nearest-neighbour cause for 'other' readings, from fit-window readings that have a real cause."""
    note: dict[str, Any] = {"switch": bool(cfg.get(f"{ROOT}.use_cause_repair_embed")), "ran": False}
    if not note["switch"]:
        return items, note
    emb = _embeddings(cfg, root)
    if emb is None:
        return items, note
    vectors, row = emb
    other = str(cfg.get("clean.causes.other_value"))
    asset = items["event_type_final"].eq("asset_event") & items[UNIT_ID].isin(row)
    pool = items[asset & items["cause_top"].ne(other) & items["cause_top"].notna() & items["split"].eq("fit")]
    todo = items[asset & items["cause_top"].eq(other)]
    if pool.empty or todo.empty:
        return items, note
    k = int(cfg.get(f"{ROOT}.neighbours_k"))
    cut = float(cfg.get(f"{ROOT}.neighbour_confidence_cut"))
    pool_vec = vectors[[row[u] for u in pool[UNIT_ID]]]
    pool_cause = pool["cause_top"].to_numpy()
    out = items.copy()
    repaired = 0
    chunk_rows = int(cfg.get(f"{ROOT}.chunk_rows"))
    for start in range(0, len(todo), chunk_rows):
        chunk = todo.iloc[start : start + chunk_rows]
        sims = vectors[[row[u] for u in chunk[UNIT_ID]]] @ pool_vec.T
        top = np.argpartition(-sims, k, axis=1)[:, :k]
        for i, idx in enumerate(chunk.index):
            votes: dict[str, float] = {}
            for j in top[i]:
                votes[pool_cause[j]] = votes.get(pool_cause[j], 0.0) + max(float(sims[i, j]), 0.0)
            best = max(votes, key=votes.get)
            if votes[best] / max(sum(votes.values()), float(np.finfo(float).eps)) >= cut:
                out.at[idx, "cause_top"] = best
                repaired += 1
    note.update({"ran": True, "other_rows": int(len(todo)), "repaired": repaired})
    return out, note


def world_themes_embed(items: pd.DataFrame, cfg: Config, root: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Embedding clusters (fit-window items without a company pin) replace the keyword themes for the world lane."""
    note: dict[str, Any] = {"switch": bool(cfg.get(f"{ROOT}.use_world_themes_embed")), "ran": False}
    if not note["switch"]:
        return items, note
    emb = _embeddings(cfg, root)
    if emb is None:
        return items, note
    from sklearn.cluster import KMeans

    vectors, row = emb
    no_pin = items["n_pins"].eq(0) & items[UNIT_ID].isin(row)
    fit = items[no_pin & items["split"].eq("fit")]
    if fit.empty:
        return items, note
    model = KMeans(int(cfg.get(f"{ROOT}.theme_clusters")), n_init=int(cfg.get("clean.sector.kmeans_n_init")), random_state=int(cfg.get(f"{ROOT}.theme_seed"))).fit(vectors[[row[u] for u in fit[UNIT_ID]]])
    out = items.copy()
    labels = model.predict(vectors[[row[u] for u in items.loc[no_pin, UNIT_ID]]])
    out.loc[no_pin, "themes"] = pd.Series([[f"emb_{k}"] for k in labels], index=items.index[no_pin])
    out["lane"] = np.where(out["n_pins"] > 0, LANE_COMPANY, np.where(out["themes"].map(len) > 0, LANE_WORLD, LANE_ATTENTION))
    note.update({"ran": True, "items_assigned": int(no_pin.sum())})
    return out, note


def direction_finbert(long: pd.DataFrame, cfg: Config, root: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Average of the reader's direction and the FinBERT tone (p_pos - p_neg), on company touch points."""
    note: dict[str, Any] = {"switch": bool(cfg.get(f"{ROOT}.use_direction_finbert")), "ran": False}
    if not note["switch"]:
        return long, note
    path = _cache(cfg, root, "finbert_file")
    if not path.exists():
        LOG.warning("FinBERT cache missing (%s); direction_finbert runs off", path)
        return long, note
    scores = pd.read_parquet(path)
    tone = (scores["p_positive"] - scores["p_negative"]).where((scores["p_positive"] - scores["p_negative"]).abs() > float(cfg.get(f"{ROOT}.finbert_neutral_band")), 0.0)
    tone_of = dict(zip(scores[UNIT_ID], tone, strict=True))
    out = long.copy()
    company = out["node_type"].eq(NODE_COMPANY)
    fb = out.loc[company, UNIT_ID].map(tone_of).fillna(0.0)
    out["direction_reader"] = out["direction"]
    out.loc[company, "direction"] = (out.loc[company, "direction"] + fb) / 2
    out["signed_weight"] = out["weight"] * out["direction"]
    note.update({"ran": True, "company_rows_with_tone": int(fb.ne(0.0).sum())})
    return out, note
