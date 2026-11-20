"""Institutional personas as clusters of real 13F filers (spec 6.4).

Per quarter and type the active (non-passive) managers are clustered by k-means on standardised
(log book, turnover, n_positions, first principal components of the sector tilt). A persona is a
cluster: its book is the sum of the books, its holdings the book-weighted aggregate weights, its
turnover and position count book-weighted means. Cluster labels are aligned to the previous quarter
by Hungarian matching on the centroids, so persona ``j`` stays the same kind of fund over time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA

from oryxis.common.config import Config

BOOK = "book_usd"
TURNOVER = "turnover"
N_POSITIONS = "n_positions"
PASSIVE_ADJACENT = "passive_adjacent"


@dataclass(frozen=True)
class FeatureSpace:
    """Standardisation and PCA fitted once (first quarter) and reused, so quarters are comparable."""

    mean: np.ndarray
    std: np.ndarray
    pca: PCA
    sectors: tuple[str, ...]


def _raw_features(managers: pd.DataFrame, tilt: pd.DataFrame, space_pca: PCA | None, sectors) -> tuple[np.ndarray, np.ndarray]:
    tilt_matrix = tilt.reindex(columns=list(sectors), fill_value=0.0).to_numpy()
    base = np.column_stack(
        [np.log(managers[BOOK].to_numpy()), managers[TURNOVER].to_numpy(), np.log1p(managers[N_POSITIONS].to_numpy())]
    )
    return base, tilt_matrix


def fit_feature_space(managers: pd.DataFrame, tilt: pd.DataFrame, cfg: Config) -> FeatureSpace:
    """Fit the PCA of the sector tilt and the feature standardisation on a reference quarter."""
    sectors = tuple(tilt.columns)
    n_pcs = min(int(cfg.get("personas.cluster.n_pcs")), len(sectors), len(managers))
    pca = PCA(n_components=n_pcs).fit(tilt.to_numpy())
    base, tilt_matrix = _raw_features(managers, tilt, pca, sectors)
    stacked = np.column_stack([base, pca.transform(tilt_matrix)])
    floor = float(cfg.get("personas.cluster.feature_floor"))
    return FeatureSpace(stacked.mean(axis=0), np.maximum(stacked.std(axis=0), floor), pca, sectors)


def features(space: FeatureSpace, managers: pd.DataFrame, tilt: pd.DataFrame) -> np.ndarray:
    base, tilt_matrix = _raw_features(managers, tilt, space.pca, space.sectors)
    stacked = np.column_stack([base, space.pca.transform(tilt_matrix)])
    return (stacked - space.mean) / space.std


def cluster_managers(
    matrix: np.ndarray, n_clusters: int, seed: int, n_init: int
) -> tuple[np.ndarray, np.ndarray]:
    """k-means labels and centroids; never more clusters than managers."""
    k = max(min(n_clusters, len(matrix)), 1)
    model = KMeans(n_clusters=k, n_init=n_init, random_state=seed).fit(matrix)
    return model.labels_, model.cluster_centers_


def align_labels(
    labels: np.ndarray, centroids: np.ndarray, previous: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Relabel clusters so centroid ``j`` is the one closest to the previous quarter's centroid ``j``."""
    cost = np.linalg.norm(previous[:, None, :] - centroids[None, :, :], axis=2)
    rows, cols = linear_sum_assignment(cost)
    mapping = {int(new): int(old) for old, new in zip(rows, cols, strict=True)}
    spare = max(mapping.values(), default=-1) + 1
    for new in range(len(centroids)):
        if new not in mapping:  # more clusters than before: new personas get fresh indices
            mapping[new] = spare
            spare += 1
    relabelled = np.array([mapping[int(label)] for label in labels])
    reordered = np.zeros((max(mapping.values()) + 1, centroids.shape[1]))
    for new, old in mapping.items():
        reordered[old] = centroids[new]
    return relabelled, reordered


def aggregate_clusters(
    managers: pd.DataFrame, holdings: pd.DataFrame, labels: np.ndarray
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Persona rows (book, turnover, positions, manager count, passive share) and holdings matrix.

    ``holdings`` is managers x securities with portfolio weights; the aggregate is book weighted.
    """
    frame = managers.assign(cluster=labels)
    rows, vectors = [], {}
    for cluster, part in frame.groupby("cluster"):
        books = part[BOOK].to_numpy()
        weights = books / books.sum()
        rows.append(
            {
                "cluster": int(cluster),
                BOOK: float(books.sum()),
                TURNOVER: float(weights @ part[TURNOVER].to_numpy()),
                N_POSITIONS: float(weights @ part[N_POSITIONS].to_numpy()),
                "n_managers": len(part),
                PASSIVE_ADJACENT: float(part[PASSIVE_ADJACENT].mean()) if PASSIVE_ADJACENT in part else 0.0,
            }
        )
        vectors[int(cluster)] = (holdings.loc[part.index].mul(weights, axis=0)).sum(axis=0)
    return pd.DataFrame(rows).set_index("cluster"), pd.DataFrame(vectors).T
