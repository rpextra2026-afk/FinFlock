"""Source dials from the reader record (spec 4.5): return, volatility change, volume change.

Return dial: ``s_ret = value_effect_pct / 100`` (log value units). Volatility and volume dials are maps from the
news features (cause probabilities, magnitude, |value effect|, certainty, kind), the stock's own recent vol and volume
ending on day -1, the market-state terciles and the reader's scheduled/surprise flag, fitted on dev against
``log_vol_ratio`` and ``log_dv_ratio`` per node class. The ridge map is the floor; an optional LightGBM arm competes under
the same time-blocked CV and is used only when it beats the ridge out of fold. Context companies without prices borrow the
small-layer map and are flagged ``source_map=borrowed``. Fitting receives labels as data; this module never imports the
labels store, and no feature reads day 0 or later.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from oryxis.common.config import Config
from oryxis.common.cv import time_blocked_folds
from oryxis.common.timing import TradingCalendar

CAUSE_PREFIX = "cause_"
KINDS = ("finance", "geo", "social")
TARGETS = {"vol": "log_vol_ratio", "volu": "log_dv_ratio"}


SCHEDULED_VALUES = ("scheduled", "surprise")  # "unknown" is the baseline of the scheduled/surprise dummies
STATE_BLOCKS = {
    "own_state": ("own_vol_trail", "own_volu_trail"),
    "market_state": ("mkt_vol_tercile", "mkt_volu_tercile"),
}


MISSING_SOURCE = "missing"  # source label of a value the model never wrote (oryxis.reader.salvage.SOURCE_MISSING)


def require_written(rows: pd.DataFrame, name: str) -> None:
    """Stop when a requested model-written feature is absent on some rows. Nothing is imputed here: either the block is removed from
    ``reader.dials.features`` (a named reader mode does this) or the rows carry a labelled fill (``certainty_source = filled``)."""
    absent = rows[name].isna().to_numpy() if name in rows.columns else np.ones(len(rows), dtype=bool)
    if f"{name}_source" in rows.columns:
        absent = absent | (rows[f"{name}_source"] == MISSING_SOURCE).to_numpy()
    if absent.any():
        raise ValueError(f"feature block {name!r} is requested but the model never wrote it on {int(absent.sum())} rows; "
                         f"drop it from reader.dials.features (reader mode salvaged_no_certainty) or use the labelled fill mode")


def feature_matrix(rows: pd.DataFrame, cfg: Config) -> np.ndarray:
    """Design matrix from the blocks named in ``reader.dials.features``. State columns may be NaN (no history): the map fills them
    with the training mean, which is zero after standardising. ``certainty`` / ``scheduled`` must have been written by the model (or
    carry a labelled fill); a missing value stops the run."""
    causes = [f"{CAUSE_PREFIX}{c}" for c in cfg.get("causes.causes")]
    blocks: list[np.ndarray] = []
    for name in cfg.get("reader.dials.features"):
        if name == "causes":
            blocks.append(rows[causes].to_numpy(dtype=float))
        elif name == "magnitude":
            blocks.append(rows[["magnitude"]].to_numpy(dtype=float))
        elif name == "abs_value_effect":
            blocks.append(rows[["value_effect_pct"]].abs().to_numpy(dtype=float))
        elif name == "certainty":
            require_written(rows, "certainty")
            blocks.append(rows[["certainty"]].to_numpy(dtype=float))
        elif name == "kind":
            blocks.append(np.column_stack([(rows["kind"] == k).to_numpy().astype(float) for k in KINDS]))
        elif name == "scheduled":
            if f"{name}_source" in rows.columns:
                require_written(rows.assign(scheduled=rows["scheduled"].where(rows["scheduled_source"] != MISSING_SOURCE)), "scheduled")
            blocks.append(np.column_stack([(rows["scheduled"] == v).to_numpy().astype(float) for v in SCHEDULED_VALUES]))
        elif name in STATE_BLOCKS:
            missing = [c for c in STATE_BLOCKS[name] if c not in rows.columns]
            if missing:
                raise KeyError(f"feature block {name!r} needs columns {missing}: add them with reader.state_features.add_state_features")
            blocks.append(rows[list(STATE_BLOCKS[name])].to_numpy(dtype=float))
        else:
            raise KeyError(f"unknown dial feature block {name!r}")
    return np.column_stack(blocks)


@dataclass(frozen=True)
class DialMap:
    """A fitted linear map; ``mean`` / ``scale`` standardise the features."""

    coef: np.ndarray
    intercept: float
    mean: np.ndarray
    scale: np.ndarray
    ridge: float
    n_rows: int

    def predict(self, X: np.ndarray) -> np.ndarray:
        filled = np.where(np.isnan(X), self.mean, X)
        return ((filled - self.mean) / self.scale) @ self.coef + self.intercept


class TreeMap:
    """A fitted gradient-boosted tree map (the optional LightGBM arm); same ``predict`` as ``DialMap``."""

    def __init__(self, model, n_rows: int, params: dict) -> None:
        self.model, self.n_rows, self.params = model, n_rows, params

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.asarray(self.model.predict(X), dtype=float)


def fit_tree(X: np.ndarray, y: np.ndarray, params: dict, seed: int) -> TreeMap:
    """Fit one LightGBM regressor. The package is imported here so the rest of the module works without it."""
    try:
        import lightgbm
    except ImportError as exc:
        raise RuntimeError("reader.dials.lgbm.enabled is true but the lightgbm package is not installed") from exc
    model = lightgbm.LGBMRegressor(random_state=seed, verbose=-1, **params)
    model.fit(X, y)
    return TreeMap(model, len(y), dict(params))


def fit_ridge(X: np.ndarray, y: np.ndarray, ridge: float) -> DialMap:
    column_mean = np.nanmean(np.where(np.isfinite(X), X, np.nan), axis=0)
    column_mean = np.where(np.isnan(column_mean), 0.0, column_mean)
    X = np.where(np.isnan(X), column_mean, X)  # a missing state value counts as an average one
    mean = X.mean(axis=0)
    scale = np.where(X.std(axis=0) > 0, X.std(axis=0), 1.0)
    Z = (X - mean) / scale
    center = y.mean()
    coef = np.linalg.solve(Z.T @ Z + ridge * np.eye(Z.shape[1]), Z.T @ (y - center))
    return DialMap(coef, float(center), mean, scale, ridge, len(y))


def _cv_loss(fit, X: np.ndarray, y: np.ndarray, folds) -> float:
    losses = []
    for train, val in folds:
        if len(train) < X.shape[1] + 1:
            continue
        model = fit(X[train], y[train])
        losses.append(float(np.mean((model.predict(X[val]) - y[val]) ** 2)))
    return float(np.mean(losses)) if losses else np.inf


def fit_dial_map(
    X: np.ndarray, y: np.ndarray, dates: pd.Series, calendar: TradingCalendar, cfg: Config
) -> DialMap | TreeMap:
    """Ridge chosen by 4-fold time-blocked CV with an embargo, then refit on all rows. When the LightGBM arm is on, its best
    grid point (same folds, same number of grid points) replaces the ridge only if it has a strictly lower CV loss."""
    spec = cfg.get("reader.dials")
    folds = time_blocked_folds(dates, int(spec["cv_folds"]), int(spec["embargo_sessions"]), calendar)
    best, best_loss = None, np.inf
    for ridge in spec["ridge_grid"]:
        loss = _cv_loss(lambda a, b, r=float(ridge): fit_ridge(a, b, r), X, y, folds)
        if loss < best_loss:
            best, best_loss = float(ridge), loss
    ridge_map = fit_ridge(X, y, best if best is not None else float(spec["ridge_grid"][-1]))
    tree = spec["lgbm"]
    if "lgbm" not in spec["arms"] or not tree["enabled"]:
        return ridge_map
    tree_best, tree_loss = None, np.inf
    for params in tree["grid"]:
        loss = _cv_loss(lambda a, b, p=dict(params): fit_tree(a, b, p, int(tree["seed"])), X, y, folds)
        if loss < tree_loss:
            tree_best, tree_loss = dict(params), loss
    if tree_best is not None and tree_loss < best_loss:
        return fit_tree(X, y, tree_best, int(tree["seed"]))
    return ridge_map


def fit_source_dial_maps(
    rows: pd.DataFrame, calendar: TradingCalendar, cfg: Config
) -> dict[tuple[str, str], DialMap | TreeMap]:
    """Maps per (node class, target). ``rows`` has the pin readings plus ``node_class``, ``day0`` and
    the label columns; classes with too few rows borrow the pooled map."""
    spec = cfg.get("reader.dials")
    pooled_X = feature_matrix(rows, cfg)
    maps: dict[tuple[str, str], DialMap | TreeMap] = {}
    for target, label in TARGETS.items():
        ok = rows[label].notna().to_numpy()
        pooled = fit_dial_map(pooled_X[ok], rows.loc[ok, label].to_numpy(), rows.loc[ok, "day0"], calendar, cfg)
        maps[("pooled", target)] = pooled
        for node_class in spec["node_classes"]:
            part = rows[(rows["node_class"] == node_class) & rows[label].notna()]
            if len(part) >= int(spec["min_rows_per_class"]):
                maps[(node_class, target)] = fit_dial_map(
                    feature_matrix(part, cfg), part[label].to_numpy(), part["day0"], calendar, cfg
                )
    return maps


def source_dials(
    pins: pd.DataFrame, maps: dict[tuple[str, str], DialMap | TreeMap], cfg: Config
) -> pd.DataFrame:
    """``ret, vol, volu`` for each pin reading and a flag saying which map produced it."""
    spec = cfg.get("reader.dials")
    out = pd.DataFrame({"ret": pins["value_effect_pct"].to_numpy() / float(spec["value_scale_pct"])}, index=pins.index)
    flags = pd.Series("own", index=pins.index)
    X = feature_matrix(pins, cfg)
    for target in TARGETS:
        values = np.zeros(len(pins))
        for position, node_class in enumerate(pins["node_class"]):
            key = (node_class, target)
            if node_class == "borrowed" or key not in maps:
                key = (str(spec["borrowed_from"]), target) if (str(spec["borrowed_from"]), target) in maps else ("pooled", target)
                flags.iloc[position] = "borrowed" if node_class == "borrowed" else "pooled"
            values[position] = maps[key].predict(X[position : position + 1])[0]
        out[target] = values
    out["source_map"] = flags
    return out
