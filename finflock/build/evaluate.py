"""Evaluation of one tape: views (linked, all, unlinked), methods, slices (family F) and placebos (family G), written as the long metrics table."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats

from finflock.build import metrics as M
from finflock.common.config import Config

ROOT = "build"
DIALS = ("ret", "vol", "volu")
VIEW_LINKED, VIEW_ALL, VIEW_UNLINKED = "linked", "all", "unlinked"
PLACEBO_SLICE = "placebo"


def evaluation_frame(df: pd.DataFrame, method: str, dial: str, base: float) -> pd.DataFrame:
    """Columns y, pred, base, day, node for one dial of a rows or panel table. The forecast is the baseline (training mean) plus the method's ripple."""
    part = df[df["dial"] == dial]
    return pd.DataFrame({"y": part["y"].to_numpy(float), "pred": base + part[f"pred_{method}"].to_numpy(float), "base": np.full(len(part), base), "day": part["day"].to_numpy(), "node": part["node"].to_numpy()}, index=part.index)


def probability(ripple: np.ndarray, sd: float) -> np.ndarray:
    """Gaussian predictive probability of an outcome above the baseline (training residual sd, no fitting on the evaluation rows)."""
    return stats.norm.cdf(ripple / sd)


def add_slice_columns(rows: pd.DataFrame, ctx: Mapping[str, Any], cfg: Config) -> pd.DataFrame:
    """Columns the slices use. Edges and thresholds come from the fit window or the config, never from the rows being scored."""
    out = rows.copy()
    day = pd.to_datetime(out["day"])
    names = list(cfg.get(f"{ROOT}.slices.weekdays"))
    out["s_weekday"] = day.dt.weekday.map(lambda i: names[i])
    out["s_month"] = day.dt.month.astype(str)
    out["s_year"] = day.dt.year.astype(str)
    strongest = out[["w_industry", "w_corr", "w_owner"]].to_numpy()
    out["s_link_type"] = np.array(["industry", "corr", "owner"])[strongest.argmax(1)]
    out["s_link_type"] = np.where(strongest.max(1) > 0, out["s_link_type"], "none")
    nonowner = np.maximum(out["w_industry"], out["w_corr"])
    edges = ctx.get("weight_edges")
    out["s_weight_decile"] = np.where(nonowner > 0, np.searchsorted(edges, nonowner, side="right").astype(str), "none") if edges is not None else "none"
    pinned = list(cfg.get(f"{ROOT}.slices.pinned_edges"))
    out["s_pinned_neighbours"] = np.searchsorted(pinned, out["n_pinned_neighbours"].to_numpy(), side="right").astype(str)
    rep = list(cfg.get(f"{ROOT}.slices.repeat_edges"))
    out["s_repeat_bucket"] = np.searchsorted(rep, out["repeat_count"].to_numpy(), side="right").astype(str)
    for src, dst in (("cause_group", "s_cause_group"), ("kind", "s_kind"), ("session_type", "s_session_type")):
        out[dst] = out[src].astype(str)
    out["s_sector"] = out["node"].map(ctx.get("sector_of", {})).fillna("unknown").astype(str)
    out["s_size_bucket"] = out.set_index(["day", "node"]).index.map(ctx.get("size_bucket", {})).fillna("na").astype(str) if ctx.get("size_bucket") is not None else "na"
    out["s_liquidity_bucket"] = out.set_index(["day", "node"]).index.map(ctx.get("liquidity_bucket", {})).fillna("na").astype(str) if ctx.get("liquidity_bucket") is not None else "na"
    reg = ctx.get("high_vol_days", set())
    out["s_regime"] = np.where(day.isin(reg), "high_market_vol", "low_market_vol")
    world = ctx.get("world_days", set())
    burst = ctx.get("burst_days", set())
    out["s_day_type"] = np.where(day.isin(burst), "burst_day", np.where(day.isin(world), "world_news_day", "quiet_day"))
    return out


SLICE_COLUMNS = ["s_link_type", "s_weight_decile", "s_cause_group", "s_kind", "s_repeat_bucket", "s_session_type", "s_weekday", "s_month", "s_year", "s_regime", "s_sector", "s_size_bucket", "s_liquidity_bucket", "s_pinned_neighbours", "s_day_type"]


def placebo_rows(rows: pd.DataFrame, panel: pd.DataFrame, method: str, dial: str, base: float, collector: M.Collector, cfg: Config, rng: np.random.Generator, holdout_days: Sequence[pd.Timestamp]) -> None:
    """Family G on the linked rows: shifted news, shuffled readings, shuffled predictions, a random node on the same day. Each is next to the real gain."""
    part = rows[(rows["dial"] == dial) & (rows["has_industry"] | rows["has_corr"])].reset_index(drop=True)
    if len(part) < int(M.mcfg(cfg, "min_rows_per_slice")):
        return
    lab = panel[panel["dial"] == dial].set_index(["day", "node"])["y"]
    days = np.array(sorted(set(pd.to_datetime(panel["day"]))))
    didx = {d: i for i, d in enumerate(days)}
    pred = base + part[f"pred_{method}"].to_numpy(float)
    y = part["y"].to_numpy(float)
    real = 1.0 - float(((y - pred) ** 2).sum() / ((y - base) ** 2).sum())
    collector.add(dial, PLACEBO_SLICE, "real", "gain", real, method=method, view=VIEW_LINKED, n=len(part))

    def gain_with(y_alt: np.ndarray, p_alt: np.ndarray) -> float:
        ok = np.isfinite(y_alt)
        if not ok.any():
            return float("nan")
        return 1.0 - float(((y_alt[ok] - p_alt[ok]) ** 2).sum() / ((y_alt[ok] - base) ** 2).sum())

    def labels_at(day_values: np.ndarray, nodes: np.ndarray) -> np.ndarray:
        idx = pd.MultiIndex.from_arrays([day_values, nodes])
        return lab.reindex(idx).to_numpy(float)

    p_day = pd.to_datetime(part["day"]).to_numpy()
    nodes = part["node"].to_numpy()
    for shift in M.mcfg(cfg, "placebo_shifts"):
        moved = np.array([days[(didx[d] + int(shift)) % len(days)] if d in didx else d for d in p_day])
        collector.add(dial, PLACEBO_SLICE, f"shift_{int(shift)}_sessions", "gain", gain_with(labels_at(moved, nodes), pred), method=method, view=VIEW_LINKED, n=len(part))
    draws = int(M.mcfg(cfg, "placebo_shuffles"))
    controls: dict[str, list[float]] = {"shuffle_days": [], "shuffle_events_nodes": [], "shuffle_links": [], "random_node_same_day": []}
    event_codes = pd.factorize(part["event_id"])[0]
    for _ in range(draws):
        per_event = days[rng.integers(0, len(days), size=event_codes.max() + 1)]
        controls["shuffle_days"].append(gain_with(labels_at(per_event[event_codes], nodes), pred))
        random_nodes = nodes[rng.permutation(len(nodes))]
        controls["shuffle_events_nodes"].append(gain_with(labels_at(p_day, random_nodes), pred))
        permuted = pred.copy()
        for code in np.unique(event_codes):
            where = np.flatnonzero(event_codes == code)
            permuted[where] = pred[rng.permutation(where)]
        controls["shuffle_links"].append(gain_with(y, permuted))
        same_day = nodes.copy()
        for d in np.unique(p_day):
            where = np.flatnonzero(p_day == d)
            same_day[where] = nodes[rng.permutation(where)]
        controls["random_node_same_day"].append(gain_with(labels_at(p_day, same_day), pred))
    level = float(M.mcfg(cfg, "ci_level"))
    for name, values in controls.items():
        arr = np.array([v for v in values if np.isfinite(v)])
        if len(arr):
            collector.add(dial, PLACEBO_SLICE, name, "gain_mean", float(arr.mean()), method=method, view=VIEW_LINKED, n=len(arr))
            collector.add(dial, PLACEBO_SLICE, name, "gain_quantile", float(np.quantile(arr, level)), method=method, view=VIEW_LINKED, n=len(arr))
            collector.add(dial, PLACEBO_SLICE, name, "share_matching_or_beating_real", float((arr >= real).mean()), method=method, view=VIEW_LINKED, n=len(arr))


def evaluate_tape(collector: M.Collector, rows: pd.DataFrame, panel: pd.DataFrame, methods: Sequence[str], main_method: str, train_mean: Mapping[str, float], scale: Mapping[str, float], resid_sd: Mapping[str, Mapping[str, float]], ctx: Mapping[str, Any], cfg: Config, rng: np.random.Generator, holdout_days: Sequence[pd.Timestamp]) -> None:
    """Every family of the spec on the hold-out rows: methods x views x dials, slices and placebos for the main method."""
    rows = add_slice_columns(rows, ctx, cfg)
    linked = rows[rows["has_industry"] | rows["has_corr"]]
    views = {VIEW_LINKED: (linked, "pred"), VIEW_ALL: (panel, "pred"), VIEW_UNLINKED: (panel[~panel["linked_today"]], "pred")}
    min_rows = int(M.mcfg(cfg, "min_rows_per_slice"))
    for dial in DIALS:
        base = float(train_mean[dial])
        for method in methods:
            for view, (df, _) in views.items():
                f = evaluation_frame(df, method, dial, base)
                if len(f) < min_rows:
                    continue
                prob = probability(f["pred"].to_numpy() - base, float(resid_sd[method][dial]))
                res = M.all_metrics(f, dial, float(scale[dial]), float(resid_sd[method][dial]), prob, cfg, rng)
                M.write_metrics(collector, dial, "view", view, res, len(f), method, view)
        for column in SLICE_COLUMNS:
            for value, part in linked[linked["dial"] == dial].groupby(rows.loc[linked.index, column] if column in rows else column):
                if len(part) < min_rows:
                    continue
                f = evaluation_frame(part, main_method, dial, base)
                prob = probability(f["pred"].to_numpy() - base, float(resid_sd[main_method][dial]))
                res = M.all_metrics(f, dial, float(scale[dial]), float(resid_sd[main_method][dial]), prob, cfg, rng, draws=int(M.mcfg(cfg, "slice_bootstrap_draws")))
                M.write_metrics(collector, dial, column[2:], value, res, len(f), main_method, VIEW_LINKED)
        placebo_rows(rows, panel, main_method, dial, base, collector, cfg, rng, holdout_days)
    tested = [(r.dial, r.method, r.view, r.metric, r.value) for r in collector.frame().itertuples() if r.metric in ("dm_p", "cw_p", "slope_p") and r.slice == "view"]
    holm, bh = M.multiple_comparison([x[-1] for x in tested])
    for (dial, method, view, metric, raw), h, b in zip(tested, holm, bh, strict=True):
        collector.add(dial, "multiple_comparison", f"{metric}_raw", "p_value", float(raw), method=method, view=view)
        collector.add(dial, "multiple_comparison", f"{metric}_holm", "p_adjusted", float(h), method=method, view=view)
        collector.add(dial, "multiple_comparison", f"{metric}_benjamini_hochberg", "p_adjusted", float(b), method=method, view=view)
