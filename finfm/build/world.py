"""The world lane (spec section 4 and family I): market-level news features per theme against day-level volatility and volume targets.

Cheap and separate from the company sheaf: it reads the cleaned tape, builds one feature row per session, and scores it on the hold-out with the same
metrics code. Versions: every day (the default), burst days only (kept as a comparison), raw and day-demeaned targets, shifted and shuffled controls.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from finfm.build import metrics as M
from finfm.clean.steps import LANE_WORLD, NODE_THEME, returns_frame
from finfm.common.config import Config

ROOT = "build.world"
TARGETS = ("vol", "volu")


def daily_targets(prices: pd.DataFrame, cfg: Config) -> dict[str, pd.DataFrame]:
    """Stock-by-session targets: |market-adjusted return| (vol) and log volume against its trailing mean (volu), raw and day-demeaned."""
    rets = returns_frame(prices, cfg)
    vol = rets.abs()
    wide = prices.pivot(index="session_date", columns="security_id", values="volume_raw").reindex(rets.index)
    lv = np.log1p(wide)
    window, floor = int(cfg.get(f"{ROOT}.volume_window")), int(cfg.get(f"{ROOT}.volume_min_obs"))
    volu = lv - lv.shift(1).rolling(window, min_periods=floor).mean()
    out = {"vol_raw": vol, "volu_raw": volu}
    for name in TARGETS:
        frame = out[f"{name}_raw"]
        out[f"{name}_demeaned"] = frame.sub(frame.mean(axis=1), axis=0)
    return out


def theme_features(clean: pd.DataFrame, themes: Sequence[str], cfg: Config) -> pd.DataFrame:
    """One row per session: for every theme the item count, the weighted count, the share of world-lane items and an intensity (log of the weighted count)."""
    world = clean[clean["lane"] == LANE_WORLD]
    theme = world[world["node_type"] == NODE_THEME]
    sessions = pd.DatetimeIndex(sorted(world["session_date"].unique()))
    total = world.groupby("session_date")["unit_id"].nunique().reindex(sessions).fillna(0.0)
    cols: dict[str, pd.Series] = {}
    for name in themes:
        part = theme[theme["node_id"] == name]
        count = part.groupby("session_date")["unit_id"].nunique().reindex(sessions).fillna(0.0)
        weighted = part.groupby("session_date")["weight"].sum().reindex(sessions).fillna(0.0)
        cols[f"count_{name}"] = count
        cols[f"weighted_count_{name}"] = weighted
        cols[f"share_{name}"] = (count / total.where(total > 0, np.nan)).fillna(0.0)
        cols[f"intensity_{name}"] = np.log1p(weighted)
    return pd.DataFrame(cols)


def _standardise(x: pd.DataFrame, fit_days: pd.DatetimeIndex) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    block = x.reindex(fit_days).fillna(0.0)
    mean, sd = block.mean(), block.std().replace(0.0, 1.0)
    return (x - mean) / sd, mean, sd


def _ridge(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[np.ndarray, float]:
    xm, ym = x.mean(0), y.mean()
    xc, yc = x - xm, y - ym
    beta = np.linalg.solve(xc.T @ xc + alpha * len(x) * np.eye(x.shape[1]), xc.T @ yc)
    return beta, float(ym - xm @ beta)


def fit_predict(features: pd.DataFrame, daily_target: pd.Series, fit_days: pd.DatetimeIndex, test_days: pd.DatetimeIndex, cfg: Config, burst: set[pd.Timestamp] | None = None, test_features: pd.DataFrame | None = None) -> tuple[pd.Series, np.ndarray, float, pd.Index]:
    """Ridge on the daily mean target. The strength is picked on the last fit year, then refitted on all fit days; returns the test prediction, the coefficients and the strength."""
    x = features.copy()
    if burst is not None:
        keep = pd.Series(x.index.isin(list(burst)), index=x.index)
        x = x.mul(keep, axis=0)
    z, mean, sd = _standardise(x, fit_days)
    tx = z if test_features is None else ((test_features.mul(pd.Series(test_features.index.isin(list(burst)), index=test_features.index), axis=0) if burst is not None else test_features) - mean) / sd
    fit = z.reindex(fit_days).fillna(0.0)
    y = daily_target.reindex(fit_days)
    ok = y.notna().to_numpy()
    last = fit_days[-1].year
    val = np.array([d.year == last for d in fit_days]) & ok
    trn = np.array([d.year != last for d in fit_days]) & ok
    best = None
    for alpha in cfg.get(f"{ROOT}.ridge_grid"):
        if trn.sum() < 2 or val.sum() < 2:
            best = (0.0, float(alpha))
            break
        beta, icpt = _ridge(fit.to_numpy()[trn], y.to_numpy()[trn], float(alpha))
        loss = float(((y.to_numpy()[val] - (fit.to_numpy()[val] @ beta + icpt)) ** 2).sum())
        best = (loss, float(alpha)) if best is None or loss < best[0] else best
    beta, icpt = _ridge(fit.to_numpy()[ok], y.to_numpy()[ok], best[1])
    pred = pd.Series(tx.reindex(test_days).fillna(0.0).to_numpy() @ beta + icpt, index=test_days)
    return pred, beta, best[1], z.columns


def shifted(frame: pd.DataFrame, days: pd.DatetimeIndex, k: int) -> pd.DataFrame:
    """The features of the hold-out days moved by ``k`` sessions inside the hold-out (circular), so no day outside the window is read."""
    block = frame.reindex(days).fillna(0.0)
    moved = np.roll(block.to_numpy(), k, axis=0)
    return pd.DataFrame(moved, index=days, columns=block.columns)


def stock_frame(pred_day: pd.Series, target: pd.DataFrame, base: float) -> pd.DataFrame:
    """Long frame (y, pred, base, day, node): every stock of a session gets the session's market-level prediction."""
    block = target.reindex(pred_day.index)
    long = block.stack().rename("y").reset_index()
    long.columns = ["day", "node", "y"]
    long["pred"] = long["day"].map(pred_day)
    long["base"] = base
    return long.dropna()


def evaluate_world(collector: M.Collector, clean: pd.DataFrame, prices: pd.DataFrame, run_cfg: Mapping[str, str], themes: Sequence[str], cfg: Config, rng: np.random.Generator) -> pd.DataFrame:
    """Writes the world-lane rows of family I and returns the per-theme coefficient table."""
    targets = daily_targets(prices, cfg)
    feats = theme_features(clean, themes, cfg)
    sessions = targets["vol_raw"].index
    fit_days = sessions[(sessions >= pd.Timestamp(run_cfg["fit_start"])) & (sessions <= pd.Timestamp(run_cfg["fit_end"]))]
    test_days = sessions[(sessions >= pd.Timestamp(run_cfg["holdout_start"])) & (sessions <= pd.Timestamp(run_cfg["holdout_end"]))]
    if len(fit_days) < int(cfg.get(f"{ROOT}.min_days")) or not len(test_days):
        return pd.DataFrame()
    feats = feats.reindex(sessions).fillna(0.0)
    total = feats[[c for c in feats.columns if c.startswith("weighted_count_")]].sum(axis=1)
    thr = float(np.quantile(total.reindex(fit_days), float(M.mcfg(cfg, "burst_quantile"))))
    bursts = set(total.index[total >= thr]) if thr > 0 else set()
    versions = {"every_day": None, "burst_only": bursts}
    coefs = []
    for name in TARGETS:
        for tkind in ("raw", "demeaned"):
            target = targets[f"{name}_{tkind}"]
            daily = target.mean(axis=1)
            base = float(target.reindex(fit_days).stack().mean())
            for vname, burst in versions.items():
                pred, beta, alpha, cols = fit_predict(feats, daily, fit_days, test_days, cfg, burst)
                variants = {"real": None, **{f"shift_{int(k)}_sessions": int(k) for k in M.mcfg(cfg, "placebo_shifts")}}
                for pname, k in variants.items():
                    if k is None:
                        p = pred
                    else:
                        p, *_ = fit_predict(feats, daily, fit_days, test_days, cfg, burst, test_features=shifted(feats, test_days, k))
                    f = stock_frame(p, target, base)
                    sd = float(f["y"].std())
                    res = M.all_metrics(f, name, sd, sd, np.full(len(f), M.HALF), cfg, rng, draws=int(M.mcfg(cfg, "slice_bootstrap_draws")), full=False)
                    M.write_metrics(collector, name, f"world_{tkind}_{vname}", pname, {k2: res[k2] for k2 in ("r2_os", "gain_ci_day", "gain_ci_block", "dm_p", "cw_p", "slope", "slope_t_clustered", "rmse", "mae")}, len(f), "world_lane", "all")
                if vname == "every_day":
                    coefs.append(pd.DataFrame({"target": name, "kind": tkind, "feature": cols, "coefficient": beta, "alpha": alpha}))
                    z = (feats - feats.reindex(fit_days).mean()) / feats.reindex(fit_days).std().replace(0.0, 1.0)
                    inten = z[[c for c in z.columns if c.startswith("intensity_")]].sum(axis=1)
                    for lag in range(int(cfg.get(f"{ROOT}.lag_range")[0]), int(cfg.get(f"{ROOT}.lag_range")[1]) + 1):
                        y_lag = daily.shift(-lag).reindex(test_days)
                        x_now = inten.reindex(test_days)
                        ok = y_lag.notna() & np.isfinite(x_now)
                        if ok.sum() > int(M.MIN_POINTS) and x_now[ok].std() > 0:
                            slope = float(np.polyfit(x_now[ok], y_lag[ok], 1)[0])
                            collector.add(name, f"world_lag_profile_{tkind}", lag, "slope_per_intensity_sd", slope * float(x_now[ok].std()), n=int(ok.sum()), method="world_lane", view="all")
    inten_year = feats[[c for c in feats.columns if c.startswith("count_")]].groupby(feats.index.year).sum()
    for year, row in inten_year.iterrows():
        tot = float(row.sum())
        for col, v in row.items():
            collector.add("all", "world_theme_share_by_year", f"{year}|{col.replace('count_', '')}", "share", v / tot if tot > 0 else float("nan"), n=int(tot), method="world_lane", view="all")
    return pd.concat(coefs, ignore_index=True) if coefs else pd.DataFrame()
