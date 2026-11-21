"""Tier A of the ablation: forecast-level numbers on the identical test events (spec section 8.1). Numbers only; nothing here compares or decides.

One row per (event, asset): the predicted cumulative increment over the first h days and the realised return over the same days, the 5-day return in
excess of the prior-year market beta, whether the asset is pinned by the reader (else linked), and the cuts. ``summarise`` turns that table into the
rank IC and the linked-stock out-of-sample R2 and slope, overall and by cut.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from finflock.common.config import Config
from finflock.eval.signal_metrics import ic_report
from finflock.framework.run import EventForecast

PRED, REAL, ABNORMAL = "pred_{h}", "real_{h}", "abnormal_{h}"
TERCILES = 3


def _cumulative(increments: np.ndarray, horizon: int) -> np.ndarray:
    return np.asarray(increments)[:horizon].sum(axis=0)


def _realised(ret: pd.DataFrame, day0: int, assets: np.ndarray, horizon: int) -> np.ndarray:
    window = ret.iloc[day0 + 1 : day0 + 1 + horizon, assets]
    if len(window) < horizon:
        return np.full(len(assets), np.nan)
    return (1.0 + window.fillna(0.0)).prod().to_numpy() - 1.0


def _betas(ret: pd.DataFrame, market: pd.Series, day0: int, assets: np.ndarray, window: int, min_fraction: float) -> np.ndarray:
    """Market beta over the ``window`` sessions before day 0 (day 0 itself excluded); NaN when fewer than ``min_fraction`` of the days are observed."""
    start = max(day0 - window, 0)
    r = ret.iloc[start:day0, assets]
    m = market.iloc[start:day0]
    ok = r.notna().sum() >= min_fraction * window
    centred_m = m - m.mean()
    var = float((centred_m**2).mean())
    cov = ((r.sub(r.mean())).mul(centred_m, axis=0)).mean()
    beta = (cov / var).to_numpy() if var > 0 else np.full(len(assets), np.nan)
    return np.where(ok.to_numpy(), beta, np.nan)


def event_table(forecasts: list[EventForecast], readings: dict, kinds: dict[str, str], ret: pd.DataFrame, market: pd.Series, cap_proxy: pd.DataFrame,
                linked: dict[str, set[int]], cfg: Config) -> pd.DataFrame:
    """The long table of one run. ``linked[event]`` is the run-independent set of global asset indices in the event's subgraph that are not pinned."""
    horizons = [int(h) for h in cfg.get("ablation.rank_ic_horizons")]
    window, min_fraction = int(cfg.get("ablation.beta_window_sessions")), float(cfg.get("ablation.beta_min_fraction"))
    longest = max(horizons)
    frames = []
    for f in forecasts:
        assets = np.asarray(f.assets)
        if f.day0 + longest >= len(ret):
            continue
        pinned_nodes = {p["node_id"] for p in readings[f.event_id].pins}
        names = ret.columns[assets]
        beta = _betas(ret, market, f.day0, assets, window, min_fraction)
        frame = pd.DataFrame({"event_id": f.event_id, "asset": assets, "pinned": [n in pinned_nodes for n in names],
                              "linked": [int(a) in linked.get(f.event_id, set()) for a in assets]})
        for h in horizons:
            realised = _realised(ret, f.day0, assets, h)
            frame[PRED.format(h=h)] = _cumulative(f.increments, h)
            frame[REAL.format(h=h)] = realised
            frame[ABNORMAL.format(h=h)] = realised - beta * float((1.0 + market.iloc[f.day0 + 1 : f.day0 + 1 + h].fillna(0.0)).prod() - 1.0)
        size = cap_proxy.iloc[f.day0, assets]
        frame["size_layer"] = pd.qcut(size.rank(method="first"), TERCILES, labels=False) if size.notna().sum() >= TERCILES else np.nan
        frame["kind"] = kinds[f.event_id]
        frame["n_pins"] = len(readings[f.event_id].pins)
        frame["year"] = int(ret.index[f.day0].year)
        frame["day0"] = ret.index[f.day0]
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def per_event(table: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """One row per event: its rank IC for each horizon and the sums behind the linked-stock fit (so differences between runs can be bootstrapped over events)."""
    minimum = int(cfg.get("ablation.event_min_assets"))
    horizons = [int(h) for h in cfg.get("ablation.rank_ic_horizons")]
    rows = []
    for event_id, part in table.groupby("event_id", sort=False):
        row = {"event_id": event_id, "day0": part["day0"].iloc[0], "kind": part["kind"].iloc[0], "n_pins": int(part["n_pins"].iloc[0]),
               "year": int(part["year"].iloc[0]), "size_layer": part.loc[part["pinned"], "size_layer"].mean() if part["pinned"].any() else np.nan}
        for h in horizons:
            pred, real = part[PRED.format(h=h)], part[REAL.format(h=h)]
            keep = pred.notna() & real.notna()
            row[f"ic_{h}d"] = float(spearmanr(pred[keep], real[keep])[0]) if keep.sum() >= minimum and pred[keep].nunique() > 1 and real[keep].nunique() > 1 else np.nan
            linked = part[part["linked"] & ~part["pinned"]]
            y, p = linked[ABNORMAL.format(h=h)].to_numpy(), linked[PRED.format(h=h)].to_numpy()
            ok = np.isfinite(y) & np.isfinite(p)
            y, p = y[ok], p[ok]
            row.update({f"linked_n_{h}d": float(ok.sum()), f"linked_sse_{h}d": float(((y - p) ** 2).sum()), f"linked_sst_{h}d": float((y**2).sum()),
                        f"linked_spy_{h}d": float((p * y).sum()), f"linked_spp_{h}d": float((p**2).sum())})
        rows.append(row)
    return pd.DataFrame(rows)


def linked_fit(events: pd.DataFrame, h: int) -> dict[str, float]:
    """Out-of-sample R2 of the linked (non-pinned) assets against a forecast of zero, and the slope of the realised excess return on the forecast, pooled over events."""
    sse, sst = float(events[f"linked_sse_{h}d"].sum()), float(events[f"linked_sst_{h}d"].sum())
    spy, spp = float(events[f"linked_spy_{h}d"].sum()), float(events[f"linked_spp_{h}d"].sum())
    return {f"linked_r2_{h}d": 1.0 - sse / sst if sst > 0 else float("nan"), f"linked_slope_{h}d": spy / spp if spp > 0 else float("nan"),
            f"n_linked_rows_{h}d": float(events[f"linked_n_{h}d"].sum())}


def summarise(events: pd.DataFrame, cfg: Config) -> dict[str, float]:
    """Mean rank IC, ICIR and Newey-West t per horizon (one value per day, averaged over its events) and the linked-stock fit; plus the counts."""
    out: dict[str, float] = {"n_events": float(len(events))}
    if events.empty:
        return out
    lag = int(cfg.get("eval.hac_lag"))
    for h in [int(x) for x in cfg.get("ablation.rank_ic_horizons")]:
        daily = events.dropna(subset=[f"ic_{h}d"]).groupby("day0")[f"ic_{h}d"].mean()
        if len(daily):
            report = ic_report(daily, lag)
            out.update({f"rank_ic_{h}d": report["ic"], f"icir_{h}d": report["icir"], f"t_newey_west_{h}d": report["t_newey_west"],
                        f"n_events_ic_{h}d": float(events[f"ic_{h}d"].notna().sum())})
        out.update(linked_fit(events, h))
    return out


def by_cut(events: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """The same summary for each cut of the data (kind, single or multiple pins, year, size layer); a cut with too few events is counted and left out."""
    minimum = int(cfg.get("ablation.linked_min_events"))
    pins = cfg.get("ablation.strata.pin_count")
    cuts: dict[str, pd.Series] = {"all": pd.Series("all", index=events.index)}
    if len(events):
        cuts.update({"kind": events["kind"], "year": events["year"].astype(str), "size_layer": events["size_layer"].round().astype("Int64").astype(str),
                     "pin_count": events["n_pins"].map(lambda n: next((name for name, (lo, hi) in pins.items() if n >= lo and (hi is None or n <= hi)), "other"))})
    rows = []
    for cut, labels in cuts.items():
        for label, index in labels.groupby(labels).groups.items():
            part = events.loc[index]
            rows.append({"cut": cut, "group": label, "n_events": float(len(part)), "reported": len(part) >= minimum,
                         **(summarise(part, cfg) if len(part) >= minimum else {})})
    return pd.DataFrame(rows)
