"""The world lane inside the framework: a market-level volatility (and volume) shift for every day, from the news that has no company pin.

Reuses ``finfm.build.world`` (daily targets, theme features, the ridge fit and prediction). Fitted on the fit window only; the prediction for a session uses the
theme counts of that session and earlier ones. One switch: config ``framework.world_lane.enabled``. Off means no shift anywhere.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from finfm.build import world as W
from finfm.common.config import Config

ROOT = "framework.world_lane"


@dataclass
class WorldShift:
    vol: pd.Series  # log-volatility shift by session (market level)
    volu: pd.Series  # log-volume shift by session
    notes: dict

    def __call__(self, date: pd.Timestamp) -> tuple[float, float]:
        key = pd.Timestamp(date)
        return float(self.vol.get(key, 0.0)), float(self.volu.get(key, 0.0))


def enabled(cfg: Config) -> bool:
    return bool(cfg.get(f"{ROOT}.enabled"))


def fit_world_shift(world_pins: pd.DataFrame, prices: pd.DataFrame, run_cfg: dict, themes: Sequence[str], cfg: Config, days: pd.DatetimeIndex) -> WorldShift:
    """Predicted market-level shift on ``days``: ``log(predicted mean |return| / fit-window mean)`` for volatility and ``predicted - fit-window mean`` for log volume, both clipped."""
    targets = W.daily_targets(prices, cfg)
    feats = W.theme_features(world_pins, themes, cfg)
    sessions = targets["vol_raw"].index
    fit_days = sessions[(sessions >= pd.Timestamp(run_cfg["fit_start"])) & (sessions <= pd.Timestamp(run_cfg["fit_end"]))]
    if len(fit_days) < int(cfg.get("build.world.min_days")) or feats.empty:
        zero = pd.Series(0.0, index=days)
        return WorldShift(zero, zero.copy(), {"stopped": "too few fit days or no world-lane items", "fit_days": int(len(fit_days))})
    feats = feats.reindex(sessions).fillna(0.0)
    burst = None
    if str(cfg.get(f"{ROOT}.version")) == "burst_only":
        total = feats[[c for c in feats.columns if c.startswith("weighted_count_")]].sum(axis=1)
        from finfm.build import metrics as M

        thr = float(np.quantile(total.reindex(fit_days), float(M.mcfg(cfg, "burst_quantile"))))
        burst = set(total.index[total >= thr]) if thr > 0 else set()
    pred_days = pd.DatetimeIndex(days).intersection(sessions)
    out: dict[str, pd.Series] = {}
    notes: dict = {"fit_days": int(len(fit_days)), "version": str(cfg.get(f"{ROOT}.version"))}
    for name in cfg.get(f"{ROOT}.targets"):
        daily = targets[f"{name}_raw"].mean(axis=1)
        base = float(daily.reindex(fit_days).mean())
        pred, _, alpha, _ = W.fit_predict(feats, daily, fit_days, pred_days, cfg, burst)
        clip = float(cfg.get(f"{ROOT}.clip_log_{name}"))
        shift = np.log(pred.clip(lower=np.finfo(float).tiny) / base) if name == "vol" else pred - base
        out[name] = shift.clip(-clip, clip).reindex(days).fillna(0.0)
        notes[f"ridge_{name}"] = alpha
        notes[f"fit_mean_{name}"] = base
    zero = pd.Series(0.0, index=days)
    return WorldShift(out.get("vol", zero), out.get("volu", zero.copy()), notes)


def shift_for(cfg: Config, shift: WorldShift | None) -> Callable[[pd.Timestamp], tuple[float, float]] | None:
    """The callable the strategy takes, or ``None`` when the lane is switched off."""
    return shift if (enabled(cfg) and shift is not None) else None
