"""Calibration of road 2's volume dial: one scale and one shift, fitted on the dev (fit) window only, with time-blocked folds.

Why: the volume dial road 2 hands the optimiser is ``log(1 + game volume / baseline dollar volume)``; on the fit window its level and spread are far from the realised log dollar-volume
ratio of the label window (d01) that road 1's dial is trained on, and it enters the cost multiplier exp(vol - alpha volu). The calibration maps it to the label's scale: ``b + a x`` with
``a`` clipped to [0, max_scale] (shrink only, never amplified) and ``b`` the shift that matches the means.

One stage, one setting: it is fitted once per tape inside the dev stage (``run_final(stop_after="calibrate")``), written to ``volu_calibration_<run>.json`` and its sha256 and values go into
FREEZE_FRAMEWORK.json. It is used only when it lowers the out-of-fold squared error in EVERY time block; otherwise the dial stays raw and the file says so. No test or sealed row is read.
``framework.calibration.enabled: false`` (or ``run_final(calibrate_volu=False)``) is the off switch, so that raw and calibrated can be reported side by side.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finfm.common.config import Config
from finfm.common.hashing import sha256_hex
from finfm.common.rng import make_rng

DIAL_NAME = "volu_road2"
MIN_PAIRS = 200  # fewer fit-window pairs than this in a block (or in all): the calibration is not fitted


def fit_scale_shift(x: np.ndarray, y: np.ndarray, max_scale: float) -> tuple[float, float]:
    """``y ~ b + a x`` by least squares, ``a`` clipped to [0, max_scale], ``b`` = mean(y) - a mean(x)."""
    var = float(((x - x.mean()) ** 2).sum())
    a = float(((x - x.mean()) * (y - y.mean())).sum() / var) if var > 0 else 0.0
    a = min(max(a, 0.0), float(max_scale))
    return a, float(y.mean() - a * x.mean())


def time_blocks(days: pd.Series, folds: int) -> pd.Series:
    """Block number 0..folds-1 of every row: contiguous blocks of the event dates with (about) the same number of EVENTS in each."""
    order = days.drop_duplicates().sort_values()
    edges = np.array_split(order.to_numpy(), folds)
    block_of = {d: k for k, part in enumerate(edges) for d in part}
    return days.map(block_of)


def fit_volu_calibration(pairs: pd.DataFrame, cfg: Config) -> dict[str, Any]:
    """``pairs``: columns event_id, day0 (Timestamp), x (raw road-2 volume dial), y (realised log dollar-volume ratio, label window d01); fit-window events only.

    Per block: the dial raw (error x - y) against the dial calibrated with the scale and shift fitted on the OTHER blocks (events within ``embargo_days`` of the block left out of that fit)."""
    spec = dict(cfg.get("framework.calibration"))
    folds, embargo, max_scale = int(spec["folds"]), pd.Timedelta(days=int(spec["embargo_days"])), float(spec["max_scale"])
    pairs = pairs.dropna(subset=["x", "y"]).reset_index(drop=True)
    out: dict[str, Any] = {"dial": DIAL_NAME, "folds": folds, "embargo_days": int(spec["embargo_days"]), "max_scale": max_scale, "n_pairs": int(len(pairs)), "n_events": int(pairs["event_id"].nunique())}
    if len(pairs) < MIN_PAIRS or pairs["day0"].nunique() < folds:
        return {**out, "used": False, "scale": 1.0, "shift": 0.0, "blocks": [], "reason": f"only {len(pairs)} fit-window pairs: not fitted, the dial stays raw"}
    day0 = pd.to_datetime(pairs["day0"])
    block = time_blocks(day0, folds)
    rows = []
    for k in range(folds):
        val = block == k
        lo, hi = day0[val].min(), day0[val].max()
        train = (~val) & ((day0 < lo - embargo) | (day0 > hi + embargo))
        a, b = fit_scale_shift(pairs.loc[train, "x"].to_numpy(float), pairs.loc[train, "y"].to_numpy(float), max_scale)
        x, y = pairs.loc[val, "x"].to_numpy(float), pairs.loc[val, "y"].to_numpy(float)
        raw, cal = float(((x - y) ** 2).mean()), float((((b + a * x) - y) ** 2).mean())
        rows.append({"block": k, "first_day": str(lo.date()), "last_day": str(hi.date()), "n_events": int(pairs.loc[val, "event_id"].nunique()), "n_pairs": int(val.sum()),
                     "scale_fitted_on_other_blocks": a, "shift_fitted_on_other_blocks": b, "raw_mean_dial": float(x.mean()), "realised_mean": float(y.mean()), "raw_rmse": float(np.sqrt(raw)),
                     "calibrated_rmse": float(np.sqrt(cal)), "raw_mse": raw, "calibrated_mse": cal})
    a, b = fit_scale_shift(pairs["x"].to_numpy(float), pairs["y"].to_numpy(float), max_scale)
    better = all(r["calibrated_mse"] < r["raw_mse"] for r in rows)
    return {**out, "used": bool(better), "scale": a, "shift": b, "blocks": rows,
            "reason": "the calibration lowers the out-of-fold squared error in every time block" if better else "the calibration does not lower the out-of-fold error in every block: the dial stays raw"}


def fit_window_pairs(dev_forecasts: Sequence, labels: pd.DataFrame, columns: Sequence[str], day0_date: Mapping[str, pd.Timestamp], per_event: int, run: str) -> pd.DataFrame:
    """Raw road-2 volume dial (``volu_parts[1]``) against the label, for a random sample of ``per_event`` assets of every fit-window event (the same sample for any run of this tape)."""
    real = labels.set_index(["event_id", "security_id"])["log_dv_ratio"].replace([np.inf, -np.inf], np.nan)
    rows = []
    for f in dev_forecasts:
        rng = make_rng("final", "calibration_sample", run, f.event_id)
        pick = np.sort(rng.permutation(len(f.assets))[:per_event])
        for j in pick:
            rows.append((f.event_id, day0_date[f.event_id], float(np.asarray(f.volu_parts[1])[j]), real.get((f.event_id, columns[f.assets[j]]), np.nan)))
    return pd.DataFrame(rows, columns=["event_id", "day0", "x", "y"])


def sampled_nodes(dev_forecasts: Sequence, columns: Sequence[str], per_event: int, run: str) -> dict[str, tuple[str, ...]]:
    out = {}
    for f in dev_forecasts:
        rng = make_rng("final", "calibration_sample", run, f.event_id)
        pick = np.sort(rng.permutation(len(f.assets))[:per_event])
        out[f.event_id] = tuple(columns[f.assets[j]] for j in pick)
    return out


def canonical(record: Mapping[str, Any]) -> str:
    return json.dumps(record, sort_keys=True, indent=1, default=float)


def calibration_sha256(record: Mapping[str, Any]) -> str:
    return sha256_hex(canonical(record))


def write(path: Path, record: Mapping[str, Any]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(canonical(record))
    return calibration_sha256(record)


def read(path: Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def apply(volu_b: np.ndarray, calibration: tuple[float, float] | None) -> np.ndarray:
    """Calibrated road-2 volume dial; unchanged when no calibration is set (switch off, or not stable across the time blocks)."""
    if calibration is None:
        return volu_b
    scale, shift = calibration
    return shift + scale * volu_b


def setting_from_record(record: Mapping[str, Any], enabled: bool) -> tuple[float, float] | None:
    return (float(record["scale"]), float(record["shift"])) if enabled and record.get("used") else None
