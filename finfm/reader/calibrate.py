"""Calibration of the reader's probabilities and certainty (spec 4.5): reliability, ECE, isotonic."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression

from finfm.common.config import Config


def equal_mass_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    """Bin index 0..n_bins-1 with (nearly) equal counts, by rank."""
    ranks = pd.Series(values).rank(method="first").to_numpy()
    return np.minimum(((ranks - 1) * n_bins / len(values)).astype(int), n_bins - 1)


def reliability_table(predicted: np.ndarray, outcome: np.ndarray, n_bins: int) -> pd.DataFrame:
    """Mean prediction versus frequency of the outcome in each equal-mass bin."""
    frame = pd.DataFrame({"p": predicted, "y": outcome.astype(float), "bin": equal_mass_bins(predicted, n_bins)})
    table = frame.groupby("bin").agg(mean_predicted=("p", "mean"), frequency=("y", "mean"), n=("y", "size"))
    return table.reset_index()


def expected_calibration_error(predicted: np.ndarray, outcome: np.ndarray, n_bins: int) -> float:
    """ECE with equal-mass bins: count-weighted mean absolute gap between prediction and frequency."""
    table = reliability_table(predicted, outcome, n_bins)
    return float((table["n"] * (table["mean_predicted"] - table["frequency"]).abs()).sum() / table["n"].sum())


def fit_isotonic(certainty: np.ndarray, exceeds_median: np.ndarray) -> IsotonicRegression:
    """Isotonic recalibration of certainty against whether |abnormal return| exceeded the dev median."""
    model = IsotonicRegression(y_min=0.0, y_max=1.0, out_of_bounds="clip")
    return model.fit(certainty, exceeds_median.astype(float))


def calibration_report(certainty: np.ndarray, outcome: np.ndarray, cfg: Config) -> dict[str, float]:
    """ECE before and after isotonic recalibration (in-sample for the isotonic fit; report dev only)."""
    bins = int(cfg.get("reader.calibration.ece_bins"))
    model = fit_isotonic(certainty, outcome)
    return {
        "ece_raw": expected_calibration_error(certainty, outcome, bins),
        "ece_isotonic": expected_calibration_error(model.predict(certainty), outcome, bins),
    }
