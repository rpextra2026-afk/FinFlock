"""MASS data modalities with US analogues (spec 10.4). Descriptions follow the repository's wording.

Fundamentals (valuation, quality, growth, dividend) need a point-in-time fundamentals panel from the SEC
financial statement data sets; the price-derived modalities are built here from prices. A modality with
no data is dropped from the draw and the share dropped is reported (never silently).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from oryxis.common.config import Config
from oryxis.common.rng import make_rng

DESCRIPTIONS: dict[str, dict[str, str]] = {
    "valuation": {
        "E/P": "The inverse of the P/E ratio (E/P) indicates the earnings yield, showing the percentage of profit generated per dollar invested in the stock.",
        "B/P": "Inverse of P/B (B/P) indicates the book yield, showing the return on book value per dollar invested.",
        "S/P": "Inverse of P/S (S/P) reflects the sales yield, showing sales generated per dollar invested.",
        "CF/P": "Inverse of P/CF (CF/P) shows the cash flow yield, representing cash flow generated per dollar invested.",
        "EBITDA/EV": "Measures a company's return on enterprise value, indicating operating earnings (EBITDA) generated per dollar of EV.",
    },
    "quality": {
        "ROE": "ROE Measures profitability, showing net income generated per dollar of shareholders' equity.",
        "ROE stability": "TS_Mean(ROE, 8) / TS_Std(ROE, 8), measuring both absolute value and stability of ROE.",
        "ROA stability": "TS_Mean(ROA, 8) / TS_Std(ROA, 8), measuring both absolute value and stability of ROA.",
    },
    "growth": {
        "Revenue TTM YoY growth rate": "Measures the percentage change in trailing twelve months' revenue compared to the same period last year.",
        "Net profit TTM YoY growth rate": "Measures the percentage change in trailing twelve months' net profit compared to the same period last year.",
    },
    "dividend": {
        "Dividend yield": "Dividend yield indicates annual dividends received per dollar invested, expressed as a percentage of the stock price",
    },
    "risk_factor": {
        "Intraday volatility": "Measuring the price fluctuation range of a stock within a single trading day.",
        "Liquidity": "Weighted average of monthly, quarterly and yearly turnover ratio.",
        "Residual volatility": "Residual volatility measures the unexplained variability in a security's returns after accounting for market or factor influences, indicating idiosyncratic risk.",
    },
    "base_data": {},
    "cross_industry": {"Daily_Return": "One-day return of holding the sector's constituent stocks."},
    "price_feature": {
        "ret_1": "Return over the last trading day.",
        "ret_5": "Return over the last five trading days.",
        "ret_20": "Return over the last twenty trading days.",
        "vol_20": "Standard deviation of daily returns over the last twenty trading days.",
        "volume_ratio": "Today's dollar volume divided by its twenty-day average.",
        "range_ratio": "Today's high-low range divided by its twenty-day average range.",
    },
    "news": {},
}
HALF = 0.5
GK_LN2_TERM = 0.38629436111989063  # 2 ln 2 - 1 of the Garman-Klass estimator
SHORT_WINDOW = 5
MEDIUM_WINDOW = 20
RESIDUAL_VOL_WINDOW = 60
FUNDAMENTAL_MODALITIES = ("valuation", "quality", "growth", "dividend")


@dataclass
class ModalityStore:
    """Per modality: a ``(Stock, Date, ...features)`` frame; descriptions are the repository's wording."""

    frames: dict[str, pd.DataFrame]

    def available(self, wanted: list[str]) -> list[str]:
        return [m for m in wanted if m in self.frames and len(self.frames[m])]

    def descriptions(self, modalities: list[str]) -> dict[str, str]:
        out: dict[str, str] = {}
        for modality in modalities:
            out.update({k: v for k, v in DESCRIPTIONS[modality].items() if k in self.frames[modality].columns})
        return out


def price_modalities(close, high, low, open_, volume, sector, sector_ret=None) -> dict[str, pd.DataFrame]:
    """Price-derived modalities from wide panels (sessions x securities); ``Date`` is the session."""
    dollar = close * volume
    ret = close.pct_change()
    gk = HALF * np.log(high / low) ** 2 - GK_LN2_TERM * np.log(close / open_) ** 2
    turnover = volume / volume.rolling(MEDIUM_WINDOW).mean()
    frames = {}

    def long(name_to_panel: dict[str, pd.DataFrame]) -> pd.DataFrame:
        parts = [panel.stack(future_stack=True).rename(name) for name, panel in name_to_panel.items()]
        out = pd.concat(parts, axis=1).reset_index()
        out.columns = ["Date", "Stock", *name_to_panel]
        return out

    frames["base_data"] = long({"Open": open_, "High": high, "Low": low, "Close": close, "Value": dollar})
    frames["risk_factor"] = long({
        "Intraday volatility": np.sqrt(gk.clip(lower=0.0)), "Liquidity": turnover,
        "Residual volatility": ret.rolling(RESIDUAL_VOL_WINDOW).std(),
    })
    frames["price_feature"] = long({
        "ret_1": ret, "ret_5": close.pct_change(SHORT_WINDOW), "ret_20": close.pct_change(MEDIUM_WINDOW),
        "vol_20": ret.rolling(MEDIUM_WINDOW).std(),
        "volume_ratio": dollar / dollar.rolling(MEDIUM_WINDOW).mean(),
        "range_ratio": (high - low) / (high - low).rolling(MEDIUM_WINDOW).mean(),
    })
    sector_daily = pd.DataFrame({s: ret.loc[:, sector[sector == s].index].mean(axis=1) for s in sector.unique()})
    frames["cross_industry"] = long({"Daily_Return": ret.copy().apply(lambda col: sector_daily[sector[col.name]])})
    return {k: v.dropna(subset=[c for c in v.columns if c not in ("Date", "Stock")], how="all") for k, v in frames.items()}


def draw_modalities(cfg: Config, available: list[str], seed_parts: tuple) -> list[str]:
    """A random subset of ``modality_count`` modalities for one investor type (repository: ``randint(1, 3)``)."""
    low, high = (int(x) for x in cfg.get("rivals.mass.modality_count"))
    rng = make_rng(*seed_parts)
    count = int(rng.integers(low, min(high, len(available)) + 1))
    return sorted(rng.choice(available, size=count, replace=False).tolist())
