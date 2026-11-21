"""Prices, returns, betas and forward returns for scoring a framework (realised returns: scoring code only).

Fill rule (spec 9.4): a decision after the close of session ``d`` is filled at the open of the next session ``f``.
Forward return over ``h`` sessions = open of ``f`` to the close of session ``f + h - 1``. The abnormal return is
the stock return minus ``beta x market``, with beta from the prior estimation window, shrunk to the cross-section
(Vasicek) and the market total return ``Mkt-RF + RF``. The market leg is close to close while the stock leg starts at
the open, so the abnormal return is an approximation; outputs say so.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from finflock.common.config import Config
from finflock.common.timing import TradingCalendar
from finflock.data.market import total_returns, wide
from finflock.portfolio.market import Market

CLOSE, OPEN, VOLUME = "close_raw", "open", "volume_raw"
FACTOR_COLUMNS = ("mkt_rf", "rf")
ACTION_COLUMNS = ["security_id", "ex_date", "action_type", "ratio", "amount"]


class FactorFileMissing(RuntimeError):
    """The Fama-French factor table is not on disk; the scorer stops (no substitute is allowed)."""


@dataclass
class ScoreMarket:
    sessions: pd.DatetimeIndex
    close: pd.DataFrame
    open_: pd.DataFrame
    volume: pd.DataFrame
    cc: pd.DataFrame            # close-to-close total return of each session
    oo: pd.DataFrame            # open of the session to the open of the next, total return
    oc: pd.DataFrame            # open to close of the same session
    last_trading_date: pd.Series
    factors: pd.DataFrame       # sessions x (mkt_rf, smb, hml, rmw, cma, rf, mom)
    market_total: pd.Series     # mkt_rf + rf
    _impact_inputs: tuple | None = None

    def execution_market(self, fill: str, cfg: Config) -> tuple[Market, pd.DataFrame]:
        """Market for the executor and the return panel it holds. ``open``: open to next open; ``close``: close to next close."""
        panel = self.oo if fill == cfg.get("cpu.score_basic.fills.headline") else self.cc.shift(-1)
        return Market(self.close, self.volume, panel, self.last_trading_date), panel

    def adv_and_sigma(self, cfg: Config) -> tuple[pd.DataFrame, pd.DataFrame]:
        """ADV in dollars and daily vol, both known through the previous session (rolling windows from the impact config)."""
        if self._impact_inputs is None:
            dollars = self.close * self.volume
            adv = dollars.rolling(int(cfg.get("portfolio.impact.adv_window_days")), min_periods=1).mean().shift(1)
            sigma = self.cc.rolling(int(cfg.get("portfolio.impact.vol_window_days")), min_periods=2).std().shift(1)
            self._impact_inputs = (adv, sigma)
        return self._impact_inputs


def build_market(close: pd.DataFrame, open_: pd.DataFrame, volume: pd.DataFrame, actions: pd.DataFrame,
                 factors: pd.DataFrame, last_trading_date: pd.Series) -> ScoreMarket:
    cc, oo = total_returns(close, open_, actions)
    sessions = close.index
    fac = factors.reindex(sessions)
    return ScoreMarket(sessions, close, open_, volume, cc, oo, close / open_ - 1.0, last_trading_date, fac,
                       fac["mkt_rf"] + fac["rf"])


def load_market(root: Path, cfg: Config, calendar: TradingCalendar) -> ScoreMarket:
    """Real prices, corporate actions and factors from the folders named in ``cpu.folders``."""
    factor_path = root / str(cfg.get("cpu.folders.factors"))
    if not factor_path.exists():
        raise FactorFileMissing(f"factor file not found: {factor_path} (scripts/fetch_factors.py builds it); stopping, no substitute is used")
    folder = root / str(cfg.get("cpu.folders.market"))
    px = pd.read_parquet(folder / "px_daily.parquet")
    actions = pd.read_parquet(folder / "corp_action.parquet")
    sessions = calendar.sessions
    panels = {c: wide(px, c, sessions) for c in (CLOSE, OPEN, VOLUME)}
    last = px.groupby("security_id")["session_date"].max().pipe(pd.to_datetime)
    factors = pd.read_parquet(factor_path)
    missing = [c for c in FACTOR_COLUMNS if c not in factors.columns]
    if missing:
        raise FactorFileMissing(f"factor file lacks columns {missing}")
    return build_market(panels[CLOSE], panels[OPEN], panels[VOLUME], actions, factors, last)


def fill_session(sessions: pd.DatetimeIndex, decision: pd.Timestamp) -> pd.Timestamp | None:
    """The next session after the decision date (its open is the headline fill)."""
    position = sessions.searchsorted(pd.Timestamp(decision), side="right")
    return sessions[position] if position < len(sessions) else None


def stock_forward(sm: ScoreMarket, horizon: int) -> pd.DataFrame:
    """Row = fill session ``f``: total return from the open of ``f`` to the close of ``f + horizon - 1``."""
    log_oo = np.log1p(sm.oo)
    total = np.log1p(sm.oc.shift(-(horizon - 1)))
    for k in range(horizon - 1):
        total = total + log_oo.shift(-k)
    return np.expm1(total)


def market_forward(sm: ScoreMarket, horizon: int) -> pd.Series:
    """Row = fill session: compounded market total return over ``horizon`` sessions starting at the fill session."""
    log_m = np.log1p(sm.market_total)
    total = sum(log_m.shift(-k) for k in range(horizon))
    return np.expm1(total)


def shrunk_betas(sm: ScoreMarket, date: pd.Timestamp, members: list[str], cfg: Config) -> pd.Series:
    """Beta of excess returns on Mkt-RF over the sessions up to ``date``, Vasicek-shrunk to the member cross-section."""
    window = int(cfg.get("cpu.score_basic.beta.window_sessions"))
    need = int(np.ceil(window * float(cfg.get("cpu.score_basic.beta.min_fraction"))))
    end = sm.sessions.searchsorted(pd.Timestamp(date), side="right")
    rows = slice(max(end - window, 0), end)
    names = [m for m in members if m in sm.cc.columns]
    y = (sm.cc[names].iloc[rows].sub(sm.factors["rf"].iloc[rows], axis=0)).to_numpy()
    x = sm.factors["mkt_rf"].iloc[rows].to_numpy()
    valid = np.isfinite(y) & np.isfinite(x)[:, None]
    n = valid.sum(axis=0)
    xv = np.where(valid, x[:, None], 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        xbar = xv.sum(axis=0) / n
        yv = np.where(valid, y, 0.0)
        ybar = yv.sum(axis=0) / n
        dx, dy = np.where(valid, x[:, None] - xbar, 0.0), np.where(valid, y - ybar, 0.0)
        sxx, sxy, syy = (dx * dx).sum(axis=0), (dx * dy).sum(axis=0), (dy * dy).sum(axis=0)
        beta = sxy / sxx
        se2 = (syy - beta * sxy) / (n - 2) / sxx
    ok = (n >= need) & np.isfinite(beta) & np.isfinite(se2)
    out = pd.Series(np.nan, index=names)
    if ok.sum() < 2:
        return out
    prior_mean = float(beta[ok].mean())
    prior_var = max(float(beta[ok].var(ddof=1)) - float(se2[ok].mean()), 0.0)
    weight = prior_var / (prior_var + se2[ok])
    out[np.array(names)[ok]] = weight * beta[ok] + (1.0 - weight) * prior_mean
    return out


def abnormal_forward(sm: ScoreMarket, decisions: list[pd.Timestamp], members: dict[pd.Timestamp, set], horizon: int,
                     cfg: Config) -> pd.DataFrame:
    """Rows = fill sessions of the given decision dates, columns = securities: stock minus beta x market, same horizon."""
    stock, market = stock_forward(sm, horizon), market_forward(sm, horizon)
    rows = {}
    for date in sorted(set(decisions)):
        fill = fill_session(sm.sessions, date)
        if fill is None:
            continue
        beta = shrunk_betas(sm, date, sorted(members[date]), cfg)
        rows[fill] = stock.loc[fill, beta.index] - beta * market.loc[fill]
    return pd.DataFrame(rows).T.sort_index()
