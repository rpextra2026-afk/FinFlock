"""Real inputs of the mean-field game: fund types from the persona panels, and a month's covariance, liquidity and game matrices.

Fund types are the institutional persona skeletons of the newest quarter whose filings are known (13F lag). The S&P share of a
type's book is the book-weighted mean over that quarter's active filers of the type (the skeletons themselves are normalised inside the
S&P 500). The sleeve is the proxy ticker in ``game.sleeve_proxy``. Approximation, stated in the paper: one S&P share per type and
quarter, not per cluster; retail's uncovered share is the configured constant.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from oryxis.common.config import Config
from oryxis.framework.run import MonthContext
from oryxis.mfg.precompute import FundInput, GameError, amihud_liquidity, build_month_game, estimate_covariance
from oryxis.personas.skeletons import RETAIL, Skeleton
from oryxis.real.market_load import RealMarket
from oryxis.stage.persona_build import panel_for_date, quarter_label

SECURITY = "security_id"


class MonthSkipped(RuntimeError):
    """A month for which no game can be built (no panel yet, too few assets, a degenerate matrix). Carries the reason."""


def panels_from_skeletons(skeletons: Mapping[str, Skeleton]) -> dict[str, list[Skeleton]]:
    panels: dict[str, list[Skeleton]] = {}
    for skeleton in skeletons.values():
        panels.setdefault(skeleton.asof_quarter, []).append(skeleton)
    return panels


def sp_cover_table(managers: pd.DataFrame, cfg: Config) -> dict[tuple[str, str], float]:
    """``(quarter label, type) -> book-weighted mean S&P share`` over active (not passive) filers."""
    active = managers[~managers["passive_flag"]].dropna(subset=["sp_cover", "book_usd"])
    out = {}
    for (period, kind), part in active.groupby(["period", "type"]):
        out[(quarter_label(pd.Timestamp(period), cfg), str(kind))] = float(np.average(part["sp_cover"], weights=part["book_usd"]))
    return out


def fund_inputs(panel: list[Skeleton], asset_ids: list[str], cover: Mapping[tuple[str, str], float]) -> list[FundInput]:
    """One ``FundInput`` per institutional skeleton: holdings on the month's assets scaled to the type's S&P share."""
    index = {a: i for i, a in enumerate(asset_ids)}
    funds = []
    for group in sorted({s.persona_type for s in panel if s.persona_type != RETAIL}):
        members = [s for s in panel if s.persona_type == group]
        percentile = pd.Series([s.turnover for s in members]).rank(pct=True, method="average").to_numpy()
        for s, rank in zip(members, percentile, strict=True):
            share = cover.get((s.asof_quarter, group))
            held = {a: w for a, w in s.holdings.items() if a in index}
            if share is None or not held or sum(held.values()) <= 0:
                continue
            weights = np.zeros(len(asset_ids))
            total = sum(held.values())
            for a, w in held.items():
                weights[index[a]] = share * w / total
            funds.append(FundInput(s.persona_id, group, s.book_usd, weights, float(rank)))
    return funds


@dataclass
class GameInputs:
    """Month contexts built lazily from real prices and the persona panels; skipped months keep their reason."""

    cfg: Config
    rm: RealMarket
    panels: dict[str, list[Skeleton]]
    cover: Mapping[tuple[str, str], float]
    uncovered_share: float
    skipped: dict[pd.Period, str] = field(default_factory=dict)
    fund_transform: Callable[[list[FundInput]], list[FundInput]] | None = None  # ablation: changes the game's fund inputs (e.g. equal books)
    _months: dict[pd.Period, MonthContext] = field(default_factory=dict)

    def month_context(self, month: pd.Period) -> MonthContext:
        if month in self._months:
            return self._months[month]
        if month in self.skipped:
            raise MonthSkipped(self.skipped[month])
        try:
            self._months[month] = self._build(month)
        except (GameError, np.linalg.LinAlgError, ValueError) as exc:
            self.skipped[month] = f"{type(exc).__name__}: {exc}"
            raise MonthSkipped(self.skipped[month]) from exc
        return self._months[month]

    def _build(self, month: pd.Period) -> MonthContext:
        cfg, rm = self.cfg, self.rm
        start = month.start_time
        sleeve = str(cfg.get("game.sleeve_proxy"))
        label = panel_for_date(start, self.panels, cfg)
        if label is None:
            raise MonthSkipped("no persona panel is known yet")
        window = int(cfg.get("game.covariance_window_days"))
        history = rm.ret.loc[: start - pd.Timedelta(days=1)].iloc[-window:]
        liq_window = int(cfg.get("game.amihud.window_days"))
        dv = (rm.close * rm.volume).loc[: start - pd.Timedelta(days=1)].iloc[-liq_window:]
        enough = history.notna().sum() >= int(cfg.get("game.min_return_obs"))
        priced = (dv > 0).sum() > 0
        members = sorted(a for a in rm.universe()(start) if a != sleeve and enough.get(a, False) and priced.get(a, False))
        if len(members) < int(cfg.get("game.min_assets")) or not enough.get(sleeve, False):
            raise MonthSkipped(f"{len(members)} usable assets, sleeve usable: {bool(enough.get(sleeve, False))}")
        funds = fund_inputs(self.panels[label], members, self.cover)
        if self.fund_transform is not None and funds:
            funds = self.fund_transform(funds)
        if not funds:
            raise MonthSkipped(f"no fund type has an S&P share and holdings in {label}")
        panel = history[members + [sleeve]].fillna(0.0)
        sigma_plus = estimate_covariance(panel.to_numpy(), float(cfg.get("game.covariance_ridge")))
        abs_ret = rm.ret.loc[dv.index, members].abs().fillna(0.0).to_numpy()
        ell = amihud_liquidity(abs_ret, dv[members].fillna(0.0).to_numpy(), float(cfg.get("game.liquidity_scale")),
                               float(cfg.get("game.amihud.winsor_low")), float(cfg.get("game.amihud.winsor_high")))
        game = build_month_game(cfg, members, sigma_plus, ell, funds)
        return MonthContext(game, members, np.full(len(members), self.uncovered_share), dv[members].mean().to_numpy())


def make_game_inputs(cfg: Config, rm: RealMarket, skeletons: Mapping[str, Skeleton], managers: pd.DataFrame) -> GameInputs:
    return GameInputs(cfg, rm, panels_from_skeletons(skeletons), sp_cover_table(managers, cfg),
                      float(cfg.get("personas.panel.default_uncovered_share")))
