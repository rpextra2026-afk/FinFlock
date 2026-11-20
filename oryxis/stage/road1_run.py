"""Fit the road-1 methods on dev graphs and score them on test graphs: one function for the reader-based run and for the oracle run.

Everything that decides what is fitted (the grid, the steps, the batch of events per step, the folds) comes from ``road1.*`` exactly as before; the
only run-time inputs are how many processes to use and a resource log. Output is a dict of tables; nothing here decides anything.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass

import numpy as np
import pandas as pd

from oryxis.common.config import Config
from oryxis.common.timing import TradingCalendar
from oryxis.road1.fit import dial_scale, select_and_fit, standardise, usable
from oryxis.road1.graph import EventGraph
from oryxis.road1.rreport import r_summary, r_table
from oryxis.road1.strata import event_strata, sheaf_strata_table
from oryxis.stage import road1_real as rr

Stage = Callable[[str], AbstractContextManager]


class NotEnoughGraphs(ValueError):
    """No labelled graph on the dev or the test side."""


@dataclass
class Road1Output:
    fitted: rr.Fitted
    scores: pd.DataFrame
    tables: dict[str, pd.DataFrame]


def fit_and_score(
    dev: Sequence[EventGraph], test: Sequence[EventGraph], day0: Mapping[str, pd.Timestamp], pin_types: Mapping[str, Sequence[str]],
    calendar: TradingCalendar, cfg: Config, seed_parts: tuple, workers: int, stage: Stage, log: Callable[[str], None] = lambda message: None,
    with_r_report: bool = True, with_sheaf: bool = True,
) -> Road1Output:
    """``with_sheaf=False``: only the weighted graph and the weighted matrices are fitted and scored; the sheaf, the sheaf test by stratum and the
    sheaf's r are left out (the sheaf is then run on its own through the framework's part 1)."""
    if not usable(dev) or not usable(test):
        raise NotEnoughGraphs("not enough labelled graphs on dev or test")
    scale = dial_scale(dev, float(cfg.get("road1.dial_scale_floor")))
    std_dev = standardise(dev, scale)
    dev_dates = [day0[g.event_id] for g in dev]
    configs: list = []
    with stage("fit wgraph"):
        wg = select_and_fit("wgraph", std_dev, dev_dates, calendar, cfg, seed_parts=seed_parts, on_config=configs.append, workers=workers)
    log(f"wgraph grounding {wg[1].grounding}, cv loss {wg[1].val_loss:.4f}")
    sh = None
    if with_sheaf:
        with stage("fit sheaf"):
            sh = select_and_fit("sheaf", std_dev, dev_dates, calendar, cfg, seed_parts=seed_parts, on_config=configs.append, workers=workers)
        log(f"sheaf grounding {sh[1].grounding} penalty {sh[1].penalty_mu}, cv loss {sh[1].val_loss:.4f}")
    with stage("fit matrices"):
        mats = rr.fit_matrices(std_dev, dev_dates, calendar, cfg)
    fitted = rr.Fitted(scale, {}, (wg[0], wg[1]), (sh[0], sh[1]) if sh is not None else None, mats, len(dev))
    tables = {"fits": pd.DataFrame([
        {"method": "wgraph", "dev_cv_loss": wg[1].val_loss, "grounding": wg[1].grounding, "penalty": None, "parameters": rr.n_params(wg[0])},
        *([{"method": "sheaf", "dev_cv_loss": sh[1].val_loss, "grounding": sh[1].grounding, "penalty": sh[1].penalty_mu, "parameters": rr.n_params(sh[0])}]
          if sh is not None else []),
        *[{"method": f"wmatrix_r{r}", "dev_cv_loss": loss, "grounding": None, "penalty": ridge, "parameters": m.n_parameters()} for r, (m, ridge, loss) in mats.items()]]),
        "configs_evaluated": pd.DataFrame([vars(c) for c in configs])}
    with stage("score test"):
        scores = rr.long_scores(fitted, list(test), day0, cfg)
    dial_names = list(cfg.get("road1.dial_names"))
    train_mean = dict(zip(dial_names, np.nanmean(np.vstack([g.y for g in usable(dev)]), axis=0), strict=True))
    n_pinned = {g.event_id: g.n_pinned for g in [*dev, *test]}
    strata = event_strata(n_pinned, pin_types, cfg)
    best_rung = min(mats, key=lambda r: mats[r][2])
    primary = str(cfg.get("cpu.road1_eval.sheaf_test.primary_table"))
    tables.update({
        "road1_scores": rr.score_table(scores, train_mean, cfg, seed_parts),
        "road1_scores_by_stratum": rr.score_table_strata(scores, strata, train_mean, cfg, seed_parts),
        "strata_counts": strata.groupby(["pin_stratum", "node_class_stratum"], dropna=False).size().rename("graphs").reset_index(),
        "rung_selection": pd.DataFrame([{"table": f"wmatrix_r{r}", "dev_cv_loss": loss, "ridge": ridge, "pre_specified": f"wmatrix_r{r}" == primary,
                                         "lowest_dev_cv_loss": r == best_rung} for r, (_, ridge, loss) in mats.items()])})
    if sh is not None:
        with stage("sheaf test by stratum"):
            tables["d10_sheaf_by_stratum"] = sheaf_strata_table(scores, strata, best_rung, cfg, seed_parts)
    if with_r_report:
        with stage("r report"):
            by_event = dict(zip(strata["event_id"], strata["pin_stratum"].fillna(""), strict=True))
            events = r_table({"wgraph": (wg[0], wg[1].grounding), **({"sheaf": (sh[0], sh[1].grounding)} if sh is not None else {})}, [*dev, *test], by_event, cfg)
            tables["r_events"], tables["r_summary"] = events, r_summary(events, cfg)
    return Road1Output(fitted, scores, tables)
