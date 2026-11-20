"""The sheaf test by stratum with a multiple-testing correction (spec 8.4, D10).

Strata come from the news side only: the number of pinned assets of the event (``single_pin``: expected tie, a free placebo; ``multi_pin``:
where a sheaf can differ from a table) and, as an exploratory family, the node type of a single pin. The tables the sheaf is compared with
are the pre-specified rung 3 of the matrix ladder and the weighted graph; the rung chosen by dev cross-validation is reported beside them as a
sensitivity row and never replaces the pre-specified one. Holm's step-down correction is applied inside each declared family. Strata with
fewer test events than the configured floor are counted and not tested (spec 11.8). Nothing here is tuned on test results.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from finfm.common.config import Config
from finfm.common.rng import make_rng
from finfm.eval.stats import holm
from finfm.road1.sheaf_test import d10_regression

POOLED = "all"
FAMILY_PRIMARY, FAMILY_EXPLORATORY = "primary", "exploratory"
ROLE_PRIMARY, ROLE_GRAPH, ROLE_SENSITIVITY = "primary_table", "graph_table", "sensitivity_table"
NODE_CLASS_PREFIX = "pin_"
MIXED_TYPES = "mixed"
COLUMNS = ("stratum", "dial", "role", "table", "family", "n_events", "n_dates", "tested", "reason", "beta1", "beta2", "se_cluster", "ci_low", "ci_high",
           "p_two_sided", "holm_group", "holm_m", "p_holm", "meets_rule")


def pin_stratum(n_pinned: int, cfg: Config) -> str | None:
    """The configured pin-count stratum of an event, or ``None`` when no stratum covers its pin count."""
    for name, bounds in dict(cfg.get("cpu.road1_eval.sheaf_test.pin_strata")).items():
        high = bounds["max_pins"]
        if int(bounds["min_pins"]) <= n_pinned and (high is None or n_pinned <= int(high)):
            return str(name)
    return None


def event_strata(n_pinned: Mapping[str, int], pin_types: Mapping[str, Sequence[str]], cfg: Config) -> pd.DataFrame:
    """One row per event: ``n_pinned``, ``pin_stratum`` and ``node_class_stratum`` (the type of the single pin, ``mixed`` for several)."""
    rows = []
    for event_id, n in n_pinned.items():
        types = set(pin_types.get(event_id, ()))
        node_class = f"{NODE_CLASS_PREFIX}{next(iter(types))}" if len(types) == 1 and n == 1 else f"{NODE_CLASS_PREFIX}{MIXED_TYPES}"
        rows.append({"event_id": event_id, "n_pinned": int(n), "pin_stratum": pin_stratum(int(n), cfg), "node_class_stratum": node_class})
    return pd.DataFrame(rows, columns=["event_id", "n_pinned", "pin_stratum", "node_class_stratum"])


def _row(stratum: str, dial: str, role: str, table: str, family: str, n_events: int, n_dates: int, tested: bool, reason: str, result=None) -> dict:
    row = dict.fromkeys(COLUMNS)
    row.update({"stratum": stratum, "dial": dial, "role": role, "table": table, "family": family, "n_events": n_events, "n_dates": n_dates,
                "tested": tested, "reason": reason})
    if result is not None:
        row.update({"beta1": result.beta1, "beta2": result.beta2, "se_cluster": result.se_cluster, "ci_low": result.ci_low,
                    "ci_high": result.ci_high, "p_two_sided": result.p_two_sided})
    return row


def sheaf_strata_table(scores: pd.DataFrame, strata: pd.DataFrame, best_rung: int, cfg: Config, seed_parts: tuple) -> pd.DataFrame:
    """D10 for every (stratum, dial, table). ``scores`` is the long score frame of ``road1_real.long_scores`` (needs ``event_id, date, dial, y,
    sheaf`` and one column per table). Returns one row per test, with Holm-adjusted p-values inside each family."""
    spec = cfg.get("cpu.road1_eval.sheaf_test")
    draws, level = int(cfg.get("road1.bootstrap.draws")), float(cfg.get("road1.confidence_level"))
    floor, min_clusters = int(spec["underpowered_below_events"]), int(spec["min_clusters"])
    tables = {ROLE_PRIMARY: str(spec["primary_table"]), ROLE_GRAPH: str(spec["graph_table"]), ROLE_SENSITIVITY: f"wmatrix_r{best_rung}"}
    primary_strata = set(spec["primary_family"])
    groups: dict[str, set[str]] = {str(spec["pooled_stratum"]): set(scores["event_id"])}
    for name in dict(spec["pin_strata"]):
        groups[name] = set(strata.loc[strata["pin_stratum"] == name, "event_id"]) & set(scores["event_id"])
    if bool(spec["node_class_strata"]):
        for name, part in strata[strata["n_pinned"] == 1].groupby("node_class_stratum"):
            groups[str(name)] = set(part["event_id"]) & set(scores["event_id"])
    rows = []
    for stratum, ids in groups.items():
        in_stratum = scores[scores["event_id"].isin(ids)]
        for dial, part in in_stratum.groupby("dial"):
            n_events, n_dates = int(part["event_id"].nunique()), int(part["date"].nunique())
            for role, table in tables.items():
                family = FAMILY_PRIMARY if stratum in primary_strata and role in (ROLE_PRIMARY, ROLE_GRAPH) else FAMILY_EXPLORATORY
                if table not in part.columns:
                    rows.append(_row(stratum, str(dial), role, table, family, n_events, n_dates, False, "table not fitted"))
                elif n_events < floor:
                    rows.append(_row(stratum, str(dial), role, table, family, n_events, n_dates, False, f"underpowered: {n_events} events < {floor}"))
                elif n_dates < min_clusters:
                    rows.append(_row(stratum, str(dial), role, table, family, n_events, n_dates, False, f"too few dates: {n_dates} < {min_clusters}"))
                else:
                    rng = make_rng(*seed_parts, "d10", stratum, str(dial), role)
                    result = d10_regression(part["y"].to_numpy(), part[table].to_numpy(), part["sheaf"].to_numpy(), part["date"].astype(str).to_numpy(), rng, draws, level)
                    rows.append(_row(stratum, str(dial), role, table, family, n_events, n_dates, True, "", result))
    out = pd.DataFrame(rows, columns=COLUMNS)
    out["holm_group"] = np.where(out["family"] == FAMILY_PRIMARY, FAMILY_PRIMARY, out["role"].astype(str) + "/" + out["stratum"].map(_exploratory_kind(spec)).astype(str))
    out["holm_m"], out["p_holm"] = np.nan, np.nan
    for _, index in out[out["tested"].astype(bool)].groupby("holm_group").groups.items():
        adjusted = holm({int(i): float(out.at[i, "p_two_sided"]) for i in index})
        for i, value in adjusted.items():
            out.at[i, "p_holm"], out.at[i, "holm_m"] = value, len(index)
    alpha = float(spec["alpha"])
    out["meets_rule"] = out["tested"].astype(bool) & (out["beta2"].astype(float) > 0.0) & (out["p_holm"].astype(float) < alpha)
    return out


def _exploratory_kind(spec):
    """Strata of the exploratory family are corrected together by kind: pooled, pin-count (outside the primary family), node class."""
    pin_names = set(dict(spec["pin_strata"]))
    pooled = str(spec["pooled_stratum"])

    def kind(stratum: str) -> str:
        if stratum == pooled:
            return "pooled"
        return "pin_count" if stratum in pin_names else "node_class"

    return kind
