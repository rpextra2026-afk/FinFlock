"""Families J, K and L: sheaf diagnostics, data and cleaning diagnostics, leakage and sanity checks. Numbers and counts only."""

from __future__ import annotations

import json
import platform
import resource
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch

from oryxis.build import metrics as M
from oryxis.common.config import Config
from oryxis.road1.arms import SheafModel, assemble_laplacian, ripple
from oryxis.road1.fit import standardise
from oryxis.road1.graph import EventGraph

ROOT = "build.diagnostics"
NAN = float("nan")
KIB = 1024.0
REPEAT_CAP = 10                 # repeat counts above this are shown together


def _hops(graph: EventGraph, max_hops: int) -> list[int]:
    """Number of free nodes first reached at hop 1, 2, ... from the pinned nodes (undirected edges)."""
    reached = np.zeros(graph.n_nodes, dtype=bool)
    reached[: graph.n_pinned] = True
    frontier = reached.copy()
    counts = []
    for _ in range(max_hops):
        nxt = np.zeros_like(reached)
        for a, b in zip(graph.src, graph.dst, strict=True):
            if frontier[a] and not reached[b]:
                nxt[b] = True
            if frontier[b] and not reached[a]:
                nxt[a] = True
        reached |= nxt
        counts.append(int(nxt.sum()))
        frontier = nxt
        if not nxt.any():
            break
    return counts + [0] * (max_hops - len(counts))


def sheaf_diagnostics(model: SheafModel, grounding: float, graphs: Sequence[EventGraph], scale: np.ndarray, link_types: Sequence[str], cfg: Config, collector: M.Collector, configs: pd.DataFrame | None, fit_seconds: Mapping[str, float]) -> dict[str, Any]:
    """Learned Delta values and norms, Laplacian spectrum, Dirichlet energy before and after the extension, harmonic residual, reach and coverage."""
    out: dict[str, Any] = {"parameters": int(model.n_parameters()), "grounding": float(grounding)}
    d = model.d
    for t, name in enumerate(link_types):
        du, dv = model.delta_u[t, 0].detach().numpy(), model.delta_v[t, 0].detach().numpy()
        collector.add("all", "sheaf_delta", name, "frobenius_norm_delta_u", float(np.linalg.norm(du)), method="sheaf")
        collector.add("all", "sheaf_delta", name, "frobenius_norm_delta_v", float(np.linalg.norm(dv)), method="sheaf")
        for a in range(d):
            for b in range(d):
                collector.add("all", "sheaf_delta", name, f"delta_u_{a}{b}", float(du[a, b]), method="sheaf")
    sample = list(graphs)[: int(cfg.get(f"{ROOT}.graph_sample"))]
    cap = int(cfg.get(f"{ROOT}.max_laplacian_size"))
    k = int(cfg.get(f"{ROOT}.smallest_eigenvalues"))
    hop_counts, shares, e_before, e_after, resid, gaps, eig_rows = [], [], [], [], [], [], []
    with torch.no_grad():
        for g in standardise(sample, scale):
            if not g.n_free:
                continue
            shares.append(g.n_pinned / g.n_nodes)
            hop_counts.append(_hops(g, int(cfg.get(f"{ROOT}.hops"))))
            if g.n_nodes * d > cap:
                continue
            L = model.laplacian(g, 0)
            p = g.n_pinned * d
            s = torch.as_tensor(g.s, dtype=L.dtype).reshape(-1)
            x_f = ripple(L, g.n_pinned, torch.as_tensor(g.s, dtype=L.dtype), grounding).reshape(-1)
            x0 = torch.cat([s, torch.zeros(L.shape[0] - p, dtype=L.dtype)])
            x1 = torch.cat([s, x_f])
            e_before.append(float(x0 @ L @ x0))
            e_after.append(float(x1 @ L @ x1))
            resid.append(float(torch.linalg.norm(L[p:, p:] @ x_f + L[p:, :p] @ s)))
            ev = torch.linalg.eigvalsh(L[p:, p:])
            gaps.append(float(ev[0]))
            eig_rows.append(ev[:k].numpy())
    if hop_counts:
        hops = np.array(hop_counts)
        for h in range(hops.shape[1]):
            collector.add("all", "sheaf_reach", f"hop_{h + 1}", "free_nodes_mean", float(hops[:, h].mean()), n=len(hops), method="sheaf")
        collector.add("all", "sheaf_reach", "all", "share_nodes_pinned_mean", float(np.mean(shares)), n=len(shares), method="sheaf")
    for name, values in (("dirichlet_energy_before_extension", e_before), ("dirichlet_energy_after_extension", e_after), ("harmonic_residual_ungrounded", resid), ("laplacian_ff_smallest_eigenvalue", gaps)):
        if values:
            collector.add("all", "sheaf_laplacian", "sample", name, float(np.mean(values)), float(np.min(values)), float(np.max(values)), len(values), method="sheaf")
    if eig_rows:
        width = min(len(r) for r in eig_rows)
        means = np.mean([r[:width] for r in eig_rows], axis=0)
        for i, v in enumerate(means):
            collector.add("all", "sheaf_laplacian", "sample", f"eigenvalue_{i + 1}_mean", float(v), n=len(eig_rows), method="sheaf")
    if configs is not None and len(configs):
        for r in configs.itertuples():
            if hasattr(r, "arm") and hasattr(r, "val_loss"):
                collector.add("all", "ridge_path", f"{r.arm}|grounding={getattr(r, 'grounding', '')}|penalty={getattr(r, 'penalty_mu', '')}", "cv_loss", float(r.val_loss), method="sheaf")
    for stage, seconds in fit_seconds.items():
        collector.add("all", "fit_time", stage, "seconds", float(seconds), method="sheaf")
    out.update({"graphs_in_sample": len(sample), "laplacian_graphs": len(e_before)})
    return out


def coverage_per_day(rows: pd.DataFrame, n_nodes: int, collector: M.Collector) -> None:
    """Nodes reached by at least one event per hold-out day, as a share of all stocks."""
    reach = rows.groupby("day")["node"].nunique() / float(n_nodes)
    for name, value in (("mean", reach.mean()), ("median", reach.median()), ("min", reach.min()), ("max", reach.max())):
        collector.add("all", "coverage_per_day", name, "share_of_stocks_reached", float(value), n=int(len(reach)), method="sheaf")


# ---- K. data and cleaning ------------------------------------------------------------------------------------------------------------------

def data_diagnostics(report: Mapping[str, Any], tape: pd.DataFrame, clean: pd.DataFrame) -> dict[str, Any]:
    """Counts after every cleaning step, parse rate, 'other' rate, pins non-empty rate, repeat-count distribution, trust weights, direction share, unmatched entities."""
    steps = {s["step"]: s for s in report["steps"]}
    parsed = tape["parse_ok"].astype(float) if "parse_ok" in tape else pd.Series(dtype=float)
    called = tape[tape["status"] == "ok"]
    items = clean.drop_duplicates("unit_id")
    pins = clean[clean["node_type"] == "company"]
    return {"steps": report["steps"], "tape_rows": int(len(tape)), "called": int(len(called)), "parse_rate": float(parsed[tape["status"] == "ok"].mean()) if len(parsed) else NAN,
            "other_rate_called": float((called["cause_top"] == "other").mean()) if len(called) else NAN,
            "pins_nonempty_rate_called": float(called["reading_pins_json"].map(lambda v: isinstance(v, str) and v not in ("", "[]")).mean()) if len(called) else NAN,
            "repeat_count_distribution": {str(k): int(v) for k, v in items["repeat_count"].clip(upper=REPEAT_CAP).value_counts().sort_index().items()},
            "trust_weights": steps.get("trust", {}).get("trust_weights", {}), "direction_share": {str(k): float(v) for k, v in pins["direction"].value_counts(normalize=True).items()},
            "unmatched_entities": {k: steps["entity_check"].get(k) for k in ("pins_seen", "pins_unmatched", "distinct_unmatched_nodes", "unmatched_examples")}}


def tape_agreement(cleaned: Mapping[str, pd.DataFrame]) -> dict[str, Any]:
    """Agreement across tapes of the same bucket: same item, same event type, same cause, same direction."""
    names = sorted(cleaned)
    out: dict[str, Any] = {}
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            ia, ib = (cleaned[a].drop_duplicates("unit_id").set_index("unit_id"), cleaned[b].drop_duplicates("unit_id").set_index("unit_id"))
            common = ia.index.intersection(ib.index)
            if not len(common):
                continue
            da = cleaned[a][cleaned[a]["node_type"] == "company"].groupby("unit_id")["direction"].first()
            db = cleaned[b][cleaned[b]["node_type"] == "company"].groupby("unit_id")["direction"].first()
            both = da.index.intersection(db.index)
            out[f"{a}|{b}"] = {"items_in_both": int(len(common)), "same_event_type": float((ia.loc[common, "event_type_final"] == ib.loc[common, "event_type_final"]).mean()), "same_cause": float((ia.loc[common, "cause_top"] == ib.loc[common, "cause_top"]).mean()),
                               "same_lane": float((ia.loc[common, "lane"] == ib.loc[common, "lane"]).mean()), "items_with_company_pin_in_both": int(len(both)), "same_direction": float((da.loc[both] == db.loc[both]).mean()) if len(both) else NAN}
    return out


# ---- L. leakage and sanity -----------------------------------------------------------------------------------------------------------------

def leakage_checks(run_cfg: Mapping[str, Any], clean: pd.DataFrame, rows: pd.DataFrame, dev_days: Sequence[pd.Timestamp], test_days: Sequence[pd.Timestamp], counts: Mapping[str, Any], cfg: Config, open_test: bool) -> dict[str, Any]:
    """Counts of violations (zero is what is wanted): sessions, split overlap, rows after the hold-out end, row reconciliation."""
    hold_end, fit_end, hold_start = (pd.Timestamp(run_cfg[k]) for k in ("holdout_end", "fit_end", "holdout_start"))
    session = pd.to_datetime(clean["session_date"])
    avail = pd.to_datetime(clean["ts_utc"]).dt.tz_localize(None).dt.normalize()
    out = {"items_with_session_before_availability_date": int((session < avail - pd.Timedelta(days=1)).sum()), "fit_days_after_fit_end": int(sum(d > fit_end for d in dev_days)),
           "holdout_days_before_holdout_start": int(sum(d < hold_start for d in test_days)), "fit_and_holdout_days_in_common": int(len(set(dev_days) & set(test_days))),
           "cleaned_rows_after_holdout_end": int((session > hold_end).sum()) if not open_test else None, "scored_rows_after_holdout_end": int((pd.to_datetime(rows["day"]) > hold_end).sum()) if len(rows) and not open_test else None,
           "scored_rows_before_holdout_start": int((pd.to_datetime(rows["day"]) < hold_start).sum()) if len(rows) else 0, "row_reconciliation": dict(counts),
           "seed": int(cfg.get("build.metrics.seed")), "config_hash": cfg.hash, "python": sys.version.split()[0], "numpy": np.__version__, "pandas": pd.__version__, "torch": torch.__version__, "platform": platform.platform(),
           "peak_memory_mb": float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / (KIB * KIB if sys.platform == "darwin" else KIB)}
    return out


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str))
