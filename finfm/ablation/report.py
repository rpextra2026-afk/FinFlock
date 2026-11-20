"""The tables and figures of the ablation, built from whatever run folders exist (spec sections 8, 9, 12). A missing run is listed as "not computed", never filled.

Numbers and labels only. The text written here is checked against ``eval.report.forbidden_words``; mock or synthetic runs are refused.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from finfm.ablation import tier_a
from finfm.ablation.execute import EVENTS_FILE, LINEAGE_FILE, NET_FILE, SUMMARY_FILE
from finfm.ablation.plan import DRAW_SEPARATOR, FIXED_SUFFIX, Plan, RunSpec
from finfm.common.config import Config
from finfm.common.rng import make_rng
from finfm.eval.lw_bootstrap import sharpe_difference_test
from finfm.eval.portfolio_metrics import sharpe
from finfm.eval.stats import block_length, circular_block_indices, holm, sharpe_difference_se, smallest_detectable_gap

TIER_B_COLUMNS = ("cum_net_return", "sharpe_gross", "sharpe_net_headline", "max_drawdown_net_headline", "cvar_net", "hit_rate_net", "turnover_annual",
                  "annual_cost", "break_even_round_trip_bp", "alpha_annual", "t_alpha")
TIER_A_COLUMNS = ("n_events", "rank_ic_1d", "rank_ic_5d", "icir_5d", "t_newey_west_5d", "linked_r2_5d", "linked_slope_5d")
KEY_METRICS = ("rank_ic_5d", "linked_r2_5d", "sharpe_net_headline")
NOT_COMPUTED = "not computed"
TIER_B_DRAWS = "tier B draws"


class RunData:
    """What one run folder holds."""

    def __init__(self, folder: Path) -> None:
        self.folder = folder
        self.lineage = json.loads((folder / LINEAGE_FILE).read_text(encoding="utf-8"))
        self.events = pd.read_parquet(folder / EVENTS_FILE) if (folder / EVENTS_FILE).exists() else pd.DataFrame()
        self.summary = pd.read_csv(folder / SUMMARY_FILE).iloc[0].to_dict() if (folder / SUMMARY_FILE).exists() else {}
        self.net = pd.read_parquet(folder / NET_FILE) if (folder / NET_FILE).exists() else pd.DataFrame()

    def tier_a(self, cfg: Config) -> dict[str, float]:
        return tier_a.summarise(self.events, cfg) if len(self.events) else {}


def load_runs(root: Path, plan: Plan, cfg: Config) -> dict[str, RunData]:
    """``{run id: RunData}`` for every run of the plan that has a done file."""
    folder = root / str(cfg.get("ablation.outputs.runs_subfolder"))
    out = {}
    for spec in plan.all_specs():
        if (folder / spec.run_id / "done.json").exists():
            out[spec.run_id] = RunData(folder / spec.run_id)
    return out


def _guard(lineages: list[dict], cfg: Config) -> None:
    mock = [x["run_id"] for x in lineages if x.get("synthetic") or x.get("mock")]
    if mock:
        raise ValueError(f"mock or synthetic runs are refused by the report builder: {mock}")


def _levels_row(spec: RunSpec, data: RunData | None, full: RunData | None, cfg: Config) -> dict:
    row = {"run_id": spec.run_id, "key": spec.key, "name": spec.name, "kind": spec.kind, "tier": spec.tier, "mode": spec.mode, "refit": ",".join(spec.refit) or "none"}
    if spec.status == "not_computed":
        return {**row, "status": NOT_COMPUTED, "reason": spec.reason}
    if data is None:
        return {**row, "status": NOT_COMPUTED, "reason": "no finished run folder"}
    source = data
    if data.lineage.get("identical_to_full") and full is not None:
        source = full
        row["note"] = data.lineage["note"]
    row.update({"status": "run", **source.tier_a(cfg)})
    row.update({k: source.summary.get(k) for k in TIER_B_COLUMNS})
    for column in (c for c in source.summary if c.startswith("sharpe_net_") and c.endswith("bp")):
        row[column] = source.summary[column]
    row["seconds"] = data.lineage.get("seconds")
    return row


def _is_fixed_draw(spec: RunSpec) -> bool:
    """One draw of a forecast-level placebo with the fitted model kept: shown in the placebo distributions, not as a row of its own."""
    return FIXED_SUFFIX in spec.key and spec.draw is not None


def levels_table(plan: Plan, runs: dict[str, RunData], cfg: Config) -> pd.DataFrame:
    full_id = plan.spec(str(cfg.get("ablation.full.id"))).run_id
    full = runs.get(full_id)
    rows = [_levels_row(spec, runs.get(spec.run_id), full, cfg) for spec in plan.all_specs() if not _is_fixed_draw(spec)]
    rows += [_levels_row(spec, None, full, cfg) for spec in plan.not_computed()]
    return pd.DataFrame(rows)


# ---- differences from Full ----------------------------------------------------------------------------------------------------------------

def _paired_ic(run: pd.DataFrame, full: pd.DataFrame, column: str, cfg: Config, rng: np.random.Generator) -> dict[str, float]:
    """Mean per-event difference (run minus Full) of ``column`` with a date-clustered circular block bootstrap interval."""
    both = run[["event_id", "day0", column]].merge(full[["event_id", column]], on="event_id", suffixes=("_run", "_full")).dropna()
    daily = both.assign(d=both[f"{column}_run"] - both[f"{column}_full"]).groupby("day0")["d"].mean().sort_index()
    return _bootstrap_mean(daily.to_numpy(), cfg, rng) | {"n_events": float(len(both))}


def _bootstrap_mean(values: np.ndarray, cfg: Config, rng: np.random.Generator) -> dict[str, float]:
    if len(values) < 2:
        return {"difference": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan")}
    draws, level = int(cfg.get("eval.bootstrap.draws")), float(cfg.get("eval.bootstrap.ci_level"))
    index = circular_block_indices(len(values), block_length(len(values)), draws, rng)
    means = values[index].mean(axis=1)
    tail = (1.0 - level) / 2.0
    low, high = np.quantile(means, [tail, 1.0 - tail])
    centred = means - means.mean()
    p = float((np.abs(centred) >= abs(values.mean())).mean())
    return {"difference": float(values.mean()), "ci_low": float(low), "ci_high": float(high), "p_value": p}


def _paired_ratio(run: pd.DataFrame, full: pd.DataFrame, h: int, cfg: Config, rng: np.random.Generator) -> dict[str, float]:
    """Difference of the pooled linked-stock R2 (run minus Full), bootstrapped over days (events of a day move together)."""
    cols = [f"linked_sse_{h}d", f"linked_sst_{h}d"]
    both = run[["event_id", "day0", *cols]].merge(full[["event_id", *cols]], on="event_id", suffixes=("_run", "_full"))
    days = both.groupby("day0")[[f"{c}_{s}" for c in cols for s in ("run", "full")]].sum().sort_index()
    if len(days) < 2:
        return {"difference": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "p_value": float("nan")}

    def r2(frame: np.ndarray) -> float:
        sse_run, sse_full, sst_run, sst_full = frame.sum(axis=0)
        return (1.0 - sse_run / sst_run if sst_run > 0 else np.nan) - (1.0 - sse_full / sst_full if sst_full > 0 else np.nan)

    values = days[[f"{cols[0]}_run", f"{cols[0]}_full", f"{cols[1]}_run", f"{cols[1]}_full"]].to_numpy()
    index = circular_block_indices(len(values), block_length(len(values)), int(cfg.get("eval.bootstrap.draws")), rng)
    draws = np.array([r2(values[i]) for i in index])
    level = float(cfg.get("eval.bootstrap.ci_level"))
    tail = (1.0 - level) / 2.0
    point = r2(values)
    low, high = np.nanquantile(draws, [tail, 1.0 - tail])
    return {"difference": float(point), "ci_low": float(low), "ci_high": float(high), "p_value": float((np.abs(draws - np.nanmean(draws)) >= abs(point)).mean())}


def deltas_table(plan: Plan, runs: dict[str, RunData], cfg: Config) -> pd.DataFrame:
    """Every run's difference from Full: rank IC (1 and 5 days), linked-stock R2, and net Sharpe at each cost, each with an interval and a Holm-adjusted p per family."""
    full_spec = plan.spec(str(cfg.get("ablation.full.id")))
    full = runs.get(full_spec.run_id)
    if full is None:
        return pd.DataFrame()
    periods, draws, level = int(cfg.get("eval.annualisation_days")), int(cfg.get("eval.bootstrap.draws")), float(cfg.get("eval.bootstrap.ci_level"))
    rows = []
    for spec in plan.all_specs():
        data = runs.get(spec.run_id)
        if data is None or spec.run_id == full_spec.run_id or data.lineage.get("identical_to_full"):
            continue
        rng = make_rng("ablation", "report", spec.run_id)
        family = plan.switches.get(spec.key.split(":")[0].split("@")[0], {}).get("family", spec.kind)
        if len(data.events) and len(full.events):
            for h in (int(x) for x in cfg.get("ablation.rank_ic_horizons")):
                rows.append({"run_id": spec.run_id, "family": family, "metric": f"rank_ic_{h}d", **_paired_ic(data.events, full.events, f"ic_{h}d", cfg, rng)})
                rows.append({"run_id": spec.run_id, "family": family, "metric": f"linked_r2_{h}d", **_paired_ratio(data.events, full.events, h, cfg, rng), "n_events": float(len(data.events))})
        if len(data.net) and len(full.net):
            common = data.net.index.intersection(full.net.index)
            for column in (c for c in data.net.columns if c.endswith("bp")):
                a, b = data.net.loc[common, column].to_numpy(), full.net.loc[common, column].to_numpy()
                test = sharpe_difference_test(a, b, periods, draws, level, rng)
                rho = float(np.corrcoef(a, b)[0, 1])
                years = len(common) / periods
                se = sharpe_difference_se(sharpe(pd.Series(a), periods), sharpe(pd.Series(b), periods), rho, years, periods)
                rows.append({"run_id": spec.run_id, "family": family, "metric": f"net_sharpe_{column}", "difference": test.delta_annual, "ci_low": test.ci_low,
                             "ci_high": test.ci_high, "p_value": test.p_value, "n_days": float(len(common)), "correlation_with_full": rho,
                             "standard_error": se, "smallest_readable_gap": smallest_detectable_gap(se, level)})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    table["p_holm_family"] = np.nan
    for (family, metric), part in table.groupby(["family", "metric"]):
        adjusted = holm({str(i): v for i, v in part["p_value"].dropna().items()})
        for i, v in adjusted.items():
            table.loc[int(i), "p_holm_family"] = v
    return table


def noise_floor_table(plan: Plan, runs: dict[str, RunData], cfg: Config) -> pd.DataFrame:
    """Full refitted with other fit seeds: mean and sd of every metric, and the sd of the difference from the first Full run."""
    ids = [s.run_id for s in plan.all_specs() if s.kind == "noise" and s.run_id in runs]
    full = runs.get(plan.spec(str(cfg.get("ablation.full.id"))).run_id)
    if not ids or full is None:
        return pd.DataFrame()
    values = pd.DataFrame([{**runs[i].tier_a(cfg), **{k: runs[i].summary.get(k) for k in (*TIER_B_COLUMNS, *[c for c in runs[i].summary if str(c).startswith("sharpe_net_")])}}
                           for i in ids])
    reference = {**full.tier_a(cfg), **{k: full.summary.get(k) for k in values.columns}}
    rows = [{"metric": c, "n_runs": float(values[c].notna().sum()), "mean": values[c].mean(), "sd": values[c].std(ddof=1) if values[c].notna().sum() > 1 else float("nan"),
             "full": reference.get(c), "sd_of_difference_from_full": (values[c] - reference.get(c, np.nan)).std(ddof=1) if values[c].notna().sum() > 1 else float("nan")}
            for c in values.columns]
    return pd.DataFrame(rows)


def placebo_table(plan: Plan, runs: dict[str, RunData], cfg: Config) -> pd.DataFrame:
    """For each placebo switch, its draws: mean, sd, 5th and 95th percentile, and the share of draws at or above Full and at or above the no-signal floor."""
    full = runs.get(plan.spec(str(cfg.get("ablation.full.id"))).run_id)
    floor = next((runs[s.run_id] for s in plan.all_specs() if s.name == "no_signal" and s.run_id in runs), None)
    low, high = cfg.get("ablation.power.tier_a_draw_summary")
    rows = []
    for switch, entry in plan.switches.items():
        if not entry.get("placebo"):
            continue
        for label, prefix in ((TIER_B_DRAWS, f"{switch}{DRAW_SEPARATOR}"), ("forecast-level draws, fitted model kept", f"{switch}{FIXED_SUFFIX}{DRAW_SEPARATOR}")):
            draws = [r for s in plan.all_specs() if s.placebo and s.key.startswith(prefix) for r in [runs.get(s.run_id)] if r is not None]
            if not draws:
                continue
            for metric in KEY_METRICS:
                series = np.array([({**d.tier_a(cfg), **d.summary}).get(metric, np.nan) for d in draws], dtype=float)
                series = series[np.isfinite(series)]
                if not len(series):
                    continue
                ref = ({**full.tier_a(cfg), **full.summary}).get(metric, np.nan) if full else np.nan
                fl = ({**floor.tier_a(cfg), **floor.summary}).get(metric, np.nan) if floor else np.nan
                rows.append({"placebo": entry["name"], "draws": label, "metric": metric, "n_draws": float(len(series)), "mean": series.mean(),
                             "sd": series.std(ddof=1) if len(series) > 1 else np.nan, "p_low": np.quantile(series, low), "p_high": np.quantile(series, high),
                             "full": ref, "no_signal_floor": fl,
                             "share_at_or_above_full": float((series >= ref).mean()) if np.isfinite(ref) else np.nan,
                             "share_at_or_above_floor": float((series >= fl).mean()) if np.isfinite(fl) else np.nan})
    return pd.DataFrame(rows)


def strata_table(plan: Plan, runs: dict[str, RunData], cfg: Config) -> pd.DataFrame:
    frames = []
    for spec in plan.all_specs():
        data = runs.get(spec.run_id)
        if data is not None and len(data.events) and not _is_fixed_draw(spec):
            frames.append(tier_a.by_cut(data.events, cfg).assign(run_id=spec.run_id))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def power_table(deltas: pd.DataFrame) -> pd.DataFrame:
    if deltas.empty or "smallest_readable_gap" not in deltas:
        return pd.DataFrame()
    return deltas[deltas["metric"].str.startswith("net_sharpe")][["run_id", "metric", "n_days", "correlation_with_full", "standard_error", "smallest_readable_gap"]]


def persona_curve(plan: Plan, runs: dict[str, RunData], cfg: Config) -> pd.DataFrame:
    rows = []
    for spec in plan.all_specs():
        if spec.kind == "persona" and spec.run_id in runs:
            rows.append({"fraction": spec.fraction, **runs[spec.run_id].tier_a(cfg)})
    full = runs.get(plan.spec(str(cfg.get("ablation.full.id"))).run_id)
    if full is not None and rows:
        rows.append({"fraction": 1.0, **full.tier_a(cfg)})
    return pd.DataFrame(rows).sort_values("fraction") if rows else pd.DataFrame()


def ladder_table(plan: Plan, levels: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Table 2: the rungs with their numbers and the change from the previous rung that has one."""
    by_key = levels.set_index("key") if len(levels) else pd.DataFrame()
    rows, previous = [], None
    for entry in cfg.get("ablation.ladder_table"):
        key = entry["key"]
        base = {"rung": entry["rung"], "key": key, "built_from": plan.ladder.get(key, {}).get("label", plan.switches.get(key, {}).get("name", key))}
        row = by_key.loc[key] if len(by_key) and key in by_key.index else None
        if row is None or row.get("status") != "run":
            rows.append({**base, "status": NOT_COMPUTED, "reason": None if row is None else row.get("reason")})
            continue
        values = {c: row.get(c) for c in ("rank_ic_5d", "linked_r2_5d", "sharpe_net_headline")}
        change = {f"change_{c}": (values[c] - previous[c]) if previous is not None and pd.notna(values[c]) and pd.notna(previous[c]) else np.nan for c in values}
        rows.append({**base, "status": "run", **values, **change})
        previous = values
    return pd.DataFrame(rows)


def forest_figure(ladder: pd.DataFrame, placebos: pd.DataFrame, target: Path, cfg: Config) -> None:
    """Layout Fig 4: the change in net Sharpe over the rung below, with the placebo rows in grey. The data are written beside it as a csv."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    palette = cfg.get("eval.report.palette")
    size = cfg.get("eval.report.figure.size")
    shown = ladder[ladder["status"] == "run"].dropna(subset=["change_sharpe_net_headline"]) if "change_sharpe_net_headline" in ladder else pd.DataFrame()
    fig, ax = plt.subplots(figsize=tuple(size), dpi=int(cfg.get("eval.report.figure_dpi")))
    labels = [f"rung {r}: {k}" for r, k in zip(shown.get("rung", []), shown.get("key", []), strict=False)]
    ax.barh(range(len(shown)), shown["change_sharpe_net_headline"] if len(shown) else [], color=palette[0])
    extra = placebos[(placebos["metric"] == "sharpe_net_headline") & (placebos["draws"] == TIER_B_DRAWS)] if len(placebos) else pd.DataFrame()
    for i, row in enumerate(extra.itertuples(), start=len(shown)):
        ax.barh(i, row.mean - row.full, xerr=[[max(row.mean - row.p_low, 0.0)], [max(row.p_high - row.mean, 0.0)]], color="grey")
        labels.append(f"placebo {row.placebo}")
    ax.set_yticks(range(len(labels)), labels, fontsize=int(cfg.get("eval.report.figure.legend_size")))
    ax.set_xlabel("change in net Sharpe over the rung below (placebo rows: draw mean minus Full)")
    fig.tight_layout()
    fig.savefig(target, dpi=int(cfg.get("eval.report.figure_dpi")))
    plt.close(fig)
    pd.concat([shown.assign(row_type="rung"), extra.assign(row_type="placebo")], ignore_index=True).to_csv(target.with_suffix(".csv"), index=False)


def build_report(root: Path, cfg: Config, bucket: str, llm: str, reader_mode: str) -> dict[str, pd.DataFrame]:
    """Reads the run folders under ``root`` and writes every table and figure beside them; returns the tables."""
    plan = Plan(cfg, bucket, llm)
    runs = load_runs(root, plan, cfg)
    _guard([r.lineage for r in runs.values()], cfg)
    levels = levels_table(plan, runs, cfg)
    deltas = deltas_table(plan, runs, cfg)
    placebos = placebo_table(plan, runs, cfg)
    ladder = ladder_table(plan, levels, cfg)
    tables = {"ablation_levels": levels, "ablation_deltas": deltas, "ladder": ladder, "noise_floor": noise_floor_table(plan, runs, cfg),
              "placebo_distributions": placebos, "strata": strata_table(plan, runs, cfg), "power": power_table(deltas),
              "persona_count_curve": persona_curve(plan, runs, cfg),
              "run_manifest": pd.DataFrame([{k: v for k, v in r.lineage.items() if k in ("run_id", "key", "name", "kind", "tier", "mode", "refit", "config_hash", "seconds",
                                                                                    "n_forecasts", "n_skipped", "identical_to_full")} for r in runs.values()])}
    for name, table in tables.items():
        if not table.empty:
            table.to_csv(root / f"{name}.csv", index=False)
    if not levels.empty:
        (root / "ablation_levels.md").write_text(_markdown(levels, cfg, bucket, llm, reader_mode, tables["noise_floor"]), encoding="utf-8")
    if not ladder.empty:
        forest_figure(ladder, placebos, root / "fig4_forest.png", cfg)
    curve = tables["persona_count_curve"]
    if not curve.empty and "rank_ic_5d" in curve:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=tuple(cfg.get("eval.report.figure.size")), dpi=int(cfg.get("eval.report.figure_dpi")))
        ax.plot(curve["fraction"], curve["rank_ic_5d"], marker="o", color=cfg.get("eval.report.palette")[0])
        ax.set_xlabel("share of the personas kept")
        ax.set_ylabel("event rank IC, 5 days")
        fig.tight_layout()
        fig.savefig(root / "persona_count_curve.png")
        plt.close(fig)
    (root / "lineage.json").write_text(json.dumps({"bucket": bucket, "llm": llm, "reader_mode": reader_mode, "label": str(cfg.get("ablation.labels.in_training")),
                                                   "runs_found": sorted(runs), "config_hash": cfg.hash}, indent=1), encoding="utf-8")
    return tables


def _markdown(levels: pd.DataFrame, cfg: Config, bucket: str, llm: str, reader_mode: str, noise: pd.DataFrame) -> str:
    """The levels table as text. The caption carries bucket, LLM, mode, the in-training label, the number of runs and the noise-floor sd of the headline Sharpe."""
    decimals = int(cfg.get("eval.report.decimals"))
    sd = noise.loc[noise["metric"] == "sharpe_net_headline", "sd"].iloc[0] if len(noise) and (noise["metric"] == "sharpe_net_headline").any() else np.nan
    caption = (f"Ablation levels. Bucket {bucket}, LLM {llm}, reader mode {reader_mode}. {cfg.get('ablation.labels.in_training')}. "
               f"Runs: {int((levels['status'] == 'run').sum())}; {NOT_COMPUTED}: {int((levels['status'] != 'run').sum())}. Noise-floor sd of the net Sharpe: {sd:.{decimals}f}.")
    text = caption + "\n\n" + _table_text(levels, decimals) + "\n"
    words = [w for w in cfg.get("eval.report.forbidden_words") if f" {w} " in f" {text.lower()} "]
    if words:
        raise ValueError(f"the generated text contains words the report must not use: {words}")
    return text


def _table_text(frame: pd.DataFrame, decimals: int) -> str:
    """A pipe table (no extra package): floats with ``decimals`` places, missing values as empty cells."""
    def cell(value) -> str:
        if isinstance(value, float):
            return "" if np.isnan(value) else f"{value:.{decimals}f}"
        return "" if value is None else str(value)

    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)
