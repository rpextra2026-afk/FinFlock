"""The report a rival notebook builds after its run: score the tape with the project's own scorer, then draw charts and zip everything.

``build_report`` never raises: the run's tape bundle is already saved before it is called, so a failure here only costs the report (the
scoring can be repeated on the Mac with ``scripts/s9_score_<job>.py``). Scoring uses the same scorers, windows, cost grid and metrics as the
main framework's evaluation, so the numbers sit in the same long format (``summary_*.csv``). A progress line is printed every
``kaggle.heartbeat_minutes`` minutes while it works.
"""

from __future__ import annotations

import json
import re
import shutil
import threading
import time
import traceback
import zipfile
from pathlib import Path

import pandas as pd

from finfm.common.config import load_config
from finfm.common.schema import STATUS
from finfm.compute.bundle import make_job_id

SECONDS_PER_MINUTE = 60
PNL_PREFIX = "pnl_"
SHARPE = "sharpe"
PORTFOLIO = "portfolio"
PANEL_WIDTH = 5.2
PANEL_HEIGHT = 4.2
MAX_VARIANTS_SHOWN = 6
BAR_SPAN = 0.8
LINE_WIDTH = 0.6
GRID_LABEL = 6
SMALL_FONT = 6
HALF = 0.5
WIDE_SIZE = (10, 7)
TALL_SIZE = (10, 4.5)
MAX_CURVES = 8
BAR_WIDTH = 0.27
THIN_LINE = 1
HEALTH_SIZE = (14, 4)
HIST_BINS = 40
SIGNAL_PANEL_WIDTH = 3.4
SIGNAL_PANEL_HEIGHT = 3.8
MAX_SIGNAL_PANELS = 6
MAX_SIGNAL_ROWS = 24
TITLE_FONT = 8
ROTATION = 90
MAX_ERROR_CHARS = 600
MAX_LOG_CHARS = 800
HEALTH_PANELS = 3
MAX_REJECTED_SHOWN = 5


class Heartbeat:
    """Prints the last message again whenever ``interval`` seconds pass without a new one."""

    def __init__(self, interval: float, log=print) -> None:
        self.started, self.interval, self.last, self.stop, self.log = time.time(), interval, "starting", threading.Event(), log
        threading.Thread(target=self._beat, daemon=True).start()

    def __call__(self, message: str) -> None:
        self.last = message
        self.log(f"[report {time.time() - self.started:6.0f}s] {message}", flush=True)

    def _beat(self) -> None:
        while not self.stop.wait(self.interval):
            self.log(f"[report {time.time() - self.started:6.0f}s] still working: {self.last}", flush=True)


def newest_bundle(out_dir: Path, job: str, bucket: str, llm: str) -> Path | None:
    found = sorted(out_dir.rglob(f"finfm_{job}__{bucket}__{llm}_*.bundle"), key=lambda p: p.stat().st_mtime)
    return found[-1] if found else None


def merged_bundle(run, input_root: Path, out_dir: Path, cfg, log) -> Path:
    """One job-level bundle from every shard bundle of this run found under ``input_root`` (and ``out_dir``), each unit once."""
    from finfm.compute.bundle import merge_bundles, scan_bundles

    job_id = make_job_id(run.job, run.bucket, run.llm)
    scan = scan_bundles([input_root, out_dir], job_id)
    if not scan.valid:
        raise FileNotFoundError(f"no valid bundle of {job_id} attached; rejected: {[(str(p), why) for p, why in scan.rejected][:MAX_REJECTED_SHOWN]}")
    shards = sorted({ref.manifest.get("shard_id") for ref in scan.valid}, key=str)
    log(f"merging {len(scan.valid)} bundle(s) of shards {shards}; rejected {len(scan.rejected)}, duplicates {len(scan.duplicates)}")
    meta = {**scan.valid[0].manifest.get("meta", {}), "merged": True}
    target = out_dir / "merged_tape"
    target.mkdir(parents=True, exist_ok=True)
    return merge_bundles([input_root, out_dir], target, job_id=job_id, meta=meta, compression=str(cfg.get("runtime.parquet_compression")),
                         compress_level=int(cfg.get("runtime.bundle_compression_level")))


def score(job: str, bundle: Path, cfg, package: Path, scoring_dir: Path, log) -> Path:
    """Run the job's scorer on ``bundle``; returns the long summary csv."""
    if job == "basic":
        from finfm.eval import score_basic as m
        manifest, tape, state = m.load_bundle(bundle)
        meta = manifest["meta"]
        inputs = m.load_real_inputs(package, cfg, meta["bucket"], meta["llm"])
        m.run_scoring(inputs, tape, manifest, state, scoring_dir, log=log)
    elif job == "finripple":
        from finfm.eval import score_finripple as m
        manifest, tape, state = m.load_bundle(bundle)
        meta = manifest["meta"]
        inputs = m.load_real_inputs(package, cfg, meta["bucket"], meta["llm"], package)
        m.run_scoring(inputs, tape, manifest, state, scoring_dir, log=log, max_tokens=int(cfg.get("rivals.finripple.max_tokens")))
    elif job == "mass":
        from finfm.eval import score_mass as m
        manifest, tape, raw, state = m.load_bundle(bundle)
        meta = manifest["meta"]
        inputs = m.load_real_inputs(package, cfg, meta["bucket"], meta["llm"])
        m.run_scoring(inputs, tape, raw, manifest, state, scoring_dir, log=log)
    else:
        raise ValueError(f"no scorer for job {job}")
    csvs = sorted(scoring_dir.glob("summary_*.csv"))
    if not csvs:
        raise FileNotFoundError("the scorer wrote no summary csv")
    return csvs[0]


def _pnl_frames(scoring_dir: Path) -> list[tuple[str, pd.DataFrame]]:
    out = []
    for path in sorted(scoring_dir.glob(f"{PNL_PREFIX}*.parquet")):
        try:
            out.append((path.stem[len(PNL_PREFIX):], pd.read_parquet(path)))
        except Exception:  # noqa: BLE001
            continue
    return out


def charts(summary: pd.DataFrame, scoring_dir: Path, tape: pd.DataFrame | None, cfg, out: Path, title: str, log) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    rep = cfg.get("rivals.report")
    dpi, costs, hcost = int(rep["chart_dpi"]), list(rep["cost_order"]), str(rep["headline_cost"])
    written: list[str] = []
    summary = summary.assign(value=pd.to_numeric(summary["value"], errors="coerce"))   # "not computed" labels become empty

    def save(fig, name):
        path = out / name
        fig.tight_layout()
        fig.savefig(path, dpi=dpi)
        plt.close(fig)
        written.append(name)
        log(f"chart {name}")

    port = summary[(summary["section"] == PORTFOLIO) & (summary["metric"] == SHARPE) & (summary["fill"].fillna(str(rep["headline_fill"])) == str(rep["headline_fill"]))]
    windows = [w for w in port["window"].dropna().unique()]
    # 1. Sharpe by cost, every variant, one panel per window
    try:
        fig, axes = plt.subplots(1, max(len(windows), 1), figsize=(PANEL_WIDTH * max(len(windows), 1), PANEL_HEIGHT), squeeze=False)
        for ax, w in zip(axes[0], windows, strict=False):
            sub = port[(port["window"] == w) & (port["cost"].isin(costs))]
            variants = list(sub["variant"].dropna().unique())[:MAX_VARIANTS_SHOWN]
            width = BAR_SPAN / max(len(variants), 1)
            for k, v in enumerate(variants):
                sv = sub[sub["variant"] == v].drop_duplicates("cost").set_index("cost").reindex(costs)
                ax.bar(np.arange(len(costs)) + k * width, sv["value"].astype(float), width, label=v)
            ax.axhline(0, color="k", lw=LINE_WIDTH)
            ax.set_xticks(np.arange(len(costs)) + BAR_SPAN * HALF - width * HALF, costs)
            ax.set_title(f"{w}")
            ax.set_ylabel("Sharpe")
        axes[0][0].legend(fontsize=SMALL_FONT)
        fig.suptitle(f"{title}: Sharpe by cost level")
        save(fig, "01_sharpe_by_cost.png")
    except Exception as exc:  # noqa: BLE001
        log(f"chart 01 failed: {exc!r}")
    # 2. cumulative net PnL and drawdown of the headline cost
    try:
        frames = [(n, f) for n, f in _pnl_frames(scoring_dir) if n.endswith(f"_{hcost}") and "net" in f and "fill_date" in f and "window" in f]
        if frames:
            fig, axes = plt.subplots(2, 1, figsize=WIDE_SIZE, sharex=True)
            for name, f in frames[:MAX_CURVES]:
                f = f[f["window"].isin(windows)] if windows else f
                series = f.groupby("fill_date")["net"].sum().sort_index()
                curve = (1 + series).cumprod()
                axes[0].plot(curve.index, curve.values, label=name[:-len(hcost) - 1], lw=THIN_LINE)
                axes[1].plot(curve.index, curve / curve.cummax() - 1, lw=THIN_LINE)
            axes[0].set_ylabel(f"growth of 1 (net, {hcost})")
            axes[1].set_ylabel("drawdown")
            axes[0].legend(fontsize=SMALL_FONT)
            fig.suptitle(f"{title}: cumulative net return and drawdown")
            save(fig, "02_cumulative_and_drawdown.png")
    except Exception as exc:  # noqa: BLE001
        log(f"chart 02 failed: {exc!r}")
    # 3. annual return / vol / turnover table as a bar of annual return by window
    try:
        ann = summary[(summary["section"] == PORTFOLIO) & (summary["metric"].isin(["annual_return", "annual_vol", "max_drawdown"])) & (summary["cost"] == hcost)
                      & (summary["fill"] == str(rep["headline_fill"]))]
        if len(ann):
            piv = ann.pivot_table(index=["window", "variant"], columns="metric", values="value", aggfunc="first").reset_index()
            fig, ax = plt.subplots(figsize=TALL_SIZE)
            labels = [f"{w}\n{v}" for w, v in zip(piv["window"], piv["variant"], strict=True)]
            x = np.arange(len(piv))
            for k, col in enumerate([c for c in ("annual_return", "annual_vol", "max_drawdown") if c in piv]):
                ax.bar(x + k * BAR_WIDTH, piv[col].astype(float), BAR_WIDTH, label=col)
            ax.set_xticks(x + BAR_WIDTH, labels, fontsize=GRID_LABEL, rotation=ROTATION)
            ax.axhline(0, color="k", lw=LINE_WIDTH)
            ax.legend()
            ax.set_title(f"{title}: annual return, vol, max drawdown ({hcost})")
            save(fig, "03_return_vol_drawdown.png")
    except Exception as exc:  # noqa: BLE001
        log(f"chart 03 failed: {exc!r}")
    # 4. run health from the tape
    try:
        if tape is not None and len(tape):
            fig, axes = plt.subplots(1, HEALTH_PANELS, figsize=HEALTH_SIZE)
            tape[STATUS].astype(str).value_counts().plot.bar(ax=axes[0], title="unit status")
            if "tokens_out" in tape:
                tape["tokens_out"].dropna().astype(float).plot.hist(bins=HIST_BINS, ax=axes[1], title="answer tokens")
            if "tokens_in" in tape:
                tape["tokens_in"].dropna().astype(float).plot.hist(bins=HIST_BINS, ax=axes[2], title="prompt tokens")
            save(fig, "04_run_health.png")
    except Exception as exc:  # noqa: BLE001
        log(f"chart 04 failed: {exc!r}")
    # 5. rank IC / hit-rate style signal metrics, whatever the scorer wrote
    try:
        sig = summary[(summary["section"] == "signal") & summary["value"].notna()]
        sig = sig[pd.to_numeric(sig["value"], errors="coerce").notna()].copy()
        sig["value"] = sig["value"].astype(float)
        if len(sig):
            top = sig.groupby("metric")["value"].count().sort_values(ascending=False).index[:MAX_SIGNAL_PANELS]
            fig, axes = plt.subplots(1, len(top), figsize=(SIGNAL_PANEL_WIDTH * len(top), SIGNAL_PANEL_HEIGHT), squeeze=False)
            for ax, m in zip(axes[0], top, strict=True):
                s = sig[sig["metric"] == m].head(MAX_SIGNAL_ROWS)
                ax.barh([f"{a}|{b}" for a, b in zip(s["set"].astype(str), s["window"].astype(str), strict=True)], s["value"])
                ax.set_title(m, fontsize=TITLE_FONT)
                ax.tick_params(labelsize=GRID_LABEL)
            save(fig, "05_signal_metrics.png")
    except Exception as exc:  # noqa: BLE001
        log(f"chart 05 failed: {exc!r}")
    return written


def build_report(run, config_dir, input_root, out_dir, log=print, merge: bool = False) -> dict:
    """Score the newest bundle of ``run`` in ``out_dir`` and write ``<run id>-report`` (folder and zip) next to it. Never raises."""
    out_dir, input_root = Path(out_dir), Path(input_root)
    cfg = load_config(config_dir)
    rep = cfg.get("rivals.report")
    beat = Heartbeat(float(cfg.get("kaggle.heartbeat_minutes")) * SECONDS_PER_MINUTE, log)
    result = {"ok": False, "run": run.run_id}
    try:
        from finfm.stage.run_stage import find_market_dir
        package = find_market_dir(input_root)
        cfg = load_config(config_dir, {"cpu.folders.upload": str(package), "cpu.folders.market": str(package),
                                       "cpu.folders.factors": str(package / "factors_daily.parquet")})
        bundle = merged_bundle(run, input_root, out_dir, cfg, beat) if merge else newest_bundle(out_dir, run.job, run.bucket, run.llm)
        if bundle is None:
            raise FileNotFoundError("no bundle of this run in the output folder")
        name = f"{run.run_id}-{rep['folder_suffix']}"
        folder = out_dir / name
        scoring = folder / "scoring"
        scoring.mkdir(parents=True, exist_ok=True)
        beat(f"scoring {bundle.name} with the {run.job} scorer (same windows, costs and metrics as the main framework)")
        done = threading.Event()
        box: dict = {}

        def work():
            try:
                box["csv"] = score(run.job, bundle, cfg, package, scoring, beat)
            except Exception:  # noqa: BLE001
                box["error"] = traceback.format_exc()
            finally:
                done.set()

        threading.Thread(target=work, daemon=True).start()
        if not done.wait(float(rep["max_minutes"]) * SECONDS_PER_MINUTE):
            raise TimeoutError(f"scoring took more than {rep['max_minutes']} minutes; the tape bundle is saved, score it on the Mac")
        if "error" in box:
            raise RuntimeError(box["error"])
        summary = pd.read_csv(box["csv"])
        from finfm.eval.score_basic import load_bundle as _lb
        tape = None
        try:
            tape = _lb(bundle)[1] if run.job != "mass" else None
        except Exception:  # noqa: BLE001
            tape = None
        written = charts(summary, scoring, tape, cfg, folder, run.run_id, beat)
        headline = summary[(summary["section"] == PORTFOLIO) & (summary["fill"] == str(rep["headline_fill"])) & summary["cost"].isin(list(rep["cost_order"]))
                           & summary["metric"].isin(["sharpe", "annual_return", "annual_vol", "max_drawdown", "cvar", "invested_days", "sessions",
                                                      "one_way_turnover_per_invested_day"])]
        headline.pivot_table(index=["set", "window", "variant", "cost"], columns="metric", values="value", aggfunc="first").reset_index().to_csv(folder / "headline_table.csv", index=False)
        code_version = next((p.read_text(encoding="utf-8") for p in Path(config_dir).resolve().parent.glob("CODE_VERSION")), None)
        info = {"run": run.run_id, "job": run.job, "bucket": run.bucket, "llm": run.llm, "code_version": code_version, "bundle": bundle.name,
                "config_hash": cfg.hash, "scored_with": f"scripts/s9_score_{run.job}.py equivalent",
                "charts": written, "summary_csv": box["csv"].name}
        if tape is not None:   # which events the method answered: two reports carry the same events when these two values are equal
            from finfm.compute.shard import units_fingerprint
            info.update(n_units=int(tape["unit_id"].nunique()), units_sha256=units_fingerprint(tape["unit_id"].drop_duplicates()))
        (folder / "report_info.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
        (folder / "README.md").write_text(
            f"# {run.run_id} report\n\nScored from `{bundle.name}` with the project's `{run.job}` scorer: same windows, cost grid and metrics as the main framework "
            f"(compare `scoring/{box['csv'].name}` row by row with the framework's `summary` table). Charts: {', '.join(written)}.\n"
            "Numbers only, no verdicts. Every window is labelled in the `label` column of the summary.\n", encoding="utf-8")
        shutil.copy2(bundle, folder / bundle.name)
        if merge:
            shutil.rmtree(out_dir / "merged_tape", ignore_errors=True)
        zip_path = out_dir / f"{name}.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(folder.rglob("*")):
                if p.is_file():
                    z.write(p, p.relative_to(out_dir).as_posix())
        shutil.rmtree(folder)
        result.update(ok=True, zip=str(zip_path), charts=written)
        beat(f"report written: {zip_path.name} ({len(written)} charts)")
    except Exception as exc:  # noqa: BLE001
        result["error"] = repr(exc)[:MAX_ERROR_CHARS]
        beat(f"REPORT FAILED (the run's .bundle is saved and complete): {exc!r}"[:MAX_LOG_CHARS])
    finally:
        beat.stop.set()
    return result
