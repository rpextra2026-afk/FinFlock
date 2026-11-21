"""No-LLM baselines scored by the Basic rival's own scorer: a baseline score replaces the LLM answer, everything else is identical.

Each baseline gives one score per Basic headline unit (higher = more bullish); the score becomes the unit's ``score`` and its sign the answer
(YES / NO / UNKNOWN), and the synthetic tape goes through ``score_basic.run_scoring``: same decision dates, S&P membership, de-duplication, fills,
cost grid, windows, metrics, signal tests. Lineage is ``nollm_baseline`` (real data, no LLM). The placebo band re-runs only the headline rule.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from finflock.baselines import lexicon, price
from finflock.baselines.finbert import finbert_tone, make_scorer
from finflock.baselines.placebo import band, placebo_scores
from finflock.common.config import load_config
from finflock.common.schema import PARSE_OK, STATUS, UNIT_ID, Status
from finflock.eval import score_basic as sb_mod
from finflock.eval.score_basic import ANSWER, SCORE, build_units, attach_abnormal, headline_cost_name, in_span, run_scoring, run_variant, window_spans, sb

YES, NO, UNKNOWN = "YES", "NO", "UNKNOWN"
LABEL_KEY = "cpu.score_basic.labels.in_sample"
SHARPE = "sharpe"
LEGEND_FONT = 7


def synthetic_tape(units: pd.DataFrame, scores: pd.Series) -> pd.DataFrame:
    s = scores.reindex(units.index).fillna(0.0).astype(float)
    answer = np.where(s > 0, YES, np.where(s < 0, NO, UNKNOWN))
    return pd.DataFrame({UNIT_ID: units[UNIT_ID].to_numpy(), STATUS: Status.OK.value, PARSE_OK: True, ANSWER: answer, SCORE: s.to_numpy()})


def baseline_scores(name: str, inp, cfg, cache_dir: Path, log) -> pd.Series:
    units = inp.units
    bl = cfg.get("baselines")
    if name == "N1_lm_tone":
        return pd.Series(lexicon.lm_tone(units["headline"]).to_numpy(), index=units.index)
    if name == "N2_finbert_tone":
        fb = bl["finbert"]
        log(f"FinBERT on {len(units)} headlines (cached by text hash)")
        tone = finbert_tone(units["headline"], cache_dir / str(fb["cache_file"]), make_scorer(str(fb["model"]), int(fb["max_length"]), int(fb["batch_size"])))
        return pd.Series(tone.to_numpy(), index=units.index)
    if name == "N3_reversal":
        return price.reversal(inp.market.cc, units, int(bl["reversal_days"]))
    if name == "N4_momentum":
        return price.momentum(inp.market.cc, units, int(bl["momentum"]["start_lag_sessions"]), int(bl["momentum"]["skip_sessions"]))
    if name == "N5_placebo":
        return placebo_scores(units.index, int(cfg.get("runtime.base_seed")), 0)
    raise ValueError(f"unknown baseline {name}")


def placebo_band(inp, cfg, log) -> pd.DataFrame:
    """Net Sharpe of the headline rule over ``baselines.placebo.draws`` random score draws, per window: mean and the configured quantile band."""
    pl = cfg.get("baselines.placebo")
    rows = []
    rows_stub = sb_mod.Rows(cfg)
    units, members = build_units(synthetic_tape(inp.units, pd.Series(0.0, index=inp.units.index)), inp, rows_stub)
    attach_abnormal(units, members, inp)
    units = units[units["group"].isin(sb(cfg, f"sets.{cfg.get('baselines.mode_basic_set')}"))]
    spans, cost, cache = window_spans(inp), headline_cost_name(cfg), {}
    headline_variant, fill = str(sb(cfg, "headline_variant")), str(sb(cfg, "fills.headline"))
    from finflock.eval.portfolio_metrics import sharpe
    periods = int(cfg.get("eval.annualisation_days"))
    for window, span in spans.items():
        if span is None:
            continue
        sub = units[in_span(units["decision_date"], span)]
        values = []
        for draw in range(int(pl["draws"])):
            part = sub.copy()
            part[SCORE] = placebo_scores(part.index, int(cfg.get("runtime.base_seed")), draw)
            values.append(sharpe(run_variant(part, span, headline_variant, fill, True, inp, cache).net[cost], periods))
            log(f"placebo {window} draw {draw + 1}/{pl['draws']}")
        mean, lo, hi = band(values, float(pl["band"][0]), float(pl["band"][1]))
        rows.append({"baseline": "N5_placebo", "window": window, "variant": headline_variant, "cost": cost, "draws": len(values), "mean_sharpe": mean, "q_low": lo, "q_high": hi})
    return pd.DataFrame(rows)


def run_nollm(package: Path, config_dir: Path, bucket: str, out_dir: Path, baselines: list[str] | None = None, log=print, llm: str | None = None) -> dict:
    """Score every baseline for one bucket; writes ``<out>/<baseline>/`` scorer outputs, ``summary_nollm.csv``, ``placebo_band.csv``, ``lineage.json``."""
    cfg = load_config(config_dir, {"cpu.folders.upload": str(package), "cpu.folders.market": str(package), "cpu.folders.factors": str(package / "factors_daily.parquet"),
                                   LABEL_KEY: str(load_config(config_dir).get("baselines.labels.no_llm"))})
    out_dir.mkdir(parents=True, exist_ok=True)
    llm = llm or str(cfg.get("llms.rival_llm"))
    inp = sb_mod.load_real_inputs(package, cfg, bucket, llm)
    names = baselines or list(cfg.get("baselines.ids"))
    started, summaries, failures = time.time(), [], {}
    for name in names:
        log(f"[{time.time() - started:6.0f}s] baseline {name} ({bucket})")
        try:
            scores = baseline_scores(name, inp, cfg, out_dir / "cache", log)
            manifest = {"bundle_id": f"nollm-{name}", "meta": {"job": "basic", "bucket": bucket, "llm": llm, "code_version": str(cfg.get("baselines.lineage"))}}
            run_scoring(inp, synthetic_tape(inp.units, scores), manifest, {}, out_dir / name, log=log)
            table = pd.read_csv(out_dir / name / str(sb(cfg, "output.summary_csv")))
            summaries.append(table.assign(baseline=name, bucket=bucket))
        except Exception as exc:  # noqa: BLE001  one baseline failing never stops the others
            failures[name] = repr(exc)[:ERROR_CHARS]
            log(f"baseline {name} FAILED: {failures[name]}")
    if summaries:
        pd.concat(summaries, ignore_index=True).to_csv(out_dir / "summary_nollm.csv", index=False)
    if "N5_placebo" in names:
        try:
            placebo_band(inp, cfg, log).to_csv(out_dir / "placebo_band.csv", index=False)
        except Exception as exc:  # noqa: BLE001
            failures["placebo_band"] = repr(exc)[:ERROR_CHARS]
    (out_dir / "lineage.json").write_text(json.dumps({"lineage": str(cfg.get("baselines.lineage")), "bucket": bucket, "config_hash": cfg.hash, "baselines": names,
                                                      "lexicon": lexicon.lm_version() if "N1_lm_tone" in names else None, "failures": failures}, indent=1), encoding="utf-8")
    return {"ok": bool(summaries), "failures": failures, "baselines": names}


SECONDS_PER_MINUTE = 60
FIGURE_SIZE = (11, 4.5)
CURVE_SIZE = (10, 5)
BAR_SPAN = 0.8
LINE = 1
MESSAGE_CHARS = 200
ERROR_CHARS = 800


def build_nollm_report(package: Path, config_dir: Path, bucket: str, out_dir: Path) -> dict:
    """Run every baseline, draw the comparison charts and write ``nollm-baselines-<bucket>-report.zip``. Never raises."""
    import shutil
    import threading
    import zipfile

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config(config_dir)
    interval = float(cfg.get("kaggle.heartbeat_minutes")) * SECONDS_PER_MINUTE
    stop, last = threading.Event(), ["starting"]

    def say(message: str) -> None:
        last[0] = str(message)[:MESSAGE_CHARS]
        print(message, flush=True)

    def beat() -> None:
        while not stop.wait(interval):
            print(f"still working: {last[0]}", flush=True)

    threading.Thread(target=beat, daemon=True).start()
    name = f"nollm-baselines-{bucket}"
    folder = out_dir / name
    result = {"ok": False, "bucket": bucket}
    try:
        result.update(run_nollm(package, config_dir, bucket, folder, log=say))
        summary = pd.read_csv(folder / "summary_nollm.csv")
        cost, fill, variant = headline_cost_name(cfg), str(sb(cfg, "fills.headline")), str(sb(cfg, "headline_variant"))
        head = summary[(summary["section"] == "portfolio") & (summary["metric"] == SHARPE) & (summary["variant"] == variant) & (summary["cost"] == cost) & (summary["fill"] == fill)
                       & (summary["set"] == str(cfg.get("baselines.mode_basic_set")))].copy()
        head["value"] = pd.to_numeric(head["value"], errors="coerce")
        table = head.pivot_table(index="baseline", columns="window", values="value", aggfunc="first")
        table.to_csv(folder / "headline_sharpe_table.csv")
        fig, ax = plt.subplots(figsize=FIGURE_SIZE)
        table.plot.bar(ax=ax)
        ax.axhline(0, color="k", lw=LINE)
        ax.set_ylabel(f"net Sharpe ({cost}, {variant})")
        ax.set_title(f"No-LLM baselines, {bucket}: headline rule, by window")
        fig.tight_layout()
        fig.savefig(folder / "01_baseline_sharpe_by_window.png", dpi=int(cfg.get("rivals.report.chart_dpi")))
        plt.close(fig)
        fig, ax = plt.subplots(figsize=CURVE_SIZE)
        for path in sorted(folder.glob(f"*/pnl_llt_{variant}_{cost}.parquet")):
            frame = pd.read_parquet(path)
            frame = frame[(frame["set"] == str(cfg.get("baselines.mode_basic_set"))) & (frame["fill"] == fill)]
            series = frame.groupby("fill_date")["net"].sum().sort_index()
            ax.plot(series.index, (1 + series).cumprod(), lw=LINE, label=path.parent.name)
        ax.set_title(f"No-LLM baselines, {bucket}: growth of 1 (net, {cost})")
        ax.legend(fontsize=LEGEND_FONT)
        fig.tight_layout()
        fig.savefig(folder / "02_baseline_cumulative.png", dpi=int(cfg.get("rivals.report.chart_dpi")))
        plt.close(fig)
        code_version = next((p.read_text(encoding="utf-8") for p in Path(config_dir).resolve().parent.glob("CODE_VERSION")), None)
        (folder / "report_info.json").write_text(json.dumps({"run": name, "job": "basic", "bucket": bucket, "llm": str(cfg.get("llms.rival_llm")), "code_version": code_version,
                                                             "config_hash": cfg.hash, "methods": list(result.get("baselines", []))}, indent=1), encoding="utf-8")
        zip_path = out_dir / f"{name}-report.zip"
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
            for p in sorted(folder.rglob("*")):
                if p.is_file() and "cache" not in p.parts:
                    z.write(p, p.relative_to(out_dir).as_posix())
        shutil.rmtree(folder)
        result.update(ok=True, zip=str(zip_path))
        say(f"report written: {zip_path.name}")
    except Exception as exc:  # noqa: BLE001
        result["error"] = repr(exc)[:ERROR_CHARS]
        say(f"BASELINE REPORT FAILED: {exc!r}"[:ERROR_CHARS])
    finally:
        stop.set()
    return result
