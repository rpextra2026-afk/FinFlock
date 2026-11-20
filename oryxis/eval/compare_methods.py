"""Paired comparison of separately run methods on one bucket, from the report zips they left behind (and our framework's daily series).

Each method's daily net series (headline variant, headline cost, open fill) is put on the common session calendar of every window (a day without a
position counts as 0), then every pair gets a Ledoit-Wolf studentised block-bootstrap interval for the Sharpe difference. Numbers only. The stamp of
every method (config hash, code version, LLM, bucket) is compared first: a difference is printed on top of every output, never hidden.
"""

from __future__ import annotations

import json
import tempfile
import threading
import zipfile
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from oryxis.common.config import load_config
from oryxis.common.rng import make_rng
from oryxis.eval.lw_bootstrap import sharpe_difference_test
from oryxis.eval.portfolio_metrics import sharpe

SECONDS_PER_MINUTE = 60
INFO = "report_info.json"
MIN_DAYS = 20          # fewer common days than this: the pair is counted, not tested
FIGURE_WIDTH = 11
FIGURE_HEIGHT = 4.5
LINE = 1
FONT = 7
MAX_CHARS = 800


def find_reports(*roots: Path) -> list[Path]:
    """Every ``*-report.zip`` under the given folders (attached datasets and the output)."""
    found = []
    for root in roots:
        found += sorted(Path(root).rglob("*-report.zip"))
    return found


def unpack(zips: list[Path], target: Path) -> list[Path]:
    folders = []
    for z in zips:
        out = target / z.stem
        with zipfile.ZipFile(z) as archive:
            archive.extractall(out)
        folders.append(out)
    return folders


def methods_in(folder: Path, cfg) -> list[dict]:
    """One entry per method found in an unpacked report: name, job, stamp, the folder holding its pnl and summary files."""
    out = []
    for info_path in sorted(folder.rglob(INFO)):
        info = json.loads(info_path.read_text(encoding="utf-8"))
        base = info_path.parent
        if info.get("methods"):                                   # a baseline report: one sub-folder per baseline
            for name in info["methods"]:
                out.append({"name": name, "job": info["job"], "stamp": {k: info.get(k) for k in cfg.get("comparison.stamp_keys")}, "folder": base / name})
        else:
            out.append({"name": info["run"], "job": info["job"], "stamp": {k: info.get(k) for k in cfg.get("comparison.stamp_keys")}, "folder": base / "scoring"})
    return out


def method_series(method: dict, cfg) -> pd.DataFrame | None:
    spec = cfg.get(f"comparison.headline.{method['job']}")
    cost, fill = str(cfg.get("comparison.cost")), str(cfg.get("comparison.fill"))
    path = method["folder"] / f"{spec['file_prefix']}_{spec['variant']}_{cost}.parquet"
    if not path.exists():
        return None
    frame = pd.read_parquet(path)
    frame = frame[frame["fill"] == fill]
    if spec.get("set") and "set" in frame:
        frame = frame[frame["set"] == spec["set"]]
    return frame[["fill_date", "window", "net"]]


def ours_series(roots: list[Path], bucket: str, cfg) -> pd.DataFrame | None:
    pattern = str(cfg.get("comparison.ours.file_pattern")).format(bucket=bucket)
    for root in roots:
        hits = sorted(Path(root).rglob(pattern))
        if hits:
            return pd.read_parquet(hits[0])[["fill_date", "window", "net"]]
    return None


def stamp_report(methods: list[dict], keys: list[str]) -> list[str]:
    """Lines saying which stamp keys differ between methods (empty list = every method carries the same stamp)."""
    lines = []
    for key in keys:
        values = {m["name"]: m["stamp"].get(key) for m in methods}
        if len({str(v) for v in values.values()}) > 1:
            lines.append(f"STAMP DIFFERS on {key}: {values}")
    return lines


def metrics_table(methods: list[dict], cfg) -> pd.DataFrame:
    """Headline metrics each method's own scorer wrote (same long format for all): Sharpe at each cost, return, drawdown, turnover, invested days, rank IC."""
    rows = []
    for m in methods:
        spec = cfg.get(f"comparison.headline.{m['job']}")
        summary_files = sorted(m["folder"].glob("summary_*.csv")) or sorted(m["folder"].parent.glob("summary_*.csv"))
        if not summary_files:
            continue
        table = pd.read_csv(summary_files[0])
        port = table[(table["section"] == "portfolio") & (table["variant"] == spec["variant"]) & (table["fill"] == str(cfg.get("comparison.fill")))]
        if spec.get("set") and "set" in port:
            port = port[port["set"] == spec["set"]]
        for window in cfg.get("comparison.windows"):
            part = port[port["window"] == window]
            if part.empty:
                continue
            row = {"method": m["name"], "window": window}
            for cost in cfg.get("comparison.metrics_costs"):
                v = part[(part["metric"] == "sharpe") & (part["cost"] == cost)]["value"]
                row[f"sharpe_{cost}"] = v.iloc[0] if len(v) else "not computed"
            headline_cost = str(cfg.get("comparison.cost"))
            for metric in ("annual_return", "annual_vol", "max_drawdown", "cvar", "sessions", "invested_days"):
                v = part[(part["metric"] == metric) & (part["cost"] == headline_cost)]["value"]
                row[metric] = v.iloc[0] if len(v) else "not computed"
            v = part[(part["metric"] == "one_way_turnover_per_invested_day")]["value"]
            row["turnover_per_invested_day"] = v.iloc[0] if len(v) else "not computed"
            rows.append(row)
    return pd.DataFrame(rows)


def paired_table(series: dict[str, pd.DataFrame], cfg, log=print) -> pd.DataFrame:
    """Sharpe difference (annual) with interval and p-value for every pair of methods in every window, on the common calendar of that window."""
    periods, draws, level = int(cfg.get("eval.annualisation_days")), int(cfg.get("eval.bootstrap.draws")), float(cfg.get("eval.bootstrap.ci_level"))
    target = float(cfg.get("comparison.vol_target_annual"))
    rows = []
    for window in cfg.get("comparison.windows"):
        wide = {}
        for name, frame in series.items():
            part = frame[frame["window"] == window]
            if len(part):
                wide[name] = part.groupby("fill_date")["net"].sum().sort_index()
        if len(wide) < 2:
            continue
        calendar = sorted(set().union(*[s.index for s in wide.values()]))
        aligned = pd.DataFrame({n: s.reindex(calendar).fillna(0.0) for n, s in wide.items()})
        for a, b in combinations(aligned.columns, 2):
            ra, rb = aligned[a].to_numpy(), aligned[b].to_numpy()
            row = {"window": window, "method_a": a, "method_b": b, "common_days": len(aligned), "sharpe_a": sharpe(aligned[a], periods), "sharpe_b": sharpe(aligned[b], periods)}
            if len(aligned) < MIN_DAYS or ra.std() == 0 or rb.std() == 0:
                row["note"] = "not tested: too few common days or a series without variation"
            else:
                test = sharpe_difference_test(ra, rb, periods, draws, level, make_rng(cfg.get("runtime.base_seed"), cfg.get("comparison.bootstrap_seed_label"), window, a, b))
                row.update(sharpe_diff_annual=test.delta_annual, ci_low=test.ci_low, ci_high=test.ci_high, p_value=test.p_value, block=test.block, draws=test.draws)
                if not (np.isfinite(test.ci_low) and np.isfinite(test.ci_high)):
                    row["note"] = "interval not computed: a series has too few invested days for the bootstrap (many zero days)"
                scale_a, scale_b = target / (ra.std() * np.sqrt(periods)), target / (rb.std() * np.sqrt(periods))
                row["annual_return_a_at_vol_target"], row["annual_return_b_at_vol_target"] = float(ra.mean() * scale_a * periods), float(rb.mean() * scale_b * periods)
            rows.append(row)
        log(f"paired {window}: {len(aligned.columns)} methods, {len(aligned)} common days")
    return pd.DataFrame(rows)


def build_comparison(config_dir: Path, bucket: str, roots: list[Path], out_dir: Path, log=print) -> dict:
    """Unpack every report zip under ``roots``, compare all methods found (plus our framework's series if present), write tables and a zip. Never raises."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    cfg = load_config(config_dir)
    interval = float(cfg.get("kaggle.heartbeat_minutes")) * SECONDS_PER_MINUTE
    stop = threading.Event()

    def beat() -> None:
        while not stop.wait(interval):
            print("still working: comparison", flush=True)

    threading.Thread(target=beat, daemon=True).start()
    result = {"ok": False, "bucket": bucket}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            zips = find_reports(*roots)
            folders = unpack(zips, Path(tmp))
            methods = [m for f in folders for m in methods_in(f, cfg) if m["stamp"].get("bucket") in (None, bucket)]
            log(f"{len(zips)} report zip(s), {len(methods)} method(s): {[m['name'] for m in methods]}")
            series = {m["name"]: s for m in methods if (s := method_series(m, cfg)) is not None}
            ours = ours_series(list(roots), bucket, cfg)
            if ours is not None:
                series[str(cfg.get("comparison.ours.label"))] = ours
            if len(series) < 2:
                raise RuntimeError(f"need at least two methods with a daily series, found {list(series)}")
            out = out_dir / f"comparison-{bucket}"
            out.mkdir(parents=True, exist_ok=True)
            stamp_lines = stamp_report(methods, list(cfg.get("comparison.stamp_keys")))
            (out / "STAMP.txt").write_text("\n".join(stamp_lines) if stamp_lines else "every method carries the same config hash, code version, LLM and bucket", encoding="utf-8")
            metrics_table(methods, cfg).to_csv(out / "metrics_by_method.csv", index=False)
            paired = paired_table(series, cfg, log)
            paired.to_csv(out / "paired_sharpe_differences.csv", index=False)
            fig, ax = plt.subplots(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))
            for name, frame in series.items():
                s = frame.groupby("fill_date")["net"].sum().sort_index()
                ax.plot(s.index, (1 + s).cumprod(), lw=LINE, label=name)
            ax.legend(fontsize=FONT)
            ax.set_title(f"{bucket}: growth of 1 (net, {cfg.get('comparison.cost')}), all windows")
            fig.tight_layout()
            fig.savefig(out / "cumulative_all_methods.png", dpi=int(cfg.get("rivals.report.chart_dpi")))
            plt.close(fig)
            zip_path = out_dir / f"comparison-{bucket}.zip"
            with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as z:
                for p in sorted(out.rglob("*")):
                    if p.is_file():
                        z.write(p, p.relative_to(out_dir).as_posix())
        result.update(ok=True, methods=list(series), stamp=stamp_lines, zip=str(zip_path))
        log(f"comparison written: {zip_path.name}; stamp differences: {stamp_lines or 'none'}")
    except Exception as exc:  # noqa: BLE001
        result["error"] = repr(exc)[:MAX_CHARS]
        log(f"COMPARISON FAILED: {exc!r}"[:MAX_CHARS])
    finally:
        stop.set()
    return result
