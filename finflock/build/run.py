"""One tape, all stages: clean -> graphs -> fit -> score -> metrics -> write. Every stage is recorded in ``manifest.json``; a finished stage whose files are
still there (same hashes) is skipped on a restart, so a run stopped at the wall-clock cutoff carries on where it stopped."""

from __future__ import annotations

import contextlib
import json
import pickle
import time
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from finflock.build import diagnostics as D
from finflock.build import evaluate as E
from finflock.build import inputs as I  # noqa: E741
from finflock.build import metrics as M
from finflock.build import scoring as S
from finflock.build import world as W
from finflock.clean import run as clean_run
from finflock.clean import steps as clean_steps
from finflock.common.config import Config, load_config
from finflock.common.hashing import sha256_file, sha256_hex
from finflock.common.resources import ResourceLog, plan_workers
from finflock.common.timing import TradingCalendar
from finflock.fit.road1_fit import training_graphs
from finflock.common import ckpt
from finflock.road1.fit import ShardPending, new_model, predict_graph, standardise, train_model, usable
from finflock.road1.graph import EventGraph
from finflock.stage import road1_real as rr
from finflock.stage.road1_run import NotEnoughGraphs, fit_and_score

ROOT = "build"
STAGES = ("clean", "graphs", "fit", "score", "metrics", "ablations", "write")
MANIFEST = "manifest.json"
FREEZE_FILE = "FREEZE.json"
FRAMEWORK_FREEZE_FILE = "FREEZE_FRAMEWORK.json"
SHORT_HASH = 12


class Manifest:
    """Stage status, times, row counts and file hashes of one run folder."""

    def __init__(self, path: Path, run: str):
        self.path = path
        self.data: dict[str, Any] = json.loads(path.read_text()) if path.exists() else {"run": run, "stages": {}}

    def done(self, stage: str, files: list[Path]) -> bool:
        rec = self.data["stages"].get(stage)
        return bool(rec and rec.get("status") == "done" and all(p.exists() and sha256_file(p) == rec["files"].get(p.name) for p in files))

    def mark(self, stage: str, status: str, started: float, files: list[Path] | None = None, **extra: Any) -> None:
        self.data["stages"][stage] = {"status": status, "start_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(started)), "end_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                                      "seconds": round(time.time() - started, 1), "files": {p.name: sha256_file(p) for p in files or [] if p.exists()}, **extra}
        self.path.write_text(json.dumps(self.data, indent=2, default=str))


def code_hash(code_root: Path) -> str:
    """Digest of every source file the build uses (finflock/, config/, the two runner scripts); caches and byte-code are left out."""
    files = sorted(p for part in ("finflock", "config") for p in (code_root / part).rglob("*") if p.is_file() and "__pycache__" not in p.parts and p.suffix not in (".pyc", ".pyo"))
    files += [code_root / "scripts" / name for name in ("run_final_build.py", "cpu_session.py")]
    return sha256_hex("\n".join(f"{p.relative_to(code_root)}:{sha256_file(p)}" for p in files if p.exists()))


def build_config(base: Config, smoke: int = 0) -> Config:
    """The run's configuration: three link types, one pooled cause group. Only values under road1 and causes change; the hashed base is untouched."""
    all_causes = [c for members in base.get("causes.groups").values() for c in members]
    pooled = str(base.get("build.sheaf.pooled_group"))
    overrides: dict[str, Any] = {"road1.link_types": list(base.get("build.sheaf.link_types")), "road1.symmetric_types": list(base.get("build.sheaf.symmetric_types")), "causes.groups": {pooled: all_causes}}
    if smoke:  # a mechanics test: one grid cell, few steps (labelled in the manifest; never a result)
        overrides.update({f"road1.{k}": v for k, v in base.get("build.smoke.road1").items()})
    return base.with_overrides(overrides)


def restrict_links(graph: EventGraph, keep: list[int]) -> EventGraph:
    """The same graph with only the edges of the kept link types (node set and pinned values unchanged)."""
    mask = np.isin(graph.link_type, keep)
    return EventGraph(graph.event_id, graph.nodes, graph.n_pinned, graph.src[mask], graph.dst[mask], graph.link_type[mask], graph.weight[mask], graph.s, graph.group_probs, graph.truncated, graph.y, graph.meta)


def with_pinned(graph: EventGraph, s: np.ndarray) -> EventGraph:
    return EventGraph(graph.event_id, graph.nodes, graph.n_pinned, graph.src, graph.dst, graph.link_type, graph.weight, s, graph.group_probs, graph.truncated, graph.y, graph.meta)


class Runner:
    def __init__(self, run: str, root: Path, out: Path, base: Config, log: Callable[[str], None], workers: int | None, smoke: int = 0, fast_fit: bool = False):
        self.run, self.root, self.out, self.base, self.log, self.smoke = run, root, out, base, log, smoke
        self.cfg = build_config(base, smoke or int(fast_fit))
        self.run_cfg = base.get(f"clean.runs.{run}")
        self.out.mkdir(parents=True, exist_ok=True)
        self.manifest = Manifest(out / MANIFEST, run)
        self.cal = TradingCalendar.from_config(self.cfg)
        self.rlog = ResourceLog(out / "resources.json")
        self.plan = plan_workers(self.cfg, workers)
        self.types = list(self.cfg.get("road1.link_types"))
        self.seconds: dict[str, float] = {}
        self.last_progress = time.time()
        self.root_code = Path(__file__).resolve().parents[2]

    def progress(self, message: str) -> None:
        self.log(message)
        self.last_progress = time.time()

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        started = time.time()
        self.progress(f"STAGE {name} start")
        with self.rlog.stage(name):
            yield
        self.seconds[name] = time.time() - started
        self.progress(f"STAGE {name} done in {time.time() - started:.1f}s")

    # ---- clean -------------------------------------------------------------------------------------------------------------------------
    def clean_path(self) -> Path:
        return self.root / str(self.base.get("clean.output_dir")) / f"clean_{self.run}.parquet"

    def check_freeze(self) -> dict[str, Any]:
        path = self.root / str(self.base.get("build.cleaned_dir")) / FREEZE_FILE
        if not path.exists():
            self.progress("FREEZE.json not found: running without a freeze check (a note is written to the manifest)")
            return {"freeze": "absent"}
        freeze = json.loads(path.read_text())
        report = json.loads((self.root / str(self.base.get("clean.output_dir")) / f"clean_{self.run}_report.json").read_text())
        want = freeze.get("clean_content_sha256", {}).get(self.run)
        have = report["content_sha256"]
        if want is not None and want != have:
            raise SystemExit(f"the cleaned tape of {self.run} does not match FREEZE.json (frozen {want[:SHORT_HASH]}, now {have[:SHORT_HASH]}); refusing to run")
        later = self.root / str(self.base.get("clean.output_dir")) / FRAMEWORK_FREEZE_FILE
        if later.exists():  # the framework run extends the build: its freeze file (written later, same cleaned tapes) holds the config and code hashes that now apply
            freeze = {**freeze, **{k: v for k, v in json.loads(later.read_text()).items() if k in ("config_hash", "config_unhashed_hash", "code_sha256")}}
        for key, now in (("config_hash", self.base.hash), ("config_unhashed_hash", self.base.unhashed_hash), ("code_sha256", code_hash(self.root_code))):
            if freeze.get(key) not in (None, now):
                raise SystemExit(f"{key} does not match FREEZE.json (frozen {str(freeze.get(key))[:SHORT_HASH]}, now {str(now)[:SHORT_HASH]}); refusing to run")
        note = {"freeze": "matched", "frozen_content_sha256": want, "config_hash": self.base.hash}
        sector_frozen = freeze.get("sector_rows_sha256", {}).get(self.run)
        if sector_frozen is not None and sector_frozen != report["sector_rows_sha256"]:
            note["sector_rows_differ_from_freeze"] = True
            self.progress("NOTE: the sector touch points differ from the frozen ones (library version); the frozen part matched, the run goes on")
        return note

    def do_clean(self) -> None:
        started = time.time()
        files = [self.clean_path()]
        if self.manifest.done("clean", files):
            self.progress("STAGE clean skipped (finished earlier, files match)")
            return
        with self.stage("clean"):
            report = clean_run.clean_run(self.run, self.base, str(self.root))
        note = self.check_freeze()
        self.manifest.mark("clean", "done", started, files, items_out=report["items_out"], rows_out=report["rows_out"], **note)

    # ---- graphs ------------------------------------------------------------------------------------------------------------------------
    def do_graphs(self) -> dict[str, Any]:
        cache = self.out / str(self.cfg.get("build.runner.graph_cache"))
        started = time.time()
        if self.manifest.done("graphs", [cache]):
            self.progress("STAGE graphs skipped (cache found)")
            return pickle.loads(cache.read_bytes())
        with self.stage("graphs"):
            clean = pd.read_parquet(self.clean_path())
            inp = I.build_inputs(clean, self.run_cfg, self.cal, self.cfg)
            if self.smoke:
                keep = sorted(inp.readings)[: self.smoke]
                for name in ("readings", "dials", "day0", "split", "kinds", "pin_types"):
                    setattr(inp, name, {k: getattr(inp, name)[k] for k in keep})
            panels = rr.load_panels(self.root, self.cfg, self.cal)
            links = pd.read_parquet(self.root / self.run_cfg["links"])
            links = links[links["link_type"].isin(self.types)]
            nodes = rr.graph_nodes(inp.readings, links, inp.day0, self.cfg)
            labels = rr.label_table(nodes, inp.day0, panels, self.cfg)
            graphs = training_graphs(inp.readings, inp.dials, links, inp.day0, labels, self.cfg)
            dev, test = rr.split_graphs(graphs, inp.split, "dev"), rr.split_graphs(graphs, inp.split, "test")
            counts = {**inp.counts, "labels": int(len(labels)), "graphs": len(graphs), "graphs_dev": len(dev), "graphs_test": len(test), "free_nodes_dev": int(sum(g.n_free for g in dev)), "free_nodes_test": int(sum(g.n_free for g in test))}
            self.progress(f"counts {counts}")
            payload = {"dev": dev, "test": test, "day0": inp.day0, "pin_types": inp.pin_types, "split": inp.split, "kinds": inp.kinds, "counts": counts}
            cache.write_bytes(pickle.dumps(payload))
        self.manifest.mark("graphs", "done", started, [cache], **counts)
        return payload

    # ---- fit ---------------------------------------------------------------------------------------------------------------------------
    def do_fit(self, g: dict[str, Any]) -> rr.Fitted | None:
        cache = self.out / str(self.cfg.get("build.runner.fit_cache"))
        started = time.time()
        if self.manifest.done("fit", [cache]):
            self.progress("STAGE fit skipped (cache found)")
            return pickle.loads(cache.read_bytes())["fitted"]
        try:
            with self.stage("fit"):
                result = fit_and_score(g["dev"], usable(g["test"])[:1], g["day0"], g["pin_types"], self.cal, self.cfg, (self.run, "final"), self.plan["workers"], lambda name: self.rlog.stage(f"fit {name}"), self.progress, with_r_report=False)
        except NotEnoughGraphs as exc:
            self.manifest.mark("fit", "failed", started, reason=str(exc))
            raise
        except ShardPending as pending:
            self.progress(f"SHARD: the fits of '{pending}' owned by this shard are stored in the checkpoint; the assemble run finishes the rest")
            self.manifest.mark("fit", "shard_stored", started)
            return None
        configs = result.tables.get("configs_evaluated", pd.DataFrame())
        pickle.dump({"fitted": result.fitted, "configs": configs, "fits": result.tables["fits"]}, cache.open("wb"))
        self.manifest.mark("fit", "done", started, [cache], dev_graphs=len(g["dev"]))
        return result.fitted

    # ---- score -------------------------------------------------------------------------------------------------------------------------
    def do_score(self, g: dict[str, Any], fitted: rr.Fitted) -> dict[str, Any]:
        rows_p, panel_p = self.out / str(self.cfg.get("build.runner.rows_cache")), self.out / str(self.cfg.get("build.runner.panel_cache"))
        started = time.time()
        clean = pd.read_parquet(self.clean_path())
        meta = clean.drop_duplicates("unit_id").rename(columns={"unit_id": "event_id"})[["event_id", "cause_group", "kind", "family", "session_type", "repeat_count"]]
        dev = usable(g["dev"])
        train_mean = dict(zip(DIALS, np.nanmean(np.vstack([x.y for x in dev]), axis=0), strict=True))
        scale = dict(zip(DIALS, fitted.scale, strict=True))
        methods = list(self.cfg.get("build.runner.methods"))
        if self.manifest.done("score", [rows_p, panel_p]):
            self.progress("STAGE score skipped (cache found)")
            return {"rows": pd.read_parquet(rows_p), "panel": pd.read_parquet(panel_p), "train_mean": train_mean, "scale": scale, "methods": methods, "clean": clean, "resid_sd": json.loads((self.out / "resid_sd.json").read_text())}
        with self.stage("score"):
            rows = S.scored_rows(fitted, g["test"], g["day0"], meta, self.types, self.cfg)
            panels = rr.load_panels(self.root, self.cfg, self.cal)
            nodes = sorted(set(pd.read_parquet(self.root / self.run_cfg["links"], columns=["src_node", "dst_node"]).stack()))
            hold = self.cal.sessions_between(self.run_cfg["holdout_start"], self.run_cfg["holdout_end"])
            panel = S.panel_labels(hold, nodes, panels, self.cfg)
            panel_long = S.panel_with_forecasts(panel, rows, methods)
            sample = dev[: int(self.cfg.get("build.runner.resid_graphs"))]
            resid: dict[str, dict[str, float]] = {m: {} for m in methods}
            res_rows = S.scored_rows(fitted, sample, g["day0"], meta, self.types, self.cfg)
            for m in methods:
                for dial in DIALS:
                    part = res_rows[res_rows["dial"] == dial]
                    resid[m][dial] = float(np.std(part["y"] - part[f"pred_{m}"])) or float(scale[dial])
            rows.to_parquet(rows_p, index=False)
            panel_long.to_parquet(panel_p, index=False)
            (self.out / "resid_sd.json").write_text(json.dumps(resid))
        self.manifest.mark("score", "done", started, [rows_p, panel_p, self.out / "resid_sd.json"], rows=len(rows), panel_rows=len(panel_long))
        return {"rows": rows, "panel": panel_long, "train_mean": train_mean, "scale": scale, "methods": methods, "clean": clean, "resid_sd": resid}

    # ---- metrics -----------------------------------------------------------------------------------------------------------------------
    def context(self, s: dict[str, Any], g: dict[str, Any]) -> dict[str, Any]:
        panels = rr.load_panels(self.root, self.cfg, self.cal)
        info = pd.read_parquet(self.root / "dataset/finflock-market/security_info.parquet").set_index("security_id")
        hold = self.cal.sessions_between(self.run_cfg["holdout_start"], self.run_cfg["holdout_end"])
        fit = self.cal.sessions_between(self.run_cfg["fit_start"], self.run_cfg["fit_end"])
        absret = panels.ret.abs().mean(axis=1)
        thr = float(absret.reindex(fit).quantile(float(M.mcfg(self.cfg, "regime_quantile"))))
        cap = panels.cap_proxy
        n_terc = int(self.cfg.get("build.slices.size_terciles"))
        size_b, liq_b = {}, {}
        for d in hold:
            prev = cap.index.searchsorted(d) - 1
            if prev < 0:
                continue
            row = cap.iloc[prev].dropna()
            if len(row) >= n_terc:
                b = pd.qcut(row.rank(method="first"), n_terc, labels=False)
                for node, v in b.items():
                    size_b[(d, node)] = str(int(v))
            vol = (panels.close.iloc[prev] * pd.Series({k: v["volume_raw"].iloc[prev] for k, v in panels.prices.items()})).dropna()
            if len(vol) >= n_terc:
                b = pd.qcut(vol.rank(method="first"), n_terc, labels=False)
                for node, v in b.items():
                    liq_b[(d, node)] = str(int(v))
        clean = s["clean"]
        world = clean[clean["lane"] == clean_steps.LANE_WORLD]
        wc = world.groupby("session_date")["unit_id"].nunique()
        total_w = world.groupby("session_date")["weight"].sum()
        fit_w = total_w.reindex(fit).fillna(0.0)
        bthr = float(fit_w.quantile(float(M.mcfg(self.cfg, "burst_quantile"))))
        strengths = []
        for graph in g["dev"][: int(self.cfg.get("build.runner.resid_graphs"))]:
            f = S.link_features(graph, self.types)
            strengths.append(np.maximum(f["w_industry"], f["w_corr"]).to_numpy())
        flat = np.concatenate(strengths) if strengths else np.array([])
        flat = flat[flat > 0]
        edges = np.quantile(flat, np.linspace(0, 1, int(M.mcfg(self.cfg, "weight_bins")) + 1)[1:-1]) if len(flat) else np.array([])
        return {"weight_edges": edges, "sector_of": info["sector"].to_dict(), "size_bucket": size_b, "liquidity_bucket": liq_b, "high_vol_days": set(absret.index[absret > thr]),
                "world_days": set(wc.index[wc >= int(self.cfg.get("build.slices.world_day_min_items"))]), "burst_days": set(total_w.index[total_w >= bthr]) if bthr > 0 else set()}

    def do_metrics(self, g: dict[str, Any], fitted: rr.Fitted, s: dict[str, Any], fit_pack: dict[str, Any]) -> pd.DataFrame:
        out_p = self.out / f"metrics_{self.run}_holdout.parquet"
        started = time.time()
        if self.manifest.done("metrics", [out_p]):
            self.progress("STAGE metrics skipped (finished earlier)")
            return pd.read_parquet(out_p)
        col = M.Collector(self.run, "holdout", str(self.cfg.get("build.metrics.tape_horizon")))
        rng = np.random.default_rng(int(self.cfg.get("build.metrics.seed")))
        with self.stage("metrics"):
            hold = list(self.cal.sessions_between(self.run_cfg["holdout_start"], self.run_cfg["holdout_end"]))
            ctx = self.context(s, g)
            E.evaluate_tape(col, s["rows"], s["panel"], s["methods"], str(self.cfg.get("build.runner.main_method")), s["train_mean"], s["scale"], s["resid_sd"], ctx, self.cfg, rng, hold)
            D.coverage_per_day(s["rows"], len(set(s["panel"]["node"])), col)
            diag = D.sheaf_diagnostics(fitted.sheaf[0], fitted.sheaf[1].grounding, g["test"], fitted.scale, self.types, self.cfg, col, fit_pack.get("configs"), self.seconds)
            prices = pd.read_parquet(self.root / self.run_cfg["prices"], columns=["security_id", "session_date", "close_raw", "volume_raw"])
            themes = list(self.base.get("clean.touch.themes"))
            coefs = W.evaluate_world(col, s["clean"], prices, self.run_cfg, themes, self.base, rng)
            if len(coefs):
                coefs.to_csv(self.out / "world_theme_coefficients.csv", index=False)
            report = json.loads((self.root / str(self.base.get("clean.output_dir")) / f"clean_{self.run}_report.json").read_text())
            tape = pd.read_parquet(self.root / self.run_cfg["tape"], columns=["status", "parse_ok", "cause_top", "reading_pins_json"])
            kdiag = D.data_diagnostics(report, tape, s["clean"])
            dev_days = list(self.cal.sessions_between(self.run_cfg["fit_start"], self.run_cfg["fit_end"]))
            ldiag = D.leakage_checks(self.run_cfg, s["clean"], s["rows"], dev_days, hold, g["counts"], self.cfg, False)
            D.write_json(self.out / f"diagnostics_{self.run}_holdout.json", {"sheaf": diag, "data_and_cleaning": kdiag, "leakage_and_sanity": ldiag, "stage_seconds": self.seconds})
            frame = col.frame()
            frame.to_parquet(out_p, index=False)
        self.manifest.mark("metrics", "done", started, [out_p, self.out / f"diagnostics_{self.run}_holdout.json"], rows=len(frame))
        return frame

    # ---- ablations ---------------------------------------------------------------------------------------------------------------------
    def do_ablations(self, g: dict[str, Any], fitted: rr.Fitted, s: dict[str, Any]) -> pd.DataFrame | None:
        if not bool(self.cfg.get("build.ablations.enabled")):
            return None
        out_p = self.out / f"metrics_{self.run}_ablations.parquet"
        started = time.time()
        if self.manifest.done("ablations", [out_p]):
            self.progress("STAGE ablations skipped (finished earlier)")
            return pd.read_parquet(out_p)
        col = M.Collector(self.run, "ablation", str(self.cfg.get("build.metrics.tape_horizon")))
        rng = np.random.default_rng(int(self.cfg.get("build.metrics.seed")) + 1)
        dev, test = usable(g["dev"]), g["test"]
        scale = fitted.scale
        dates = [g["day0"][x.event_id] for x in dev]
        variants: dict[str, tuple[Callable[[EventGraph], EventGraph], str]] = {}
        for name, keep in self.cfg.get("build.ablations.link_sets").items():
            idx = [self.types.index(t) for t in keep]
            variants[name] = (lambda gr, idx=idx: restrict_links(gr, idx), "links")
        trust = json.loads((self.root / str(self.base.get("clean.output_dir")) / f"clean_{self.run}_report.json").read_text())
        trust = {sx["step"]: sx for sx in trust["steps"]}["trust"]["trust_weights"]
        kinds = g["kinds"]
        variants["no_trust"] = (lambda gr: with_pinned(gr, gr.s / float(trust.get(kinds.get(gr.event_id), 1.0))), "inputs")
        variants["no_direction_weight"] = (lambda gr: with_pinned(gr, np.column_stack([np.abs(gr.s[:, 0]), gr.s[:, 1:]])), "inputs")
        rows_all = []
        with self.stage("ablations"):
            for name, (fn, _) in variants.items():
                self.progress(f"ablation {name}: fitting the sheaf")
                t0 = time.time()
                d2 = standardise([fn(x) for x in dev], scale)
                cv_main = fitted.sheaf[1]
                steps = max(int(np.mean(cv_main.best_steps)), 1)
                model = ckpt.get("ablation", self.run, name, cv_main.grounding, cv_main.penalty_mu, steps)
                if model is None:  # one fit with the main sheaf's chosen settings, not a new grid
                    model = train_model(new_model("sheaf", self.cfg), d2, None, grounding=cv_main.grounding, penalty_mu=cv_main.penalty_mu, cfg=self.cfg, seed_parts=(self.run, "ablation", name), fixed_steps=steps).model
                    ckpt.put("ablation", model, self.run, name, cv_main.grounding, cv_main.penalty_mu, steps)
                cv = cv_main
                rows = []
                for graph in test:
                    gx = fn(graph)
                    if graph.y is None or not graph.n_free:
                        continue
                    pred = predict_graph(model, gx, cv.grounding, scale, self.cfg)
                    for j, node in enumerate(graph.nodes[graph.n_pinned :]):
                        for k, dial in enumerate(DIALS):
                            if np.isfinite(graph.y[j, k]):
                                rows.append({"event_id": graph.event_id, "day": g["day0"][graph.event_id], "node": node, "dial": dial, "y": graph.y[j, k], "pred_ablation": pred[j, k]})
                r = pd.DataFrame(rows)
                base_rows = s["rows"][["event_id", "node", "dial", "has_industry", "has_corr"]].drop_duplicates(["event_id", "node", "dial"])
                r = r.merge(base_rows, on=["event_id", "node", "dial"], how="left")
                r["has_industry"], r["has_corr"] = r["has_industry"].fillna(False), r["has_corr"].fillna(False)
                linked = r[r["has_industry"] | r["has_corr"]]
                for dial in DIALS:
                    f = E.evaluation_frame(linked.rename(columns={"pred_ablation": "pred_ablation"}), "ablation", dial, float(s["train_mean"][dial]))
                    if len(f) < int(M.mcfg(self.cfg, "min_rows_per_slice")):
                        continue
                    sd = float(np.std(f["y"] - f["pred"])) or float(scale[DIALS.index(dial)])
                    res = M.all_metrics(f, dial, float(scale[DIALS.index(dial)]), sd, E.probability(f["pred"].to_numpy() - float(s["train_mean"][dial]), sd), self.cfg, rng, draws=int(M.mcfg(self.cfg, "slice_bootstrap_draws")))
                    M.write_metrics(col, dial, "ablation", name, res, len(f), name, E.VIEW_LINKED)
                col.add("all", "fit_time", name, "seconds", time.time() - t0, method=name)
                rows_all.append(name)
            for name in self.cfg.get("build.ablations.needs_a_new_cleaning"):
                col.add("all", "ablation_not_run", name, "needs_a_new_cleaned_tape", 1.0, method=name)
            for name in ("none",):
                col.add("all", "ablation_note", name, "no_ripple_baseline_is_the_training_mean", 1.0, method=name)
        frame = col.frame()
        frame.to_parquet(out_p, index=False)
        self.manifest.mark("ablations", "done", started, [out_p], variants=rows_all)
        return frame

    # ---- write -------------------------------------------------------------------------------------------------------------------------
    def do_write(self, parts: list[pd.DataFrame]) -> None:
        started = time.time()
        frame = pd.concat([p for p in parts if p is not None and len(p)], ignore_index=True)
        path = self.out / f"ALL_METRICS_{self.run}.parquet"
        frame.to_parquet(path, index=False)
        tables = self.out / "tables"
        tables.mkdir(exist_ok=True)
        view = frame[(frame["slice"] == "view") & (frame["metric"].isin(["r2_os", "gain_pct", "rmse", "dm_p", "cw_p", "slope", "slope_t_clustered", "ic_mean", "rank_ic_mean", "dir_auc"]))]
        view.pivot_table(index=["dial", "method", "view"], columns="metric", values="value").reset_index().to_csv(tables / f"main_{self.run}.csv", index=False)
        frame[frame["slice"] == "placebo"].to_csv(tables / f"placebo_{self.run}.csv", index=False)
        frame[frame["slice"].str.startswith("world_")].to_csv(tables / f"world_lane_{self.run}.csv", index=False)
        frame[frame["slice"] == "ablation"].to_csv(tables / f"ablation_{self.run}.csv", index=False)
        frame[frame["slice"].isin(["link_type", "weight_decile"])].to_csv(tables / f"link_type_dose_response_{self.run}.csv", index=False)
        frame[~frame["slice"].isin(["view", "placebo", "ablation", "link_type", "weight_decile"]) & ~frame["slice"].str.startswith("world_")].to_csv(tables / f"appendix_slices_and_diagnostics_{self.run}.csv", index=False)
        self.manifest.mark("write", "done", started, [path], rows=len(frame))
        self.progress(f"WROTE {path} ({len(frame)} rows) and tables/")


DIALS = ("ret", "vol", "volu")


def run_all(args: Any, log: Callable[[str], None]) -> int:
    base = load_config(Path(args.config))
    runner = Runner(args.run, Path(args.root), Path(args.out), base, log, args.workers, args.smoke, bool(args.fast_fit))
    if args.open_test:
        log("--open_test: the sealed test window of this build is opened only through the separate test stage after FREEZE.json matches; that stage is not part of this run")
        return 2
    runner.do_clean()
    graphs = runner.do_graphs()
    fitted = runner.do_fit(graphs)
    if fitted is None:
        return 0  # shard mode: the fits owned by this shard are stored
    fit_pack = pickle.loads((runner.out / str(runner.cfg.get("build.runner.fit_cache"))).read_bytes())
    scored = runner.do_score(graphs, fitted)
    main = runner.do_metrics(graphs, fitted, scored, fit_pack)
    runner.do_write([main])  # main results are on disk before any ablation starts
    abl = runner.do_ablations(graphs, fitted, scored)
    runner.do_write([main, abl])
    return 0
