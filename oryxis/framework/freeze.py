"""FREEZE_FRAMEWORK.json: the hashes the framework run is locked to (cleaned tapes, config, code, persona tapes, dev fit).

The run refuses to start when any hash differs. The older FREEZE.json (the cleaning and sheaf build) is never edited; this is a new file. ``dev_fit`` may be ``null``
in the file when the fit is made inside the run (a Kaggle notebook): the run then records the hash of ``dev_fit.pkl`` the moment it is written and refuses a changed file
afterwards (``FREEZE_FRAMEWORK_runtime.json`` in the results folder).
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path
from typing import Any

from oryxis.build.run import code_hash
from oryxis.common.config import Config
from oryxis.common.hashing import sha256_file
from oryxis.compute.bundle import make_job_id, scan_bundles
from oryxis.real.tapes import tape_dirs

FREEZE_NAME = "FREEZE_FRAMEWORK.json"
RUNTIME_NAME = "FREEZE_FRAMEWORK_runtime.json"
SHORT = 12
FRAMEWORK_SCRIPTS = ("run_framework_final.py", "run_framework_sealed.py", "run_framework_sharded.py")


def git_head(code_root: Path) -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=code_root, capture_output=True, text=True, check=False).stdout.strip()


def persona_files(cfg: Config, root: Path, run: str) -> dict[str, str]:
    """sha256 of every valid persona bundle of the run's bucket and LLM, by file name."""
    bucket, llm = run.split("-")
    scan = scan_bundles(tape_dirs(cfg, root), make_job_id("personas", bucket, llm))
    return {ref.path.name: sha256_file(ref.path) for ref in scan.valid}


def current_hashes(base: Config, root: Path, code_root: Path, runs: list[str], dev_fit: dict[str, Path | None] | None = None) -> dict[str, Any]:
    clean_dir = root / str(base.get("clean.output_dir"))
    content, files = {}, {}
    for run in runs:
        report = json.loads((clean_dir / f"clean_{run}_report.json").read_text())
        content[run] = report["content_sha256"]
        files[run] = sha256_file(clean_dir / f"clean_{run}.parquet")
    extra = {name: sha256_file(code_root / "scripts" / name) for name in FRAMEWORK_SCRIPTS if (code_root / "scripts" / name).exists()}
    dev = {run: (sha256_file(path) if path is not None and Path(path).exists() else None) for run, path in (dev_fit or {}).items()}
    final = base.get("framework.final")
    settings = {"headline_set": final["headline_set"], "second_set": final["second_set"], "headline_sets": list(final["headline_sets"]), "delisting_rules": list(final["delisting_rules"]),
                "risk_target_annual_vol": float(base.get("portfolio.vol_match.target_annual_vol")), "delisting_default_return": float(base.get("portfolio.delisting.default_return")),
                "world_lane": dict(base.get("framework.world_lane"))}
    cal_dir = root / str(base.get("framework.calibration.dir"))
    file_of = lambda name: cal_dir / f"volu_calibration_{name}.json"  # noqa: E731
    calibration = {run: (sha256_file(file_of(run)) if file_of(run).exists() else None) for run in runs}
    all_runs = list(base.get("clean.runs"))   # the settings record is the same for every run: the values of ALL tapes, not only the ones being checked
    values = {run: ({k: json.loads(file_of(run).read_text())[k] for k in ("used", "scale", "shift")} if file_of(run).exists() else None) for run in all_runs}
    settings["calibration"] = {"enabled": bool(base.get("framework.calibration.enabled")), "folds": int(base.get("framework.calibration.folds")), "max_scale": float(base.get("framework.calibration.max_scale")), "values": values}
    return {"settings": settings, "calibration_sha256": calibration, "config_hash": base.hash, "config_unhashed_hash": base.unhashed_hash, "code_sha256": code_hash(code_root), "framework_scripts_sha256": extra,
            "clean_content_sha256": content, "clean_file_sha256": files, "persona_tape_sha256": {run: persona_files(base, root, run) for run in runs}, "dev_fit_sha256": dev}


def write_freeze(path: Path, hashes: dict[str, Any], code_root: Path) -> dict[str, Any]:
    if path.exists():
        raise SystemExit(f"{path} exists; the freeze file is written once and never overwritten")
    git = subprocess.run(["git", "rev-parse", "HEAD"], cwd=code_root, capture_output=True, text=True, check=False).stdout.strip()
    record = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git_head": git, **hashes}
    path.write_text(json.dumps(record, indent=1, sort_keys=True))
    return record


def problems(frozen: dict[str, Any], now: dict[str, Any], runs: list[str]) -> list[str]:
    """Differences between the frozen hashes and the current ones for ``runs`` (empty = the run may start)."""
    out = []
    if frozen.get("settings") is not None and frozen["settings"] != now.get("settings"):
        out.append(f"settings: frozen {frozen['settings']}, now {now.get('settings')}")
    for key in ("config_hash", "config_unhashed_hash", "code_sha256", "framework_scripts_sha256"):
        if frozen.get(key) is not None and frozen[key] != now[key]:
            out.append(f"{key}: frozen {str(frozen[key])[:SHORT]}, now {str(now[key])[:SHORT]}")
    for key in ("clean_content_sha256", "clean_file_sha256", "persona_tape_sha256", "dev_fit_sha256", "calibration_sha256"):
        for run in runs:
            want, have = (frozen.get(key) or {}).get(run), (now.get(key) or {}).get(run)
            if want is not None and want != have:
                out.append(f"{key}[{run}]: frozen {str(want)[:SHORT]}, now {str(have)[:SHORT]}")
    return out


def check_static(freeze_path: Path, base: Config, root: Path, code_root: Path, run: str) -> dict[str, Any]:
    """The freeze check without the dev fit (which is rebuilt later in the run and checked then)."""
    if not freeze_path.exists():
        raise SystemExit(f"{freeze_path} is missing: the framework run does not start without its freeze file (scripts/freeze_framework.py writes it)")
    frozen = json.loads(freeze_path.read_text())
    bad = [b for b in problems(frozen, current_hashes(base, root, code_root, [run], None), [run]) if not b.startswith("dev_fit_sha256")]
    if bad:
        raise SystemExit("FREEZE_FRAMEWORK.json does not match this run; refusing to start:\n  " + "\n  ".join(bad))
    return {"freeze": "matched", "freeze_file": str(freeze_path), "created_utc": frozen.get("created_utc")}


def check(freeze_path: Path, base: Config, root: Path, code_root: Path, run: str, dev_fit_path: Path | None, runtime_path: Path) -> dict[str, Any]:
    """Stop unless the run's hashes match the freeze file; pin ``dev_fit.pkl`` at its first appearance when the file did not carry its hash."""
    if not freeze_path.exists():
        raise SystemExit(f"{freeze_path} is missing: the framework run does not start without its freeze file (scripts/freeze_framework.py writes it)")
    frozen = json.loads(freeze_path.read_text())
    now = current_hashes(base, root, code_root, [run], {run: dev_fit_path})
    bad = problems(frozen, now, [run])
    runtime = json.loads(runtime_path.read_text()) if runtime_path.exists() else {}
    frozen_fit = (frozen.get("dev_fit_sha256") or {}).get(run)
    if frozen_fit is None and now["dev_fit_sha256"].get(run) is not None:
        pinned = runtime.get("dev_fit_sha256")
        if pinned is None:
            runtime_path.write_text(json.dumps({"dev_fit_sha256": now["dev_fit_sha256"][run], "run": run}, indent=1))
        elif pinned != now["dev_fit_sha256"][run]:
            bad.append(f"dev_fit.pkl changed after it was first pinned ({pinned[:SHORT]} -> {now['dev_fit_sha256'][run][:SHORT]})")
    if bad:
        raise SystemExit("FREEZE_FRAMEWORK.json does not match this run; refusing to start:\n  " + "\n  ".join(bad))
    return {"freeze": "matched", "freeze_file": str(freeze_path), "created_utc": frozen.get("created_utc")}
