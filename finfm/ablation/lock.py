"""Locking the ablation plan with proof before any run (spec section 10).

``write_lock`` records the sha256 of ``config/ablation.yaml``, of the spec copy, of the reader and persona bundles and of the unit table, the list of run ids, the
configuration hash, the UTC time and the git commit of the tree. Commit and push the lock file: the commit time on the remote is the proof. ``verify_lock`` refuses a
run when any of those files differs from what was locked.
"""

from __future__ import annotations

import datetime as dt
import json
import subprocess
from pathlib import Path

from finfm.ablation.plan import Plan
from finfm.common.config import Config
from finfm.common.hashing import sha256_file


def lock_path(root: Path, cfg: Config, bucket: str, llm: str) -> Path:
    return root / str(cfg.get("ablation.lock_file")).format(bucket=bucket, llm=llm)


def _git(root: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=root, capture_output=True, text=True, check=False).stdout.strip()


def locked_files(root: Path, cfg: Config, tapes: list[Path], persona_tapes: list[Path], units: Path) -> dict[str, Path]:
    files = {"config/ablation.yaml": root / "config" / "ablation.yaml", str(cfg.get("ablation.spec_file")): root / str(cfg.get("ablation.spec_file")), "unit_table": units}
    files.update({f"reader_tape/{p.name}": p for p in tapes})
    files.update({f"persona_tape/{p.name}": p for p in persona_tapes})
    return files


def write_lock(root: Path, cfg: Config, bucket: str, llm: str, tapes: list[Path], persona_tapes: list[Path], units: Path) -> Path:
    plan = Plan(cfg, bucket, llm)
    payload = {"locked_utc": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"), "bucket": bucket, "llm": llm,
               "config_hash": cfg.hash, "git_commit": _git(root, "rev-parse", "HEAD"), "git_tree_dirty": bool(_git(root, "status", "--porcelain", "--", "finfm", "config", "scripts")),
               "sha256": {name: sha256_file(path) for name, path in locked_files(root, cfg, tapes, persona_tapes, units).items()},
               "run_ids": [s.run_id for s in plan.all_specs()], "not_computed": {s.name: s.reason for s in plan.not_computed()}}
    target = lock_path(root, cfg, bucket, llm)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(payload, indent=1), encoding="utf-8")
    return target


def verify_lock(root: Path, cfg: Config, bucket: str, llm: str, tapes: list[Path], persona_tapes: list[Path], units: Path) -> list[str]:
    """The reasons a run must not start (empty when every locked file is byte-identical and the run list is unchanged)."""
    target = lock_path(root, cfg, bucket, llm)
    if not target.exists():
        return [f"no lock file at {target}: run scripts/lock_ablation.py, commit and push it before any run"]
    locked = json.loads(target.read_text(encoding="utf-8"))
    problems = []
    for name, path in locked_files(root, cfg, tapes, persona_tapes, units).items():
        want = locked["sha256"].get(name)
        if want is None:
            problems.append(f"{name}: not in the lock file")
        elif not path.exists() or sha256_file(path) != want:
            problems.append(f"{name}: differs from the locked file")
    if [s.run_id for s in Plan(cfg, bucket, llm).all_specs()] != locked["run_ids"]:
        problems.append("the run list differs from the locked run list")
    return problems
