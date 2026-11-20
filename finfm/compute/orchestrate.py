"""Planning, pushing, polling and pulling Kaggle jobs (spec 2.3). All CLI calls go through ``KaggleCLI``.

The syntax is the installed ``kaggle`` command's own (read from its help): ``kernels push -p <folder> -t
<seconds> [--accelerator <name>]``, ``kernels status <owner>/<slug>``, ``kernels output <owner>/<slug> -p
<dir>``, ``datasets create|version -p <folder>``, ``quota``. Each account has its own credentials folder,
selected through an environment variable (a VERIFY item); credentials never enter the repository.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import pandas as pd

from finfm.common.config import Config
from finfm.common.units import SECONDS_PER_MINUTE
from finfm.compute.bundle import iter_bundle_paths, scan_bundles, verify_bundle
from finfm.compute.shard import Shard, assign_shards, plan_shards

Runner = Callable[[Sequence[str], Mapping[str, str]], subprocess.CompletedProcess]
KERNEL_METADATA = "kernel-metadata.json"
DATASET_METADATA = "dataset-metadata.json"


def _run(command: Sequence[str], env: Mapping[str, str]) -> subprocess.CompletedProcess:
    return subprocess.run(list(command), env={**os.environ, **env}, capture_output=True, text=True, check=False)


class KaggleCLI:
    """Thin wrapper over the ``kaggle`` command for one account (its credentials folder)."""

    def __init__(self, cfg: Config, account_dir: str | Path, runner: Runner = _run) -> None:
        self.cfg, self.runner = cfg, runner
        self.env = {str(cfg.get("kaggle.config_dir_env")): str(account_dir)}

    def call(self, *args: str) -> subprocess.CompletedProcess:
        result = self.runner(["kaggle", *args], self.env)
        if result.returncode != 0:
            raise RuntimeError(f"kaggle {' '.join(args)} failed: {result.stderr.strip() or result.stdout.strip()}")
        return result

    def push_kernel(self, folder: Path) -> None:
        args = ["kernels", "push", "-p", str(folder), "-t", str(int(float(self.cfg.get("kaggle.session_cutoff_minutes")) * SECONDS_PER_MINUTE))]
        accelerator = self.cfg.get("kaggle.accelerator")
        if accelerator:
            args += ["--accelerator", str(accelerator)]
        self.call(*args)

    def kernel_status(self, owner: str, slug: str) -> str:
        return self.call("kernels", "status", f"{owner}/{slug}").stdout.strip()

    def pull_output(self, owner: str, slug: str, folder: Path) -> None:
        folder.mkdir(parents=True, exist_ok=True)
        self.call("kernels", "output", f"{owner}/{slug}", "-p", str(folder), "-o")

    def create_dataset(self, folder: Path) -> None:
        self.call("datasets", "create", "-p", str(folder), "-r", "tar")

    def new_dataset_version(self, folder: Path, message: str) -> None:
        self.call("datasets", "version", "-p", str(folder), "-m", message, "-r", "tar")

    def quota(self) -> list[dict]:
        out = self.call("quota", "--format", "json").stdout
        return json.loads(out) if out.strip() else []


def slug_for(prefix: str, *parts: str) -> str:
    """Kaggle slug: lower-case letters, digits and hyphens."""
    return "-".join([prefix, *parts]).lower().replace("_", "-")


def write_dataset_folder(folder: Path, owner: str, slug: str, title: str) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / DATASET_METADATA).write_text(json.dumps({"title": title, "id": f"{owner}/{slug}", "licenses": [{"name": "other"}]}), encoding="utf-8")
    return folder


def write_kernel_folder(
    folder: Path, owner: str, slug: str, title: str, dataset_sources: Sequence[str], cfg: Config, script_source: Path,
) -> Path:
    """kernel-metadata.json plus the script. GPU on; internet per config; every source is a private dataset."""
    folder.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(script_source, folder / str(cfg.get("kaggle.kernel_main")))
    metadata = {
        "id": f"{owner}/{slug}", "title": title, "code_file": str(cfg.get("kaggle.kernel_main")), "language": "python",
        "kernel_type": "script", "is_private": True, "enable_gpu": True,
        "enable_internet": bool(cfg.get("kaggle.enable_internet")), "dataset_sources": list(dataset_sources),
        "competition_sources": [], "kernel_sources": [],
    }
    (folder / KERNEL_METADATA).write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return folder


def plan_job(job_id: str, units: pd.DataFrame, cfg: Config) -> tuple[list[Shard], pd.DataFrame]:
    """Shards of one job and their static account/wave assignment."""
    shards = plan_shards(job_id, units, cfg)
    return shards, assign_shards(shards, cfg)


def collect_outputs(pulled: Path, job_id: str) -> dict:
    """Verify every bundle pulled for a job and say which were usable (exactly-once merging comes next)."""
    scan = scan_bundles([pulled], job_id)
    return {"valid": [str(r.path) for r in scan.valid], "rejected": [[str(p), why] for p, why in scan.rejected],
            "files": [str(p) for p in iter_bundle_paths(pulled)], "all_verify": all(verify_bundle(p).ok for p in iter_bundle_paths(pulled))}


def ledger_rows(account: str, quota: list[dict]) -> pd.DataFrame:
    """One row per quota entry reported by ``kaggle quota`` for an account (reconciled by hand with the quota page)."""
    return pd.DataFrame([{"account": account, **entry} for entry in quota])
