"""What a Kaggle notebook does inside a session: check the machine, find the attached files, run one shard.

The notebook only unpacks the code and calls ``run_session``. Everything it needs is found under the input
folder (the attached datasets and models): the staged shard folders of this run, the model weights, and any
earlier output bundles to resume from. Nothing is written to the output folder except the one bundle.
"""

from __future__ import annotations

import json
import os
import signal
import sys
import subprocess
import tempfile
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from oryxis.common.config import Config, load_config
from oryxis.common.schema import PARSE_OK, REASON, TERMINAL_STATUSES, UNIT_ID, STATUS, Status
from oryxis.compute.bundle import make_job_id, read_consolidated, scan_bundles, unit_state
from oryxis.compute.factory import resolve_model_path
from oryxis.compute.runner import run_shard_from_folder
from oryxis.compute.shard import is_single_run, shard_sort_key
from oryxis.compute.staging import MANIFEST, read_manifest
from oryxis.compute.worker import StopFlag

WEIGHTS_MARKER = "config.json"
WEIGHT_SUFFIXES = (".safetensors", ".bin")
COMMAND_TIMEOUT_SECONDS = 30
ERROR_TAIL_CHARS = 600  # how much of a failing import's message is shown
LOG_TAIL_CHARS = 120  # and how much of it goes into a one-line log message


class SessionError(RuntimeError):
    """The session cannot start (wrong accelerator, missing attachment ...). Raised before any GPU time is spent."""


@dataclass(frozen=True)
class RunSpec:
    run_id: str
    job: str
    bucket: str
    llm: str

    def matches(self, manifest: Mapping[str, Any]) -> bool:
        return (manifest["job"], manifest["bucket"], manifest["llm"]) == (self.job, self.bucket, self.llm)


def gpu_count() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECONDS, check=False)
    except (OSError, subprocess.SubprocessError):
        return 0
    return sum(1 for line in out.stdout.splitlines() if line.startswith("GPU"))


def required_gpus(cfg: Config, llm: str) -> int:
    spec = cfg.get(f"engine.models.{llm}")
    return int(spec["tensor_parallel"]) * int(spec["replicas"])


def check_machine(cfg: Config, llm: str, gpus: int | None = None) -> int:
    """Stop at once if the notebook is not on enough GPUs (no model load, no wasted session)."""
    if str(cfg.get("engine.engine")) == "mock":
        return 0
    have = gpu_count() if gpus is None else gpus
    need = required_gpus(cfg, llm)
    if have < need:
        raise SessionError(f"this run needs {need} GPUs and the session has {have}: run it on a machine with "
                           f"{need} GPUs (the 2 x A30 machine, or GPU T4 x2 on Kaggle) and run again")
    return have


def find_weights(input_root: Path, tokens: Sequence[str]) -> Path | None:
    """The attached folder that holds model weights and whose path contains every token (case-insensitive)."""
    wanted = [t.lower() for t in tokens]
    hits = []
    for marker in sorted(input_root.rglob(WEIGHTS_MARKER)):
        folder = marker.parent
        if not any(f.suffix in WEIGHT_SUFFIXES for f in folder.iterdir()):
            continue
        text = str(folder.relative_to(input_root)).lower()
        if all(t in text for t in wanted):
            hits.append(folder)
    return hits[0] if hits else None


def link_weights(cfg: Config, llm: str, input_root: Path, work: Path) -> Path | None:
    """Make ``<work>/weights/<org>--<name>`` point at the attached weights and select it through the environment."""
    spec = cfg.get(f"engine.models.{llm}")
    folder = find_weights(input_root, spec["weights_match"])
    if folder is None:
        return None
    root = work / "weights"
    root.mkdir(parents=True, exist_ok=True)
    link = root / str(spec["path"]).replace("/", "--")
    if not link.exists():
        link.symlink_to(folder, target_is_directory=True)
    os.environ[str(cfg.get("engine.weights_root_env"))] = str(root)
    return folder


def ensure_runtime(cfg: Config, input_root: Path, run=subprocess.run, log=print) -> None:
    """vLLM must import its ``LLM`` class. Repairs the torch add-ons first, installs vLLM if missing, repairs again, then checks.

    Installing vLLM can pull a newer torch than the image's torchaudio / torchvision were built for, and ``transformers`` then fails
    while importing them ("PyTorch and TorchAudio were compiled with different CUDA versions"). vLLM's text path needs neither, so a
    companion that fails to import is uninstalled (``engine.broken_companions``). Every step is logged to the notebook.
    """
    if str(cfg.get("engine.engine")) == "mock":
        return

    def attempt(code: str):
        return run([sys.executable, "-c", code], capture_output=True, text=True, check=False)

    def repair() -> None:
        for package in cfg.get("engine.broken_companions"):
            result = attempt(f"import {package}")
            if result.returncode != 0:
                log(f"runtime: {package} does not import ({(result.stderr or '').strip()[-LOG_TAIL_CHARS:]}); uninstalling it")
                run([sys.executable, "-m", "pip", "uninstall", "-y", package], capture_output=True, check=False)

    repair()
    if attempt("from vllm import LLM").returncode == 0:
        log("runtime: vllm imports")
        return
    wheels = sorted({p.parent for p in input_root.rglob("*.whl")})
    log(f"runtime: installing vllm ({'attached wheels' if wheels else 'from the internet'})")
    command = [sys.executable, "-m", "pip", "install", "-q"] + (["--no-index", "--find-links", str(wheels[0]), "vllm"] if wheels else ["vllm"])
    run(command, capture_output=True, check=False)
    repair()
    last = attempt("from vllm import LLM")
    if last.returncode != 0:
        raise SessionError("vllm cannot be imported after install and repair; last error:\n" + (last.stderr or "").strip()[-ERROR_TAIL_CHARS:])
    log("runtime: vllm imports after install")


def download_weights(cfg: Config, llm: str, work: Path, token: str | None) -> Path | None:
    """Fetch the model from the Hugging Face hub when it is not attached (needs internet and a token that accepted the licence)."""
    spec = cfg.get(f"engine.models.{llm}")
    if not token:
        return None
    from huggingface_hub import snapshot_download

    root = work / "weights"
    target = root / str(spec["path"]).replace("/", "--")
    snapshot_download(repo_id=str(spec["path"]), local_dir=str(target), token=token)
    os.environ[str(cfg.get("engine.weights_root_env"))] = str(root)
    return target


def staged_shards(input_root: Path, run: RunSpec) -> tuple[list[tuple[Path, dict]], list[str]]:
    """Shard folders of this run, in shard order, and a note for every shard folder of another run."""
    mine, other = [], []
    for path in sorted(input_root.rglob(MANIFEST)):
        manifest = read_manifest(path.parent)
        if run.matches(manifest):
            mine.append((path.parent, manifest))
        else:
            other.append(f"{path.parent}: {manifest['job']}/{manifest['bucket']}/{manifest['llm']}")
    mine.sort(key=lambda item: shard_sort_key(item[1]["shard_id"]))
    return mine, other


def remaining_units(manifest: Mapping[str, Any], input_root: Path) -> int:
    """Units of the shard that no attached bundle has finished yet."""
    state = unit_state(scan_bundles([input_root], manifest["job_id"]).valid)
    done = set(state.loc[state[STATUS].isin(TERMINAL_STATUSES), UNIT_ID]) if len(state) else set()
    return sum(1 for u in manifest["unit_ids"] if u not in done)


def pick_shard(input_root: Path, run: RunSpec, shard_id: str | None = None, staged_root: Path | None = None) -> tuple[Path, dict]:
    """The requested shard, else the first one of this run that is not finished, judged from the bundles attached to the session.

    Shard folders are looked up in the attached data and in ``staged_root`` (shards the session built itself).
    """
    mine, other = staged_shards(input_root, run)
    if staged_root is not None:
        extra_mine, extra_other = staged_shards(staged_root, run)
        mine, other = sorted(mine + extra_mine, key=lambda item: shard_sort_key(item[1]["shard_id"])), other + extra_other
    if shard_id is not None:
        shard_id = str(shard_id)
        wanted = [item for item in mine if item[1]["shard_id"] == shard_id]
        if not wanted:
            raise SessionError(f"run {run.run_id} has no shard {shard_id}; its shards are {[m['shard_id'] for _, m in mine]}")
        if not remaining_units(wanted[0][1], input_root):
            raise SessionError(f"shard {shard_id} of {run.run_id} is already finished in the attached bundles")
        return wanted[0]
    if any(m.get("explicit_shard") for _, m in mine):
        raise SessionError(f"run {run.run_id} is split over several accounts: set SHARD to this account's shard number")
    if not mine:
        found = "; ".join(other) or "no shard folders at all"
        raise SessionError(f"no data for run {run.run_id} is attached (found: {found}). Attach the data dataset of this run")
    for folder, manifest in mine:
        if remaining_units(manifest, input_root):
            return folder, manifest
    raise SessionError(f"every shard of run {run.run_id} is already finished in the attached bundles; nothing to do")


REPORT_DECIMALS = 3
CUTOFF_KEY = "kaggle.session_cutoff_minutes"
MINUTES_PER_HOUR = 60
NO_CUTOFF_MINUTES = 10**9  # a single-run job (``kaggle.single_run_jobs``) never stops for time: it ends when its units are done


def run_cutoff_minutes(cfg: Config, run_id: str) -> float:
    """Session cutoff of one run: its own entry in ``kaggle.session_cutoff_minutes_by_run``, else the general cutoff."""
    return float((cfg.get("kaggle.session_cutoff_minutes_by_run") or {}).get(run_id, cfg.get(CUTOFF_KEY)))


def describe_minutes(minutes: float) -> str:
    """``585`` -> ``9 h 45 min``, ``660`` -> ``11 h``."""
    hours, rest = divmod(int(round(minutes)), MINUTES_PER_HOUR)
    return f"{hours} h" + (f" {rest} min" if rest else "")


def quality_report(out_dir: str | Path, job_id: str, cfg: Config) -> dict[str, Any]:
    """What the finished bundle says about the answers: status and reason counts, parse rate among called units, output tokens.

    The verdict compares the parse rate with ``runtime.min_parse_rate``; it is for a human to read (the run is never stopped by it)."""
    tape = read_consolidated(scan_bundles([Path(out_dir)], job_id).valid)
    if tape.empty:
        return {"rows": 0, "verdict": "no rows"}
    called = tape[tape[STATUS] == Status.OK.value]
    parsed = called[PARSE_OK].map(lambda v: bool(v) if v == v else False) if PARSE_OK in called.columns and len(called) else None
    rate = float(parsed.mean()) if parsed is not None and len(parsed) else None
    threshold = float(cfg.get("runtime.min_parse_rate"))
    report: dict[str, Any] = {
        "rows": int(len(tape)), "status": {str(k): int(v) for k, v in tape[STATUS].value_counts().items()},
        "reasons": {str(k): int(v) for k, v in tape[REASON].value_counts().items()} if REASON in tape.columns else {},
        "called": int(len(called)), "parse_rate_of_called": None if rate is None else round(rate, REPORT_DECIMALS),
        "min_parse_rate": threshold,
    }
    if "tokens_out" in called.columns and len(called):
        report["mean_tokens_out"] = round(float(called["tokens_out"].mean()), 1)
    if "flag_cause_defaulted" in called.columns and len(called):
        report["share_cause_defaulted"] = round(float(called["flag_cause_defaulted"].fillna(False).astype(bool).mean()), REPORT_DECIMALS)
    report["verdict"] = "no called units" if rate is None else ("OK: parse rate at or above the threshold" if rate >= threshold
                                                             else "LOW: parse rate below the threshold, read the reasons before running more")
    return report


def run_session(
    run: RunSpec, config_dir: str | Path, input_root: str | Path, out_dir: str | Path, *, overrides: dict | None = None,
    engine=None, tokenizer=None, clock=None, gpus: int | None = None, shard_id: str | None = None,
    limit_units: int | None = None, stage: Callable[[Path], Path] | None = None, session_started: float | None = None,
) -> dict[str, Any]:
    """Run the next unfinished shard of ``run`` and leave one bundle in ``out_dir``. Returns the summary printed to the notebook."""
    cfg = load_config(config_dir, overrides)
    cutoff = NO_CUTOFF_MINUTES if is_single_run(cfg, make_job_id(run.job, run.bucket, run.llm)) else run_cutoff_minutes(cfg, run.run_id)
    if cutoff != float(cfg.get(CUTOFF_KEY)):  # this run has its own cutoff: the worker reads it from the config it is given
        overrides = {**(overrides or {}), CUTOFF_KEY: cutoff}
        cfg = load_config(config_dir, overrides)
    input_root, out_dir = Path(input_root), Path(out_dir)
    if engine is None:  # an injected engine (tests) brings its own hardware
        check_machine(cfg, run.llm, gpus)
        ensure_runtime(cfg, input_root)
    with tempfile.TemporaryDirectory() as scratch:
        real = engine is None and str(cfg.get("engine.engine")) != "mock"
        weights = link_weights(cfg, run.llm, input_root, Path(scratch)) if real else None  # before staging: some jobs need the tokenizer
        if real and weights is None:
            weights = download_weights(cfg, run.llm, Path(scratch), os.environ.get(str(cfg.get("sources.hf.env_token"))))
        if engine is None and str(cfg.get("engine.engine")) != "mock" and weights is None:  # before staging: fail in seconds, not after minutes of work
            raise SessionError(f"model weights for {run.llm} are not attached (looked for a folder whose path contains "
                               f"{cfg.get(f'engine.models.{run.llm}.weights_match')}), and no HF token is set to download it")
        staged_root = stage(Path(scratch)) if stage and not staged_shards(input_root, run)[0] else None
        folder, manifest = pick_shard(input_root, run, shard_id, staged_root)
        stop = StopFlag()
        signal.signal(signal.SIGTERM, lambda *_: stop.set())
        result = run_shard_from_folder(folder, out_dir, config_dir, input_dirs=[input_root], engine=engine, tokenizer=tokenizer,
                                       clock=clock, stop_flag=stop, overrides=overrides, repo_root=Path(config_dir).resolve().parent,
                                       limit_units=limit_units, session_started=session_started)
    summary = {"run": run.run_id, "shard": manifest["shard_id"], "outcome": result.outcome.value,
               "planned": result.n_planned, "terminal": result.n_terminal, "remaining": result.n_remaining,
               "bundle": str(result.bundle_path), "seconds": round(result.seconds, 1), "weights": str(weights) if weights else None,
               "rejected_inputs": [[str(p), why] for p, why in result.rejected_inputs]}
    try:
        summary["quality"] = quality_report(out_dir, manifest["job_id"], cfg)
    except Exception as exc:  # reporting only: the bundle is saved
        summary["quality"] = {"verdict": f"report failed: {exc!r}"}
    print(json.dumps(summary, indent=2))
    return summary
