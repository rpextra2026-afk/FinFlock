"""What one Kaggle session runs: read the shard manifest, build the job, run the shard, save one bundle.

Command line: ``python -m finfm.compute.runner --shard <folder> --out <dir> [--inputs <dir> ...]``. Bundles found
in ``--inputs`` (previously uploaded datasets) are verified and resumed from. The session clock starts when
this module starts; the worker stops itself before ``kaggle.session_cutoff_minutes``.
"""

from __future__ import annotations

import argparse
import json
import signal
import sys
import time
from collections.abc import Sequence
from pathlib import Path

from finfm.common.config import Config, load_config
from finfm.compute.engine import Engine
from finfm.compute.factory import HFTokenizer, build_engine, resolve_model_path
from finfm.compute.jobs import JOBS, planned_for_shard
from finfm.compute.parallel import describe
from finfm.compute.shard import is_single_run
from finfm.compute.staging import INFRASTRUCTURE_SECTIONS, read_manifest
from finfm.compute.worker import Clock, ShardResult, StopFlag, run_shard
from finfm.reader.prompt import Tokenizer, WhitespaceTokenizer

CODE_VERSION_FILE = "CODE_VERSION"


def code_version(repo_root: Path) -> str | None:
    path = repo_root / CODE_VERSION_FILE
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def run_shard_from_folder(
    folder: str | Path, out_dir: str | Path, config_dir: str | Path, *, input_dirs: Sequence[str | Path] = (),
    engine: Engine | None = None, tokenizer: Tokenizer | None = None, clock: Clock | None = None,
    stop_flag: StopFlag | None = None, overrides: dict | None = None, repo_root: Path | None = None,
    limit_units: int | None = None, session_started: float | None = None,
) -> ShardResult:
    """Run one staged shard. ``engine`` / ``tokenizer`` are injected for tests; otherwise built from config."""
    cfg: Config = load_config(config_dir, overrides)
    manifest = read_manifest(folder)
    if manifest["science_hash"] != cfg.hash_without(INFRASTRUCTURE_SECTIONS):
        raise RuntimeError("config hash differs from the one the shard was staged with; refuse to run")
    own_engine = engine is None
    engine = engine or build_engine(cfg, manifest["llm"], str(config_dir))
    if tokenizer is None:
        mock = str(cfg.get("engine.engine")) == "mock"
        tokenizer = WhitespaceTokenizer() if mock else HFTokenizer(
            resolve_model_path(cfg, str(cfg.get(f"engine.models.{manifest['llm']}.path"))))
    job = JOBS[manifest["job"]](cfg, Path(folder), engine, tokenizer)
    planned = planned_for_shard(job.planned, manifest["unit_ids"])
    if limit_units:  # a trial: the first N units of the shard; a later full run continues from them
        planned = planned.head(int(limit_units))
    meta = {"job": manifest["job"], "bucket": manifest["bucket"], "llm": manifest["llm"], "engine": describe(engine),
            "code_version": code_version(repo_root) if repo_root else None}
    try:
        return run_shard(cfg=cfg, job_id=manifest["job_id"], shard_id=manifest["shard_id"], planned=planned,
                         processor=job.processor, input_dirs=list(input_dirs), out_dir=out_dir, meta=meta,
                         clock=clock, stop_flag=stop_flag, retry_errors=is_single_run(cfg, manifest["job_id"]),
                         session_offset_seconds=max(time.time() - session_started, 0.0) if session_started else 0.0)
    finally:
        if own_engine:
            try:
                engine.close()
            except Exception as exc:  # the result is already saved; a failing shutdown must not turn a finished run into an error
                print(f"WARNING engine shutdown failed after the run: {exc!r}", flush=True)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run one staged FinFM shard")
    parser.add_argument("--shard", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--config", default="config")
    parser.add_argument("--inputs", nargs="*", default=[])
    args = parser.parse_args(argv)
    stop = StopFlag()
    signal.signal(signal.SIGTERM, lambda *_: stop.set())
    result = run_shard_from_folder(args.shard, args.out, args.config, input_dirs=args.inputs, stop_flag=stop,
                                   repo_root=Path(args.config).resolve().parent)
    print(json.dumps({"outcome": result.outcome.value, "abort": result.abort_reason.value if result.abort_reason else None,
                      "planned": result.n_planned, "terminal": result.n_terminal, "remaining": result.n_remaining,
                      "bundle": str(result.bundle_path), "seconds": round(result.seconds, 1),
                      "rejected_inputs": [[str(p), why] for p, why in result.rejected_inputs]}))
    return 0 if result.n_remaining == 0 or result.outcome.value in ("session_limit", "terminated") else 1


if __name__ == "__main__":
    sys.exit(main())
