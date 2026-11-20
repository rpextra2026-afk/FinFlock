"""Source-only entry points for the reader, sheaf, personas and merged framework."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODELS = {"gemma": "llm1", "olmo": "llm2", "llama": "llm3"}
BUCKETS = {"databucket1": "db1", "tessera": "tessera", "databucket3": "db3"}

def main(stage=None, bucket=None, model=None, argv=None):
    parser = argparse.ArgumentParser(description="Run a stage using local, separately supplied inputs.")
    if stage is None:
        parser.add_argument("--stage", required=True, choices=["reader", "sheaf", "personas", "final_framework"])
        parser.add_argument("--bucket", required=True, choices=BUCKETS)
        parser.add_argument("--model", required=True, choices=MODELS)
    parser.add_argument("--config", type=Path, default=ROOT / "config")
    parser.add_argument("--root", type=Path, default=ROOT, help="Local data tree with dataset/, store/ and generated outputs/.")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--check-inputs", action="store_true", help="Validate required input paths and exit without running models or fitting.")
    args, extra = parser.parse_known_args(argv)
    if stage is None:
        stage, bucket, model = args.stage, args.bucket, args.model
    bucket_id, llm = BUCKETS[bucket], MODELS[model]
    run = f"{bucket_id}-{llm}"
    from finfm.common.config import load_config
    cfg = load_config(args.config)
    if stage in ("sheaf", "final_framework"):
        run_cfg = cfg.get(f"clean.runs.{run}")
        required = [args.root / run_cfg[k] for k in ("tape", "units", "links", "prices")]
        required += [args.root / cfg.get("build.paths.factors_file")]
        if stage == "final_framework":
            market = args.root / cfg.get("cpu.folders.market")
            required += [market / "f13_managers.parquet", market / "prestaged" / f"personas-{bucket_id}" / "skeletons.json"]
        missing = [str(p) for p in required if not p.exists()]
        if missing:
            parser.error("Missing local inputs:\n" + "\n".join(missing))
        if args.check_inputs:
            return 0
        params = [run, "--config", str(args.config), "--root", str(args.root)]
        if args.out:
            params += ["--out", str(args.out)]
        params += extra
        if stage == "sheaf":
            from scripts.run_final_build import main as run_stage
        else:
            from scripts.run_framework_final import main as run_stage
        return run_stage(params)
    gpu = argparse.ArgumentParser(description="GPU stage arguments")
    gpu.add_argument("--input-root", type=Path, required=True)
    gpu.add_argument("--shard-id")
    gpu.add_argument("--limit-units", type=int)
    opts = gpu.parse_args(extra)
    if not opts.input_root.is_dir():
        parser.error("--input-root must be a directory containing local inputs")
    if args.check_inputs:
        parser.error("GPU inputs need their stage-specific manifests; --check-inputs is supported for sheaf and final_framework")
    out = args.out or args.root / "outputs" / stage / run
    if stage == "reader":
        from reader2.session import run_reader2
        run_reader2(bucket=bucket_id, llm=llm, pack_root=opts.input_root,
                    config_dir=args.config, input_root=opts.input_root, out_dir=out,
                    shard=opts.shard_id, limit_units=opts.limit_units)
    else:
        from finfm.compute.session import RunSpec, run_session
        from finfm.stage.run_stage import make_stager
        spec = RunSpec(f"personas-{run}", "personas", bucket_id, llm)
        run_session(spec, args.config, opts.input_root, out,
                    shard_id=opts.shard_id, limit_units=opts.limit_units,
                    stage=make_stager(spec, args.config, opts.input_root))
    return 0

if __name__ == "__main__":
    sys.exit(main())
