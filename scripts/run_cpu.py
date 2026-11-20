"""Run one CPU cell on real tapes.

  python3 scripts/run_cpu.py framework db1 llm2                       # our framework, original design (reader mode "original")
  python3 scripts/run_cpu.py framework db3 llm2 --reader-mode salvaged_no_certainty
  python3 scripts/run_cpu.py framework db3 llm2 --reader-mode reader2_clean       # the framework final on the cleaned tape and the pooled sheaf (needs the build's fit cache and FREEZE_FRAMEWORK.json)

The framework can run in two parts. Part 1 fits only the dial maps and the sheaf and saves them (needs no game inputs, no 13F, no VXF):
  python3 scripts/run_cpu.py framework db3 llm2 --reader-mode salvaged_no_certainty --stop-after-dev     # -> results/framework-db3-llm2-sheaf/dev_fit.pkl
Part 2 loads that file instead of fitting again, then runs the personas, the merge, the optimiser and the backtests:
  python3 scripts/run_cpu.py framework db3 llm2 --reader-mode salvaged_no_certainty --load-dev PATH/dev_fit.pkl

Reads the finished bundles from the folders in ``cpu.tape_dirs`` (config/cpu.yaml), the market tables from the market dataset folder and
the unit table the job was run on, and writes ``results/<job>-<bucket>-<llm>[-<mode>]/``: tables (parquet and csv), diagnostics.json,
lineage.json. A reader mode other than ``off`` writes to its own folder, so runs of different modes never overwrite each other, and records
the mode, the configuration differences, the unit-table alignment report and the unit table's hash in lineage.json.
Numbers only; nothing here decides anything.
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from oryxis.common.config import load_config  # noqa: E402
from oryxis.real.framework_cell import run_framework_cell  # noqa: E402
from oryxis.real.game_inputs import make_game_inputs  # noqa: E402
from oryxis.real.results import write_json, write_lineage, write_tables  # noqa: E402
from oryxis.real.world import load_world, units_sha256  # noqa: E402
from oryxis.road1.fit import ShardPending  # noqa: E402

CELLS = ("framework",)
DEFAULT_MODE = "original"


from oryxis.real.devfit_io import DEV_FIT_IGNORED_SECTIONS, DEV_FILE, dev_fit_problems, read_dev_fit  # noqa: E402,F401


def run_framework(bucket: str, llm: str, cfg, mode_name: str = DEFAULT_MODE, stop_after_dev: bool = False, load_dev: Path | None = None, build_out: Path | None = None) -> Path:
    from oryxis.reader.salvage import mode_settings  # noqa: PLC0415

    if mode_settings(cfg, mode_name).get("clean_tape"):  # reader2_clean: the framework final on the cleaned tape and the pooled sheaf (oryxis/framework/final_run.py)
        from oryxis.framework.final_run import run_final  # noqa: PLC0415

        if stop_after_dev:
            raise SystemExit("--stop-after-dev belongs to the old modes; in reader2_clean the dev fit is made from the build's fit cache (outputs/final-build/runs/<run>)")
        run = f"{bucket}-{llm}"
        out = Path(build_out) if build_out else ROOT / "outputs" / "final-build" / "runs" / run
        return run_final(ROOT, run, cfg, out, lambda m: print(m, flush=True), workers=os.cpu_count() or 1, dev_fit_path=load_dev)
    world = load_world(ROOT, bucket, llm, cfg, mode_name, with_personas=not stop_after_dev)  # part 1 uses no persona answers
    run_cfg, mode, calendar, windows, rm = world.cfg, world.mode, world.calendar, world.windows, world.rm
    used, persona_used, units_path, alignment, fill_info = world.used, world.persona_used, world.units_path, world.alignment, world.fill_info
    skeletons = world.skeletons
    game = None if stop_after_dev else make_game_inputs(run_cfg, rm, skeletons, world.managers)  # part 1 needs no game inputs
    arm = str(run_cfg.get("framework.road1_arm_legacy"))  # the old modes keep the cause-split arm; framework.road1_arm = pooled_sheaf is read by the clean mode only
    meta = {"bucket": bucket, "llm": llm, "reader_mode": mode_name, "arm": arm, "config_hash": run_cfg.hash,
            "fit_config_hash": run_cfg.hash_without(DEV_FIT_IGNORED_SECTIONS)}
    dev_fit = None
    if load_dev is not None:
        dev_fit = read_dev_fit(Path(load_dev), meta)
        print(f"loaded the dev fit (dial maps and {arm}) from {load_dev}; the dev events are not fitted again", flush=True)
    suffix = "" if mode_name == DEFAULT_MODE else f"-{mode_name}"
    folder = ROOT / str(cfg.get("cpu.results_dir")) / f"framework-{bucket}-{llm}{suffix}{'-sheaf' if stop_after_dev else ''}"

    def save_dev_fit(dev, dials) -> None:  # called by the cell right after the dev fit, before the sheaf test score is computed
        folder.mkdir(parents=True, exist_ok=True)
        with (folder / DEV_FILE).open("wb") as handle:
            pickle.dump({"meta": meta, "dev": dev, "dials": dials}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"saved the dev fit to {folder / DEV_FILE}", flush=True)

    out = run_framework_cell(run_cfg, rm, world.factors, world.events, world.reader_tape, world.persona_tape, skeletons, game,
                             world.links, windows, calendar, arm,
                             workers=os.cpu_count() or 1, reader_mode=mode, dev_fit=dev_fit, stop_after_dev=stop_after_dev,
                             after_dev=save_dev_fit if stop_after_dev else None)
    tables = {"summary": out.summary, "skipped": out.skipped, **out.extra}
    tables.update({f"net_{key}": frame for key, frame in out.net.items()})
    tables.update({f"gross_{key}": result.gross.to_frame("gross") for key, result in out.backtests.items()})
    write_tables(folder, tables)
    write_json(folder, "diagnostics.json", {**out.diagnostics, "fits": out.fits})
    if stop_after_dev:
        if out.dev_fit is None or out.dev_fit[0].road1 is None:
            raise SystemExit(f"part 1 produced no sheaf fit: {out.fits.get('stopped')}")
        with (folder / DEV_FILE).open("wb") as handle:
            pickle.dump({"meta": meta, "dev": out.dev_fit[0], "dials": out.dev_fit[1]}, handle, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"saved the dev fit to {folder / DEV_FILE}", flush=True)
    write_lineage(folder, cfg, ROOT, used + persona_used, {"cell": "framework", "bucket": bucket, "llm": llm,
                                                           "test_start": str(windows.test_start.date()), "test_end": str(windows.test_end.date()),
                                                           "persona_tape": bool(persona_used), "reader_mode": mode, "fill": fill_info,
                                                           "unit_table": str(units_path), "unit_table_sha256": units_sha256(world),
                                                           "alignment": alignment, "deviations": mode if mode_name != DEFAULT_MODE else None,
                                                           "part": "1 (dev fit only)" if stop_after_dev else ("2 (dev fit loaded)" if load_dev else "whole"),
                                                           "dev_fit_file": str(load_dev) if load_dev else None})
    return folder


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("cell", choices=CELLS)
    parser.add_argument("bucket")
    parser.add_argument("llm")
    parser.add_argument("--reader-mode", default=DEFAULT_MODE, help="a mode of config/cpu.yaml reader_salvage.modes")
    parser.add_argument("--stop-after-dev", action="store_true", help="part 1: fit only the dial maps and the sheaf, save them to dev_fit.pkl and stop")
    parser.add_argument("--load-dev", type=Path, default=None, help="part 2: a dev_fit.pkl from part 1 (same bucket, llm, reader mode and configuration)")
    parser.add_argument("--build-out", type=Path, default=None, help="reader2_clean only: the build's run folder (fit cache); default outputs/final-build/runs/<bucket>-<llm>")
    args = parser.parse_args(argv)
    if args.stop_after_dev and args.load_dev:
        parser.error("--stop-after-dev and --load-dev cannot be used together")
    cfg = load_config(ROOT / "config")
    try:
        folder = {"framework": run_framework}[args.cell](args.bucket, args.llm, cfg, args.reader_mode, args.stop_after_dev, args.load_dev, args.build_out)
    except ShardPending as pending:  # shard mode (ORYXIS_FIT_SHARD): this shard's cross-validation fits are stored; the assemble run finishes the stage
        print(f"SHARD RUN FINISHED: the fits of '{pending}' owned by this shard are stored in the checkpoint; no result was written", flush=True)
        return 0
    print(f"wrote {folder}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
