"""CPU experiments of one bucket (db1 by default, llm2) inside a Kaggle CPU notebook. No GPU, no model call, no reader or persona rerun.

  python kaggle_db1_runner.py <stage> [--bucket db1|db3] [--no-sheaf] [--input-root DIR] [--work DIR] [--modes a,b,c] [--fetch-vxf] [--fetch-f13]

Stages: ``validate`` (checks every input, writes ``validation_report.json``, runs nothing heavy) and ``framework`` (validate, then the framework
cell under each reader mode, then the with/without-defaulted-cause comparison, then one zip of everything). The framework needs three inputs that
are not in the pack: the VXF sleeve prices and ``f13_managers.parquet``; ``--fetch-vxf`` / ``--fetch-f13`` fetch them with the
repository's own scripts (internet on; the 13F build needs the Kaggle secret SEC_CONTACT and downloads about 2.5 GB). Without them the framework
stage stops after validation and says what is missing. Original files are only read; everything is written under the work folder.
"""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path

START = time.time()
MARKER = "PACK_MANIFEST.json"
MAX_START_SECONDS = 8 * 3600  # no new mode starts after this (a Kaggle session ends at 12 h and loses its output)
BUCKET, LLM = "db1", "llm2"  # BUCKET is set from --bucket in main()
READER2_MAX_MISSING = 100  # events a reader2 run may leave without a row (tessera-llm2 stopped with 55 open by the owner's decision)
SHARED_HALF_BUCKETS = 50  # owner decision 2026-11-04: tessera on llm1 and llm3 reads only the shared half of the events (hash bucket below 50)
PARTIAL_BY_DESIGN = {("tessera", "llm1"), ("tessera", "llm3")}
ROAD1_MODES = ["salvaged_no_certainty", "salvaged_no_certainty_no_defaulted", "original"]
DEFAULT_MODES = ["salvaged_no_certainty", "salvaged_no_certainty_no_defaulted", "original", "original_no_defaulted"]
# Counts the salvage must reproduce, from results/reader-salvage-<bucket>-llm2/validation.json. db1 holds the first 9,312-unit tape only: update it from a new
# salvage run once the full db1 tape is in.
EXPECTED_SALVAGE_BY_BUCKET = {"db1": {"natural": 11617, "salvaged": 12453, "unrepaired": 24, "not_called": 5906},
                              "db3": {"natural": 7260, "salvaged": 18113, "unrepaired": 3, "not_called": 4624}}  # re-run 2026-11-03 on the merged tape (reader-db3.bundle); the earlier stored report says 18110 / 6
MARKET_LINKED = ("coverage.parquet", "sp500_changes.parquet", "sp500_snapshot.parquet", "survivorship_audit.parquet")
MARKET_COPIED = ("px_daily.parquet", "corp_action.parquet")  # fetch_sleeve rewrites these two
LIMITATIONS = """# Limitations (read before using any number)
- Reader: answers that parsed on their own, answers repaired by CPU salvage (tail completed or cut back to their last complete pin) and answers that stay
  unparsed are counted in validation_report.json (`salvage_crosscheck`); units with no usable pin were never sent to the model.
- `flag_cause_defaulted`: for these readings the cause is an assumption (`other = 1.0`), not a model statement. The `*_no_defaulted` modes drop them,
  so those runs use fewer events; the comparison with the full runs mixes "defaulted readings removed" with "fewer events".
- Salvaged readings have no `certainty` and mostly no `scheduled`; nothing is filled. Salvaged rows are not a random draw (more geo and
  social, other cause mix, more `down` pins); pins after a repetition loop are lost.
- Personas: events without persona answers are forecast without them (see `framework_events.with_persona_answers` in validation_report.json).
- All reader and persona answers are the existing ones; nothing was rerun.
"""


def log(message: str) -> None:
    print(f"[{time.time() - START:7.0f}s] {message}", flush=True)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def pick_dev_fit(saved: list[str], bucket: str, llm: str, mode: str) -> list[str]:
    """The dev_fit.pkl of this bucket, LLM and reader mode. Copies with identical bytes (a dataset that holds the same output twice, for instance a zip and its
    unpacked folder) count as one; different files are never merged, so more than one entry back means the attachment is ambiguous."""
    mine = [p for p in saved if f"framework-{bucket}-{llm}-{mode}-sheaf" in p] or (saved if len(saved) == 1 else [])
    unique: dict[str, str] = {}
    for path in mine:
        unique.setdefault(sha256(Path(path)), path)
    return list(unique.values())


def find_pack(root: str) -> Path:
    hits = glob.glob(f"{root}/**/{MARKER}", recursive=True)
    if not hits:
        raise SystemExit("INPUT MISSING: in the notebook choose Add Input and attach the pack dataset (no PACK_MANIFEST.json found under /kaggle/input).")
    return Path(hits[0]).parent


def link(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.symlink(source, target, target_is_directory=source.is_dir())
    except OSError:
        shutil.copytree(source, target) if source.is_dir() else shutil.copyfile(source, target)


def build_tree(pack: Path, work: Path) -> Path:
    root = work / "oryxis-exp"
    if root.exists():
        shutil.rmtree(root)
    shutil.copytree(pack / "code", root)
    market = root / "dataset" / "oryxis-market"
    market.mkdir(parents=True)
    for name in MARKET_COPIED:
        shutil.copyfile(pack / "market" / name, market / name)
    for name in MARKET_LINKED:
        link(pack / "market" / name, market / name)
    upload = pack / "oryxis-upload"
    for name in ("links.parquet", "aliases.parquet"):
        link(upload / name, market / name)
    link(upload / "prestaged", market / "prestaged")
    link(upload, root / "dataset" / "oryxis-upload")
    shutil.copyfile(pack / "french" / "ff_factors_daily.parquet", market / "factors_daily.parquet")  # same table; run_cpu expects this name
    (root / "store" / "raw").mkdir(parents=True)
    link(pack / "french", root / "store" / "raw" / "french")
    link(pack / "tapes", root / "tapes")
    if (pack / "market" / "f13_managers.parquet").exists():  # present when the pack was built with --with-f13
        link(pack / "market" / "f13_managers.parquet", market / "f13_managers.parquet")
    results = work / "results"
    results.mkdir(exist_ok=True)
    link(results, root / "results")
    return root


def run(command: list[str], cwd: Path, log_file: Path) -> int:
    log("$ " + " ".join(command))
    with log_file.open("w", encoding="utf-8") as handle:
        code = subprocess.run(command, cwd=cwd, stdout=handle, stderr=subprocess.STDOUT, check=False).returncode
    log(f"exit {code}; last lines: {log_file.read_text(encoding='utf-8', errors='replace').splitlines()[-3:]}")
    return code


def ensure(package: str) -> None:
    try:
        __import__(package)
    except ImportError:
        log(f"pip install {package}")
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", package], check=True)


def validate(pack: Path, root: Path, report: dict) -> bool:
    import pandas as pd

    manifest = json.loads((pack / MARKER).read_text(encoding="utf-8"))
    bad = [n for n, m in manifest["files"].items() if not (pack / n).exists() or sha256(pack / n) != m["sha256"]]
    report["pack"] = {"files": len(manifest["files"]), "git_commit": manifest["git_commit"], "working_tree_clean": manifest["working_tree_clean"],
                      "bucket": manifest["bucket"], "llm": manifest["llm"], "files_not_matching_manifest": bad[:10]}
    ok = not bad and manifest["bucket"] == BUCKET and LLM in manifest.get("llms", [manifest["llm"]])  # a reader2 pack carries every LLM of its bucket
    sys.path.insert(0, str(root))
    os.chdir(root)
    from oryxis.common.config import load_config
    from oryxis.reader.salvage import config_for_mode, prepare_reader_tape, verify_alignment
    from oryxis.real.tapes import TapeError, is_reader_v2, load_reader_tape, load_tape

    cfg = load_config(root / "config")
    reader, used = load_reader_tape(cfg, root, BUCKET, LLM)
    v2 = is_reader_v2(reader)
    try:
        personas, _ = load_tape(cfg, root, "personas", BUCKET, LLM)
    except TapeError:
        if not v2:
            raise
        personas = pd.DataFrame()  # a reader2 pack carries no persona answers: the sheaf stage (part 1) uses none
    units = pd.read_parquet(root / "dataset" / "oryxis-market" / "prestaged" / f"reader-{BUCKET}" / "units.parquet")
    pins_json = dict(zip(units["unit_id"], units["pins_json"], strict=True))
    alignment = verify_alignment(reader, pins_json, cfg)
    events = pd.read_parquet(root / "dataset" / "oryxis-market" / "prestaged" / f"personas-{BUCKET}" / "events.parquet")
    report["tapes"] = {"reader": "reader2" if v2 else "reader", "reader_rows": len(reader), "reader_bundles": [p.name for p in used], "persona_rows": len(personas),
                       "persona_events_with_answers": int(personas["event_id"].nunique()) if "event_id" in personas else None}
    report["alignment"] = {k: v for k, v in alignment.items() if k != "examples"}
    in_tape = events["event_id"].isin(reader["unit_id"])
    report["framework_events"] = {"events": len(events), "in_reader_tape": int(in_tape.sum()),
                                  "with_persona_answers": int(events["event_id"].isin(personas["event_id"]).sum()) if "event_id" in personas else None}
    if (BUCKET, LLM) in PARTIAL_BY_DESIGN:  # a half-sample run: every event of the shared half must be on the tape and no other event may be
        from oryxis.data.buckets.events import hash_bucket

        expected = events["event_id"].map(hash_bucket) < SHARED_HALF_BUCKETS
        report["framework_events"].update({"partial_by_design": True, "expected_shared_half": int(expected.sum()),
                                           "expected_on_tape": int((expected & in_tape).sum()), "outside_shared_half_on_tape": int((~expected & in_tape).sum())})
        events_ok = bool((expected & in_tape).sum() == expected.sum() and (~expected & in_tape).sum() == 0)
        if v2:  # the shared half read in full is what was asked; units outside it are scoped out, not missing
            events_ok = bool((~expected & in_tape).sum() == 0 and (expected & ~in_tape).sum() <= READER2_MAX_MISSING)
    elif v2:  # reader2 reads every event but may leave a handful open (an owner decision to stop a run short): reported, never hidden
        missing = int((~in_tape).sum())
        report["framework_events"]["events_without_a_reader_row"] = missing
        events_ok = missing <= READER2_MAX_MISSING
    else:
        events_ok = report["framework_events"]["events"] == report["framework_events"]["in_reader_tape"]
    ok = ok and bool(alignment["ok"]) and events_ok
    if v2:  # the final reader needs no salvage and no cross-check against recorded salvage counts
        report["reader_modes"] = {"reader2": "stored columns used as they are"}
        report["salvage_crosscheck"] = {"matches": True, "skipped": "reader2 tape"}
        return validate_inputs(report, ok, cfg, root, pd)
    modes = {}
    for mode in DEFAULT_MODES:
        run_cfg, rec = config_for_mode(cfg, mode)
        tape, rep, _ = prepare_reader_tape(reader, pins_json, run_cfg, rec)
        usable = int(tape["parse_ok"].astype(bool).sum() + (tape.get("salvage_status", pd.Series(dtype=object)) == "salvaged").sum() * int(rec["include_salvaged"]))
        modes[mode] = {"excluded_defaulted_cause": rep.get("excluded_defaulted_cause", 0), "usable_readings_before_pin_filter": usable,
                       "salvage_status": {str(k): int(v) for k, v in tape.get("salvage_status", pd.Series(dtype=object)).value_counts().items()}}
    report["reader_modes"] = modes
    got = modes["salvaged_no_certainty"]["salvage_status"]
    expected_salvage = EXPECTED_SALVAGE_BY_BUCKET.get(f"{BUCKET}-{LLM}") or (EXPECTED_SALVAGE_BY_BUCKET.get(BUCKET) if LLM == "llm2" else None)
    if expected_salvage is None:  # a bucket / LLM whose salvage counts have not been recorded yet: they are reported, not compared
        expected_salvage = {k: got.get(k) for k in ("natural", "salvaged", "unrepaired", "not_called")}
        log(f"no recorded salvage counts for {BUCKET}/{LLM}; recomputed here (reported, not compared): {expected_salvage}")
    report["salvage_crosscheck"] = {"expected": expected_salvage, "recomputed_here": {k: got.get(k) for k in expected_salvage},
                                    "matches": all(got.get(k) == v for k, v in expected_salvage.items())}
    ok = ok and report["salvage_crosscheck"]["matches"]
    return validate_inputs(report, ok, cfg, root, pd)


def validate_inputs(report: dict, ok: bool, cfg, root: Path, pd) -> bool:
    market = root / "dataset" / "oryxis-market"
    px = pd.read_parquet(market / "px_daily.parquet", columns=["security_id"])
    needs = {"f13_managers.parquet": (market / "f13_managers.parquet").exists(),
             "factors_daily.parquet": (market / "factors_daily.parquet").exists(), "links.parquet": (market / "links.parquet").exists(),
             "VXF prices in px_daily": bool((px["security_id"] == str(cfg.get("game.sleeve_proxy"))).any()),
             "persona skeletons": (market / "prestaged" / f"personas-{BUCKET}" / "skeletons.json").exists()}
    report["framework_inputs_present"] = needs
    report["validation_passed"] = bool(ok)
    report["framework_ready"] = bool(ok and all(needs.values()))
    return report["framework_ready"]


def main(argv=None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["validate", "framework", "road1", "sheaf", "optimise", "ablation"])
    parser.add_argument("--no-sheaf", action="store_true", help="road1 only: fit and score just the weighted graph and the weighted matrices")
    parser.add_argument("--bucket", default="db1", choices=["db1", "tessera", "db3"])
    parser.add_argument("--llm", default="llm2", choices=["llm1", "llm2", "llm3"])
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--work", default="/kaggle/working")
    parser.add_argument("--modes", default="")
    parser.add_argument("--fetch-vxf", action="store_true")
    parser.add_argument("--fetch-f13", action="store_true")
    parser.add_argument("--notebook", default="", help="ablation stage: the notebook name of config ablation.notebooks")
    args = parser.parse_args(argv)
    global BUCKET, LLM
    BUCKET, LLM = args.bucket, args.llm
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    pack = find_pack(args.input_root)
    root = build_tree(pack, work)
    ensure("exchange_calendars")
    report: dict = {"stage": args.stage}
    codes: dict[str, int] = {}
    if args.fetch_vxf:
        ensure("yfinance")
        codes["fetch_vxf"] = run([sys.executable, "scripts/fetch_sleeve.py"], root, work / "fetch_vxf.log")
    if args.fetch_f13:
        if not os.environ.get("SEC_CONTACT"):
            raise SystemExit("--fetch-f13 needs the Kaggle secret SEC_CONTACT (Add-ons > Secrets); see the notebook instructions")
        codes["fetch_f13"] = run([sys.executable, "scripts/build_f13.py"], root, work / "fetch_f13.log")
    ready = validate(pack, root, report)
    (work / "validation_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    (work / "LIMITATIONS.md").write_text(LIMITATIONS, encoding="utf-8")
    log(f"validation passed: {report['validation_passed']}; framework inputs: {report['framework_inputs_present']}")
    if args.stage == "road1":  # experiment E: weighted graph vs sheaf, matrix rungs, oracle, stratified test; needs no 13F and no VXF
        if not report["validation_passed"]:
            log("STOP: validation failed (see validation_report.json); no experiment was started")
        else:
            for mode in [m for m in (args.modes or ",".join(ROAD1_MODES)).split(",") if m]:
                if time.time() - START > MAX_START_SECONDS:
                    log(f"SKIPPED {mode}: more than {MAX_START_SECONDS // 3600} h used already")
                    continue
                out = work / "results" / f"road1-{BUCKET}-{LLM}-{mode}{'-nosheaf' if args.no_sheaf else ''}"
                codes[f"road1_{mode}"] = run([sys.executable, "scripts/run_road1.py", BUCKET, LLM, str(pack / "tapes" / f"reader-{BUCKET}-{LLM}"), "--reader-mode", mode, "--out", str(out), *(["--no-sheaf"] if args.no_sheaf else [])], root, work / f"road1_{mode}.log")
                if codes[f"road1_{mode}"] == 0 and not os.environ.get("ORYXIS_FIT_SHARD"):  # a shard writes no tables: there is nothing to validate
                    codes[f"validate_{mode}"] = run([sys.executable, "scripts/validate_road1_outputs.py", str(out), "--expect-oracle"], root, work / f"validate_{mode}.log")
    if args.stage == "sheaf":  # part 1 of the framework: only the dial maps and the sheaf; needs no 13F, no VXF, no game inputs
        if not report["validation_passed"]:
            log("STOP: validation failed (see validation_report.json); no experiment was started")
        else:
            for mode in [m for m in (args.modes or ",".join(DEFAULT_MODES)).split(",") if m]:
                if time.time() - START > MAX_START_SECONDS:
                    log(f"SKIPPED {mode}: more than {MAX_START_SECONDS // 3600} h used already")
                    continue
                codes[f"sheaf_{mode}"] = run([sys.executable, "scripts/run_cpu.py", "framework", BUCKET, LLM, "--reader-mode", mode, "--stop-after-dev"], root, work / f"sheaf_{mode}.log")
    if args.stage == "optimise":  # part 2 of the framework: loads part 1's dev_fit.pkl (attach part 1's output as a dataset), then personas, merge, optimiser, backtests
        if not report["validation_passed"]:
            log("STOP: validation failed (see validation_report.json); no experiment was started")
        elif not ready:
            missing = [k for k, v in report["framework_inputs_present"].items() if not v]
            log(f"STOP: framework inputs missing: {missing}. Nothing was started. Fetch them (notebook switch FETCH_VXF) or add them to the pack.")
        else:
            saved = sorted(glob.glob(f"{args.input_root}/**/dev_fit.pkl", recursive=True))
            log(f"dev_fit.pkl files found under {args.input_root}: {saved}")
            for mode in [m for m in (args.modes or ",".join(DEFAULT_MODES)).split(",") if m]:
                if time.time() - START > MAX_START_SECONDS:
                    log(f"SKIPPED {mode}: more than {MAX_START_SECONDS // 3600} h used already")
                    continue
                mine = pick_dev_fit(saved, BUCKET, LLM, mode)
                if len(mine) != 1:
                    log(f"STOP for {mode}: expected exactly one dev_fit.pkl for this mode, found {mine}; attach the output of part 1 (the 'sheaf' stage) as a dataset")
                    codes[f"optimise_{mode}"] = 2
                    continue
                codes[f"optimise_{mode}"] = run([sys.executable, "scripts/run_cpu.py", "framework", BUCKET, LLM, "--reader-mode", mode, "--load-dev", mine[0]], root, work / f"optimise_{mode}.log")
    if args.stage == "ablation":  # the runs of one ablation notebook: loads the sheaf stage's dev_fit.pkl, then each run in turn (finished runs are skipped)
        if not report["validation_passed"]:
            log("STOP: validation failed (see validation_report.json); no experiment was started")
        elif not ready:
            missing = [k for k, v in report["framework_inputs_present"].items() if not v]
            log(f"STOP: framework inputs missing: {missing}. Nothing was started. Fetch them (notebook switch FETCH_VXF) or add them to the pack.")
        else:
            saved = sorted(glob.glob(f"{args.input_root}/**/dev_fit.pkl", recursive=True))
            log(f"dev_fit.pkl files found under {args.input_root}: {saved}")
            mode = ([m for m in args.modes.split(",") if m] or ["salvaged_no_certainty"])[0]
            mine = pick_dev_fit(saved, BUCKET, LLM, mode)
            if len(mine) != 1:
                log(f"STOP: expected exactly one dev_fit.pkl for {BUCKET}/{LLM}/{mode}, found {mine}; attach the output of the sheaf notebook as a dataset")
                codes["ablation"] = 2
            else:
                codes["ablation"] = run([sys.executable, "scripts/run_ablation.py", BUCKET, LLM, "--reader-mode", mode, "--notebook", args.notebook, "--dev-fit", mine[0]], root, work / f"ablation_{args.notebook}.log")
    if args.stage == "framework":
        if not report["validation_passed"]:
            log("STOP: validation failed (see validation_report.json); no experiment was started")
        elif not ready:
            missing = [k for k, v in report["framework_inputs_present"].items() if not v]
            log(f"STOP: framework inputs missing: {missing}. Nothing was started. Fetch them (notebook switches FETCH_VXF / FETCH_F13) or add them to the pack.")
        else:
            for mode in [m for m in (args.modes or ",".join(DEFAULT_MODES)).split(",") if m]:
                if time.time() - START > MAX_START_SECONDS:
                    log(f"SKIPPED {mode}: more than {MAX_START_SECONDS // 3600} h used already, so the session can still finish and save what is done")
                    continue
                codes[f"framework_{mode}"] = run([sys.executable, "scripts/run_cpu.py", "framework", BUCKET, LLM, "--reader-mode", mode], root, work / f"framework_{mode}.log")
            if len([m for m in (args.modes or ",".join(DEFAULT_MODES)).split(",") if m]) > 1:
                codes["compare"] = run([sys.executable, "scripts/compare_framework_modes.py", str(work / "results"), BUCKET, LLM, str(work / "comparison")], root, work / "compare.log")
    report.update({"exit_codes": codes, "seconds": round(time.time() - START, 1)})
    (work / "validation_report.json").write_text(json.dumps(report, indent=1, default=str), encoding="utf-8")
    outputs = work / f"{BUCKET}_cpu_outputs.zip"
    with zipfile.ZipFile(outputs, "w", zipfile.ZIP_DEFLATED) as z:
        for path in [work / "validation_report.json", work / "LIMITATIONS.md", *work.glob("*.log"), *(work / "results").rglob("*"), *(work / "comparison").rglob("*")]:
            if path.is_file():
                z.write(path, path.relative_to(work))
    log(f"wrote {outputs}")
    return 0 if all(c == 0 for c in codes.values()) and report["validation_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
