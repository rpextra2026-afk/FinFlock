"""Supervisor of one CPU notebook run on Kaggle (also runs on a laptop).

  python cpu_session.py --label db3-llm2-sheaf --stage sheaf --bucket db3 --llm llm2 --modes salvaged_no_certainty --input-root /kaggle/input --work /kaggle/working

It starts scripts/kaggle_db1_runner.py as a child and does everything around it:
  * names every bundle it opens (the data bundles in the pack and any checkpoint bundle attached from an earlier session) and says whether it resumes;
  * every ``--tick-seconds`` (10) prints one METRICS line (elapsed, time left, CPU, memory, work done, last result) and appends it to metrics_stream.csv,
    and shows the new lines of the child's step logs, so every result the code prints reaches the notebook output;
  * stops the child at ``--cutoff-seconds`` (11.5 h in the notebooks) sharp, writes ``<label>.ckpt.bundle`` and says how to continue: download it, upload it as a dataset,
    attach it to the same notebook, Run all - the finished pieces of work are then read from the bundle and not computed again;
  * at the end prints the final results table (metrics as rows, values as columns) and saves it as final_results.md / final_results.csv.
Exit code of the supervisor: 0 finished, 3 stopped by the cutoff (checkpoint written), 1 the child failed (checkpoint written).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import zipfile
from pathlib import Path

import pandas as pd

CKPT_SUFFIX = ".ckpt.bundle"
EXIT_FINISHED, EXIT_FAILED, EXIT_CUTOFF = 0, 1, 3
KILL_WAIT_SECONDS = 60
MAX_LOG_LINES_PER_TICK = 200
METRIC_COLUMNS = ["time_utc", "elapsed_s", "left_s", "cpu_pct", "mem_gb", "ckpt_pieces", "fits_done", "fits_total", "last_val_loss", "best_cv_loss", "log_lines", "last_line"]

START = time.time()
CUTOFF = 41400.0
SESSION_LOG: Path | None = None
SESSION_LOG_NAME = "session.log"
PROCESSES: dict = {}  # psutil objects kept between ticks: a CPU percentage needs the previous reading of the same process


def stamp() -> str:
    elapsed = time.time() - START
    left = max(CUTOFF - elapsed, 0)
    return f"[{int(elapsed // 3600):02d}:{int(elapsed % 3600 // 60):02d}:{int(elapsed % 60):02d} | {int(left // 3600):02d}:{int(left % 3600 // 60):02d}:{int(left % 60):02d} left]"


def log(message: str) -> None:
    line = f"{stamp()} {message}"
    print(line, flush=True)
    if SESSION_LOG is not None:
        with SESSION_LOG.open("a", encoding="utf-8") as handle:
            handle.write(line + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---- bundles ------------------------------------------------------------------------------------------------------------------------------

def describe_data_bundles(pack: Path) -> None:
    """Names every data bundle (the reader and persona tapes) in the pack that the run is going to open."""
    found = sorted((pack / "tapes").rglob("*.bundle")) if (pack / "tapes").exists() else []
    if not found:
        log("DATA BUNDLES: none found under the pack's tapes/ folder")
    for path in found:
        try:
            manifest = json.loads(zipfile.ZipFile(path).read("MANIFEST.json"))
            rows = sum(int(part.get("rows", 0)) for part in manifest.get("parts", []))
            log(f"DATA BUNDLE opened: {path.relative_to(pack)}  job={manifest.get('job_id')}  shard={manifest.get('shard_id')}  seq={manifest.get('seq')}  "
                f"rows={rows}  size={path.stat().st_size / 1e6:.1f} MB  sha256={sha256_file(path)[:16]}")
        except Exception as exc:  # noqa: BLE001
            log(f"DATA BUNDLE {path.name}: cannot read its manifest ({exc})")


def shard_label_pattern(label: str, shards: int) -> re.Pattern:
    """Labels of the shard runs that feed the assemble run ``label``: ``<label>-s<k>of<N>``."""
    return re.compile(rf"^{re.escape(label)}-s(\d+)of{shards}$")


def find_checkpoints(input_root: Path, label: str, merge_shards: int = 0) -> list[tuple[Path, dict | None, str]]:
    """Every checkpoint bundle under the input root with the verdict on it: (path, manifest or None, 'ok' or the reason it is not used).
    With ``merge_shards`` N the bundles of the N shard runs ``<label>-s<k>of<N>`` count as well as the run's own."""
    out = []
    family = shard_label_pattern(label, merge_shards) if merge_shards else None
    for path in sorted(input_root.rglob(f"*{CKPT_SUFFIX}")) if input_root.exists() else []:
        try:
            archive = zipfile.ZipFile(path)
            manifest = json.loads(archive.read("MANIFEST.json"))
            bad = [name for name, digest in manifest["files"].items() if hashlib.sha256(archive.read(name)).hexdigest() != digest]
            if bad:
                out.append((path, manifest, f"REJECTED: {len(bad)} file(s) do not match their hashes (first: {bad[:2]})"))
            elif manifest.get("label") != label and not (family is not None and family.match(str(manifest.get("label")))):
                out.append((path, manifest, f"REJECTED: it belongs to run '{manifest.get('label')}', this run is '{label}'"))
            else:
                out.append((path, manifest, "ok"))
        except Exception as exc:  # noqa: BLE001
            out.append((path, None, f"REJECTED: unreadable ({type(exc).__name__}: {str(exc)[:80]})"))
    return out


def unpack_all(valid: list[tuple[Path, dict]], label: str, ckpt_dir: Path, merge_shards: int) -> dict:
    """Assemble run: every valid bundle of the run and of its shards is unpacked into one folder. A piece is named by the hash of its inputs, so the same
    name is the same piece; two bundles holding one name with different bytes are an error, never a silent overwrite."""
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    pieces: dict[str, int] = {}
    seen: dict[str, tuple[str, str]] = {}
    shard_numbers = set()
    for path, manifest in valid:
        family = shard_label_pattern(label, merge_shards).match(str(manifest.get("label")))
        if family:
            shard_numbers.add(int(family.group(1)))
        with zipfile.ZipFile(path) as archive:
            for name, digest in manifest["files"].items():
                if not name.startswith("ckpt/"):
                    continue
                target = Path(name).name
                if target in seen and seen[target][0] != digest:
                    raise SystemExit(f"CHECKPOINT CONFLICT: piece {target} differs between {path.name} and {seen[target][1]}; refusing to merge")
                if target not in seen:
                    (ckpt_dir / target).write_bytes(archive.read(name))
                    seen[target] = (digest, path.name)
                    kind = target.split("-", 1)[0]
                    pieces[kind] = pieces.get(kind, 0) + 1
        log(f"MERGED bundle {path.name}: label={manifest.get('label')} pieces={manifest.get('pieces')}")
    missing = sorted(set(range(merge_shards)) - shard_numbers)
    if missing:
        log(f"WARNING: shard bundle(s) {missing} of {merge_shards} are not attached; the fits they own are computed again in THIS session (slow). "
            f"Stop and attach them if that is not what you want.")
    else:
        log(f"all {merge_shards} shard bundles are attached")
    return {"session": 0, "pieces": pieces, "label": label}


def open_checkpoint(input_root: Path, label: str, ckpt_dir: Path, merge_shards: int = 0) -> dict | None:
    """Logs every checkpoint bundle found, unpacks the best valid one into ``ckpt_dir`` (all of them, in an assemble run) and says plainly whether this run resumes."""
    log(f"CHECKPOINT SEARCH: looking for *{CKPT_SUFFIX} under {input_root} for run '{label}'" + (f" and its {merge_shards} shard runs" if merge_shards else ""))
    found = find_checkpoints(input_root, label, merge_shards)
    if not found:
        log("CHECKPOINT: none attached -> STARTING FRESH (nothing to resume from)")
        return None
    for path, manifest, verdict in found:
        extra = f"  stage={manifest.get('stage')} session={manifest.get('session')} created={manifest.get('created_utc')} pieces={manifest.get('pieces')}" if manifest else ""
        log(f"CHECKPOINT found: {path}  ({path.stat().st_size / 1e6:.1f} MB){extra}  -> {verdict}")
    valid = [(p, m) for p, m, v in found if v == "ok"]
    if not valid:
        log("CHECKPOINT: none of the attached bundles can be used -> STARTING FRESH")
        return None
    if merge_shards:
        merged = unpack_all(valid, label, ckpt_dir, merge_shards)
        log(f"ASSEMBLING from {len(valid)} bundle(s): {sum(merged['pieces'].values())} finished piece(s) of work unpacked {merged['pieces']}; the run skips them")
        return merged
    path, manifest = max(valid, key=lambda item: (sum(item[1].get("pieces", {}).values()), item[1].get("created_utc", "")))
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(path) as archive:
        for name in manifest["files"]:
            if name.startswith("ckpt/"):
                (ckpt_dir / Path(name).name).write_bytes(archive.read(name))
    log(f"RESUMING from {path.name}: {sum(manifest.get('pieces', {}).values())} finished piece(s) of work unpacked {manifest.get('pieces')}; "
        f"the run skips them and continues where session {manifest.get('session')} stopped")
    return manifest


def write_checkpoint(work: Path, ckpt_dir: Path, label: str, stage: str, session: int, reason: str) -> Path:
    """One verified bundle with the finished pieces of work and this session's logs; written atomically."""
    files, pieces = {}, {}
    target = work / f"{label}{CKPT_SUFFIX}"
    temporary = target.with_suffix(".tmp")
    with zipfile.ZipFile(temporary, "w", zipfile.ZIP_STORED) as archive:
        for path in sorted(ckpt_dir.glob("*.pkl")) if ckpt_dir.exists() else []:
            data = path.read_bytes()
            archive.writestr(f"ckpt/{path.name}", data)
            files[f"ckpt/{path.name}"] = hashlib.sha256(data).hexdigest()
            kind = path.name.split("-", 1)[0]
            pieces[kind] = pieces.get(kind, 0) + 1
        for name in ("session.log", "metrics_stream.csv"):
            if (work / name).exists():
                data = (work / name).read_bytes()
                archive.writestr(f"logs/{name}", data)
                files[f"logs/{name}"] = hashlib.sha256(data).hexdigest()
        manifest = {"label": label, "stage": stage, "session": session, "reason": reason, "pieces": pieces, "files": files,
                    "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "elapsed_s": round(time.time() - START)}
        archive.writestr("MANIFEST.json", json.dumps(manifest, indent=1))
    os.replace(temporary, target)
    return target


# ---- live metrics ---------------------------------------------------------------------------------------------------------------------------

def machine_numbers(pid: int | None) -> tuple[float, float]:
    """(cpu percent, resident memory GB) of the child's process tree, or of the machine when psutil is not there."""
    try:
        import psutil

        alive = []
        if pid is not None:
            root = psutil.Process(pid)
            alive = [root, *root.children(recursive=True)]
        procs = [PROCESSES.setdefault(p.pid, p) for p in alive]
        cpu = sum(p.cpu_percent(interval=None) for p in procs)
        mem = sum(p.memory_info().rss for p in procs) / 1e9
        return cpu, mem
    except Exception:  # noqa: BLE001
        return (os.getloadavg()[0] * 100.0 if hasattr(os, "getloadavg") else float("nan")), float("nan")


class Follower:
    """Prints the new lines of every step log of the child and keeps the numbers the metrics line shows."""

    def __init__(self, work: Path) -> None:
        self.work, self.offsets, self.lines = work, {}, 0
        self.fits_done = self.fits_total = 0
        self.last_val = self.best_cv = float("nan")
        self.last_line = ""
        self.stage, self.stages_done, self.last_result = "", [], ""

    def see(self, text: str) -> None:
        self.lines += 1
        self.last_line = text.strip()[:90]
        started = re.search(r"STAGE ([\w ()+,-]+?) (start|done|skipped)", text)
        if started:
            if started.group(2) == "start":
                self.stage = started.group(1)
            else:
                self.stages_done.append(started.group(1))
        result = re.search(r"RESULT (.*)", text)
        if result:
            self.last_result = result.group(1)[:300]
        done = re.search(r"\| (\d+)/(\d+) fits done", text)
        if done:
            self.fits_done, self.fits_total = int(done.group(1)), int(done.group(2))
        total = re.search(r"= (\d+) fits; (\d+) already stored", text)
        if total:
            self.fits_total, self.fits_done = int(total.group(1)), int(total.group(2))
        val = re.search(r"val_loss=([0-9.eE+-]+)", text)
        if val:
            self.last_val = float(val.group(1))
        cv = re.search(r"cv_loss=([0-9.eE+-]+)", text)
        if cv:
            value = float(cv.group(1))
            self.best_cv = value if self.best_cv != self.best_cv else min(self.best_cv, value)

    def flush(self) -> None:
        for path in sorted(self.work.glob("*.log")):
            if path.name == SESSION_LOG_NAME:  # the supervisor's own log: tailing it would echo every line again and again
                continue
            try:
                with path.open("r", encoding="utf-8", errors="replace") as handle:
                    handle.seek(self.offsets.get(path.name, 0))
                    new = handle.read()
                    self.offsets[path.name] = handle.tell()
            except OSError:
                continue
            lines = [line for line in new.splitlines() if line.strip()]
            for line in lines[:MAX_LOG_LINES_PER_TICK]:
                self.see(line)
                log(f"| {path.name}: {line[:300]}")
            if len(lines) > MAX_LOG_LINES_PER_TICK:
                log(f"| {path.name}: ... {len(lines) - MAX_LOG_LINES_PER_TICK} more line(s) in the file")


NEXT_STAGE = {"clean": "graphs", "graphs": "fit", "fit": "score", "score": "metrics", "metrics": "write", "write": "ablations", "ablations": "framework stage (load, dev fit, freeze check, forecasts, backtests, metrics)",
              "load": "dev fit", "dev_fit": "freeze check", "combine": "forecasts", "forecast": "backtests", "backtest": "metrics", "metrics ": "report"}


def status_block(args: argparse.Namespace, follower: "Follower", elapsed: float, mem: float, session: int) -> str:
    """The block printed every ``--status-seconds`` (10 minutes): time left, stage and shard in progress, last result with its interval, memory, next steps."""
    left = max(CUTOFF - elapsed, 0.0)
    shard = getattr(args, "shard", None) or (f"assemble of {args.merge_shards} shards" if getattr(args, "merge_shards", 0) else "none (one notebook)")
    nxt = NEXT_STAGE.get(follower.stage, "see the stage list printed at the start")
    fits = f"{follower.fits_done}/{follower.fits_total} cross-validation fits" if follower.fits_total else "no fit counter yet"
    return "\n".join([
        "=" * 100,
        f"STATUS {time.strftime('%H:%M:%S', time.gmtime())} UTC  session {session}  run {args.label}",
        f"  time used {elapsed / 3600:.2f} h   time left before the cutoff {left / 3600:.2f} h ({left:.0f} s)",
        f"  stage in progress: {follower.stage or '(none yet)'}   shard: {shard}   progress: {fits}",
        f"  stages finished: {', '.join(follower.stages_done) or '(none)'}",
        f"  last result: {follower.last_result or '(none yet)'}",
        f"  memory of the child: {mem:.2f} GB   last log line: {follower.last_line}",
        f"  next: {nxt}",
        "=" * 100])


def append_metrics(work: Path, row: dict) -> None:
    path = work / "metrics_stream.csv"
    new = not path.exists()
    with path.open("a", encoding="utf-8") as handle:
        if new:
            handle.write(",".join(METRIC_COLUMNS) + "\n")
        handle.write(",".join(f'"{row[c]}"' if c == "last_line" else str(row[c]) for c in METRIC_COLUMNS) + "\n")


# ---- final results table ----------------------------------------------------------------------------------------------------------------------

def main_metrics() -> list[str]:
    """The headline metrics, from config/eval.yaml (report.main_metrics); the pack carries the same file."""
    import yaml

    path = Path(__file__).resolve().parents[1] / "config" / "eval.yaml"
    return list(yaml.safe_load(path.read_text(encoding="utf-8"))["report"]["main_metrics"])


def _mode_suffix(mode: str) -> str:
    return "" if mode == "original" else f"-{mode}"


def _flat(prefix: str, value, out: dict) -> None:
    if isinstance(value, dict):
        for k, v in value.items():
            _flat(f"{prefix}.{k}" if prefix else str(k), v, out)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        out[prefix] = float(value)


def build_tables(stage: str, results: Path, bucket: str, llm: str, modes: list[str], no_sheaf: bool) -> list[tuple[str, pd.DataFrame]]:
    """Metrics as rows, values as columns."""
    tables: list[tuple[str, pd.DataFrame]] = []
    if stage == "sheaf":
        columns = {}
        for mode in modes:
            path = results / f"framework-{bucket}-{llm}{_mode_suffix(mode)}-sheaf" / "diagnostics.json"
            if path.exists():
                flat: dict = {}
                _flat("", json.loads(path.read_text(encoding="utf-8")), flat)
                columns[mode] = flat
        if columns:
            tables.append((f"SHEAF fit ({bucket}, {llm})", pd.DataFrame(columns)))
    elif stage == "road1":
        for mode in modes:
            folder = results / f"road1-{bucket}-{llm}{_mode_suffix(mode)}{'-nosheaf' if no_sheaf else ''}"
            for sub, name in ((folder, "test window"), (folder / "oracle", "ORACLE (realised source dials)")):
                fits, scores = sub / "fits.csv", sub / "road1_scores.csv"
                if fits.exists():
                    tables.append((f"MATRIX / GRAPH dev fit - {name} [{mode}]", pd.read_csv(fits).set_index("method").T))
                if scores.exists():
                    frame = pd.read_csv(scores)
                    for dial, part in frame.groupby("dial"):
                        keep = [c for c in ("r2", "ci_low", "ci_high", "rmse", "n", "n_dates") if c in part.columns]
                        tables.append((f"MATRIX / GRAPH out-of-sample - {name} - dial {dial} [{mode}]", part.set_index("method")[keep].T))
    elif stage == "optimise":
        for mode in modes:
            path = results / f"framework-{bucket}-{llm}{_mode_suffix(mode)}" / "summary.csv"
            if path.exists():
                frame = pd.read_csv(path)
                frame.index = frame["variant"].astype(str) + " | " + frame["delisting"].astype(str)
                metrics = frame.drop(columns=["variant", "delisting"]).T
                main = [m for m in main_metrics() if m in metrics.index]
                tables.append((f"FINAL RUN - MAIN metrics ({bucket}, {llm}) [{mode}]", metrics.loc[main]))
                tables.append((f"FINAL RUN - APPENDIX metrics ({bucket}, {llm}) [{mode}]", metrics.drop(index=main)))
                for extra in sorted(path.parent.glob("appendix_*.csv")):  # the appendix tables computed in the same run (finflock/real/appendix.py)
                    tables.append((f"FINAL RUN - APPENDIX - {extra.stem.removeprefix('appendix_').replace('_', ' ')} [{mode}]", pd.read_csv(extra).set_index("metric")))
    elif stage == "ablation":
        for folder in sorted(results.glob(f"ablation-{bucket}-{llm}-*")):
            columns, tier_a_columns = {}, {}
            for run in sorted((folder / "runs").glob("*")):
                if (run / "summary.csv").exists():
                    columns[run.name.removeprefix(f"abl-{bucket}-{llm}-")] = pd.read_csv(run / "summary.csv").iloc[0].drop(labels=["run_id", "variant", "delisting"], errors="ignore")
                if (run / "events.parquet").exists():
                    events = pd.read_parquet(run / "events.parquet")
                    tier_a_columns[run.name.removeprefix(f"abl-{bucket}-{llm}-")] = {f"mean_{c}": float(events[c].mean()) for c in events.columns if c.startswith("ic_")} | {"n_events": float(len(events))}
            if columns:
                metrics = pd.DataFrame(columns)
                main = [m for m in main_metrics() if m in metrics.index]
                tables.append((f"ABLATION - MAIN metrics, one column per run ({bucket}, {llm})", metrics.loc[main]))
                tables.append((f"ABLATION - APPENDIX metrics ({bucket}, {llm})", metrics.drop(index=main)))
            if tier_a_columns:
                tables.append((f"ABLATION - forecast level (event rank IC) ({bucket}, {llm})", pd.DataFrame(tier_a_columns)))
    return tables


def render(frame: pd.DataFrame) -> str:
    return frame.to_string(float_format=lambda x: f"{x:.0f}" if float(x).is_integer() else f"{x:.4f}", na_rep="-")  # counts without decimals


def save_tables(tables: list[tuple[str, pd.DataFrame]], work: Path) -> None:
    lines, long_rows = [], []
    for title, frame in tables:
        lines += [f"## {title}", "", "| metric | " + " | ".join(map(str, frame.columns)) + " |", "|---|" + "---|" * len(frame.columns)]
        for metric, row in frame.iterrows():
            lines.append(f"| {metric} | " + " | ".join(f"{v:.4f}" if isinstance(v, float) else str(v) for v in row) + " |")
            long_rows += [{"table": title, "metric": metric, "column": c, "value": row[c]} for c in frame.columns]
        lines.append("")
    (work / "final_results.md").write_text("\n".join(lines), encoding="utf-8")
    pd.DataFrame(long_rows).to_csv(work / "final_results.csv", index=False)


# ---- the session ------------------------------------------------------------------------------------------------------------------------------

def run_session(args: argparse.Namespace, child: list[str] | None = None) -> int:
    global START, CUTOFF, SESSION_LOG
    START, CUTOFF = time.time(), float(args.cutoff_seconds)
    work, input_root = Path(args.work), Path(args.input_root)
    work.mkdir(parents=True, exist_ok=True)
    SESSION_LOG = work / SESSION_LOG_NAME
    modes = [m for m in args.modes.split(",") if m]
    label = args.label
    ckpt_dir = work / "ckpt"
    log(f"SESSION START  run='{label}'  stage={args.stage}  bucket={args.bucket}  llm={args.llm}  modes={modes}  no_sheaf={args.no_sheaf}  fetch_vxf={args.fetch_vxf}")
    log(f"SETTINGS  cutoff {CUTOFF / 3600:.2f} h ({CUTOFF:.0f} s)  metrics every {args.tick_seconds} s  python {sys.version.split()[0]}  cpus {os.cpu_count()}")
    log(f"INPUTS under {input_root}: " + (", ".join(sorted(p.name for p in input_root.iterdir())) if input_root.exists() else "(folder missing)"))
    pack = Path(args.pack) if args.pack else None
    if pack is not None:
        manifest = json.loads((pack / "PACK_MANIFEST.json").read_text(encoding="utf-8")) if (pack / "PACK_MANIFEST.json").exists() else {}
        log(f"PACK {pack}: {len(manifest.get('files', {}))} files, built from git {str(manifest.get('git_commit'))[:10]}, config hash {str(manifest.get('config_hash'))[:12]}")
        describe_data_bundles(pack)
    shard = getattr(args, "shard", None)
    merge_shards = int(getattr(args, "merge_shards", 0) or 0)
    if shard and merge_shards:
        raise SystemExit("--shard and --merge-shards cannot be used together")
    if shard:
        match = re.match(r"^(\d+)/(\d+)$", shard)
        if not match or not 0 <= int(match.group(1)) < int(match.group(2)):
            raise SystemExit(f"--shard {shard!r} is not 'k/N' with 0 <= k < N")
        log(f"SHARD RUN {shard}: only the cross-validation fits owned by this shard are computed; the checkpoint bundle is the result")
    previous = open_checkpoint(input_root, label, ckpt_dir, merge_shards)
    session = (previous.get("session", 0) + 1) if previous else 1
    log(f"SESSION NUMBER {session} of this run" + (" (a resumed session)" if previous else " (a fresh start)"))
    command = child if child is not None else [sys.executable, str(Path(args.runner)), args.stage, "--bucket", args.bucket, "--llm", args.llm, "--modes", args.modes,
                                               "--input-root", str(input_root), "--work", str(work)] + (["--fetch-vxf"] if args.fetch_vxf else []) + (["--no-sheaf"] if args.no_sheaf else []) \
        + (["--notebook", args.notebook] if getattr(args, "notebook", None) else [])
    env = {**os.environ, "FINFLOCK_CKPT_DIR": str(ckpt_dir), "PYTHONUNBUFFERED": "1"}
    env.pop("FINFLOCK_FIT_SHARD", None)
    if shard:
        env["FINFLOCK_FIT_SHARD"] = shard
    log("STARTING the child: " + " ".join(command))
    proc = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1, env=env, start_new_session=True)
    follower = Follower(work)

    def pump() -> None:
        for line in proc.stdout:  # type: ignore[union-attr]
            follower.see(line)
            log(line.rstrip())
    reader = threading.Thread(target=pump, daemon=True)
    reader.start()
    cutoff_hit, last_tick, last_status = False, 0.0, 0.0
    status_every = float(getattr(args, "status_seconds", 600.0))
    while proc.poll() is None:
        time.sleep(1.0)
        elapsed = time.time() - START
        if elapsed >= CUTOFF:
            cutoff_hit = True
            log(f"CUTOFF reached at {elapsed / 3600:.3f} h: stopping the child now (SIGTERM, then SIGKILL after {KILL_WAIT_SECONDS} s)")
            try:
                os.killpg(proc.pid, signal.SIGTERM)
                proc.wait(timeout=KILL_WAIT_SECONDS)
            except (subprocess.TimeoutExpired, ProcessLookupError):
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            break
        if elapsed - last_tick >= args.tick_seconds:
            last_tick = elapsed
            follower.flush()
            cpu, mem = machine_numbers(proc.pid)
            row = {"time_utc": time.strftime("%H:%M:%S", time.gmtime()), "elapsed_s": round(elapsed), "left_s": round(CUTOFF - elapsed), "cpu_pct": round(cpu, 1),
                   "mem_gb": round(mem, 2), "ckpt_pieces": len(list(ckpt_dir.glob("*.pkl"))) if ckpt_dir.exists() else 0,
                   "fits_done": follower.fits_done, "fits_total": follower.fits_total, "last_val_loss": follower.last_val, "best_cv_loss": follower.best_cv,
                   "log_lines": follower.lines, "last_line": follower.last_line.replace('"', "'")}
            append_metrics(work, row)
            log("METRICS " + " ".join(f"{k}={row[k]}" for k in METRIC_COLUMNS if k != "last_line"))
            if elapsed - last_status >= status_every:
                last_status = elapsed
                for line in status_block(args, follower, elapsed, mem, session).splitlines():
                    log(line)
    reader.join(timeout=10)
    follower.flush()
    code = proc.returncode
    log(f"CHILD ENDED with exit code {code}" + (" (stopped by the cutoff)" if cutoff_hit else ""))
    results = work / "results"
    status = EXIT_CUTOFF if cutoff_hit else (EXIT_FINISHED if code == 0 else EXIT_FAILED)
    if status == EXIT_FINISHED and shard:  # a finished shard has no result tables: its stored fits are the result, so the bundle is always written
        bundle = write_checkpoint(work, ckpt_dir, label, args.stage, session, "shard finished")
        log(f"SHARD CHECKPOINT BUNDLE written: {bundle} ({bundle.stat().st_size / 1e6:.1f} MB, {len(list(ckpt_dir.glob('*.pkl')))} stored fit(s)). Download it from this notebook's output.")
    if status != EXIT_FINISHED:
        bundle = write_checkpoint(work, ckpt_dir, label, args.stage, session, "cutoff" if cutoff_hit else f"child exit {code}")
        pieces = len(list(ckpt_dir.glob("*.pkl"))) if ckpt_dir.exists() else 0
        log(f"CHECKPOINT BUNDLE written: {bundle} ({bundle.stat().st_size / 1e6:.1f} MB, {pieces} finished piece(s) of work)")
        log(f"TO CONTINUE: download {bundle.name} from this notebook's output, upload it as a private dataset, attach it to THIS notebook (with the same other inputs) and Run all. "
            f"It reads the bundle, skips the {pieces} finished piece(s) and carries on.")
    tables = build_tables(args.stage, results, args.bucket, args.llm, modes, args.no_sheaf)
    if tables:
        log("FINAL RESULTS TABLE" + (" (PARTIAL: the run did not finish)" if status != EXIT_FINISHED else ""))
        for title, frame in tables:
            log(f"=== {title}")
            for line in render(frame).splitlines():
                log(line)
        save_tables(tables, work)
        log("saved final_results.md and final_results.csv")
    elif status == EXIT_FINISHED:
        log("no result tables were found to print (check the step logs above)")
    outputs = work / f"{args.bucket}_cpu_outputs.zip"
    if outputs.exists():  # the child's own zip: add what the supervisor wrote after it
        with zipfile.ZipFile(outputs, "a", zipfile.ZIP_DEFLATED) as archive:
            present = set(archive.namelist())
            for name in ("session.log", "metrics_stream.csv", "final_results.md", "final_results.csv"):
                if (work / name).exists():
                    archive.write(work / name, name if name not in present else f"final_{name}")  # the child's zip already holds an earlier session.log
    log(f"SESSION END  status={'finished' if status == EXIT_FINISHED else 'stopped by the cutoff' if status == EXIT_CUTOFF else 'child failed'}  exit={status}")
    return status


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--label", required=True)
    parser.add_argument("--stage", required=True, choices=["validate", "sheaf", "road1", "optimise", "framework", "ablation"])
    parser.add_argument("--bucket", required=True)
    parser.add_argument("--llm", default="llm2")
    parser.add_argument("--modes", default="salvaged_no_certainty")
    parser.add_argument("--input-root", default="/kaggle/input")
    parser.add_argument("--work", default="/kaggle/working")
    parser.add_argument("--pack", default=None)
    parser.add_argument("--runner", default=str(Path(__file__).resolve().parent / "kaggle_db1_runner.py"))
    parser.add_argument("--cutoff-seconds", type=float, default=11.5 * 3600)
    parser.add_argument("--tick-seconds", type=float, default=10.0)
    parser.add_argument("--status-seconds", type=float, default=600.0, help="a status block (time left, stage and shard, last result with intervals, memory, next steps) at this interval")
    parser.add_argument("--no-sheaf", action="store_true")
    parser.add_argument("--notebook", default=None, help="ablation stage: the notebook name of config ablation.notebooks (its list of runs)")
    parser.add_argument("--fetch-vxf", action="store_true")
    parser.add_argument("--shard", default=None, help="'k/N': compute only the cross-validation fits owned by shard k of N; the checkpoint bundle is the result")
    parser.add_argument("--merge-shards", type=int, default=0, help="N: assemble run; unpack the checkpoint bundles of the N shard runs '<label>-s<k>of<N>' as well as this run's own")
    return run_session(parser.parse_args(argv))


if __name__ == "__main__":
    sys.exit(main())
