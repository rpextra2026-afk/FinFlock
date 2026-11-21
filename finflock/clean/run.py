"""Run the cleaning for one or more tapes and write ``clean_<run>.parquet`` and ``clean_<run>_report.json``."""

from __future__ import annotations

import argparse
import json
import logging
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import pandas as pd

from finflock.clean import placeholders, steps
from finflock.common.config import Config, load_config
from finflock.common.hashing import sha256_file, sha256_hex
from finflock.common.timing import TradingCalendar

LOG = logging.getLogger(__name__)
SECTOR_NODE = steps.NODE_SECTOR
KEEP_LOG_LEVEL = logging.INFO


def content_hashes(long: pd.DataFrame, decimals: int) -> dict[str, str]:
    """Hashes of the cleaned table that do not depend on the parquet writer: the frozen part (every touch point except the sector ones) and the sector rows
    (their assignment comes from a scikit-learn classifier and may differ by library version; reported, never blocking)."""
    def digest(frame: pd.DataFrame) -> str:
        frame = frame.sort_values(["unit_id", "node_type", "node_id"], na_position="first").reset_index(drop=True)
        frame = frame.round(decimals)
        text = frame.astype(str).agg("|".join, axis=1)
        return sha256_hex("\n".join(text))
    is_sector = long["node_type"].eq(SECTOR_NODE)
    return {"content_sha256": digest(long[~is_sector].drop(columns=["ts_utc"])), "sector_rows_sha256": digest(long[is_sector].drop(columns=["ts_utc"])), "rows_hashed": int((~is_sector).sum())}


def clean_run(name: str, cfg: Config, root: str, run_cfg: Mapping[str, Any] | None = None, out_dir: Path | None = None, tag: str = "") -> dict[str, Any]:
    """Cleans one tape. ``run_cfg`` / ``out_dir`` / ``tag`` are used only by the sealed-window stage (finflock.framework.sealed): the same steps and the same code on a longer window,
    written under another name; the defaults give the frozen build's behaviour unchanged."""
    started = time.time()
    run_cfg = run_cfg if run_cfg is not None else cfg.get(f"clean.runs.{name}")
    out_dir = Path(out_dir) if out_dir is not None else Path(root) / str(cfg.get("clean.output_dir"))
    out_dir.mkdir(parents=True, exist_ok=True)
    counter = steps.Counter()
    cal = TradingCalendar.from_config(cfg)
    themes = steps.compile_themes(cfg)
    items, load_note = steps.load_items(run_cfg, root, cfg)
    counter.record("load", load_note["tape_rows"], load_note["joined_rows"], **load_note)
    items, sealed = steps.split_windows(items, run_cfg, counter)
    items = steps.gate(items, themes, cfg, counter)
    links = pd.read_parquet(f"{root}/{run_cfg['links']}", columns=["src_node", "dst_node"])
    nodes = set(links["src_node"]) | set(links["dst_node"])
    items = steps.entity_check(items, nodes, cfg, counter)
    items = steps.session_timing(items, cal, cfg, counter)
    items = steps.dedupe(items, cal, cfg, counter)
    items = steps.assign_lanes(items, themes, cfg, counter)
    items, note_repair = placeholders.cause_repair_embed(items, cfg, root)
    items, note_themes = placeholders.world_themes_embed(items, cfg, root)
    group_of = {c: g for g, causes in cfg.get("causes.groups").items() for c in causes}
    items = steps.cause_groups(items, cfg, group_of, counter)
    prices = pd.read_parquet(f"{root}/{run_cfg['prices']}", columns=["security_id", "session_date", "close_raw"])
    rets = steps.returns_frame(prices, cfg)
    trust = steps.learn_trust(items, rets, run_cfg, cfg, counter)
    sector = steps.fit_sector_model(items, rets, sorted(nodes), cfg, counter)
    long = steps.touch_points(items, trust, sector, cfg, counter)
    long, note_direction = placeholders.direction_finbert(long, cfg, root)
    path = out_dir / f"clean_{name}{tag}.parquet"
    long.to_parquet(path, index=False)
    sealed_path = out_dir / f"clean_{name}{tag}_sealed_rows.parquet"
    sealed[[steps.UNIT_ID, "available_date"]].to_parquet(sealed_path, index=False)  # ids and dates only: the sealed text is never cleaned
    report = {"run": name, "bucket": run_cfg["bucket"], "windows": {k: str(run_cfg[k]) for k in ("fit_start", "fit_end", "holdout_start", "holdout_end")}, "steps": counter.steps, "placeholders": {"cause_repair_embed": note_repair, "world_themes_embed": note_themes, "direction_finbert": note_direction}, "items_out": int(long[steps.UNIT_ID].nunique()), "rows_out": int(len(long)), "sealed_rows": int(len(sealed)), "file": str(path.relative_to(root)) if path.is_relative_to(root) else str(path), "file_sha256": sha256_file(path), **content_hashes(long, int(cfg.get("clean.report.decimals"))), "config_hash": cfg.hash, "clean_section_hash": cfg.unhashed_hash, "seconds": round(time.time() - started, int(cfg.get("clean.report.decimals")))}
    (out_dir / f"clean_{name}{tag}_report.json").write_text(json.dumps(report, indent=2, default=str))
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Final-build cleaning of the reader tapes")
    parser.add_argument("--config", default="config")
    parser.add_argument("--root", default=".")
    parser.add_argument("runs", nargs="*", help="run names; default is every run in clean.runs")
    args = parser.parse_args()
    logging.basicConfig(level=KEEP_LOG_LEVEL, format="%(message)s")
    cfg = load_config(args.config)
    for name in args.runs or list(cfg.get("clean.runs")):
        report = clean_run(name, cfg, args.root)
        LOG.info("%s: %d items, %d rows, %s s", name, report["items_out"], report["rows_out"], report["seconds"])


if __name__ == "__main__":
    main()
