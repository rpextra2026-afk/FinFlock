"""Per-job stagers: how a notebook session builds its own shard folders (so nothing but the market dataset is attached)."""

from __future__ import annotations

import os
import tempfile
import zipfile
from collections.abc import Callable
from pathlib import Path

from finfm.common.config import Config, load_config
from finfm.compute.session import RunSpec, SessionError
from finfm.stage.acquire import download_hf_subsets
from finfm.stage.mass_build import stage_mass_job
from finfm.stage.persona_stage import stage_bios_job, stage_persona_job, stage_persona_retry
from finfm.stage.rival_stage import stage_finripple_job
from finfm.stage.reader_stage import PRESTAGED, UNITS_FILE, sources_needed, stage_job

MARKET_MARKER = "aliases.parquet"
LISTING_MAX_ENTRIES = 60
LISTING_FILES_SHOWN = 6
GUARDIAN = "guardian"


def _walk(input_root: Path):
    """``(folder, folder names, file names)`` under ``input_root``, following symlinks (``Path.rglob`` does not; an attached dataset may be one)."""
    return os.walk(input_root, followlinks=True)


def _listing(input_root: Path) -> str:
    """What is attached, every level, so a missing dataset or an odd layout is obvious from the message alone."""
    if not input_root.exists():
        return f"{input_root} does not exist"
    lines = []
    for folder, _, files in sorted(_walk(input_root)):
        rel = Path(folder).relative_to(input_root)
        lines.append(f"{rel}/ [{len(files)} files{': ' + ', '.join(sorted(files)[:LISTING_FILES_SHOWN]) if files else ''}]")
        if len(lines) >= LISTING_MAX_ENTRIES:
            lines.append("...")
            break
    return "; ".join(lines) or "nothing at all"


def _unpack_market_zip(input_root: Path) -> Path | None:
    """If Kaggle left the market dataset as a zip (one holding ``aliases.parquet``), unpack it to a scratch folder."""
    for folder, _, files in sorted(_walk(input_root)):
        for name in sorted(files):
            if not name.lower().endswith(".zip"):
                continue
            with zipfile.ZipFile(Path(folder) / name) as bundle:
                if any(n == MARKET_MARKER or n.endswith("/" + MARKET_MARKER) for n in bundle.namelist()):
                    target = Path(tempfile.mkdtemp(prefix="finfm_market_"))
                    bundle.extractall(target)
                    return target
    return None


def _folders_with(input_root: Path, name: str) -> list[Path]:
    """Folders under ``input_root`` that hold a file called ``name``."""
    return sorted(Path(folder) for folder, _, files in _walk(input_root) if name in files)


def find_market_dir(input_root: Path) -> Path:
    """The attached folder that holds the market dataset (prices, membership, names, Guardian files)."""
    hits = _folders_with(input_root, MARKET_MARKER)
    if hits:
        return hits[0]
    unpacked = _unpack_market_zip(input_root)
    inner = _folders_with(unpacked, MARKET_MARKER) if unpacked else []
    if inner:
        return inner[0]
    raise SessionError(f"the data dataset of this run is not attached: no folder with {MARKET_MARKER} under {input_root}. "
                       f"Attach the tessera data dataset (Add Input). What is attached now: {_listing(input_root)}")


def make_stager(run: RunSpec, config_dir: Path, input_root: Path, shard: str | int | None = None, retry: tuple[int, int] | None = None) -> Callable[[Path], Path] | None:
    """The function that stages this run's shards into a scratch folder, or None when the run's shards must be attached."""
    if run.job not in ("reader", "basic", "personas", "bios", "finripple", "mass"):
        raise SessionError(f"in-notebook staging for job {run.job} is not built yet")

    def stage(scratch: Path) -> Path:
        cfg: Config = load_config(config_dir)
        market = find_market_dir(input_root)
        if run.job == "personas" and retry:  # (part, parts): one shard holding this part of the error units of the attached bundles
            stage_persona_retry(run.bucket, run.llm, cfg, market, input_root, scratch / "shards", part=retry[0], parts=retry[1])
            return scratch / "shards"
        if run.job == "personas":
            stage_persona_job(run.bucket, run.llm, cfg, market, input_root, scratch / "shards", only_shard=shard)
            return scratch / "shards"
        if run.job == "mass":
            stage_mass_job(run.bucket, run.llm, cfg, market, scratch / "shards")
            return scratch / "shards"
        if run.job == "finripple":
            from finfm.compute.factory import HFTokenizer, resolve_model_path
            from finfm.reader.prompt import WhitespaceTokenizer

            mock = str(cfg.get("engine.engine")) == "mock"
            tokenizer = WhitespaceTokenizer() if mock else HFTokenizer(resolve_model_path(cfg, str(cfg.get(f"engine.models.{run.llm}.path"))))
            stage_finripple_job(run.bucket, run.llm, cfg, market, tokenizer, scratch / "shards")
            return scratch / "shards"
        if run.job == "bios":
            stage_bios_job(run.llm, cfg, market, scratch / "shards")
            return scratch / "shards"
        hf = None
        if not (market / PRESTAGED / f"{run.job}-{run.bucket}" / UNITS_FILE).exists():  # no prebuilt units: build from raw news
            token = os.environ.get(str(cfg.get("sources.hf.env_token")), "")
            download_hf_subsets(cfg, token, scratch / "hf", [s for s in sources_needed(cfg, run.bucket) if s != GUARDIAN])
            hf = scratch / "hf"
        stage_job(run.job, run.bucket, run.llm, cfg, config_dir, market, hf, scratch / "shards", only_shard=shard)
        return scratch / "shards"

    return stage
