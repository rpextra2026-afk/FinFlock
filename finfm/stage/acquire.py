"""Resolve separately supplied local news subsets without remote dataset links."""
from __future__ import annotations
from collections.abc import Callable, Iterable
from pathlib import Path
from finfm.common.config import Config

class AcquireError(RuntimeError):
    """A required local input is unavailable."""

def download_hf_subsets(cfg: Config, token: str, out_dir: Path,
                        sources: Iterable[str], log: Callable[[str], None] = print) -> list[Path]:
    """Compatibility entry point: resolve local files; no download or token use.

    Each requested source lives under out_dir/<source>/. Supply licensed inputs
    separately; source repository URLs are not part of this release.
    """
    files = []
    for source in sources:
        folder = out_dir / source
        found = sorted(p for p in folder.rglob("*") if p.is_file()) if folder.is_dir() else []
        if not found:
            raise AcquireError(f"Missing local news subset: {folder}. Supply the input files separately.")
        files.extend(found)
        log(f"{source}: {len(found)} local files")
    return files
