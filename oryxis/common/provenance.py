"""Provenance stamped into every bundle and result: code commit, versions, time."""

from __future__ import annotations

import datetime as dt
import importlib.metadata
import platform
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Provenance:
    git_commit: str | None
    git_dirty: bool | None
    python: str
    platform: str
    packages: dict[str, str | None]
    created_utc: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _git(args: list[str], cwd: Path, timeout: float) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, timeout=timeout, check=True
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip()


def package_versions(names: list[str]) -> dict[str, str | None]:
    """Installed version of each package, ``None`` when it is not installed."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def collect_provenance(
    repo_dir: str | Path, packages: list[str], git_timeout_seconds: float
) -> Provenance:
    """Collect the commit, dirty flag, interpreter, platform and package versions."""
    cwd = Path(repo_dir)
    commit = _git(["rev-parse", "HEAD"], cwd, git_timeout_seconds)
    status = _git(["status", "--porcelain"], cwd, git_timeout_seconds)
    return Provenance(
        git_commit=commit,
        git_dirty=None if status is None else bool(status),
        python=platform.python_version(),
        platform=platform.platform(),
        packages=package_versions(packages),
        created_utc=dt.datetime.now(dt.timezone.utc).isoformat(),
    )
