"""Reading and checking a saved dev fit (``dev_fit.pkl``, written by part 1 of the framework run; a pickle of this project only, never load one from anywhere else)."""

from __future__ import annotations

import pickle
from pathlib import Path

DEV_FILE = "dev_fit.pkl"
# top-level config sections that cannot change the dial maps or the fitted sheaf (reporting, portfolio, infrastructure): they are left out of the hash a saved dev fit is checked against
DEV_FIT_IGNORED_SECTIONS = ("eval", "portfolio", "kaggle", "runs", "runtime", "engine", "smoke")
DEV_FIT_KEYS = ("bucket", "llm", "reader_mode", "arm")


def dev_fit_problems(saved: dict, current: dict) -> tuple[list[str], list[str]]:
    """(problems that stop the run, warnings) when a saved dev fit is compared with this run. Files written before the fit hash existed are checked on the
    four keys only and say so."""
    problems = [f"{k}: saved {saved.get(k)!r}, this run {current[k]!r}" for k in DEV_FIT_KEYS if saved.get(k) != current[k]]
    warnings = []
    if "fit_config_hash" in saved:
        if saved["fit_config_hash"] != current["fit_config_hash"]:
            problems.append("the settings that shape the fit (everything except reporting, portfolio and infrastructure) differ from the run that made the file")
    else:
        warnings.append("this dev_fit.pkl was written before the fit hash existed: bucket, llm, reader mode and arm match, but its configuration cannot be verified")
    return problems, warnings


def read_dev_fit(path: Path, meta: dict) -> tuple:
    """The (DevFit, source dials) of ``path``, refused when it was made for another bucket, LLM, reader mode, arm or fit configuration."""
    with Path(path).open("rb") as handle:
        saved = pickle.load(handle)
    problems, warnings = dev_fit_problems(saved["meta"], meta)
    for warning in warnings:
        print(f"WARNING: {warning}", flush=True)
    if problems:
        raise SystemExit(f"{path} does not fit this run: " + "; ".join(problems) + "; run part 1 again with the same settings")
    return saved["dev"], saved["dials"]
