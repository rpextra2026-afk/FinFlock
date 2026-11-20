"""Day-0 smoke test measurements (spec 2.7): fp16 overflow gate, throughput, quantised-vs-fp16 agreement.

These functions only measure and report numbers; the budget script and the owner decide what to do. They run
against any ``Engine`` so the plumbing is tested on CPU with the mock engine; the first real run on Kaggle
fills in the numbers.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from oryxis.common.config import Config
from oryxis.common.io import atomic_write_json
from oryxis.compute.engine import Engine, GenParams, GenRequest, GenResult
from oryxis.compute.parallel import describe
from oryxis.common.units import SECONDS_PER_MINUTE

TINY = 1e-9  # guards a division by a zero duration
COMMAND_TIMEOUT_SECONDS = 30


def environment_report() -> dict[str, Any]:
    """Versions and hardware facts (written into experiment configuration section A by the owner)."""
    info: dict[str, Any] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in ("torch", "vllm", "transformers", "numpy", "pandas"):
        try:
            info[name] = __import__(name).__version__
        except Exception:  # not installed in this session
            info[name] = None
    try:
        smi = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=COMMAND_TIMEOUT_SECONDS, check=False)
        info["gpus"] = [line.strip() for line in smi.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.SubprocessError):
        info["gpus"] = []
    return info


def nan_gate(engine: Engine, requests: Sequence[GenRequest], params: GenParams) -> dict[str, float]:
    """Share of outputs that look like NaN / inf logits (empty text for a non-empty prompt)."""
    results = engine.generate(list(requests), params)
    flagged = sum(r.nan_suspect for r in results)
    return {"n": len(results), "nan_suspect": flagged, "nan_share": flagged / max(len(results), 1)}


def measure_throughput(engine: Engine, requests: Sequence[GenRequest], params: GenParams, cfg: Config,
                       parse: Callable[[GenResult], bool] | None = None) -> dict[str, float]:
    """Prefill and decode speed, cache effect and parse rate.

    Pass 1 generates one token per request (time ~ prefill). Pass 2 is the real generation on the same
    prompts (prefix cache warm). Pass 3 repeats it, so ``cache_speedup`` compares pass 2 with pass 3.
    """
    probe = replace(params, max_tokens=int(cfg.get("smoke.decode_probe_tokens")))
    start = time.perf_counter()
    prefill = engine.generate(list(requests), probe)
    t_prefill = time.perf_counter() - start
    start = time.perf_counter()
    full = engine.generate(list(requests), params)
    t_full = time.perf_counter() - start
    start = time.perf_counter()
    engine.generate(list(requests), params)
    t_repeat = time.perf_counter() - start
    tokens_in = sum(r.tokens_in for r in prefill)
    tokens_out = sum(r.tokens_out for r in full)
    decode_time = max(t_full - t_prefill, TINY)
    return {
        "n_prompts": len(requests), "tokens_in": tokens_in, "tokens_out": tokens_out,
        "prefill_tps_uncached": tokens_in / max(t_prefill, TINY), "decode_tps": tokens_out / decode_time,
        "cache_speedup": t_full / max(t_repeat, TINY), "seconds_prefill": t_prefill, "seconds_full": t_full,
        "parse_rate": float(np.mean([parse(r) for r in full])) if parse else float("nan"),
        "mean_tokens_in": tokens_in / max(len(requests), 1), "mean_tokens_out": tokens_out / max(len(requests), 1),
    }


def agreement(numeric_a: pd.DataFrame, numeric_b: pd.DataFrame, categorical_a: pd.DataFrame | None = None,
              categorical_b: pd.DataFrame | None = None) -> dict[str, float]:
    """Spearman correlation per numeric field and agreement share per categorical field (quantised vs fp16)."""
    out: dict[str, float] = {}
    for column in numeric_a.columns:
        both = pd.concat([numeric_a[column], numeric_b[column]], axis=1).dropna()
        out[f"spearman_{column}"] = float(spearmanr(both.iloc[:, 0], both.iloc[:, 1])[0]) if len(both) > 2 else float("nan")
    for column in (categorical_a.columns if categorical_a is not None else []):
        out[f"agree_{column}"] = float((categorical_a[column] == categorical_b[column]).mean())
    return out


def equivalence_verdict(values: Mapping[str, float], cfg: Config) -> dict[str, Any]:
    """Numbers against the configured thresholds. The choice of weights (quantised or fp16) follows it."""
    low_num = float(cfg.get("smoke.min_spearman"))
    low_cat = float(cfg.get("smoke.min_agreement"))
    worst_numeric = min((v for k, v in values.items() if k.startswith("spearman_")), default=float("nan"))
    worst_cat = min((v for k, v in values.items() if k.startswith("agree_")), default=float("nan"))
    return {"worst_spearman": worst_numeric, "worst_agreement": worst_cat,
            "use_quantised": bool(worst_numeric >= low_num and (np.isnan(worst_cat) or worst_cat >= low_cat))}


def measured_overrides(model_load_seconds: float, final_write_seconds: float) -> dict[str, float]:
    """Candidate lines for ``config/measured.yaml``: minutes with a rounded-up margin."""
    return {"kaggle.model_load_minutes": float(np.ceil(model_load_seconds / SECONDS_PER_MINUTE)),
            "kaggle.final_write_reserve_minutes": float(np.ceil(final_write_seconds / SECONDS_PER_MINUTE))}


def write_report(path: Path, report: Mapping[str, Any]) -> None:
    atomic_write_json(path, dict(report))


def timed_engine_load(build: Callable[[], Engine]) -> tuple[Engine, float]:
    start = time.perf_counter()
    engine = build()
    return engine, time.perf_counter() - start


def summarize(engine: Engine, throughput: Mapping[str, float], nan: Mapping[str, float], load_seconds: float) -> dict[str, Any]:
    return {"engine": describe(engine), "environment": environment_report(), "model_load_seconds": load_seconds,
            "throughput": dict(throughput), "nan_gate": dict(nan), "json": json.dumps({"ok": nan["nan_suspect"] == 0})}
