"""Engine and tokenizer construction from config (spec 2.2): quantised Gemma with a fallback, local weights."""

from __future__ import annotations

import os
from pathlib import Path

from finfm.common.config import Config
from finfm.compute.engine import Engine, MockEngine
from finfm.compute.parallel import DataParallelEngine
from finfm.compute.vllm_engine import VLLMEngine

QUANT_MARKERS = ("w4a16", "qat")


def resolve_model_path(cfg: Config, path: str) -> str:
    """A weights directory staged under ``local_weights_root`` wins over the hub id (no internet in sessions)."""
    root = Path(os.environ.get(str(cfg.get("engine.weights_root_env")), str(cfg.get("engine.local_weights_root"))))
    local = root / path.replace("/", "--")
    return str(local) if local.exists() else path


def _vllm_factory(cfg_dir: str, path: str, tensor_parallel: int, overrides: dict, _index: int) -> Engine:
    from finfm.common.config import load_config

    cfg = load_config(cfg_dir, overrides)
    return VLLMEngine(resolve_model_path(cfg, path), cfg, tensor_parallel_size=tensor_parallel)


class _ReplicaFactory:
    """Picklable per-replica factory for ``DataParallelEngine``."""

    def __init__(self, cfg_dir: str, path: str, tensor_parallel: int, overrides: dict) -> None:
        self.args = (cfg_dir, path, tensor_parallel, overrides)

    def __call__(self, index: int) -> Engine:
        return _vllm_factory(*self.args, index)


def build_engine(cfg: Config, llm_key: str, config_dir: str, *, use_fallback: bool = False, overrides: dict | None = None) -> Engine:
    """The engine of one LLM. ``engine: mock`` in config gives the deterministic mock for CPU runs."""
    if str(cfg.get("engine.engine")) == "mock":
        return MockEngine()
    spec = cfg.get(f"engine.models.{llm_key}")
    if use_fallback and spec.get("fallback"):
        spec = {**spec, **spec["fallback"]}
    replicas, tensor_parallel = int(spec["replicas"]), int(spec["tensor_parallel"])
    if replicas > 1:
        return DataParallelEngine(_ReplicaFactory(config_dir, spec["path"], tensor_parallel, overrides or {}), replicas)
    return _vllm_factory(config_dir, spec["path"], tensor_parallel, overrides or {}, 0)


class HFTokenizer:
    """Model tokenizer for text truncation (the reader's 1,500 tokens, the persona depths, the KG budget)."""

    def __init__(self, model_path: str) -> None:
        from transformers import AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

    def truncate(self, text: str, max_tokens: int) -> tuple[str, bool]:
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        if len(ids) <= max_tokens:
            return text, False
        return self.tokenizer.decode(ids[:max_tokens]), True
