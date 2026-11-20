"""Engines of the final reader.

``ReaderV2VLLM`` is the engine of the proof-of-concept that parsed 100 % of the answers on 2 x T4: JSON-schema constrained
decoding with ``disable_any_whitespace`` (the earlier reader lost most answers to endless whitespace after the pins).
One class serves all three LLMs; Gemma additionally gets the Turing attention patch of the production engine.

``EngineSupervisor`` wraps whichever engine is built and rebuilds it (a bounded number of times) when it dies, so a crashed
vLLM process costs one batch retry, not the run.
"""
from __future__ import annotations

import importlib
import os
from collections.abc import Callable, Sequence
from typing import Any

from oryxis.common.config import Config, load_config
from oryxis.compute.engine import Engine, GenParams, GenRequest, GenResult
from oryxis.compute.factory import resolve_model_path
from oryxis.compute.parallel import DataParallelEngine
from oryxis.compute.vllm_engine import STRUCTURED_CANDIDATES, VLLMEngine, patch_triton_attention

from reader2 import settings as S

TASK = "reader_v2"
WIDE_HEAD_LLMS = ("llm1",)  # Gemma-4 has 256- and 512-wide heads, which a T4's shared memory cannot take without the patch


class ReaderV2VLLM(VLLMEngine):
    """VLLMEngine with the POC's compact structured decoding for the reader task."""

    def __init__(self, model_path: str, cfg: Config, *, tensor_parallel_size: int, patch_attention: bool = False,
                 vllm_module: Any = None, llm_factory: Callable[..., Any] | None = None) -> None:
        self.cfg = cfg
        if patch_attention:
            patch_triton_attention()
        backend = cfg.get("engine.attention_backend_env")
        if backend:
            os.environ[str(backend["name"])] = str(backend["value"])
        self.vllm = vllm_module or importlib.import_module("vllm")
        self._factory = llm_factory or self.vllm.LLM
        self._kwargs = dict(
            model=model_path, dtype=str(cfg.get("engine.dtype")), tensor_parallel_size=tensor_parallel_size,
            max_model_len=int(cfg.get("engine.max_model_len")), gpu_memory_utilization=float(cfg.get("engine.gpu_memory_utilization")),
            enable_prefix_caching=True, enforce_eager=bool(cfg.get("engine.enforce_eager")), max_num_seqs=int(cfg.get("engine.max_num_seqs")))
        self.llm = self._build_llm()
        self.tokenizer = self.llm.get_tokenizer()
        self.model_path, self.system_ok = model_path, cfg.get("engine.system_role_supported")
        self.structured_keyword, self.whitespace_option, self.compact_enabled = None, None, True
        self.restarts = 0

    def _build_llm(self):
        """CUDA-graph capture can fail on some cards: the same engine is then built again without graphs."""
        try:
            return self._factory(**self._kwargs)
        except RuntimeError as exc:
            text = str(exc).lower()
            if "graph" not in text or not any(word in text for word in ("capture", "cuda")) or self._kwargs.get("enforce_eager"):
                raise
            self._kwargs["enforce_eager"] = True
            self.fell_back_to_eager = True
            return self._factory(**self._kwargs)

    def restart(self) -> None:  # VLLMEngine.restart rebuilds through self._factory(**self._kwargs): keep the graph fallback
        try:
            self.llm.llm_engine.engine_core.shutdown()
        except Exception:
            pass
        del self.llm
        import gc
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
        self.llm = self._build_llm()
        self.tokenizer = self.llm.get_tokenizer()

    def _sampling(self, params: GenParams, request: GenRequest) -> Any:
        if request.task != TASK or not self.compact_enabled:
            return super()._sampling(params, request)
        sampling = getattr(self.vllm, "sampling_params", None) or importlib.import_module("vllm.sampling_params")
        for keyword, class_name in STRUCTURED_CANDIDATES:
            cls = getattr(sampling, class_name, None)
            if cls is None:
                continue
            for option in ({"disable_any_whitespace": True}, {"whitespace_pattern": ""}, {}):
                try:
                    structured = cls(json=dict(params.json_schema), **option)
                    result = self.vllm.SamplingParams(
                        n=request.n, temperature=params.temperature, top_p=params.top_p, max_tokens=params.max_tokens,
                        seed=self._seed(request.seed if request.seed is not None else params.seed), **{keyword: structured})
                except (TypeError, ValueError):
                    continue
                self.structured_keyword, self.whitespace_option = keyword, (next(iter(option)) if option else "backend_default")
                return result
        raise RuntimeError("no_supported_structured_output")

    def generate(self, requests: Sequence[GenRequest], params: GenParams) -> list[GenResult]:
        try:
            return super().generate(requests, params)
        except (TypeError, ValueError) as exc:
            if any(r.task == TASK for r in requests) and self.compact_enabled and "whitespace" in str(exc).lower():
                self.compact_enabled = False  # this vLLM rejects the option: continue with its default grammar
                self.whitespace_option = "backend_default_after_rejection"
                return super().generate(requests, params)
            raise

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "whitespace_option": self.whitespace_option, "dtype": self._kwargs["dtype"],
                "tensor_parallel_size": self._kwargs["tensor_parallel_size"], "enforce_eager": self._kwargs["enforce_eager"],
                "restarts": getattr(self, "restarts", 0)}


class _ReplicaFactory:
    """Picklable per-replica factory (a replica is a separate process that sees exactly one GPU)."""

    def __init__(self, config_dir: str, path: str, tensor_parallel: int, overrides: dict, patch_attention: bool) -> None:
        self.args = (config_dir, path, tensor_parallel, dict(overrides), patch_attention)

    def __call__(self, index: int) -> Engine:
        config_dir, path, tensor_parallel, overrides, patch = self.args
        cfg = load_config(config_dir, overrides)
        return ReaderV2VLLM(resolve_model_path(cfg, path), cfg, tensor_parallel_size=tensor_parallel, patch_attention=patch)


def build_engine_v2(cfg: Config, llm: str, config_dir: str, overrides: dict | None = None) -> Engine:
    """The engine of one LLM: one tensor-parallel engine over both GPUs (llm2, llm3) or one replica per GPU (llm1)."""
    spec = cfg.get(f"engine.models.{llm}")
    replicas, tensor_parallel = int(spec["replicas"]), int(spec["tensor_parallel"])
    patch = llm in WIDE_HEAD_LLMS
    if replicas > 1:
        return DataParallelEngine(_ReplicaFactory(str(config_dir), spec["path"], tensor_parallel, overrides or {}, patch), replicas)
    return ReaderV2VLLM(resolve_model_path(cfg, spec["path"]), cfg, tensor_parallel_size=tensor_parallel, patch_attention=patch)


class EngineSupervisor(Engine):
    """Rebuilds the wrapped engine after a fatal error (dead engine process, CUDA error, dead replica) and repeats the call."""

    name = "supervised"

    def __init__(self, factory: Callable[[], Engine], *, max_rebuilds: int = S.ENGINE_REBUILDS, log=print) -> None:
        self.factory, self.max_rebuilds, self.log = factory, max_rebuilds, log
        self.rebuilds = 0
        self.engine = factory()

    @property
    def tokenizer(self):
        return getattr(self.engine, "tokenizer", None)

    @staticmethod
    def is_fatal(exc: BaseException) -> bool:
        text = f"{type(exc).__name__} {exc}"
        return any(marker in text for marker in S.FATAL_ENGINE_MARKERS)

    def rebuild(self) -> None:
        try:
            self.engine.close()
        except Exception as exc:
            self.log(f"READER2 old engine did not close cleanly ({exc!r}); building a new one anyway")
        self.engine = self.factory()

    def generate(self, requests: Sequence[GenRequest], params: GenParams) -> list[GenResult]:
        consecutive = 0  # rebuilds in a row for this call; a call that works resets it, so a long run is not limited to a few crashes
        while True:
            try:
                return self.engine.generate(requests, params)
            except Exception as exc:
                if not self.is_fatal(exc) or consecutive >= self.max_rebuilds or self.rebuilds >= self.max_rebuilds * S.REBUILD_TOTAL_FACTOR:
                    raise
                consecutive += 1
                self.rebuilds += 1
                self.log(f"READER2 engine error ({type(exc).__name__}: {str(exc)[:200]}); rebuilding the engine "
                         f"({consecutive}/{self.max_rebuilds} in a row, {self.rebuilds} in this session)")
                self.rebuild()

    def describe(self) -> dict[str, Any]:
        inner = self.engine.describe() if hasattr(self.engine, "describe") else {"engine": getattr(self.engine, "name", "unknown")}
        return {**inner, "rebuilds": self.rebuilds}

    def close(self) -> None:
        self.engine.close()
