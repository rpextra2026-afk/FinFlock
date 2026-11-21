"""The vLLM engine (offline ``LLM`` class), written to survive API differences between versions.

Nothing here is exercised on the CPU test machine except through fakes: the first real run is the Day-0
smoke test. Version-sensitive points are detected at run time (structured-output parameter name,
chat-template system-role support) instead of being assumed; the detected values are recorded in
``describe()`` and written into the smoke-test report and every bundle's metadata.
"""

from __future__ import annotations

import importlib
import importlib.util
import os
import re
from pathlib import Path
from collections.abc import Callable, Sequence
from typing import Any

from finflock.common.config import Config
from finflock.compute.engine import Engine, GenParams, GenRequest, GenResult

STRUCTURED_CANDIDATES = (
    ("structured_outputs", "StructuredOutputsParams"),  # newer vLLM
    ("guided_decoding", "GuidedDecodingParams"),  # older vLLM
)


def structured_kwargs(vllm_module: Any, schema: dict) -> tuple[dict[str, Any], str]:
    """Keyword argument for ``SamplingParams`` that requests JSON-schema constrained output.

    Tries the newer parameter first, then the older one; returns the kwargs and the name that worked.
    """
    sampling = getattr(vllm_module, "sampling_params", None) or importlib.import_module(f"{vllm_module.__name__}.sampling_params")
    errors = []
    for keyword, class_name in STRUCTURED_CANDIDATES:
        cls = getattr(sampling, class_name, None)
        if cls is None:
            continue
        try:
            return {keyword: cls(json=schema)}, keyword
        except TypeError as exc:  # the class exists but takes a different argument
            errors.append(f"{class_name}: {exc}")
    raise RuntimeError(f"no supported structured-output parameter in this vLLM ({errors})")


SEED_BITS = 63  # vLLM passes the seed to a C long: a 64-bit hash seed made the engine crash ("Python int too large to convert to C long")
PROMPT_TOO_LONG = "prompt_too_long"  # finish reason of a request that was not sent because its prompt left no room for an answer


TRITON_ATTENTION_FILE = ("vllm", "v1", "attention", "ops", "triton_unified_attention.py")
TRITON_MARKER = "# finflock: small tile and one stage for wide heads (Turing shared memory)"
TRITON_STAGES_ANCHOR = re.compile(r"^(?P<indent>[ \t]*)if tuned_large_head:.*?\n(?:(?P=indent)[ \t]+.*\n|[ \t]*\n)+", re.M)
TRITON_TILE_ANCHOR = re.compile(r"^(?P<indent>[ \t]*)TILE_SIZE_DECODE = _get_tile_size\(.*?\n(?P=indent)\)\n", re.M | re.S)
WIDE_HEAD = 256            # heads this wide or wider (Gemma-4: 256 local, 512 global) overflow a T4's 64 KB of shared memory per block
SMALL_TILE = 16            # key/value rows per step; the kernel's own decode tile
CONTEXT_LINES = 18


def patch_triton_attention(log=print) -> bool:
    """Turing (T4) has 64 KB of shared memory per block; vLLM's Triton attention asks 96 KB for Gemma-4's 256- and 512-wide heads
    (32-row tiles, 3 pipeline stages). Use 16-row tiles and one stage for such heads (vLLM issue 38918); same maths, slower steps.
    Edits the installed vLLM file before the engine process starts; says what it did; never raises."""
    try:
        spec = importlib.util.find_spec("vllm")
        if spec is None or not spec.submodule_search_locations:
            return False
        path = Path(next(iter(spec.submodule_search_locations)), *TRITON_ATTENTION_FILE[1:])
        source = path.read_text()
        if TRITON_MARKER in source:
            log(f"vllm triton attention: already patched ({path})")
            return True
        stages, tile = TRITON_STAGES_ANCHOR.search(source), TRITON_TILE_ANCHOR.search(source)
        if "launch_num_stages" not in source or "head_size" not in source or not stages or not tile:
            lines = [ln for ln in source.splitlines() if "num_stages" in ln or "tuned_large_head" in ln or "TILE_SIZE_" in ln][:CONTEXT_LINES]
            log("vllm triton attention: patch anchor not found; relevant lines: " + " | ".join(ln.strip() for ln in lines))
            return False
        pad, pad2 = tile.group("indent"), stages.group("indent")
        small = (f"{pad}{TRITON_MARKER}\n{pad}if head_size >= {WIDE_HEAD}:\n"
                 f"{pad}    TILE_SIZE_PREFILL = min(TILE_SIZE_PREFILL, {SMALL_TILE})\n{pad}    TILE_SIZE_DECODE = min(TILE_SIZE_DECODE, {SMALL_TILE})\n")
        one = f"{pad2}if launch_num_stages is None and head_size >= {WIDE_HEAD}:\n{pad2}    launch_num_stages = 1\n"
        source = source[:tile.end()] + small + source[tile.end():]  # later in the file first, so the earlier offset stays valid
        source = source[:stages.end()] + one + source[stages.end():]
        path.write_text(source)
        log(f"vllm triton attention: patched ({path})")
        return True
    except Exception as exc:  # a failed patch must not hide the engine's own error
        log(f"vllm triton attention: patch failed ({exc!r})")
        return False


class VLLMEngine(Engine):
    name = "vllm"

    def __init__(
        self, model_path: str, cfg: Config, *, tensor_parallel_size: int, quantization: str | None = None,
        llm_factory: Callable[..., Any] | None = None, vllm_module: Any | None = None,
    ) -> None:
        self.cfg = cfg
        patch_triton_attention()
        backend = cfg.get("engine.attention_backend_env")
        if backend:  # [VERIFY] name/value from the Day-0 check, set before vLLM is imported
            os.environ[str(backend["name"])] = str(backend["value"])
        self.vllm = vllm_module or importlib.import_module("vllm")
        factory = llm_factory or self.vllm.LLM
        kwargs: dict[str, Any] = dict(
            model=model_path, dtype=str(cfg.get("engine.dtype")), tensor_parallel_size=tensor_parallel_size,
            max_model_len=int(cfg.get("engine.max_model_len")),
            gpu_memory_utilization=float(cfg.get("engine.gpu_memory_utilization")),
            enable_prefix_caching=bool(cfg.get("engine.enable_prefix_caching")),
            enforce_eager=bool(cfg.get("engine.enforce_eager")), max_num_seqs=int(cfg.get("engine.max_num_seqs")),
        )
        if quantization:
            kwargs["quantization"] = quantization
        self._factory, self._kwargs = factory, kwargs
        self.llm = factory(**kwargs)
        self.model_path, self.structured_keyword = model_path, None
        self.system_ok = cfg.get("engine.system_role_supported")
        self.tokenizer = self.llm.get_tokenizer()

    def describe(self) -> dict[str, Any]:
        return {"engine": self.name, "model": self.model_path, "vllm": getattr(self.vllm, "__version__", None),
                "structured_keyword": self.structured_keyword, "system_role_supported": self.system_ok}

    def _prompt(self, request: GenRequest) -> str:
        messages = [{"role": "user", "content": request.prompt}]
        if request.system:
            with_system = [{"role": "system", "content": request.system}, *messages]
            if self.system_ok is not False:
                try:
                    text = self.tokenizer.apply_chat_template(with_system, tokenize=False, add_generation_prompt=True)
                    self.system_ok = True
                    return text
                except Exception:  # the template refuses a system turn: merge it into the user turn
                    self.system_ok = False
            messages = [{"role": "user", "content": f"{request.system}\n\n{request.prompt}"}]
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    @staticmethod
    def _seed(seed: int | None) -> int | None:
        return None if seed is None else int(seed) & ((1 << SEED_BITS) - 1)

    def _sampling(self, params: GenParams, request: GenRequest) -> Any:
        kwargs: dict[str, Any] = dict(
            n=request.n, temperature=params.temperature, top_p=params.top_p, max_tokens=params.max_tokens,
            seed=self._seed(request.seed if request.seed is not None else params.seed),
        )
        if params.top_k is not None:
            kwargs["top_k"] = params.top_k
        if params.json_schema:
            extra, self.structured_keyword = structured_kwargs(self.vllm, dict(params.json_schema))
            kwargs.update(extra)
        return self.vllm.SamplingParams(**kwargs)

    def _engine_died(self, exc: BaseException) -> bool:
        names = set(self.cfg.get("engine.restart_error_names"))
        return any(cls.__name__ in names for cls in type(exc).__mro__) or "EngineCore" in str(exc)

    def restart(self) -> None:
        """Shut the dead engine down, free the GPU and build a new one with the same settings."""
        try:
            self.llm.llm_engine.engine_core.shutdown()
        except Exception:  # already gone
            pass
        del self.llm
        import gc

        gc.collect()
        try:
            import torch

            torch.cuda.empty_cache()
        except Exception:  # no CUDA in this process (tests, CPU)
            pass
        self.llm = self._factory(**self._kwargs)
        self.tokenizer = self.llm.get_tokenizer()

    def _generate_with_restart(self, prompts: list[str], sampling: list[Any]):
        """``llm.generate``, restarting the engine (up to ``engine.max_restarts`` times) if its process died; the batch is then redone."""
        for attempt in range(int(self.cfg.get("engine.max_restarts")) + 1):
            try:
                return self.llm.generate(prompts, sampling, use_tqdm=False)
            except Exception as exc:
                if not self._engine_died(exc) or attempt >= int(self.cfg.get("engine.max_restarts")):
                    raise
                self.restarts = getattr(self, "restarts", 0) + 1
                self.restart()

    def _n_tokens(self, text: str) -> int | None:
        encode = getattr(self.tokenizer, "encode", None)
        return len(encode(text, add_special_tokens=False)) if encode else None

    def generate(self, requests: Sequence[GenRequest], params: GenParams) -> list[GenResult]:
        """One result per request. A prompt that cannot fit the context with its answer is returned empty (``prompt_too_long``)
        instead of making vLLM reject, or even kill, the whole batch."""
        prompts = [self._prompt(r) for r in requests]
        room = int(self.cfg.get("engine.max_model_len")) - int(params.max_tokens)
        lengths = [self._n_tokens(p) for p in prompts]
        fits = [n is None or n <= room for n in lengths]
        run = [i for i, ok in enumerate(fits) if ok]
        results: list[GenResult | None] = [None] * len(requests)
        if run:
            sampling = [self._sampling(params, requests[i]) for i in run]
            outputs = self._generate_with_restart([prompts[i] for i in run], sampling)
            for i, output in zip(run, outputs, strict=True):
                texts = tuple(o.text for o in output.outputs)
                tokens_out = sum(len(o.token_ids) for o in output.outputs)
                results[i] = GenResult(
                    requests[i].unit_id, texts, len(output.prompt_token_ids), tokens_out,
                    str(output.outputs[0].finish_reason) if output.outputs else "none",
                    nan_suspect=bool(requests[i].prompt) and not any(t.strip() for t in texts),
                )
        for i, ok in enumerate(fits):
            if not ok:
                results[i] = GenResult(requests[i].unit_id, ("",), int(lengths[i] or 0), 0, PROMPT_TOO_LONG, nan_suspect=False)
        return [r for r in results if r is not None]

    def close(self) -> None:
        del self.llm
