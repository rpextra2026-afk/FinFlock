"""Language-model engine interface. All pipeline code talks to ``Engine`` and nothing else.

``MockEngine`` is deterministic and has no GPU dependency; it lets every stage run on a laptop.
The vLLM and HTTP-server engines are added after the Day-0 VERIFY of the vLLM version, structured
output parameter name and attention backend (experiment configuration); until then asking for them fails
loudly instead of guessing an API.
"""

from __future__ import annotations

import hashlib
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from oryxis.common.hashing import canonical_json
from oryxis.common.rng import derive_seed

DEFAULT_TASK = "generic"
MOCK_ENGINE_NAME = "mock"
FINISH_STOP = "stop"


@dataclass(frozen=True)
class GenRequest:
    """One generation request: ``n`` independent draws for one prompt."""

    unit_id: str
    prompt: str
    n: int = 1
    task: str = DEFAULT_TASK
    system: str | None = None
    seed: int | None = None  # per-request seed derived from the unit identity


@dataclass(frozen=True)
class GenParams:
    """Sampling parameters. Every value comes from config at the call site."""

    temperature: float
    top_p: float
    top_k: int | None
    max_tokens: int
    json_schema: Mapping[str, Any] | None = None
    seed: int | None = None


@dataclass(frozen=True)
class GenResult:
    unit_id: str
    texts: tuple[str, ...]
    tokens_in: int
    tokens_out: int
    finish: str = FINISH_STOP
    nan_suspect: bool = False  # empty output for a non-empty prompt: the signature of NaN / inf logits


class Engine(ABC):
    """Interface of every language-model backend."""

    name: str

    @abstractmethod
    def generate(self, requests: Sequence[GenRequest], params: GenParams) -> list[GenResult]:
        """Generate for each request; results come back in request order."""

    def close(self) -> None:  # noqa: B027 - optional hook
        """Release GPU memory or servers; the default does nothing."""


Responder = Callable[[str, np.random.Generator], Mapping[str, Any]]


def _default_responder(prompt: str, rng: np.random.Generator) -> Mapping[str, Any]:
    """Generic mock answer: one number derived from the prompt and the draw."""
    return {"value": float(rng.random())}


def count_tokens(text: str) -> int:
    """Whitespace token count used by the mock engine and by dry-run budgeting."""
    return len(text.split())


@dataclass
class MockEngine(Engine):
    """Deterministic JSON from ``sha256(prompt)``, with task-specific responders for tests.

    A responder receives the prompt text and a generator seeded from (prompt, draw), so it can plant
    a dependence of the answer on the text for downstream code to find.
    """

    name: str = MOCK_ENGINE_NAME
    responders: dict[str, Responder] = field(default_factory=dict)

    def register(self, task: str, responder: Responder) -> None:
        self.responders[task] = responder

    def generate(self, requests: Sequence[GenRequest], params: GenParams) -> list[GenResult]:
        results: list[GenResult] = []
        for request in requests:
            responder = self.responders.get(request.task, _default_responder)
            digest = hashlib.sha256(
                f"{request.system or ''}\x1f{request.prompt}".encode("utf-8")
            ).hexdigest()
            texts = []
            for draw in range(request.n):
                rng = np.random.Generator(
                    np.random.PCG64(derive_seed(digest, draw, request.seed if request.seed is not None else (params.seed or 0)))
                )
                texts.append(canonical_json(dict(responder(request.prompt, rng))))
            results.append(
                GenResult(
                    unit_id=request.unit_id,
                    texts=tuple(texts),
                    tokens_in=count_tokens(request.prompt) + count_tokens(request.system or ""),
                    tokens_out=sum(count_tokens(t) for t in texts),
                )
            )
        return results


def get_engine(name: str) -> Engine:
    """Engine factory. Real backends are intentionally not guessed before Day-0 verification."""
    if name == MOCK_ENGINE_NAME:
        return MockEngine()
    raise NotImplementedError(
        f"engine '{name}' is not implemented yet: it is written after the Day-0 VERIFY items "
        "(vLLM version, structured-output parameter, attention backend) are recorded in experiment configuration"
    )
