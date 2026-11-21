"""Seeded random number generators. No module may use global RNG state.

Seeds are derived from the identity of the unit of work, e.g. ``(bucket, llm, event_id, persona_id,
draw)``, so that a unit gets the same random numbers however the work is split into shards.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

SEED_BYTES = 8


def derive_seed(*parts: Any) -> int:
    """64-bit seed from the first ``SEED_BYTES`` bytes of sha256 of ``str(parts)``.

    Parts must have a stable ``repr`` (strings, ints, tuples of those). Floats and objects are
    rejected so that a seed can never depend on formatting.
    """
    for part in parts:
        if not isinstance(part, (str, int, tuple)):
            raise TypeError(f"seed parts must be str, int or tuple, got {type(part).__name__}")
    digest = hashlib.sha256(str(tuple(parts)).encode("utf-8")).digest()[:SEED_BYTES]
    return int.from_bytes(digest, "big")


def make_rng(*parts: Any) -> np.random.Generator:
    """A fresh PCG64 generator seeded from the identity ``parts``."""
    return np.random.Generator(np.random.PCG64(derive_seed(*parts)))


class SeedFactory:
    """Derives named sub-seeds from one base seed (``runtime.base_seed``)."""

    def __init__(self, base_seed: int):
        self.base_seed = int(base_seed)

    def seed(self, *parts: Any) -> int:
        return derive_seed(self.base_seed, *parts)

    def rng(self, *parts: Any) -> np.random.Generator:
        return make_rng(self.base_seed, *parts)
