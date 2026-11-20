"""A mock reader answer with a planted dependence on the text, for end-to-end CPU tests.

The mock reads the ``Text:`` and ``Allowed assets:`` parts of the user prompt, so downstream code
(dials, sheaf, portfolio) has a real signal to find. It is never used outside tests and smoke runs.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np

REF_LINE = re.compile(r"^(A[0-9]+):", re.MULTILINE)
CAUSE_LINE = re.compile(r"^Cause list: (.*)$", re.MULTILINE)
POSITIVE_WORDS = ("beat", "raises", "wins", "record", "upgrade")
NEGATIVE_WORDS = ("miss", "cuts", "recall", "lawsuit", "downgrade")
EFFECT_PER_WORD = 3.0
NOISE_SCALE = 0.5
MAX_MAGNITUDE_WORDS = 5
EFFECT_DECIMALS = 3
PROB_DECIMALS = 6
CERTAINTY_DECIMALS = 4


def reader_responder(prompt: str, rng: np.random.Generator) -> Mapping[str, Any]:
    """Deterministic JSON reading: tone words move the value effect of every allowed asset."""
    causes = [c.strip() for c in CAUSE_LINE.search(prompt).group(1).split(",")]
    text = prompt.split("Text:", 1)[-1].lower()
    tone = sum(w in text for w in POSITIVE_WORDS) - sum(w in text for w in NEGATIVE_WORDS)
    probs = rng.dirichlet(np.ones(len(causes)))
    refs = REF_LINE.findall(prompt)
    pins = []
    for position, ref in enumerate(refs):
        effect = tone * EFFECT_PER_WORD + float(rng.normal(0.0, NOISE_SCALE))
        pins.append(
            {
                "ref": ref,
                "direction": "up" if effect > 0 else "down" if effect < 0 else "none",
                "value_effect_pct": round(effect, EFFECT_DECIMALS),
                "magnitude": min(abs(tone), MAX_MAGNITUDE_WORDS),
                "role": "source" if position == 0 else "linked",
            }
        )
    return {
        "cause": {c: round(float(p), PROB_DECIMALS) for c, p in zip(causes, probs, strict=True)},
        "pins": pins,
        "scheduled": "unknown",
        "certainty": round(float(rng.random()), CERTAINTY_DECIMALS),
    }
