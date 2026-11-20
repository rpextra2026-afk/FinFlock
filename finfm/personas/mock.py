"""Mock persona answer with planted dependences on the feed and the persona, for CPU tests only."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import numpy as np

REF_LINE = re.compile(r"^(A[0-9]+):", re.MULTILINE)
POSITIVE_WORDS = ("beat", "raises", "wins", "record", "upgrade")
NEGATIVE_WORDS = ("miss", "cuts", "recall", "lawsuit", "downgrade")
VALUE_PER_WORD = 2.5
VALUE_NOISE = 0.8
EMOTION_NOISE = 0.5
BASE_EMOTION = 3.0
FEAR_PER_NEGATIVE_WORD = 1.5
DECIMALS = 3
SCALE_MAX = 10.0
BASE_CONVICTION = 5.0
CONVICTION_NOISE = 1.0


def persona_responder(prompt: str, rng: np.random.Generator) -> Mapping[str, Any]:
    """Tone words move the valuation; negative words raise fear; the rest is draw noise."""
    text = prompt.lower()
    positive = sum(w in text for w in POSITIVE_WORDS)
    negative = sum(w in text for w in NEGATIVE_WORDS)
    values = {
        ref: round(float((positive - negative) * VALUE_PER_WORD + rng.normal(0.0, VALUE_NOISE)), DECIMALS)
        for ref in REF_LINE.findall(prompt)
    }
    fear = BASE_EMOTION + FEAR_PER_NEGATIVE_WORD * negative + rng.normal(0.0, EMOTION_NOISE)
    return {
        "fear": round(float(min(max(fear, 0.0), SCALE_MAX)), DECIMALS),
        "anger": round(float(min(max(BASE_EMOTION + rng.normal(0.0, EMOTION_NOISE), 0.0), SCALE_MAX)), DECIMALS),
        "calm": round(float(min(max(BASE_EMOTION + rng.normal(0.0, EMOTION_NOISE), 0.0), SCALE_MAX)), DECIMALS),
        "value_change_pct": values,
        "conviction": round(float(min(max(BASE_CONVICTION + rng.normal(0.0, CONVICTION_NOISE), 0.0), SCALE_MAX)), DECIMALS),
    }
