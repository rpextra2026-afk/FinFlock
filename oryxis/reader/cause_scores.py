"""Full-label conditional log probabilities; no truncated vocabulary approximation."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np


@dataclass(frozen=True)
class Continuation:
    token_ids: list[int]
    start: int
    boundary_merge: bool
    label_tokens: int


def continuation_tokens(tokenizer, prompt, label):
    before = list(tokenizer.encode(prompt, add_special_tokens=False))
    after = list(tokenizer.encode(prompt + label, add_special_tokens=False))
    common = 0
    for a, b in zip(before, after):
        if a != b:
            break
        common += 1
    return Continuation(after, common, common < len(before), len(after) - common)


def sum_label_logprobs(tokens, prompt_logprobs):
    values = []
    for index in range(tokens.start, len(tokens.token_ids)):
        if index == 0 or index >= len(prompt_logprobs) or prompt_logprobs[index] is None:
            raise ValueError('missing_label_logprob')
        item = prompt_logprobs[index].get(tokens.token_ids[index])
        if item is None:
            raise ValueError('missing_label_logprob')
        value = float(getattr(item, 'logprob', item))
        if math.isnan(value) or value == math.inf:
            raise ValueError('nonfinite_label_logprob')
        values.append(value)
    if not values:
        raise ValueError('empty_label_tokens')
    return sum(values)


def score_distribution(logps, groups):
    labels = list(logps)
    values = np.asarray([logps[c] for c in labels], dtype=float)
    if np.isnan(values).any() or np.isposinf(values).any() or not np.isfinite(values).any():
        raise ValueError('invalid_label_scores')
    weights = np.exp(values - np.max(values))
    weights /= weights.sum()
    probs = dict(zip(labels, map(float, weights)))
    positive = weights[weights > 0]
    return probs, {g: sum(probs[c] for c in cs) for g, cs in groups.items()}, float(-np.sum(positive * np.log(positive)))
