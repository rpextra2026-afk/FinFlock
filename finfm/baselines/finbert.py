"""N2: FinBERT tone P(positive) - P(negative), CPU, every score cached by the sha256 of its text (a rerun costs nothing)."""

from __future__ import annotations

import hashlib
from collections.abc import Callable, Iterable
from pathlib import Path

import pandas as pd

KEY, VALUE = "text_sha256", "tone"


def text_hash(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def load_model(name: str):
    """(tokenizer, model, index of positive, index of negative); the labels are read from the model's own config."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tokenizer, model = AutoTokenizer.from_pretrained(name), AutoModelForSequenceClassification.from_pretrained(name).eval()
    labels = {str(v).lower(): int(k) for k, v in model.config.id2label.items()}
    return tokenizer, model, labels["positive"], labels["negative"]


def make_scorer(name: str, max_length: int, batch_size: int) -> Callable[[list[str]], list[float]]:
    import torch

    tokenizer, model, pos, neg = load_model(name)

    def score(texts: list[str]) -> list[float]:
        out: list[float] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            enc = tokenizer(batch, padding=True, truncation=True, max_length=max_length, return_tensors="pt")
            with torch.no_grad():
                probs = torch.softmax(model(**enc).logits, dim=-1)
            out += [float(p[pos] - p[neg]) for p in probs]
        return out

    return score


def finbert_tone(texts: Iterable[str], cache_path: Path, scorer: Callable[[list[str]], list[float]]) -> pd.Series:
    """Scores for ``texts`` in order. Only texts missing from the cache go to ``scorer``; the cache is written back."""
    texts = [str(t) for t in texts]
    cache = pd.read_parquet(cache_path).set_index(KEY)[VALUE].to_dict() if cache_path.exists() else {}
    hashes = [text_hash(t) for t in texts]
    todo = {h: t for h, t in zip(hashes, texts, strict=True) if h not in cache}
    if todo:
        for h, value in zip(todo, scorer(list(todo.values())), strict=True):
            cache[h] = value
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame({KEY: list(cache), VALUE: list(cache.values())}).to_parquet(cache_path, index=False)
    return pd.Series([cache[h] for h in hashes], dtype=float)
