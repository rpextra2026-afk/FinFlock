"""A progress line for long loops: one line at most every ``seconds`` seconds (the owner's rule: new scripts report what they are doing at a fixed interval)."""

from __future__ import annotations

import time


class Heartbeat:
    def __init__(self, label: str, total: int, seconds: float) -> None:
        self.label, self.total, self.seconds = label, total, seconds
        self.began = self.last = time.time()

    def tick(self, done: int, extra: str = "") -> None:
        now = time.time()
        if now - self.last >= self.seconds:
            self.last = now
            print(f"[progress] {self.label}: {done}/{self.total} after {now - self.began:.0f} s {extra}".rstrip(), flush=True)
