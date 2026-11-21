"""Masking of company names, former names, tickers and cashtags (spec 4.2, 12.2 #7).

Masking is a leakage control, never a fix: its failure rate is measured by the masked-text probe.
Companies are replaced by ``COMPANY_A``, ``COMPANY_B`` ... in order of first appearance. Countries,
commodities, agencies and currencies are not masked. Tickers of at most ``short_ticker_max_len``
letters are masked only as cashtags (``$AB``), because they collide with ordinary words.
"""

from __future__ import annotations

import re
import string
from dataclasses import dataclass

import pandas as pd

ALIAS_NODE = "node_id"
ALIAS_TEXT = "alias"
ALIAS_TYPE = "alias_type"
ALIAS_FROM = "valid_from"
ALIAS_TO = "valid_to"
TYPE_NAME = "name"
TYPE_FORMER = "former_name"
TYPE_TICKER = "ticker"
NAME_TYPES = (TYPE_NAME, TYPE_FORMER)
CASHTAG = "$"
WORD_CHARS = r"A-Za-z0-9"
LETTERS = string.ascii_uppercase


def label_for(index: int, prefix: str) -> str:
    """``COMPANY_A`` ... ``COMPANY_Z``, ``COMPANY_AA`` ... (spreadsheet style)."""
    letters = ""
    n = index
    while True:
        letters = LETTERS[n % len(LETTERS)] + letters
        n = n // len(LETTERS) - 1
        if n < 0:
            return prefix + letters


@dataclass(frozen=True)
class MaskResult:
    text: str
    mapping: dict[str, str]  # node_id -> label, in order of first appearance
    n_replaced: int


@dataclass(frozen=True)
class _Hit:
    start: int
    end: int
    node_id: str


class Masker:
    """Masks every alias valid at any time in ``[window_start, window_end]`` (point-in-time)."""

    def __init__(
        self,
        aliases: pd.DataFrame,
        window_start: pd.Timestamp,
        window_end: pd.Timestamp,
        *,
        label_prefix: str,
        short_ticker_max_len: int,
    ):
        self.label_prefix = label_prefix
        active = aliases[
            (aliases[ALIAS_FROM].isna() | (pd.to_datetime(aliases[ALIAS_FROM]) <= window_end))
            & (aliases[ALIAS_TO].isna() | (pd.to_datetime(aliases[ALIAS_TO]) >= window_start))
        ]
        names: dict[str, str] = {}
        tickers: dict[str, str] = {}
        cashtags: dict[str, str] = {}
        self.conflicts = 0
        for node, text, kind in sorted(
            zip(active[ALIAS_NODE], active[ALIAS_TEXT], active[ALIAS_TYPE], strict=True)
        ):
            text = str(text).strip()
            if not text:
                continue
            if kind in NAME_TYPES:
                key = text.casefold()
                if names.setdefault(key, node) != node:
                    self.conflicts += 1
            elif kind == TYPE_TICKER:
                key = text.upper()
                cashtags.setdefault(key, node)
                if len(key) > short_ticker_max_len:
                    if tickers.setdefault(key, node) != node:
                        self.conflicts += 1
        self._names, self._tickers, self._cashtags = names, tickers, cashtags
        self._name_re = self._compile(names, ignore_case=True)
        self._ticker_re = self._compile(tickers, ignore_case=False)
        self._cashtag_re = self._compile(cashtags, ignore_case=True, prefix=re.escape(CASHTAG))

    @staticmethod
    def _compile(table: dict[str, str], *, ignore_case: bool, prefix: str = "") -> re.Pattern | None:
        if not table:
            return None
        body = "|".join(re.escape(k) for k in sorted(table, key=len, reverse=True))
        left = "" if prefix else rf"(?<![{WORD_CHARS}$])"
        pattern = rf"{left}{prefix}(?:{body})(?![{WORD_CHARS}])"
        return re.compile(pattern, re.IGNORECASE if ignore_case else 0)

    def _hits(self, text: str) -> list[_Hit]:
        hits: list[_Hit] = []
        if self._name_re:
            hits += [
                _Hit(m.start(), m.end(), self._names[m.group(0).casefold()])
                for m in self._name_re.finditer(text)
            ]
        if self._ticker_re:
            hits += [
                _Hit(m.start(), m.end(), self._tickers[m.group(0)])
                for m in self._ticker_re.finditer(text)
            ]
        if self._cashtag_re:
            hits += [
                _Hit(m.start(), m.end(), self._cashtags[m.group(0)[len(CASHTAG) :].upper()])
                for m in self._cashtag_re.finditer(text)
            ]
        hits.sort(key=lambda h: (h.start, -(h.end - h.start)))
        kept: list[_Hit] = []
        last_end = -1
        for hit in hits:  # drop overlaps, keeping the earliest and longest
            if hit.start >= last_end:
                kept.append(hit)
                last_end = hit.end
        return kept

    def find(self, text: str) -> list[str]:
        """Node ids named in ``text`` (names, former names, tickers, cashtags), in order of first appearance."""
        seen: dict[str, None] = {}
        for hit in self._hits(text):
            seen.setdefault(hit.node_id, None)
        return list(seen)

    def mask(self, text: str) -> MaskResult:
        hits = self._hits(text)
        mapping: dict[str, str] = {}
        for hit in hits:
            if hit.node_id not in mapping:
                mapping[hit.node_id] = label_for(len(mapping), self.label_prefix)
        out, cursor = [], 0
        for hit in hits:
            out.append(text[cursor : hit.start])
            out.append(mapping[hit.node_id])
            cursor = hit.end
        out.append(text[cursor:])
        return MaskResult("".join(out), mapping, len(hits))


class MaskerCache:
    """One masker per calendar month, built lazily (aliases change slowly)."""

    def __init__(self, aliases: pd.DataFrame, *, label_prefix: str, short_ticker_max_len: int):
        self._aliases = aliases
        self._prefix = label_prefix
        self._short = short_ticker_max_len
        self._cache: dict[pd.Period, Masker] = {}

    def for_date(self, date: pd.Timestamp) -> Masker:
        key = pd.Period(date, freq="M")
        if key not in self._cache:
            self._cache[key] = Masker(
                self._aliases,
                key.start_time.normalize(),
                key.end_time.normalize(),
                label_prefix=self._prefix,
                short_ticker_max_len=self._short,
            )
        return self._cache[key]
