"""R1, finripple: FinRipple-lite. The paper's zero-shot-with-graph-text variant, no reinforcement learning.

The full FinRipple needs more than 9,000 GPU-hours and has no confirmed code release; claims are made
against this lite variant only. The knowledge-graph text is our rendering from the point-in-time link
snapshots (the paper's graphs are patent, supply-chain and shared-leadership graphs), stated as such.
"""

from __future__ import annotations

import ast
import json
import re
import string
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

from finflock.common.config import Config
from finflock.common.rng import make_rng
from finflock.common.schema import DETAIL, PARSE_OK, REASON, STATUS, TOKENS_IN, TOKENS_OUT, UNIT_ID, Reason, Status
from finflock.compute.engine import Engine, GenParams, GenRequest
from finflock.data.f13 import normalise_name
from finflock.reader.prompt import Tokenizer
from finflock.road1.graph import LINK_DST, LINK_SRC, LINK_TYPE, LINK_WEIGHT, available_links

TASK = "finripple"
NEWS = "news"
KG = "kg_text"
RAW = "raw_output"
AFFECTED = "affected_json"
ARROW = "--{relation}-->"


LEGACY, BALANCED = "legacy", "balanced"
ANCHOR = "anchor"
EDGE_KEY = [LINK_SRC, LINK_DST, LINK_TYPE]


def _edge_text(table: pd.DataFrame, labels: Mapping[str, str]) -> list[str]:
    return [f"{labels.get(r.src_node, r.src_node)} {ARROW.format(relation=r.link_type)} {labels.get(r.dst_node, r.dst_node)}"
            for r in table.itertuples()]


def kg_lines(
    seeds: Sequence[str], links: pd.DataFrame, labels: Mapping[str, str], asof: pd.Timestamp, cfg: Config,
    *, order: str = BALANCED,
) -> list[str]:
    """Edges within ``hops`` of the named companies as ``NAME (TICKER) --relation--> NAME (TICKER)``.

    ``balanced`` (default): first-hop edges come before second-hop edges, and inside a hop the edges are dealt out round-robin over
    (relation, named company), strongest first. With a token budget that cuts the list, every named company and every relation is
    seen. ``legacy`` is the first version (all hops pooled, strongest first, ties alphabetical), kept to reproduce old tables: it
    let one alphabetical company's industry edges (weight 1.0 for all) fill the whole budget.
    """
    spec = cfg.get("rivals.finripple")
    known = available_links(links, asof)
    known = known[known[LINK_TYPE].isin(set(spec["relations"]))]
    frontier, seen_nodes = set(seeds), set(seeds)
    hops = []
    for _ in range(int(spec["hops"])):
        touching = known[known[LINK_SRC].isin(frontier) | known[LINK_DST].isin(frontier)]
        hops.append((touching, frozenset(frontier)))
        frontier = (set(touching[LINK_SRC]) | set(touching[LINK_DST])) - seen_nodes
        seen_nodes |= frontier
    by_weight = [LINK_WEIGHT, LINK_SRC, LINK_DST]
    if order == LEGACY:
        table = pd.concat([t for t, _ in hops]).drop_duplicates(EDGE_KEY).sort_values(by_weight, ascending=[False, True, True], kind="mergesort")
        return _edge_text(table, labels)
    parts, taken = [], set()
    for touching, anchors in hops:
        unseen = np.array([key not in taken for key in zip(touching[LINK_SRC], touching[LINK_DST], touching[LINK_TYPE], strict=True)], dtype=bool)
        fresh = touching.loc[unseen]                     # a boolean row mask (a plain empty list would select columns)
        fresh = fresh.drop_duplicates(EDGE_KEY).sort_values(by_weight, ascending=[False, True, True], kind="mergesort")
        taken |= set(zip(fresh[LINK_SRC], fresh[LINK_DST], fresh[LINK_TYPE], strict=True))
        fresh = fresh.assign(**{ANCHOR: [s if s in anchors else d for s, d in zip(fresh[LINK_SRC], fresh[LINK_DST], strict=True)]})
        fresh["deal"] = fresh.groupby([LINK_TYPE, ANCHOR], sort=False).cumcount()
        parts.append(fresh.sort_values(["deal", LINK_WEIGHT, LINK_SRC, LINK_DST, LINK_TYPE], ascending=[True, False, True, True, True], kind="mergesort"))
    return _edge_text(pd.concat(parts), labels) if parts else []


def random_lines(links: pd.DataFrame, labels: Mapping[str, str], asof: pd.Timestamp, cfg: Config, seed_parts: tuple) -> list[str]:
    """Random edges (hashed, seeded) when the event names no company, as the paper does."""
    known = available_links(links, asof)
    known = known[known[LINK_TYPE].isin(set(cfg.get("rivals.finripple.relations")))]
    if known.empty:
        return []
    order = make_rng(*seed_parts).permutation(len(known))
    return [f"{labels.get(r.src_node, r.src_node)} {ARROW.format(relation=r.link_type)} {labels.get(r.dst_node, r.dst_node)}"
            for r in known.iloc[order].itertuples()]


def kg_text(lines: Sequence[str], tokenizer: Tokenizer, max_tokens: int) -> str:
    text, _ = tokenizer.truncate("\n".join(lines), max_tokens)
    return text


def parse_impacts(text: str, cfg: Config) -> list[dict] | None:
    """Affected companies from the answer: strict JSON first, then a literal parse (models often copy the
    single-quoted example). ``None`` means refusal or malformed output (counted as the refusal rate)."""
    bound = float(cfg.get("rivals.finripple.score_bound"))
    data = None
    for body in _candidate_bodies(text):
        for loader in (json.loads, ast.literal_eval):
            try:
                data = loader(body)
                break
            except (ValueError, SyntaxError, TypeError, RecursionError, MemoryError):
                data = None
        if isinstance(data, dict):
            break
    if not isinstance(data, dict):
        return None
    analysis = data.get("impact_analysis")
    companies = analysis.get("affected_companies") if isinstance(analysis, dict) else None
    if not isinstance(companies, list):
        return None
    out = []
    for item in companies:
        try:
            score = float(item["impact_score"])
            if not np.isfinite(score):
                continue
            out.append({"name": str(item["name"]), "impact_type": str(item.get("impact_type", "")),
                        "score": float(np.clip(score, -bound, bound))})
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return out


WHITESPACE_AND_COMMA = string.whitespace + ","


ENTRY = re.compile(
    r"""['"]name['"]\s*:\s*(['"])(?P<name>.*?)\1\s*,\s*['"]impact_type['"]\s*:\s*(['"])(?P<kind>.*?)\3\s*,\s*['"]impact_score['"]\s*:\s*(?P<score>[-+]?\d+(?:\.\d+)?)""",
    re.DOTALL)
CUT, QUOTE_STYLE = "cut", "quote_style"


def quote_style_entries(text: str, cfg: Config) -> list[dict] | None:
    """The company entries of an answer written with Python-style quotes whose strings hold unescaped apostrophes (so it is neither JSON nor a Python literal).

    An entry is ``name``, ``impact_type`` and ``impact_score`` in the order the prompt gives, in either quote style; the closing quote of a string is
    the one followed by the next key, so an apostrophe inside a name or an analysis does not end it. Nothing is repaired or invented: only values the model
    wrote are read, scores are clipped to the bound as everywhere. ``None`` when no complete entry is found.
    """
    bound = float(cfg.get("rivals.finripple.score_bound"))
    out = [{"name": m.group("name"), "impact_type": m.group("kind"), "score": float(np.clip(float(m.group("score")), -bound, bound))} for m in ENTRY.finditer(text)]
    return out or None


def recover_answer(text: str, cfg: Config) -> tuple[list[dict] | None, str | None]:
    """``(entries, how)``: ``how`` is ``cut`` when the complete entries before a cut-off were read as JSON, ``quote_style`` when only the tolerant reading
    (``cpu.score_finripple.salvage_quote_style``) found entries, ``None`` when there is nothing."""
    items = recover_entries(text, cfg, quote_style=False)
    if items:
        return items, CUT
    if cfg.get("cpu.score_finripple.salvage_quote_style"):
        items = quote_style_entries(text, cfg)
        if items:
            return items, QUOTE_STYLE
    return None, None


def recover_entries(text: str, cfg: Config, quote_style: bool = True) -> list[dict] | None:
    """Complete company entries that precede a cut-off in an answer that does not parse as a whole (output stopped at the token limit).

    Walks the ``affected_companies`` array object by object and keeps every complete one; stops at the first incomplete object. ``None``
    when nothing complete is found. CPU-side only (the scorer reports these answers separately from strictly parsed ones). With ``quote_style``
    (and ``cpu.score_finripple.salvage_quote_style``) an answer the walk cannot read is read by ``quote_style_entries``.
    """
    items = _walk_entries(text, cfg)
    if items or not quote_style or not cfg.get("cpu.score_finripple.salvage_quote_style"):
        return items
    return quote_style_entries(text, cfg)


def _walk_entries(text: str, cfg: Config) -> list[dict] | None:
    bound = float(cfg.get("rivals.finripple.score_bound"))
    key = text.find('"affected_companies"')
    start = text.find("[", key) if key >= 0 else -1
    if start < 0:
        return None
    decoder, position, out = json.JSONDecoder(), start + 1, []
    while position < len(text):
        while position < len(text) and text[position] in WHITESPACE_AND_COMMA:
            position += 1
        if position >= len(text) or text[position] != "{":
            break
        try:
            item, position = decoder.raw_decode(text, position)
        except ValueError:
            break
        try:
            score = float(item["impact_score"])
            if np.isfinite(score):
                out.append({"name": str(item["name"]), "impact_type": str(item.get("impact_type", "")), "score": float(np.clip(score, -bound, bound))})
        except (KeyError, TypeError, ValueError, AttributeError):
            continue
    return out or None


FENCE = re.compile(r"^```[A-Za-z]*\s*(.*?)\s*```\s*$", re.DOTALL)


def _candidate_bodies(text: str) -> list[str]:
    """The answer as written, without a markdown code fence, and cut to the outermost braces (text before or after the JSON)."""
    body = text.strip()
    fenced = FENCE.match(body)
    bodies = [body] + ([fenced.group(1)] if fenced else [])
    first, last = body.find("{"), body.rfind("}")
    if 0 <= first < last:
        bodies.append(body[first:last + 1])
    return bodies


class FinRippleProcessor:
    """Worker processor: units are events with columns ``news`` and ``kg_text``."""

    def __init__(self, cfg: Config, engine: Engine) -> None:
        self.cfg, self.engine = cfg, engine
        spec = cfg.get("rivals.finripple")
        self.params = GenParams(temperature=float(spec["temperature"]), top_p=1.0, top_k=None,
                                max_tokens=int(spec["max_tokens"]), seed=int(cfg.get("runtime.base_seed")))

    def __call__(self, batch: pd.DataFrame) -> pd.DataFrame:
        template = str(self.cfg.get("prompts.finripple.user"))
        requests = [GenRequest(str(r[UNIT_ID]), template.format(news=r[NEWS], kg_text=r[KG]), task=TASK)
                    for _, r in batch.iterrows()]
        rows = []
        for result in self.engine.generate(requests, self.params):
            impacts = parse_impacts(result.texts[0], self.cfg)
            rows.append({UNIT_ID: result.unit_id, STATUS: Status.OK.value,
                         REASON: None if impacts is not None else Reason.PARSE_FAILED.value, DETAIL: None,
                         PARSE_OK: impacts is not None, AFFECTED: json.dumps(impacts or [], sort_keys=True),
                         RAW: result.texts[0], TOKENS_IN: result.tokens_in, TOKENS_OUT: result.tokens_out})
        return pd.DataFrame(rows)


def impacts_to_signals(
    tape: pd.DataFrame, decision_dates: Mapping[str, pd.Timestamp], index: Mapping[str, str], cfg: Config,
) -> pd.DataFrame:
    """Signal rows ``(decision_date, security_id, score)``: affected companies mapped back to S&P securities
    by normalised name; names that do not map are dropped and counted by the caller."""
    suffixes = set(cfg.get("f13.name_match.suffixes"))
    rows = []
    for record in tape[tape[PARSE_OK].fillna(False).astype(bool)].itertuples():
        for item in json.loads(getattr(record, AFFECTED)):
            security = index.get(normalise_name(item["name"], suffixes))
            if security is not None:
                rows.append({"decision_date": decision_dates[getattr(record, UNIT_ID)], "security_id": security,
                             "score": item["score"], "event_id": getattr(record, UNIT_ID)})
    return pd.DataFrame(rows, columns=["decision_date", "security_id", "score", "event_id"])
