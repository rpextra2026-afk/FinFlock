"""Building the reader prompt: pin selection, masking, truncation, asset lines (spec 4.2, 4.4)."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol

from finfm.common.config import Config
from finfm.common.hashing import sha256_hex
from finfm.reader.masking import MaskResult, Masker

FIELD_SEPARATOR = "\n\x1f\n"
NODE_COMPANY = "company"
PIN_NODE_ID = "node_id"
PIN_NODE_TYPE = "node_type"
PIN_SECTOR = "sector"
PIN_LAYER = "layer"
PIN_PRIORITY = "priority"


class Tokenizer(Protocol):
    def truncate(self, text: str, max_tokens: int) -> tuple[str, bool]:
        """First ``max_tokens`` tokens of ``text`` and whether anything was cut."""


class WhitespaceTokenizer:
    """Whitespace tokens, for the mock engine and for CPU tests. Real runs use the model tokenizer."""

    def truncate(self, text: str, max_tokens: int) -> tuple[str, bool]:
        parts = text.split(" ")
        if len(parts) <= max_tokens:
            return text, False
        return " ".join(parts[:max_tokens]), True


@dataclass(frozen=True)
class BuiltPrompt:
    system: str
    user: str
    ref_to_node: dict[str, str]
    n_pins_given: int
    n_pins_dropped: int
    text_truncated: bool
    mask: MaskResult

    @property
    def template_hash(self) -> str:
        return sha256_hex(self.system + "\x1f" + self.user.split("Allowed assets:")[0])


def ref_to_node_of(pins_json: str, cfg: Config) -> dict[str, str]:
    """``A1 -> node id`` of the pins offered to the model for one event (what ``build_reader_prompt`` shows)."""
    pins, _ = select_pins(json.loads(pins_json), int(cfg.get("reader.max_pins")))
    prefix = str(cfg.get("reader.ref_prefix"))
    return {f"{prefix}{index}": str(pin[PIN_NODE_ID]) for index, pin in enumerate(pins, start=1)}


def select_pins(pins: list[Mapping[str, Any]], max_pins: int) -> tuple[list[Mapping[str, Any]], int]:
    """Keep the ``max_pins`` pins with the largest dataset priority (ties by node id)."""
    ordered = sorted(pins, key=lambda p: (-float(p.get(PIN_PRIORITY, 0)), str(p[PIN_NODE_ID])))
    return ordered[:max_pins], max(len(ordered) - max_pins, 0)


def _asset_line(ref: str, pin: Mapping[str, Any], label: str | None, cfg: Config) -> str:
    kind = str(pin[PIN_NODE_TYPE])
    parts = [kind]
    if kind == NODE_COMPANY:
        parts.append(f"sector={pin.get(PIN_SECTOR, 'unknown')}")
        parts.append(f"size={pin.get(PIN_LAYER, cfg.get('reader.non_sp_layer'))}")
    line = f"{ref}: {', '.join(parts)}"
    return f"{line} (= {label})" if label else line


def build_reader_prompt(
    *,
    cfg: Config,
    masker: Masker,
    tokenizer: Tokenizer,
    kind: str,
    family: str,
    available_date: str,
    title: str,
    text: str,
    pins_json: str,
) -> BuiltPrompt:
    """Assemble system and user prompts. Text and title are masked together so labels agree."""
    pins, dropped = select_pins(json.loads(pins_json), int(cfg.get("reader.max_pins")))
    masked = masker.mask(f"{title}{FIELD_SEPARATOR}{text}")
    masked_title, _, masked_text = masked.text.partition(FIELD_SEPARATOR)
    masked_text, truncated = tokenizer.truncate(masked_text, int(cfg.get("reader.max_text_tokens")))

    ref_prefix = str(cfg.get("reader.ref_prefix"))
    ref_to_node: dict[str, str] = {}
    lines = []
    for index, pin in enumerate(pins, start=1):
        ref = f"{ref_prefix}{index}"
        ref_to_node[ref] = str(pin[PIN_NODE_ID])
        lines.append(_asset_line(ref, pin, masked.mapping.get(str(pin[PIN_NODE_ID])), cfg))
    user = str(cfg.get("prompts.reader.user")).format(
        kind=kind,
        family=family,
        published=available_date,
        asset_lines="\n".join(lines),
        cause_list=", ".join(cfg.get("causes.causes")),
        title=masked_title,
        text=masked_text,
    )
    return BuiltPrompt(
        system=str(cfg.get("prompts.reader.system")),
        user=user,
        ref_to_node=ref_to_node,
        n_pins_given=len(pins),
        n_pins_dropped=dropped,
        text_truncated=truncated,
        mask=masked,
    )
