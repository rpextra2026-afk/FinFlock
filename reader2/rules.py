"""What the code does with the model's answer (the model is never trusted to follow these itself).

1. Gate (news and chatter only): the model's own ``event_type`` decides. ``not_market_news`` -> cause ``other`` and every pin
   becomes a zero-effect pin; ``analyst_opinion`` -> cause ``other``, pins keep their effect.
2. Analyst-headline rule (finance and world news only): a headline such as "Goldman Upgrades Merck" or "Morgan Stanley Raises PT"
   is an analyst opinion whatever the model said (``not_market_news`` is left alone: it is already stricter).
3. Filings, Federal Register, OFAC and GDELT records are events by construction: the model's ``event_type`` is only recorded.
4. ``other`` means "no listed event happened". Downstream it must carry no cause-group effect; ``is_other`` marks those rows.
"""
from __future__ import annotations

from oryxis.data.pin_cleanup import is_rating_headline
from oryxis.reader.output_v2 import ReadingV2, apply_gate

from reader2 import settings as S

EVENT = "asset_event"
OPINION = "analyst_opinion"
NOT_NEWS = "not_market_news"


def finalize(reading: ReadingV2, family: str, title: str, cfg) -> dict:
    """Apply the rules to a parsed (``ok``) reading in place; return the audit fields stored with the row."""
    audit = {"event_type_model": reading.event_type, "gate_enforced": False, "rating_rule": False}
    if family in S.GATED_FAMILIES:
        audit["gate_enforced"] = True
        if family in S.RATING_FAMILIES and reading.event_type == EVENT and is_rating_headline(title):
            reading.event_type = OPINION
            audit["rating_rule"] = True
        apply_gate(reading, cfg)
        audit["event_type_final"] = reading.event_type
    else:
        audit["event_type_final"] = EVENT  # an event by construction; the model's own answer stays in event_type_model
    return audit


def is_other(reading: ReadingV2) -> bool:
    return bool(reading.ok and reading.cause and max(reading.cause, key=reading.cause.get) == "other")
