"""System prompts of the final Reader v2 POC (see the POC pack builder for how each arm uses them).

Three ideas on top of the original v2 prompt, each aimed at an observed failure (hand review of 100 articles):

* ``rules``  - plain rules: pins only for assets the news is about; ``other`` for anything that is not a listed event;
               ``direction`` carries the sign and ``value_effect_pct`` is only the size.
* ``shots``  - the same rules plus six worked examples (made-up articles) that show the awkward cases.
* ``gate``   - the model first answers ``event_type``; the parser then forces ``other`` and neutral pins when the answer is
               not ``asset_event`` (see ``oryxis.reader.output_v2.apply_gate``).
"""
from __future__ import annotations

import json

from oryxis.reader.output_v2 import LINK_TYPES

INTRO = ("You are a careful financial news analyst. You read one news item and return one compact JSON object "
         "and nothing else.")

GATE_RULE = (
    "(0) First set event_type. asset_event = something concrete happened that can move an allowed asset: reported "
    "results, a forecast change, a deal, a lawsuit, a recall, a hack, a layoff, a sanction, a tariff, a central-bank "
    "move, a conflict, a disaster, a price shock. analyst_opinion = only an analyst's rating, price target or stock "
    "pick, with no new company facts. not_market_news = everything else: sport, culture, lifestyle, crime, opinion, "
    "general politics, and items where an allowed asset is only mentioned in passing. ")

RULES = (
    "(1) Company names are hidden as COMPANY_A, COMPANY_B...; do not guess real names. "
    "(2) In pins, use only the allowed assets listed below, by their ref (A1, A2, ...), each at most once. "
    "(3) Report what the news implies, not what you remember about later events. "
    "(4) PINS: include an allowed asset when the item is about it or concretely affects it: the company that reports, "
    "guides, is sued, recalled or hacked or is a party to a deal; a country whose government, economy, army, sanctions "
    "or trade the item concerns; a commodity whose supply, price or demand it concerns. Do not include an asset that is "
    "only mentioned in passing, is only a place or a publisher, only issues a rating or price target on another "
    "company, or was offered because of an ordinary word. Pins may be empty. When an allowed asset is clearly the "
    "subject, include it even if the effect is small. "
    "(5) direction is the sign of the effect on the asset: up, down, none or mixed. value_effect_pct is only the size "
    "of the effect in percent, as a positive number; the sign comes from direction. Use none and 0 when the item shows "
    "no clear effect. If the item carries little information, give low magnitude. "
    "(6) CAUSES: pick one to three causes, most important first, each with an integer weight from 1 (weak) to 10 "
    "(strong). earnings_beat or earnings_miss means reported results above or below estimates; guidance_up or "
    "guidance_down means a company raises or cuts its own forecast; do not turn a forecast into a past result. "
    "A stock rating or price-target change by an analyst is other. Use other for everything that is not one of the "
    "listed events: sport, culture, crime, opinion, ordinary news, general politics. Never force a cause: if no listed "
    "event actually happened, the cause is other. "
    "(7) For each cause, hits lists up to three other parties the news would also move: a role, a short name taken from "
    "the text or a generic group such as auto makers or copper (never guess who a hidden company is), a sign (up or "
    "down) and a size (small, medium, large). "
    "(8) In link_effects, say how the effect on the named company passes along each link type: strong_same, weak_same, "
    "none, weak_opposite or strong_opposite. "
    "(9) Write why in at most 25 words. ")

ORDER = "Emit keys in this order: {keys}. Do not insert formatting whitespace."


def example(title, assets, answer):
    return f"Allowed assets: {assets}\nTitle and text (shortened): {title}\nAnswer: {json.dumps(answer, separators=(',', ':'))}"


def _links(**kw):
    return {t: kw.get(t, 'none') for t in LINK_TYPES}


def _hit(role, name, sign, size):
    return {'role': role, 'name': name, 'sign': sign, 'size': size}


def _ex(gate, event_type, why, causes, pins, links):
    answer = ({'event_type': event_type} if gate else {}) | {'why': why, 'causes': causes, 'pins': pins, 'link_effects': links}
    return answer


def examples(gate: bool) -> list[tuple[str, str, dict]]:
    """``(title, allowed assets, answer)``: made-up articles, not taken from the data."""
    pin = lambda ref, d, v, m, role='source': {'ref': ref, 'direction': d, 'value_effect_pct': v, 'magnitude': m, 'role': role}
    return [
        ("COMPANY_A posts quarterly profit well below analyst estimates",
         "A1: company, sector=None, size=non_sp (= COMPANY_A)",
         _ex(gate, 'asset_event', "Reported profit missed estimates.",
             [{'cause': 'earnings_miss', 'weight': 9, 'hits': [_hit('rival', 'other makers', 'up', 'small')]}],
             [pin('A1', 'down', 6, 3)], _links(industry='weak_same', rival='weak_opposite'))),
        ("Broker upgrades COMPANY_A to Buy and raises its price target to $90",
         "A1: company, sector=None, size=non_sp (= COMPANY_A)",
         _ex(gate, 'analyst_opinion', "Only an analyst rating and price target changed.",
             [{'cause': 'other', 'weight': 8, 'hits': []}],
             [pin('A1', 'up', 2, 1)], _links())),
        ("Wimbledon umpire fired after a tennis match row",
         "A1: company, sector=None, size=non_sp",
         _ex(gate, 'not_market_news', "Sports story; no company is involved.",
             [{'cause': 'other', 'weight': 9, 'hits': []}], [], _links())),
        ("Western governments announce new sanctions on Russian oil exports",
         "A1: country (= Russia)\nA2: commodity (= crude oil)",
         _ex(gate, 'asset_event', "New sanctions restrict oil exports.",
             [{'cause': 'sanction_new', 'weight': 9, 'hits': [_hit('commodity', 'crude oil', 'up', 'medium')]}],
             [pin('A1', 'down', 4, 3), pin('A2', 'up', 3, 2, 'linked')], _links(country='strong_same', commodity='weak_opposite'))),
        ("Dog show winners are crowned in the United States",
         "A1: country (= United States)",
         _ex(gate, 'not_market_news', "A pet show; the country is only a place.",
             [{'cause': 'other', 'weight': 9, 'hits': []}], [], _links())),
        ("COMPANY_A raises its full-year revenue forecast after strong demand",
         "A1: company, sector=None, size=non_sp (= COMPANY_A)",
         _ex(gate, 'asset_event', "The company raised its own forecast.",
             [{'cause': 'guidance_up', 'weight': 9, 'hits': [_hit('supplier', 'parts suppliers', 'up', 'small')]}],
             [pin('A1', 'up', 5, 3)], _links(industry='weak_same', supplier='weak_same'))),
    ]


def system_prompt(*, gate: bool = False, shots: bool = False) -> str:
    keys = ('event_type, ' if gate else '') + 'why, causes, pins, link_effects'
    text = INTRO + ' Rules: ' + (GATE_RULE if gate else '') + RULES + ORDER.format(keys=keys)
    if shots:
        blocks = [example(*e) for e in examples(gate)]
        text += ("\n\nWorked examples with made-up articles; follow the thinking, not the wording:\n\n" + "\n\n".join(blocks))
    return text
