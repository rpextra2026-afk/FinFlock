"""Strict, isolated short-form reader; invalid fields are counted and never repaired."""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Any, Mapping

from finfm.reader.output import reader_json_schema

MAX_CAUSES = 3
MAX_HITS = 3
MAX_WHY_CHARS = 200
MAX_WHY_WORDS = 25
MAX_NAME_CHARS = 40
MAX_WEIGHT = 10
LINK_VALUES = {'strong_same': 1.0, 'weak_same': 0.5, 'none': 0.0, 'weak_opposite': -0.5, 'strong_opposite': -1.0}
LINK_TYPES = ('industry', 'rival', 'supplier', 'customer', 'country', 'commodity', 'etf_member')
HIT_ROLES = ('rival', 'supplier', 'customer', 'country', 'commodity', 'industry_group')
HIT_SIZES = ('small', 'medium', 'large')
EVENT_TYPES = ('asset_event', 'analyst_opinion', 'not_market_news')  # the optional "gate" answer, see apply_gate
FLAG_NAMES = ('unknown_causes', 'value_clamped', 'magnitude_clamped', 'hallucinated_refs',
              'duplicate_refs', 'invalid_pins', 'empty_hits', 'invalid_hits', 'duplicate_causes', 'schema_errors', 'why_word_limit')


@dataclass
class ReadingV2:
    ok: bool = False
    fail_reason: str | None = None
    why: str = ''
    cause: dict[str, float] = field(default_factory=dict)
    groups: dict[str, float] = field(default_factory=dict)
    pins: list[dict[str, Any]] = field(default_factory=list)
    link_effects: dict[str, str] = field(default_factory=dict)
    link_values: dict[str, float] = field(default_factory=dict)
    hits: list[dict[str, Any]] = field(default_factory=list)
    n_causes: int = 0
    event_type: str | None = None  # only when the gate is on
    model_cause: dict[str, float] = field(default_factory=dict)  # the model's own cause weights before the gate acts
    flags: dict[str, int] = field(default_factory=lambda: dict.fromkeys(FLAG_NAMES, 0))


def reader_v2_json_schema(cfg, gate: bool = False) -> dict:
    pin = reader_json_schema(cfg)['properties']['pins']
    pin['items']['additionalProperties'] = False
    hit = {'type': 'object', 'properties': {
        'role': {'enum': list(HIT_ROLES)}, 'name': {'type': 'string', 'minLength': 1, 'maxLength': MAX_NAME_CHARS},
        'sign': {'enum': ['up', 'down']}, 'size': {'enum': list(HIT_SIZES)}},
        'required': ['role', 'name', 'sign', 'size'], 'additionalProperties': False}
    props = {
        'why': {'type': 'string', 'maxLength': MAX_WHY_CHARS},
        'causes': {'type': 'array', 'minItems': 1, 'maxItems': MAX_CAUSES, 'items': {
            'type': 'object', 'properties': {'cause': {'enum': cfg.get('causes.causes')},
                'weight': {'type': 'integer', 'minimum': 1, 'maximum': MAX_WEIGHT},
                'hits': {'type': 'array', 'maxItems': MAX_HITS, 'items': hit}},
            'required': ['cause', 'weight', 'hits'], 'additionalProperties': False}},
        'pins': pin,
        'link_effects': {'type': 'object', 'properties': {t: {'enum': list(LINK_VALUES)} for t in LINK_TYPES},
                         'required': list(LINK_TYPES), 'additionalProperties': False}}
    if gate:
        props = {'event_type': {'enum': list(EVENT_TYPES)}, **props}
    return {'type': 'object', 'properties': props, 'required': list(props), 'additionalProperties': False}


def signed_pin(pin: Mapping[str, Any]) -> dict[str, Any]:
    """Make ``value_effect_pct`` agree with ``direction``; the model's raw number is kept as ``value_effect_raw``.

    Observed: the model writes the size as a positive number even for ``down`` (142/152 in the POC baseline), while
    the downstream return dial uses ``value_effect_pct / 100`` and ignores ``direction``. So ``direction`` decides
    the sign and the number supplies only the size: up -> +|v|, down -> -|v|, none or mixed -> 0.
    """
    sign = {'up': 1.0, 'down': -1.0}.get(pin['direction'], 0.0)
    return {**pin, 'value_effect_pct': sign * abs(float(pin['value_effect_pct'])) + 0.0, 'value_effect_raw': pin['value_effect_pct']}


def apply_gate(out: ReadingV2, cfg) -> None:
    """Make the cause and pin effects agree with the model's own answer to "is this a market event?".

    analyst_opinion (a rating or price-target change and nothing else): the cause is ``other``; pins keep their effect.
    not_market_news (sport, culture, general politics, a place merely mentioned): the cause is ``other`` and every pin
    is kept only as a zero-effect pin, so no return or hit is passed on.
    The model's own cause weights stay in ``model_cause`` so the gate's effect can be audited.
    """
    if out.event_type == 'asset_event':
        return
    out.cause = {c: (1.0 if c == 'other' else 0.0) for c in cfg.get('causes.causes')}
    out.groups = {g: sum(out.cause[c] for c in cs) for g, cs in cfg.get('causes.groups').items()}
    if out.event_type == 'not_market_news':
        out.pins = [{**p, 'direction': 'none', 'value_effect_pct': 0.0, 'magnitude': 0} for p in out.pins]
        out.hits = []
        out.link_effects = dict.fromkeys(LINK_TYPES, 'none')
        out.link_values = dict.fromkeys(LINK_TYPES, 0.0)


def parse_reading_v2(text: str, ref_to_node: Mapping[str, str], cfg, gate: bool = False, enforce_gate: bool = True) -> ReadingV2:
    from jsonschema import Draft202012Validator

    out = ReadingV2()
    try:
        def reject_constant(value):
            raise ValueError(value)
        def unique_keys(pairs):
            data = {}
            for key, value in pairs:
                if key in data:
                    raise ValueError('duplicate JSON key: ' + key)
                data[key] = value
            return data
        data = json.loads(text, parse_constant=reject_constant, object_pairs_hook=unique_keys)
    except (ValueError, TypeError):
        out.fail_reason = 'invalid_json'
        return out
    if not isinstance(data, dict):
        out.fail_reason = 'invalid_json'
        return out
    flags = out.flags
    causes = cfg.get('causes.causes')
    seen_causes, seen_refs = set(), set()
    for item in data.get('causes', []) if isinstance(data.get('causes'), list) else []:
        if not isinstance(item, dict):
            continue
        name = item.get('cause')
        flags['unknown_causes'] += int(not isinstance(name, str) or name not in causes)
        if isinstance(name, str):
            flags['duplicate_causes'] += int(name in seen_causes)
            seen_causes.add(name)
        hits = item.get('hits')
        flags['empty_hits'] += int(hits == [])  # legal: a descriptive count, not a parse error
        if isinstance(hits, list):
            validator = Draft202012Validator(reader_v2_json_schema(cfg)['properties']['causes']['items']['properties']['hits']['items'])
            flags['invalid_hits'] += sum(not validator.is_valid(h) for h in hits)
    for pin in data.get('pins', []) if isinstance(data.get('pins'), list) else []:
        if not isinstance(pin, dict):
            flags['invalid_pins'] += 1
            continue
        ref = pin.get('ref')
        if ref != cfg.get('reader.none_ref'):
            flags['hallucinated_refs'] += int(not isinstance(ref, str) or ref not in ref_to_node)
        if isinstance(ref, str):
            flags['duplicate_refs'] += int(ref in seen_refs)
            seen_refs.add(ref)
        value, mag = pin.get('value_effect_pct'), pin.get('magnitude')
        flags['value_clamped'] += int(isinstance(value, (int, float)) and not isinstance(value, bool)
                                      and (not math.isfinite(value) or abs(value) > cfg.get('reader.value_effect_pct_bound')))
        flags['magnitude_clamped'] += int(isinstance(mag, (int, float)) and not isinstance(mag, bool)
                                          and (not math.isfinite(mag) or mag < 0 or mag > cfg.get('reader.magnitude_max')))
        schema = reader_v2_json_schema(cfg)['properties']['pins']['items']
        flags['invalid_pins'] += int(not Draft202012Validator(schema).is_valid(pin))
    errors = list(Draft202012Validator(reader_v2_json_schema(cfg, gate)).iter_errors(data))
    flags['schema_errors'] = len(errors)
    flags['why_word_limit'] = int(isinstance(data.get('why'), str) and len(data['why'].split()) > MAX_WHY_WORDS)
    fatal = errors or any(flags[k] for k in ('hallucinated_refs', 'duplicate_refs', 'why_word_limit'))
    if fatal:
        required = ('why', 'causes', 'pins', 'link_effects') + (('event_type',) if gate else ())
        out.fail_reason = 'missing_key' if any(k not in data for k in required) else 'invalid_form'
        return out
    weights = dict.fromkeys(causes, 0)
    for item in data['causes']:
        weights[item['cause']] += item['weight']
        out.hits.extend({'cause': item['cause'], **hit} for hit in item['hits'])
    total = sum(weights.values())
    out.cause = {c: w / total for c, w in weights.items()}
    out.groups = {g: sum(out.cause[c] for c in cs) for g, cs in cfg.get('causes.groups').items()}
    out.pins = [{**signed_pin(p), 'node_id': ref_to_node[p['ref']]} for p in data['pins'] if p['ref'] != cfg.get('reader.none_ref')]
    out.link_effects = data['link_effects']
    out.link_values = {k: LINK_VALUES[v] for k, v in out.link_effects.items()}
    out.why, out.n_causes, out.ok = data['why'], len(data['causes']), True
    if gate:
        out.event_type, out.model_cause = data['event_type'], dict(out.cause)
        if enforce_gate:  # False: the caller decides per item whether the gate may act (filings and structured records are events by construction)
            apply_gate(out, cfg)
    return out
