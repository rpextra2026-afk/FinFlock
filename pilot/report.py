"""Descriptive numbers and Wilson intervals, including explicitly uncompleted steps."""
from __future__ import annotations
from collections import Counter
import json
import math
from pathlib import Path
import numpy as np
import pandas as pd
from finflock.reader.output_v2 import LINK_TYPES


def rate(k, n):
    if not n:
        return {'count': int(k), 'n': 0, 'rate': None, 'ci95': None}
    z = 1.959963984540054
    p = k / n
    mid = (p + z*z/(2*n)) / (1 + z*z/n)
    half = z * math.sqrt(p*(1-p)/n + z*z/(4*n*n)) / (1+z*z/n)
    return {'count': int(k), 'n': int(n), 'rate': p, 'ci95': [mid-half, mid+half]}


def numeric(values):
    a = np.asarray(list(values), dtype=float)
    a = a[np.isfinite(a)]
    return {'n': len(a), 'mean': float(a.mean()) if len(a) else None,
            'median': float(np.median(a)) if len(a) else None,
            'p95': float(np.quantile(a, .95)) if len(a) else None, 'max': float(a.max()) if len(a) else None}


def stage_metrics(frame, elapsed):
    n = len(frame)
    if not n:
        return {'n': 0, 'strict_parse': rate(0, 0), 'seconds': elapsed}
    tokens = pd.to_numeric(frame['tokens_out'], errors='coerce').fillna(0)
    called = frame.get('status', pd.Series(index=frame.index, dtype=str)).ne('skipped')
    count = int(called.sum())
    result = {'n': n, 'strict_parse': rate(int(frame['parse_ok'].fillna(False).sum()), count),
        'no_usable_pin': rate(int((frame.get('fail_reason', pd.Series(index=frame.index, dtype=str)) == 'no_usable_pin').sum()), n),
        'finish_reasons': dict(Counter(frame['finish_reason'].fillna('unknown'))),
        'tokens_out': numeric(tokens), 'flag_counts': {c:int(pd.to_numeric(frame[c],errors='coerce').fillna(0).sum()) for c in frame if c.startswith('flag_')}, 'seconds': elapsed, 'called_n': count,
        'seconds_per_unit': elapsed/count if count else None,
        'tokens_per_second': float(tokens.sum())/elapsed if elapsed else None,
        'units_per_hour': count*3600/elapsed if elapsed else None,
        'projected_hours_30000': elapsed*30000/count/3600 if count else None,
        'projected_hours_60000': elapsed*60000/count/3600 if count else None}
    if 'salvage_usable' in frame:
        result['after_salvage'] = rate(int(frame['salvage_usable'].fillna(False).sum()), count)
    return result


def build_metrics(v2, baseline, scores, cfg, state):
    m = {'partial': state.get('partial', True), 'n_selected': state.get('n_selected', 0),
         'sample': state.get('sample', {}), 'cut_steps': state.get('cut_steps', []),
         'baseline_plan_n': state.get('baseline_plan_n'), 'score_plan_n': state.get('score_plan_n'),
         'source_no_usable_pin': rate(state.get('sample',{}).get('no_usable_pin_n',0),state.get('sample',{}).get('source_n',0)),
         'source_dial_check': {'status': 'not run', 'reason': 'optional check not wired; no forecast accuracy measured'},
         'new_form': stage_metrics(v2, state.get('timing', {}).get('new_form', 0)),
         'baseline': stage_metrics(baseline, state.get('timing', {}).get('baseline', 0))}
    good = v2[v2['parse_ok'].fillna(False)] if len(v2) else v2
    if len(good):
        hits = [h for raw in good['hits_json'] for h in json.loads(raw)]
        m['event_counts'] = {'causes': dict(Counter(map(int, good['n_causes']))),
            'pins': numeric(len(json.loads(s)) for s in good['pins_json']),
            'hits': numeric(len(json.loads(s)) for s in good['hits_json']),
            'with_hit': rate(sum(bool(json.loads(s)) for s in good['hits_json']), len(good)),
            'link_none': {t: rate(int((good['link_'+t] == 0).sum()), len(good)) for t in LINK_TYPES}}
        m['resolution'] = {role: rate(sum(h['resolved'] for h in hits if h['role'] == role), sum(h['role'] == role for h in hits))
                           for role in sorted({h['role'] for h in hits})}
        m['unresolved_names'] = Counter(h['name'] for h in hits if not h['resolved']).most_common(20)
        m['resolution_kind'] = dict(Counter(h['resolution_kind'] for h in hits))
        m['cause_top'] = dict(Counter(good['cause_top']))
        m['group_top'] = dict(Counter(good['group_top']))
        m['other'] = rate(int((good['cause_top'] == 'other').sum()), len(good))
        compared = good[good.get('old_cause_top', pd.Series(index=good.index, dtype=object)).notna()]
        m['old_tape_agreement'] = rate(int((compared['cause_top'] == compared['old_cause_top']).sum()), len(compared))
        m['old_tape_group_agreement'] = rate(int((compared['group_top'] == compared['old_group_top']).sum()), len(compared))
    valid = scores[scores['status'] == 'ok'] if len(scores) else scores
    if len(scores):
        single_n = sum(int((scores['label_tokens_'+c] == 1).sum()) for c in cfg.get('causes.causes'))
        token_n = sum(int(scores['label_tokens_'+c].notna().sum()) for c in cfg.get('causes.causes'))
        m['cause_scores'] = {'single_token_labels': rate(single_n,token_n), 'n': len(scores), 'n_complete': len(valid),
            'entropy': numeric(valid['entropy']) if len(valid) else numeric([]),
            'boundary_merge_events': rate(int(scores['boundary_merge'].fillna(False).sum()), len(scores)),
            'single_token_by_label': {c: rate(int((scores['label_tokens_'+c] == 1).sum()), int(scores['label_tokens_'+c].notna().sum())) for c in cfg.get('causes.causes')}}
        if len(valid):
            m['cause_scores']['mean_groups'] = {g: numeric(valid['gs_'+g]) for g in cfg.get('causes.groups')}
            together = valid.merge(good[['unit_id','cause_top']], on='unit_id') if len(good) else pd.DataFrame()
            m['cause_scores']['form_agreement'] = rate(int((together['score_top'] == together['cause_top']).sum()), len(together)) if len(together) else rate(0,0)
    return m


def write_report(out, cfg, state):
    def load(name):
        p = out/(name+'.parquet')
        return pd.read_parquet(p) if p.exists() else pd.DataFrame()
    m = build_metrics(load('tape_v2'), load('tape_baseline'), load('cause_scores'), cfg, state)
    # JSON disallows NaNs; missing values are explicit nulls.
    payload = json.dumps(m, indent=2, allow_nan=False, default=lambda x: x.item() if isinstance(x, np.generic) else str(x))
    atomic_text(out/'metrics.json', payload)
    atomic_text(out/'report.md', '# Reader v2 pilot\n\nDescriptive counts, rates and Wilson 95% intervals. No forecast-accuracy result is implied.\n\n```json\n'+payload+'\n```\n')


def atomic_text(path, text):
    tmp = Path(str(path)+'.tmp')
    tmp.write_text(text)
    tmp.replace(path)
