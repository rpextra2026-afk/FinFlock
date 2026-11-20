"""Bounded, checkpointed pilot; the parent owns the deadline and kills the GPU process group."""
from __future__ import annotations
import argparse
import gc
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import threading
import time
from dataclasses import replace
from datetime import datetime, timezone

import pandas as pd
import yaml

from finfm.common.config import Config
from finfm.common.schema import UNIT_ID, AVAILABLE_DATE, KIND, PINS_JSON, TITLE, TEXT
from finfm.compute.bundle import read_consolidated, scan_bundles
from finfm.compute.engine import GenParams, GenRequest
from finfm.reader.cause_scores import score_distribution
from finfm.reader.hits_v2 import HitResolver
from finfm.reader.masking import MaskerCache
from finfm.reader.output_v2 import LINK_TYPES, FLAG_NAMES, parse_reading_v2, reader_v2_json_schema
from finfm.reader.prompt import ref_to_node_of
from finfm.reader.run import ReaderProcessor
from finfm.reader.salvage import reparse_with_salvage
from finfm.data.names import drop_generic_aliases
from pilot.report import atomic_text, write_report

ROOT = Path(__file__).resolve().parents[1]
NEWLINE = chr(10)


def utc():
    return datetime.now(timezone.utc).isoformat()


def identity_hash(unit):
    return hashlib.sha256(str(unit).encode()).hexdigest()


def stratified_sample(units, n, preferred=(), proportions=None):
    """Largest-remainder kind quotas, capped at capacity; stable hash within each kind."""
    units = units.copy()
    if units[UNIT_ID].duplicated().any():
        raise ValueError('duplicate_unit_ids')
    n = min(n, len(units))
    capacities = units[KIND].value_counts().to_dict()
    weights = proportions or capacities
    quotas = dict.fromkeys(capacities, 0)
    for _ in range(n):
        eligible = [k for k in capacities if quotas[k] < capacities[k]]
        k = max(eligible, key=lambda k: (weights.get(k, 0) * n / sum(weights.values()) - quotas[k], k))
        quotas[k] += 1
    preferred = set(preferred)
    units['_hash'] = units[UNIT_ID].map(identity_hash)
    units['_preferred'] = ~units[UNIT_ID].isin(preferred)
    selected = pd.concat([part.sort_values(['_preferred','_hash',UNIT_ID]).head(quotas[k]) for k, part in units.groupby(KIND)]) if n else units.head(0)
    return selected.sort_values(['_hash',UNIT_ID]).drop(columns=['_hash','_preferred']).reset_index(drop=True)


def load_units(settings):
    """Reject stale two-kind tables; a configured, audited alternative may supply the specified mix."""
    candidates = [settings['units']]
    if settings.get('units_fallback'):
        candidates.append(settings['units_fallback'])
    inspected = []
    for relative in candidates:
        path = ROOT/relative
        if not path.exists():
            inspected.append({'path':str(path),'reason':'missing'})
            continue
        frame = pd.read_parquet(path)
        counts = {str(k):int(v) for k,v in frame[KIND].value_counts().items()}
        inspected.append({'path':str(path),'kind_counts':counts,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
        if settings.get('expected_kind_counts') and counts != settings['expected_kind_counts']:
            inspected[-1]['reason'] = 'kind_counts_differ_from_spec'
            continue
        return frame, {'selected_path':str(path),'candidates':inspected,'used_fallback':relative!=settings['units']}
    raise ValueError('no_unit_table_matches_spec: '+json.dumps(inspected))


def load_old_tape(path):
    if not path.exists():
        return pd.DataFrame(), {'available': False, 'path': str(path)}
    scan = scan_bundles([path.parent], 'reader__db3__llm2')
    # The scan validates manifests and consolidation deduplicates unit ids.
    refs = [ref for ref in scan.valid if Path(ref.path).resolve() == path.resolve()]
    if not refs:
        raise ValueError('old_tape_not_valid_or_wrong_job')
    return read_consolidated(refs), {'available': True, 'path': str(path)}


class TokenizerAdapter:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def truncate(self, text, max_tokens):
        ids = self.tokenizer.encode(text, add_special_tokens=False)
        return (self.tokenizer.decode(ids[:max_tokens]), True) if len(ids) > max_tokens else (text, False)


def asset_names(cfg, countries, commodities):
    """``c:US -> United States``, ``m:oil -> crude oil``: the first gazetteer word of each country and commodity."""
    prefix = cfg.get('events.pins.node_prefix')
    names = {}
    for _, row in countries.iterrows():
        names.setdefault(prefix['country'] + str(row['iso']), str(row['alias']))
    for _, row in commodities.iterrows():
        names.setdefault(prefix['commodity'] + str(row['tag']), str(row['keyword']))
    return names


def name_assets(user, ref_to_node, names):
    """Say WHICH country or commodity an offered asset is; the plain prompt only says ``A2: country``."""
    lines = []
    for line in user.split(NEWLINE):
        ref, sep, rest = line.partition(': ')
        if sep and rest in ('country', 'commodity') and names.get(ref_to_node.get(ref)):
            line = f'{line} (= {names[ref_to_node[ref]]})'
        lines.append(line)
    return NEWLINE.join(lines)


def v2_prompt(old, settings, names=None):
    glossary = settings['reader_v2']['glossary']
    user = name_assets(old.user, old.ref_to_node, names) if names else old.user
    marker = 'Return the JSON.'
    if marker in user:
        user = user.rsplit(marker, 1)[0] + glossary + '\n' + marker
    else:
        user += '\n' + glossary
    return replace(old, system=settings['reader_v2']['system'], user=user)


def scoring_request(old, settings):
    marker = 'Return the JSON.'
    user = old.user.rsplit(marker, 1)[0] if marker in old.user else old.user
    return GenRequest('score', user + '\n' + settings['reader_v2']['scoring_instruction'], system=settings['reader_v2']['scoring_system'])


def table_columns(cfg, name):
    common = ['unit_id','status','raw_output','tokens_in','tokens_out','finish_reason','parse_ok','fail_reason']
    if name == 'tape_v2':
        return common+['why','cause_top','group_top','old_cause_top','old_group_top','n_causes','event_type','model_cause_top','pins_json','hits_json'] + ['flag_'+f for f in FLAG_NAMES] + ['pi_'+c for c in cfg.get('causes.causes')]+['gw_'+g for g in cfg.get('causes.groups')]+['link_'+t for t in LINK_TYPES]
    if name == 'tape_baseline':
        return common+['reason','salvage_status','salvage_usable']+['cause_'+c for c in cfg.get('causes.causes')]
    return ['unit_id','status','fail_reason','entropy','score_top','boundary_merge','boundary_merge_labels']+['logp_'+c for c in cfg.get('causes.causes')]+['p_'+c for c in cfg.get('causes.causes')]+['label_tokens_'+c for c in cfg.get('causes.causes')]+['gs_'+g for g in cfg.get('causes.groups')]


def save_frame(out, name, rows, cfg):
    frame = pd.DataFrame(rows) if len(rows) else pd.DataFrame(columns=table_columns(cfg,name))
    required = table_columns(cfg,name)
    frame = frame.reindex(columns=list(dict.fromkeys(list(frame.columns)+required)))
    # Each checkpoint is replace-atomic, so a watchdog termination cannot corrupt the previous one.
    tmp = out/(name+'.parquet.tmp')
    frame.to_parquet(tmp,index=False)
    tmp.replace(out/(name+'.parquet'))


def save_state(out, state):
    atomic_text(out/'lineage.json', json.dumps(state, indent=2, default=str))


def gpu_preflight(expected_model='A30', run=subprocess.run):
    try:
        g = run(['nvidia-smi','--query-gpu=index,name,memory.used,utilization.gpu','--format=csv,noheader,nounits'], capture_output=True,text=True,check=True,timeout=10)
        procs = run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True,timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f'{expected_model} GPU preflight unavailable') from exc
    rows = [line.split(',') for line in g.stdout.strip().splitlines() if line.strip()]
    if len(rows)!=2 or any(expected_model not in row[1] for row in rows):
        raise RuntimeError(f'requires_two_{expected_model}_GPUs')
    if procs.stdout.strip() or any(float(row[3]) != 0 for row in rows):
        raise RuntimeError('GPUs busy; no pilot started')
    return {'gpu_query':g.stdout.strip(),'compute_apps':procs.stdout.strip()}


class MetricsPulse:
    def __init__(self, state, seconds):
        self.state, self.seconds = state, seconds
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.loop,daemon=True)
        self.start = time.monotonic()

    def loop(self):
        while not self.stop.wait(self.seconds):
            elapsed = time.monotonic()-self.start
            done = self.state.get('units_done',0)
            print(f"METRICS stage={self.state.get('stage','startup')} units_done={done} units_per_second={done/max(elapsed,1e-9):.6f} tokens_out={self.state.get('tokens_out',0)} elapsed_seconds={elapsed:.1f}",flush=True)

    def __enter__(self):
        self.thread.start()
        return self

    def __exit__(self,*args):
        self.stop.set();self.thread.join()


def model_path(cfg, settings, explicit=None):
    if explicit:
        return explicit
    from finfm.compute.session import find_weights
    key = settings['llm']
    weights_root = Path(os.environ.get(str(cfg.get('engine.weights_root_env')),str(cfg.get('engine.local_weights_root'))))
    found = find_weights(weights_root,cfg.get(f'engine.models.{key}.weights_match')) if weights_root.exists() else None
    return str(found) if found else cfg.get(f'engine.models.{key}.path')


def execute(cfg, settings, out, state, deadline, engine_factory=None, explicit_model=None):
    """Runs with an injectable engine for CPU integration tests; no synthetic production mode."""
    from pilot.engine import ScoringVLLM
    factory = engine_factory or ScoringVLLM
    engine = None
    start = time.monotonic()
    timings = state.setdefault('timing',{})
    for name in ('tape_v2','tape_baseline','cause_scores'):
        save_frame(out,name,[],cfg)
    save_state(out,state);write_report(out,cfg,state)
    def checkpoint():
        save_state(out,state)
        write_report(out,cfg,state)
    def time_left():
        return time.monotonic() < deadline
    dtype = settings['dtype']
    gate = bool(settings['reader_v2'].get('gate', False))
    def make_engine(dtype):
        override = cfg.with_overrides({'engine.dtype':dtype,'engine.enforce_eager':False,'engine.enable_prefix_caching':True})
        try:
            return factory(state['model_path'],override,tensor_parallel_size=settings['tensor_parallel_size'])
        except RuntimeError as exc:
            text = str(exc).lower()
            if 'graph' not in text or not any(x in text for x in ('capture','cuda')):
                raise
            state.setdefault('engine_fallbacks',[]).append({'reason':'cuda_graph_capture','enforce_eager':True})
            return factory(state['model_path'],override.with_overrides({'engine.enforce_eager':True}),tensor_parallel_size=settings['tensor_parallel_size'])
    def fallback(reason):
        nonlocal engine,dtype,processor
        if dtype == settings['fallback_dtype']:
            return False
        engine.close();del engine;gc.collect()
        dtype = settings['fallback_dtype']
        state.setdefault('engine_fallbacks',[]).append({'reason':reason,'dtype':dtype})
        engine = make_engine(dtype)
        processor = ReaderProcessor(cfg,engine,maskers,TokenizerAdapter(engine.tokenizer))
        state['engine'] = engine.describe()
        return True
    try:
        state['stage'] = 'loading_data'
        units, state['unit_table'] = load_units(settings)
        aliases = pd.read_parquet(ROOT/settings['aliases'])
        snapshots = pd.read_parquet(ROOT/settings['snapshots'])
        state['table_columns'] = {'units':list(units.columns),'aliases':list(aliases.columns),'snapshots':list(snapshots.columns)}
        old, state['old_tape'] = load_old_tape(ROOT/settings['old_tape'])
        if len(old):
            from finfm.reader.salvage import verify_alignment
            state['old_tape_alignment'] = verify_alignment(old,dict(zip(units[UNIT_ID],units[PINS_JSON])),cfg,expected_units=len(units))
            if state['old_tape_alignment']['pin_count_mismatches'] or state['old_tape_alignment']['node_id_mismatches']:
                raise ValueError('old_tape_unit_table_mismatch')
        valid = units[PINS_JSON].map(lambda x: bool(ref_to_node_of(x,cfg)))
        eligible = units[valid].copy()
        state['sample'] = {'source_n':len(units),'eligible_n':len(eligible),'no_usable_pin_n':int((~valid).sum()),
                           'no_usable_pin_rate':float((~valid).mean())}
        overlap_ids = set(old[UNIT_ID]) if len(old) else set()
        sample = stratified_sample(eligible,settings['sample_size'],overlap_ids,units[KIND].value_counts().to_dict())
        prompt_aliases = aliases if settings.get('aliases_prefiltered') else drop_generic_aliases(aliases, units[TITLE] + ' ' + units[TEXT], cfg)
        state['prompt_aliases'] = {'input_n': len(aliases), 'used_n': len(prompt_aliases)}
        maskers = MaskerCache(prompt_aliases,label_prefix=cfg.get('reader.masked_company_prefix'),short_ticker_max_len=cfg.get('reader.short_ticker_max_len'))
        countries,commodities = pd.read_csv(ROOT/'config/countries.csv'),pd.read_csv(ROOT/'config/commodity_keywords.csv')
        names = asset_names(cfg,countries,commodities) if settings.get('name_assets') else None
        resolver = HitResolver(aliases,snapshots,countries,commodities,pd.read_csv(ROOT/'config/sic_sector.csv'),cfg.get('events.pins.node_prefix'))
        state['stage'] = 'loading_model'
        state['model_path'] = model_path(cfg,settings,explicit_model)
        save_state(out,state)
        if not time_left():
            raise TimeoutError('deadline_before_model')
        engine = make_engine(dtype)
        state['engine'] = engine.describe()
        processor = ReaderProcessor(cfg,engine,maskers,TokenizerAdapter(engine.tokenizer))
        params = GenParams(temperature=0.0,top_p=1.0,top_k=None,max_tokens=settings['max_tokens'],json_schema=reader_v2_json_schema(cfg,gate),seed=cfg.get('runtime.base_seed'))
        state['stage'] = 'smoke'
        smoke = sample.head(settings['smoke_size'])
        def generate(events):
            built = {str(row[UNIT_ID]):v2_prompt(processor._build(row),settings,names) for _,row in events.iterrows()}
            requests = [GenRequest(u,b.user,system=b.system,task='reader_v2') for u,b in built.items()]
            results = engine.generate(requests,params)
            if any(r.nan_suspect for r in results) and fallback('nan_or_empty_form'):
                results = engine.generate(requests,params)
            return built,results
        smoke_start = time.monotonic()
        built,results = generate(smoke)
        smoke_seconds = time.monotonic()-smoke_start
        smoke_readings = [parse_reading_v2(r.texts[0],built[r.unit_id].ref_to_node,cfg,gate) for r in results]
        state['smoke'] = {'n':len(results),'seconds':smoke_seconds,'seconds_per_unit':smoke_seconds/max(len(results),1),
            'unit_ids':[r.unit_id for r in results], 'strict_parsed':sum(reading.ok for reading in smoke_readings),
            'nonempty_pins':sum(reading.ok and bool(reading.pins) for reading in smoke_readings)}
        minimum = settings.get('min_smoke_nonempty_pin_rate')
        if minimum is not None and state['smoke']['nonempty_pins']/max(len(results),1) < float(minimum):
            raise RuntimeError('smoke_nonempty_pin_rate_below_threshold')
        for r in results[:3]:
            print('SMOKE_RAW unit='+r.unit_id+' '+r.texts[0],flush=True)
        if state['smoke']['seconds_per_unit'] > settings['slow_seconds_per_unit']:
            sample = stratified_sample(sample,settings['slow_sample_size'],overlap_ids,units[KIND].value_counts().to_dict())
            state['sample_reduction'] = {'reason':'smoke_seconds_per_unit','n':len(sample)}
        state['n_selected'] = len(sample)
        state['sample_unit_ids'] = sample[UNIT_ID].astype(str).tolist()
        state['sample']['kind_counts'] = {str(k):int(v) for k,v in sample[KIND].value_counts().items()}
        state['sample']['old_tape_overlap'] = int(sample[UNIT_ID].isin(overlap_ids).sum())
        baseline_units = stratified_sample(sample,settings['baseline_size'],proportions=units[KIND].value_counts().to_dict())
        state['baseline_plan_n'] = len(baseline_units)
        state['baseline_unit_ids'] = baseline_units[UNIT_ID].astype(str).tolist()
        old_lookup = {}
        if len(old):
            subset = old[old[UNIT_ID].isin(sample[UNIT_ID])]
            repaired = reparse_with_salvage(subset,dict(zip(sample[UNIT_ID],sample[PINS_JSON])),cfg)
            for _, row in repaired.iterrows():
                if row.get('salvage_status') in ('natural','salvaged'):
                    cause = {c:row.get('cause_'+c,0) for c in cfg.get('causes.causes')}
                    groups = {g:sum(cause[c] for c in cs) for g,cs in cfg.get('causes.groups').items()}
                    old_lookup[str(row[UNIT_ID])] = (max(cause,key=cause.get),max(groups,key=groups.get))
        records=[];state['stage']='new_form';phase=time.monotonic()
        smoke_results={r.unit_id:r for r in results}
        for offset in range(0,len(sample),settings['batch_size']):
            if not time_left():break
            batch=sample.iloc[offset:offset+settings['batch_size']]
            try:
                if all(str(u) in smoke_results for u in batch[UNIT_ID]):
                    prompts={str(row[UNIT_ID]):v2_prompt(processor._build(row),settings,names) for _,row in batch.iterrows()}
                    batch_results=[smoke_results[str(u)] for u in batch[UNIT_ID]]
                else:
                    prompts,batch_results=generate(batch)
                result_map={r.unit_id:r for r in batch_results}
                for _,row in batch.iterrows():
                    u=str(row[UNIT_ID]);b=prompts[u];r=result_map[u]
                    reading=parse_reading_v2(r.texts[0],b.ref_to_node,cfg,gate)
                    rec=dict.fromkeys(table_columns(cfg,'tape_v2'))
                    rec.update(dtype=dtype,unit_id=u,status='ok',raw_output=r.texts[0],tokens_in=r.tokens_in,tokens_out=r.tokens_out,
                        finish_reason=r.finish,parse_ok=reading.ok,fail_reason=reading.fail_reason,why=reading.why,
                        n_causes=reading.n_causes,event_type=reading.event_type,pins_json=json.dumps(reading.pins),
                        hits_json=json.dumps(resolver.resolve_all(reading.hits,row[AVAILABLE_DATE],{node: label for node,label in b.mask.mapping.items() if node in b.ref_to_node.values()})),
                        old_cause_top=old_lookup.get(u,(None,None))[0],old_group_top=old_lookup.get(u,(None,None))[1])
                    rec.update({'flag_'+k:v for k,v in reading.flags.items()})
                    rec.update({'pi_'+c:reading.cause.get(c,0.0) for c in cfg.get('causes.causes')})
                    rec.update({'gw_'+g:reading.groups.get(g,0.0) for g in cfg.get('causes.groups')})
                    rec.update({'link_'+t:reading.link_values.get(t) for t in LINK_TYPES})
                    if reading.ok:
                        rec.update(cause_top=max(reading.cause,key=reading.cause.get),group_top=max(reading.groups,key=reading.groups.get),
                            model_cause_top=max(reading.model_cause,key=reading.model_cause.get) if reading.model_cause else None)
                    records.append(rec)
                    state['tokens_out']=state.get('tokens_out',0)+r.tokens_out
            except Exception as exc:
                completed={r['unit_id'] for r in records}
                for u in batch[UNIT_ID]:
                    if str(u) not in completed:
                        rec=dict.fromkeys(table_columns(cfg,'tape_v2'));rec.update(unit_id=str(u),status='error',parse_ok=False,fail_reason=type(exc).__name__+': '+str(exc),tokens_in=0,tokens_out=0,finish_reason='error',pins_json='[]',hits_json='[]');records.append(rec)
            state['units_done']=len(records)
            timings['new_form']=smoke_seconds+time.monotonic()-phase
            state['engine']=engine.describe()
            save_frame(out,'tape_v2',records,cfg);checkpoint()
        # Save the new-form report before spending the remaining budget on the old format.
        state['stage']='baseline';phase=time.monotonic();baseline=[]
        for offset in range(0,len(baseline_units),settings['batch_size']):
            if not time_left():break
            batch=baseline_units.iloc[offset:offset+settings['batch_size']]
            try:
                frame=processor(batch)
                if frame.get('nan_detected', pd.Series(False,index=frame.index)).any() and fallback('nan_or_empty_baseline'):
                    frame=processor(batch)
                # ReaderProcessor is unchanged. Capture finish reasons via a pilot adapter below.
                finishes=getattr(engine,'last_finishes',{})
                frame['finish_reason']=[finishes.get(str(u),'unknown') for u in frame[UNIT_ID]]
                frame['fail_reason']=frame['reason']
                frame['dtype']=dtype
                repaired=reparse_with_salvage(frame,dict(zip(batch[UNIT_ID],batch[PINS_JSON])),cfg)
                frame['salvage_status']=repaired['salvage_status']
                frame['salvage_usable']=repaired['salvage_status'].isin(['natural','salvaged'])
                baseline.extend(frame.to_dict('records'))
                state['tokens_out']+=int(frame['tokens_out'].sum())
            except Exception as exc:
                baseline.extend(dict(unit_id=str(u),status='error',parse_ok=False,tokens_in=0,tokens_out=0,finish_reason='error',fail_reason=type(exc).__name__+': '+str(exc)) for u in batch[UNIT_ID])
            timings['baseline']=time.monotonic()-phase
            state['units_done']=len(records)+len(baseline)
            save_frame(out,'tape_baseline',baseline,cfg);checkpoint()
        score_units = sample.head(int(settings.get('score_sample_size',len(sample))))
        state['score_plan_n'] = len(score_units)
        state['stage']='cause_scores';phase=time.monotonic();scores=[];labels=cfg.get('causes.causes')
        for offset in range(0,len(score_units),settings['score_batch_events']):
            if not time_left():break
            batch=score_units.iloc[offset:offset+settings['score_batch_events']]
            try:
                prompts=[engine._prompt(scoring_request(processor._build(row),settings)) for _,row in batch.iterrows()]
                scored=engine.score_continuations([p for p in prompts for c in labels],labels*len(prompts))
                if any(i['error'] == 'nonfinite_label_logprob' for i in scored) and fallback('nan_cause_score'):
                    prompts=[engine._prompt(scoring_request(processor._build(row),settings)) for _,row in batch.iterrows()]
                    scored=engine.score_continuations([p for p in prompts for c in labels],labels*len(prompts))
                if len(scored) != len(batch)*len(labels):
                    raise ValueError('incomplete_label_scores')
                for index,(_,row) in enumerate(batch.iterrows()):
                    items=scored[index*len(labels):(index+1)*len(labels)]
                    rec=dict.fromkeys(table_columns(cfg,'cause_scores'))
                    rec.update(unit_id=str(row[UNIT_ID]),status='error',fail_reason=None,boundary_merge=any(i['boundary_merge'] for i in items),boundary_merge_labels=json.dumps([c for c,i in zip(labels,items) if i['boundary_merge']]))
                    rec.update({'label_tokens_'+c:i['label_tokens'] for c,i in zip(labels,items)})
                    rec.update({'logp_'+c:i['logp'] for c,i in zip(labels,items)})
                    errors=[i['error'] for i in items if i['error']]
                    if errors:
                        rec['fail_reason']=';'.join(sorted(set(errors)))
                    else:
                        p,gs,entropy=score_distribution({c:i['logp'] for c,i in zip(labels,items)},cfg.get('causes.groups'))
                        rec.update(status='ok',entropy=entropy,score_top=max(p,key=p.get))
                        rec.update({'p_'+c:v for c,v in p.items()});rec.update({'gs_'+g:v for g,v in gs.items()})
                    scores.append(rec)
            except Exception as exc:
                completed={r['unit_id'] for r in scores}
                scores.extend(dict(unit_id=str(u),status='error',fail_reason=type(exc).__name__+': '+str(exc),boundary_merge=False,**{'label_tokens_'+c:None for c in labels}) for u in batch[UNIT_ID] if str(u) not in completed)
            timings['cause_scores']=time.monotonic()-phase
            state['units_done']=len(records)+len(baseline)+len(scores)
            save_frame(out,'cause_scores',scores,cfg);checkpoint()
        for name,actual,expected in [('new_form',len(records),len(sample)),('baseline',len(baseline),len(baseline_units)),('cause_scores',len(scores),len(score_units))]:
            if actual<expected:
                state.setdefault('cut_steps',[]).append({'step':name,'reason':'deadline','n':actual,'planned_n':expected})
        state['partial']=bool(state.get('cut_steps'))
        state['engine']=engine.describe()
        state['stage']='complete' if not state['partial'] else 'partial'
    finally:
        timings['worker_seconds']=time.monotonic()-start
        state['ended_utc']=utc()
        if engine is not None:
            state['engine']=engine.describe()
        checkpoint()
        if engine is not None:
            engine.close()


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--settings',type=Path,default=ROOT/'pilot/pilot_reader_v2.yaml')
    parser.add_argument('--out',type=Path)
    parser.add_argument('--model-path')
    parser.add_argument('--max-minutes',type=float)
    parser.add_argument('--worker',action='store_true',help=argparse.SUPPRESS)
    parser.add_argument('--deadline',type=float,help=argparse.SUPPRESS)
    args=parser.parse_args(argv)
    settings=yaml.safe_load(args.settings.read_text())
    max_minutes=args.max_minutes if args.max_minutes is not None else settings['max_minutes']
    if not 0<max_minutes<=100:
        parser.error('--max-minutes must be > 0 and <= 100')
    cfg=Config.load(ROOT/'config')
    out=(args.out or ROOT/settings['output']).resolve()
    if args.worker:
        state=json.loads((out/'lineage.json').read_text())
        try:
            with MetricsPulse(state,settings['metrics_seconds']):
                execute(cfg,settings,out,state,args.deadline,explicit_model=args.model_path)
        except BaseException as exc:
            state['partial']=True
            state.setdefault('cut_steps',[]).append({'step':state.get('stage','startup'),'reason':type(exc).__name__+': '+str(exc)})
            save_state(out,state);write_report(out,cfg,state)
            raise
        return 0
    # Never overwrite a previous experiment or combine results across runs.
    if out.exists() and any(out.iterdir()):
        parser.error('output directory is nonempty; choose --out for a new run')
    out.mkdir(parents=True,exist_ok=True)
    started=float(os.environ.get('FINFM_PILOT_STARTED',time.monotonic()))
    deadline=started+max_minutes*60-settings['final_write_reserve_seconds']
    state={'started_utc':utc(),'partial':True,'stage':'preflight','cut_steps':[], 'config_hash':cfg.hash,
        'settings_sha256':hashlib.sha256(args.settings.read_bytes()).hexdigest(), 'settings':settings,
        'max_minutes':max_minutes,'model_path':args.model_path,'timing':{},
        'seed':cfg.get('runtime.base_seed'), 'source_dial_check':'not run', 'rule_lock':'uncommitted pilot; no pre-run GitHub lock; no claim of preregistration'}
    try:
        state['git_hash']=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip()
        state['git_working_tree_clean']=not bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True).strip())
    except subprocess.SubprocessError:
        state['git_hash']=os.environ.get('FINFM_PILOT_BUILD_GIT_HASH')
        state['git_working_tree_clean']=False
    state['source_archive_sha256']=os.environ.get('FINFM_PILOT_ARCHIVE_SHA256')
    for name in ('tape_v2','tape_baseline','cause_scores'):save_frame(out,name,[],cfg)
    try:
        state['gpus']=gpu_preflight(str(settings.get('gpu_model','A30')))
    except RuntimeError as exc:
        state['cut_steps']=[{'step':'gpu_preflight','reason':str(exc)}]
        state['ended_utc']=utc();save_state(out,state);write_report(out,cfg,state)
        print(str(exc),flush=True)
        return 2
    save_state(out,state);write_report(out,cfg,state)
    cmd=[sys.executable,str(ROOT/'scripts/run_reader_v2_pilot.py'),'--worker','--settings',str(args.settings.resolve()),'--out',str(out),'--deadline',str(deadline)]
    if args.model_path:cmd+=['--model-path',args.model_path]
    child=subprocess.Popen(cmd,start_new_session=True)
    cutoff=False
    try:
        remaining=max(0,deadline-time.monotonic())
        child.wait(timeout=remaining)
    except (subprocess.TimeoutExpired,KeyboardInterrupt):
        cutoff=True
    finally:
        # Also terminate any descendant engine processes after a worker exits or is killed.
        try:os.killpg(child.pid,signal.SIGTERM)
        except ProcessLookupError:pass
        if child.poll() is None:
            try:child.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:os.killpg(child.pid,signal.SIGKILL)
                except ProcessLookupError:pass
                child.wait(timeout=2)
        # A worker that exited can still leave descendants in its process group.
        try:os.killpg(child.pid,signal.SIGKILL)
        except ProcessLookupError:pass
    state=json.loads((out/'lineage.json').read_text())
    if cutoff or child.returncode:
        state['partial']=True
        state.setdefault('cut_steps',[]).append({'step':state.get('stage'),'reason':'wall_time_cutoff' if cutoff else 'worker_exit','exit_code':child.returncode})
    state['ended_utc']=utc();state['elapsed_seconds']=time.monotonic()-started
    state['config_hash_after']=Config.load(ROOT/'config').hash
    save_state(out,state);write_report(out,cfg,state)
    print('OUTPUT '+str(out),flush=True)
    return 3 if state['partial'] else 0
