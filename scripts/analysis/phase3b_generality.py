"""Full nominal common-observation bank, resumable native replay, and episode-level analysis.

CPU stages never import torch or a simulator. Full GPU replay is an explicit worker stage.
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3a_core import REVISION, SUITES, array_hash, category, protocol_hash, require, sha
from phase3b import ROOT, MODEL_SHA, fresh, require_server_audit, seal, verify_phase3a, verify_seal
from phase3b_distribution import capture_native_logits, distribution_metrics, load_model
from scripts.common.io import csv_save, read, save

ORIGINS = ('sft', 'grpo', 'sft_stochastic', 'grpo_stochastic')
TEMPERATURES = (1.0, 1.6)
RAW_FIELDS = ('main_image_policy', 'proprio_policy', 'pixel_values', 'input_ids', 'attention_mask')
STAGE_SOURCE = Path(__file__).resolve()


def uniform_queries(count, maximum=10):
    require(type(count) is int and count > 0 and maximum > 0, 'Invalid query count')
    selected=np.linspace(0,count-1,num=min(count,maximum),dtype=int).tolist()
    require(len(selected)==len(set(selected)), 'Duplicate uniform query index')
    return selected


def source_records(info, root, origin, mapping):
    if origin in ('sft','grpo'):
        return info['records'][origin]
    if origin=='grpo_stochastic':
        return info['records']['stochastic']
    run=root/'outputs/nominal/nominal-sft_base_only_stochastic-v1'
    require(run.is_dir(), f'Missing SFT stochastic source: {run}')
    manifest=read(run/'manifest.json'); protocol=manifest['protocol']
    require(protocol['checkpoint_revision']==REVISION['sft'] and protocol['do_sample'] is True
            and protocol['lora_applied'] is False and protocol['action_chunks']==8,
            'SFT stochastic source protocol mismatch')
    sys.path.insert(0,str(root/'scripts/evaluate'))
    from nominal_validate import validate_episode
    rows=[]
    for path in sorted((run/'rollouts').iterdir()):
        if not path.is_dir() or path.name.startswith('.partial-'): continue
        m=validate_episode(path,manifest['protocol_hash'])
        key=(m['suite'],m['task_id'])
        require(key in mapping and m['initial_state_sha256']==mapping[key]['trial_sha256'][m['trial_id']],
                'Stochastic source mapping mismatch')
        rows.append(dict(m,source_dir=str(path),policy='sft_stochastic'))
    require(len(rows)==150 and len({r['episode_id'] for r in rows})==150,
            'SFT stochastic source must have 150 complete episodes')
    return rows


def nominal_inventory(root, info):
    paths=dict(info['manifest']['inputs'])
    paths['sft_stochastic']=str(root/'outputs/nominal/nominal-sft_base_only_stochastic-v1')
    result={}
    for name,path in paths.items():
        p=Path(path)
        require((p/'manifest.json').is_file() and (p/'rollouts').is_dir(),f'Missing nominal run: {p}')
        mf=read(p/'manifest.json'); protocol=mf['protocol']
        expected_model='sft' if name.startswith('sft') else 'grpo'
        expected_sample=name in ('stochastic','sft_stochastic')
        require(protocol_hash(protocol)==mf['protocol_hash']
                and protocol['checkpoint_revision']==REVISION[expected_model]
                and protocol['do_sample'] is expected_sample
                and protocol['lora_applied'] is False
                and protocol['action_chunks']==8
                and protocol['unnorm_key']=='libero_130_no_noops_trajall',
                f'Nominal protocol mismatch: {name}')
        episode_dirs=[x for x in (p/'rollouts').iterdir()
                      if x.is_dir() and not x.name.startswith('.partial-')]
        counts=[read(x/'metadata.json')['num_queries'] for x in episode_dirs]
        require(len(counts)==150 and all(type(q) is int and q>0 for q in counts),
                f'Nominal episode/query inventory incomplete: {name}')
        result[name]=dict(path=str(p.resolve()),completed_episode_dirs=len(counts),
            query_count_min=min(counts),query_count_max=max(counts),query_count_total=sum(counts),
            manifest_sha256=sha(p/'manifest.json'),checkpoint_revision=protocol['checkpoint_revision'])
    return result


def _bank_readback(run, raw=True):
    bank=run/'bank'; manifest=read(bank/'manifest.json'); entries=manifest['observations']
    require(len(entries)==len({e['observation_id'] for e in entries}),'Duplicate observation ID')
    groups=collections.defaultdict(list)
    for e in entries:
        p=bank/e['file']; require(sha(p)==e['file_sha256'],'Bank observation checksum')
        with np.load(p,allow_pickle=False) as z:
            require(set(z.files)==set(e['arrays']),'Bank array fields')
            for k,v in e['arrays'].items():
                a=z[k]
                require(array_hash(a)==v['sha256'] and list(a.shape)==v['shape']
                        and str(a.dtype)==v['dtype'] and np.isfinite(a).all(),'Bank array mismatch')
        groups[e['source']].append(e)
    if raw:
        for source,rows in groups.items():
            require(sha(source)==rows[0]['source_sha256'],'Original observation changed')
            require(all(r['source_sha256']==rows[0]['source_sha256'] for r in rows),'Source hash conflict')
            metadata=Path(source).parent/'metadata.json'
            require(sha(metadata)==rows[0]['source_metadata_sha256'],'Original metadata changed')
            require(all(r['source_metadata_sha256']==rows[0]['source_metadata_sha256'] for r in rows),
                    'Source metadata hash conflict')
            with np.load(source,allow_pickle=False) as z:
                for e in rows:
                    q=e['query_index']
                    require(int(z['query_step'][q])==e['simulator_step'],'Original query index changed')
                    for k,v in e['arrays'].items():
                        require(array_hash(z[k][q])==v['sha256'],'Original observation readback')
    return dict(status='PASS',source_episodes=len(groups),observations=len(entries),
                raw_readback=raw,origins=dict(collections.Counter(e['source_policy'] for e in entries)))


def build_bank(args):
    require(__debug__,'PYTHONOPTIMIZE must be unset')
    require(len(set(args.origins))==len(args.origins) and set(args.origins)<=set(ORIGINS),'Invalid origins')
    primary=set(args.origins)=={'sft','grpo'}
    secondary=set(args.origins)=={'sft_stochastic','grpo_stochastic'}
    require(primary or secondary,'Keep deterministic and stochastic origins in separate banks')
    require(args.max_queries==10,'Stage 1 selection is pinned to at most 10 queries per episode')
    info=verify_phase3a(args.phase3a)
    require_server_audit(args.audit,args.phase3a,args.research_root)
    mapping_data=read(Path(info['manifest']['mapping']))
    mapping={(r['suite'],r['subset_task_id']):r for r in mapping_data['tasks']}
    require(len(mapping)==30,'Expected 30 mapped tasks')
    pairs={r['episode_id']:r for r in read(args.phase3a/'paired/pairs.json') if r['comparison']=='primary'}
    require(len(pairs)==150,'Expected 150 deterministic outcome pairs')
    deterministic={name:{r['episode_id']:r for r in info['records'][name]} for name in ('sft','grpo')}
    for eid,pair in pairs.items():
        require(eid in deterministic['sft'] and eid in deterministic['grpo'] and
                pair['category']==category(bool(deterministic['sft'][eid]['success']),
                                           bool(deterministic['grpo'][eid]['success'])),
                'Paired deterministic outcome mismatch')
    inventory=nominal_inventory(args.research_root,info)
    require(all(v['completed_episode_dirs']==150 for v in inventory.values()),'Nominal run incomplete')
    records={origin:source_records(info,args.research_root,origin,mapping) for origin in args.origins}
    require(all(len(v)==150 for v in records.values()),'Source episode count')
    index={(origin,r['episode_id']):r for origin,rr in records.items() for r in rr}
    require(len(index)==150*len(args.origins),'Duplicate source episode')
    if primary:
        require(set(r['episode_id'] for r in records['sft'])==set(r['episode_id'] for r in records['grpo']),
                'Deterministic source pairing')
    estimate=0
    for origin,rr in records.items():
        for m in rr:
            source=Path(m['source_dir'])/'observations.npz'
            require(source.is_file(),'Missing raw observations')
            estimate+=source.stat().st_size*min(args.max_queries,m['num_queries'])/m['num_queries']
    args.output.parent.mkdir(parents=True,exist_ok=True)
    free=shutil.disk_usage(args.output.parent).free
    require(free>2*estimate,'Insufficient free disk for bank estimate')
    run=fresh(args.output); bank=run/'bank'; (bank/'observations').mkdir(parents=True)
    entries=[]; episodes=[]
    source_hashes=info['manifest']['source_files']
    for origin in args.origins:
        for m in sorted(records[origin],key=lambda r:r['episode_id']):
            eid=m['episode_id']; pair=pairs[eid]; source=Path(m['source_dir'])/'observations.npz'
            metadata=Path(m['source_dir'])/'metadata.json'
            expected_sha=source_hashes.get(str(source))
            source_sha=sha(source)
            if expected_sha: require(source_sha==expected_sha,'Phase 3A source observation changed')
            require(read(metadata)['episode_id']==eid,'Raw source metadata identity')
            require(m['initial_state_sha256']==pair['initial_state_sha256'],'Paired initial state mismatch')
            require(m['suite'] in SUITES and m['task_id']==pair['task_id'] and m['trial_id']==pair['trial_id'],
                    'Source task/initial-state linkage')
            key=(m['suite'],m['task_id'])
            require(m['initial_state_sha256']==mapping[key]['trial_sha256'][m['trial_id']],
                    'Mapped initial-state mismatch')
            with np.load(source,allow_pickle=False) as z:
                qcount=m['num_queries']; require(len(z['query_step'])==qcount,'Query count changed')
                require(np.array_equal(z['query_step'],np.arange(qcount)*8),'Query-step alignment')
                require(set(RAW_FIELDS)<=set(z.files),'Missing native input fields')
                chosen=uniform_queries(qcount,args.max_queries)
                for q in chosen:
                    oid=f'{origin}-{eid}-q{q:04d}'
                    data={k:z[k][q] for k in z.files if k!='query_step'}
                    require(all(np.isfinite(a).all() for a in data.values()),'Nonfinite raw observation')
                    target=bank/'observations'/(oid+'.npz')
                    with target.open('xb') as f: np.savez_compressed(f,**data)
                    arrays={k:dict(sha256=array_hash(a),shape=list(a.shape),dtype=str(a.dtype)) for k,a in data.items()}
                    entries.append(dict(observation_id=oid,source_policy=origin,source_episode=eid,
                        suite=m['suite'],task_id=m['task_id'],trial_id=m['trial_id'],
                        outcome=pair['category'],source_success=bool(m['success']),
                        initial_state_sha256=m['initial_state_sha256'],language=m['language'],
                        query_index=q,simulator_step=int(z['query_step'][q]),source=str(source),
                        source_sha256=source_sha,source_metadata_sha256=sha(metadata),
                        file=str(target.relative_to(bank)),file_sha256=sha(target),arrays=arrays))
            episodes.append(dict(origin=origin,episode_id=eid,suite=m['suite'],task_id=m['task_id'],
                                 trial_id=m['trial_id'],queries=qcount,selected=len(chosen),
                                 success=bool(m['success']),outcome=pair['category']))
    require(len(episodes)==150*len(args.origins) and len(entries)<=3000,'Bank episode/observation count')
    require(len({e['observation_id'] for e in entries})==len(entries),'Duplicate selected observation')
    csv_save(bank/'observation_index.csv',[{k:e[k] for k in ('observation_id','source_policy','source_episode','suite','task_id','trial_id','outcome','source_success','initial_state_sha256','query_index','simulator_step','file','file_sha256')} for e in entries])
    save(bank/'manifest.json',dict(scope='primary' if primary else 'secondary',origins=args.origins,
         phase3a=str(args.phase3a.resolve()),phase3a_validation_sha256=sha(args.phase3a/'validation.json'),
         audit=str(args.audit.resolve()),audit_validation_sha256=sha(args.audit/'validation.json'),
         nominal_inventory=inventory,mapping_sha256=sha(info['manifest']['mapping']),
         protocols={k:info['manifest']['protocols'][k] for k in ('sft','grpo')},
         source_episodes=episodes,observations=entries,selection='uniform np.linspace integer indices including first/last; at most 10 unique queries per source episode',
         raw_fields=list(RAW_FIELDS),estimated_bank_bytes=int(estimate),free_bytes_before=free,
         stage_source_sha256=sha(STAGE_SOURCE),native_model_sha256=MODEL_SHA))
    check=_bank_readback(run,raw=True)
    seal(bank,dict(check,expected_episodes=len(episodes),disk_estimate_bytes=int(estimate)))
    commit=subprocess.run(['git','rev-parse','HEAD'],cwd=args.research_root,capture_output=True,text=True,check=True).stdout.strip()
    save(run/'run_manifest.json',dict(scope='primary' if primary else 'secondary',phase3a=str(args.phase3a.resolve()),
         bank_validation_sha256=sha(bank/'validation.json'),expected_source_episodes=len(episodes),
         expected_observations=len(entries),temperatures=list(TEMPERATURES),models=['sft','grpo'],
         git_commit=commit,upstream_commit=info['manifest']['protocols']['sft']['upstream_commit'],
         model_revisions={model:REVISION[model] for model in ('sft','grpo')},
         full_gpu_replay='manual worker commands only',stage_source_sha256=sha(STAGE_SOURCE)))
    print(json.dumps(dict(check,disk_estimate_bytes=int(estimate),free_bytes_before=free),indent=2))


def validate_bank(args):
    run=args.run.resolve(); verify_seal(run/'bank')
    result=_bank_readback(run,raw=True)
    require(result['source_episodes']==read(run/'run_manifest.json')['expected_source_episodes'],
            'Source episode count changed')
    print(json.dumps(result,indent=2))


def _worker_dir(run,model,smoke=False):
    return run/('smoke' if smoke else 'workers')/model


def _check_worker_record(directory,e,model,manifest):
    path=directory/'records'/e['observation_id']; record=read(path/'record.json')
    require(record['observation_id']==e['observation_id'] and record['model']==model,
            'Worker record identity')
    require(record['input_checksum']==e['file_sha256'] and
            record['bank_validation_sha256']==manifest['bank_validation_sha256'],
            'Worker input/protocol mismatch')
    require(sha(path/'logits.npz')==record['file_sha256'],'Completed logits checksum changed')
    with np.load(path/'logits.npz',allow_pickle=False) as z:
        require(set(z.files)=={'logits','action_tokens','decoded_actions','normalized_actions'},
                'Worker artifact schema')
        require(z['logits'].shape==(56,manifest['bins']) and np.isfinite(z['logits']).all(),
                'Worker logits shape/finite')
        require(z['action_tokens'].shape==(8,7) and z['decoded_actions'].shape==(8,7),
                'Worker action schema')
        require(z['normalized_actions'].shape==(8,7) and
                np.isfinite(z['decoded_actions']).all() and np.isfinite(z['normalized_actions']).all(),
                'Worker action finite/shape')
        require(np.array_equal(z['action_tokens'].reshape(-1),z['logits'].argmax(-1)+manifest['token_start']),
                'Worker token/logit mismatch')
    return record


def worker(args):
    run=args.run.resolve(); bank=run/'bank'; verify_seal(bank)
    bm=read(bank/'manifest.json'); rm=read(run/'run_manifest.json')
    require(rm['bank_validation_sha256']==sha(bank/'validation.json'),'Run/bank mismatch')
    require(bm['stage_source_sha256']==rm['stage_source_sha256']==sha(STAGE_SOURCE),
            'Bank/worker code changed')
    require_server_audit(Path(bm['audit']),Path(bm['phase3a']),args.research_root)
    entries=bm['observations'][:5] if args.smoke else bm['observations']
    directory=_worker_dir(run,args.model,args.smoke)
    if directory.exists():
        require(args.resume,'Existing worker requires --resume')
        for partial in sorted(directory.glob('.partial-*')):
            destination=run/'interrupted'/args.model/partial.name
            destination.parent.mkdir(parents=True,exist_ok=True)
            require(not destination.exists(),'Interrupted worker archive collision')
            partial.rename(destination)
            print('ARCHIVE_INCOMPLETE',partial,destination,flush=True)
        manifest=read(directory/'manifest.json') if (directory/'manifest.json').exists() else None
        if manifest is not None:
            require(manifest['model']==args.model and manifest['mode']==('smoke' if args.smoke else 'full')
                    and manifest['bank_validation_sha256']==sha(bank/'validation.json')
                    and manifest['checkpoint_revision']==REVISION[args.model]
                    and manifest['stage_source_sha256']==sha(STAGE_SOURCE)
                    and manifest['capture_source_sha256']==sha(Path(capture_native_logits.__code__.co_filename)),
                    'Worker protocol/checkpoint/source changed')
        (directory/'records').mkdir(exist_ok=True)
        existing={p.name for p in (directory/'records').iterdir() if p.is_dir()}
        require(manifest is not None or not existing,'Records without worker manifest')
        require(existing<={e['observation_id'] for e in entries},'Unexpected worker record')
        lookup={e['observation_id']:e for e in entries}
        for oid in sorted(existing): _check_worker_record(directory,lookup[oid],args.model,manifest)
        if len(existing)==len(entries):
            if (directory/'validation.json').exists(): verify_seal(directory)
            else: seal(directory,dict(status='PASS',model=args.model,mode=manifest['mode'],
                                      observations=len(existing),finite_logits=True,native_slots=56))
            print(json.dumps(dict(status='PASS',model=args.model,completed=len(existing),resumed=True),indent=2))
            return
        require(not (directory/'validation.json').exists(),'Sealed worker incomplete')
    else:
        directory.mkdir(parents=True); (directory/'records').mkdir()
        existing=set()
    remaining=[e for e in entries if e['observation_id'] not in existing]
    from phase3b_distribution import load_model
    model,core,cfg=load_model(args.research_root,Path(bm['phase3a']),args.model)
    import torch
    lo=core.vocab_size-core.config.n_action_bins; hi=core.vocab_size
    require(core.action_dim==7 and core.num_action_chunks==8,'Native 56-slot action shape')
    free,total=torch.cuda.mem_get_info(); print('GPU_AFTER_LOAD',args.model,'free_mib',free//2**20,'total_mib',total//2**20,flush=True)
    expected=dict(model=args.model,mode='smoke' if args.smoke else 'full',
        bank_validation_sha256=sha(bank/'validation.json'),checkpoint_revision=REVISION[args.model],
        model_config_hash=protocol_hash(cfg),
        runtime_model_config=cfg,token_start=lo,bins=hi-lo,
        token_to_bin_index=np.clip(core.vocab_size-np.arange(lo,hi)-1,0,len(core.bin_centers)-1).tolist(),
        action_norm_stats=core._get_action_stats(),stage_source_sha256=sha(STAGE_SOURCE),
        capture_source_sha256=sha(Path(capture_native_logits.__code__.co_filename)),
        native_model_sha256=MODEL_SHA,temperature_inference=1.0,
        temperature_analysis=list(TEMPERATURES),native_parallel_action_slots=56)
    if (directory/'manifest.json').exists():
        require(read(directory/'manifest.json')==expected,'Worker runtime configuration changed')
    else: save(directory/'manifest.json',expected)
    count=0
    for index,e in enumerate(entries):
        if e['observation_id'] in existing: continue
        source=bank/e['file']; require(sha(source)==e['file_sha256'],'Bank input changed')
        with np.load(source,allow_pickle=False) as z: raw={k:v.copy() for k,v in z.items()}
        require(all(array_hash(v)==e['arrays'][k]['sha256'] for k,v in raw.items()),
                'Raw input array changed')
        logits,actions,fi,actual_lo,actual_hi=capture_native_logits(
            model,core,raw,e['language'],3101+index,check_equivalence=(count==0))
        require((actual_lo,actual_hi)==(lo,hi),'Action token support changed')
        normalized=core.bin_centers[np.clip(core.vocab_size-fi['action_tokens']-1,0,len(core.bin_centers)-1)]
        record_dir=directory/'records'/e['observation_id']
        require(not record_dir.exists(),'Duplicate worker output')
        temp=Path(tempfile.mkdtemp(prefix='.partial-'+e['observation_id']+'-',dir=directory))
        try:
            with (temp/'logits.npz').open('xb') as f:
                np.savez_compressed(f,logits=logits,action_tokens=fi['action_tokens'],
                                    decoded_actions=actions,normalized_actions=normalized)
            hashes={k:array_hash(fi[k]) for k in ('input_ids','attention_mask','pixel_values')}
            save(temp/'record.json',dict(observation_id=e['observation_id'],model=args.model,
                input_checksum=e['file_sha256'],bank_validation_sha256=sha(bank/'validation.json'),
                file_sha256=sha(temp/'logits.npz'),processed_tensor_hashes=hashes,
                processed_tensor_shapes={k:list(fi[k].shape) for k in hashes},
                saved_preprocessing_equal={k:hashes[k]==e['arrays'][k]['sha256'] for k in hashes}))
            os.replace(temp,record_dir)
        except Exception:
            # Preserve a partial directory for inspection; it is never a committed record.
            raise
        count+=1
        print(args.model,len(existing)+count,'/',len(entries),e['observation_id'],flush=True)
        if args.max_new and count>=args.max_new: break
    completed=len(existing)+count
    if completed==len(entries):
        seal(directory,dict(status='PASS',model=args.model,mode=expected['mode'],observations=completed,
                            finite_logits=True,native_slots=56))
    print(json.dumps(dict(status='PASS' if completed==len(entries) else 'PARTIAL',
                          model=args.model,completed=completed,expected=len(entries)),indent=2))


def validate_workers(args):
    run=args.run.resolve(); verify_seal(run/'bank'); bm=read(run/'bank/manifest.json')
    entries=bm['observations'][:5] if args.smoke else bm['observations']
    manifests={}
    for model in ('sft','grpo'):
        directory=_worker_dir(run,model,args.smoke); verify_seal(directory)
        require(not list(directory.glob('.partial-*')),'Uncommitted worker partial remains')
        m=read(directory/'manifest.json')
        require(m['model']==model and m['bank_validation_sha256']==sha(run/'bank/validation.json')
                and m['checkpoint_revision']==REVISION[model] and m['native_model_sha256']==MODEL_SHA
                and m['stage_source_sha256']==sha(STAGE_SOURCE)
                and m['capture_source_sha256']==sha(Path(capture_native_logits.__code__.co_filename))
                and m['mode']==('smoke' if args.smoke else 'full'),
                'Worker manifest mismatch')
        found={p.name for p in (directory/'records').iterdir() if p.is_dir()}
        require(found=={e['observation_id'] for e in entries},'Missing/extra worker observation')
        for e in entries: _check_worker_record(directory,e,model,m)
        manifests[model]=m
    require(manifests['sft']['token_start']==manifests['grpo']['token_start'] and
            manifests['sft']['bins']==manifests['grpo']['bins'] and
            manifests['sft']['token_to_bin_index']==manifests['grpo']['token_to_bin_index'],
            'Cross-model action support mismatch')
    print(json.dumps(dict(status='PASS',mode='smoke' if args.smoke else 'full',
                          observations=len(entries),models=['sft','grpo']),indent=2))


OBS_METRICS=('entropy_sft','entropy_grpo','delta_entropy','top1_sft','top1_grpo',
             'delta_top1','top5_sft','top5_grpo','top10_sft','top10_grpo',
             'greedy_disagreement','kl_sft_grpo','kl_grpo_sft','js')


def observation_metric_rows(e,logits_sft,logits_grpo):
    rows=[]
    for temperature in TEMPERATURES:
        metrics=distribution_metrics(logits_sft,logits_grpo,temperature)
        row={k:float(np.mean(v)) for k,v in metrics.items()}
        row['delta_entropy']=row['entropy_grpo']-row['entropy_sft']
        row['delta_top1']=row['top1_grpo']-row['top1_sft']
        rows.append(dict(observation_id=e['observation_id'],episode_id=e['source_episode'],
            suite=e['suite'],task_id=e['task_id'],trial_id=e['trial_id'],
            origin=e['source_policy'],outcome=e['outcome'],source_success=e['source_success'],
            query_index=e['query_index'],simulator_step=e['simulator_step'],
            temperature=temperature,**row))
    return rows


def aggregate_episodes(observations):
    grouped=collections.defaultdict(list)
    for r in observations:
        grouped[(r['origin'],r['episode_id'],r['temperature'])].append(r)
    episodes=[]
    for (_,_,_),rows in sorted(grouped.items()):
        first=rows[0]
        require(all((r['suite'],r['task_id'],r['trial_id'],r['outcome'])==
                    (first['suite'],first['task_id'],first['trial_id'],first['outcome']) for r in rows),
                'Observation/episode linkage')
        means={k:float(np.mean([r[k] for r in rows])) for k in OBS_METRICS}
        episodes.append(dict(origin=first['origin'],episode_id=first['episode_id'],suite=first['suite'],
            task_id=first['task_id'],trial_id=first['trial_id'],outcome=first['outcome'],
            source_success=first['source_success'],temperature=first['temperature'],
            n_observations=len(rows),median_observation_delta_entropy=float(np.median([r['delta_entropy'] for r in rows])),
            fraction_observations_delta_negative=float(np.mean([r['delta_entropy']<0 for r in rows])),
            **means))
    return episodes


def _distribution_summary(rows):
    require(rows,'Empty summary group')
    delta=np.array([r['delta_entropy'] for r in rows])
    return dict(source_episodes=len(rows),observations=sum(r['n_observations'] for r in rows),
        mean_delta_entropy=float(delta.mean()),median_delta_entropy=float(np.median(delta)),
        variance_delta_entropy=float(delta.var()),p25_delta_entropy=float(np.quantile(delta,.25)),
        p75_delta_entropy=float(np.quantile(delta,.75)),fraction_negative=float(np.mean(delta<0)),
        mean_entropy_sft=float(np.mean([r['entropy_sft'] for r in rows])),
        mean_entropy_grpo=float(np.mean([r['entropy_grpo'] for r in rows])),
        mean_delta_top1=float(np.mean([r['delta_top1'] for r in rows])),
        mean_js=float(np.mean([r['js'] for r in rows])),
        mean_greedy_disagreement=float(np.mean([r['greedy_disagreement'] for r in rows])))


def task_cluster_ci(episodes,seed=3101,iterations=2000):
    """Resample whole tasks within suite; keep both origins and all trials together."""
    require(iterations>0 and episodes,'Invalid task bootstrap')
    grouped=collections.defaultdict(list)
    for r in episodes: grouped[(r['suite'],r['task_id'])].append(r['delta_entropy'])
    suites=collections.defaultdict(list)
    for (suite,task),values in sorted(grouped.items()): suites[suite].append(values)
    rng=np.random.default_rng(seed); draws=[]
    for _ in range(iterations):
        sampled=[]
        for values in suites.values():
            for idx in rng.integers(len(values),size=len(values)):
                sampled.extend(values[idx])
        draws.append(float(np.mean(sampled)))
    return dict(ci95=np.quantile(draws,[.025,.975]).tolist(),seed=seed,iterations=iterations,
                clusters=len(grouped),scheme='Whole tasks sampled within each suite; observations and episodes stay in their task')


def aggregate_groups(episodes):
    output={}
    for name,fields in [('tasks',('suite','task_id')),('suites',('suite',)),('outcomes',('outcome',))]:
        grouped=collections.defaultdict(list)
        for r in episodes:
            for origin in ('all',r['origin']):
                grouped[tuple(r[f] for f in fields)+(origin,r['temperature'])].append(r)
        rows=[]
        for key,group in sorted(grouped.items()):
            row=dict(zip(fields+('origin','temperature'),key),**_distribution_summary(group))
            if name=='suites':
                task_means=collections.defaultdict(list)
                for r in group: task_means[r['task_id']].append(r['delta_entropy'])
                row['macro_task_mean_delta_entropy']=float(np.mean([np.mean(v) for v in task_means.values()]))
                row['task_count']=len(task_means)
            rows.append(row)
        output[name]=rows
    return output


def summarize(args):
    run=args.run.resolve(); validate_workers(argparse.Namespace(run=run,smoke=False))
    bm=read(run/'bank/manifest.json'); entries=bm['observations']
    require(bm['stage_source_sha256']==sha(STAGE_SOURCE),'Bank/summary code changed')
    observations=[]; processed_equal=0
    for e in entries:
        records=[]; logits=[]
        for model in ('sft','grpo'):
            directory=_worker_dir(run,model); manifest=read(directory/'manifest.json')
            record=_check_worker_record(directory,e,model,manifest)
            records.append(record)
            with np.load(directory/'records'/e['observation_id']/'logits.npz',allow_pickle=False) as z:
                logits.append(z['logits'].copy())
        require(records[0]['input_checksum']==records[1]['input_checksum']==e['file_sha256'],
                'Cross-model raw observation mismatch')
        equal=records[0]['processed_tensor_hashes']==records[1]['processed_tensor_hashes']
        processed_equal+=int(equal)
        for row in observation_metric_rows(e,*logits):
            row['processed_inputs_equal']=equal
            observations.append(row)
    episodes=aggregate_episodes(observations)
    require(len(episodes)==2*len(bm['source_episodes']), 'Episode/temperature count')
    grouped=aggregate_groups(episodes)
    overall={}
    for t in TEMPERATURES:
        current=[r for r in episodes if r['temperature']==t]
        current_observations=[r for r in observations if r['temperature']==t]
        overall[str(t)]=dict(_distribution_summary(current),
            observation_count=len(current_observations),
            observation_fraction_negative=float(np.mean([r['delta_entropy']<0 for r in current_observations])),
            observation_fraction_positive=float(np.mean([r['delta_entropy']>0 for r in current_observations])),
            episode_fraction_positive=float(np.mean([r['delta_entropy']>0 for r in current])),
            macro_task_mean_delta_entropy=float(np.mean([r['mean_delta_entropy'] for r in grouped['tasks']
                if r['origin']=='all' and r['temperature']==t])),
            task_cluster_bootstrap=task_cluster_ci(current))
    analysis=fresh(run/'analysis')
    sft_worker=read(run/'workers/sft/manifest.json')
    grpo_worker=read(run/'workers/grpo/manifest.json')
    csv_save(analysis/'observations.csv',observations)
    csv_save(analysis/'episodes.csv',episodes)
    csv_save(analysis/'delta_entropy.csv',[{k:r[k] for k in ('episode_id','suite','task_id','trial_id','origin','outcome','temperature','n_observations','entropy_sft','entropy_grpo','delta_entropy','delta_top1','js','greedy_disagreement')} for r in episodes])
    for name,rows in grouped.items(): csv_save(analysis/(name+'.csv'),rows)
    save(analysis/'metrics.json',dict(scope=bm['scope'],temperatures=list(TEMPERATURES),
        source_episodes=len(bm['source_episodes']),observations=len(entries),
        paired_raw_observations=len(entries),processed_inputs_equal=processed_equal,
        processed_inputs_different=len(entries)-processed_equal,overall=overall,
        action_normalization_equal=sft_worker['action_norm_stats']==grpo_worker['action_norm_stats'],
        model_revisions={model:REVISION[model] for model in ('sft','grpo')},
        input_validation_sha256=dict(bank=sha(run/'bank/validation.json'),
            sft=sha(run/'workers/sft/validation.json'),grpo=sha(run/'workers/grpo/validation.json')),
        stage_source_sha256=sha(STAGE_SOURCE),
        aggregation='56 slots mean per observation; selected observations mean per source episode; equal source episodes for overall and grouped micro means; equal tasks for macro means',
        uncertainty='Task cluster bootstrap is descriptive for these 30 fixed nominal tasks; no OOD claim',
        source_origin_bias='SFT-origin and GRPO-origin trajectories may differ despite matching initial state; never treated as identical observations'))
    print(json.dumps(dict(status='PASS',source_episodes=len(bm['source_episodes']),observations=len(entries),
                          episode_rows=len(episodes),task_rows=len(grouped['tasks'])),indent=2))


def _csv_rows(path):
    with path.open(newline='') as f: return list(csv.DictReader(f))


def _plot_inputs(analysis,temperature):
    episodes=[r for r in _csv_rows(analysis/'episodes.csv') if float(r['temperature'])==temperature]
    tasks=[r for r in _csv_rows(analysis/'tasks.csv') if float(r['temperature'])==temperature and r['origin']=='all']
    outcomes=[r for r in _csv_rows(analysis/'outcomes.csv') if float(r['temperature'])==temperature and r['origin']=='all']
    require(len(episodes)==read(analysis/'metrics.json')['source_episodes'] and len(tasks)==30,
            'Figure input episode/task coverage')
    require(len({r['observation_id'] for r in _csv_rows(analysis/'observations.csv')
                 if float(r['temperature'])==temperature})==read(analysis/'metrics.json')['observations'],
            'Figure observation coverage')
    return episodes,tasks,outcomes


def _save_figure(fig,folder,stem):
    fig.tight_layout()
    fig.savefig(folder/(stem+'.png'),dpi=180)
    fig.savefig(folder/(stem+'.pdf'))
    import matplotlib.pyplot as plt
    plt.close(fig)


def figures(args):
    run=args.run.resolve(); analysis=run/'analysis'; metrics=read(analysis/'metrics.json')
    require(metrics['scope']=='primary','Stage 1 figures require the primary bank')
    folder=analysis/'figures'; require(not folder.exists(),'Figures already exist')
    folder.mkdir()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors={'libero_goal':'#2c7bb6','libero_spatial':'#d95f02','libero_10':'#1b9e77'}
    names={'libero_goal':'Goal','libero_spatial':'Spatial','libero_10':'Long'}
    markers={'sft':'o','grpo':'^'}
    stems=[]
    for temperature in TEMPERATURES:
        episodes,tasks,outcomes=_plot_inputs(analysis,temperature)
        suffix='T'+str(temperature).replace('.','p')
        delta=np.array([float(r['delta_entropy']) for r in episodes])
        top1=np.array([float(r['delta_top1']) for r in episodes])
        js=np.array([float(r['js']) for r in episodes])
        overall=metrics['overall'][str(temperature)]
        # Figure 1: episode is the paired unit; lines are deliberately faint.
        fig,axes=plt.subplots(1,3,figsize=(13,4.2))
        for row in episodes:
            color=colors[row['suite']]
            axes[0].plot((0,1),(float(row['entropy_sft']),float(row['entropy_grpo'])),
                         color=color,alpha=.055,lw=.7,ls='-' if row['origin']=='sft' else '--')
            axes[1].plot((0,1),(float(row['top1_sft']),float(row['top1_grpo'])),
                         color=color,alpha=.055,lw=.7,ls='-' if row['origin']=='sft' else '--')
        for ax,key1,key2,label in ((axes[0],'entropy_sft','entropy_grpo','Entropy (nats)'),
                                    (axes[1],'top1_sft','top1_grpo','Top-1 probability')):
            ax.plot((0,1),(np.mean([float(r[key1]) for r in episodes]),
                             np.mean([float(r[key2]) for r in episodes])),
                    marker='D',markersize=8,color='black',lw=2,label='Equal episode mean')
            ax.set(xticks=(0,1),xticklabels=('SFT','GRPO'),ylabel=label)
        for suite in SUITES:
            values=sorted(float(r['js']) for r in episodes if r['suite']==suite)
            axes[2].step(values,np.arange(1,len(values)+1)/len(values),where='post',
                         label=names[suite],color=colors[suite])
        axes[2].set(xlabel='Episode mean JS divergence (nats)',ylabel='ECDF')
        axes[0].plot([],[],color='gray',ls='-',label='SFT origin')
        axes[0].plot([],[],color='gray',ls='--',label='GRPO origin')
        axes[0].legend(loc='best',fontsize=8); axes[2].legend(loc='lower right',fontsize=8)
        fig.suptitle(f'Same observations, two native distributions | T={temperature:g} | '
                     f'{len(episodes)} source episodes, {metrics["observations"]} observations')
        stem='figure1_expanded_paired_entropy_'+suffix; _save_figure(fig,folder,stem); stems.append(stem)
        # Figure 2: episode-level delta distribution and clustered uncertainty.
        fig,ax=plt.subplots(figsize=(7.8,4.8))
        ax.hist(delta,bins=35,color='#6485a8',alpha=.8)
        ax.axvline(0,color='black',lw=1); ax.axvline(delta.mean(),color='#b2182b',lw=1.5,label='Mean')
        ax.axvline(np.median(delta),color='#2166ac',lw=1.5,ls='--',label='Median')
        ci=overall['task_cluster_bootstrap']['ci95']
        ax.set(xlabel='ΔH = H(GRPO) − H(SFT), nats',ylabel='Source episodes',
               title=f'T={temperature:g}; mean {delta.mean():.3g}, median {np.median(delta):.3g}; '
                     f'negative {np.mean(delta<0):.1%}; task CI [{ci[0]:.3g}, {ci[1]:.3g}]')
        ax.legend(); stem='figure2_paired_entropy_change_'+suffix; _save_figure(fig,folder,stem); stems.append(stem)
        # Figure 3: one episode-level point per source trajectory.
        fig,ax=plt.subplots(figsize=(6.3,6.0))
        for suite in SUITES:
            for origin in ('sft','grpo'):
                rr=[r for r in episodes if r['suite']==suite and r['origin']==origin]
                ax.scatter([float(r['entropy_sft']) for r in rr],[float(r['entropy_grpo']) for r in rr],
                    c=colors[suite],marker=markers[origin],s=24,alpha=.58,
                    label=f'{names[suite]} / {origin} origin')
        limits=[min(float(r['entropy_sft']) for r in episodes),max(float(r['entropy_grpo']) for r in episodes)]
        lo=min(limits); hi=max(max(float(r['entropy_sft']) for r in episodes),limits[1])
        ax.plot((lo,hi),(lo,hi),color='black',ls='--',lw=1)
        ax.set(xlabel='SFT entropy (nats)',ylabel='GRPO entropy (nats)',title=f'Episode means, T={temperature:g}')
        ax.legend(fontsize=7,ncol=2); stem='figure3_episode_entropy_scatter_'+suffix; _save_figure(fig,folder,stem); stems.append(stem)
        # Figure 4: all 30 mapped tasks, with episode IQR and median.
        ordered=sorted(tasks,key=lambda r:(SUITES.index(r['suite']),int(r['task_id'])))
        fig,ax=plt.subplots(figsize=(8.2,9.6)); y=np.arange(len(ordered))
        for i,r in enumerate(ordered):
            mean=float(r['mean_delta_entropy']); p25=float(r['p25_delta_entropy']); p75=float(r['p75_delta_entropy'])
            ax.plot((p25,p75),(i,i),color=colors[r['suite']],lw=2,alpha=.7)
            ax.scatter(mean,i,color=colors[r['suite']],s=27)
            ax.scatter(float(r['median_delta_entropy']),i,color='black',marker='|',s=80)
        ax.axvline(0,color='black',lw=1)
        ax.set(yticks=y,yticklabels=[f'{names[r["suite"]]} t{int(r["task_id"]):02d} '
            f'(n={int(r["source_episodes"])}/{int(r["observations"])})' for r in ordered],
            xlabel='Episode mean ΔH (dot), median (tick), episode IQR (line)',title=f'Task variation, T={temperature:g}')
        ax.invert_yaxis(); stem='figure4_taskwise_change_'+suffix; _save_figure(fig,folder,stem); stems.append(stem)
        # Figure 5: outcomes use the same paired deterministic state label for both origins.
        cats=('both_success','gained','lost','both_fail')
        fig,axes=plt.subplots(2,2,figsize=(11,7))
        for ax,(key,label) in zip(axes.flat,(('delta_entropy','Δ entropy (nats)'),
                                            ('delta_top1','Δ top-1 probability'),
                                            ('js','JS divergence (nats)'),
                                            ('greedy_disagreement','Greedy slot disagreement'))):
            groups=[[float(r[key]) for r in episodes if r['outcome']==cat] for cat in cats]
            available=[(i+1,v) for i,v in enumerate(groups) if v]
            ax.boxplot([v for _,v in available],positions=[i for i,_ in available],showfliers=False)
            ax.set(xticks=range(1,5),xticklabels=[f'{cat}\n(n={len(groups[i])})' for i,cat in enumerate(cats)],ylabel=label)
            if key.startswith('delta'): ax.axhline(0,color='black',lw=.8)
        fig.suptitle(f'Paired-outcome strata | T={temperature:g}; small groups are unadjusted')
        stem='figure5_outcome_distribution_'+suffix; _save_figure(fig,folder,stem); stems.append(stem)
        # Figure 6: compact descriptive dashboard, all quantities episode weighted.
        fig,axes=plt.subplots(2,3,figsize=(12.5,7))
        axes[0,0].bar(['Overall ΔH'],[delta.mean()],color='#6485a8'); axes[0,0].axhline(0,color='black',lw=.8)
        axes[0,0].set_ylabel('nats'); axes[0,0].set_title('Equal episode mean')
        axes[0,1].bar(['ΔH < 0'],[np.mean(delta<0)],color='#1b9e77'); axes[0,1].set_ylim(0,1)
        axes[0,1].set_title('Fraction of source episodes')
        suite_means=[np.mean([float(r['delta_entropy']) for r in episodes if r['suite']==s]) for s in SUITES]
        axes[0,2].bar([names[s] for s in SUITES],suite_means,color=[colors[s] for s in SUITES]); axes[0,2].axhline(0,color='black',lw=.8)
        axes[0,2].set_title('Suite mean ΔH')
        axes[1,0].hist([float(r['mean_delta_entropy']) for r in tasks],bins=15,color='#a6761d')
        axes[1,0].axvline(0,color='black',lw=.8); axes[1,0].set_title('30 task means')
        axes[1,1].hist(top1,bins=25,color='#7570b3'); axes[1,1].axvline(0,color='black',lw=.8)
        axes[1,1].set_title('Episode Δ top-1')
        axes[1,2].hist(js,bins=25,color='#e7298a'); axes[1,2].set_title('Episode JS')
        fig.suptitle(f'Common-observation generality | T={temperature:g}; n={len(episodes)} episodes')
        stem='figure6_generality_dashboard_'+suffix; _save_figure(fig,folder,stem); stems.append(stem)
    save(analysis/'figure_index.json',dict(stems=stems,formats=['png','pdf'],
        source_episodes=metrics['source_episodes'],observations=metrics['observations'],
        temperatures=list(TEMPERATURES),figures_per_temperature=6))
    print(json.dumps(dict(status='PASS',figures=len(stems),formats=['png','pdf']),indent=2))


def report(args):
    run=args.run.resolve(); analysis=run/'analysis'; metrics=read(analysis/'metrics.json')
    require(not (analysis/'validation.json').exists() and not (analysis/'summary.md').exists(),
            'Report already exists')
    figures=read(analysis/'figure_index.json')
    require(len(figures['stems'])==12 and all((analysis/'figures'/(stem+'.'+ext)).is_file()
            for stem in figures['stems'] for ext in ('png','pdf')),'Missing figure coverage')
    episodes=_csv_rows(analysis/'episodes.csv'); tasks=_csv_rows(analysis/'tasks.csv')
    suites=_csv_rows(analysis/'suites.csv'); outcomes=_csv_rows(analysis/'outcomes.csv')
    require(len(episodes)==2*metrics['source_episodes'],'Episode summary count')
    lines=['# Common-observation distribution generality','',
        f'Scope: {metrics["scope"]}; {metrics["source_episodes"]} source episodes, '
        f'{metrics["observations"]} distinct selected observations, two model forwards per observation.',
        f'Processed input hashes equal in {metrics["processed_inputs_equal"]}/{metrics["observations"]} paired forwards.',
        f'Native action normalization stats equal across models: {metrics["action_normalization_equal"]}.',
        'Each observation is compared at T=1.0 and T=1.6 by offline softmax of the same raw logits.',
        'Slots are averaged within observation, observations within source episode, then episodes with equal weight. '
        'Macro task means weight each mapped task equally.','']
    for t in TEMPERATURES:
        s=metrics['overall'][str(t)]; ci=s['task_cluster_bootstrap']['ci95']
        lines.extend([f'## T={t:g}','',
            f'Mean ΔH (GRPO − SFT): {s["mean_delta_entropy"]:.6g} nats; median {s["median_delta_entropy"]:.6g}; '
            f'IQR [{s["p25_delta_entropy"]:.6g}, {s["p75_delta_entropy"]:.6g}]; '
            f'fraction negative {s["fraction_negative"]:.3%}.',
            f'Positive ΔH: {s["episode_fraction_positive"]:.3%} of source episodes and '
            f'{s["observation_fraction_positive"]:.3%} of selected observations '
            f'({s["observation_count"]} observations; observations are clustered within episodes).',
            f'Task-cluster bootstrap 95% interval [{ci[0]:.6g}, {ci[1]:.6g}] nats '
            f'({s["task_cluster_bootstrap"]["clusters"]} tasks, seed {s["task_cluster_bootstrap"]["seed"]}).',
            f'Macro task mean ΔH {s["macro_task_mean_delta_entropy"]:.6g}; '
            f'mean Δ top-1 {s["mean_delta_top1"]:.6g}; mean JS {s["mean_js"]:.6g}.',''])
    lines.extend(['## Task outcomes at T=1.6','',
        '| Suite | Task | Episodes | Observations | Mean ΔH | Median ΔH | Fraction ΔH<0 |',
        '|---|---:|---:|---:|---:|---:|---:|'])
    for r in sorted((r for r in tasks if r['origin']=='all' and float(r['temperature'])==1.6),
                    key=lambda r:(SUITES.index(r['suite']),int(r['task_id']))):
        lines.append(f'| {r["suite"]} | {int(r["task_id"])} | {int(r["source_episodes"])} | '
            f'{int(r["observations"])} | {float(r["mean_delta_entropy"]):.5g} | '
            f'{float(r["median_delta_entropy"]):.5g} | {float(r["fraction_negative"]):.1%} |')
    lines.extend(['','## Source and outcome strata at T=1.6','',
        '| Stratum | Source episodes | Mean ΔH | Fraction ΔH<0 | Mean Δ top-1 | Mean JS | Greedy disagreement |',
        '|---|---:|---:|---:|---:|---:|---:|'])
    for r in (x for x in suites if x['origin']!='all' and float(x['temperature'])==1.6):
        lines.append(f'| {r["suite"]} / {r["origin"]} | {int(r["source_episodes"])} | '
            f'{float(r["mean_delta_entropy"]):.5g} | {float(r["fraction_negative"]):.1%} | '
            f'{float(r["mean_delta_top1"]):.5g} | {float(r["mean_js"]):.5g} | '
            f'{float(r["mean_greedy_disagreement"]):.1%} |')
    for r in (x for x in outcomes if x['origin']=='all' and float(x['temperature'])==1.6):
        lines.append(f'| {r["outcome"]} | {int(r["source_episodes"])} | '
            f'{float(r["mean_delta_entropy"]):.5g} | {float(r["fraction_negative"]):.1%} | '
            f'{float(r["mean_delta_top1"]):.5g} | {float(r["mean_js"]):.5g} | '
            f'{float(r["mean_greedy_disagreement"]):.1%} |')
    lines.extend(['','## Interpretation limits','',
        'Source-origin strata are reported separately in tasks.csv, suites.csv, and outcomes.csv. '
        'SFT-origin and GRPO-origin trajectories begin at matched states but their later observations can differ.',
        'Action slots and observations within one episode are not independent episodes. '
        'Task-cluster intervals are descriptive for these fixed nominal tasks.',
        'This analysis does not establish a causal link between entropy and success, useful strategy diversity, '
        'or out-of-distribution robustness.'])
    (analysis/'summary.md').write_text('\n'.join(lines)+'\n')
    seal(analysis,dict(status='PASS',source_episodes=metrics['source_episodes'],
        observations=metrics['observations'],temperature_count=2,figures=12,
        figure_formats=['png','pdf'],scope=metrics['scope']))
    print(analysis/'summary.md')


def verify(args):
    run=args.run.resolve(); bank=verify_seal(run/'bank')
    _bank_readback(run,raw=True)
    rm=read(run/'run_manifest.json')
    require(rm['bank_validation_sha256']==sha(run/'bank/validation.json')
            and rm['stage_source_sha256']==sha(STAGE_SOURCE),
            'Run manifest/bank seal mismatch')
    validate_workers(argparse.Namespace(run=run,smoke=False))
    analysis=verify_seal(run/'analysis')
    inputs=read(run/'analysis/metrics.json')['input_validation_sha256']
    require(inputs==dict(bank=sha(run/'bank/validation.json'),
        sft=sha(run/'workers/sft/validation.json'),grpo=sha(run/'workers/grpo/validation.json')),
        'Analysis input seals changed')
    require(analysis['source_episodes']==bank['expected_episodes'] and
            analysis['observations']==bank['observations'],'Final bank/analysis count mismatch')
    print(json.dumps(dict(status='PASS',bank_observations=bank['observations'],
                          analysis_episodes=analysis['source_episodes'],figures=analysis['figures']),indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='stage',required=True)
    a=sub.add_parser('bank'); a.add_argument('--phase3a',type=Path,required=True)
    a.add_argument('--audit',type=Path,required=True); a.add_argument('--research-root',type=Path,default=ROOT)
    a.add_argument('--output',type=Path,required=True)
    a.add_argument('--origins',choices=ORIGINS,nargs='+',default=['sft','grpo'])
    a.add_argument('--max-queries',type=int,default=10)
    for stage in ('bank-validate','worker-validate','summarize','figures','report','verify'):
        a=sub.add_parser(stage); a.add_argument('--run',type=Path,required=True)
        if stage=='worker-validate': a.add_argument('--smoke',action='store_true')
    a=sub.add_parser('worker'); a.add_argument('--run',type=Path,required=True)
    a.add_argument('--model',choices=('sft','grpo'),required=True)
    a.add_argument('--research-root',type=Path,default=ROOT)
    a.add_argument('--smoke',action='store_true'); a.add_argument('--resume',action='store_true')
    a.add_argument('--max-new',type=int,default=0)
    args=p.parse_args()
    require(not hasattr(args,'max_new') or args.max_new>=0,'Negative chunk size')
    dispatch={'bank':build_bank,'bank-validate':validate_bank,'worker':worker,
              'worker-validate':validate_workers,'summarize':summarize,
              'figures':figures,'report':report,'verify':verify}
    dispatch[args.stage](args)


if __name__=='__main__': main()
