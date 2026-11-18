"""CPU-only loader and episode-weighted aggregation for Stage 2/3."""
from __future__ import annotations
import collections
import csv
import math
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT)); sys.path.insert(0,str(ROOT/'scripts/analysis'))
from scripts.common.integrity import require, sha, verify_seal
from scripts.common.io import read
from scripts.common.action_schema import validate_saved
from phase3b_generality import task_cluster_ci

IDENTITY=('observation_id','origin','episode_id','suite','task_id','trial_id','outcome','source_success')
BASE=('origin','episode_id','suite','task_id','trial_id','outcome','source_success')
KEYS=('temperature','group','scope')


def csv_rows(path):
    with Path(path).open(newline='') as f: return list(csv.DictReader(f))


def clean(x):
    if isinstance(x,dict): return {k:clean(v) for k,v in x.items()}
    if isinstance(x,np.ndarray): return clean(x.tolist())
    if isinstance(x,(list,tuple)): return [clean(v) for v in x]
    if isinstance(x,(np.bool_,)): return bool(x)
    if isinstance(x,(np.integer,)): return int(x)
    if isinstance(x,(np.floating,float)): return float(x) if np.isfinite(x) else None
    return x


def avg(values):
    x=np.asarray([np.nan if v is None else v for v in values],float)
    return float(np.nanmean(x)) if np.isfinite(x).any() else None


def load_record(run,model,entry,manifest):
    d=run/'workers'/model/'records'/entry['observation_id']; record=read(d/'record.json')
    require(record['observation_id']==entry['observation_id'] and record['model']==model,'Record identity')
    require(record['input_checksum']==entry['file_sha256'] and
            record['bank_validation_sha256']==manifest['bank_validation_sha256'],'Record bank linkage')
    require(sha(d/'logits.npz')==record['file_sha256'],'Logits checksum')
    with np.load(d/'logits.npz',allow_pickle=False) as z: data={k:z[k].copy() for k in z.files}
    validate_saved(data,manifest)
    return data,record


def check_analysis_copy(path):
    """A transferred missing display file is reported; missing/changed numeric input fails."""
    v=read(path/'validation.json'); require(v['status']=='PASS','Analysis was not sealed PASS')
    missing=[]
    for name,digest in v['output_hashes'].items():
        p=path/name
        if not p.is_file() and name.startswith('figures/'):
            missing.append(name); continue
        require(p.is_file() and sha(p)==digest,'Stage 1 numeric/derived file changed: '+name)
    return dict(status='PASS' if not missing else 'PARTIAL_COPY_FIGURES_MISSING',
                required_numeric_inputs='PASS',missing_figures=missing)


def latest_pass(root,stage1=False):
    good=[]
    for p in Path(root).glob('*'):
        if not p.is_dir(): continue
        try:
            if stage1:
                for d in ('bank','workers/sft','workers/grpo'): verify_seal(p/d)
                check_analysis_copy(p/'analysis')
            else: verify_seal(p)
            good.append(p)
        except (ValueError,KeyError,FileNotFoundError): continue
    require(good,f'No validated input at {root}')
    return max(good,key=lambda p:p.name).resolve()


def episode_means(rows,metrics):
    groups=collections.defaultdict(list)
    for r in rows: groups[tuple(r[k] for k in KEYS)+tuple(r[k] for k in ('origin','episode_id'))].append(r)
    result=[]
    for _,rr in sorted(groups.items()):
        first=rr[0]
        require(all(tuple(r[k] for k in BASE)==tuple(first[k] for k in BASE) for r in rr),'Outcome/episode alignment')
        result.append(dict({k:first[k] for k in (*BASE,*KEYS)},n_observations=len(rr),
                           **{k:avg([r[k] for r in rr]) for k in metrics}))
    return result


def initial_states(episodes,metrics):
    groups=collections.defaultdict(list)
    for r in episodes: groups[tuple(r[k] for k in (*KEYS,'episode_id'))].append(r)
    return [dict({k:rr[0][k] for k in (*KEYS,'episode_id','suite','task_id','trial_id','outcome')},
                 n_source_episodes=len(rr),**{k:avg([r[k] for r in rr]) for k in metrics}) for rr in groups.values()]


def summarize(episodes,metrics,fields=(),bootstrap=False):
    groups=collections.defaultdict(list)
    for r in episodes:
        for origin in ('all',r['origin']):
            groups[tuple(r[k] for k in (*KEYS,*fields))+(origin,)].append(r)
    result=[]
    for key,rr in sorted(groups.items()):
        base=dict(zip((*KEYS,*fields,'origin'),key)); tasks={(r['suite'],r['task_id']) for r in rr}
        for metric in metrics:
            valid=[r for r in rr if r[metric] is not None and np.isfinite(r[metric])]
            x=np.asarray([r[metric] for r in valid],float)
            bytask=collections.defaultdict(list)
            for r in valid: bytask[(r['suite'],r['task_id'])].append(r[metric])
            ci=None
            if bootstrap and base['origin']=='all' and len(bytask)>1:
                ci=task_cluster_ci([dict(r,delta_entropy=r[metric]) for r in valid],seed=3101,iterations=2000)['ci95']
            result.append(dict(base,metric=metric,n_source_episodes=len(rr),n_valid_episodes=len(valid),
                n_tasks=len(tasks),n_valid_tasks=len(bytask),mean=float(x.mean()) if len(x) else None,
                median=float(np.median(x)) if len(x) else None,p25=float(np.quantile(x,.25)) if len(x) else None,
                p75=float(np.quantile(x,.75)) if len(x) else None,
                fraction_positive=float(np.mean(x>0)) if len(x) else None,
                fraction_negative=float(np.mean(x<0)) if len(x) else None,
                task_macro_mean=avg([np.mean(v) for v in bytask.values()]),
                ci95_low=ci[0] if ci else None,ci95_high=ci[1] if ci else None,
                bootstrap_seed=3101 if ci else None))
    return result


def correlations(episode_rows,xkey,ykey):
    rr=[r for r in episode_rows if r[xkey] is not None and r[ykey] is not None]
    x=np.array([r[xkey] for r in rr]); y=np.array([r[ykey] for r in rr])
    value=float(np.corrcoef(x,y)[0,1]) if len(rr)>2 and np.std(x)>0 and np.std(y)>0 else None
    return dict(x=xkey,y=ykey,n_episodes=len(rr),pearson=value,uncertainty='descriptive only; episode means, no independent-observation p-value')
