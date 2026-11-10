"""Phase 3B audit and CPU stagnation analysis. Existing artifacts are read-only."""
from __future__ import annotations
import argparse
import collections
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import platform
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3a_core import require, sha, array_hash, protocol_hash, SUITES, REVISION
from nominal_behavior import read, save, csv_save

ROOT = Path(__file__).resolve().parents[2]
UPSTREAM = 'e05f71e80d14ab4b55e5027ee739cfbcc6b12b9d'
MODEL_SOURCE = 'rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py'
MODEL_SHA = '22a5b7864bc125bb8ca757b1ed14e901972eea96ab243e9b5d663f61b3f219e9'
SOURCE_FILES = ('phase3b.py', 'phase3b_distribution.py', 'phase3b_pilot.py', 'phase3b_sft_sampling.py', 'phase3a_core.py', 'nominal_behavior.py', 'run_phase3b.sh')


def command(argv, cwd=None):
    r = subprocess.run(argv, cwd=cwd, capture_output=True, text=True)
    return dict(command=argv, returncode=r.returncode, stdout=r.stdout.strip(), stderr=r.stderr.strip())


def fresh(out):
    out = Path(out).resolve()
    require(not out.exists(), f'Output already exists: {out}')
    out.mkdir(parents=True)
    return out


def sources():
    return {str(Path(__file__).parent/n): sha(Path(__file__).parent/n) for n in SOURCE_FILES if (Path(__file__).parent/n).exists()}


def seal(out, details):
    details = dict(details, output_hashes={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    save(out/'validation.json', details)
    return details


def verify_seal(out):
    v = read(out/'validation.json')
    require(v['status']=='PASS', f'Validation did not pass: {out}')
    for name, digest in v['output_hashes'].items():
        require(sha(out/name)==digest, f'Changed derived output: {name}')
    return v


def verify_phase3a(p):
    p = p.resolve()
    v = read(p/'validation.json'); c = read(p/'completion.json')
    require(v['status']==c['status']=='PASS', 'Phase 3A incomplete')
    require(sha(p/'validation.json')==c['validation_sha256'], 'Phase 3A validation hash')
    require(sha(p/'summary.md')==c['summary_sha256'], 'Phase 3A summary hash')
    for name,digest in v['output_hashes'].items():
        require(sha(p/name)==digest, f'Phase 3A file changed: {name}')
    records=read(p/'records.json')
    require(set(records)=={'sft','grpo','stochastic'}, 'Unexpected Phase 3A policies')
    for label,rows in records.items():
        require(len(rows)==150 and len({r['episode_id'] for r in rows})==150, 'Missing/duplicate records')
        require(all(r['checkpoint_revision']==REVISION[label] and not r['lora_applied'] for r in rows), 'Checkpoint/LoRA mismatch')
    pairs=read(p/'paired/pairs.json')
    for comparison in ('primary','secondary'):
        rr=[r for r in pairs if r['comparison']==comparison]
        require(len(rr)==150 and len({r['episode_id'] for r in rr})==150, 'Pair identity mismatch')
    return dict(phase3a=str(p), derived_files_checked=len(v['output_hashes']), records=records,
                manifest=read(p/'manifest.json'), phase3a_validation_sha256=sha(p/'validation.json'))


def bank_check(p, raw=False):
    idx=read(p/'replay_bank/index.json'); seen=set(); counts=collections.Counter(); source_cache={}
    for e in idx['observations']:
        oid=e['observation_id']; require(oid not in seen, 'Duplicate observation'); seen.add(oid)
        require(e['source_policy'] in ('sft','grpo'), 'Unexpected replay origin'); counts[e['source_policy']]+=1
        m=e['source_metadata']; require(m['language'] and m['initial_state_sha256'], 'Missing language/state identity')
        require(m['episode_id']==e['source_episode'] and m['policy']==e['source_policy'], 'Replay identity')
        require(e['simulator_step']==8*e['query_index'], 'Replay query index')
        f=p/e['file']; require(sha(f)==e['file_sha256'], 'Replay file checksum')
        with np.load(f,allow_pickle=False) as z:
            require(set(z.files)==set(e['arrays']), 'Replay field mismatch')
            for k,desc in e['arrays'].items():
                a=z[k]; require(array_hash(a)==desc['sha256'], 'Replay array hash')
                require(list(a.shape)==desc['shape'] and str(a.dtype)==desc['dtype'] and np.isfinite(a).all(), 'Replay dtype/shape/finite')
        if raw:
            src=Path(e['source'])
            if str(src) not in source_cache:
                require(sha(src)==e['source_sha256'], 'Original observation changed')
                source_cache.clear()  # Index is grouped by origin episode; bound memory to one source NPZ.
                source_cache[str(src)]={k:v.copy() for k,v in np.load(src,allow_pickle=False).items()}
            original=source_cache[str(src)]; q=e['query_index']
            require(int(original['query_step'][q])==e['simulator_step'], 'Original query index')
            original_m=read(src.parent/'metadata.json')
            require(all(m.get(k)==v for k,v in original_m.items()), 'Original replay metadata mismatch')
            for k,desc in e['arrays'].items(): require(array_hash(original[k][q])==desc['sha256'],'Original replay readback')
    require(dict(counts)==idx['origin_policy_counts'], 'Origin counts')
    return dict(observations=len(seen), origin_policy_counts=dict(counts), raw_readback_verified=raw)


def checkpoint_audit(root):
    checked={}
    for item in read(root/'manifests/models_130.json')['models']:
        if item['revision'] not in REVISION.values(): continue
        directory=root/'checkpoints'/item['repo_id'].split('/')[-1]; files={}
        for f in item['files']:
            path=directory/f['path']; require(path.stat().st_size==f['size_bytes'], f'Checkpoint size: {path}')
            digest=sha(path)
            if f['lfs_sha256']:
                require(digest==f['lfs_sha256'], f'Checkpoint LFS hash: {path}')
            else:
                b=path.read_bytes(); blob=hashlib.sha1(b'blob '+str(len(b)).encode()+b'\0'+b).hexdigest()
                require(blob==f['blob_id'], f'Checkpoint git blob hash: {path}')
            files[str(path)]=dict(sha256=digest,size=path.stat().st_size,mtime_ns=path.stat().st_mtime_ns)
        checked[item['revision']]=dict(repo=item['repo_id'],directory=str(directory),files=files)
    require(set(checked)==set(REVISION.values()), 'Missing checkpoint inventory')
    return checked


def audit(args):
    out=fresh(args.output); info=verify_phase3a(args.phase3a)
    result={k:v for k,v in info.items() if k not in ('records','manifest')}
    result.update(mode='derived-only' if args.derived_only else 'server-full', source_hashes=sources(),
                  python=sys.executable,python_version=platform.python_version(),numpy=np.__version__,
                  git_status=command(['git','status','--short'],args.research_root),
                  git_branch=command(['git','branch','--show-current'],args.research_root),
                  source_files_verified=0)
    result['bank']=bank_check(args.phase3a,raw=not args.derived_only)
    if not args.derived_only:
        root=args.research_root.resolve(); upstream=root/'upstream/RLinf'
        rev=command(['git','rev-parse','HEAD'],upstream); dirty=command(['git','status','--porcelain'],upstream)
        require(rev['stdout']==UPSTREAM and dirty['returncode']==0 and not dirty['stdout'], 'Pinned upstream must be clean')
        require(sha(upstream/MODEL_SOURCE)==MODEL_SHA,'Native model source differs from audited pinned source')
        result.update(upstream=rev,upstream_dirty=dirty,native_model_source=str(upstream/MODEL_SOURCE),
                      native_model_sha256=MODEL_SHA,native_factorization='parallel 56 action slots; zero action embeddings; no autoregressive prefix')
        all_sources=dict(info['manifest']['source_files'])
        regression=read(args.phase3a/'expert_reference/regression/regression.json')
        all_sources.update(regression.get('source_files',{}))
        for path,digest in all_sources.items(): require(sha(path)==digest, f'Original source changed: {path}')
        result['source_files_verified']=len(all_sources)
        sys.path.insert(0,str(root/'scripts/evaluate'))
        from nominal_validate import validate_episode
        require(__debug__,'PYTHONOPTIMIZE must be unset')
        for label,rows in info['records'].items():
            for m in rows: validate_episode(Path(m['source_dir']),m['protocol_hash'])
        result['raw_episodes_validated']=450
        result['checkpoints']=checkpoint_audit(root)
        import torch
        result['torch']=torch.__version__; result['cuda_available']=torch.cuda.is_available()
        result['pip_freeze']=command([sys.executable,'-m','pip','freeze'])
        result['gpu']=command(['nvidia-smi'])
        result['collector_sha256']=sha(root/'scripts/evaluate/nominal_collect.py')
        result['collector_help']=command([sys.executable,'-B',str(root/'scripts/evaluate/nominal_collect.py'),'--help'])
        result['legacy_metrics_source']=str(Path(info['manifest']['expert_root'])/'experiments/dataset_analysis/analysis.py')
    save(out/'manifest.json',result); seal(out,dict(status='PASS',scope=result['mode'],raw_sources_verified=not args.derived_only))
    print(json.dumps({k:v for k,v in result.items() if k in ('mode','bank','source_files_verified','raw_episodes_validated','cuda_available')},indent=2))


def require_server_audit(path,phase3a,root):
    verify_seal(path); a=read(path/'manifest.json')
    require(a['source_hashes']==sources(),'Analysis source changed after audit; use a new run ID')
    require(a['mode']=='server-full' and a['phase3a']==str(phase3a.resolve()), 'Full server audit for this Phase 3A is required')
    require(a['phase3a_validation_sha256']==sha(phase3a/'validation.json'),'Phase 3A validation changed')
    require(sha(phase3a/'manifest.json')==read(phase3a/'validation.json')['output_hashes']['manifest.json'],'Phase 3A model/input manifest changed')
    up=root/'upstream/RLinf'
    require(command(['git','rev-parse','HEAD'],up)['stdout']==UPSTREAM,'Upstream revision changed')
    require(command(['git','status','--porcelain'],up)['stdout']=='','Upstream dirty')
    require(sha(up/MODEL_SOURCE)==MODEL_SHA,'Native source changed')
    require(sha(root/'scripts/evaluate/nominal_collect.py')==a['collector_sha256'],'Collector changed after audit')
    for item in a['checkpoints'].values():
        for name,f in item['files'].items():
            st=Path(name).stat(); require(st.st_size==f['size'] and st.st_mtime_ns==f['mtime_ns'],'Checkpoint changed since full hash audit')
    return a


def stationary_runs(mask):
    mask=np.asarray(mask,dtype=bool)
    edges=np.diff(np.r_[False,mask,False].astype(int))
    starts=np.flatnonzero(edges==1); ends=np.flatnonzero(edges==-1)
    return starts, ends-starts


def motion_metrics(x,threshold=1e-4):
    x=np.asarray(x); require(x.ndim==2 and x.shape[1]==3 and len(x)>1 and np.isfinite(x).all(),'Invalid EEF path')
    d=np.linalg.norm(np.diff(x,axis=0),axis=1); mask=d<threshold; starts,runs=stationary_runs(mask)
    return dict(steps=len(d),path_length=float(d.sum()),net_displacement=float(np.linalg.norm(x[-1]-x[0])),
        initial_xyz=x[0].tolist(),final_xyz=x[-1].tolist(),stationary_steps=int(mask.sum()),stationary_fraction=float(mask.mean()),
        stagnation=bool(mask.mean()>=.5),legacy_strict_gt_half=bool(mask.mean()>.5),
        displacement_min=float(d.min()),displacement_p25=float(np.quantile(d,.25)),displacement_median=float(np.median(d)),
        displacement_p75=float(np.quantile(d,.75)),displacement_p95=float(np.quantile(d,.95)),displacement_max=float(d.max()),
        stationary_runs=len(runs),stationary_longest_run=int(runs.max()) if len(runs) else 0,
        stationary_mean_run=float(runs.mean()) if len(runs) else 0),d,starts,runs


def stagnation(args):
    require(__debug__,'PYTHONOPTIMIZE must be unset')
    info=verify_phase3a(args.phase3a); p=args.phase3a; out=fresh(args.output)
    raw=args.raw; files={str(p/'validation.json'):sha(p/'validation.json')}
    prior={(r['policy'],r['episode_id']):r for r in csv.DictReader((p/'trajectories/metrics.csv').open())}
    rows=[]; motion=[]; sensitivity=[]; run_rows=[]; displacement={}; originals={}
    for policy,records in info['records'].items():
        for m in records:
            eid=m['episode_id']; canonical=p/'trajectories/canonical'/policy/(eid+'.npz'); files[str(canonical)]=sha(canonical)
            with np.load(canonical,allow_pickle=False) as z: x=z['xyz'].copy()
            metrics,d,starts,runs=motion_metrics(x); old=prior[(policy,eid)]
            require(metrics['stationary_steps']==int(old['stationary_steps_lt_1e_4_m']),'Phase 3A stationary count mismatch')
            require(np.isclose(metrics['path_length'],float(old['path_length']),rtol=1e-7,atol=1e-9),'Phase 3A path length mismatch')
            row=dict(policy=policy,episode_id=eid,suite=m['suite'],task_id=m['task_id'],success=m['success'],**metrics)
            rows.append(row); originals[(policy,eid)]=row; displacement[(policy,eid)]=d
            run_rows.extend(dict(policy=policy,episode_id=eid,start=int(s),length=int(n)) for s,n in zip(starts,runs))
            for mm in (.05,.1,.2,.5):
                frac=float((d<mm/1000).mean())
                for cutoff in (.25,.5,.75):
                    sensitivity.append(dict(policy=policy,episode_id=eid,suite=m['suite'],task_id=m['task_id'],success=m['success'],threshold_mm=mm,fraction_cutoff=cutoff,stationary_fraction=frac,stagnation=frac>=cutoff))
            cm=dict(policy=policy,episode_id=eid,status='RAW_NOT_LOCAL',command_units='native decoded translation; NOT assumed meters',object_failure_label='UNKNOWN')
            if raw:
                src=Path(m['source_dir']); mf=read(src.parent.parent/'manifest.json')
                for name in ('metadata.json','states.npz','actions.npz','observations.npz'):
                    require(sha(src/name)==info['manifest']['source_files'].get(str(src/name)), 'Raw input differs from Phase 3A: '+str(src/name))
                sys.path.insert(0,str(args.research_root/'scripts/evaluate'))
                from nominal_validate import validate_episode
                validate_episode(src,mf['protocol_hash'])
                for n in ('states.npz','actions.npz','metadata.json'): files[str(src/n)]=sha(src/n)
                with np.load(src/'states.npz',allow_pickle=False) as z:
                    require(np.array_equal(x,z['robot0_eef_pos']),'Raw/canonical state mismatch')
                    finger_motion=np.linalg.norm(np.diff(z['robot0_gripper_qpos'],axis=0),axis=1)
                with np.load(src/'actions.npz',allow_pickle=False) as z: act=z['executed_actions'].copy()
                c=np.linalg.norm(act[:,:3],axis=1); low=d<1e-4
                corr=float(np.corrcoef(c,d)[0,1]) if c.std()>0 and d.std()>0 else None
                cm.update(status='MEASURED',mean_translation_command_norm=float(c.mean()),median_translation_command_norm=float(np.median(c)),
                    stationary_command_median=float(np.median(c[low])) if low.any() else None,
                    moving_command_median=float(np.median(c[~low])) if (~low).any() else None,
                    exactly_zero_translation_commands=int((c==0).sum()),command_motion_correlation=corr,
                    stationary_nonzero_translation_commands=int(np.count_nonzero(low & (c>0))),
                    stationary_zero_translation_commands=int(np.count_nonzero(low & (c==0))),
                    stationary_finger_qpos_motion_mean=float(finger_motion[low].mean()) if low.any() else None,
                    gripper_command_transitions=int(np.count_nonzero(np.diff(act[:,6]))),
                    classification='No causal low-action/contact/saturation label; native command scale not calibrated')
            motion.append(cm)
    paired=[]; pairs=[r for r in read(p/'paired/pairs.json') if r['comparison'] in ('primary','secondary')]
    for pair in pairs:
        a=originals[(pair['left'],pair['episode_id'])]; b=originals[(pair['right'],pair['episode_id'])]
        paired.append(dict(comparison=pair['comparison'],episode_id=pair['episode_id'],suite=pair['suite'],task_id=pair['task_id'],outcome=pair['category'],
            left_stagnation=a['stagnation'],right_stagnation=b['stagnation'],left_fraction=a['stationary_fraction'],right_fraction=b['stationary_fraction'],
            fraction_change=b['stationary_fraction']-a['stationary_fraction'],left_steps=a['steps'],right_steps=b['steps'],
            left_path=a['path_length'],right_path=b['path_length']))
    def grouped(keys):
        groups=collections.defaultdict(list)
        for r in paired: groups[tuple(r[k] for k in keys)].append(r)
        return [dict(zip(keys,k),n=len(v),left_stagnant=sum(r['left_stagnation'] for r in v),right_stagnant=sum(r['right_stagnation'] for r in v),
                     left_rate=sum(r['left_stagnation'] for r in v)/len(v),right_rate=sum(r['right_stagnation'] for r in v)/len(v),
                     change_pp=100*sum(int(r['right_stagnation'])-int(r['left_stagnation']) for r in v)/len(v),
                     both_stagnant=sum(r['left_stagnation'] and r['right_stagnation'] for r in v),
                     resolved=sum(r['left_stagnation'] and not r['right_stagnation'] for r in v),
                     appeared=sum(not r['left_stagnation'] and r['right_stagnation'] for r in v)) for k,v in sorted(groups.items())]
    sweep=[]
    for policy in ('sft','grpo','stochastic'):
        for mm in (.05,.1,.2,.5):
            for cutoff in (.25,.5,.75):
                rr=[r for r in sensitivity if r['policy']==policy and r['threshold_mm']==mm and r['fraction_cutoff']==cutoff]
                sweep.append(dict(policy=policy,threshold_mm=mm,fraction_cutoff=cutoff,n=len(rr),stagnant=sum(r['stagnation'] for r in rr),rate=sum(r['stagnation'] for r in rr)/len(rr)))
    contingency=[]
    for policy in ('sft','grpo','stochastic'):
        for success in (False,True):
            rr=[r for r in rows if r['policy']==policy and r['success']==success]
            contingency.append(dict(policy=policy,success=success,n=len(rr),stagnant=sum(r['stagnation'] for r in rr),nonstagnant=sum(not r['stagnation'] for r in rr)))
    for name,data in [('per_episode',rows),('paired_stagnation',paired),('outcome_summary',grouped(('comparison','outcome'))),
                      ('suite_summary',grouped(('comparison','suite'))),('task_summary',grouped(('comparison','suite','task_id'))),
                      ('overall_summary',grouped(('comparison',))),('threshold_sensitivity',sweep),('threshold_per_episode',sensitivity),
                      ('command_motion_analysis',motion),('stationary_runs',run_rows),('stationarity_vs_success',contingency)]: csv_save(out/(name+'.csv'),data)
    gained=[r for r in paired if r['comparison']=='primary' and r['outcome']=='gained']
    legacy_counts={pol:sum(originals[(pol,r['episode_id'])]['legacy_strict_gt_half'] for r in gained) for pol in ('sft','grpo')}
    equality=[dict(policy=r['policy'],episode_id=r['episode_id']) for r in rows if r['stationary_fraction']==.5]
    save(out/'manifest.json',dict(phase3a=str(p.resolve()),source_hashes=sources(),input_hashes=files,
        definition='per-step EEF Euclidean displacement < 0.0001 m; stationary fraction >= 0.5',
        frequency='one position per executed simulator action; T+1 states; no wall-clock rate assumed',
        legacy_definition='Phase 3A metric uses strict displacement <; earlier ad-hoc episode classification used fraction > 0.5',
        boundary_episodes=equality,legacy_gained_counts=legacy_counts,command_motion='raw-verified' if raw else 'UNAVAILABLE: raw rollouts not present',
        interpretation='Descriptive fixed task/state population. Success terminates episodes early; failures are horizon-censored. Stationarity is not a failure label. Object roles UNKNOWN.'))
    plot_stagnation(out,rows,sweep)
    overall=grouped(('comparison',)); lines=['# Phase 3B stagnation','',json.dumps(overall,indent=2),'',f'Legacy gained replication: {legacy_counts}; n={len(gained)}.',
        f'Exactly 50% boundary episodes: {len(equality)}.',f'Command-motion: {"measured" if raw else "UNAVAILABLE (raw files not local)"}.',
        'EEF stationarity is measured displacement, not absence of policy commands or task failure. Threshold grid is descriptive, not optimized.']
    (out/'summary.md').write_text('\n'.join(lines)+'\n')
    for f,h in files.items(): require(sha(f)==h,'Input changed during analysis')
    seal(out,dict(status='PASS',episodes=len(rows),pairs=len(paired),primary_pairs=150,legacy_gained_counts=legacy_counts,raw_command_motion=raw))
    print('\n'.join(lines))


def plot_stagnation(out,rows,sweep):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dest=out/'figures'; dest.mkdir()
    fig,axes=plt.subplots(1,3,figsize=(13,4))
    for policy,color in zip(('sft','grpo','stochastic'),('C0','C1','C2')):
        rr=[r for r in rows if r['policy']==policy]
        for ax,key in zip(axes,('stationary_fraction','path_length','steps')):
            a=np.sort([r[key] for r in rr]); ax.plot(a,np.arange(1,len(a)+1)/len(a),label=policy,color=color); ax.set(xlabel=key,ylabel='Empirical CDF')
    axes[0].legend(); fig.tight_layout(); fig.savefig(dest/'all_episodes_ecdf.png',dpi=140); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(12,4))
    for ax,policy in zip(axes,('sft','grpo','stochastic')):
        a=np.array([[next(r['stagnant']/r['n'] for r in sweep if r['policy']==policy and r['threshold_mm']==mm and r['fraction_cutoff']==c) for mm in (.05,.1,.2,.5)] for c in (.25,.5,.75)])
        im=ax.imshow(a,vmin=0,vmax=1); ax.set(xticks=range(4),xticklabels=['0.05','0.1','0.2','0.5'],yticks=range(3),yticklabels=['25%','50%','75%'],xlabel='Displacement threshold (mm)',ylabel='Fraction cutoff',title=policy)
        for i in range(3):
            for j in range(4): ax.text(j,i,f'{a[i,j]:.0%}',ha='center',va='center',color='white' if a[i,j]<.5 else 'black')
    fig.tight_layout(); fig.savefig(dest/'threshold_sensitivity.png',dpi=140); plt.close(fig)


def report(args):
    out=fresh(args.output); sections=[]
    for label,path in [('stagnation',args.stagnation),('sft_sampling',args.sft_sampling),('distribution',args.distribution),('pilot_diversity',args.diversity)]:
        if path:
            v=verify_seal(path); sections.extend([f'## {label}',f'Input: {path.resolve()}',(path/'summary.md').read_text()])
        else: sections.extend([f'## {label}','NOT RUN'])
    (out/'summary.md').write_text('# Phase 3B report\n\n'+'\n\n'.join(sections))
    seal(out,dict(status='PASS',scope='Report of supplied stages; omitted stages explicitly NOT RUN'))
    print(out/'summary.md')


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='stage',required=True)
    for name in ('audit','stagnation'):
        a=sub.add_parser(name); a.add_argument('--phase3a',type=Path,required=True); a.add_argument('--output',type=Path,required=True); a.add_argument('--research-root',type=Path,default=ROOT)
        if name=='audit': a.add_argument('--derived-only',action='store_true')
        else: a.add_argument('--raw',action='store_true',help='Require and validate raw states/actions for command-motion analysis')
    r=sub.add_parser('report'); r.add_argument('--output',type=Path,required=True)
    for name in ('stagnation','sft-sampling','distribution','diversity'): r.add_argument('--'+name,type=Path)
    args=p.parse_args(); globals()[args.stage](args)

if __name__=='__main__': main()
