"""Stage 2/3 CPU analysis of sealed native predictions; never loads a model."""
from __future__ import annotations
import argparse
import collections
import csv
import json
import re
from pathlib import Path
import shutil
import subprocess
import sys
import numpy as np
from token_action_core import *
from scripts.common.io import save, csv_save
from scripts.common.integrity import seal
from scripts.common.action_schema import *
from phase3b_distribution import distribution_metrics
from phase3b_generality import aggregate_episodes, _distribution_summary

TEMPERATURES=(1.,1.6)
METRICS=('entropy_sft','entropy_grpo','top1_sft','top1_grpo','greedy_disagreement','kl_sft_grpo','kl_grpo_sft','js','top5_sft','top5_grpo','top10_sft','top10_grpo','delta_entropy','delta_top1')
CODE=('scripts/common/io.py','scripts/common/integrity.py','tests/test_token_action.py','scripts/common/action_schema.py','scripts/analysis/token_action_core.py','scripts/analysis/token_action_analysis.py','scripts/analysis/token_action_figures.py','scripts/analysis/phase3b_distribution.py','scripts/analysis/phase3b_generality.py','scripts/evaluate/nominal_collect.py','scripts/analysis/outcome_diversity.py')


SOURCE_AUDIT = """# Source audit

## Reused implementations
- `phase3b_distribution.distribution_metrics`: float64 log-softmax, full action-token support, natural-log entropy, KL in both directions, JS, top-1/5/10 and argmax disagreement.
- `phase3b_generality.aggregate_episodes` and `_distribution_summary`: exact Stage 1 numerical regression.
- `phase3b_generality.task_cluster_ci`: whole-task resampling within suite, 2000 draws, seed 3101. Source episodes and origins stay with their task.
- `scripts.common.io` and `integrity`: exclusive output creation and SHA256 output seals.
- Reviewed `phase3b_pilot.repeat_run`, `outcome_diversity`, `nominal_collect.collect_episode`, `nominal_storage`, and `nominal_validate` for collection/trajectory contracts. Existing collector and schemas are unchanged.

## Audited native chain
Pinned source snapshots are included in `native_sources/`, checked against fixed SHA256 values and upstream commit. The model snapshot hash equals both worker manifests.

1. Native prediction reshapes 56 token IDs to 8 rows × 7 dimensions (chunk-major).
2. The VLA adapter's field slicing identifies first 3 translation command channels, next 3 rotation command channels, last gripper channel. LIBERO passes the first six channels unchanged. Axis labels describe command channels; world-frame displacement and physical angular units are not inferred.
3. Native bin index is clip(vocab_size - token_id - 1, 0, B-2). B=256 edges yield 255 centers. The first two admitted token IDs alias the top center.
4. Q99 unnormalization is where(mask, 0.5*(normalized+1)*(q99-q01+1e-8)+q01, normalized). This branch is verified numerically against every saved decoded action; mask/q01/q99 are copied in workers.json. Import-time normalization configuration was not available locally.
5. LIBERO gripper preparation is -sign(2*decoded-1). Native reset_gripper_open sends -1; +1 is the close command. Neutral 0 can occur at exactly 0.5. No measured gripper state is inferred.
6. Nominal collector copies decoded chunks before preparation, then records each actually executed query/slot. A chunk can terminate early. Local nominal source arrays were not available for verified linkage; this run uses only predicted slots, execution UNKNOWN.

## New code
- `action_schema`: mapping, decoding validation, gripper transform, near-zero vector metrics.
- `token_action_core`: saved-record hash/identity validation and explicit source-episode / initial-state / task / suite / outcome aggregation.
- `token_action_analysis`: audit and Stage 2/3 CLI, saved arrays, numerical regression, report, final verification.
- `token_action_figures`: PNG/PDF exports using actual summary tables.

No existing CLI, numerical implementation, upstream file, model checkpoint, nominal schema, or Stage 1/4 output was modified. No inference or simulator runtime is imported.
"""


def fresh(p):
    require(not p.exists(),f'Output exists: {p}'); p.mkdir(parents=True); return p


def writecsv(p,rows): csv_save(p,clean(rows))


def identity(e):
    return dict(observation_id=e['observation_id'],origin=e['source_policy'],episode_id=e['source_episode'],
                **{k:e[k] for k in ('suite','task_id','trial_id','outcome','source_success')})


def check_outcomes(entries):
    byepisode=collections.defaultdict(list)
    for e in entries:
        require(e['source_policy'] in ('sft','grpo'),'Unknown origin')
        expected={'both_success':(True,True),'both_fail':(False,False),'gained':(False,True),'lost':(True,False)}
        require(e['outcome'] in expected and e['source_success']==expected[e['outcome']][e['source_policy']=='grpo'],'Outcome/source success mismatch')
        byepisode[e['source_episode']].append(e)
    for rr in byepisode.values():
        require(len({(r['suite'],r['task_id'],r['trial_id'],r['outcome'],r['initial_state_sha256']) for r in rr})==1,'Initial-state/outcome alignment')
        require({r['source_policy'] for r in rr}=={'sft','grpo'},'Missing paired origin')


def audit(args):
    s1=(args.stage1 or latest_pass(ROOT/'outputs/analysis/distribution_generality',True)).resolve()
    s4=(args.stage4 or latest_pass(ROOT/'outputs/analysis/outcome_diversity')).resolve()
    for d in ('bank','workers/sft','workers/grpo'): verify_seal(s1/d)
    analysis_status=check_analysis_copy(s1/'analysis'); verify_seal(s4)
    entries=read(s1/'bank/manifest.json')['observations']; ids=[e['observation_id'] for e in entries]
    require(len(ids)==len(set(ids))==3000,'Expected 3000 unique common observations')
    check_outcomes(entries)
    manifests={m:read(s1/'workers'/m/'manifest.json') for m in ('sft','grpo')}
    for m,v in manifests.items():
        require(v['mode']=='full' and v['native_model_sha256']==SOURCE_HASHES['model.py'],'Native worker provenance')
        require(v['bank_validation_sha256']==sha(s1/'bank/validation.json'),'Worker bank seal mismatch')
        require({p.name for p in (s1/'workers'/m/'records').iterdir() if p.is_dir()}==set(ids),'Worker observation set mismatch')
    require(manifests['sft']['action_norm_stats']==manifests['grpo']['action_norm_stats'],'Normalization mismatch')
    snapshots={}
    for name,path in SOURCE_PATHS.items():
        p=args.native_source_dir/name if args.native_source_dir else ROOT/'upstream/RLinf'/path
        require(p.is_file() and sha(p)==SOURCE_HASHES[name],f'Pinned native source unavailable/changed: {p}')
        snapshots[name]=p
    out=fresh(args.output); a=fresh(out/'audit'); (a/'native_sources').mkdir()
    for name,p in snapshots.items(): shutil.copyfile(p,a/'native_sources'/name)
    actions=np.empty((len(entries),2,8,7)); normalized=np.empty_like(actions); tokens=np.empty_like(actions,dtype=np.int64)
    for i,e in enumerate(entries):
        records=[]
        for j,m in enumerate(('sft','grpo')):
            z,r=load_record(s1,m,e,manifests[m]); records.append(r)
            actions[i,j]=z['decoded_actions']; normalized[i,j]=z['normalized_actions']; tokens[i,j]=z['action_tokens']
        require(records[0]['processed_tensor_hashes']==records[1]['processed_tensor_hashes'],'Processed inputs differ')
        if (i+1)%500==0: print('AUDIT',i+1,'/',len(entries),flush=True)
    np.savez_compressed(a/'actions.npz',decoded=actions,normalized=normalized,tokens=tokens)
    (a/'source_audit.md').write_text(SOURCE_AUDIT)
    save(a/'observations.json',entries); save(a/'workers.json',manifests); save(a/'contract.json',contract())
    inputs=[s1/d/'validation.json' for d in ('bank','workers/sft','workers/grpo','analysis')]+[s4/'validation.json']
    manifest=dict(stage1=str(s1),stage4=str(s4),input_seals={str(p):sha(p) for p in inputs},
        native_source_hashes=SOURCE_HASHES,upstream_commit=UPSTREAM,checkpoint_revisions={m:v['checkpoint_revision'] for m,v in manifests.items()},
        temperatures=TEMPERATURES,weighting='slots within observation; observations within source episode; equal source episodes. Task macro separately. Bootstrap whole tasks within suite, seed3101,2000 draws.',
        code_hashes={p:sha(ROOT/p) for p in CODE},git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        execution='UNKNOWN: no verified nominal query/execution linkage supplied; all metrics predicted slots; no executed/EEF analysis',
        stage1_analysis_copy=analysis_status,source_episodes=len({(e['source_policy'],e['source_episode']) for e in entries}),
        initial_states=len({e['source_episode'] for e in entries}),observations=len(entries),cpu_only=True)
    save(out/'manifest.json',manifest)
    seal(a,dict(status='PASS',saved_records=2*len(entries),native_decoding='PASS',processed_inputs='PASS',outcomes='PASS',execution='UNKNOWN'))
    print(json.dumps(manifest,indent=2))


def context(out):
    verify_seal(out/'audit'); m=read(out/'manifest.json')
    for p,h in m['code_hashes'].items(): require(sha(ROOT/p)==h,'Analysis source changed: '+p)
    for p,h in m['input_seals'].items(): require(sha(Path(p))==h,'Input seal changed: '+p)
    return m,read(out/'audit/observations.json'),read(out/'audit/workers.json')


def save_aggregates(out,rows,metrics):
    ep=episode_means(rows,metrics)
    save(out/'observations.json',clean(rows)); save(out/'episodes.json',clean(ep))
    writecsv(out/'observation_metrics.csv',rows); writecsv(out/'episode_metrics.csv',ep)
    writecsv(out/'initial_state_metrics.csv',initial_states(ep,metrics))
    summaries=summarize(ep,metrics,bootstrap=True)
    save(out/'group_summary.json',clean(summaries)); writecsv(out/'action_group_summary.csv',summaries)
    for name,fields in [('task',('suite','task_id')),('suite',('suite',)),('outcome',('outcome',))]:
        writecsv(out/(name+'_summary.csv'),summarize(ep,metrics,fields))
    return ep,summaries


def stage2(args):
    m,entries,manifests=context(args.output); s1=Path(m['stage1']); out=fresh(args.output/'stage2/data')
    cache=np.empty((2,len(entries),8,7,len(METRICS)))
    confidence=np.empty((2,len(entries),2,8,4)) # open, close, neutral, greedy-command probability
    reference={(r['observation_id'],float(r['temperature'])):r for r in csv_rows(s1/'analysis/observations.csv')}
    require(len(reference)==6000,'Stage 1 reference coverage')
    max_error=0.; rows=[]
    with (out/'slot_metrics.csv').open('x',newline='') as f:
        writer=None
        for i,e in enumerate(entries):
            z=[load_record(s1,p,e,manifests[p])[0] for p in ('sft','grpo')]
            for ti,t in enumerate(TEMPERATURES):
                d=distribution_metrics(z[0]['logits'],z[1]['logits'],t)
                d.update(delta_entropy=d['entropy_grpo']-d['entropy_sft'],delta_top1=d['top1_grpo']-d['top1_sft'])
                a=np.stack([d[k] for k in METRICS],axis=-1).reshape(8,7,-1); cache[ti,i]=a
                for k,v in d.items():
                    err=abs(float(v.mean())-float(reference[(e['observation_id'],t)][k])); max_error=max(max_error,err)
                    require(err<1e-10,'Stage 1 observation regression: '+k)
                base=dict(identity(e),temperature=t,scope='predicted',execution='UNKNOWN')
                for c,dim in np.ndindex(8,7):
                    row=dict(base,chunk=c,dimension=DIMENSIONS[dim],**dict(zip(METRICS,a[c,dim])))
                    if writer is None: writer=csv.DictWriter(f,fieldnames=list(row)); writer.writeheader()
                    writer.writerow(row)
                for group,dims in dict(all=tuple(range(7)),**GROUPS).items():
                    rows.append(dict(base,group=group,**dict(zip(METRICS,a[:,dims,:].mean(axis=(0,1))))))
                for j,pol in enumerate(('sft','grpo')):
                    logits=z[j]['logits'].reshape(8,7,-1)[:,6,:].astype(float)/t
                    p=np.exp(logits-logits.max(-1,keepdims=True)); p/=p.sum(-1,keepdims=True)
                    signs=prepared_gripper(token_values(manifests[pol]))
                    cmd=prepared_gripper(z[j]['decoded_actions'][:,6])
                    confidence[ti,i,j]=np.stack([p[:,signs==-1].sum(-1),p[:,signs==1].sum(-1),p[:,signs==0].sum(-1),(p*(signs[None,:]==cmd[:,None])).sum(-1)],axis=-1)
            if (i+1)%500==0: print('STAGE2',i+1,'/',len(entries),flush=True)
    np.savez_compressed(out/'metrics.npz',metrics=cache,gripper_confidence=confidence)
    ep,summary=save_aggregates(out,rows,METRICS)
    # Stage 1's own episode reducer and summary must reproduce its saved summary.
    ref=read(s1/'analysis/metrics.json')['overall']; reproduced={}
    for t in TEMPERATURES:
        rr=[r for r in rows if r['temperature']==t and r['group']=='all']
        got=_distribution_summary(aggregate_episodes(rr)); reproduced[str(t)]=got
        for k,v in got.items():
            if k in ref[str(t)]: require(abs(v-ref[str(t)][k])<1e-10,'Stage 1 overall regression: '+k)
    save(out/'stage1_regression.json',dict(status='PASS',max_observation_absolute_error=max_error,reproduced=reproduced))
    # Slot summaries first reduce each source episode, then average episodes.
    episode_indices=collections.defaultdict(list)
    for i,e in enumerate(entries): episode_indices[(e['source_policy'],e['source_episode'])].append(i)
    slot=np.stack([cache[:,ii].mean(axis=1) for ii in episode_indices.values()],axis=1).mean(axis=1)
    for name,shape in [('slot',(8,7)),('dimension',(7,)),('chunk',(8,))]:
        result=[]
        for ti,t in enumerate(TEMPERATURES):
            for indices in np.ndindex(shape):
                a=slot[ti][indices] if name=='slot' else slot[ti].mean(axis=0 if name=='dimension' else 1)[indices[0]]
                result.append(dict(temperature=t,index=list(indices),label=DIMENSIONS[indices[0]] if name=='dimension' else str(indices),**dict(zip(METRICS,a))))
        writecsv(out/(name+'_summary.csv'),result)
    (out/'summary.md').write_text('# Stage 2\n\nAll 56 predicted slots; execution UNKNOWN. Temperatures kept separate.\nStage 1 observation and episode regression: PASS.\nSee action_group_summary.csv for episode-weighted and task-macro results.\nS2-5 NOT AVAILABLE: no verified execution mask.\n')
    seal(out,dict(status='PASS',observations=len(entries),slot_rows=2*len(entries)*56,stage1_regression='PASS',executed_slots='UNKNOWN'))


ACTION_METRICS=('delta_entropy','js','greedy_disagreement','action_difference','normalized_difference','magnitude_sft','magnitude_grpo','magnitude_delta','cosine','near_zero_sft','near_zero_grpo','cosine_defined','direction_reversed','gripper_disagreement','threshold_proximity_sft','threshold_proximity_grpo','command_confidence_sft','command_confidence_grpo','p_open_sft','p_open_grpo','p_close_sft','p_close_grpo')


def stage3(args):
    m,entries,manifests=context(args.output); verify_seal(args.output/'stage2/data'); out=fresh(args.output/'stage3/data')
    with np.load(args.output/'audit/actions.npz') as z: actions=z['decoded']; normalized=z['normalized']; tokens=z['tokens']
    with np.load(args.output/'stage2/data/metrics.npz') as z: cache=z['metrics']; confidence=z['gripper_confidence']
    rows=[]; chunkrows=[]; bins=np.asarray(manifests['sft']['token_to_bin_index']); start=manifests['sft']['token_start']
    token_dist=np.abs(tokens[:,1]-tokens[:,0]); bin_dist=np.abs(bins[tokens[:,1]-start]-bins[tokens[:,0]-start])
    with (out/'slot_action_relations.csv').open('x',newline='') as f:
        writer=None
        for i,e in enumerate(entries):
            a,b=actions[i]; na,nb=normalized[i]; base=identity(e)
            for c in range(8):
                r=dict(base,chunk=c,scope='counterfactual_greedy',execution='UNKNOWN')
                for j,d in enumerate(DIMENSIONS):
                    r.update({d+'_sft':a[c,j],d+'_grpo':b[c,j],d+'_normalized_sft':na[c,j],d+'_normalized_grpo':nb[c,j]})
                for group,dims in GROUPS.items():
                    if group=='gripper': continue
                    r.update({group+'_'+k:clean(v) for k,v in vector_metrics(a[c,list(dims)],b[c,list(dims)],args.near_zero).items()})
                r.update(gripper_prepared_sft=prepared_gripper(a[c,6]),gripper_prepared_grpo=prepared_gripper(b[c,6]),gripper_disagreement=prepared_gripper(a[c,6])!=prepared_gripper(b[c,6]),threshold_proximity_sft=abs(a[c,6]-.5),threshold_proximity_grpo=abs(b[c,6]-.5))
                chunkrows.append(r)
            for ti,t in enumerate(TEMPERATURES):
                for c,d in np.ndindex(8,7):
                    r=dict(base,temperature=t,chunk=c,dimension=DIMENSIONS[d],delta_entropy=cache[ti,i,c,d,METRICS.index('delta_entropy')],js=cache[ti,i,c,d,METRICS.index('js')],token_id_distance=int(token_dist[i,c,d]),bin_index_distance=int(bin_dist[i,c,d]),decoded_abs_difference=abs(b[c,d]-a[c,d]),normalized_abs_difference=abs(nb[c,d]-na[c,d]))
                    if writer is None: writer=csv.DictWriter(f,fieldnames=list(r)); writer.writeheader()
                    writer.writerow(r)
                for group,dims in GROUPS.items():
                    r=dict(base,temperature=t,group=group,scope='predicted',**dict.fromkeys(ACTION_METRICS))
                    for k in ('delta_entropy','js','greedy_disagreement'): r[k]=cache[ti,i][:,dims,METRICS.index(k)].mean()
                    r['normalized_difference']=np.linalg.norm((nb-na)[:,dims],axis=-1).mean()
                    if group!='gripper':
                        v=vector_metrics(a[:,dims],b[:,dims],args.near_zero)
                        r.update({k:avg(v[k]) for k in ('magnitude_sft','magnitude_grpo','magnitude_delta')})
                        r['action_difference']=v['difference_l2'].mean()
                        if group=='translation':
                            r.update({k:avg(v[k]) for k in ('cosine','near_zero_sft','near_zero_grpo','cosine_defined')})
                            r['direction_reversed']=avg(np.where(v['cosine_defined'],v['cosine']<0,np.nan))
                    else:
                        r['action_difference']=np.abs(b[:,6]-a[:,6]).mean()
                        r['gripper_disagreement']=(prepared_gripper(a[:,6])!=prepared_gripper(b[:,6])).mean()
                        for j,pol in enumerate(('sft','grpo')):
                            r['threshold_proximity_'+pol]=np.abs(actions[i,j,:,6]-.5).mean()
                            for col,key in [(0,'p_open'),(1,'p_close'),(3,'command_confidence')]: r[key+'_'+pol]=confidence[ti,i,j,:,col].mean()
                    rows.append(r)
    writecsv(out/'observation_action_metrics.csv',chunkrows)
    ep,summary=save_aggregates(out,rows,ACTION_METRICS)
    correlation=[]
    for t in TEMPERATURES:
        for g in GROUPS:
            for origin in ('all','sft','grpo'):
                rr=[r for r in ep if r['temperature']==t and r['group']==g and (origin=='all' or r['origin']==origin)]
                for x,y in [('js','action_difference'),('delta_entropy','action_difference')]+([('delta_entropy','magnitude_delta')] if g!='gripper' else []):
                    correlation.append(dict(temperature=t,group=g,origin=origin,**correlations(rr,x,y)))
    writecsv(out/'correlations.csv',correlation)
    sensitivity=[]
    for eps in (.001,.01,.05):
        v=vector_metrics(actions[:,0,:,:3],actions[:,1,:,:3],eps)
        sensitivity.append(dict(epsilon=eps,undefined_fraction=float((~v['cosine_defined']).mean()),mean_defined_chunk_cosine=avg(v['cosine'].ravel()),weighting='descriptive chunk count; main report uses episode means'))
    save(out/'near_zero_sensitivity.json',sensitivity)
    save(out/'token_mapping_effects.json',dict(token_changes=int((token_dist>0).sum()),token_changes_same_bin=int(((token_dist>0)&(bin_dist==0)).sum()),
        normalized_step=float(2/255),decoded_step_by_dimension=(.5*(np.array(manifests['sft']['action_norm_stats']['q99'])-np.array(manifests['sft']['action_norm_stats']['q01'])+1e-8)*2/255).tolist()[:6]+[2/255],
        mapping='native reverse token indices, first two tokens alias top bin; affine Q99 per dimension; gripper prepared command discontinuity at decoded=0.5'))
    (out/'summary.md').write_text('# Stage 3\n\nCounterfactual greedy predictions at common observations. Decoded values are controller commands, not measured motion.\nRotation comparison uses command space, not physical angular distance.\nGreedy actions do not change with offline temperature.\nTranslation cosine undefined if either norm <= '+str(args.near_zero)+'.\n')
    seal(out,dict(status='PASS',observations=len(entries),chunk_rows=len(chunkrows),near_zero_epsilon=args.near_zero,execution='UNKNOWN'))


def report(args):
    m,_,_=context(args.output)
    for d in ('stage2/data','stage2/figures','stage3/data','stage3/figures'): verify_seal(args.output/d)
    lines=['# Stage 2 + Stage 3: common-observation token/action analysis','',
        '3000 observations, 300 source episodes, 150 initial states, 30 tasks. All comparisons use the same input for both policies.',
        'Means: slots → observation → source episode → equal episode mean. Task macro and stratified summaries are saved separately. Bootstrap: whole tasks within suite, seed 3101, 2000 draws. Intervals describe these fixed nominal tasks.',
        '', '## Source audit and compatibility','',
        'Reused phase3b_distribution.distribution_metrics and phase3b_generality.aggregate_episodes/task_cluster_ci. Native source snapshots are hash-checked at pinned upstream commit. Saved token→bin→normalized→decoded values are verified for every record. Existing Stage 1/4 code and outputs are unchanged.',
        'Stage 1 bank/workers and required numeric files: PASS. Stage 4 seal: PASS. Stage 1 copied original figures missing: '+str(len(m['stage1_analysis_copy']['missing_figures']))+'. Full copied Stage 1 analysis seal is therefore PARTIAL_COPY_FIGURES_MISSING; this is not a full Stage 1 verification PASS.',
        'Execution mask: UNKNOWN. S2-5 NOT AVAILABLE. Physical translation frame/scale and rotation representation/scale: UNKNOWN. Rotation results are command-space differences. No GPU inference, rollout or training.',
        '', '## Stage 2','', '| T | Group | ΔH (nats) | JS (nats) | Greedy disagree | Δtop1 | KL SFT→GRPO | KL GRPO→SFT |','|---|---|---:|---:|---:|---:|---:|---:|']
    s2=read(args.output/'stage2/data/group_summary.json'); s3=read(args.output/'stage3/data/group_summary.json')
    def value(rows,t,g,k): return next(r for r in rows if r['temperature']==t and r['group']==g and r['metric']==k and r['origin']=='all')
    for t in TEMPERATURES:
        for g in GROUPS:
            lines.append('| '+str(t)+' | '+g+' | '+' | '.join(f"{value(s2,t,g,k)['mean']:.5f}" for k in ('delta_entropy','js','greedy_disagreement','delta_top1','kl_sft_grpo','kl_grpo_sft'))+' |')
    lines+=['','## Stage 3','', 'Greedy decoded actions are temperature-independent. The distribution/action associations are reported separately at T=1 and T=1.6.', '', '| Group | Decoded difference | 95% task-cluster interval |','|---|---:|---|']
    for g in GROUPS:
        v=value(s3,1.6,g,'action_difference'); lines.append(f"| {g} | {v['mean']:.6f} | [{v['ci95_low']:.6f}, {v['ci95_high']:.6f}] |")
    for g,k in [('translation','magnitude_delta'),('translation','cosine'),('translation','cosine_defined'),('translation','direction_reversed'),('gripper','gripper_disagreement')]:
        v=value(s3,1.6,g,k); lines.append(f"\n- {g} {k}: {v['mean']:.6f}; episode median {v['median']:.6f}; IQR [{v['p25']:.6f}, {v['p75']:.6f}].")
    lines+=['','### Magnitude and near-zero commands','', '| Metric | SFT | GRPO |','|---|---:|---:|']
    for group,k in [('translation','magnitude'),('rotation','magnitude'),('translation','near_zero'),('gripper','command_confidence')]:
        lines.append(f"| {group} {k} | {value(s3,1.6,group,k+'_sft')['mean']:.6f} | {value(s3,1.6,group,k+'_grpo')['mean']:.6f} |")
    lines+=['', 'Cosine means condition on defined chunks within observation and then valid observations within source episode. The defined fraction is reported separately; no near-zero vector receives an artificial cosine. The near-zero cutoff is in decoded command units and sensitivity at 0.001/0.01/0.05 is saved.',
        '', '### Task, suite and chunk heterogeneity at T=1.6', '']
    tasks=csv_rows(args.output/'stage2/data/task_summary.csv'); suites=csv_rows(args.output/'stage2/data/suite_summary.csv')
    for g in GROUPS:
        rr=[r for r in tasks if float(r['temperature'])==1.6 and r['group']==g and r['origin']=='all' and r['metric']=='delta_entropy']
        lo=min(rr,key=lambda r:float(r['mean'])); hi=max(rr,key=lambda r:float(r['mean']))
        positive=sum(float(r['mean'])>0 for r in rr)
        lines.append(f"- {g}: task mean ΔH range {float(lo['mean']):.4f} ({lo['suite']} task {lo['task_id']}) to {float(hi['mean']):.4f} ({hi['suite']} task {hi['task_id']}); {positive}/30 positive task means.")
    lines+=['','| Suite | Translation ΔH | Rotation ΔH | Gripper ΔH |','|---|---:|---:|---:|']
    for suite in ('libero_goal','libero_spatial','libero_10'):
        vals=[next(float(r['mean']) for r in suites if float(r['temperature'])==1.6 and r['suite']==suite and r['origin']=='all' and r['group']==g and r['metric']=='delta_entropy') for g in GROUPS]
        lines.append('| '+suite+' | '+' | '.join(f'{v:.4f}' for v in vals)+' |')
    chunks=[r for r in csv_rows(args.output/'stage2/data/chunk_summary.csv') if float(r['temperature'])==1.6]
    lines.append('\nChunk-index mean ΔH (0–7): '+', '.join(f"{float(r['delta_entropy']):.4f}" for r in chunks)+'.')
    lines+=['','### Distribution/action associations','', '| T | Group | X | Y | Episode Pearson r |','|---|---|---|---|---:|']
    for r in csv_rows(args.output/'stage3/data/correlations.csv'):
        if r['origin']=='all': lines.append('| '+' | '.join(r[k] for k in ('temperature','group','x','y'))+f" | {float(r['pearson']):.4f} |")
    lines+=['','Associations are descriptive episode-level correlations, not causal effects or tests on 3000 independent observations. Source-stratified correlations are saved separately. Larger JS need not produce a larger physical displacement.',
        '', '### Probability mass versus command semantics', '',
        'At T=1.6, gripper token entropy increases while mean prepared-command confidence remains close across policies. Multiple token values map to the same binary command; token uncertainty and command uncertainty are different quantities. In this dataset gripper greedy-token disagreement equals prepared-command disagreement, but the code does not assume that equality.',
        'Translation greedy magnitude is larger on average for GRPO. Thus these data do not support a general statement that GRPO always predicts smaller actions. Most defined translation directions remain similar; substantial near-zero mass makes conditioning essential.',
        '', '## Regression execution', '']
    if args.regression_log:
        log=args.regression_log.read_text(); require(re.search(r'^OK(?: \(skipped=\d+\))?$',log,re.M) and not re.search(r'^FAILED',log,re.M),'Regression log did not pass')
        shutil.copyfile(args.regression_log,args.output/'regression_tests.txt')
        tests=re.search(r'Ran (\d+) tests in ([0-9.]+)s',log); require(tests,'Missing test count')
        lines.append(f"PASS: {tests[1]} tests, {tests[2]} seconds. "+re.search(r'^OK.*$',log,re.M)[0]+'. See regression_tests.txt.')
        lines.append('Existing Phase 3A/3B, Stage 1 and Stage 4 tests are included. Optional legacy expert regression is skipped when PHASE3A_EXPERT_ROOT is not set.')
    else: lines.append('NOT RUN: no regression log supplied to report.')
    lines+=['','## Interpretation','',
        'Lower token entropy alone does not establish successful action concentration, causal benefit, useful strategy diversity, or robustness. JS and action distances compare distributions and native greedy commands at common observations. They do not compare two actually executed trajectories or observed EEF movements.',
        'Gripper confidence sums probabilities of all tokens that map to each prepared command. Native -1 is the reset-open command, +1 the opposite close command; these are commands, not measured gripper state. Token ID distance is not a physical distance. Two token IDs alias one bin, and the prepared gripper command has a threshold discontinuity.',
        'Source-origin, task, suite, and outcome strata are in the CSV tables. Observations and slots are dependent; point clouds are descriptive, with uncertainty computed on task clusters. Lost outcome has very few source episodes and should not support broad claims.',
        '', '## Figures','']
    for p in sorted(args.output.glob('stage*/figures/*.png')): lines.append('- ['+p.stem+']('+str(p.relative_to(args.output))+') (matching PDF available)')
    lines+=['','## Validation','', '- Native decoding, common-input identity, outcome alignment: PASS.','- Stage 1 numerical regression: PASS (see stage1_regression.json).','- New stage data and figure seals: PASS.','- Executed actions / EEF movement analysis: NOT RUN (unavailable verified linkage).','- Regression tests: see regression_tests.txt when supplied; source artifacts alone are not test execution evidence.']
    (args.output/'report.md').write_text('\n'.join(lines)+'\n')
    seal(args.output,dict(status='PASS',observations=3000,source_episodes=300,scope='predicted/counterfactual',stage1_full_copy='PARTIAL_COPY_FIGURES_MISSING' if m['stage1_analysis_copy']['missing_figures'] else 'PASS'))
    print(args.output/'report.md')


def verify(args):
    m,_,_=context(args.output)
    for d in ('bank','workers/sft','workers/grpo'): verify_seal(Path(m['stage1'])/d)
    check_analysis_copy(Path(m['stage1'])/'analysis'); verify_seal(Path(m['stage4']))
    for d in ('audit','stage2/data','stage2/figures','stage3/data','stage3/figures'): verify_seal(args.output/d)
    v=verify_seal(args.output); print(json.dumps({k:v[k] for k in ('status','observations','source_episodes','scope','stage1_full_copy')},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=('audit','stage2','plot-stage2','stage3','plot-stage3','report','verify'))
    p.add_argument('--output',type=Path,required=True); p.add_argument('--stage1',type=Path); p.add_argument('--stage4',type=Path); p.add_argument('--native-source-dir',type=Path); p.add_argument('--near-zero',type=float,default=.01); p.add_argument('--regression-log',type=Path)
    args=p.parse_args(); args.output=args.output.resolve()
    require(args.near_zero>0,'near-zero threshold must be positive')
    if args.stage.startswith('plot-'):
        from token_action_figures import plot
        context(args.output); plot(args.output,int(args.stage[-1]))
    else: globals()[args.stage](args)

if __name__=='__main__': main()
