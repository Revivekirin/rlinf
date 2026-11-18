"""Stage 3 extension: saved action-space marginals, no inference or rollouts."""
from __future__ import annotations
import argparse
import collections
import csv
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import numpy as np
from token_action_core import (ROOT,read,clean,avg,csv_rows,latest_pass,load_record,check_analysis_copy,episode_means)
from scripts.common.io import save,csv_save
from scripts.common.integrity import require,sha,seal,verify_seal
from scripts.common.action_schema import DIMENSIONS,GROUPS,token_values,prepared_gripper,SOURCE_HASHES,contract
from phase3b_distribution import distribution_metrics
from token_action_analysis import identity,check_outcomes,METRICS as OLD_METRICS
from action_space_core import *

TEMPS=(1.,1.6)
FIELDS=('js','delta_entropy','mu_sft','mu_grpo','delta_mu','abs_delta_mu','variance_sft','variance_grpo','delta_variance','w1','w1_normalized','greedy_abs','greedy_disagreement','delta_abs_mu','entropy_down_variance_up')
GROUP_FIELDS=FIELDS+('greedy_action_l2',)
BINARY=('p_open_sft','p_open_grpo','p_close_sft','p_close_grpo','p_other_sft','p_other_grpo','binary_entropy_sft','binary_entropy_grpo','delta_binary_entropy','delta_p_open','threshold_mass_sft','threshold_mass_grpo','token_entropy_sft','token_entropy_grpo','delta_token_entropy','decoded_command_disagreement','modal_command_disagreement')
SOURCES=('scripts/analysis/action_space_core.py','scripts/analysis/action_space_distribution.py','scripts/analysis/action_space_figures.py','tests/test_action_space_distribution.py','scripts/common/action_schema.py','scripts/analysis/token_action_core.py','scripts/analysis/token_action_analysis.py','scripts/analysis/phase3b_distribution.py','scripts/analysis/phase3b_generality.py','scripts/common/io.py','scripts/common/integrity.py')


def fresh(path):
    require(not path.exists(),f'Output already exists: {path}'); path.mkdir(parents=True); return path


def dump(path,rows): csv_save(path,clean(rows))


def audit(args):
    previous=(args.stage23 or latest_pass(ROOT/'outputs/analysis/token_action')).resolve()
    verify_seal(previous); previous_manifest=read(previous/'manifest.json')
    s1=(args.stage1 or latest_pass(ROOT/'outputs/analysis/distribution_generality',True)).resolve()
    require(Path(previous_manifest['stage1']).resolve()==s1,'Stage 2/3 and Stage 1 input lineage differs')
    for d in ('bank','workers/sft','workers/grpo'): verify_seal(s1/d)
    status=check_analysis_copy(s1/'analysis')
    for p,h in previous_manifest['code_hashes'].items(): require(sha(ROOT/p)==h,'Existing analysis source changed: '+p)
    for name,h in SOURCE_HASHES.items(): require(sha(previous/'audit/native_sources'/name)==h,'Pinned decoder source mismatch')
    entries=read(s1/'bank/manifest.json')['observations']; check_outcomes(entries)
    require(len(entries)==3000 and len({e['observation_id'] for e in entries})==3000,'Common-observation coverage')
    require(entries==read(previous/'audit/observations.json'),'Stage 2/3 observation ordering')
    require(set(collections.Counter((e['source_policy'],e['source_episode']) for e in entries).values())=={10},'This fixed-bank analysis requires 10 observations per source episode')
    manifests={p:read(s1/'workers'/p/'manifest.json') for p in ('sft','grpo')}
    require(all(v['mode']=='full' for v in manifests.values()),'Full replay required')
    out=fresh(args.output); d=fresh(out/'audit')
    save(d/'observations.json',entries); save(d/'workers.json',manifests); save(d/'contract.json',contract())
    (d/'source_audit.md').write_text('''# Stage 3 extension source audit

Existing modules are reused without modifying them: phase3b_distribution.distribution_metrics (JS/entropy), action_schema (native bin/decoder/gripper), token_action_core (loader, episode weighting), common integrity/io (seals).
Saved logits are [56,256], action arrays [8,7]. All 256 tokens in the saved native action support participate in offline softmax; no new token mask or top-k truncation is applied. Slot = chunk*7+dimension.
Native token IDs run in the opposite direction from action value. Two IDs alias the highest bin. W1 sorts decoded scalar support and integrates absolute CDF differences; repeated values have zero width, so alias mass is handled exactly. Expectations and variances sum token mass with its actual decoded value.
Native Q99 decoder, mask and +1e-8 range term come from the previous audited pinned source and are checked against every saved greedy action again. Models must have identical decoded support and normalization before comparison.
Translation/rotation are controller command channels; physical scale/frame/rotation representation remain UNKNOWN. W1 is a scalar marginal command distance, not joint 3-D Wasserstein or measured displacement. Group W1 is a descriptive average across three scalar marginals; cross-group physical magnitude comparisons are not supported. Normalized-bin W1 is saved separately.
Gripper preparation is -sign(2*decoded-1): -1 open, +1 close, 0 other. Binary entropy is defined only if other mass is exactly zero; otherwise it is unavailable rather than silently renormalized. Threshold proximity band is a configurable decoded-command width, not a native tolerance.
Predicted slots only; execution UNKNOWN. No simulator, model inference, training, or upstream changes. Stage 1 copied original figure files may be missing: this is disclosed separately from required numeric/worker validation.
''')
    manifest=dict(stage1=str(s1),stage23=str(previous),stage1_copy_status=status,temperatures=TEMPS,
        input_seals={str(p):sha(p) for p in [previous/'validation.json']+[s1/x/'validation.json' for x in ('bank','workers/sft','workers/grpo','analysis')]},
        source_hashes={p:sha(ROOT/p) for p in SOURCES},git_head=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
        bootstrap_seed=SEED,bootstrap_iterations=args.bootstrap,threshold_band=args.threshold_band,
        weighting='slots/chunks within observation; observations within source episode; equal source episodes. Whole tasks resampled within suite. Conditional bins keep baseline equal-episode weights; quantile edges fixed.',
        quantile='inverse weighted empirical CDF; ties may create empty bins; no invented points',execution='UNKNOWN',cpu_only=True)
    save(out/'manifest.json',manifest); seal(d,dict(status='PASS',observations=3000,source_episodes=300,stage23='PASS',stage1_numeric='PASS'))
    print(json.dumps(dict(status='PASS',stage23=str(previous),stage1=str(s1),stage1_copy_status=status),indent=2))


def context(out):
    verify_seal(out/'audit'); m=read(out/'manifest.json')
    for p,h in m['source_hashes'].items(): require(sha(ROOT/p)==h,'Analysis source changed: '+p)
    for p,h in m['input_seals'].items(): require(sha(Path(p))==h,'Input validation changed: '+p)
    return m,read(out/'audit/observations.json'),read(out/'audit/workers.json')


def mapping(args):
    m,entries,manifests=context(args.output)
    support={p:action_support(v) for p,v in manifests.items()}
    require(manifests['sft']['action_norm_stats']==manifests['grpo']['action_norm_stats'],'Normalization mismatch')
    require(np.array_equal(support['sft'],support['grpo']),'Action-value support differs between models')
    require(manifests['sft']['token_start']==manifests['grpo']['token_start'],'Token support differs')
    out=fresh(args.output/'mapping'); rows=[]; x=support['sft']
    for dim in range(7):
        order=np.argsort(x[dim],kind='stable'); ranks=np.argsort(order)
        for b in range(x.shape[-1]):
            rows.append(dict(dimension=DIMENSIONS[dim],token_id=manifests['sft']['token_start']+b,token_offset=b,bin_index=manifests['sft']['token_to_bin_index'][b],normalized_value=token_values(manifests['sft'])[b],decoded_value=x[dim,b],sorted_rank=int(ranks[b]),gripper_command=int(prepared_gripper(x[dim,b])) if dim==6 else None))
    dump(out/'action_bins.csv',rows); np.save(out/'decoded_support.npy',x)
    for i,e in enumerate(entries):
        for pol in ('sft','grpo'): load_record(Path(m['stage1']),pol,e,manifests[pol])
        if (i+1)%500==0: print('MAPPING',i+1,'/ 3000',flush=True)
    seal(out,dict(status='PASS',saved_records=6000,decoded_consistency='PASS',identical_support=True,unique_decoded_values=[len(np.unique(v)) for v in x],other_gripper_bins=int((prepared_gripper(x[6])==0).sum())))


def aggregate_summary(ep,metrics,iterations,fields=()):
    groups=collections.defaultdict(list)
    for r in ep:
        for origin in ('all',r['origin']): groups[tuple(r[k] for k in ('temperature','group','scope',*fields))+(origin,)].append(r)
    rows=[]
    for key,rr in sorted(groups.items()):
        base=dict(zip(('temperature','group','scope',*fields,'origin'),key)); vals=np.array([[np.nan if r[k] is None else r[k] for k in metrics] for r in rr],float)
        task,counts=task_bootstrap(rr,iterations) if iterations else (None,None)
        for j,metric in enumerate(metrics):
            y=vals[:,j]; good=np.isfinite(y); v=y[good]; ci=[]
            if len(v) and iterations:
                weights=counts[:,task[good]]; den=weights.sum(1); draws=np.divide(weights@v,den,out=np.full(iterations,np.nan),where=den>0); ci=np.quantile(draws[np.isfinite(draws)],[.025,.975])
            bytask=collections.defaultdict(list)
            for r,z in zip(rr,y):
                if np.isfinite(z): bytask[(r['suite'],r['task_id'])].append(z)
            rows.append(dict(base,metric=metric,n_episodes=len(rr),n_valid_episodes=len(v),mean=float(v.mean()) if len(v) else None,median=float(np.median(v)) if len(v) else None,p25=float(np.quantile(v,.25)) if len(v) else None,p75=float(np.quantile(v,.75)) if len(v) else None,positive_fraction=float((v>0).mean()) if len(v) else None,negative_fraction=float((v<0).mean()) if len(v) else None,task_macro_mean=avg([np.mean(z) for z in bytask.values()]),ci_low=ci[0] if len(ci) else None,ci_high=ci[1] if len(ci) else None))
    return rows


def write_aggregates(out,rows,metrics,iterations,prefix=''):
    ep=episode_means(rows,metrics)
    dump(out/(prefix+'episode_summary.csv'),ep); save(out/(prefix+'episodes.json'),clean(ep))
    summary=aggregate_summary(ep,metrics,iterations)
    dump(out/(prefix+'summary.csv'),summary); save(out/(prefix+'summary.json'),clean(summary))
    for name,fields in [('task',('suite','task_id')),('suite',('suite',)),('outcome',('outcome',))]: dump(out/(prefix+name+'_summary.csv'),aggregate_summary(ep,metrics,0,fields))
    return ep,summary


def analyze(args):
    m,entries,manifests=context(args.output); verify_seal(args.output/'mapping'); out=fresh(args.output/'data')
    old=Path(m['stage23']); s1=Path(m['stage1']); x=np.load(args.output/'mapping/decoded_support.npy'); n=token_values(manifests['sft'])
    N=len(entries); cache=np.empty((2,N,8,7,len(FIELDS))); binary=np.empty((2,N,8,len(BINARY)))
    greedy_l2=np.empty((N,8,3)); rows=[]; chunks=[]; gripper=[]; dimensions=[]
    with np.load(old/'stage2/data/metrics.npz') as z: old_metrics=z['metrics']; old_confidence=z['gripper_confidence']
    old_observation={(r['observation_id'],r['temperature'],r['group']):r for r in read(old/'stage3/data/observations.json')}
    old_s1={(r['observation_id'],float(r['temperature'])):r for r in csv_rows(s1/'analysis/observations.csv')}
    max_errors=collections.defaultdict(float)
    with (out/'slot_metrics.csv').open('x',newline='') as file:
        writer=None
        for i,e in enumerate(entries):
            data=[load_record(s1,p,e,manifests[p])[0] for p in ('sft','grpo')]; base=identity(e)
            decoded=[d['decoded_actions'] for d in data]; absdiff=np.abs(decoded[1]-decoded[0]); disagreement=data[0]['action_tokens']!=data[1]['action_tokens']
            for gi,(g,ds) in enumerate(GROUPS.items()): greedy_l2[i,:,gi]=np.linalg.norm((decoded[1]-decoded[0])[:,ds],axis=-1)
            for ti,t in enumerate(TEMPS):
                probs=[softmax(d['logits'].reshape(8,7,-1),t) for d in data]
                dm=distribution_metrics(data[0]['logits'],data[1]['logits'],t)
                dm['delta_entropy']=dm['entropy_grpo']-dm['entropy_sft']; dm['delta_top1']=dm['top1_grpo']-dm['top1_sft']
                for k in OLD_METRICS:
                    error=np.max(np.abs(dm[k].reshape(8,7)-old_metrics[ti,i,:,:,OLD_METRICS.index(k)])); max_errors['stage2_slots']=max(max_errors['stage2_slots'],float(error)); require(error<1e-10,'Stage 2 slot regression '+k)
                    error=abs(float(dm[k].mean())-float(old_s1[(e['observation_id'],t)][k])); max_errors['stage1_observations']=max(max_errors['stage1_observations'],error); require(error<1e-10,'Stage 1 regression '+k)
                mus,vars_=zip(*(moments(p,x) for p in probs))
                w=wasserstein1(probs[0],probs[1],x); wn=wasserstein1(probs[0],probs[1],n)
                dh=dm['delta_entropy'].reshape(8,7); dv=vars_[1]-vars_[0]
                values=dict(js=dm['js'].reshape(8,7),delta_entropy=dh,mu_sft=mus[0],mu_grpo=mus[1],delta_mu=mus[1]-mus[0],abs_delta_mu=abs(mus[1]-mus[0]),variance_sft=vars_[0],variance_grpo=vars_[1],delta_variance=dv,w1=w,w1_normalized=wn,greedy_abs=absdiff,greedy_disagreement=disagreement,delta_abs_mu=abs(mus[1])-abs(mus[0]),entropy_down_variance_up=(dh<0)&(dv>0))
                arr=np.stack([values[k] for k in FIELDS],-1); cache[ti,i]=arr
                for c,d in np.ndindex(8,7):
                    r=dict(base,temperature=t,chunk=c,dimension=DIMENSIONS[d],**dict(zip(FIELDS,arr[c,d])))
                    if writer is None: writer=csv.DictWriter(file,fieldnames=list(r)); writer.writeheader()
                    writer.writerow(r)
                for gi,(g,ds) in enumerate(GROUPS.items()):
                    group=arr[:,ds,:].mean(axis=1); pooled=group.mean(axis=0)
                    r=dict(base,temperature=t,group=g,scope='predicted',**dict(zip(FIELDS,pooled)),greedy_action_l2=float(greedy_l2[i,:,gi].mean())); rows.append(r)
                    oldr=old_observation[(e['observation_id'],t,g)]
                    for newkey,oldkey in [('js','js'),('delta_entropy','delta_entropy'),('greedy_disagreement','greedy_disagreement'),('greedy_action_l2','action_difference')]:
                        error=abs(r[newkey]-oldr[oldkey]); max_errors['stage3_observations']=max(max_errors['stage3_observations'],error); require(error<1e-10,'Stage 3 regression '+newkey)
                    for c in range(8): chunks.append(dict(base,temperature=t,group=g,scope='chunk:'+str(c),**dict(zip(FIELDS,group[c])),greedy_action_l2=float(greedy_l2[i,c,gi])))
                for d,name in enumerate(DIMENSIONS): dimensions.append(dict(base,temperature=t,group=name,scope='predicted',**dict(zip(FIELDS,arr[:,d].mean(0)))))
                bg=[gripper_distribution(p[:,6],x[6],m['threshold_band']) for p in probs]
                bv={}
                for j,pol in enumerate(('sft','grpo')):
                    for k in ('p_open','p_close','p_other','binary_entropy','threshold_mass'): bv[k+'_'+pol]=bg[j][k]
                    bv['token_entropy_'+pol]=dm['entropy_'+pol].reshape(8,7)[:,6]
                    error=np.max(abs(bg[j]['p_open']-old_confidence[ti,i,j,:,0])); max_errors['gripper_probabilities']=max(max_errors['gripper_probabilities'],float(error)); require(error<1e-10,'Stage 3 gripper confidence regression')
                bv.update(delta_p_open=bg[1]['p_open']-bg[0]['p_open'],delta_binary_entropy=bg[1]['binary_entropy']-bg[0]['binary_entropy'],delta_token_entropy=dh[:,6],decoded_command_disagreement=prepared_gripper(decoded[0][:,6])!=prepared_gripper(decoded[1][:,6]),modal_command_disagreement=np.stack([bg[0][k] for k in ('p_open','p_close','p_other')],-1).argmax(-1)!=np.stack([bg[1][k] for k in ('p_open','p_close','p_other')],-1).argmax(-1))
                binary[ti,i]=np.stack([bv[k] for k in BINARY],-1)
                gripper.append(dict(base,temperature=t,group='gripper',scope='predicted',**{k:avg(v) for k,v in bv.items()}))
            if (i+1)%500==0: print('METRICS',i+1,'/ 3000',flush=True)
    np.savez_compressed(out/'metrics.npz',metrics=cache,binary=binary,greedy_l2=greedy_l2)
    dump(out/'observation_metrics.csv',rows); save(out/'observations.json',clean(rows)); dump(out/'chunk_observations.csv',chunks)
    dump(out/'gripper_binary.csv',gripper); save(out/'gripper_observations.json',clean(gripper))
    ep,summary=write_aggregates(out,rows,GROUP_FIELDS,m['bootstrap_iterations'])
    write_aggregates(out,gripper,BINARY,m['bootstrap_iterations'],'gripper_')
    dim_ep,_=write_aggregates(out,dimensions,FIELDS,0,'dimension_')
    chunk_ep=episode_means(chunks,GROUP_FIELDS); dump(out/'chunk_summary.csv',aggregate_summary(chunk_ep,GROUP_FIELDS,0))
    # Chunk means must reproduce pooled results before interpreting aggregation effects.
    for ti,t in enumerate(TEMPS):
        for gi,g in enumerate(GROUPS):
            rr=[r for r in rows if r['temperature']==t and r['group']==g]
            cc=[r for r in chunks if r['temperature']==t and r['group']==g]
            for key in GROUP_FIELDS: require(abs(np.mean([r[key] for r in rr])-np.mean([r[key] for r in cc]))<1e-10,'Chunk/pooled regression')
    corr=[]; binned=[]
    for ti,t in enumerate(TEMPS):
        for gi,(g,ds) in enumerate(GROUPS.items()):
            obs=[r for r in rows if r['temperature']==t and r['group']==g]; episodes=[r for r in ep if r['temperature']==t and r['group']==g]
            tasks=[]
            for suite,taskid in sorted({(r['suite'],r['task_id']) for r in episodes}):
                rr=[r for r in episodes if r['suite']==suite and r['task_id']==taskid]
                tasks.append(dict(rr[0],episode_id=suite+str(taskid),origin='task',**{k:avg([r[k] for r in rr]) for k in GROUP_FIELDS}))
            for level,selected in [('observation',obs),('episode',episodes),('task',tasks)]:
                for y in ('w1','greedy_action_l2'):
                    corr.append(dict(temperature=t,group=g,level=level,scope='pooled',stratum='all',x='js',y=y,**correlation(selected,'js',y,m['bootstrap_iterations'])))
            for field in ('origin','outcome'):
                for label in sorted({r[field] for r in episodes}):
                    rr=[r for r in episodes if r[field]==label]
                    for y in ('w1','greedy_action_l2'): corr.append(dict(temperature=t,group=g,level='episode',scope='pooled',stratum=field+':'+str(label),x='js',y=y,**correlation(rr,'js',y,0)))
            # A is defined at scalar slots, not binary-scatter observation means.
            slotrows=[]
            for i,e in enumerate(entries):
                base=identity(e)
                for c in range(8):
                    for d in ds: slotrows.append(dict(base,js=cache[ti,i,c,d,FIELDS.index('js')],greedy_disagreement=cache[ti,i,c,d,FIELDS.index('greedy_disagreement')]))
            for r in quantile_bins(slotrows,'js','greedy_disagreement',m['bootstrap_iterations'],'mean'): binned.append(dict(temperature=t,group=g,figure='A',scope='slot_pooled',x='js',y='greedy_disagreement',**r))
            del slotrows
            for y,fig in [('w1','B'),('greedy_action_l2','C')]:
                for r in quantile_bins(obs,'js',y,m['bootstrap_iterations']): binned.append(dict(temperature=t,group=g,figure=fig,scope='pooled',x='js',y=y,**r))
                for c in range(8):
                    cc=[r for r in chunks if r['temperature']==t and r['group']==g and r['scope']=='chunk:'+str(c)]
                    corr.append(dict(temperature=t,group=g,level='observation',scope='chunk:'+str(c),stratum='all',x='js',y=y,**correlation(cc,'js',y,0)))
                    if fig=='C':
                        for r in quantile_bins(cc,'js',y,0): binned.append(dict(temperature=t,group=g,figure=fig,scope='chunk:'+str(c),x='js',y=y,**r))
            print('RELATIONSHIPS',t,g,flush=True)
    dump(out/'correlations.csv',corr); dump(out/'quantile_bins.csv',binned)
    # Same argmax does not imply equal marginals; report within-dimension conditioning.
    conditional=[]
    for ti,t in enumerate(TEMPS):
        for d,name in enumerate(DIMENSIONS):
            arr=cache[ti,:,:,d]; changed=arr[:,:,FIELDS.index('greedy_disagreement')]>0
            for selected,label in [(~changed,'same_argmax'),(changed,'different_argmax'),(arr[:,:,FIELDS.index('delta_entropy')]<0,'entropy_decrease')]:
                count=int(selected.sum())
                result=dict(temperature=t,dimension=name,condition=label,slot_count=count)
                for key in ('w1','abs_delta_mu','delta_variance','delta_abs_mu','js','greedy_abs'):
                    a=arr[:,:,FIELDS.index(key)][selected]; result['mean_'+key]=float(a.mean()) if count else None
                result['fraction_variance_increase']=float((arr[:,:,FIELDS.index('delta_variance')][selected]>0).mean()) if count else None
                conditional.append(result)
    dump(out/'conditional_summary.csv',conditional)
    save(out/'regression.json',dict(status='PASS',max_absolute_errors=max_errors,chunk_pooled='PASS',stage1='PASS',stage2='PASS',stage3='PASS'))
    seal(out,dict(status='PASS',observations=N,slot_rows=2*N*56,temperatures=TEMPS,bootstrap_iterations=m['bootstrap_iterations'],units='scalar marginal native command; no physical cross-dimension comparison'))


def cases(args):
    m,entries,manifests=context(args.output); verify_seal(args.output/'data'); out=fresh(args.output/'cases')
    with np.load(args.output/'data/metrics.npz') as z: cache=z['metrics']; binary=z['binary']
    x=np.load(args.output/'mapping/decoded_support.npy'); selected=[]; rules=[]
    # Selection rules are fixed before inspecting any individual probability plot.
    definitions=['high_js_low_greedy','low_js_high_greedy','high_js_high_w1','entropy_down_variance_down','gripper_token_up_binary_down']
    for ti,t in enumerate(TEMPS):
        for label in definitions:
            candidates=[]
            for d in ([6] if label.startswith('gripper') else range(6)):
                a=cache[ti,:,:,d]; js=a[:,:,FIELDS.index('js')]; greedy=a[:,:,FIELDS.index('greedy_abs')]; w=a[:,:,FIELDS.index('w1')]; dh=a[:,:,FIELDS.index('delta_entropy')]; dv=a[:,:,FIELDS.index('delta_variance')]
                if label=='high_js_low_greedy': mask=(js>=np.quantile(js,.9))&(greedy<=np.quantile(greedy,.1)); score=js
                elif label=='low_js_high_greedy': mask=(js<=np.quantile(js,.1))&(greedy>=np.quantile(greedy,.9)); score=greedy/max(float(np.max(greedy)),1e-12)
                elif label=='high_js_high_w1': mask=(js>=np.quantile(js,.9))&(w>=np.quantile(w,.9)); score=w/max(float(np.max(w)),1e-12)
                elif label=='entropy_down_variance_down': mask=(dh<=np.quantile(dh,.1))&(dv<=np.quantile(dv,.1))&(dh<0)&(dv<0); score=-dh
                else:
                    db=binary[ti,:,:,BINARY.index('delta_binary_entropy')]; mask=(dh>0)&(db<0); score=dh-db
                for i,c in np.argwhere(mask): candidates.append((float(score[i,c]),entries[i]['observation_id'],int(c),d,int(i)))
            rules.append(dict(temperature=t,rule=label,candidates=len(candidates),status='SELECTED' if candidates else 'NO_QUALIFYING_CASE',definition='Within-dimension 10th/90th percentile intersections; largest stated score; deterministic observation/chunk/dimension tie-break. No example is generalized.'))
            if not candidates: continue
            score,oid,c,d,i=sorted(candidates,key=lambda v:(-v[0],v[1],v[2],v[3]))[0]; e=entries[i]
            probs=[]; greeds=[]
            for pol in ('sft','grpo'):
                z,_=load_record(Path(m['stage1']),pol,e,manifests[pol]); probs.append(softmax(z['logits'],t)[c*7+d]); greeds.append(float(z['decoded_actions'][c,d]))
            unique,inv=np.unique(x[d],return_inverse=True); mass=np.array([np.bincount(inv,weights=p,minlength=len(unique)) for p in probs]); expected=mass@unique
            name=label+'_T'+str(t).replace('.','p'); np.savez_compressed(out/(name+'.npz'),support=unique,probability=mass,greedy=greeds,expected=expected)
            selected.append(dict(temperature=t,rule=label,case_file=name+'.npz',observation_id=oid,origin=e['source_policy'],episode_id=e['source_episode'],suite=e['suite'],task_id=e['task_id'],outcome=e['outcome'],chunk=c,dimension=DIMENSIONS[d],score=score,js=cache[ti,i,c,d,FIELDS.index('js')],w1=cache[ti,i,c,d,FIELDS.index('w1')],greedy_abs=cache[ti,i,c,d,FIELDS.index('greedy_abs')],delta_entropy=cache[ti,i,c,d,FIELDS.index('delta_entropy')],delta_variance=cache[ti,i,c,d,FIELDS.index('delta_variance')]))
    dump(out/'representative_cases.csv',selected); save(out/'selection_rules.json',rules)
    from action_space_figures import plot_cases
    plot_cases(out,selected)
    seal(out,dict(status='PASS',selected_cases=len(selected),rules=len(rules),not_generalizable=True))


def finalize(args):
    m,_,_=context(args.output)
    for d in ('mapping','data','figures','cases'): verify_seal(args.output/d)
    require(args.test_log is not None,'Provide executed regression test log')
    log=args.test_log.read_text(); require(re.search(r'^OK(?: \(skipped=\d+\))?$',log,re.M) and not re.search(r'^FAILED',log,re.M),'Regression tests did not pass')
    shutil.copyfile(args.test_log,args.output/'regression_tests.txt')
    from action_space_figures import write_report
    write_report(args.output)
    seal(args.output,dict(status='PASS',observations=3000,source_episodes=300,scope='predicted scalar marginals and greedy counterfactual commands',cpu_only=True))
    verify(args)


def verify(args):
    m,_,_=context(args.output); verify_seal(Path(m['stage23']))
    for d in ('bank','workers/sft','workers/grpo'): verify_seal(Path(m['stage1'])/d)
    check_analysis_copy(Path(m['stage1'])/'analysis')
    for d in ('audit','mapping','data','figures','cases'): verify_seal(args.output/d)
    v=verify_seal(args.output); print(json.dumps({k:v[k] for k in ('status','observations','source_episodes','scope','cpu_only')},indent=2))


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('stage',choices=('audit','mapping','analyze','figures','cases','finalize','verify'))
    p.add_argument('--output',type=Path,required=True); p.add_argument('--stage1',type=Path); p.add_argument('--stage23',type=Path)
    p.add_argument('--bootstrap',type=int,default=500); p.add_argument('--threshold-band',type=float,default=.05); p.add_argument('--test-log',type=Path)
    a=p.parse_args(); a.output=a.output.resolve(); require(a.bootstrap>0 and a.threshold_band>0,'Invalid analysis settings')
    if a.stage=='figures':
        context(a.output)
        from action_space_figures import plot
        plot(a.output)
    else: globals()[a.stage](a)

if __name__=='__main__': main()
