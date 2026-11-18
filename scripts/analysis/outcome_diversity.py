"""R3: outcome-conditioned, same-initial-state trajectory diversity (CPU only)."""
from __future__ import annotations
import argparse
import collections
import csv
import itertools
import json
import math
import os
import shutil
from pathlib import Path
import sys
import tempfile

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts/analysis'))
from scripts.common.io import read, save, csv_save
from scripts.common.integrity import require, sha, verify_seal, seal
from phase3a_core import legacy
from phase3b_pilot import repeat_validate_state, REPEAT_POLICIES, check_repeat_entry

CATEGORIES = ('SS','SF','FF')
REFERENCE = {'sft': .154012, 'grpo': .133010, 'cross': .185928}
REFERENCE_SUCCESS = {'sft': 9, 'grpo': 38}


def rows(path, required):
    with path.open(newline='') as f:
        reader = csv.DictReader(f)
        require(reader.fieldnames is not None and set(required) <= set(reader.fieldnames), f'CSV schema: {path}')
        return list(reader)


def truth(x):
    require(str(x) in ('True','False'), f'Invalid success: {x}')
    return str(x) == 'True'


def category(a, b):
    return 'SS' if a and b else 'FF' if not a and not b else 'SF'


def canonical_pair(a, b):
    require(a != b, 'Self pair')
    return tuple(sorted((a,b)))


def finite_mean(values):
    return float(np.mean(values)) if values else None


def quantiles(values):
    if not values: return dict(median_m=None,p25_m=None,p75_m=None)
    return dict(median_m=float(np.median(values)),p25_m=float(np.quantile(values,.25)),p75_m=float(np.quantile(values,.75)))


def decompose(pair_rows, episode_rows, *, expected_states=10):
    """Pairs are descriptive; state is the unit of policy comparison."""
    jobs={r['job_id']:r for r in episode_rows}
    require(len(jobs)==len(episode_rows), 'Duplicate episode job')
    groups=collections.defaultdict(list)
    for r in episode_rows:
        groups[(r['episode_id'],r['policy'])].append(r)
    require(len(groups)==2*expected_states, 'Unexpected state/policy groups')
    pairs=collections.defaultdict(list); seen=set()
    for r in pair_rows:
        a,b=r['job_a'],r['job_b']; key=canonical_pair(a,b)
        require(a in jobs and b in jobs and key not in seen, 'Unknown or duplicate pair')
        seen.add(key); left,right=jobs[a],jobs[b]
        require(left['episode_id']==right['episode_id']==r['episode_id'], 'Different initial states in pair')
        kind='cross' if left['policy']!=right['policy'] else left['policy']
        require(r['pair_type']==kind, 'Pair policy mismatch')
        require(left['success']==truth(r['success_a']) and right['success']==truth(r['success_b']), 'Pair outcome mismatch')
        value=float(r['trajectory_rms_32_m']); require(np.isfinite(value) and value>=0,'Invalid RMS')
        r['outcome_category']=category(left['success'],right['success'])
        pairs[(r['episode_id'],kind)].append(r)
    state_rows=[]; decomposition=[]
    for (eid,policy), group in sorted(groups.items()):
        require(policy in REPEAT_POLICIES and len(group)==5, 'Five rollouts per state/policy required')
        require(len({r['policy_seed'] for r in group})==5, 'Duplicate seed')
        require(len({r['initial_state_sha256'] for r in group})==1, 'Initial state hash mismatch')
        pp=pairs[(eid,policy)]
        expected={canonical_pair(a['job_id'],b['job_id']) for a,b in itertools.combinations(group,2)}
        require(len(pp)==10 and {canonical_pair(r['job_a'],r['job_b']) for r in pp}==expected, 'Incomplete within-policy pairs')
        m=sum(r['success'] for r in group); n=len(group); counts={'SS':math.comb(m,2),'SF':m*(n-m),'FF':math.comb(n-m,2)}
        require(sum(counts.values())==math.comb(n,2),'Category count identity')
        actual=collections.Counter(r['outcome_category'] for r in pp)
        require(all(actual[c]==counts[c] for c in CATEGORIES), 'Outcome count mismatch')
        values={c:[float(r['trajectory_rms_32_m']) for r in pp if r['outcome_category']==c] for c in CATEGORIES}
        D={c:finite_mean(values[c]) for c in CATEGORIES}; all_mean=finite_mean([float(r['trajectory_rms_32_m']) for r in pp])
        contrib={c:counts[c]/10*D[c] if D[c] is not None else 0.0 for c in CATEGORIES}
        require(math.isclose(all_mean,sum(contrib.values()),rel_tol=1e-12,abs_tol=1e-12),'Weighted decomposition mismatch')
        row=dict(episode_id=eid,task_id=int(group[0]['task_id']),trial_id=int(group[0]['trial_id']),policy=policy,
                 initial_state_sha256=group[0]['initial_state_sha256'],n_rollouts=n,n_success=m,n_failure=n-m,
                 success_rate=m/n,n_all_pairs=10,D_all_m=all_mean,
                 success_steps=json.dumps([r['steps'] for r in group if r['success']]),
                 failure_steps=json.dumps([r['steps'] for r in group if not r['success']]))
        for c in CATEGORIES:
            row[f'n_{c}_pairs']=counts[c]; row[f'D_{c}_m']=D[c]
            row[f'{c}_status']='DEFINED' if D[c] is not None else 'UNDEFINED'
            row.update({f'{c}_{k}':v for k,v in quantiles(values[c]).items()})
        state_rows.append(row)
        decomposition.append(dict(episode_id=eid,task_id=row['task_id'],trial_id=row['trial_id'],policy=policy,
                                  D_all_m=all_mean,**{f'w_{c}':counts[c]/10 for c in CATEGORIES},
                                  **{f'C_{c}_m':contrib[c] for c in CATEGORIES}))
    for eid in {r['episode_id'] for r in episode_rows}:
        require(len(pairs[(eid,'cross')])==25, 'Cross-policy pair grid')
        a=groups[(eid,'sft')]; b=groups[(eid,'grpo')]
        require({canonical_pair(x['job_id'],y['job_id']) for x in a for y in b}==
                {canonical_pair(r['job_a'],r['job_b']) for r in pairs[(eid,'cross')]}, 'Cross-policy pair coverage')
        require(a[0]['initial_state_sha256']==b[0]['initial_state_sha256'], 'Cross-policy initial state mismatch')
    require(len(seen)==45*expected_states,'Total pair count')
    return state_rows,decomposition


def aggregate(state_rows, pair_rows):
    by_policy=collections.defaultdict(list); by_task=collections.defaultdict(dict)
    for r in state_rows:
        by_policy[r['policy']].append(r); by_task[(r['task_id'],r['trial_id'])][r['policy']]=r
    require(all(set(v)==set(REPEAT_POLICIES) for v in by_task.values()),'Missing matched policy state')
    policy_rows=[]
    for policy,rr in sorted(by_policy.items()):
        all_d=[r['D_all_m'] for r in rr]
        row=dict(policy=policy,n_states=len(rr),n_rollouts=sum(r['n_rollouts'] for r in rr),n_success=sum(r['n_success'] for r in rr),
                 success_rate=sum(r['n_success'] for r in rr)/sum(r['n_rollouts'] for r in rr),
                 state_success_rate_min=min(r['success_rate'] for r in rr),state_success_rate_max=max(r['success_rate'] for r in rr),
                 D_all_state_mean_m=finite_mean(all_d),**{f'D_all_{k}':v for k,v in quantiles(all_d).items()})
        for c in CATEGORIES:
            valid=[r[f'D_{c}_m'] for r in rr if r[f'D_{c}_m'] is not None]
            row[f'D_{c}_state_mean_m']=finite_mean(valid); row[f'D_{c}_valid_states']=len(valid)
            row[f'{c}_pair_count']=sum(r[f'n_{c}_pairs'] for r in rr)
            row.update({f'D_{c}_{k}':v for k,v in quantiles(valid).items()})
        policy_rows.append(row)
    task_rows=[]; paired=[]
    for (task,trial), v in sorted(by_task.items()):
        s,g=v['sft'],v['grpo']; paired.append(dict(task_id=task,trial_id=trial,episode_id=s['episode_id'],
            D_all_sft_m=s['D_all_m'],D_all_grpo_m=g['D_all_m'],D_all_diff_grpo_minus_sft_m=g['D_all_m']-s['D_all_m'],
            success_rate_sft=s['success_rate'],success_rate_grpo=g['success_rate'],
            D_SS_sft_m=s['D_SS_m'],D_SS_grpo_m=g['D_SS_m']))
    for task in sorted({r['task_id'] for r in paired}):
        rr=[r for r in paired if r['task_id']==task]; diffs=[r['D_all_diff_grpo_minus_sft_m'] for r in rr]
        task_rows.append(dict(task_id=task,n_states=len(rr),D_all_diff_mean_m=finite_mean(diffs),
            **{f'D_all_diff_{k}':v for k,v in quantiles(diffs).items()},
            grpo_higher_states=sum(d>0 for d in diffs),grpo_lower_states=sum(d<0 for d in diffs),
            sft_success_rate=float(np.mean([r['success_rate_sft'] for r in rr])),
            grpo_success_rate=float(np.mean([r['success_rate_grpo'] for r in rr])),
            common_SS_states=sum(r['D_SS_sft_m'] is not None and r['D_SS_grpo_m'] is not None for r in rr)))
    cross=[float(r['trajectory_rms_32_m']) for r in pair_rows if r['pair_type']=='cross']
    return policy_rows,task_rows,paired,dict(cross_pair_mean_m=finite_mean(cross),
        common_SS_states=sum(r['D_SS_sft_m'] is not None and r['D_SS_grpo_m'] is not None for r in paired),
        common_SS_sft_mean_m=finite_mean([r['D_SS_sft_m'] for r in paired if r['D_SS_sft_m'] is not None and r['D_SS_grpo_m'] is not None]),
        common_SS_grpo_mean_m=finite_mean([r['D_SS_grpo_m'] for r in paired if r['D_SS_sft_m'] is not None and r['D_SS_grpo_m'] is not None]))


def inspect_inputs(run, diversity, *, metric=None):
    run=run.resolve(); diversity=diversity.resolve()
    old=verify_seal(diversity)
    dm=read(diversity/'manifest.json')
    require(Path(dm['source_run']).resolve()==run,'Diversity source run mismatch')
    require(old['episodes']==100 and old['states']==10 and old['pairs']==450,'Old diversity validation counts')
    for name,digest in dm['input_hashes'].items(): require(sha(name)==digest,f'Old diversity input changed: {name}')
    rm=read(run/'manifest.json'); plan=Path(rm['plan']); verify_seal(plan); pl=read(plan/'manifest.json')
    state=repeat_validate_state(run,pl)
    require(state['status']=='COMPLETE_VALID' and state['completed_jobs']==100,'Repeat run incomplete')
    require(pl['experiment']['suite']=='libero_10' and set(pl['experiment']['task_ids'])=={3,8}
            and set(pl['experiment']['trial_ids'])==set(range(5)) and set(pl['experiment']['policies'])==set(REPEAT_POLICIES)
            and pl['experiment']['repeats']==5,'Unexpected repeated rollout plan')
    er=rows(diversity/'per_episode.csv',('job_id','episode_id','policy','task_id','trial_id','policy_seed','success','steps','initial_state_sha256'))
    pr=rows(diversity/'pairwise_diversity.csv',('episode_id','pair_type','job_a','job_b','success_a','success_b','trajectory_rms_32_m'))
    sr=rows(diversity/'same_state_summary.csv',('episode_id','sft_success','grpo_success','sft_pairwise_rms_mean_m','grpo_pairwise_rms_mean_m','cross_pairwise_rms_mean_m'))
    require((len(er),len(pr),len(sr))==(100,450,10),'Old CSV row counts')
    jobs={j['job_id']:j for j in pl['jobs']}; require(len(jobs)==100 and {r['job_id'] for r in er}==set(jobs),'Episode/job mismatch')
    episodes=[]; paths={}; hashes={str(diversity/n):sha(diversity/n) for n in ('manifest.json','validation.json','per_episode.csv','pairwise_diversity.csv','same_state_summary.csv')}
    for row in er:
        j=jobs[row['job_id']]; entry=read(run/'entries'/(j['job_id']+'.json'))
        source=Path(entry['episode_path']); meta=read(source/'metadata.json')
        require(all(str(row[k])==str(j[k]) for k in ('episode_id','policy','task_id','trial_id','policy_seed','initial_state_sha256')), 'Episode CSV identity mismatch')
        require(truth(row['success'])==bool(meta['success']) and int(row['steps'])==meta['episode_length'], 'Episode CSV/metadata mismatch')
        with np.load(source/'states.npz',allow_pickle=False) as z: xyz=z['robot0_eef_pos'].copy()
        require(xyz.shape==(meta['episode_length']+1,3) and np.isfinite(xyz).all(), 'Missing/invalid EEF trajectory')
        paths[j['job_id']]=xyz
        episodes.append(dict(job_id=j['job_id'],episode_id=j['episode_id'],task_id=j['task_id'],trial_id=j['trial_id'],
            policy=j['policy'],policy_seed=j['policy_seed'],success=bool(meta['success']),steps=int(meta['episode_length']),
            initial_state_sha256=j['initial_state_sha256']))
        hashes[str(source/'metadata.json')]=sha(source/'metadata.json'); hashes[str(source/'states.npz')]=sha(source/'states.npz')
    require(all(sum(r['success'] for r in episodes if r['policy']==p)==REFERENCE_SUCCESS[p]
                for p in REPEAT_POLICIES),'Success regression mismatch')
    if metric is None:
        phase3a=Path(pl['phase3a']); expert_root=Path(read(phase3a/'manifest.json')['expert_root'])
        _,metric=legacy(expert_root)
    labeled=[]
    for row in pr:
        a,b=row['job_a'],row['job_b']; require(a in paths and b in paths,'Missing pair trajectory')
        actual=float(metric.variation([paths[a],paths[b]])['world_pairs'][0])
        require(np.isfinite(actual) and math.isclose(actual,float(row['trajectory_rms_32_m']),rel_tol=1e-7,abs_tol=1e-9),
                f'Original RMS regression: {a}, {b}')
        labeled.append(dict(row))
    for kind,count in [('sft',100),('grpo',100),('cross',250)]:
        vv=[float(r['trajectory_rms_32_m']) for r in labeled if r['pair_type']==kind]
        require(len(vv)==count and abs(float(np.mean(vv))-REFERENCE[kind])<1e-6, f'Original {kind} RMS regression')
    original={r['episode_id']:r for r in sr}; require(len(original)==10,'Duplicate old state summary')
    return episodes,labeled,original,hashes,dict(run=run,diversity=diversity,plan=plan,paths=paths,validation=state)


def compute(run, diversity, output, *, metric=None, figures=True):
    episodes,pairs,old,hashes,ctx=inspect_inputs(run,diversity,metric=metric)
    state_rows,decomp=decompose(pairs,episodes)
    require(len(state_rows)==20 and len(decomp)==20,'State count')
    for eid in old:
        rr={r['policy']:r for r in state_rows if r['episode_id']==eid}; prior=old[eid]
        for policy in REPEAT_POLICIES:
            require(rr[policy]['n_success']==int(prior[policy+'_success']) and
                    math.isclose(rr[policy]['D_all_m'],float(prior[policy+'_pairwise_rms_mean_m']),rel_tol=1e-10,abs_tol=1e-12),
                    'Original state summary regression')
    policies,tasks,paired,extra=aggregate(state_rows,pairs)
    labeled=[dict(r,task_id=int(r['episode_id'].split('-t')[1].split('-')[0]),
                  trial_id=int(r['episode_id'].split('-i')[1]),outcome_category=r['outcome_category']) for r in pairs]
    output=output.resolve(); require(not output.exists(),'Output already exists'); output.mkdir(parents=True)
    csv_save(output/'pairwise_labeled.csv',labeled); csv_save(output/'per_state_outcome_diversity.csv',state_rows)
    csv_save(output/'per_task_summary.csv',tasks); csv_save(output/'policy_summary.csv',policies)
    csv_save(output/'decomposition.csv',decomp); csv_save(output/'paired_state_comparison.csv',paired)
    save(output/'manifest.json',dict(schema_version=1,source_run=str(ctx['run']),source_diversity=str(ctx['diversity']),
        plan=str(ctx['plan']),input_hashes=hashes,metric='legacy task_plots.variation world_pairs, 32-point normalized-index EEF RMS',
        analysis_unit='state; pairs descriptive only',code_sha256=sha(__file__),reference_rms_m=REFERENCE))
    if figures: draw_figures(output,ctx['paths'],episodes)
    write_report(output,policies,tasks,paired,extra)
    for name,digest in hashes.items(): require(sha(name)==digest,f'Input changed during analysis: {name}')
    seal(output,dict(status='PASS',episodes=100,states=10,pairs=450,within_policy_pairs=200,cross_policy_pairs=250,
                     source_validation=ctx['validation']['status'],common_SS_states=extra['common_SS_states']))
    return output


def _csv_float(row,key):
    value=row[key]; return None if value in ('','None','UNDEFINED') else float(value)


def _figsave(fig, path):
    fig.tight_layout()
    for ext in ('png','pdf'): fig.savefig(path.with_suffix('.'+ext),dpi=180,bbox_inches='tight')


def draw_figures(output, paths, episodes):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.size':10,'axes.titlesize':12,'axes.labelsize':10,'legend.fontsize':8,'pdf.fonttype':42})
    colors={'sft':'#3568a8','grpo':'#d16628'}; catcolors={'SS':'#2c7a64','SF':'#bf8641','FF':'#8b5a86'}
    state=rows(output/'per_state_outcome_diversity.csv',('episode_id','policy','D_all_m','success_rate'))
    dec=rows(output/'decomposition.csv',('episode_id','policy','D_all_m','C_SS_m','C_SF_m','C_FF_m'))
    paired=rows(output/'paired_state_comparison.csv',('episode_id','D_all_sft_m','D_all_grpo_m','success_rate_sft','success_rate_grpo'))
    policy=rows(output/'policy_summary.csv',('policy','success_rate'))
    task=rows(output/'per_task_summary.csv',('task_id','D_all_diff_mean_m'))
    by_state={(r['episode_id'],r['policy']):r for r in state}; by_dec={(r['episode_id'],r['policy']):r for r in dec}
    figdir=output/'figures'; figdir.mkdir(exist_ok=True)
    # 1. Same-state paired scatter.
    fig,ax=plt.subplots(figsize=(6.8,6)); x=np.array([float(r['D_all_sft_m']) for r in paired]); y=np.array([float(r['D_all_grpo_m']) for r in paired])
    delta=np.array([float(r['success_rate_grpo'])-float(r['success_rate_sft']) for r in paired]); lim=(min(x.min(),y.min())-.01,max(x.max(),y.max())+.01)
    for marker,t in [('o','03'),('s','08')]:
        ii=[i for i,r in enumerate(paired) if '-t'+t+'-' in r['episode_id']]
        ax.scatter(x[ii],y[ii],c=delta[ii],vmin=-1,vmax=1,cmap='coolwarm',marker=marker,edgecolors='black',s=90,label='t'+t)
    for i,r in enumerate(paired): ax.annotate(r['episode_id'].split('-t')[1],(x[i],y[i]),xytext=(4,3),textcoords='offset points',fontsize=7)
    ax.plot(lim,lim,'k--',lw=1); ax.set(xlim=lim,ylim=lim,xlabel='SFT D_all (m)',ylabel='GRPO D_all (m)',title='Same-state trajectory diversity')
    ax.legend(title='Task'); fig.colorbar(ax.collections[0],ax=ax,label='GRPO − SFT success rate',shrink=.8); _figsave(fig,figdir/'paired_rms'); plt.close(fig)
    # 2. Outcome-conditioned state means; no undefined zeros.
    fig,ax=plt.subplots(figsize=(8,5)); cats=list(CATEGORIES)
    for pi,p in enumerate(REPEAT_POLICIES):
        for ci,c in enumerate(cats):
            vals=[_csv_float(r,'D_'+c+'_m') for r in state if r['policy']==p]; vals=[v for v in vals if v is not None]
            xx=ci+(pi-.5)*.3
            if vals:
                ax.scatter(np.full(len(vals),xx),vals,color=colors[p],alpha=.45,s=28)
                ax.scatter([xx],[float(np.mean(vals))],color=colors[p],edgecolors='black',marker='D',s=75)
            ax.text(xx,1.02,f'n={len(vals)}',transform=ax.get_xaxis_transform(),ha='center',va='bottom',fontsize=8)
    ax.set_xticks(range(3),cats); ax.set(xlabel='Pair outcome',ylabel='State mean EEF RMS (m)',title='Outcome-conditioned diversity; dots = states, diamond = state mean')
    ax.legend(handles=[Line2D([0],[0],marker='o',ls='',color=colors[p],label=p.upper()) for p in REPEAT_POLICIES]); _figsave(fig,figdir/'outcome_diversity'); plt.close(fig)
    # 3. Pair-count-weighted decomposition.
    fig,ax=plt.subplots(figsize=(11,5)); order=sorted({r['episode_id'] for r in state}); xpos=[]; labels=[]
    for i,eid in enumerate(order):
        for pi,p in enumerate(REPEAT_POLICIES):
            pos=i*2.3+pi*.85; bottom=0
            for c in CATEGORIES:
                v=float(by_dec[(eid,p)]['C_'+c+'_m']); ax.bar(pos,v,bottom=bottom,width=.72,color=catcolors[c],label=c if i==0 and pi==0 else None); bottom+=v
            require(math.isclose(bottom,float(by_dec[(eid,p)]['D_all_m']),abs_tol=1e-9),'Figure decomposition mismatch')
            xpos.append(pos); labels.append(('S' if p=='sft' else 'G')+'\n'+eid.split('-t')[1])
    ax.set_xticks(xpos,labels,fontsize=7); ax.set(ylabel='Contribution to D_all (m)',title='SS / SF / FF contributions by state and policy'); ax.legend(); _figsave(fig,figdir/'decomposition'); plt.close(fig)
    # 4. Empirical success versus diversity. Paired states connected.
    fig,axes=plt.subplots(1,2,figsize=(11,4.8))
    for ai,field in enumerate(('D_all_m','D_SS_m')):
        ax=axes[ai]; excluded=0
        for r in paired:
            rr=[by_state[(r['episode_id'],p)] for p in REPEAT_POLICIES]
            xy=[(_csv_float(z,field),float(z['success_rate'])) for z in rr]
            if all(v is not None for v,_ in xy): ax.plot([v for v,_ in xy],[rate for _,rate in xy],color='#aaaaaa',lw=.8,zorder=1)
        for r in state:
            value=_csv_float(r,field)
            if value is None: excluded+=1; continue
            ax.scatter(value,float(r['success_rate']),color=colors[r['policy']],marker='o' if '-t03-' in r['episode_id'] else 's',edgecolors='black',s=55,zorder=2)
        ax.set(xlabel=field.replace('_m','')+' (m)',ylabel='Empirical success rate',ylim=(-.05,1.05),title=f'{field.replace("_m", "")}; excluded {excluded}/20')
    axes[0].legend(handles=[Line2D([0],[0],marker='o',ls='',color=colors[p],label=p.upper()) for p in REPEAT_POLICIES]); _figsave(fig,figdir/'success_vs_diversity'); plt.close(fig)
    # 5. Ten raw world-frame XYZ overlays with shared limits across policies.
    by_episode=collections.defaultdict(list)
    for r in episodes: by_episode[r['episode_id']].append(r)
    for eid,rr in sorted(by_episode.items()):
        fig=plt.figure(figsize=(12,5)); all_xyz=np.concatenate([paths[r['job_id']] for r in rr]); lo=all_xyz.min(axis=0); hi=all_xyz.max(axis=0); pad=np.maximum((hi-lo)*.08,.005)
        for pi,p in enumerate(REPEAT_POLICIES):
            ax=fig.add_subplot(1,2,pi+1,projection='3d')
            for r in sorted((r for r in rr if r['policy']==p),key=lambda r:r['policy_seed']):
                xyz=paths[r['job_id']]; ax.plot(*xyz.T,ls='-' if r['success'] else '--',color=colors[p],alpha=.75,label=f"{r['policy_seed']} {'S' if r['success'] else 'F'}")
                ax.scatter(*xyz[0],marker='o',color='black',s=12); ax.scatter(*xyz[-1],marker='x',color=colors[p],s=28)
            ax.set(xlim=(lo[0]-pad[0],hi[0]+pad[0]),ylim=(lo[1]-pad[1],hi[1]+pad[1]),zlim=(lo[2]-pad[2],hi[2]+pad[2]),
                   xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title=p.upper()); ax.legend(fontsize=6,loc='upper right')
        fig.suptitle(f'{eid}: raw world-frame EEF paths (circle=start, x=end)')
        _figsave(fig,figdir/('overlay_'+eid)); plt.close(fig)
    # 6. Compact dashboard from the same saved CSVs.
    fig,axes=plt.subplots(2,2,figsize=(11,8)); ax=axes[0,0]
    ax.bar([p['policy'].upper() for p in policy],[float(p['success_rate']) for p in policy],color=[colors[p['policy']] for p in policy]); ax.set(ylabel='Success rate',ylim=(0,1),title='Repeated rollout success')
    ax=axes[0,1]; ax.axhline(0,color='black',lw=.8); ax.bar([r['episode_id'].split('-t')[1] for r in paired],[float(r['D_all_diff_grpo_minus_sft_m']) for r in paired],color='#8166a2'); ax.tick_params(axis='x',rotation=45,labelsize=7); ax.set(ylabel='GRPO − SFT D_all (m)',title='Matched initial states')
    ax=axes[1,0]; cmeans={p:{c:float(np.mean([float(by_dec[(e,p)]['C_'+c+'_m']) for e in order])) for c in CATEGORIES} for p in REPEAT_POLICIES}
    for pi,p in enumerate(REPEAT_POLICIES):
        bottom=0
        for c in CATEGORIES: ax.bar(pi,cmeans[p][c],bottom=bottom,color=catcolors[c],label=c if pi==0 else None); bottom+=cmeans[p][c]
    ax.set_xticks([0,1],['SFT','GRPO']); ax.set(ylabel='Mean state contribution (m)',title='D_all decomposition'); ax.legend()
    ax=axes[1,1]; ax.axhline(0,color='black',lw=.8); ax.bar(['t'+str(r['task_id']).zfill(2) for r in task],[float(r['D_all_diff_mean_m']) for r in task],color='#8166a2')
    coverage={p['policy']:{c:p[f'D_{c}_valid_states'] for c in CATEGORIES} for p in rows(output/'policy_summary.csv',('policy','D_SS_valid_states'))}
    ax.set(ylabel='Mean paired ΔD_all (m)',title='Task difference and valid-state coverage'); ax.text(.98,.96,'Valid states: '+str(coverage),transform=ax.transAxes,ha='right',va='top',fontsize=8)
    fig.text(.5,.015,'Trajectory diversity is measured from repeated simulator rollouts, not token probability distributions.',ha='center',fontsize=9)
    _figsave(fig,figdir/'summary_dashboard'); plt.close(fig)
    return sorted(figdir.glob('*'))


def write_report(output,policies,tasks,paired,extra):
    lines=['# Outcome-conditioned trajectory diversity','',
        '100 validated rollouts; 10 initial states in selected LIBERO-10 tasks t03/t08; five stochastic seeds per policy and state. 450 original pairs (100 SFT, 100 GRPO, 250 cross-policy).','',
        'RMS is the original 32-point normalized-index world-frame EEF path distance. Matching normalized indices do not imply matching physical events. Pair values are descriptive; state is the comparison unit.','',
        '| Policy | Success | State mean D_all (m) | SS valid/10 | D_SS (m) | SF valid/10 | D_SF (m) | FF valid/10 | D_FF (m) |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    fmt=lambda x: 'UNDEFINED' if x is None else f'{x:.6f}'
    for p in policies:
        lines.append(f"| {p['policy'].upper()} | {p['n_success']}/{p['n_rollouts']} | {fmt(p['D_all_state_mean_m'])} | {p['D_SS_valid_states']}/10 | {fmt(p['D_SS_state_mean_m'])} | {p['D_SF_valid_states']}/10 | {fmt(p['D_SF_state_mean_m'])} | {p['D_FF_valid_states']}/10 | {fmt(p['D_FF_state_mean_m'])} |")
    diffs=[r['D_all_diff_grpo_minus_sft_m'] for r in paired]; q=quantiles(diffs)
    lines+=['',f"Matched-state ΔD_all (GRPO−SFT): mean {fmt(finite_mean(diffs))} m; median {fmt(q['median_m'])} m; IQR {fmt(q['p25_m'])}–{fmt(q['p75_m'])} m. GRPO higher in {sum(x>0 for x in diffs)} states, lower in {sum(x<0 for x in diffs)} states.",
        f"Common-state SS coverage: {extra['common_SS_states']}/10; SFT D_SS {fmt(extra['common_SS_sft_mean_m'])} m, GRPO D_SS {fmt(extra['common_SS_grpo_mean_m'])} m on these same states. All-state SS means have different coverage and are not directly comparable.",'',
        '| Task | States | Mean paired ΔD_all (m) | SFT success | GRPO success | Common SS states |','|---|---:|---:|---:|---:|---:|']
    for t in tasks: lines.append(f"| t{t['task_id']:02d} | {t['n_states']} | {fmt(t['D_all_diff_mean_m'])} | {t['sft_success_rate']:.3f} | {t['grpo_success_rate']:.3f} | {t['common_SS_states']} |")
    lines+=['','D_all equals the pair-count-weighted SS, SF and FF contributions in every state and policy. The stacked bars report these contributions; category means alone do not explain the overall change.','',
        'Each trajectory appears in multiple pairs. The 450 pair values are not independent samples. The selected two tasks and five seeds per state limit generalization. Success-rate correlations are descriptive, not causal. Distinct path overlays show observed path differences, not proof of distinct learned strategies.','',
        'Trajectory diversity is measured from repeated simulator rollouts, not token probability distributions.','']
    (output/'summary.md').write_text('\n'.join(lines))


def validate_output(output, *, check_inputs=True):
    output=output.resolve(); v=verify_seal(output); manifest=read(output/'manifest.json')
    require(v['episodes']==100 and v['states']==10 and v['pairs']==450,'Output counts changed')
    if check_inputs:
        for name,digest in manifest['input_hashes'].items(): require(sha(name)==digest,f'Input hash changed: {name}')
    ep=rows(output/'per_state_outcome_diversity.csv',('episode_id','policy','D_all_m'))
    pp=rows(output/'pairwise_labeled.csv',('episode_id','pair_type','job_a','job_b','outcome_category'))
    dec=rows(output/'decomposition.csv',('episode_id','policy','D_all_m','C_SS_m','C_SF_m','C_FF_m'))
    require((len(ep),len(pp),len(dec))==(20,450,20),'Output CSV counts')
    for r in dec:
        require(math.isclose(float(r['D_all_m']),sum(float(r['C_'+c+'_m']) for c in CATEGORIES),abs_tol=1e-10),'Saved decomposition mismatch')
    figs=list((output/'figures').glob('*')) if (output/'figures').is_dir() else []
    require(not figs or len(figs)==30,'Expected 15 PNG/PDF figure pairs')
    return dict(status='PASS',episodes=100,states=10,pairs=450,figures=len(figs),inputs_verified=check_inputs)


def regenerate_figures(output):
    output=output.resolve(); validate_output(output)
    manifest=read(output/'manifest.json'); run=Path(manifest['source_run']); pl=read(Path(manifest['plan'])/'manifest.json')
    jobs={j['job_id']:j for j in pl['jobs']}; episodes=[]; paths={}
    for jid,j in jobs.items():
        source=Path(read(run/'entries'/(jid+'.json'))['episode_path']); m=read(source/'metadata.json')
        with np.load(source/'states.npz',allow_pickle=False) as z: paths[jid]=z['robot0_eef_pos'].copy()
        episodes.append(dict(j,success=bool(m['success'])))
    # A complete temporary figure set is prepared before replacing the prior set.
    with tempfile.TemporaryDirectory(dir=output) as td:
        temp=Path(td)
        for filename in ('per_state_outcome_diversity.csv','decomposition.csv','paired_state_comparison.csv',
                         'policy_summary.csv','per_task_summary.csv'):
            shutil.copyfile(output/filename,temp/filename)
        draw_figures(temp,paths,episodes)
        new=sorted((temp/'figures').glob('*')); require(len(new)==30,'Incomplete regenerated figures')
        for p in new: os.replace(p,output/'figures'/p.name)
    details={k:v for k,v in read(output/'validation.json').items() if k!='output_hashes'}
    details['output_hashes']={str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*')) if p.is_file() and p.name!='validation.json'}
    fd,name=tempfile.mkstemp(prefix='.validation-',dir=output)
    try:
        with os.fdopen(fd,'w') as f: json.dump(details,f,indent=2,sort_keys=True,allow_nan=False); f.write('\n')
        os.replace(name,output/'validation.json')
    finally:
        if os.path.exists(name): os.unlink(name)
    return validate_output(output)


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='command',required=True)
    a=sub.add_parser('run'); a.add_argument('--run',type=Path,required=True); a.add_argument('--diversity',type=Path,required=True); a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('figures'); a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('validate'); a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('summary'); a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('audit'); a.add_argument('--run',type=Path,required=True); a.add_argument('--diversity',type=Path,required=True)
    args=p.parse_args(argv)
    if args.command=='run': print(json.dumps(dict(output=str(compute(args.run,args.diversity,args.output)),status='PASS'),indent=2))
    elif args.command=='figures': print(json.dumps(regenerate_figures(args.output),indent=2))
    elif args.command=='validate': print(json.dumps(validate_output(args.output),indent=2))
    elif args.command=='summary': validate_output(args.output); print((args.output/'summary.md').read_text())
    else:
        episodes,pairs,_,_,ctx=inspect_inputs(args.run,args.diversity)
        state,_=decompose(pairs,episodes)
        print(json.dumps(dict(status='PASS',episodes=len(episodes),pairs=len(pairs),states=len(state)//2,
                              source_validation=ctx['validation']['status'],schemas='PASS',rms_regression='PASS'),indent=2))

if __name__=='__main__': main()
