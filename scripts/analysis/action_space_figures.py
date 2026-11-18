"""Figures A–G from sealed action-space tables, with no fitted regression lines."""
from pathlib import Path
import json
import re
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from token_action_core import read,csv_rows,clean
from scripts.common.integrity import require,sha,verify_seal,seal
from scripts.common.io import save
from scripts.common.action_schema import DIMENSIONS,GROUPS

COLORS={'translation':'#287e9f','rotation':'#8463ad','gripper':'#be8032','sft':'#28618a','grpo':'#be593c','libero_goal':'#258c86','libero_spatial':'#8269a5','libero_10':'#c67b40'}
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','pdf.fonttype':42,'savefig.dpi':170})
FOOT='3,000 observations • 300 source episodes • 30 tasks. Predicted slots; execution UNKNOWN. No physical-distance inference.'


def export(fig,out,name,title,subtitle,footer=FOOT,top=.81,bottom=.16):
    fig.suptitle(title,x=.065,y=.97,ha='left',fontsize=16,fontweight='bold')
    fig.text(.065,.91,subtitle,fontsize=9,color='#536170')
    fig.text(.065,.035,footer,fontsize=8,color='#536170')
    fig.subplots_adjust(left=.08,right=.97,top=top,bottom=bottom,wspace=.4,hspace=.65)
    for ext in ('png','pdf'): fig.savefig(out/(name+'.'+ext),facecolor='white')
    plt.close(fig)


def num(r,k): return float(r[k]) if r[k] not in ('',None) else np.nan


def selection(rows,t,g=None,**kw):
    return [r for r in rows if float(r['temperature'])==t and (g is None or r['group']==g) and all(str(r[k])==str(v) for k,v in kw.items())]


def value(rows,t,g,k,**kw):
    rr=selection(rows,t,g,metric=k,origin='all',**kw); require(len(rr)==1,f'Nonunique plot summary {t,g,k,kw}'); return num(rr[0],'mean')


def curve(ax,rr,mean=False,band=True,color='#bd593b',label=None):
    rr=sorted(rr,key=lambda r:int(r['bin'])); x=[num(r,'x_median') for r in rr]; y=[num(r,'mean' if mean else 'median') for r in rr]
    if band: ax.fill_between(x,[num(r,'p25') for r in rr],[num(r,'p75') for r in rr],color=color,alpha=.18,label='Within-bin IQR')
    ax.plot(x,y,'o-',color=color,lw=1.5,ms=4,label=label)
    low=np.array([num(r,'ci_low') for r in rr]); high=np.array([num(r,'ci_high') for r in rr]); good=np.isfinite(low)&np.isfinite(high)
    if good.any():
        # A percentile interval can exclude the point estimate; draw endpoints directly.
        ax.vlines(np.asarray(x)[good],low[good],high[good],color=color,lw=1.2)
    ax.grid(alpha=.12)


def plot(run):
    verify_seal(run/'data'); out=run/'figures'; require(not out.exists(),'Figures already exist'); out.mkdir()
    data=run/'data'; summary=csv_rows(data/'summary.csv'); bins=csv_rows(data/'quantile_bins.csv'); corr=csv_rows(data/'correlations.csv'); chunks=csv_rows(data/'chunk_summary.csv'); tasks=csv_rows(data/'task_summary.csv')
    obs=read(data/'observations.json'); dimensions=read(data/'dimension_episodes.json'); grip=read(data/'gripper_observations.json'); grip_ep=read(data/'gripper_episodes.json'); gs=csv_rows(data/'gripper_summary.csv'); gt=csv_rows(data/'gripper_task_summary.csv')
    # JSON summaries and plotted CSV means must agree, not merely have valid hashes.
    js=read(data/'summary.json')
    for row in summary:
        r=next(r for r in js if all(str(r[k])==str(row[k]) for k in ('temperature','group','scope','origin','metric')))
        require((r['mean'] is None and row['mean']=='') or abs(r['mean']-float(row['mean']))<1e-12,'Plot CSV/JSON value mismatch')
    inventory={}
    for t in (1.,1.6):
        suffix='T'+str(t).replace('.','p')
        fig,axes=plt.subplots(1,3,figsize=(13,5))
        for ax,g in zip(axes,GROUPS):
            rr=selection(bins,t,g,figure='A'); curve(ax,rr,mean=True,band=False,color=COLORS[g]); ax.set(xlabel='Scalar-slot JS quantile median (nats)',ylabel='Greedy token disagreement rate',ylim=(-.02,1.02)); ax.set_title(g.capitalize(),loc='left')
        export(fig,out,'A_greedy_disagreement_js_'+suffix,'Can token distributions change without changing the greedy token?',f'T={t:g} • 10 weighted JS quantile bins at scalar slots • Vertical intervals: 95% whole-task bootstrap')
        fig,axes=plt.subplots(1,3,figsize=(13,5.5))
        for ax,g in zip(axes,GROUPS):
            rr=selection(obs,t,g); ax.hexbin([r['js'] for r in rr],[r['w1'] for r in rr],gridsize=35,mincnt=1,bins='log',cmap='Greys',alpha=.65,rasterized=True)
            curve(ax,selection(bins,t,g,figure='B',scope='pooled'),color=COLORS[g],label='Quantile median')
            c=selection(corr,t,g,level='observation',scope='pooled',stratum='all',y='w1')[0]
            ax.set_title(g.capitalize()+f"\nρ={num(c,'spearman'):.2f} [{num(c,'ci_low'):.2f}, {num(c,'ci_high'):.2f}]",loc='left',fontsize=10)
            ax.set_xlabel('Observation mean token JS (nats)'); ax.set_ylabel('Mean scalar marginal W1\n(native command units)')
        axes[0].legend(frameon=False,fontsize=8)
        export(fig,out,'B_js_wasserstein_'+suffix,'Does token-space difference track action-value redistribution?',f'T={t:g} • Gray: log observation counts; line: median; band: IQR; whiskers and Spearman ρ: task-cluster 95% CI',footer='Group axes use their native command scales; do not compare physical magnitudes across groups. Gripper here is pre-threshold decoded space.')
        fig,axes=plt.subplots(1,3,figsize=(13,5.5)); cmap=plt.get_cmap('viridis')
        for ax,g in zip(axes,GROUPS):
            for c in range(8):
                rr=selection(bins,t,g,figure='C',scope='chunk:'+str(c)); ax.plot([num(r,'x_median') for r in rr],[num(r,'median') for r in rr],color=cmap(c/7),lw=.8,alpha=.55)
            curve(ax,selection(bins,t,g,figure='C',scope='pooled'),color='#b34e35',label='Pooled median / IQR')
            ax.set_title(g.capitalize(),loc='left'); ax.set_xlabel('JS quantile median (nats)'); ax.set_ylabel('Greedy decoded difference\n(vector L2; gripper absolute)')
        axes[0].legend(frameon=False,fontsize=8)
        export(fig,out,'C_js_greedy_action_'+suffix,'Does chunk pooling hide the JS–greedy-action relationship?',f'T={t:g} • Thin colored lines: chunks 0–7; thick line: pooled observation median • Pooled whiskers: task-cluster CI')
        fig,axes=plt.subplots(5,3,figsize=(13,14))
        metrics=[('js','Mean token JS (nats)'),('w1','Mean scalar W1 (command)'),('greedy_disagreement','Token disagreement'),('greedy_action_l2','Greedy difference (command)'),('delta_variance','Δ marginal variance (command²)')]
        for col,g in enumerate(GROUPS):
            for ri,(k,label) in enumerate(metrics):
                ax=axes[ri,col]; vals=[value(chunks,t,g,k,scope='chunk:'+str(c)) for c in range(8)]; ax.plot(range(8),vals,'o-',color=COLORS[g],ms=4); ax.set_xticks(range(8)); ax.set_ylabel(label); ax.grid(alpha=.15)
                if ri==0: ax.set_title(g.capitalize(),loc='left')
                if ri==4: ax.set_xlabel('Predicted chunk index')
                if k=='delta_variance': ax.axhline(0,color='gray',lw=.7)
        export(fig,out,'D_chunkwise_'+suffix,'How do distribution changes vary through the predicted chunk?',f'T={t:g} • Equal source-episode means • Each metric and action group has its own axis',top=.84,bottom=.11)
        fig,axes=plt.subplots(4,7,figsize=(18,10))
        for col,d in enumerate(DIMENSIONS):
            rr=selection(dimensions,t,d)
            for ri,(k,label) in enumerate([('delta_mu','Signed Δ mean'),('abs_delta_mu','Absolute Δ mean'),('delta_variance','Δ variance'),('w1','Scalar W1')]):
                ax=axes[ri,col]; vals=np.array([r[k] for r in rr]); ax.boxplot(vals,widths=.5,showfliers=False,patch_artist=True,boxprops=dict(facecolor='#b8cbd8'),medianprops=dict(color='#263c50')); ax.scatter([1],[vals.mean()],marker='D',s=22,color='#bb5537',zorder=3); ax.set_xticks([]); ax.axhline(0,color='gray',lw=.6); ax.tick_params(axis='y',labelsize=8); ax.ticklabel_format(axis='y',style='sci',scilimits=(-2,2))
                if col==0: ax.set_ylabel(label)
                if ri==0: ax.set_title(d)
        export(fig,out,'E_action_mean_variance_'+suffix,'How does probability concentration change in decoded value space?',f'T={t:g} • Source-episode distributions; box=IQR, line=median, diamond=mean; whiskers=1.5 IQR • Independent native scales',footer='Signed shifts can cancel: |Δmean| is computed per slot before averaging. Scalar W1 and variance are not cross-dimension physical distances.',top=.82,bottom=.12)
        fig,axes=plt.subplots(2,2,figsize=(12,9)); rr=selection(grip,t); ee=selection(grip_ep,t)
        for ax,key,label,lim in [(axes[0,0],'p_open','P(open)',1),(axes[0,1],'binary_entropy','Binary command entropy (nats)',np.log(2))]:
            xx=[r[key+'_sft'] for r in rr if r[key+'_sft'] is not None and r[key+'_grpo'] is not None]; yy=[r[key+'_grpo'] for r in rr if r[key+'_sft'] is not None and r[key+'_grpo'] is not None]
            ax.hexbin(xx,yy,gridsize=30,extent=(0,lim,0,lim),mincnt=1,bins='log',cmap='Blues',rasterized=True); ax.plot([0,lim],[0,lim],'--',color='#ad654b'); ax.set(xlim=(0,lim),ylim=(0,lim),xlabel='SFT '+label,ylabel='GRPO '+label); ax.set_aspect('equal')
        ax=axes[1,0]
        for origin,marker in [('sft','o'),('grpo','^')]:
            ss=[r for r in ee if r['origin']==origin and r['delta_binary_entropy'] is not None]; ax.scatter([r['delta_token_entropy'] for r in ss],[r['delta_binary_entropy'] for r in ss],s=14,alpha=.5,marker=marker,color=COLORS[origin],label=origin.upper()+' origin')
        ax.axhline(0,color='gray',lw=.7); ax.axvline(0,color='gray',lw=.7); ax.set(xlabel='Episode Δ token entropy (nats)',ylabel='Episode Δ binary entropy (nats)'); ax.legend(frameon=False,fontsize=8)
        ax=axes[1,1]; tr=selection(gt,t,'gripper',metric='delta_p_open',origin='all'); tr=sorted(tr,key=lambda r:(['libero_goal','libero_spatial','libero_10'].index(r['suite']),int(r['task_id'])))
        ax.bar(range(30),[num(r,'mean') for r in tr],color=[COLORS[r['suite']] for r in tr]); ax.axhline(0,color='gray',lw=.7); ax.set_xticks([4.5,14.5,24.5],['Goal 0–9','Spatial 0–9','Long 0–9']); ax.set_ylabel('Task mean Δ P(open)'); ax.set_ylim(-1,1)
        export(fig,out,'F_gripper_binary_'+suffix,'Does more token uncertainty imply more gripper-command uncertainty?',f'T={t:g} • Native −sign(2 × decoded − 1) • Gray/blue densities: observations; entropy-change points: source episodes',top=.82,bottom=.13)
        fig,axes=plt.subplots(2,4,figsize=(16,9))
        def bars(ax,key,title):
            vals=[value(summary,t,g,key,scope='predicted') for g in GROUPS]; ax.bar(range(3),vals,color=[COLORS[g] for g in GROUPS]); ax.set_xticks(range(3),['Trans.','Rot.','Grip.']); ax.set_title(title,fontsize=10,loc='left'); ax.grid(axis='y',alpha=.1); ax.set_axisbelow(True)
        bars(axes[0,0],'js','Token distribution difference\nJS (nats)'); bars(axes[0,1],'w1_normalized','Action-space distribution difference\nMean normalized-bin W1'); bars(axes[0,2],'greedy_disagreement','Greedy token disagreement\nFraction of scalar slots')
        for ax,key,title in [(axes[0,3],'greedy_action_l2','Greedy action difference'),(axes[1,0],'delta_variance','Action-space variance change')]:
            ax.axis('off'); ax.set_title(title,loc='left',fontsize=10)
            for i,g in enumerate(GROUPS): ax.text(0,.8-i*.24,f'{g.capitalize()}: {value(summary,t,g,key,scope="predicted"):.5f}',color=COLORS[g],fontsize=11)
            ax.text(0,.03,'Native command scales differ;\nvalues are not physical comparisons.',fontsize=8,color='#5b6874')
        ax=axes[1,1]; vals=[value(gs,t,'gripper','binary_entropy_'+p,scope='predicted') for p in ('sft','grpo')]; ax.plot([0,1],vals,'D-',color=COLORS['gripper']); ax.set_xticks([0,1],['SFT','GRPO']); ax.set_ylabel('Binary entropy (nats)'); ax.set_title('Gripper command uncertainty',fontsize=10,loc='left'); ax.set_ylim(bottom=0)
        ax=axes[1,2]
        for gi,g in enumerate(GROUPS):
            tr=selection(tasks,t,g,metric='w1_normalized',origin='all'); ax.scatter([gi]*len(tr),[num(r,'mean') for r in tr],color=COLORS[g],s=20,alpha=.5)
        ax.set_xticks(range(3),['Trans.','Rot.','Grip.']); ax.set_ylabel('Task mean normalized-bin W1'); ax.set_title('Task heterogeneity (30 tasks)',fontsize=10,loc='left')
        ax=axes[1,3]
        for gi,g in enumerate(GROUPS):
            r=selection(corr,t,g,level='episode',scope='pooled',stratum='all',y='w1')[0]; ax.scatter([num(r,'spearman')],[gi],color=COLORS[g]); ax.hlines(gi,num(r,'ci_low'),num(r,'ci_high'),color=COLORS[g])
        ax.axvline(0,color='gray',ls=':'); ax.set_xlim(-1,1); ax.set_yticks(range(3),['Trans.','Rot.','Grip.']); ax.set_xlabel('Spearman ρ (95% task CI)'); ax.set_title('Episode JS versus marginal W1',fontsize=10,loc='left')
        export(fig,out,'G_dashboard_'+suffix,'Token difference, action-space redistribution, and greedy choice',f'T={t:g} • Equal source-episode means • W1 is a scalar marginal measure; no joint 3-D control distribution assumed',top=.81,bottom=.15)
        inventory[str(t)]={g:{k:value(summary,t,g,k,scope='predicted') for k in ('js','w1','w1_normalized','greedy_disagreement','greedy_action_l2','delta_variance')} for g in GROUPS}
        print('FIGURES A–G',t,flush=True)
    save(out/'plotted_values.json',clean(inventory))
    seal(out,dict(status='PASS',png=len(list(out.glob('*.png'))),pdf=len(list(out.glob('*.pdf'))),csv_json_consistency='PASS',data_validation_sha256=sha(data/'validation.json')))


def plot_cases(out,rows):
    for r in rows:
        with np.load(out/r['case_file']) as z: x=z['support']; p=z['probability']; greedy=z['greedy']; expected=z['expected']
        require(np.allclose(p.sum(-1),1,atol=1e-12),'Case probability mass')
        fig,ax=plt.subplots(figsize=(11,5))
        for j,pol in enumerate(('sft','grpo')):
            ax.plot(x,p[j],color=COLORS[pol],lw=1.5,label=pol.upper()+' mass'); ax.axvline(greedy[j],color=COLORS[pol],ls='--',lw=1,label=pol.upper()+' greedy'); ax.axvline(expected[j],color=COLORS[pol],ls=':',lw=1.5,label=pol.upper()+' expected')
        ax.set_xlabel('Decoded scalar action value (native command units)'); ax.set_ylabel('Probability mass'); ax.legend(ncol=3,fontsize=8,frameon=False); ax.grid(alpha=.1)
        export(fig,out,Path(r['case_file']).stem,r['rule'].replace('_',' ').capitalize(),f"T={r['temperature']:g} • {r['observation_id']} • chunk {r['chunk']} / {r['dimension']} • {r['outcome']}",footer=f"JS={r['js']:.4f} nats; W1={r['w1']:.5f}; greedy difference={r['greedy_abs']:.5f}. Rule-selected example, not a general result.")


def write_report(run):
    m=read(run/'manifest.json'); data=run/'data'; s=csv_rows(data/'summary.csv'); g=csv_rows(data/'gripper_summary.csv'); corr=csv_rows(data/'correlations.csv'); cond=csv_rows(data/'conditional_summary.csv'); chunks=csv_rows(data/'chunk_summary.csv')
    lines=['# Stage 3 extension: action-space probability distributions','',
        'Scope: 3000 common observations, 300 source episodes, 30 nominal tasks; T=1.0 and 1.6 are offline temperatures of the same saved logits. Predicted action slots; execution UNKNOWN.',
        '', '## Source audit','',
        'Reused native action_schema mapping/decoder, saved-record loader, episode aggregator, distribution_metrics and common integrity seals. Existing Stage 1/2/3 files were not modified. The pinned native source chain and Q99 decoding match all 6000 saved records. See audit/source_audit.md and mapping/action_bins.csv.',
        'Support is reverse-ordered by token ID, with two token IDs sharing the highest decoded value. Exact scalar W1 sorts support and integrates absolute CDF differences. It accounts for distances between action values; JS does not. Gripper binary mass uses the native postprocessing, not an assumed zero threshold.',
        'Input Stage 2/3 root seal: PASS. Stage 1 bank/workers and numerical files: PASS. Copied Stage 1 original figures missing: '+str(len(m['stage1_copy_status']['missing_figures']))+'. Full copied Stage 1 analysis remains '+m['stage1_copy_status']['status']+'.',
        '', '## Aggregation and units','',m['weighting'],
        f"Bootstrap: seed {m['bootstrap_seed']}, {m['bootstrap_iterations']} whole-task draws within suite; 95% percentile intervals. Quantile edges fixed before resampling. Binned intervals are pointwise, not simultaneous bands. IQR describes within-bin dispersion and is not a confidence interval.",
        'W1 is scalar marginal transport. Group W1 averages three native scalar marginals (one for gripper), not joint-vector optimal transport. Cross-dimension physical scale/frame is unknown; native W1 cannot be ranked as physical motion across groups. Normalized-bin W1 is separately saved and used for a common-scale dashboard. Rotation quantities are command values, not radians or SO(3) distances.',
        '', '## Action-space metrics','', '| T | Group | Token JS | Native marginal W1 | Normalized W1 | Δ variance | Greedy L2 / absolute |','|---|---|---:|---:|---:|---:|---:|']
    for t in (1.,1.6):
        for group in GROUPS: lines.append('| '+str(t)+' | '+group+' | '+' | '.join(f'{value(s,t,group,k,scope="predicted"):.6f}' for k in ('js','w1','w1_normalized','delta_variance','greedy_action_l2'))+' |')
    lines+=['','### Expected-action shifts','', '| T | Group | Signed Δ mean | Absolute Δ mean | Δ absolute expected command |','|---|---|---:|---:|---:|']
    for t in (1.,1.6):
        for group in GROUPS: lines.append('| '+str(t)+' | '+group+' | '+' | '.join(f'{value(s,t,group,k,scope="predicted"):.6f}' for k in ('delta_mu','abs_delta_mu','delta_abs_mu'))+' |')
    lines+=['','These expected-command measures are scalar marginal quantities. Δ absolute expected command is |μ_GRPO|−|μ_SFT| averaged within observation/episode; it is not expected vector norm or measured motion.']
    lines+=['','### Same greedy token, different distribution','', '| T | Dimension | Same-argmax slots | Mean W1 | Mean absolute expected-action shift | Mean Δ variance |','|---|---|---:|---:|---:|---:|']
    for r in cond:
        if r['condition']=='same_argmax': lines.append('| '+' | '.join(r[k] for k in ('temperature','dimension','slot_count'))+' | '+' | '.join(f'{num(r,k):.6f}' for k in ('mean_w1','mean_abs_delta_mu','mean_delta_variance'))+' |')
    lines+=['','A retained greedy token does not imply an unchanged sampling marginal. The mean, variance, and W1 above use the entire native token support; no new samples or rollouts were generated.',
        '', '### Entropy decrease versus variance change','', '| T | Dimension | P(variance increases given entropy decreases) |','|---|---|---:|']
    for r in cond:
        if r['condition']=='entropy_decrease': lines.append(f"| {r['temperature']} | {r['dimension']} | {num(r,'fraction_variance_increase'):.2%} |")
    lines+=['','## Correlations and aggregation effects','', '| T | Group | Level | JS versus | Spearman ρ | 95% task interval |','|---|---|---|---|---:|---|']
    for r in corr:
        if r['scope']=='pooled' and r['stratum']=='all': lines.append('| '+' | '.join(r[k] for k in ('temperature','group','level','y'))+f" | {num(r,'spearman'):.3f} | [{num(r,'ci_low'):.3f}, {num(r,'ci_high'):.3f}] |")
    lines+=['','Chunk-level correlations are descriptive (no bootstrap interval), as are origin/outcome-stratified correlations; pooled observation, episode and task intervals resample whole tasks. No independent-slot p-values or fitted regression lines are used.', '', '| T | Group | Chunk JS–greedy ρ range | Pooled observation ρ |','|---|---|---|---:|']
    for t in (1.,1.6):
        for group in GROUPS:
            cc=selection(corr,t,group,level='observation',stratum='all',y='greedy_action_l2'); vals=[num(r,'spearman') for r in cc if r['scope'].startswith('chunk:')]; pooled=next(r for r in cc if r['scope']=='pooled')
            lines.append(f"| {t} | {group} | [{min(vals):.3f}, {max(vals):.3f}] | {num(pooled,'spearman'):.3f} |")
    lines+=['','These scales test aggregation sensitivity; they do not identify a causal mechanism. A weak or sign-changing association is reported as such. Temperature can change JS/W1 associations even though the stored greedy actions are unchanged.',
        '', '## Binary gripper distribution','', '| T | Token ΔH | Binary ΔH | ΔP(open) | Threshold mass SFT | Threshold mass GRPO |','|---|---:|---:|---:|---:|---:|']
    for t in (1.,1.6): lines.append('| '+str(t)+' | '+' | '.join(f'{value(g,t,"gripper",k,scope="predicted"):.6f}' for k in ('delta_token_entropy','delta_binary_entropy','delta_p_open','threshold_mass_sft','threshold_mass_grpo'))+' |')
    lines += ['',f"Threshold band: |decoded−0.5| ≤ {m['threshold_band']}. Open/close/other masses are explicit. Binary entropy is unavailable if other mass is nonzero; no implicit renormalization. Native-decoded greedy command and marginal modal command disagreements are separate columns. Neither is an observed gripper state.",
        '', '## Figure guide and interpretation','',
        '- **A:** scalar-slot JS quantiles versus greedy-token disagreement, with whole-task intervals. Tests whether a changed distribution retains its argmax; rates are conditional on the bin and equal-source-episode baseline weights.',
        '- **B:** observation density, median marginal W1, IQR and task-cluster uncertainty. Spearman estimates above quantify monotonic association without a linear model. Native group scales differ.',
        '- **C:** pooled observation medians/IQR and separate chunk curves. Pooling can change the apparent association; chunk ranges above quantify how much. Chunk curves alone have no uncertainty bands.',
        '- **D:** chunk means for JS, W1, disagreement, greedy difference and variance change, each on its own group/metric axis. These are predicted, not verified executed slots.',
        '- **E:** signed and absolute expected-action shifts, variance shifts and W1 by dimension. Boxes summarize source-episode values. Signed cancellation and scale differences prevent interpreting a near-zero overall mean as no redistribution.',
        '- **F:** binary gripper probabilities, binary entropy, token/binary entropy changes, and task-wise open-mass changes. A token-entropy increase need not imply a binary-entropy increase; the table above provides the measured direction.',
        '- **G:** separates token distribution difference, action-space distribution difference, and greedy action difference. Task heterogeneity and cluster intervals remain visible. Normalized W1 is used only where common-scale comparisons are shown.',
        '', '## Representative cases','',
        'Five fixed rules per temperature use within-dimension 10th/90th quantile intersections or positive token/negative binary entropy changes; ties use observation/chunk/dimension order. Missing categories are reported as NO_QUALIFYING_CASE. Probability plots collapse duplicate decoded support values by summing their masses. Selection is explanatory, not representative estimation or evidence of successful behavior. See cases/selection_rules.json and representative_cases.csv.',
        '', '## Limits and next use','',
        'These are conditional single-slot marginals from saved native parallel logits. They do not identify cross-slot dependence or a calibrated joint trajectory probability. To study successful-trajectory probability later, predefine reference trajectories, map actions to native bins (including aliases and gripper classes), specify conditioning/state provenance, and report action-marginal log mass separately from any assumed factorization. The present outcomes are nominal paired labels, not a causal demonstration that RLVR increases successful-action probability.',
        '', '## Validation','',
        '- Input numerical/native decoding and previous Stage 1/2/3 metrics: PASS (data/regression.json).',
        '- Exact 1D W1 is cross-tested against SciPy, including unordered/duplicate support.',
        '- Figure CSV/JSON consistency and figure/data/case seals: PASS.',
        '- No GPU inference, simulator rollout, training, or upstream modification.',
        '- Physical displacement / executed-slot inference: NOT RUN (UNKNOWN linkage).',
        '- CPU regression execution: '+re.search(r'Ran \d+ tests in [0-9.]+s', (run/'regression_tests.txt').read_text())[0]+'; '+re.search(r'^OK.*$',(run/'regression_tests.txt').read_text(),re.M)[0]+'.',
        '', '## Files','']
    for p in sorted(run.glob('figures/*.png'))+sorted(run.glob('cases/*.png')): lines.append('- ['+p.stem+']('+str(p.relative_to(run))+') — PDF also saved')
    (run/'summary.md').write_text('\n'.join(lines)+'\n')
