"""Publication exports for CPU-only Stage 2/3 analysis (PNG and vector PDF)."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from token_action_core import read,csv_rows,avg,require,verify_seal
from scripts.common.integrity import seal
from scripts.common.action_schema import GROUPS,DIMENSIONS

COLORS={'sft':'#28618a','grpo':'#bd5739','translation':'#327a9b','rotation':'#8c6bb1','gripper':'#ba7e33',
        'libero_goal':'#258d88','libero_spatial':'#7b65a9','libero_10':'#cc793c',
        'both_success':'#237f73','both_fail':'#a25062','gained':'#5d77b1','lost':'#d08d33'}
SUITES=('libero_goal','libero_spatial','libero_10')
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'axes.titleweight':'bold','axes.labelcolor':'#27313b','text.color':'#17232d','pdf.fonttype':42,'savefig.dpi':180})
FOOT='3,000 common observations • 300 source episodes • 30 tasks. Predicted slots; execution UNKNOWN.'


def savefig(fig,out,name,title,subtitle,footer=FOOT):
    fig.suptitle(title,x=.07,y=.97,ha='left',fontsize=16,fontweight='bold')
    fig.text(.07,.905,subtitle,fontsize=9,color='#586572')
    fig.text(.07,.045,footer,fontsize=8,color='#586572')
    fig.subplots_adjust(left=.08,right=.97,bottom=.19,top=.79,wspace=.37,hspace=.65)
    for ext in ('png','pdf'): fig.savefig(out/(name+'.'+ext),facecolor='white')
    plt.close(fig)


def val(summary,t,g,metric,origin='all'):
    return next(r for r in summary if r['temperature']==t and r['group']==g and r['origin']==origin and r['metric']==metric)


def filterrows(rows,t,g): return [r for r in rows if r['temperature']==t and r['group']==g]


def bar(ax,summary,t,metric,label):
    vals=[val(summary,t,g,metric)['mean'] for g in GROUPS]
    ax.bar(range(3),vals,color=[COLORS[g] for g in GROUPS],width=.57)
    ax.set_xticks(range(3),['Translation','Rotation','Gripper'],rotation=20,ha='right'); ax.set_ylabel(label)
    ax.axhline(0,color='#8e959b',lw=.7); ax.grid(axis='y',alpha=.12); ax.set_axisbelow(True)
    for i,y in enumerate(vals): ax.annotate(f'{y:.3f}',(i,y),xytext=(0,5 if y>=0 else -13),textcoords='offset points',ha='center',fontsize=9)
    ax.margins(y=.23)


def plot_stage2(run,out,t):
    data=run/'stage2/data'; summary=read(data/'group_summary.json'); suffix=f'T{t:g}'.replace('.','p')
    dims=[r for r in csv_rows(data/'dimension_summary.csv') if float(r['temperature'])==t]
    fig,ax=plt.subplots(figsize=(11,5))
    x=np.arange(7)
    for i,r in enumerate(dims): ax.plot([i-.12,i+.12],[float(r['entropy_sft']),float(r['entropy_grpo'])],color='#c0c7ce',lw=2)
    for pol,shift in [('sft',-.12),('grpo',.12)]: ax.scatter(x+shift,[float(r['entropy_'+pol]) for r in dims],s=75,marker='D',color=COLORS[pol],label=pol.upper(),zorder=3)
    ax.set_xticks(x,DIMENSIONS); ax.set_ylabel('Mean entropy (nats)'); ax.legend(frameon=False,ncol=2,loc='lower left'); ax.grid(axis='y',alpha=.15)
    for edge in (2.5,5.5): ax.axvline(edge,color='#b8c0c8',ls=':',lw=1)
    ax.set_ylim(bottom=0)
    savefig(fig,out,'S2-1_dimensions_'+suffix,'Where does token uncertainty change?',f'T = {t:g} • Equal source-episode means; each dimension averaged over 8 predicted chunks')
    slot=[r for r in csv_rows(data/'slot_summary.csv') if float(r['temperature'])==t]; matrix=np.empty((8,7))
    for r in slot: matrix[tuple(json.loads(r['index']))]=float(r['delta_entropy'])
    lim=max(abs(matrix).max(),1e-9); fig,ax=plt.subplots(figsize=(10,6.5))
    im=ax.imshow(matrix,cmap='RdBu_r',norm=TwoSlopeNorm(0,-lim,lim),aspect='auto')
    ax.set_xticks(range(7),DIMENSIONS); ax.set_yticks(range(8)); ax.set_ylabel('Predicted chunk index'); fig.colorbar(im,ax=ax,label='ΔH: GRPO − SFT (nats)',pad=.025)
    for c,d in np.ndindex(8,7): ax.text(d,c,f'{matrix[c,d]:.2f}',ha='center',va='center',fontsize=9,color='white' if abs(matrix[c,d])>lim*.58 else '#17232d')
    savefig(fig,out,'S2-2_slot_heatmap_'+suffix,'Entropy restructuring across all 56 slots',f'T = {t:g} • Blue: lower GRPO entropy; red: higher GRPO entropy • Equal source-episode means')
    fig,axes=plt.subplots(1,4,figsize=(14,5))
    for ax,k,label in zip(axes,('delta_entropy','greedy_disagreement','js','delta_top1'),('ΔH (nats)','Greedy token disagreement','JS divergence (nats)','Δ top-1 probability')): bar(ax,summary,t,k,label)
    savefig(fig,out,'S2-3_groups_'+suffix,'Distribution changes by action group',f'T = {t:g} • Metrics use separate axes • Slots → observation → equal source-episode mean')
    tasks=csv_rows(data/'task_summary.csv'); keys=[(s,i) for s in SUITES for i in range(10)]
    mat=np.array([[float(next(r['mean'] for r in tasks if float(r['temperature'])==t and r['suite']==s and int(r['task_id'])==i and r['group']==g and r['origin']=='all' and r['metric']=='delta_entropy')) for g in GROUPS] for s,i in keys])
    fig,ax=plt.subplots(figsize=(10,11)); lim=max(abs(mat).max(),1e-9)
    im=ax.imshow(mat,cmap='RdBu_r',norm=TwoSlopeNorm(0,-lim,lim),aspect='auto'); ax.set_xticks(range(3),['Translation','Rotation','Gripper'])
    ax.set_yticks(range(30),[s.replace('libero_','').replace('10','Long')+f' / {i:02d}' for s,i in keys],fontsize=8)
    for boundary in (9.5,19.5): ax.axhline(boundary,color='white',lw=3)
    for i,j in np.ndindex(mat.shape): ax.text(j,i,f'{mat[i,j]:.2f}',ha='center',va='center',fontsize=8,color='white' if abs(mat[i,j])>lim*.6 else '#17232d')
    fig.colorbar(im,ax=ax,label='Mean ΔH (nats)',pad=.03)
    savefig(fig,out,'S2-4_tasks_'+suffix,'Task-level entropy changes',f'T = {t:g} • 10 source episodes / task • Goal, Spatial and Long separated')
    # Chunk plot makes the axis distinct from the dimension analysis.
    chunks=[r for r in csv_rows(data/'chunk_summary.csv') if float(r['temperature'])==t]
    fig,ax=plt.subplots(figsize=(10,5))
    for pol in ('sft','grpo'): ax.plot(range(8),[float(r['entropy_'+pol]) for r in chunks],marker='o',color=COLORS[pol],label=pol.upper())
    ax.set_xlabel('Predicted chunk index'); ax.set_ylabel('Entropy (nats)'); ax.set_xticks(range(8)); ax.legend(frameon=False); ax.grid(alpha=.15)
    savefig(fig,out,'S2-S1_chunks_'+suffix,'Entropy along the predicted action chunk',f'T = {t:g} • All seven dimensions averaged within each chunk • Equal source-episode means')


def scatter_group(ax,rows,x,y,group,mode='suite'):
    valid=[r for r in rows if r[x] is not None and r[y] is not None]
    if mode=='outcome':
        for outcome in ('both_success','gained','both_fail','lost'):
            for origin,marker in [('sft','o'),('grpo','^')]:
                rr=[r for r in valid if r['outcome']==outcome and r['origin']==origin]
                ax.scatter([r[x] for r in rr],[r[y] for r in rr],s=9,alpha=.25,c=COLORS[outcome],marker=marker,rasterized=True)
    else:
        for s in SUITES:
            rr=[r for r in valid if r['suite']==s]
            ax.scatter([r[x] for r in rr],[r[y] for r in rr],s=10,alpha=.22,color=COLORS[s],label=s.replace('libero_','').replace('10','Long'),rasterized=True)
    ax.set_title(group.capitalize(),loc='left',fontsize=11); ax.grid(alpha=.12)


def plot_stage3(run,out,t):
    data=run/'stage3/data'; rows=read(data/'observations.json'); ep=read(data/'episodes.json'); summary=read(data/'group_summary.json'); s2=read(run/'stage2/data/group_summary.json'); suffix=f'T{t:g}'.replace('.','p')
    fig,axes=plt.subplots(1,3,figsize=(13,5.5))
    for ax,g in zip(axes,GROUPS):
        scatter_group(ax,filterrows(rows,t,g),'js','action_difference',g); ax.set_xlabel('JS divergence (nats)'); ax.set_ylabel('Decoded difference (command units)')
    axes[0].legend(frameon=False,fontsize=8)
    savefig(fig,out,'S3-1_JS_action_'+suffix,'Does distribution divergence imply a different command?',f'T = {t:g} • Each point: observation mean over predicted chunks • Group-specific y-axis scales')
    rr=filterrows(rows,t,'translation'); ee=filterrows(ep,t,'translation'); fig,ax=plt.subplots(figsize=(8,6.5))
    scatter_group(ax,rr,'magnitude_sft','magnitude_grpo','translation'); hi=max(max(r['magnitude_sft'],r['magnitude_grpo']) for r in rr)*1.05
    ax.plot([0,hi],[0,hi],ls='--',color='#8794a0',lw=1); ax.set(xlim=(0,hi),ylim=(0,hi),xlabel='SFT translation magnitude',ylabel='GRPO translation magnitude'); ax.set_aspect('equal'); ax.legend(frameon=False)
    delta=[r['magnitude_delta'] for r in ee]
    savefig(fig,out,'S3-2_magnitude_'+suffix,'Greedy translation magnitude at the same observation',f'Episode Δ magnitude: mean {np.mean(delta):+.4f}; median {np.median(delta):+.4f} command units • Actions independent of T')
    fig,axes=plt.subplots(1,2,figsize=(11,5)); values=[r['cosine'] for r in ee if r['cosine'] is not None]
    axes[0].hist(values,bins=np.linspace(-1,1,26),color=COLORS['translation'],alpha=.8); axes[0].set(xlabel='Episode mean cosine (defined chunks only)',ylabel='Source episodes'); axes[0].axvline(0,color='gray',ls=':')
    defined=np.mean([r['cosine_defined'] for r in ee]); axes[1].bar(['Defined','Near-zero / undefined'],[defined,1-defined],color=['#327a9b','#adb5bc']); axes[1].set_ylabel('Episode-weighted chunk fraction'); axes[1].set_ylim(0,1)
    eps=read(data/'validation.json')['near_zero_epsilon']
    savefig(fig,out,'S3-3_direction_'+suffix,'Translation direction, with near-zero commands separated',f'Cosine undefined when either decoded translation norm ≤ {eps:g} • No substitution of zero cosine')
    task=csv_rows(data/'task_summary.csv'); outcome=csv_rows(data/'outcome_summary.csv'); fig,axes=plt.subplots(1,3,figsize=(15,6))
    for s in SUITES:
        values=[float(next(r['mean'] for r in task if float(r['temperature'])==t and r['group']=='gripper' and r['metric']=='gripper_disagreement' and r['origin']=='all' and r['suite']==s and int(r['task_id'])==i)) for i in range(10)]
        axes[0].plot(range(10),values,marker='o',ms=4,label=s.replace('libero_','').replace('10','Long'),color=COLORS[s])
    axes[0].set(xlabel='Task index (within suite)',ylabel='Prepared open/close disagreement'); axes[0].legend(frameon=False,fontsize=8); axes[0].set_xticks(range(10))
    labels=['both_success','gained','both_fail','lost']; vals=[next(r for r in outcome if float(r['temperature'])==t and r['group']=='gripper' and r['metric']=='gripper_disagreement' and r['origin']=='all' and r['outcome']==o) for o in labels]
    axes[1].bar(range(4),[float(v['mean']) for v in vals],color=[COLORS[o] for o in labels]); axes[1].set_xticks(range(4),[o.replace('_','\n')+'\nn='+v['n_source_episodes'] for o,v in zip(labels,vals)],fontsize=8); axes[1].set_ylabel('Prepared disagreement'); axes[1].set_title('Nominal paired outcome',fontsize=11)
    axes[2].bar(['SFT-origin','GRPO-origin'],[val(summary,t,'gripper','gripper_disagreement',o)['mean'] for o in ('sft','grpo')],color=[COLORS['sft'],COLORS['grpo']]); axes[2].set_ylabel('Prepared disagreement'); axes[2].set_title('Source observation policy',fontsize=11)
    for ax in axes: ax.set_ylim(bottom=0); ax.grid(axis='y',alpha=.12); ax.set_axisbelow(True)
    savefig(fig,out,'S3-4_gripper_'+suffix,'Gripper command disagreement across strata','Native postprocessing: −sign(2 × decoded − 1) • Counterfactual commands, not measured gripper states')
    fig,axes=plt.subplots(1,3,figsize=(13,6))
    for ax,g in zip(axes,GROUPS):
        scatter_group(ax,filterrows(rows,t,g),'delta_entropy','action_difference',g,'outcome'); ax.axvline(0,ls=':',color='#8794a0'); ax.set_xlabel('ΔH: GRPO − SFT (nats)'); ax.set_ylabel('Decoded difference (command units)')
    from matplotlib.lines import Line2D
    handles=[Line2D([],[],color=COLORS[o],marker='o',ls='',label=o.replace('_',' ')) for o in ('both_success','gained','both_fail','lost')]
    handles += [Line2D([],[],color='#4d5660',marker=marker,ls='',label=label) for marker,label in [('o','SFT-origin'),('^','GRPO-origin')]]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.085),ncol=6,frameon=False,fontsize=8)
    savefig(fig,out,'S3-5_entropy_action_'+suffix,'Entropy change versus decoded command change',f'T = {t:g} • Observation means • Color: nominal paired outcome; marker: source policy',footer='Points are dependent within episodes/tasks; associations are descriptive and do not establish a causal success benefit.')
    fig,axes=plt.subplots(2,3,figsize=(14,9))
    bar(axes[0,0],s2,t,'delta_entropy','Δ entropy (nats)'); bar(axes[0,1],s2,t,'js','JS divergence (nats)'); bar(axes[0,2],summary,t,'action_difference','Decoded difference (command units)')
    axes[1,0].hist([r['cosine'] for r in ee if r['cosine'] is not None],bins=np.linspace(-1,1,21),color=COLORS['translation']); axes[1,0].set(xlabel='Episode mean translation cosine',ylabel='Source episodes'); axes[1,0].set_title(f'Undefined chunks: {(1-defined):.1%}',fontsize=10)
    axes[1,1].bar(['Token changes','Command flips'],[val(s2,t,'gripper','greedy_disagreement')['mean'],val(summary,t,'gripper','gripper_disagreement')['mean']],color=['#bba4ce',COLORS['gripper']]); axes[1,1].set_ylabel('Gripper disagreement fraction')
    for g in GROUPS:
        vals=[float(r['mean']) for r in task if float(r['temperature'])==t and r['metric']=='action_difference' and r['group']==g and r['origin']=='all']
        axes[1,2].scatter([list(GROUPS).index(g)]*len(vals),vals,s=18,alpha=.65,color=COLORS[g])
    axes[1,2].set_xticks(range(3),['Translation','Rotation','Gripper'],rotation=20); axes[1,2].set_ylabel('Task mean command difference'); axes[1,2].set_title('30 tasks per group',fontsize=10)
    savefig(fig,out,'S3-6_dashboard_'+suffix,'From action-token distributions to greedy commands',f'T = {t:g} • Episode-weighted means • Same observations for SFT and GRPO • Commands are counterfactual')
    fig,axes=plt.subplots(1,2,figsize=(11,5))
    grip=filterrows(ep,t,'gripper')
    for ax,key,title in zip(axes,('command_confidence','threshold_proximity'),('Prepared-command confidence','Distance from decoded threshold 0.5')):
        for pol in ('sft','grpo'): ax.hist([r[key+'_'+pol] for r in grip],bins=20,histtype='step',lw=2,color=COLORS[pol],label=pol.upper())
        ax.set_xlabel(title); ax.set_ylabel('Source episodes'); ax.legend(frameon=False)
    savefig(fig,out,'S3-S1_gripper_confidence_'+suffix,'Gripper confidence and threshold proximity',f'T = {t:g} • Confidence sums all token probabilities mapping to the greedy prepared command • Episode means')


def plot(run,stage):
    verify_seal(run/f'stage{stage}/data'); out=run/f'stage{stage}/figures'; require(not out.exists(),f'Figure output exists: {out}'); out.mkdir()
    for t in (1.,1.6):
        (plot_stage2 if stage==2 else plot_stage3)(run,out,t)
        print('FIGURES',stage,'T',t,flush=True)
    seal(out,dict(status='PASS',png=len(list(out.glob('*.png'))),pdf=len(list(out.glob('*.pdf'))),source_validation_sha256=__import__('scripts.common.integrity',fromlist=['sha']).sha(run/f'stage{stage}/data/validation.json')))
