"""Reference-style three-panel view of sealed Stage 1 distribution results.

This script reads saved episode metrics and adds independently sealed figures under
analysis/figures/reference_style. It does not alter the original Stage 1 outputs.
"""
from __future__ import annotations
import argparse
import collections
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT))
from scripts.common.integrity import require, seal, sha, verify_seal
from scripts.common.io import read, save

SUITES=('libero_goal','libero_spatial','libero_10')
TEMPERATURES=(1.0,1.6)


def load_data(analysis):
    verified=verify_seal(analysis)
    metrics=read(analysis/'metrics.json')
    require(verified['source_episodes']==300 and verified['observations']==3000,
            'Unexpected sealed Stage 1 counts')
    require(metrics['processed_inputs_equal']==3000 and metrics['processed_inputs_different']==0,
            'Paired model inputs differ')
    with (analysis/'episodes.csv').open(newline='') as f:
        episode_rows=list(csv.DictReader(f))
    require(len(episode_rows)==600,'Expected 300 episodes at each of two temperatures')
    output={}
    for temperature in TEMPERATURES:
        rr=[r for r in episode_rows if float(r['temperature'])==temperature]
        require(len(rr)==300 and len({(r['origin'],r['episode_id']) for r in rr})==300,
                'Incomplete episode data')
        groups=collections.defaultdict(list)
        for r in rr:
            require(r['suite'] in SUITES,'Unexpected suite')
            groups[(r['suite'],int(r['task_id']))].append(r)
        require(len(groups)==30 and all(len(v)==10 for v in groups.values()),
                'Expected ten source episodes per task')
        fields=('entropy_sft','entropy_grpo','top1_sft','top1_grpo','js')
        tasks=[]
        for (suite,task),episodes in sorted(groups.items(),key=lambda x:(SUITES.index(x[0][0]),x[0][1])):
            tasks.append(dict(suite=suite,task_id=task,**{
                key:float(np.mean([float(e[key]) for e in episodes])) for key in fields}))
        overall={key:float(np.mean([float(e[key]) for e in rr])) for key in fields}
        for key in fields:
            require(math.isclose(overall[key],float(np.mean([r[key] for r in tasks])),rel_tol=1e-12),
                    'Task/episode weighting mismatch')
        published=metrics['overall'][str(temperature)]
        for key,metric_key in [('entropy_sft','mean_entropy_sft'),('entropy_grpo','mean_entropy_grpo'),
                               ('js','mean_js')]:
            require(math.isclose(overall[key],published[metric_key],rel_tol=1e-10,abs_tol=1e-12),
                    'Published mean regression mismatch')
        require(math.isclose(overall['top1_grpo']-overall['top1_sft'],
                             published['mean_delta_top1'],rel_tol=1e-10,abs_tol=1e-12),
                'Published top-1 change mismatch')
        output[temperature]=dict(tasks=tasks,overall=overall,episodes=300,observations=3000)
    return output


def make_plot(data,temperature,folder):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':11,
                         'axes.labelsize':9,'pdf.fonttype':42,'svg.fonttype':'none'})
    blue='#336c9c'; orange='#c66b47'; violet='#8163a5'; gray='#b8bfc6'
    tasks=data['tasks']; means=data['overall']
    fig,axes=plt.subplots(1,3,figsize=(13.8,5.4),gridspec_kw={'wspace':.33})
    fig.patch.set_facecolor('white')
    for ax in axes:
        ax.set_facecolor('white')
        ax.spines[['top','right']].set_visible(False)
        ax.spines[['left','bottom']].set_color('#9aa3aa')
        ax.grid(axis='y',color='#e9edf0',linewidth=.8)
        ax.set_axisbelow(True)
        ax.tick_params(axis='both',labelsize=9)
    panels=(('entropy_sft','entropy_grpo','Entropy','Entropy (nats)'),
            ('top1_sft','top1_grpo','Top-1 token probability','Top-1 probability'))
    for ax,(left,right,title,ylabel) in zip(axes[:2],panels):
        for row in tasks:
            ax.plot((0,1),(row[left],row[right]),color=gray,linewidth=.9,alpha=.65,zorder=1)
            ax.scatter((0,1),(row[left],row[right]),c=(blue,orange),s=13,alpha=.72,
                       edgecolors='white',linewidths=.3,zorder=2)
        ax.plot((0,1),(means[left],means[right]),color='#444b53',linewidth=2.1,zorder=3)
        ax.scatter((0,1),(means[left],means[right]),marker='D',s=100,c=(blue,orange),
                   edgecolors='white',linewidths=1,zorder=4)
        vals=np.array([r[key] for r in tasks for key in (left,right)])
        spread=max(float(np.ptp(vals)),.1)
        ax.set(xlim=(-.23,1.23),ylim=(float(vals.min())-.12*spread,float(vals.max())+.18*spread),
               xticks=(0,1),xticklabels=('SFT','GRPO'),ylabel=ylabel,title=title)
        ax.annotate(f'{means[left]:.2f}',(0,means[left]),xytext=(-8,12),textcoords='offset points',
                    ha='right',color=blue,fontsize=10,fontweight='bold')
        ax.annotate(f'{means[right]:.2f}',(1,means[right]),xytext=(8,12),textcoords='offset points',
                    ha='left',color=orange,fontsize=10,fontweight='bold')
        change=means[right]-means[left]
        ax.text(.97,.96,f'Δ = {change:+.2f}'+(' nats' if left.startswith('entropy') else ''),
                transform=ax.transAxes,ha='right',va='top',color='#3f4952',fontsize=9)
    ax=axes[2]
    # One small point per task. Fixed horizontal offsets identify the three suites.
    markers={'libero_goal':'o','libero_spatial':'s','libero_10':'^'}
    offsets={'libero_goal':-.085,'libero_spatial':0,'libero_10':.085}
    for suite in SUITES:
        group=[r for r in tasks if r['suite']==suite]
        x=np.array([1+offsets[suite]+(i-4.5)*.005 for i in range(10)])
        ax.scatter(x,[r['js'] for r in group],color=violet,marker=markers[suite],s=27,
                   alpha=.75,edgecolors='white',linewidths=.4,zorder=2)
    ax.scatter([1.19],[means['js']],marker='D',s=115,color=violet,
               edgecolors='white',linewidths=1,zorder=4)
    ax.annotate(f"{means['js']:.3f}",(1.19,means['js']),xytext=(7,7),textcoords='offset points',
                color=violet,fontsize=10,fontweight='bold')
    jsvals=np.array([r['js'] for r in tasks]); margin=max(float(np.ptp(jsvals))*.15,.012)
    ax.set(xlim=(.77,1.42),ylim=(max(0,float(jsvals.min())-margin),float(jsvals.max())+margin),
           xticks=(1,),xticklabels=('SFT vs GRPO',),ylabel='JS divergence (nats)',
           title='Distribution difference')
    ax.legend(handles=[Line2D([0],[0],marker=markers[s],color='none',markerfacecolor=violet,
                              markeredgecolor='white',markersize=6,label=label)
                       for s,label in zip(SUITES,('Goal','Spatial','Long'))],
              loc='upper right',frameon=False,fontsize=8)
    fig.suptitle('Same observations, two model distributions',x=.055,y=.98,ha='left',
                 fontsize=16,fontweight='bold')
    fig.text(.055,.91,f'T={temperature:g}  |  300 source episodes  |  3,000 paired observations  |  30 mapped tasks',
             fontsize=9,color='#59646d')
    fig.text(.055,.075,'Each thin line / small dot = one task mean (10 source episodes). Large diamonds = equal-episode means.',
             fontsize=8.2,color='#55616a')
    fig.text(.055,.047,'Saved-logit probabilities; lower entropy or higher top-1 alone does not establish success. JS compares the two distributions.',
             fontsize=8.2,color='#55616a')
    fig.subplots_adjust(left=.07,right=.965,bottom=.20,top=.80)
    stem='same_observations_T'+str(temperature).replace('.','p')
    for ext in ('png','pdf'):
        fig.savefig(folder/(stem+'.'+ext),dpi=200,facecolor='white')
    plt.close(fig)
    return stem


def generate(analysis,output):
    analysis=analysis.resolve(); output=output.resolve()
    require(not output.exists() or (not (output/'validation.json').exists()
            and not (output/'manifest.json').exists()),f'Completed output exists: {output}')
    require(output.parent==analysis/'figures','Output must be a new subdirectory of analysis/figures')
    data=load_data(analysis)
    output.mkdir(parents=True,exist_ok=True)
    stems=[make_plot(data[t],t,output) for t in TEMPERATURES]
    source_hashes={name:sha(analysis/name) for name in ('validation.json','episodes.csv','tasks.csv','metrics.json')}
    save(output/'manifest.json',dict(schema_version=1,source_analysis=str(analysis),
        source_hashes=source_hashes,temperatures=list(TEMPERATURES),source_episodes=300,
        selected_observations=3000,task_count=30,unit='task mean line; equal episode mean diamond',
        means={str(t):data[t]['overall'] for t in TEMPERATURES},stems=stems,
        code_sha256=sha(__file__)))
    seal(output,dict(status='PASS',figures=2,formats=['png','pdf'],source_episodes=300,
                     observations=3000,tasks=30))
    return output


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--analysis',required=True,type=Path)
    p.add_argument('--output',type=Path)
    a=p.parse_args(); analysis=a.analysis.resolve()
    output=a.output if a.output else analysis/'figures/reference_style'
    result=generate(analysis,output)
    print(json.dumps(dict(status='PASS',output=str(result),figures=[str(p) for p in sorted(result.glob('*.png'))]),indent=2))

if __name__=='__main__': main()
