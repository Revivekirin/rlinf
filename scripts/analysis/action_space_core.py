"""Exact scalar action distributions and task-cluster descriptive statistics."""
from __future__ import annotations
import collections
import numpy as np
from token_action_core import require
from scripts.common.action_schema import token_values,decode,prepared_gripper

SEED=7303


def softmax(logits,temperature):
    z=np.asarray(logits,dtype=float)
    require(z.ndim>=2 and np.isfinite(z).all() and np.isfinite(temperature) and temperature>0,'Invalid logits or temperature')
    z=z/temperature; z-=z.max(axis=-1,keepdims=True); p=np.exp(z); p/=p.sum(-1,keepdims=True)
    require(np.allclose(p.sum(-1),1,atol=1e-12,rtol=0),'Probability normalization')
    return p


def action_support(manifest):
    n=token_values(manifest)
    return decode(np.broadcast_to(n[:,None],(len(n),7)),manifest['action_norm_stats']).T


def validate_probability(p):
    require(np.isfinite(p).all() and np.all(p>=0) and np.allclose(p.sum(-1),1,atol=1e-12,rtol=0),'Invalid probability mass')


def moments(p,x):
    p=np.asarray(p,float); x=np.asarray(x,float); validate_probability(p)
    require(p.shape[-1]==x.shape[-1] and np.isfinite(x).all(),'Invalid scalar support')
    mu=np.sum(p*x,axis=-1); var=np.sum(p*(x-mu[...,None])**2,axis=-1)
    return mu,var


def wasserstein1(p,q,x):
    """Integral |CDF_P-CDF_Q| dx. Sort support; zero-width aliases are exact."""
    p=np.asarray(p,float); q=np.asarray(q,float); x=np.asarray(x,float)
    require(p.shape==q.shape and p.shape[-1]==x.shape[-1] and np.isfinite(x).all(),'W1 shape/support')
    validate_probability(p); validate_probability(q)
    order=np.argsort(x,axis=-1,kind='stable')
    xs=np.take_along_axis(x,order,-1)
    ps=np.take_along_axis(p,np.broadcast_to(order,p.shape),-1); qs=np.take_along_axis(q,np.broadcast_to(order,q.shape),-1)
    return np.sum(np.abs(np.cumsum(ps-qs,axis=-1)[...,:-1])*np.diff(xs,axis=-1),axis=-1)


def entropy(p):
    p=np.asarray(p,float); require(np.all(p>=0) and np.isfinite(p).all(),'Invalid entropy input')
    lp=np.zeros_like(p); np.log(p,out=lp,where=p>0)
    return -(p*lp).sum(-1)


def gripper_distribution(p,x,threshold_band=.05):
    validate_probability(p); require(threshold_band>0,'Threshold band must be positive')
    sign=prepared_gripper(x)
    opened=p[...,sign==-1].sum(-1); closed=p[...,sign==1].sum(-1); other=p[...,sign==0].sum(-1)
    require(np.allclose(opened+closed+other,1,atol=1e-12,rtol=0),'Gripper command mass')
    binary=entropy(np.stack([opened,closed],-1))
    # Binary entropy is only defined for a binary distribution. Never silently renormalize.
    binary=np.where(other==0,binary,np.nan)
    command_h=entropy(np.stack([opened,closed,other],-1))
    return dict(p_open=opened,p_close=closed,p_other=other,binary_entropy=binary,command_entropy=command_h,
                threshold_mass=p[...,np.abs(x-.5)<=threshold_band].sum(-1))


def task_bootstrap(rows,iterations=500,seed=SEED):
    """Integer multiplicities of whole tasks, stratified by suite."""
    require(iterations>0 and rows,'Empty bootstrap')
    keys=sorted({(r['suite'],int(r['task_id'])) for r in rows}); lookup={k:i for i,k in enumerate(keys)}
    suites=collections.defaultdict(list)
    for i,(suite,_) in enumerate(keys): suites[suite].append(i)
    rng=np.random.default_rng(seed); counts=np.zeros((iterations,len(keys)),int)
    for inds in suites.values():
        draws=rng.choice(inds,size=(iterations,len(inds)),replace=True)
        for b in range(iterations): counts[b]+=np.bincount(draws[b],minlength=len(keys))
    return np.array([lookup[(r['suite'],int(r['task_id']))] for r in rows]),counts


def episode_weights(rows):
    counts=collections.Counter((r['origin'],r['episode_id']) for r in rows)
    return np.array([1/counts[(r['origin'],r['episode_id'])] for r in rows])


def weighted_quantile(x,w,qs):
    """Inverse empirical CDF; fixed convention used in plots and bootstrap."""
    x=np.asarray(x,float); w=np.asarray(w,float); good=np.isfinite(x)&(w>0)
    if not good.any(): return np.full(np.asarray(qs).shape,np.nan)
    x=x[good]; w=w[good]; order=np.argsort(x,kind='stable'); x=x[order]; c=np.cumsum(w[order]); c/=c[-1]
    return x[np.minimum(np.searchsorted(c,qs,side='left'),len(x)-1)]


def weighted_spearman(x,y,w):
    """Spearman on weighted midranks; equals ranks on replicated integer weights."""
    x=np.asarray(x,float); y=np.asarray(y,float); w=np.asarray(w,float)
    good=np.isfinite(x)&np.isfinite(y)&(w>0); x=x[good]; y=y[good]; w=w[good]
    if len(x)<3: return np.nan
    def rank(a):
        unique,inv=np.unique(a,return_inverse=True); mass=np.bincount(inv,weights=w,minlength=len(unique))
        return (np.cumsum(mass)-.5*mass)[inv]
    a=rank(x); b=rank(y); a-=np.average(a,weights=w); b-=np.average(b,weights=w)
    denom=np.sqrt(np.sum(w*a*a)*np.sum(w*b*b))
    return float(np.sum(w*a*b)/denom) if denom>0 else np.nan


def correlation(rows,xkey,ykey,iterations=500):
    valid=[r for r in rows if r[xkey] is not None and r[ykey] is not None and np.isfinite(r[xkey]) and np.isfinite(r[ykey])]
    if len(valid)<3: return dict(n=len(valid),spearman=None,ci_low=None,ci_high=None,bootstrap_valid=0)
    x=np.array([r[xkey] for r in valid]); y=np.array([r[ykey] for r in valid]); w=episode_weights(valid)
    value=weighted_spearman(x,y,w); draws=[]
    if iterations:
        task,counts=task_bootstrap(valid,iterations)
        draws=[weighted_spearman(x,y,w*c[task]) for c in counts]
    finite=np.asarray(draws)[np.isfinite(draws)]; ci=np.quantile(finite,[.025,.975]) if len(finite) else [np.nan,np.nan]
    return dict(n=len(valid),spearman=value,ci_low=ci[0],ci_high=ci[1],bootstrap_valid=len(finite))


def quantile_bins(rows,xkey,ykey,iterations=500,stat='median'):
    """Ten fixed weighted JS quantile bins; whole-task conditional-statistic CIs."""
    x=np.array([r[xkey] for r in rows],float); y=np.array([r[ykey] for r in rows],float); w=episode_weights(rows)
    edges=weighted_quantile(x,w,np.linspace(0,1,11)); labels=np.searchsorted(edges[1:-1],x,side='right')
    task,counts=task_bootstrap(rows,iterations) if iterations else (None,None)
    result=[]
    for k in range(10):
        ix=labels==k
        if not ix.any(): continue
        xx=x[ix]; yy=y[ix]; ww=w[ix]; q=weighted_quantile(yy,ww,[.25,.5,.75])
        center=weighted_quantile(xx,ww,[.5])[0]; mean=float(np.average(yy,weights=ww))
        ci=[]
        if iterations:
            for count in counts:
                bw=ww*count[task[ix]]
                if bw.sum()>0: ci.append(np.average(yy,weights=bw) if stat=='mean' else weighted_quantile(yy,bw,[.5])[0])
        low,high=np.quantile(ci,[.025,.975]) if ci else (np.nan,np.nan)
        result.append(dict(bin=k,x_low=edges[k],x_high=edges[k+1],x_median=center,n_points=int(ix.sum()),
                           n_source_episodes=len({(rows[i]['origin'],rows[i]['episode_id']) for i in np.flatnonzero(ix)}),
                           mean=mean,p25=q[0],median=q[1],p75=q[2],ci_low=low,ci_high=high,ci_statistic=stat,bootstrap_valid=len(ci)))
    return result
