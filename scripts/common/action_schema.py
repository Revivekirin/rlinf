"""Audited native OpenVLA-OFT 8x7 action contract, with no simulator imports."""
import numpy as np
from .integrity import require

UPSTREAM='e05f71e80d14ab4b55e5027ee739cfbcc6b12b9d'
SOURCE_HASHES={
    'model.py':'22a5b7864bc125bb8ca757b1ed14e901972eea96ab243e9b5d663f61b3f219e9',
    'action_utils.py':'64ff4501b89b178bbdcf17ed76219d4d3075f9ba18dd7a86efdd939ae6be28cd',
    'libero_env.py':'15e78dd2be1cd8225683cf392a1ffc0c4958d715236fe2c41f9faa5df3631679',
}
SOURCE_PATHS={
 'model.py':'rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py',
 'action_utils.py':'rlinf/envs/action_utils.py',
 'libero_env.py':'rlinf/envs/sim/libero/libero_env.py',
}
DIMENSIONS=('tx','ty','tz','rx','ry','rz','gripper')
GROUPS={'translation':(0,1,2),'rotation':(3,4,5),'gripper':(6,)}


def slot_indices():
    return [(i,i//7,i%7,DIMENSIONS[i%7]) for i in range(56)]


def token_values(manifest):
    """Native setup has B edges, B-1 centers; clipping aliases the first two tokens."""
    b=int(manifest['bins']); require(b>=10,'Too few action bins')
    edges=np.linspace(-1,1,b); centers=(edges[:-1]+edges[1:])/2
    indices=np.asarray(manifest['token_to_bin_index'])
    expected=np.clip(b-np.arange(b)-1,0,b-2)
    require(np.array_equal(indices,expected),'Token/bin mapping differs from audited native implementation')
    return centers[indices]


def decode(normalized,stats):
    """Native BOUNDS_Q99 formula, including its 1e-8 range term."""
    a=np.asarray(normalized,dtype=float); require(a.shape[-1]==7,'Expected seven dimensions')
    low=np.asarray(stats['q01']); high=np.asarray(stats['q99']); mask=np.asarray(stats['mask'],dtype=bool)
    require(low.shape==high.shape==mask.shape==(7,) and np.all(high>=low),'Invalid normalization')
    return np.where(mask,.5*(a+1)*(high-low+1e-8)+low,a)


def prepared_gripper(decoded):
    """Native LIBERO command: -1 open (reset_gripper_open), +1 close, 0 exact threshold."""
    return -np.sign(2*np.asarray(decoded)-1)


def validate_saved(arrays,manifest):
    require(set(arrays)=={'logits','action_tokens','decoded_actions','normalized_actions'},'Saved action fields')
    b=int(manifest['bins'])
    require(arrays['logits'].shape==(56,b),'Saved logits shape')
    require(all(arrays[k].shape==(8,7) for k in ('action_tokens','decoded_actions','normalized_actions')),'Action shape')
    require(all(np.isfinite(v).all() for v in arrays.values()),'Nonfinite saved tensor')
    ids=np.asarray(arrays['action_tokens'])-int(manifest['token_start'])
    require(np.issubdtype(ids.dtype,np.integer) and np.all((ids>=0)&(ids<b)),'Action token support')
    require(np.array_equal(ids.reshape(-1),arrays['logits'].argmax(-1)),'Greedy token mismatch')
    normalized=token_values(manifest)[ids]
    require(np.allclose(normalized,arrays['normalized_actions'],rtol=0,atol=1e-12),'Normalized token decoding mismatch')
    require(np.allclose(decode(normalized,manifest['action_norm_stats']),arrays['decoded_actions'],rtol=0,atol=1e-12),
            'Native Q99 decoding regression mismatch')
    return True


def vector_metrics(a,b,epsilon=.01):
    a=np.asarray(a,float); b=np.asarray(b,float)
    require(a.shape==b.shape and a.shape[-1]==3 and epsilon>0,'Invalid vector comparison')
    na=np.linalg.norm(a,axis=-1); nb=np.linalg.norm(b,axis=-1); valid=(na>epsilon)&(nb>epsilon)
    cosine=np.full(na.shape,np.nan)
    np.divide(np.sum(a*b,axis=-1),na*nb,out=cosine,where=valid)
    return dict(magnitude_sft=na,magnitude_grpo=nb,magnitude_delta=nb-na,difference_l2=np.linalg.norm(b-a,axis=-1),
                cosine=np.clip(cosine,-1,1),cosine_defined=valid,near_zero_sft=na<=epsilon,near_zero_grpo=nb<=epsilon)


def contract():
    return dict(chunks=8,dimensions=list(DIMENSIONS),slot_rule='slot = chunk*7 + dimension',
        translation='native controller command vector; physical frame/scale UNKNOWN in copied runtime',
        rotation='native rotation command vector; physical rotation representation/scale UNKNOWN; no SO(3) angles inferred',
        normalization='BOUNDS_Q99 verified against every saved native decoded action; mask and +1e-8 range from native source',
        gripper='prepared = -sign(2*decoded-1); -1 open from native reset_gripper_open; +1 close command; 0 threshold',
        unit='command units, not measured meters or radians',
        distinction='decoded is native prediction; normalized is bin-center space; prepared sign is counterfactual, not executed',
        semantics_evidence={n:'https://github.com/RLinf/RLinf/blob/'+UPSTREAM+'/'+p for n,p in SOURCE_PATHS.items()})
