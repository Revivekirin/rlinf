"""Native parallel OpenVLA-OFT action-bin distributions on a common observation bank."""
from __future__ import annotations
import argparse
import collections
import json
from pathlib import Path
import sys
import random
import inspect
import numpy as np
from phase3b import (ROOT, UPSTREAM, MODEL_SOURCE, MODEL_SHA, fresh, seal, verify_seal,
    verify_phase3a, bank_check, require_server_audit, motion_metrics, sources)
from phase3a_core import require, sha, array_hash, protocol_hash, REVISION
from nominal_behavior import read, save, csv_save


def distribution_metrics(a,b,temperature=1.0):
    a=np.asarray(a,dtype=np.float64)/temperature; b=np.asarray(b,dtype=np.float64)/temperature
    require(a.shape==b.shape and a.ndim==2 and np.isfinite(a).all() and np.isfinite(b).all(),'Invalid full-support logits')
    def logsoftmax(x):
        x=x-x.max(-1,keepdims=True); return x-np.log(np.exp(x).sum(-1,keepdims=True))
    lp,lq=logsoftmax(a),logsoftmax(b); p,q=np.exp(lp),np.exp(lq); lm=np.logaddexp(lp,lq)-np.log(2.)
    require(np.allclose(p.sum(-1),1) and np.allclose(q.sum(-1),1),'Probability normalization')
    result=dict(entropy_sft=-(p*lp).sum(-1),entropy_grpo=-(q*lq).sum(-1),
        top1_sft=p.max(-1),top1_grpo=q.max(-1),greedy_disagreement=a.argmax(-1)!=b.argmax(-1),
        kl_sft_grpo=(p*(lp-lq)).sum(-1),kl_grpo_sft=(q*(lq-lp)).sum(-1),
        js=.5*((p*(lp-lm)).sum(-1)+(q*(lq-lm)).sum(-1)))
    for k in (5,10):
        result[f'top{k}_sft']=np.sort(p,axis=-1)[:,-min(k,p.shape[-1]):].sum(-1)
        result[f'top{k}_grpo']=np.sort(q,axis=-1)[:,-min(k,q.shape[-1]):].sum(-1)
    require(np.all(result['js']>=-1e-12) and np.all(result['js']<=np.log(2)+1e-12),'JS bounds')
    return result


def preflight(args):
    a=require_server_audit(args.audit,args.phase3a,args.research_root)
    verify_phase3a(args.phase3a); b=bank_check(args.phase3a,raw=True)
    out=fresh(args.output); idx=read(args.phase3a/'replay_bank/index.json'); manifest=read(args.phase3a/'manifest.json')
    entries=[]
    for e in idx['observations']:
        row={k:e[k] for k in ('observation_id','source_policy','source_episode','category','query_index','simulator_step','file','file_sha256','arrays','source_metadata')}
        for pol in ('sft','grpo'):
            path=args.phase3a/'trajectories/canonical'/pol/(e['source_episode']+'.npz')
            with np.load(path,allow_pickle=False) as z: row[pol+'_stagnation']=motion_metrics(z['xyz'])[0]['stagnation']
        entries.append(row)
    save(out/'manifest.json',dict(phase3a=str(args.phase3a.resolve()),audit=str(args.audit.resolve()),audit_sha256=sha(args.audit/'validation.json'),
        observations=entries,bank=b,protocols=manifest['protocols'],source_hashes=sources(),native_model_sha256=MODEL_SHA,
        prefix_contract='NOT_APPLICABLE: native action embeddings are zero; 56 parallel slots, no action-token prefix enters inference',
        probability_support='All n_action_bins tokens admitted by native mask; no top-k truncation; metrics at T=1 and T=1.6',
        scope='Selected nine Long cases; source and episode stratification required; not representative of all 150 pairs',
        sft_loading='Pinned Base-Lora checkpoint with is_lora=False and lora_path=None, matching validated baseline'))
    seal(out,dict(status='PASS',observations=len(entries),raw_bank_verified=True)); print(json.dumps(b,indent=2))


def load_model(root,phase3a,model):
    import torch
    from omegaconf import OmegaConf
    from rlinf.models import get_model
    require(torch.cuda.is_available(),'GPU inference required')
    manifest=read(phase3a/'manifest.json'); protocol=manifest['protocols'][model]
    require(protocol['checkpoint_revision']==REVISION[model] and not protocol['lora_applied'],'Wrong baseline loading')
    cfg=OmegaConf.create(protocol['model_config']); cfg.model_path=protocol['checkpoint_path']; cfg.is_lora=False; cfg.lora_path=None
    obj=get_model(cfg).to('cuda').eval(); obj.requires_grad_(False)
    core=obj.get_base_model() if hasattr(obj,'get_base_model') else obj
    require(Path(inspect.getfile(type(core))).resolve()==(root/'upstream/RLinf'/MODEL_SOURCE).resolve(),'Loaded model class is not from audited checkout')
    require(core.action_dim==7 and core.num_action_chunks==8,'Native action shape mismatch')
    return obj,core,OmegaConf.to_container(cfg,resolve=True)


def worker(args):
    verify_seal(args.preflight); pf=read(args.preflight/'manifest.json'); p=Path(pf['phase3a'])
    require(sha(Path(pf['audit'])/'validation.json')==pf['audit_sha256'],'Audit changed')
    require_server_audit(Path(pf['audit']),p,args.research_root)
    if not args.smoke:
        require(args.smoke_result is not None,'Full worker requires completed one-observation smoke')
        verify_seal(args.smoke_result)
        sm=read(args.smoke_result/'manifest.json')
        require(sm['mode']=='smoke' and sm['preflight_sha256']==sha(args.preflight/'validation.json'),'Wrong smoke gate')
        require(sm['source_hashes']==sources(),'Analysis source changed since smoke')
    entries=pf['observations'][:1] if args.smoke else pf['observations']
    out=fresh(args.output/args.model)
    import torch
    model,core,cfg=load_model(args.research_root,p,args.model)
    runtime_sources={}
    for component in (type(core),core._build_multimodal_attention,core._process_vision_features,type(core.input_processor)):
        path=Path(inspect.getfile(component)).resolve(); runtime_sources[str(path)]=sha(path)
    lo=core.vocab_size-core.config.n_action_bins; hi=core.vocab_size; n=core.action_dim*core.num_action_chunks
    captured=[]
    def hook(module,inputs,output):
        v=output.logits[:, -n-1:-1,lo:hi].detach().float().cpu().numpy().copy()
        require(v.shape==(1,56,hi-lo) and np.isfinite(v).all(),'Invalid captured action logits')
        captured.append(v[0])
    results=[]
    for index,e in enumerate(entries):
        path=p/e['file']; require(sha(path)==e['file_sha256'],'Observation changed')
        with np.load(path,allow_pickle=False) as z: bank={k:v.copy() for k,v in z.items()}
        require(all(array_hash(v)==e['arrays'][k]['sha256'] for k,v in bank.items()),'Observation array changed')
        def native(sample=False):
            torch.manual_seed(3101+index); torch.cuda.manual_seed_all(3101+index); random.seed(3101+index); np.random.seed(3101+index)
            env=dict(main_images=torch.from_numpy(bank['main_image_policy'].copy()).unsqueeze(0),
                     states=torch.from_numpy(bank['proprio_policy'].copy()).unsqueeze(0),
                     task_descriptions=[e['source_metadata']['language']])
            if 'wrist_image_policy' in bank: env['wrist_images']=torch.from_numpy(bank['wrist_image_policy'].copy()).unsqueeze(0)
            with torch.inference_mode():
                actions,result=model.predict_action_batch(env_obs=env,do_sample=sample,temperature=1.6 if sample else 1.0,top_k=-1,top_p=1.0,calculate_logprobs=False,calculate_values=False)
            def arr(x): return x.detach().float().cpu().numpy() if x.dtype==torch.bfloat16 else x.detach().cpu().numpy()
            return arr(actions)[0],{k:arr(v)[0] for k,v in result['forward_inputs'].items()}
        plain=native() if args.smoke else None
        captured.clear(); handle=core.language_model.register_forward_hook(hook)
        try: actions,fi=native()
        finally: handle.remove()
        require(len(captured)==1,'Native inference must have one language forward')
        logits=captured[0]
        require(actions.shape==(8,7) and np.isfinite(actions).all(),'Invalid native decoded actions')
        require(np.array_equal(fi['action_tokens'].reshape(-1),logits.argmax(-1)+lo),'Hook action slice does not reproduce native greedy tokens')
        if plain:
            require(np.array_equal(actions,plain[0]) and all(np.array_equal(fi[k],plain[1][k]) for k in fi),'Instrumentation changes greedy behavior')
            # Sampling uses the same native path and seed, hook must not consume RNG.
            plain_sample=native(True); handle=core.language_model.register_forward_hook(hook)
            try: hooked_sample=native(True)
            finally: handle.remove()
            require(np.array_equal(plain_sample[0],hooked_sample[0]) and all(np.array_equal(plain_sample[1][k],hooked_sample[1][k]) for k in plain_sample[1]),'Instrumentation changes stochastic behavior')
        tensors={k:array_hash(fi[k]) for k in ('input_ids','attention_mask','pixel_values')}
        saved_equal={k:array_hash(fi[k])==e['arrays'][k]['sha256'] for k in tensors}
        require(fi['input_ids'].shape==bank['input_ids'].shape,'Token position alignment changed')
        normalized=core.bin_centers[np.clip(core.vocab_size-fi['action_tokens']-1,0,len(core.bin_centers)-1)]
        fname=e['observation_id']+'.npz'
        with (out/fname).open('xb') as f: np.savez_compressed(f,logits=logits,action_tokens=fi['action_tokens'],decoded_actions=actions,normalized_actions=normalized)
        results.append(dict(observation_id=e['observation_id'],file=fname,file_sha256=sha(out/fname),input_checksum=e['file_sha256'],processed_tensor_hashes=tensors,
            saved_preprocessing_equal=saved_equal,action_token_mask=[True]*56,action_logit_sequence_indices=list(range(core.vision_backbone.get_num_patches()*core.vision_backbone.get_num_images_in_input()+fi['input_ids'].shape[-1]-1,core.vision_backbone.get_num_patches()*core.vision_backbone.get_num_images_in_input()+fi['input_ids'].shape[-1]-1+56))))
        print(args.model,index+1,'/',len(entries),e['observation_id'],flush=True)
    save(out/'manifest.json',dict(model=args.model,revision=REVISION[args.model],mode='smoke' if args.smoke else 'full',preflight_sha256=sha(args.preflight/'validation.json'),source_hashes=sources(),
        config=cfg,config_hash=protocol_hash(cfg),records=results,action_token_ids=list(range(lo,hi)),bin_centers=core.bin_centers.tolist(),
        action_norm_stats=core._get_action_stats(),native_model_sha256=MODEL_SHA,
        token_to_bin_index=np.clip(core.vocab_size-np.arange(lo,hi)-1,0,len(core.bin_centers)-1).tolist(),
        free_generation='native greedy parallel action prediction; stochastic native path also equivalence-tested in smoke',
        prefix='NOT_APPLICABLE_NATIVE_PARALLEL',instrumentation_equivalence='greedy and stochastic exact tokens/actions/processed tensors' if args.smoke else 'gated by one-observation smoke',
        runtime_sources=runtime_sources,torch=torch.__version__,cuda=torch.version.cuda,device=torch.cuda.get_device_name(),dtype=str(next(model.parameters()).dtype)))
    seal(out,dict(status='PASS',observations=len(results),smoke=args.smoke))


def summarize(args):
    verify_seal(args.preflight); pf=read(args.preflight/'manifest.json')
    for name in ('sft','grpo'): verify_seal(args.output/name)
    a,b=[read(args.output/name/'manifest.json') for name in ('sft','grpo')]
    require(a['mode']==b['mode'] and a['preflight_sha256']==b['preflight_sha256']==sha(args.preflight/'validation.json'),'Worker mismatch')
    require(a['source_hashes']==b['source_hashes']==sources(),'Worker source mismatch')
    require(a.get('runtime_sources')==b.get('runtime_sources'),'Workers imported different native implementation sources')
    require(a['action_token_ids']==b['action_token_ids'] and a['bin_centers']==b['bin_centers'],'Action support/tokenization mismatch')
    aa={r['observation_id']:r for r in a['records']}; bb={r['observation_id']:r for r in b['records']}
    expected=pf['observations'][:1] if a['mode']=='smoke' else pf['observations']
    require(set(aa)==set(bb)=={e['observation_id'] for e in expected},'Missing/duplicate model observations')
    require(len(a['records'])==len(aa) and len(b['records'])==len(bb),'Duplicate model rows')
    tokens=[]; actions=[]; observations=[]; obs_stats=[]
    for e in expected:
        oid=e['observation_id']; ra,rb=aa[oid],bb[oid]
        require(ra['input_checksum']==rb['input_checksum']==e['file_sha256'],'Common raw observation violation')
        tensors_equal=ra['processed_tensor_hashes']==rb['processed_tensor_hashes']
        base=dict(observation_id=oid,origin=e['source_policy'],episode_id=e['source_episode'],outcome=e['category'],sft_stagnation=e['sft_stagnation'],grpo_stagnation=e['grpo_stagnation'])
        observations.append(dict(base,query_index=e['query_index'],simulator_step=e['simulator_step'],initial_state_sha256=e['source_metadata']['initial_state_sha256'],language=e['source_metadata']['language'],input_checksum=e['file_sha256'],processed_inputs_equal=tensors_equal,prefix_source='N/A_NATIVE_PARALLEL'))
        with np.load(args.output/'sft'/ra['file'],allow_pickle=False) as za,np.load(args.output/'grpo'/rb['file'],allow_pickle=False) as zb:
            require(all(np.isfinite(z[k]).all() for z in (za,zb) for k in z.files),'Nonfinite worker artifact')
            for temperature in (1.,1.6):
                metrics=distribution_metrics(za['logits'],zb['logits'],temperature)
                for pos in range(56): tokens.append(dict(base,temperature=temperature,position=pos,chunk_slot=pos//7,action_dimension=pos%7,**{k:float(v[pos]) for k,v in metrics.items()}))
                obs_stats.append(dict(base,temperature=temperature,**{k:float(np.mean(v)) for k,v in metrics.items()}))
            for normalized,key in ((False,'decoded_actions'),(True,'normalized_actions')):
                delta=zb[key]-za[key]
                for slot in range(8):
                    actions.append(dict(base,space='normalized_bins' if normalized else 'native_decoded',chunk_slot=slot,translation_l2=float(np.linalg.norm(delta[slot,:3])),rotation_l2=float(np.linalg.norm(delta[slot,3:6])),gripper_abs=float(abs(delta[slot,6])),
                        executed_gripper_disagreement=bool(np.sign(2*za['decoded_actions'][slot,6]-1)!=np.sign(2*zb['decoded_actions'][slot,6]-1))))
    groups=collections.defaultdict(list)
    for r in obs_stats: groups[(r['origin'],r['outcome'],r['temperature'])].append(r)
    summaries=[]
    keys=list(distribution_metrics(np.zeros((1,16)),np.zeros((1,16))))
    for (origin,outcome,t),rr in sorted(groups.items()):
        episodes=collections.defaultdict(list)
        for r in rr: episodes[r['episode_id']].append(r)
        summaries.append(dict(origin=origin,outcome=outcome,temperature=t,observations=len(rr),episodes=len(episodes),weighting='equal episode; mean positions then queries within episode',
            **{k:float(np.mean([np.mean([r[k] for r in er]) for er in episodes.values()])) for k in keys}))
    csv_save(args.output/'observations.csv',observations); csv_save(args.output/'token_metrics.csv',tokens); csv_save(args.output/'action_metrics.csv',actions); csv_save(args.output/'group_summary.csv',summaries)
    action_groups=collections.defaultdict(list)
    for r in actions: action_groups[(r['origin'],r['outcome'],r['space'])].append(r)
    action_summary=[]
    for (origin,outcome,space),rr in sorted(action_groups.items()):
        episodes=collections.defaultdict(list)
        for r in rr: episodes[r['episode_id']].append(r)
        action_summary.append(dict(origin=origin,outcome=outcome,space=space,episodes=len(episodes),weighting='equal source episode',
            **{k:float(np.mean([np.mean([r[k] for r in er]) for er in episodes.values()])) for k in ('translation_l2','rotation_l2','gripper_abs','executed_gripper_disagreement')}))
    csv_save(args.output/'action_group_summary.csv',action_summary)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,ax=plt.subplots(figsize=(6,4))
    for origin in ('sft','grpo'):
        rr=[r for r in obs_stats if r['temperature']==1.6 and r['origin']==origin]
        ax.scatter([r['entropy_sft'] for r in rr],[r['js'] for r in rr],label=origin+' origin',alpha=.5)
    ax.set(xlabel='SFT token entropy (nats), T=1.6',ylabel='Mean token JS (nats)'); ax.legend(); fig.tight_layout()
    (args.output/'figures').mkdir(); fig.savefig(args.output/'figures/entropy_js.png',dpi=140); plt.close(fig)
    heat=np.array([np.mean([r['js'] for r in tokens if r['temperature']==1.6 and r['position']==pos]) for pos in range(56)]).reshape(8,7)
    fig,ax=plt.subplots(figsize=(7,5)); im=ax.imshow(heat,aspect='auto'); ax.set(xlabel='Action dimension',ylabel='Chunk slot',xticks=range(7),xticklabels=['tx','ty','tz','rx','ry','rz','gripper'],title='Mean token JS, T=1.6 (query weighted)'); fig.colorbar(im,ax=ax,label='nats'); fig.tight_layout(); fig.savefig(args.output/'figures/position_js.png',dpi=140); plt.close(fig)
    save(args.output/'manifest.json',dict(mode=a['mode'],preflight_sha256=a['preflight_sha256'],source_hashes=sources(),observations=len(expected),
        normalization_equal=a['action_norm_stats']==b['action_norm_stats'],processed_inputs_equal=all(r['processed_inputs_equal'] for r in observations),
        prefix_conditioning='N/A: native masked-placeholder parallel inference. No fabricated teacher-forced prefix.',
        support='All native allowed action bins, untruncated; outside-support probabilities are exactly zero; nats',
        interpretation='T=1 entropy is underlying categorical logits, not entropy of the deterministic argmax policy. No causal mechanism or useful-strategy claim.'))
    (args.output/'summary.md').write_text('# Common-observation distributions\n\n'+json.dumps(summaries,indent=2)+'\n\nNative parallel slots: no autoregressive prefix exists. Selected Long cases only. Native normalization is retained; decoded action differences include any normalization differences.\n')
    seal(args.output,dict(status='PASS',observations=len(expected),full_support=True,prefix='N/A_NATIVE_PARALLEL'))
    print(args.output/'summary.md')


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='stage',required=True)
    a=sub.add_parser('preflight'); a.add_argument('--phase3a',type=Path,required=True); a.add_argument('--audit',type=Path,required=True); a.add_argument('--research-root',type=Path,default=ROOT); a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('worker'); a.add_argument('--preflight',type=Path,required=True); a.add_argument('--output',type=Path,required=True); a.add_argument('--model',choices=('sft','grpo'),required=True); a.add_argument('--research-root',type=Path,default=ROOT); a.add_argument('--smoke',action='store_true'); a.add_argument('--smoke-result',type=Path)
    a=sub.add_parser('summarize'); a.add_argument('--preflight',type=Path,required=True); a.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); globals()[args.stage](args)

if __name__=='__main__': main()
