"""Legacy GRPO pilot and SFT/GRPO repeated rollouts via nominal collect_episode."""
from __future__ import annotations
import argparse
import itertools
import json
import gc
import random
import inspect
import hashlib
import shlex
import os
from pathlib import Path
import sys
import numpy as np
from phase3b import (ROOT,fresh,seal,verify_seal,verify_phase3a,require_server_audit,sources,motion_metrics)
from phase3a_core import require,sha,array_hash,protocol_hash,legacy,REVISION
from nominal_behavior import read,save,csv_save

REPEAT_ARTIFACTS = ('metadata.json','states.npz','actions.npz','observations.npz')
REPEAT_POLICIES = ('sft','grpo')
REPEAT_SUITES = ('libero_goal','libero_spatial','libero_10')
REPEAT_REPOS = {'sft':'RLinf/RLinf-OpenVLAOFT-LIBERO-130-Base-Lora',
                'grpo':'RLinf/RLinf-OpenVLAOFT-LIBERO-130'}


def repeat_seed(suite,task,trial,replicate):
    require(suite in REPEAT_SUITES and 0<=task<10 and 0<=trial<5 and 1<=replicate<=5,
            'Repeated-rollout seed bounds')
    return 100000*REPEAT_SUITES.index(suite)+100*task+trial+1000000*replicate


def repeat_job_id(episode_id,policy,seed):
    require(policy in REPEAT_POLICIES and 0<=seed<2**32,'Invalid policy/sampling seed')
    return f'{episode_id}-{policy}-s{seed:06d}'


def repeat_plan(args):
    info=verify_phase3a(args.phase3a)
    mapping_path=Path(args.mapping) if args.mapping else Path(info['manifest']['mapping'])
    mapping=read(mapping_path)
    require(mapping['schema_version']==1 and mapping['trials_compared']>=5,'Mapping audit insufficient')
    require(args.suite in REPEAT_SUITES,'Unsupported suite')
    tasks=list(args.task_ids); trials=list(args.trial_ids); policies=list(args.policies)
    require(tasks and trials and policies and len(tasks)==len(set(tasks)) and len(trials)==len(set(trials))
            and len(policies)==len(set(policies)),'Empty or duplicate plan selection')
    require(all(0<=t<10 for t in tasks) and all(0<=i<5 for i in trials),'Task/trial out of range')
    require(all(p in REPEAT_POLICIES for p in policies),'Unsupported policy')
    require(all(protocol_hash(mapping)==info['manifest']['protocols'][p]['mapping_sha256'] for p in policies),
            'Mapping differs from Phase 3A audited mapping')
    require(np.isfinite(args.temperature) and args.temperature>0,'Invalid temperature')
    # collect_episode uses these native values; accepting others would mislabel the run.
    require(args.top_p==1.0 and args.top_k==-1,'Native collector requires top_p=1.0 and top_k=-1')
    require(args.sampling_seeds is None or args.repeats is None,'Choose seeds or repeats')
    if args.sampling_seeds is None:
        repeats=5 if args.repeats is None else args.repeats
        require(1<=repeats<=5,'Repeats must be 1..5')
    else:
        require(args.sampling_seeds and len(set(args.sampling_seeds))==len(args.sampling_seeds)
                and all(0<=s<2**32 for s in args.sampling_seeds),'Invalid sampling seeds')
        repeats=None
    lookup={(r['suite'],r['subset_task_id']):r for r in mapping['tasks']}
    records={p:{r['episode_id']:r for r in info['records'][p]} for p in policies}
    base_protocols=info['manifest']['protocols']
    experiment=dict(schema_version=2,suite=args.suite,task_ids=tasks,trial_ids=trials,
        policies=policies,sampling_mode='stochastic',temperature=args.temperature,
        top_p=args.top_p,top_k=args.top_k,horizon=520,action_chunks=8,simulator_seed=0,
        mapping_sha256=sha(mapping_path),phase3a_validation_sha256=sha(args.phase3a/'validation.json'),
        upstream_commit='e05f71e80d14ab4b55e5027ee739cfbcc6b12b9d',
        collector_sha256=sha(ROOT/'scripts/evaluate/nominal_collect.py'),
        seed_rule='100000*suite_index+100*task_id+trial_id+1000000*replicate; suite order goal,spatial,10' if repeats else 'explicit sampling seeds',
        repeats=repeats,explicit_sampling_seeds=args.sampling_seeds)
    protocols={}
    for policy in policies:
        p=base_protocols[policy]
        require(p['checkpoint_repo']==REPEAT_REPOS[policy] and p['checkpoint_revision']==REVISION[policy]
                and not p['lora_applied'],'Wrong baseline checkpoint/LoRA')
        require(p['unnorm_key']=='libero_130_no_noops_trajall','Wrong normalization')
        protocols[policy]=dict(experiment=experiment,policy=policy,checkpoint_repo=p['checkpoint_repo'],
            checkpoint_revision=p['checkpoint_revision'],checkpoint_path=p['checkpoint_path'],
            model_config=p['model_config'],lora_applied=False,unnorm_key=p['unnorm_key'],
            environment=dict(seed=0,task_suite_name=args.suite,is_eval=True,auto_reset=False,
                ignore_terminations=False,reset_gripper_open=False,use_ordered_reset_state_ids=True,
                max_episode_steps=520,max_steps_per_rollout_epoch=520,save_video=False))
    hashes={p:protocol_hash(v) for p,v in protocols.items()}
    jobs=[]
    for task in tasks:
        mr=lookup.get((args.suite,task)); require(mr is not None,'Task absent from mapping')
        for trial in trials:
            eid=f'{args.suite}-t{task:02d}-i{trial:02d}'
            references={p:records[p][eid] for p in policies}
            require(all(r['task_name']==mr['task_name'] and r['initial_state_sha256']==mr['trial_sha256'][trial]
                        and r['task_id']==task and r['trial_id']==trial for r in references.values()),'Task/state mismatch')
            require(len({r['language'] for r in references.values()})==1,'Cross-policy instruction mismatch')
            seeds=args.sampling_seeds if repeats is None else [repeat_seed(args.suite,task,trial,k) for k in range(1,repeats+1)]
            for policy in policies:
                for seed in seeds:
                    jobs.append(dict(job_id=repeat_job_id(eid,policy,seed),episode_id=eid,
                        suite=args.suite,task_id=task,trial_id=trial,policy=policy,policy_seed=seed,
                        initial_state_sha256=mr['trial_sha256'][trial],task_name=mr['task_name'],
                        language=references[policy]['language'],libero_130_task_id=mr['libero_130_task_id'],
                        mapping=mr,protocol_hash=hashes[policy]))
    require(len({j['job_id'] for j in jobs})==len(jobs),'Job ID collision')
    out=fresh(args.output)
    save(out/'manifest.json',dict(phase3a=str(args.phase3a.resolve()),mapping=str(mapping_path.resolve()),
        experiment=experiment,experiment_hash=protocol_hash(experiment),protocols=protocols,
        protocol_hashes=hashes,jobs=jobs,expected=len(jobs),
        rng_scope='Collector seeds Python, NumPy, torch CPU and CUDA after reset; simulator progression may consume global RNG. Same seed across policies is an identity label, not common random numbers.'))
    seal(out,dict(status='PASS',expected=len(jobs)))
    print(json.dumps(dict(plan=str(out),expected=len(jobs),experiment_hash=protocol_hash(experiment)),indent=2))


def repeat_preflight(args):
    require(__debug__,'PYTHONOPTIMIZE must be unset')
    verify_seal(args.plan); pl=read(args.plan/'manifest.json')
    p=Path(pl['phase3a']); verify_phase3a(p)
    require(sha(p/'validation.json')==pl['experiment']['phase3a_validation_sha256'],'Phase 3A changed')
    require(sha(pl['mapping'])==pl['experiment']['mapping_sha256'],'Mapping changed')
    require(sha(args.research_root/'scripts/evaluate/nominal_collect.py')==pl['experiment']['collector_sha256'],'Collector changed')
    require_server_audit(args.audit,p,args.research_root)
    require(os.environ.get('LIBERO_TYPE','standard')=='standard','Only original LIBERO supported')
    require(Path(os.environ.get('EMBODIED_PATH','')).resolve()==args.research_root/'upstream/RLinf/examples/embodiment','EMBODIED_PATH mismatch')
    print(json.dumps(dict(status='PASS',jobs=pl['expected'],experiment_hash=pl['experiment_hash'],
        audit=str(args.audit.resolve()),gpu_inference='NOT RUN'),indent=2))


def check_repeat_entry(entry,job,pl):
    require(__debug__,'PYTHONOPTIMIZE must be unset')
    sys.path.insert(0,str(ROOT/'scripts/evaluate'))
    from nominal_validate import validate_episode
    path=Path(entry['episode_path'])
    require(entry['job_id']==job['job_id'] and entry['protocol_hash']==job['protocol_hash'],'Entry identity/protocol mismatch')
    require(path.name==job['episode_id'] and path.parent.parent.name==job['job_id'],'Episode path mismatch')
    protocol=read(path.parent.parent/'manifest.json')
    require(protocol==dict(protocol=pl['protocols'][job['policy']],protocol_hash=job['protocol_hash']),'Job protocol mismatch')
    require(entry['hashes']=={n:sha(path/n) for n in REPEAT_ARTIFACTS},'Episode file hash mismatch')
    m=validate_episode(path,job['protocol_hash'])
    expected=dict(episode_id=job['episode_id'],run_id=path.parents[3].name,
        repeated_job_id=job['job_id'],policy=job['policy'],suite=job['suite'],task_id=job['task_id'],trial_id=job['trial_id'],
        task_name=job['task_name'],libero_130_task_id=job['libero_130_task_id'],
        initial_state_sha256=job['initial_state_sha256'],policy_seed=job['policy_seed'],
        simulator_seed=0,checkpoint_repo=pl['protocols'][job['policy']]['checkpoint_repo'],
        checkpoint_revision=pl['protocols'][job['policy']]['checkpoint_revision'],
        lora_applied=False,sampling_mode='stochastic',temperature=pl['experiment']['temperature'],
        language=job['language'],protocol_hash=job['protocol_hash'])
    require(all(m.get(k)==v for k,v in expected.items()),'Episode metadata mismatch')
    for name in ('states.npz','actions.npz','observations.npz'):
        with np.load(path/name,allow_pickle=False) as z:
            require(all(np.isfinite(z[k]).all() for k in z.files if np.issubdtype(z[k].dtype,np.number)),
                    f'Nonfinite {name}')
    return m


def repeat_signature(path,language):
    with np.load(path/'states.npz',allow_pickle=False) as z:
        state={k:array_hash(z[k][0]) for k in z.files}
    with np.load(path/'observations.npz',allow_pickle=False) as z:
        obs={k:array_hash(z[k][0]) for k in z.files if k!='query_step'}
    return dict(state=state,observation=obs,language=language)


def repeat_validate_state(output,pl,partial=False,selected=None):
    require(__debug__,'PYTHONOPTIMIZE must be unset')
    mf=read(output/'manifest.json')
    require(mf==dict(plan=str(Path(mf['plan']).resolve()),plan_sha256=sha(Path(mf['plan'])/'validation.json'),
                     experiment_hash=pl['experiment_hash']),'Run manifest mismatch')
    jobs={j['job_id']:j for j in pl['jobs']}; selected=set(selected or ())
    require(selected<=set(jobs),'Unknown requested job')
    entries_dir=output/'entries'; jobs_dir=output/'jobs'
    found={p.stem for p in entries_dir.glob('*.json')} if entries_dir.exists() else set()
    require(found<=set(jobs),'Unexpected entry')
    present={p.name for p in jobs_dir.iterdir() if p.is_dir()} if jobs_dir.exists() else set()
    require(present<=set(jobs),'Unexpected job directory')
    signatures={}; completed=[]; recoverable=[]
    for jid in sorted(found):
        j=jobs[jid]; entry=read(entries_dir/(jid+'.json'))
        require(Path(entry['episode_path']).resolve()==(jobs_dir/jid/'rollouts'/j['episode_id']).resolve(),
                'Entry points outside this run')
        m=check_repeat_entry(entry,j,pl)
        sig=repeat_signature(Path(entry['episode_path']),m['language'])
        key=j['episode_id']
        if key in signatures: require(sig==signatures[key],'Cross-seed/policy initial state differs')
        signatures[key]=sig; completed.append(jid)
    for jid in sorted(present-found):
        dest=jobs_dir/jid; path=dest/'rollouts'/jobs[jid]['episode_id']
        if (dest/'manifest.json').is_file():
            require(read(dest/'manifest.json')==dict(protocol=pl['protocols'][jobs[jid]['policy']],
                protocol_hash=jobs[jid]['protocol_hash']),'Incomplete job has incompatible protocol')
        if path.is_dir() and all((path/n).is_file() for n in REPEAT_ARTIFACTS):
            entry=dict(job_id=jid,episode_path=str(path.resolve()),protocol_hash=jobs[jid]['protocol_hash'],
                hashes={n:sha(path/n) for n in REPEAT_ARTIFACTS})
            check_repeat_entry(entry,jobs[jid],pl)
            recoverable.append(jid)
    pending=sorted(set(jobs)-set(completed))
    require(not selected or selected<=set(completed),'Requested subset incomplete')
    status='COMPLETE_VALID' if not pending else 'PARTIAL_VALID' if partial else 'FAILED'
    return dict(status=status,expected_jobs=len(jobs),completed_jobs=len(completed),
        pending_job_ids=pending,recoverable_job_ids=recoverable,selected_job_ids=sorted(selected),
        initial_state_groups_verified=len(signatures))


def collect_repeat_job(args,pl,job,model,model_cfg):
    sys.path.insert(0,str(args.research_root/'scripts/evaluate'))
    import nominal_collect as nc
    from scripts.diagnostics.audit_libero_mapping import task_identity,state_hash
    import torch
    from omegaconf import OmegaConf
    from rlinf.envs.sim.libero.libero_env import LiberoEnv
    require(model_cfg==pl['protocols'][job['policy']]['model_config'],'Runtime model config differs from plan')
    random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
    cfg=OmegaConf.load(args.research_root/'upstream/RLinf/examples/embodiment/config/env/libero_10.yaml')
    cfg.task_suite_name=job['suite']; cfg.total_num_envs=1; cfg.is_eval=True; cfg.auto_reset=False
    cfg.ignore_terminations=False; cfg.reset_gripper_open=False; cfg.use_ordered_reset_state_ids=True
    cfg.max_episode_steps=520; cfg.max_steps_per_rollout_epoch=520; cfg.video_cfg.save_video=False; cfg.seed=0
    dest=args.output/'jobs'/job['job_id']
    require(not dest.exists(),f'Incomplete or conflicting job directory: {dest}')
    dest.mkdir(parents=True)
    save(dest/'manifest.json',dict(protocol=pl['protocols'][job['policy']],protocol_hash=job['protocol_hash']))
    env=LiberoEnv(cfg,num_envs=1,seed_offset=0,total_num_processes=1,worker_info=None)
    try:
        mr=job['mapping']; task=env.task_suite.get_task(job['task_id'])
        require(task_identity(task)==tuple(mr[k] for k in ('task_name','problem_folder','bddl_file','init_states_file')),
                'Live task identity mismatch')
        require(state_hash(env.task_suite.get_task_init_states(job['task_id'])[job['trial_id']])==job['initial_state_sha256'],
                'Live initial state mismatch')
        metadata=dict(complete=True,episode_id=job['episode_id'],run_id=args.output.name,
            repeated_job_id=job['job_id'],policy=job['policy'],protocol_hash=job['protocol_hash'],
            suite=job['suite'],task_id=job['task_id'],trial_id=job['trial_id'],
            libero_130_task_id=job['libero_130_task_id'],task_name=job['task_name'],
            initial_state_sha256=job['initial_state_sha256'],simulator_seed=0,policy_seed=job['policy_seed'],
            checkpoint_repo=pl['protocols'][job['policy']]['checkpoint_repo'],
            checkpoint_revision=pl['protocols'][job['policy']]['checkpoint_revision'],lora_applied=False,
            sampling_mode='stochastic',temperature=pl['experiment']['temperature'],horizon=520,
            wait_steps=15,action_chunks=8,unnorm_key='libero_130_no_noops_trajall',
            upstream_commit=pl['experiment']['upstream_commit'],collector_sha256=pl['experiment']['collector_sha256'])
        path=nc.collect_episode(env,model,suite=job['suite'],task_id=job['task_id'],trial_id=job['trial_id'],
            sampling=True,temperature=pl['experiment']['temperature'],policy_seed=job['policy_seed'],
            metadata=metadata,run_dir=dest)
    finally:
        env.env.close()
    return path


def repeat_run(args):
    repeat_preflight(args)
    pl=read(args.plan/'manifest.json')
    jobs=pl['jobs']
    if args.smoke:
        sample=[j for j in jobs if j['suite']=='libero_10' and j['task_id']==8 and j['trial_id']==0]
        require({j['policy'] for j in sample}==set(REPEAT_POLICIES),'Smoke needs SFT and GRPO task 8/state 0')
        seeds=sorted({j['policy_seed'] for j in sample})[:2]
        require(len(seeds)==2,'Smoke needs two sampling seeds')
        jobs=[j for j in sample if j['policy_seed'] in seeds]
        require(len(jobs)==4,'Smoke must contain four jobs')
    manifest=dict(plan=str(args.plan.resolve()),plan_sha256=sha(args.plan/'validation.json'),
                  experiment_hash=pl['experiment_hash'])
    if args.output.exists():
        require(args.resume and read(args.output/'manifest.json')==manifest,
                'Existing run needs --resume with the same plan')
    else:
        fresh(args.output); save(args.output/'manifest.json',manifest)
    (args.output/'entries').mkdir(exist_ok=True)
    (args.output/'jobs').mkdir(exist_ok=True)
    model=None; model_core=None; loaded_policy=None; model_cfg=None
    from phase3b_distribution import load_model
    for job in jobs:
        jid=job['job_id']; entry_file=args.output/'entries'/(jid+'.json')
        dest=args.output/'jobs'/jid; episode_path=dest/'rollouts'/job['episode_id']
        try:
            if entry_file.exists():
                entry=read(entry_file)
                require(Path(entry['episode_path']).resolve()==episode_path.resolve(),'Entry points outside this run')
                check_repeat_entry(entry,job,pl)
                print('SKIP',jid,flush=True)
                continue
            if dest.exists():
                require(episode_path.is_dir(),f'Incomplete job preserved for inspection: {dest}')
                entry=dict(job_id=jid,episode_path=str(episode_path.resolve()),
                    protocol_hash=job['protocol_hash'],hashes={n:sha(episode_path/n) for n in REPEAT_ARTIFACTS},
                    provenance='recovered_after_interruption')
                check_repeat_entry(entry,job,pl); save(entry_file,entry)
                print('RECOVER',jid,flush=True)
                continue
            if loaded_policy!=job['policy']:
                if model is not None:
                    model=None; model_core=None
                    gc.collect()
                    import torch
                    torch.cuda.empty_cache()
                model,model_core,model_cfg=load_model(args.research_root,Path(pl['phase3a']),job['policy'])
                loaded_policy=job['policy']
            path=collect_repeat_job(args,pl,job,model,model_cfg)
            entry=dict(job_id=jid,episode_path=str(path.resolve()),protocol_hash=job['protocol_hash'],
                hashes={n:sha(path/n) for n in REPEAT_ARTIFACTS},provenance='new')
            m=check_repeat_entry(entry,job,pl)
            save(entry_file,entry)
            print('COMPLETE',jid,'success',m['success'],'steps',m['episode_length'],flush=True)
        except Exception as exc:
            failures=args.output/'failures'; failures.mkdir(exist_ok=True)
            failure=failures/(jid+'.json')
            if not failure.exists():
                save(failure,dict(job_id=jid,error_type=type(exc).__name__,message=str(exc)))
            raise
    state=repeat_validate_state(args.output,pl,partial=True)
    print(json.dumps(state,indent=2))


def repeat_validate(args):
    try:
        verify_seal(Path(read(args.output/'manifest.json')['plan']))
        pl=read(Path(read(args.output/'manifest.json')['plan'])/'manifest.json')
        result=repeat_validate_state(args.output,pl,partial=args.partial,selected=args.jobs)
    except (ValueError,AssertionError,FileNotFoundError,KeyError) as exc:
        print(json.dumps(dict(status='FAILED',error=str(exc)),indent=2))
        raise SystemExit(1) from exc
    print(json.dumps(result,indent=2))
    if result['status']=='FAILED': raise SystemExit(1)


def seed_for(task,trial,replicate):
    require(task in (3,8) and 0<=trial<5 and 1<=replicate<=5,'Pilot A bounds')
    return 200000+100*task+trial+1000000*replicate


def plan(args):
    require_server_audit(args.audit,args.phase3a,args.research_root); info=verify_phase3a(args.phase3a)
    out=fresh(args.output); mapping=read(Path(info['manifest']['mapping']))
    lookup={(r['suite'],r['subset_task_id']):r for r in mapping['tasks']}
    records={r['episode_id']:r for r in info['records']['stochastic']}; jobs=[]
    for task in (8,3):
        for trial in range(5):
            eid=f'libero_10-t{task:02d}-i{trial:02d}'; m=records[eid]; mr=lookup[('libero_10',task)]
            require(m['task_id']==task and m['trial_id']==trial and m['task_name']==mr['task_name'],'Zero-based task identity mismatch')
            require(m['initial_state_sha256']==mr['trial_sha256'][trial],'Initial state mismatch')
            for rep in range(1,6):
                seed=seed_for(task,trial,rep); require(seed!=m['policy_seed'],'Unexpected existing-seed overlap')
                jobs.append(dict(job_id=f'{eid}-s{seed}',episode_id=eid,task_id=task,trial_id=trial,replicate=rep,policy_seed=seed,
                    initial_state_sha256=m['initial_state_sha256'],task_name=m['task_name'],mapping=mr,reference=m))
    save(out/'manifest.json',dict(phase3a=str(args.phase3a.resolve()),audit=str(args.audit.resolve()),audit_sha256=sha(args.audit/'validation.json'),
        jobs=jobs,expected=50,source_hashes=sources(),checkpoint_revision=REVISION['grpo'],
        config=info['manifest']['protocols']['stochastic'],task_indexing='zero-based, checked against mapping and saved metadata',
        seed_rule='200000 + 100*task_id + trial_id + 1000000*replicate; replicate=1..5',
        reuse='No overlap with existing nominal sampling seeds; pilot reuses its two smoke episodes exactly',
        rng_control='Environment config seed=0 and explicit fixed init state. Collector seeds Python, NumPy, torch and CUDA with policy_seed after reset. CPU environment RNG coupling cannot be excluded; not pure policy variance.'))
    seal(out,dict(status='PASS',expected=50,smoke_expected=2)); print(out/'manifest.json')


def check_entry(e,job):
    path=Path(e['episode_path']); sys.path.insert(0,str(ROOT/'scripts/evaluate'))
    from nominal_validate import validate_episode
    m=validate_episode(path,e['protocol_hash'])
    require(m['episode_id']==job['episode_id'] and m['policy_seed']==job['policy_seed'] and m['initial_state_sha256']==job['initial_state_sha256'],'Pilot identity/seed mismatch')
    require(m['checkpoint_revision']==REVISION['grpo'] and m['sampling_mode']=='stochastic' and m['temperature']==1.6 and not m['lora_applied'],'Pilot policy protocol mismatch')
    require(m['simulator_seed']==0 and m['task_id']==job['task_id'] and m['trial_id']==job['trial_id'] and m['task_name']==job['task_name'],'Pilot environment identity')
    require(e['hashes']=={n:sha(path/n) for n in ('metadata.json','states.npz','actions.npz','observations.npz')},'Pilot artifact changed')
    protocol=read(path.parent.parent/'manifest.json')
    require(protocol_hash(protocol['protocol'])==e['protocol_hash']==protocol['protocol_hash'],'Pilot config hash')
    return m


def run(args):
    verify_seal(args.plan); pl=read(args.plan/'manifest.json'); p=Path(pl['phase3a'])
    require(sha(Path(pl['audit'])/'validation.json')==pl['audit_sha256'],'Audit changed')
    require_server_audit(Path(pl['audit']),p,args.research_root)
    jobs=pl['jobs'][:2] if args.mode=='smoke' else pl['jobs']
    reused={}
    if args.mode=='pilot' and args.smoke is not None:
        verify_seal(args.smoke); sm=read(args.smoke/'manifest.json'); sv=read(args.smoke/'validation.json')
        require(sm['mode']=='smoke' and sm['plan_sha256']==sha(args.plan/'validation.json') and sv['seed_divergence_verified'],'Wrong pilot smoke')
        reused={e['job_id']:e for e in read(args.smoke/'index.json')['entries']}
    manifest=dict(mode=args.mode,plan=str(args.plan.resolve()),plan_sha256=sha(args.plan/'validation.json'),source_hashes=sources(),expected=len(jobs),
        smoke=str(args.smoke.resolve()) if args.smoke else None,rng_control=pl['rng_control'])
    if args.output.exists():
        require(args.resume and read(args.output/'manifest.json')==manifest,'Output exists; explicit --resume with same plan required')
    else:
        fresh(args.output); save(args.output/'manifest.json',manifest)
    (args.output/'entries').mkdir(exist_ok=True)
    sys.path.insert(0,str(args.research_root/'scripts/evaluate'))
    import nominal_collect as nc
    from scripts.diagnostics.audit_libero_mapping import task_identity,state_hash
    from phase3b_distribution import load_model
    import torch
    from omegaconf import OmegaConf
    from rlinf.envs.sim.libero.libero_env import LiberoEnv
    model=None
    for job in jobs:
        entry_file=args.output/'entries'/(job['job_id']+'.json')
        if entry_file.exists(): check_entry(read(entry_file),job); continue
        if job['job_id'] in reused:
            entry=dict(reused[job['job_id']],provenance='reused_smoke',smoke_entry_sha256=sha(args.smoke/'entries'/(job['job_id']+'.json')))
            check_entry(entry,job); save(entry_file,entry); continue
        if model is None: model,core,model_cfg=load_model(args.research_root,p,'grpo')
        random.seed(0); np.random.seed(0); torch.manual_seed(0); torch.cuda.manual_seed_all(0)
        cfg=OmegaConf.load(args.research_root/'upstream/RLinf/examples/embodiment/config/env/libero_10.yaml')
        cfg.task_suite_name='libero_10'; cfg.total_num_envs=1; cfg.is_eval=True; cfg.auto_reset=False; cfg.ignore_terminations=False
        cfg.reset_gripper_open=False; cfg.use_ordered_reset_state_ids=True; cfg.max_episode_steps=520; cfg.max_steps_per_rollout_epoch=520; cfg.video_cfg.save_video=False; cfg.seed=0
        dest=args.output/'new'/job['job_id']; require(not dest.exists(),f'Incomplete job exists: {dest}; preserve it and use a new run ID')
        dest.mkdir(parents=True)
        protocol=dict(pl['config'],collector_sha256=sha(args.research_root/'scripts/evaluate/nominal_collect.py'),pilot_job={k:v for k,v in job.items() if k!='reference'},
            # The pinned env YAML has an inactive video_base_dir interpolation into
            # runner.logger.log_path. The original collector passes cfg to LiberoEnv
            # without resolving it; save that exact config without forcing the
            # unrelated video field to resolve (video saving is disabled above).
            environment_config=OmegaConf.to_container(cfg,resolve=False),pilot_source_hashes=sources(),model_config=model_cfg,
            collect_episode_source_sha256=hashlib.sha256(inspect.getsource(nc.collect_episode).encode()).hexdigest(),rng_control=pl['rng_control'])
        ph=protocol_hash(protocol); save(dest/'manifest.json',dict(protocol=protocol,protocol_hash=ph))
        env=LiberoEnv(cfg,num_envs=1,seed_offset=0,total_num_processes=1,worker_info=None)
        try:
            mr=job['mapping']; task=env.task_suite.get_task(job['task_id'])
            require(task_identity(task)==tuple(mr[k] for k in ('task_name','problem_folder','bddl_file','init_states_file')),'Live task identity mismatch')
            require(state_hash(env.task_suite.get_task_init_states(job['task_id'])[job['trial_id']])==job['initial_state_sha256'],'Live initial state hash')
            m={k:v for k,v in job['reference'].items() if k not in ('source_dir','policy')}
            m.update(run_id=args.output.name,protocol_hash=ph,policy_seed=job['policy_seed'],collector_sha256=protocol['collector_sha256'])
            path=nc.collect_episode(env,model,suite='libero_10',task_id=job['task_id'],trial_id=job['trial_id'],sampling=True,temperature=1.6,
                policy_seed=job['policy_seed'],metadata=m,run_dir=dest)
        finally: env.env.close()
        entry=dict(job_id=job['job_id'],episode_path=str(path.resolve()),protocol_hash=ph,provenance='new',
                   hashes={n:sha(path/n) for n in ('metadata.json','states.npz','actions.npz','observations.npz')})
        check_entry(entry,job); save(entry_file,entry)
        print(job['job_id'],'success',m['success'],'length',m['episode_length'],flush=True)
    print('Collection finished; run validate before interpreting results.')


def validate(args):
    require(__debug__,'Disable PYTHONOPTIMIZE')
    mf=read(args.output/'manifest.json'); plan_dir=Path(mf['plan']); verify_seal(plan_dir); pl=read(plan_dir/'manifest.json')
    require(sha(plan_dir/'validation.json')==mf['plan_sha256'],'Plan changed')
    jobs=pl['jobs'][:2] if mf['mode']=='smoke' else pl['jobs']
    entries=[]; initial={}; tokens={}; expected={j['job_id'] for j in jobs}
    require({p.stem for p in (args.output/'entries').glob('*.json')}==expected,'Missing/unexpected pilot entries')
    for j in jobs:
        e=read(args.output/'entries'/(j['job_id']+'.json')); m=check_entry(e,j); path=Path(e['episode_path'])
        with np.load(path/'observations.npz',allow_pickle=False) as z:
            first={k:array_hash(z[k][0]) for k in z.files if k!='query_step'}
            require(np.array_equal(z['query_step'],np.arange(m['num_queries'])*8),'Query indexing')
            require(all(np.isfinite(z[k]).all() for k in z.files),'Invalid observation')
        with np.load(path/'states.npz',allow_pickle=False) as z: state={k:array_hash(z[k][0]) for k in z.files}
        key=j['episode_id']; signature=dict(observation=first,state=state,language=m['language'])
        if key in initial: require(signature==initial[key],'Same-state initial observation differs across seeds')
        initial[key]=signature
        # Compare the same initial observation against the audited original nominal run.
        ref=Path(j['reference']['source_dir'])
        old_hashes=read(Path(pl['phase3a'])/'manifest.json')['source_files']
        for name in ('metadata.json','states.npz','actions.npz','observations.npz'):
            require(sha(ref/name)==old_hashes[str(ref/name)],'Original nominal input changed')
        with np.load(ref/'observations.npz',allow_pickle=False) as z:
            require(first=={k:array_hash(z[k][0]) for k in z.files if k!='query_step'},'Pilot/nominal initial observation mismatch')
        with np.load(path/'actions.npz',allow_pickle=False) as z:
            require(all(np.isfinite(z[k]).all() for k in z.files),'Invalid actions')
            tokens[j['job_id']]=array_hash(z['raw_action_tokens'])
        entries.append(dict(e,policy_seed=j['policy_seed'],task_id=j['task_id'],trial_id=j['trial_id'],episode_id=j['episode_id'],success=m['success'],episode_length=m['episode_length']))
    if mf['mode']=='smoke': require(len(set(tokens.values()))==2,'Smoke sampling tokens identical across seeds; inspect before pilot')
    save(args.output/'index.json',dict(entries=entries))
    (args.output/'summary.md').write_text('# Stochastic pilot\n\n'+json.dumps(dict(mode=mf['mode'],episodes=len(entries),successes=sum(e['success'] for e in entries),provenance=dict(__import__('collections').Counter(e['provenance'] for e in entries))),indent=2)+'\n\n'+pl['rng_control']+'\n')
    seal(args.output,dict(status='PASS',episodes=len(entries),same_initial_observation_verified=True,seed_divergence_verified=len(set(tokens.values()))>1,
                         scope='Sampling seed intervention; Python/NumPy environment RNG coupling not excluded'))
    if mf['mode']=='smoke':
        script=Path(__file__).resolve()
        cmd=[sys.executable,'-B',str(script),'run','--plan',str(plan_dir),'--mode','pilot','--smoke',str(args.output.resolve()),'--output',str(args.output.resolve().with_name(args.output.name+'-pilot-a'))]
        print('SMOKE PASS. Pilot A command (50 total, 2 reused + 48 new):\n'+shlex.join(cmd))
    else: print('PILOT_A_INTEGRITY: PASS',len(entries))


def diversity(args):
    verify_seal(args.pilot); mf=read(args.pilot/'manifest.json'); require(mf['mode']=='pilot','Diversity requires full Pilot A')
    pl=read(Path(mf['plan'])/'manifest.json'); p=Path(pl['phase3a']); expert_root=Path(read(p/'manifest.json')['expert_root'])
    a,v=legacy(expert_root); out=fresh(args.output); entries=read(args.pilot/'index.json')['entries']
    require(len(entries)==50,'Expected 50 episodes')
    grouped={}; metrics=[]; pair_rows=[]; stats=[]; files={str(args.pilot/'validation.json'):sha(args.pilot/'validation.json')}
    baseline=read(p/'records.json'); baseline_det={r['episode_id']:r['success'] for r in baseline['grpo']}; baseline_sample={r['episode_id']:r['success'] for r in baseline['stochastic']}
    for e in entries:
        path=Path(e['episode_path'])
        for n,digest in e['hashes'].items(): require(sha(path/n)==digest,'Pilot source changed'); files[str(path/n)]=digest
        with np.load(path/'states.npz',allow_pickle=False) as z: xyz=z['robot0_eef_pos'].copy(); quat=z['robot0_eef_quat'][-1].copy()
        with np.load(path/'actions.npz',allow_pickle=False) as z: act=z['executed_actions'].copy(); tok=z['raw_action_tokens'].copy(); dec=z['decoded_action_chunks'].copy()
        row=dict(job_id=e['job_id'],episode_id=e['episode_id'],policy_seed=e['policy_seed'],success=e['success'],steps=len(act),
                 path_length=a.path_statistics(xyz)['path_length'],gripper_transitions=int(np.count_nonzero(np.diff(act[:,6]))),final_xyz=xyz[-1].tolist(),final_quaternion=quat.tolist())
        metrics.append(row); grouped.setdefault(e['episode_id'],[]).append(dict(row,xyz=xyz,quat=quat,tokens=tok,decoded=dec))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    (out/'figures').mkdir()
    for eid,rr in grouped.items():
        require(len(rr)==5 and len({r['policy_seed'] for r in rr})==5,'Same-state seed grid mismatch')
        fig=plt.figure(figsize=(6,5)); ax=fig.add_subplot(111,projection='3d')
        for r in rr: ax.plot(*r['xyz'].T,label=f"{r['policy_seed']} {'S' if r['success'] else 'F'}")
        ax.set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title=eid); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(out/'figures'/(eid+'.png'),dpi=120); plt.close(fig)
        for x,y in itertools.combinations(rr,2):
            category='success_success' if x['success'] and y['success'] else 'failure_failure' if not x['success'] and not y['success'] else 'success_failure'
            rms=v.variation([x['xyz'],y['xyz']])['world_pairs'][0]
            q=min(len(x['tokens']),len(y['tokens'])); dot=abs(float(np.dot(x['quat']/np.linalg.norm(x['quat']),y['quat']/np.linalg.norm(y['quat']))))
            pair_rows.append(dict(episode_id=eid,seed_a=x['policy_seed'],seed_b=y['policy_seed'],category=category,trajectory_rms_32_m=rms,
                final_position_distance_m=float(np.linalg.norm(x['xyz'][-1]-y['xyz'][-1])),final_orientation_angle_rad=float(2*np.arccos(np.clip(dot,0,1))),
                gripper_transition_difference=abs(x['gripper_transitions']-y['gripper_transitions']),path_length_difference=abs(x['path_length']-y['path_length']),
                token_disagreement_common_query_prefix=float(np.mean(x['tokens'][:q]!=y['tokens'][:q])),decoded_action_rmse_common_query_prefix=float(np.sqrt(np.mean((x['decoded'][:q]-y['decoded'][:q])**2))),
                common_query_count=q,action_comparison='common query ordinal prefix, NOT common observations or semantic events'))
        for category in ('all','success_success','failure_failure','success_failure'):
            pr=[r for r in pair_rows if r['episode_id']==eid and (category=='all' or r['category']==category)]
            subset=[r for r in rr if (r['success'] if category=='success_success' else not r['success'] if category=='failure_failure' else True)]
            stats.append(dict(episode_id=eid,category=category,episodes=5,successes=sum(r['success'] for r in rr),pairs=len(pr),
                baseline_deterministic_success=baseline_det[eid],baseline_stochastic_success=baseline_sample[eid],subset_episodes=len(subset),
                status='DEFINED' if pr else 'UNDEFINED',mean_rms_m=float(np.mean([r['trajectory_rms_32_m'] for r in pr])) if pr else None,
                path_length_std=float(np.std([r['path_length'] for r in subset])) if len(subset)>1 else None,final_xyz_std=np.std([r['xyz'][-1] for r in subset],axis=0).tolist() if len(subset)>1 else None))
    csv_save(out/'per_episode.csv',metrics); csv_save(out/'pairwise_diversity.csv',pair_rows); csv_save(out/'same_state_summary.csv',stats)
    save(out/'manifest.json',dict(source_hashes=sources(),input_hashes=files,reused_functions=['analysis.path_statistics','task_plots.variation (32-point normalized-index RMS)'],
        interpretation='Different normalized indices do not imply matching events. Unsuccessful motion is not useful diversity. Successful pairwise diversity UNDEFINED with <=1 success. Sampling/global CPU RNG coupling remains.'))
    (out/'summary.md').write_text('# Same-state diversity\n\n'+json.dumps(stats,indent=2)+'\n\nDescriptive path differences, not multiple learned strategies.\n')
    for f,digest in files.items(): require(sha(f)==digest,'Pilot source changed during diversity analysis')
    seal(out,dict(status='PASS',episodes=50,states=10,pairs=len(pair_rows))); print(out/'summary.md')


def main():
    p=argparse.ArgumentParser(description=__doc__); sub=p.add_subparsers(dest='stage',required=True)
    a=sub.add_parser('plan'); a.add_argument('--phase3a',type=Path,required=True); a.add_argument('--audit',type=Path,required=True); a.add_argument('--research-root',type=Path,default=ROOT); a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('run'); a.add_argument('--plan',type=Path,required=True); a.add_argument('--mode',choices=('smoke','pilot'),required=True); a.add_argument('--smoke',type=Path); a.add_argument('--output',type=Path,required=True); a.add_argument('--research-root',type=Path,default=ROOT); a.add_argument('--resume',action='store_true')
    a=sub.add_parser('validate'); a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('diversity'); a.add_argument('--pilot',type=Path,required=True); a.add_argument('--output',type=Path,required=True)
    a=sub.add_parser('repeat-plan'); a.add_argument('--phase3a',type=Path,required=True); a.add_argument('--mapping',type=Path)
    a.add_argument('--output',type=Path,required=True); a.add_argument('--suite',choices=REPEAT_SUITES,default='libero_10')
    a.add_argument('--task-ids',type=int,nargs='+',default=[3,8]); a.add_argument('--trial-ids',type=int,nargs='+',default=list(range(5)))
    a.add_argument('--policies',choices=REPEAT_POLICIES,nargs='+',default=list(REPEAT_POLICIES))
    a.add_argument('--sampling-seeds',type=int,nargs='+'); a.add_argument('--repeats',type=int)
    a.add_argument('--temperature',type=float,default=1.6); a.add_argument('--top-p',type=float,default=1.0); a.add_argument('--top-k',type=int,default=-1)
    a=sub.add_parser('repeat-preflight'); a.add_argument('--plan',type=Path,required=True); a.add_argument('--audit',type=Path,required=True)
    a.add_argument('--research-root',type=Path,default=ROOT)
    a=sub.add_parser('repeat-run'); a.add_argument('--plan',type=Path,required=True); a.add_argument('--audit',type=Path,required=True)
    a.add_argument('--output',type=Path,required=True); a.add_argument('--research-root',type=Path,default=ROOT)
    a.add_argument('--smoke',action='store_true'); a.add_argument('--resume',action='store_true')
    a=sub.add_parser('repeat-validate'); a.add_argument('--output',type=Path,required=True)
    a.add_argument('--partial',action='store_true'); a.add_argument('--jobs',nargs='+')
    args=p.parse_args()
    dispatch={'repeat-plan':repeat_plan,'repeat-preflight':repeat_preflight,
              'repeat-run':repeat_run,'repeat-validate':repeat_validate}
    (dispatch[args.stage] if args.stage in dispatch else globals()[args.stage])(args)

if __name__=='__main__': main()
