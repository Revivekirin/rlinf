"""Phase 3A: saved-data-only paired outcomes, trajectories, and replay inputs.

No RLinf, simulator, torch, checkpoint loader, or model is imported.
Every derived stage creates new files; no refresh/overwrite option exists.
"""
from __future__ import annotations
import argparse
import collections
import csv
import importlib.util
import json
import os
from pathlib import Path
import sys
import textwrap
from datetime import datetime, timezone
import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase3a_core import (SUITES, LABELS, COMPARISONS, CATEGORIES, REVISION, CONDITIONS,
    require, sha, array_hash, protocol_hash, legacy, identity, match_records,
    outcome_stats, canonical, choose_cases, select_queries, category)

ROOT = Path(__file__).resolve().parents[2]


def read(path):
    return json.loads(Path(path).read_text())


def save(path, data):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x') as f:
        json.dump(data, f, indent=2, sort_keys=True, allow_nan=False)
        f.write('\n')


def csv_save(path, rows):
    path = Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('x',newline='') as f:
        if rows:
            writer = csv.DictWriter(f,fieldnames=list(rows[0]))
            writer.writeheader()
            for row in rows:
                writer.writerow({k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()})


def remember(files, path):
    path = Path(path).resolve(); require(path.is_file(), f'Missing input: {path}')
    files[str(path)] = sha(path)


def source_modules(root):
    return sorted((root/'experiments/dataset_analysis').glob('*.py')) + sorted((root/'experiments/shot_scaling').glob('*.py')) + sorted((root/'experiments/fewshot_sft').glob('*.py')) + sorted((root/'experiments/dataset_analysis/evidence_sources').glob('*.py'))


def expert_regression(expert_root, results, output):
    """Recalculate original formulas on original saved expert sequences; no TFDS decoding."""
    require(not output.exists(), f'Output exists: {output}')
    a, v = legacy(expert_root)
    from experiments.dataset_analysis.stage1 import validate as legacy_validate
    from experiments.dataset_analysis.semantics import sanity
    from experiments.shot_scaling.core import digest as legacy_digest
    output.mkdir(parents=True)
    files = {}
    for p in source_modules(expert_root): remember(files,p)
    for name in ('manifest.json','validation.json','report.md','episode_features.csv','task_variation.csv','analysis.json'):
        remember(files,results/name)
    old = read(results/'analysis.json'); manifest = read(results/'manifest.json')
    require(manifest['coordinate_semantics']['orientation_representation']=='axis_angle_radians', 'Expert orientation units')
    with (results/'episode_features.csv').open() as f: feature_rows=list(csv.DictReader(f))
    with (results/'task_variation.csv').open() as f: variation_rows=list(csv.DictReader(f))
    require(len(feature_rows)==len({(r['suite'],r['episode_id']) for r in feature_rows}), 'Duplicate expert features')
    features={(r['suite'],r['episode_id']):r for r in feature_rows}
    tolerance=dict(rtol=1e-7,atol=1e-9)
    checks={}; metric_rows=[]; task_rows=[]
    for suite, expected in (('goal',(428,52042)),('spatial',(432,52970)),('long',(379,101469))):
        path=results/'trajectory_data'/f'{suite}.json'; remember(files,path)
        raw=read(path); count=legacy_validate(raw)
        require(legacy_digest(raw['episodes']) == raw['episodes_sha256'], 'Expert episode digest mismatch')
        require(count==expected, f'Expert count mismatch: {suite}: {count}')
        numerical=sanity(raw['episodes']); require(numerical['finite'], 'Expert finite check')
        prior={e['episode_id']:e for e in old['suites'][suite]['episodes']}
        require(len(prior)==len(raw['episodes']), 'Expert saved analysis count')
        grouped=collections.defaultdict(list)
        for e in raw['episodes']:
            state=np.asarray(e['observation_state']); x=state[:,:3]; st=a.path_statistics(x)
            prev=prior[e['episode_id']]; f=features[(suite,e['episode_id'])]
            np.testing.assert_allclose(x,prev['xyz'],**tolerance)
            require(len(x)==e['steps']==prev['steps']==int(f['steps']), 'Expert steps regression')
            for key in st: np.testing.assert_allclose(st[key],prev['physical'][key],**tolerance)
            np.testing.assert_allclose(st['path_length'],float(f['physical_path_m']),**tolerance)
            np.testing.assert_allclose(st['net_displacement'],float(f['net_displacement_m']),**tolerance)
            require(a.align_path(x).shape==(32,3),'Resampled shape')
            grouped[(e['split'],e['task'])].append(x)
            metric_rows.append(dict(source='expert',suite=suite,task=e['task'],episode_id=e['episode_id'],
                steps=e['steps'],**st,gripper_transitions=len(prev['gripper_events']),
                endpoint_definition='last retained pre-action state; no post-final state',
                initial_state_match='UNKNOWN; unpaired',source_file=str(path)))
        for (split,task),paths in grouped.items():
            values=v.variation(paths)['world_pairs']
            ref=[r for r in variation_rows if (r['suite'],r['split'],r['task'])==(suite,split,task)]
            require(len(ref)==1, 'Expert task variation identity')
            np.testing.assert_allclose(np.mean(values),float(ref[0]['pairwise_rms_mean_m']),**tolerance)
            np.testing.assert_allclose(np.median(values),float(ref[0]['pairwise_rms_median_m']),**tolerance)
            require(len(values)==int(ref[0]['pair_count']), 'Expert pair count')
            old_pairs=[r['aligned_xyz_rms_m'] for r in old['suites'][suite]['physical_pairs'] if (r['split'],r['task'])==(split,task)]
            np.testing.assert_allclose(sorted(values),sorted(old_pairs),**tolerance)
            task_rows.append(dict(source='expert',suite=suite,task=task,n=len(paths),
                pair_count=len(values),rms_mean=float(np.mean(values)),rms_median=float(np.median(values))))
        checks[suite]=dict(episodes=count[0],transitions=count[1],regression='PASS',sanity=numerical['status'])
        print('Expert regression:',suite,count,flush=True)
    for path,digest in files.items(): require(sha(path)==digest, f'Expert source changed: {path}')
    csv_save(output/'metrics.csv',metric_rows); csv_save(output/'variation.csv',task_rows)
    result=dict(status='PASS',suites=checks,tolerance=tolerance,source_files=files,
        reused=['analysis.path_statistics','analysis.align_path','task_plots.variation','stage1.validate','semantics.sanity'],
        coordinate_semantics=manifest['coordinate_semantics'],
        scope='Regression on extracted real sequences, not a fresh TFRecord decode; initial states unpaired.')
    save(output/'regression.json',result)
    return result


def semantics(research, expert_root, files):
    """Read source evidence without importing any robot/simulator module."""
    candidates=[]
    for site in (research/'upstream/RLinf').glob('.venv*/lib/python*/site-packages'):
        candidates.extend(site.glob('robosuite/robots/*/*.py'))
        candidates.extend(site.glob('robosuite/robots/*.py'))
    evidence=[]; object_evidence=[]
    for site in (research/'upstream/RLinf').glob('.venv*/lib/python*/site-packages'):
        for path in site.glob('libero/**/bddl_base_domain.py'):
            text=path.read_text()
            if 'def obj_pos(' in text and 'body_xpos' in text and 'def obj_quat(' in text:
                remember(files,path)
                object_evidence.append(str(path))
    for path in candidates:
        text=path.read_text()
        if 'def eef_pos(' in text and 'site_xpos' in text.split('def eef_pos(',1)[1].split('def ',1)[0] and 'def eef_quat(' in text and 'get_body_xquat' in text.split('def eef_quat(',1)[1].split('def ',1)[0] and 'xyzw' in text.split('def eef_quat(',1)[1].split('def ',1)[0]:
            remember(files,path)
            lines=text.splitlines()
            excerpts=[{'line':i+1,'text':'\n'.join(lines[max(0,i-2):i+4])}
                      for i,line in enumerate(lines) if 'def eef_pos(' in line or 'def eef_quat(' in line or 'def gripper_qpos(' in line]
            evidence.append(dict(path=str(path),excerpts=excerpts))
    collector=research/'scripts/evaluate/nominal_collect.py'
    remember(files,collector)
    # Installed simulator source is required for the active-run frame label.
    status='SOURCE_DOCUMENTED' if evidence else 'UNKNOWN'
    return dict(status=status,frame='simulator_world' if evidence else 'UNKNOWN',
        position_units='m (simulation convention, not calibrated)' if evidence else 'UNKNOWN',
        orientation='quaternion_xyzw; dimensionless' if evidence else 'UNKNOWN',
        gripper_state='joint qpos; joint units require model-specific confirmation',
        gripper_command='executed -sign(2*decoded-1); polarity not independently verified; grasp success UNKNOWN',
        timing='state[k] before action[k]; state[k+1] after; query_step indexes pre-action state; 15 settling steps excluded',
        object_pose='named *_pos/*_quat saved simulator sensors; role/target mapping requires explicit review',
        evidence=evidence,object_evidence=object_evidence,object_frame_status='SOURCE_DOCUMENTED' if object_evidence else 'UNKNOWN',wall_clock_timestamp='NOT AVAILABLE')


def preflight(args):
    out=args.output.resolve()
    require(not out.exists(), f'Output exists: {out}; use a new run ID')
    runs={l:Path(getattr(args,l)).resolve() for l in LABELS}
    for path in [*runs.values(),args.expert_root,args.expert_results]:
        require(not out.is_relative_to(Path(path).resolve()), 'Output must not be inside source data/repo')
    out.mkdir(parents=True)
    files={}; remember(files,args.mapping)
    mapping_data=read(args.mapping)
    mapping={(r['suite'],r['subset_task_id']):r for r in mapping_data['tasks']}
    require(len(mapping)==30==len(mapping_data['tasks']), 'Expected 30 unique mapping rows')
    require(len({r['libero_130_task_id'] for r in mapping.values()})==30,'Duplicate global task ID')
    sys.path.insert(0,str(args.research_root/'scripts/evaluate'))
    from nominal_validate import validate_episode
    require(__debug__, 'PYTHONOPTIMIZE disables legacy validator assertions')
    records={}; protocols={}; schema=[]; partials={}
    for name in ('nominal_validate.py','nominal_storage.py'):
        remember(files,args.research_root/'scripts/evaluate'/name)
    for label,run in runs.items():
        remember(files,run/'manifest.json'); mfest=read(run/'manifest.json'); p=mfest['protocol']
        require(protocol_hash(p)==mfest['protocol_hash'],'Protocol hash mismatch')
        require(p['mapping_sha256']==protocol_hash(mapping_data),'Mapping hash mismatch')
        require(p['condition']==CONDITIONS[label] and p['checkpoint_revision']==REVISION[label], 'Wrong condition/checkpoint')
        require(p['do_sample'] is (label=='stochastic') and p['lora_applied'] is False,'Wrong decoding/LoRA')
        require(p['temperature']==(1.6 if label=='stochastic' else 1.0),'Wrong temperature')
        require(p['upstream_commit']=='e05f71e80d14ab4b55e5027ee739cfbcc6b12b9d','Wrong upstream revision')
        require(p['trials_per_task']==5 and p['horizon']==520 and p['action_chunks']==8,'Wrong protocol dimensions')
        protocols[label]=p; records[label]=[]; partials[label]=[]
        require(p['checkpoint_repo']=='RLinf/RLinf-OpenVLAOFT-LIBERO-130'+('-Base-Lora' if label=='sft' else ''),'Checkpoint repository mismatch')
        for path in sorted((run/'rollouts').iterdir()):
            if path.name.startswith('.partial-'):
                partials[label].append(str(path)); continue
            for name in ('metadata.json','states.npz','actions.npz','observations.npz'): remember(files,path/name)
            m=validate_episode(path,mfest['protocol_hash']); identity(m,mapping)
            require(m['run_id']==run.name and m['complete'] is True and type(m['success']) is bool,'Invalid completed metadata')
            for k in ('checkpoint_repo','checkpoint_revision','sampling_mode','temperature','lora_applied','horizon','action_chunks','unnorm_key','upstream_commit','collector_sha256'):
                require(m[k]==p[k],f'Metadata/protocol mismatch: {k}')
            T,Q=m['episode_length'],m['num_queries']
            require(0<T<=520 and Q==(T+7)//8,'Invalid episode/query length')
            require(m['termination_reason']==('success' if m['success'] else 'horizon'),'Terminal mismatch')
            require(m['success'] or T==520,'Early failed episode')
            require(m['simulator_seed']==p['seed'] and m['policy_seed']==100000*SUITES.index(m['suite'])+100*m['task_id']+m['trial_id'],'Seed mismatch')
            require(m['wait_steps']==p['num_steps_wait'] and bool(m['language']),'Missing wait/language metadata')
            for filename in ('states.npz','actions.npz','observations.npz'):
                with np.load(path/filename,allow_pickle=False) as data:
                    required={'states.npz':('robot0_gripper_qpos',), 'actions.npz':(),
                              'observations.npz':('proprio_policy',)}[filename]
                    require(all(k in data.files for k in required),'Missing required arrays')
                    for key in data.files:
                        a=data[key]
                        require(a.dtype.kind in 'biuf' and np.isfinite(a).all(),f'Invalid array: {path}/{key}')
                        if filename=='observations.npz':
                            require(len(a)==Q,'Observation query length')
                            if 'image' in key: require(a.ndim==4 and a.shape[-1]==3 and a.dtype==np.uint8,'Image format')
                        schema.append(dict(policy=label,episode_id=m['episode_id'],file=filename,key=key,shape=list(a.shape),dtype=str(a.dtype),source=str(path/filename)))
                    if filename=='actions.npz':
                        require(data['raw_action_tokens'].dtype.kind in 'iu','Token dtype')
                        require(np.array_equal(data['executed_query_index'],np.arange(T)//8),'Chunk indices')
                        require(np.array_equal(data['executed_action_slot'],np.arange(T)%8),'Action slots')
                    if filename=='observations.npz': require(np.array_equal(data['query_step'],np.arange(Q)*8),'Query steps')
            records[label].append(dict(m,source_dir=str(path),policy=label))
            if len(records[label])%25==0: print('Audited',label,len(records[label]),flush=True)
        if (run/'summary.json').exists(): remember(files,run/'summary.json')
    # Only explicit model/decoding fields may differ, including the known smoke-selector source change.
    ignored={'condition','sampling_mode','temperature','do_sample','checkpoint_repo','checkpoint_path','checkpoint_revision','collector_sha256','model_config'}
    base=protocols['sft']
    for label in LABELS[1:]:
        p=protocols[label]
        require(all(base.get(k)==p.get(k) for k in base.keys()|p.keys() if k not in ignored),'Unexpected protocol difference')
        require({k:v for k,v in base['model_config'].items() if k!='model_path'} ==
                {k:v for k,v in p['model_config'].items() if k!='model_path'}, 'Model/processor config mismatch')
    match_records(records,mapping)
    evidence=semantics(args.research_root,args.expert_root,files)
    for path in Path(__file__).parent.glob('*.py'): remember(files,path)
    for path in source_modules(args.expert_root): remember(files,path)
    if args.object_roles: remember(files,args.object_roles)
    manifest=dict(created_utc=datetime.now(timezone.utc).isoformat(),inputs={k:str(v) for k,v in runs.items()},
        research_root=str(args.research_root.resolve()),mapping=str(args.mapping.resolve()),
        expert_root=str(args.expert_root.resolve()),expert_results=str(args.expert_results.resolve()),
        object_roles=str(args.object_roles.resolve()) if args.object_roles else None,
        source_files=files,protocols=protocols,seed=args.seed,iterations=args.iterations,
        semantics=evidence,analysis_code={str(p):sha(p) for p in Path(__file__).parent.glob('*.py')})
    for row in schema:
        k,f=row['key'],row['file']
        row.update(category='state' if f=='states.npz' else 'action' if f=='actions.npz' else 'observation',
                   frame='not applicable',units='not applicable',missing=False,
                   timing='T+1 measured simulator states' if f=='states.npz' else 'query or executed-step indices; see key')
        if f=='states.npz':
            if k=='robot0_eef_pos': row.update(frame=evidence['frame'],units=evidence['position_units'])
            elif k=='robot0_eef_quat': row.update(frame=evidence['frame'],units='dimensionless quaternion XYZW')
            elif k.endswith('_pos') and not k.startswith('robot') and '_to_' not in k:
                row.update(frame='simulator_world' if evidence['object_evidence'] else 'UNKNOWN',units='m (simulation convention)' if evidence['object_evidence'] else 'UNKNOWN')
            elif k.endswith('_quat') and not k.startswith('robot') and '_to_' not in k:
                row.update(frame='simulator_world' if evidence['object_evidence'] else 'UNKNOWN',units='dimensionless quaternion; inspect sensor evidence')
            else: row.update(frame='UNKNOWN / sensor-specific',units='native sensor units; not independently authenticated')
        elif f=='actions.npz' and k in ('decoded_action_chunks','executed_actions'):
            row.update(frame='controller action space, not world EEF coordinates',units='action command; translation/rotation commands are not measured displacement')
        elif 'image' in k or k=='pixel_values':
            row.update(frame='camera/image tensor',units='uint8 intensity' if 'image' in k else 'processed model input')
        elif 'step' in k or 'index' in k or 'slot' in k:
            row.update(units='discrete index, no physical timestamp')
    metadata_inventory=[]
    for label,items in records.items():
        for key in sorted(set().union(*(r.keys() for r in items))):
            metadata_inventory.append(dict(policy=label,key=key,missing=sum(key not in r for r in items),
                types=sorted({type(r[key]).__name__ for r in items if key in r})))
    save(out/'metadata_inventory.json',metadata_inventory)
    save(out/'manifest.json',manifest); save(out/'records.json',records)
    save(out/'schema_inventory.json',schema)
    save(out/'integrity.json',dict(status='PASS',episodes={k:len(v) for k,v in records.items()},excluded_partials=partials,
        source_hashes_verified=True,semantics=evidence,scope='Fresh structural audit; no simulator or model execution'))


def paired(out):
    manifest=read(out/'manifest.json'); records=read(out/'records.json')
    mapping={(r['suite'],r['subset_task_id']):r for r in read(manifest['mapping'])['tasks']}
    indices=match_records(records,mapping)
    rows=[]; stats={}; overall=[]; suite_rows=[]; task_rows=[]
    for comparison,left,right,expected_delta in COMPARISONS:
        group=[]
        for key in sorted(indices[left]):
            a,b=indices[left][key],indices[right][key]
            row=dict(comparison=comparison,left=left,right=right,episode_id=a['episode_id'],suite=a['suite'],task_id=a['task_id'],
                task_name=a['task_name'],global_task_id=a['libero_130_task_id'],trial_id=a['trial_id'],initial_state_sha256=a['initial_state_sha256'],
                bddl_identity=list(key[3:6]),environment_seed=a['simulator_seed'],policy_seed=a['policy_seed'],
                left_success=a['success'],right_success=b['success'],category=category(a['success'],b['success']),
                left_length=a['episode_length'],right_length=b['episode_length'],left_source=a['source_dir'],right_source=b['source_dir'])
            group.append(row)
        st=outcome_stats(group,manifest['seed'],manifest['iterations'])
        require(st['gained']-st['lost']==expected_delta, f'{comparison}: reported success delta mismatch')
        stats[comparison]=st; overall.append(dict(comparison=comparison,**st)); rows+=group
        for suite in SUITES:
            subset=[r for r in group if r['suite']==suite]
            suite_rows.append(dict(comparison=comparison,suite=suite,**outcome_stats(subset,manifest['seed'],manifest['iterations'])))
            for t in range(10):
                rr=[r for r in subset if r['task_id']==t]; c=collections.Counter(r['category'] for r in rr)
                task_rows.append(dict(comparison=comparison,suite=suite,task_id=t,task_name=rr[0]['task_name'],n=len(rr),
                    **{k:c[k] for k in CATEGORIES},difference_pp=100*(c['gained']-c['lost'])/len(rr)))
    csv_save(out/'paired/paired_episodes.csv',rows); save(out/'paired/pairs.json',rows)
    save(out/'paired/overall_summary.json',overall); save(out/'paired/statistics.json',stats)
    csv_save(out/'paired/suite_summary.csv',suite_rows); csv_save(out/'paired/task_summary.csv',task_rows)


def cases(out):
    pairs=read(out/'paired/pairs.json'); records=read(out/'records.json')
    long=[r for r in pairs if r['comparison']=='primary' and r['suite']=='libero_10']
    lookup={l:{r['episode_id']:r for r in records[l]} for l in LABELS}
    for r in long:
        r['conditions']={l:{k:lookup[l][r['episode_id']][k] for k in ('success','episode_length','source_dir','checkpoint_revision','sampling_mode')} for l in LABELS}
        available={}
        for l in LABELS:
            with np.load(Path(r['conditions'][l]['source_dir'])/'states.npz',allow_pickle=False) as s:
                available[l]=dict(eef=True,gripper='robot0_gripper_qpos' in s.files,
                    object_keys=[k for k in s.files if k.endswith('_pos') and not k.startswith('robot') and '_to_' not in k])
        r['availability']=available
    selected=choose_cases(long)
    csv_save(out/'cases/long_all_cases.csv',long); save(out/'cases/selected_cases.json',selected)
    save(out/'cases/case_selection_manifest.json',dict(rule='Within each category, maximize unused task then unused initial-state index; SHA256 episode ID tie-break. At most 3 per category. No visual or outcome-strength ranking.',
        candidate_count=len(long),selected_ids=[r['episode_id'] for r in selected],
        unselected_ids=[r['episode_id'] for r in long if r not in selected],
        categories={k:sum(r['category']==k for r in selected) for k in CATEGORIES},
        pattern_selection='Patterns are inspected after selection; not a hypothesis-dependent selection criterion'))


def load_trajectory(m):
    p=Path(m['source_dir'])
    with np.load(p/'states.npz',allow_pickle=False) as s, np.load(p/'actions.npz',allow_pickle=False) as a:
        return canonical({k:s[k] for k in s.files},{k:a[k] for k in a.files},m)


def trajectories(out):
    mf=read(out/'manifest.json'); require(mf['semantics']['status']=='SOURCE_DOCUMENTED', 'Active simulator frame source unavailable; inspect installed robosuite source before physical plots')
    a,v=legacy(Path(mf['expert_root'])); records=read(out/'records.json')
    rows=[]; variation_rows=[]; event_rows=[]; role_inventory=[]
    roles=read(mf['object_roles']) if mf['object_roles'] else {}
    for label in LABELS:
        groups=collections.defaultdict(list)
        for m in records[label]:
            c=load_trajectory(m); x=c['xyz']; st=a.path_statistics(x)
            orientation=np.stack([a.quat2axisangle(q.copy()) for q in c['orientation']])
            dest=out/'trajectories/canonical'/label; dest.mkdir(parents=True,exist_ok=True)
            with (dest/(m['episode_id']+'.npz')).open('xb') as f:
                np.savez_compressed(f,xyz=x,quaternion_xyzw=c['orientation'],axis_angle_radians=orientation,
                    gripper_state=c['gripper'],gripper_command=c['command'],resampled_xyz_32=a.align_path(x),
                    **{'object__'+k+'__pos':z for k,z in c['objects'].items()},
                    **{'object__'+k+'__quat':z for k,z in c['object_quaternions'].items()},
                    step_displacement=np.linalg.norm(np.diff(x,axis=0),axis=1))
            pre=a.path_statistics(x[:-1]); delta=np.linalg.norm(np.diff(x,axis=0),axis=1)
            groups[(m['suite'],m['task_id'])].append(x)
            rows.append(dict(policy=label,episode_id=m['episode_id'],suite=m['suite'],task_id=m['task_id'],task_name=m['task_name'],
                trial_id=m['trial_id'],success=m['success'],steps=m['episode_length'],**st,
                pre_action_path_length=pre['path_length'],pre_action_last_xyz=pre['last_recorded_xyz'],
                stationary_steps_lt_1e_4_m=int((delta<1e-4).sum()),stationary_step_indices=np.flatnonzero(delta<1e-4).tolist(),step_displacement_mean=float(delta.mean()),
                gripper_transitions=len(c['events']),object_names=list(c['objects']),source=m['source_dir']))
            for step in c['events']:
                event_rows.append(dict(policy=label,episode_id=m['episode_id'],step=int(step),command=float(c['command'][step]),
                    eef_xyz=x[step].tolist(),object_distances={k:float(np.linalg.norm(z[step]-x[step])) for k,z in c['objects'].items()},
                    eef_movement_previous_8_steps=float(delta[max(0,step-8):step].sum()),
                    eef_movement_next_8_steps=float(delta[step:step+8].sum()),grasp_success='UNKNOWN',polarity='numeric command transition; unverified open/close'))
            role_inventory.append(dict(policy=label,episode_id=m['episode_id'],task_name=m['task_name'],objects=list(c['objects']),
                roles=roles.get(m['task_name']),status='REVIEW_REQUIRED' if m['task_name'] not in roles else 'USER_MAPPED'))
        for (suite,t),paths in groups.items():
            values=v.variation(paths)['world_pairs']
            variation_rows.append(dict(policy=label,suite=suite,task_id=t,n=len(paths),pairs=len(values),rms_mean=float(np.mean(values)),rms_median=float(np.median(values)),definition='32 normalized-index points, within task; full T+1 path; descriptive only'))
    csv_save(out/'trajectories/metrics.csv',rows); csv_save(out/'trajectories/variation.csv',variation_rows)
    save(out/'trajectories/gripper_events.json',event_rows); save(out/'trajectories/object_role_inventory.json',role_inventory)
    lookup={l:{r['episode_id']:r for r in records[l]} for l in LABELS}
    for case in read(out/'cases/selected_cases.json'):
        plot_case(out,case,{l:lookup[l][case['episode_id']] for l in LABELS},a,roles)


def plot_case(out,case,metadata,metrics,roles):
    # Legacy stage1.plot is coupled to its expert bundle; reuse its measured-path
    # conventions (start circle/end cross, equal metric axes), with a new schema adapter.
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig=plt.figure(figsize=(18,14))
    axes=[fig.add_subplot(3,3,1,projection='3d')]+[fig.add_subplot(3,3,i) for i in range(2,10)]
    styles=dict(sft=('#377eb8','-'),grpo=('#e41a1c','-'),stochastic=('#4daf4a','--'))
    object_rows=[]; object_available=False; target_available=False
    for label,m in metadata.items():
        c=load_trajectory(m); x=c['xyz']; color,style=styles[label]
        name=f"{label} / {m['sampling_mode']} / {'success' if m['success'] else 'fail'} / T={m['episode_length']}"
        axes[0].plot(*x.T,color=color,ls=style,label=name)
        axes[0].scatter(*x[0],color=color,marker='o'); axes[0].scatter(*x[-1],color=color,marker='x')
        for ax,(j,k) in zip(axes[1:4],((0,1),(0,2),(1,2))):
            ax.plot(x[:,j],x[:,k],color=color,ls=style)
            ax.set(xlabel='XYZ'[j]+' (m)',ylabel='XYZ'[k]+' (m)'); ax.set_aspect('equal',adjustable='datalim')
        for j in range(3):
            axes[4].plot(x[:,j],color=color,ls=('-','--',':')[j],label=label+' '+'XYZ'[j])
        axes[5].step(np.arange(len(c['command'])),c['command'],where='post',color=color,ls=style,label=label+' command')
        # Separate axis for qpos: command is unitless and joint qpos units are not authenticated.
        qax=getattr(axes[5],'_qpos_axis',None)
        if qax is None:
            qax=axes[5].twinx(); axes[5]._qpos_axis=qax; qax.set_ylabel('Finger joint qpos (native units)')
        for j in range(2): qax.plot(c['gripper'][:,j],color=color,ls=':',alpha=.5)
        for obj,z in c['objects'].items() if read(out/'manifest.json')['semantics']['object_frame_status']=='SOURCE_DOCUMENTED' else []:
            distance=np.linalg.norm(z-x,axis=1); object_available=True
            axes[6].plot(distance,color=color,alpha=.5,label=label+': '+obj)
            object_rows.append(dict(policy=label,object=obj,displacement=float(np.linalg.norm(z[-1]-z[0])),
                distance_eef_min=float(distance.min()),distance_eef_initial=float(distance[0]),distance_eef_final=float(distance[-1])))
        role=roles.get(m['task_name'])
        if role:
            require(read(out/'manifest.json')['semantics']['object_frame_status']=='SOURCE_DOCUMENTED','Object frame source not verified')
            require(role.get('evidence'), 'Object role map needs evidence')
            obj,target=role['object'],role['target']
            require(obj in c['objects'] and target in c['objects'],'Mapped object/target missing')
            axes[7].plot(np.linalg.norm(c['objects'][obj]-c['objects'][target],axis=1),color=color,ls=style,label=label)
            target_available=True
        aligned=metrics.align_path(x)
        for j in range(3): axes[8].plot(np.linspace(0,1,32),aligned[:,j],color=color,ls=('-','--',':')[j])
    axes[0].set(xlabel='X (m)',ylabel='Y (m)',zlabel='Z (m)',title='Measured EEF world path')
    axes[0].legend(fontsize=7)
    for ax in axes[4:8]: ax.set_xlabel('Simulator step')
    axes[4].set(title='EEF position',ylabel='World coordinate (m)'); axes[4].legend(fontsize=6,ncol=3)
    axes[5].set(title='Gripper state / executed command',ylabel='Command (unitless)'); axes[5].legend(fontsize=7)
    axes[6].set(title='Named object sensor–EEF distances',ylabel='Distance (m)')
    if object_available: axes[6].legend(fontsize=5,ncol=2)
    else: axes[6].text(.1,.5,'NOT AVAILABLE',transform=axes[6].transAxes)
    axes[7].set(title='Object–target distance',ylabel='Distance (m)')
    if not target_available: axes[7].text(.05,.5,'ROLE MAPPING UNKNOWN\nSee object_role_inventory.json',transform=axes[7].transAxes)
    axes[8].set(title='32-point linear interpolation: XYZ',xlabel='Normalized index (not time)',ylabel='World coordinate (m)')
    title=f"{case['episode_id']} | {case['category']} | "+case['task_name']
    fig.suptitle(textwrap.fill(title,110),fontsize=12)
    fig.text(.02,.015,'Source: each condition source_dir/states.npz and actions.npz (manifest hashes). World/m: installed source documented, not calibrated.\nMeasured simulator steps except normalized panel. Gripper transitions are not proof of grasp. Expert is not overlaid: init-state correspondence unverified.',fontsize=9)
    fig.tight_layout(rect=(0,.06,1,.94))
    directory=out/'trajectories/figures'; directory.mkdir(parents=True,exist_ok=True)
    path=directory/(case['episode_id']+'.png'); require(not path.exists(),'Figure exists')
    fig.savefig(path,dpi=140); plt.close(fig)
    save(directory/(case['episode_id']+'.json'),dict(case=case,sources=metadata,objects=object_rows,frame='simulator_world',units='m',expert_overlay=False))
    print('Figure:',path.name,flush=True)


def expert(out):
    mf=read(out/'manifest.json')
    result=expert_regression(Path(mf['expert_root']),Path(mf['expert_results']),out/'expert_reference/regression')
    # Match exact task names, never indices or fuzzy instruction matching.
    with (out/'expert_reference/regression/metrics.csv').open() as f: er=list(csv.DictReader(f))
    with (out/'trajectories/metrics.csv').open() as f: pr=list(csv.DictReader(f))
    # Reuse original 3D/XY/XZ plotting on a filtered copy of its own schema.
    selected_tasks={r['task_name'] for r in read(out/'cases/selected_cases.json')}
    bundle=read(Path(mf['expert_results'])/'analysis.json')
    for suite,d in bundle['suites'].items():
        keep=selected_tasks if suite=='long' else set()
        d['episodes']=[e for e in d['episodes'] if e['task'] in keep]
        d['tasks']['train']={k:v for k,v in d['tasks']['train'].items() if k in keep}
        d['physical_pairs']=[r for r in d['physical_pairs'] if r['task'] in keep]
    filtered=out/'expert_reference/selected_expert_bundle.json'
    save(filtered,bundle)
    _,legacy_plots=legacy(Path(mf['expert_root']))
    legacy_plots.generate(filtered,out/'expert_reference/figures')
    # Correct the legacy generator's fixed all-30-tasks caption in this NEW derived index.
    index=out/'expert_reference/figures/task_trajectory_initial_index.md'
    index.write_text(index.read_text().replace('All 30 tasks; all train expert episodes.',
        'Selected Long tasks; all matching train expert episodes. Unpaired reference.'))
    comparisons=[]; aliases=dict(zip(SUITES,('goal','spatial','long')))
    for label in LABELS:
        for suite in SUITES:
            for task in sorted({r['task_name'] for r in pr if r['suite']==suite}):
                policy=[r for r in pr if (r['policy'],r['suite'],r['task_name'])==(label,suite,task)]
                ex=[r for r in er if r['suite']==aliases[suite] and r['task']==task]
                row=dict(policy=label,suite=suite,task=task,expert_n=len(ex),policy_n=len(policy),paired=False,
                    compatibility='source-documented world meters; unpaired descriptive; retained RLDS steps vs all simulator steps',
                    endpoint='RLDS last pre-action vs policy last pre-action for comparison',
                    expert_path_mean=float(np.mean([float(r['path_length']) for r in ex])) if ex else None,
                    policy_pre_action_path_mean=float(np.mean([float(r['pre_action_path_length']) for r in policy])),
                    expert_gripper_event_definition='raw command threshold 0.5; polarity unverified',
                    policy_gripper_event_definition='executed command changes; polarity unverified')
                for prefix,group,start,end in (('expert',ex,'initial_xyz','last_recorded_xyz'),('policy',policy,'initial_xyz','pre_action_last_xyz')):
                    for key in (start,end):
                        vals=np.array([json.loads(r[key]) for r in group])
                        row[prefix+'_'+key+'_mean']=vals.mean(0).tolist() if len(vals) else None
                        row[prefix+'_'+key+'_std']=vals.std(0).tolist() if len(vals) else None
                comparisons.append(row)
    reference_distances=[]
    long_experts=bundle['suites']['long']['episodes']
    lookup={l:{r['episode_id']:r for r in read(out/'records.json')[l]} for l in LABELS}
    for case in read(out/'cases/selected_cases.json'):
        experts=[e for e in long_experts if e['task']==case['task_name']]
        for label in LABELS:
            x=load_trajectory(lookup[label][case['episode_id']])['xyz'][:-1]
            values=[legacy_plots.variation([x,np.asarray(e['xyz'])])['world_pairs'][0] for e in experts]
            reference_distances.append(dict(episode_id=case['episode_id'],policy=label,task=case['task_name'],
                expert_episode_ids=[e['episode_id'] for e in experts],rms_m=values,
                pairing='UNPAIRED; init-state equivalence UNKNOWN',
                definition='Original 32-point normalized-index world RMS; retained-step timing differs; not quality'))
    save(out/'expert_reference/reference_path_distances.json',reference_distances)
    save(out/'expert_reference/comparison_summary.json',dict(status='UNPAIRED_DESCRIPTIVE',regression=result['status'],tasks=comparisons,
        no_overlay_reason='Identical initial state not established; initial XYZ does not identify simulator state.',
        variation_files=['trajectories/variation.csv','expert_reference/regression/variation.csv'],
        reference_path_similarity='Within-task RMS distributions provided; no matched expert trajectory claim.'))


def replay(out):
    mf=read(out/'manifest.json'); records=read(out/'records.json')
    lookup={l:{r['episode_id']:r for r in records[l]} for l in LABELS}
    entries=[]
    for case in read(out/'cases/selected_cases.json'):
        for label in ('sft','grpo'):
            m=lookup[label][case['episode_id']]; source=Path(m['source_dir'])
            with np.load(source/'observations.npz',allow_pickle=False) as obs, np.load(source/'actions.npz',allow_pickle=False) as actions:
                chosen=select_queries(obs['query_step'],actions['executed_actions'][:,6])
                for q,reasons in chosen.items():
                    oid=f"{label}-{m['episode_id']}-q{q:04d}"
                    data={k:obs[k][q] for k in obs.files if k!='query_step'}
                    path=out/'replay_bank/observations'/(oid+'.npz'); path.parent.mkdir(parents=True,exist_ok=True)
                    with path.open('xb') as f: np.savez_compressed(f,**data)
                    step=int(obs['query_step'][q]); end=min(step+8,m['episode_length'])
                    entries.append(dict(observation_id=oid,source_policy=label,source_episode=m['episode_id'],category=case['category'],
                        source=str(source/'observations.npz'),source_sha256=sha(source/'observations.npz'),source_metadata=m,
                        query_index=q,simulator_step=step,executed_action_range=[step,end],reasons=reasons,
                        file=str(path.relative_to(out)),file_sha256=sha(path),
                        arrays={k:dict(sha256=array_hash(a),shape=list(a.shape),dtype=str(a.dtype)) for k,a in data.items()},
                        preprocessing=mf['protocols'][label]['model_config'],
                        image_semantics='main_image_raw=raw simulator; main_image_policy=policy image; pixel_values=processed tensor, bfloat16 stored as float32',
                        prefix_contract='Preserve saved input_ids/attention_mask identically for both policies. Teacher-forced action prefixes and free generation are separate future experiments; no action prefix synthesized.'))
    require(len({e['observation_id'] for e in entries})==len(entries),'Duplicate observation ID')
    save(out/'replay_bank/index.json',dict(observations=entries,origin_policy_counts=dict(collections.Counter(e['source_policy'] for e in entries)),
        selection='Same initial/early/middle/late fractions and gripper-query-neighbor rule per source; event counts may differ. No inferred failure event.',
        balancing='Equal episode coverage per policy; per-query analyses must stratify/reweight by source and episode.'))
    validate_bank(out)


def validate_bank(out):
    bank=read(out/'replay_bank/index.json'); seen=set(); verified_sources=set()
    for e in bank['observations']:
        require(e['observation_id'] not in seen,'Duplicate observation ID'); seen.add(e['observation_id'])
        source=Path(e['source']); require(source.is_file(),'Missing source reference')
        if (str(source),e['source_sha256']) not in verified_sources:
            require(sha(source)==e['source_sha256'],'Source observation changed')
            verified_sources.add((str(source),e['source_sha256']))
        if 'source_metadata' in e:
            original_metadata=read(source.parent/'metadata.json')
            require(all(e['source_metadata'].get(k)==v for k,v in original_metadata.items()),'Source metadata/provenance mismatch')
            require(e['source_metadata']['policy']==e['source_policy'],'Source policy mismatch')
            require(e['source_episode']==original_metadata['episode_id'],'Source episode mismatch')
            require(e['executed_action_range']==[e['simulator_step'],min(e['simulator_step']+8,original_metadata['episode_length'])],'Linked action range mismatch')
        path=out/e['file']; require(sha(path)==e['file_sha256'],'Bank file checksum')
        with np.load(source,allow_pickle=False) as original, np.load(path,allow_pickle=False) as copied:
            q=e['query_index']; require(int(original['query_step'][q])==e['simulator_step'],'Query reference alignment')
            for k,desc in e['arrays'].items():
                a=copied[k]
                require(list(a.shape)==desc['shape'] and str(a.dtype)==desc['dtype'],'Bank shape/dtype')
                require(array_hash(a)==desc['sha256']==array_hash(original[k][q]),'Array checksum/readback')
    result=dict(status='PASS',observations=len(seen),origin_policy_counts=bank['origin_policy_counts'],
                source_file_hashes_verified=True,array_readback_verified=True)
    path=out/'replay_bank/validation.json'
    if not path.exists(): save(path,result)
    return result


def validate(out):
    mf=read(out/'manifest.json'); hashes=dict(mf['source_files'])
    expert_path=out/'expert_reference/regression/regression.json'
    require(expert_path.is_file(),'Expert regression not run')
    hashes.update(read(expert_path)['source_files'])
    for path,h in hashes.items(): require(sha(path)==h, f'Source changed: {path}')
    bank=validate_bank(out)
    for name in ('paired/statistics.json','cases/selected_cases.json','trajectories/metrics.csv','expert_reference/comparison_summary.json'):
        require((out/name).is_file(),f'Missing stage: {name}')
    output_hashes={str(p.relative_to(out)):sha(p) for p in out.rglob('*') if p.is_file()}
    save(out/'validation.json',dict(status='PASS',input_files_unchanged=True,input_files_checked=len(hashes),replay=bank,
        output_hashes=output_hashes,scope='Structural/provenance validation, not physical calibration or causal inference'))


def report(out):
    mf=read(out/'manifest.json'); validation=read(out/'validation.json'); stats=read(out/'paired/statistics.json')
    selected=read(out/'cases/selected_cases.json'); bank=read(out/'replay_bank/validation.json')
    lines=['# Nominal paired behavior — Phase 3A','',f"Validation: {validation['status']}. Source hashes checked: {validation['input_files_checked']}.",'',
           '| Comparison | N | Both success | Gained | Lost | Both fail | Difference pp | Exact McNemar p |',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for k,s in stats.items(): lines.append(f"| {k} | {s['n']} | {s['both_success']} | {s['gained']} | {s['lost']} | {s['both_fail']} | {s['difference_pp']:.3f} | {s['mcnemar_exact_two_sided_p']:.6g} |")
    lines+=['','Primary: SFT deterministic → GRPO deterministic. Secondary: GRPO deterministic → stochastic. Additional changes both checkpoint and decoding.','',
            f"Bootstrap seed {mf['seed']}, iterations {mf['iterations']}; paired and suite-stratified task-cluster intervals in paired/statistics.json. These fixed tasks are not a random task-population sample; p-values are not mechanism evidence.",'',
            f"Selected Long cases: {len(selected)}; plots: {len(list((out/'trajectories/figures').glob('*.png')))}.",
            f"Replay observations: {bank['observations']}; origins: {bank['origin_policy_counts']}.",
            '','Expert metrics reuse original functions and are regression-tested on 428 Goal / 432 Spatial / 379 Long saved expert sequences. Expert comparisons are unpaired. RLDS ends at the last retained pre-action state; policy metrics include full T+1 and pre-action-only variants.',
            '','Coordinate evidence: '+mf['semantics']['status']+'. See manifest.json. Object roles require explicit mapping; missing roles remain UNKNOWN. Gripper command transitions do not establish successful grasp.',
            '','Measured trajectory differences and outcome transitions are observations. No claim of new skills, causal bifurcation, failure-gradient contribution, or OOD robustness. No wall-clock timestamps are reconstructed.','',
            'Files: paired/ (all pairs and statistics), cases/ (all 50 Long cases and selection), trajectories/ (metrics/figures), expert_reference/ (regression/comparison), replay_bank/ (lossless inputs and validation).']
    with (out/'summary.md').open('x') as f: f.write('\n'.join(lines)+'\n')
    save(out/'completion.json',dict(status='PASS',summary_sha256=sha(out/'summary.md'),validation_sha256=sha(out/'validation.json')))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('stage',choices=('preflight','paired','cases','trajectories','expert','replay','validate','report','all','expert-audit'))
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--research-root',type=Path,default=ROOT)
    for name in LABELS: p.add_argument('--'+name,type=Path)
    p.add_argument('--mapping',type=Path)
    p.add_argument('--expert-root',type=Path)
    p.add_argument('--expert-results',type=Path)
    p.add_argument('--object-roles',type=Path)
    p.add_argument('--seed',type=int,default=3101)
    p.add_argument('--iterations',type=int,default=10000)
    args=p.parse_args()
    require(args.iterations > 0, 'Bootstrap iterations must be positive')
    if args.stage=='expert-audit':
        require(args.expert_root and args.expert_results,'Provide expert root/results')
        expert_regression(args.expert_root.resolve(),args.expert_results.resolve(),args.output.resolve()); return
    if args.stage in ('preflight','all'):
        require(all(getattr(args,k) is not None for k in (*LABELS,'mapping','expert_root','expert_results')),'Preflight requires all fixed inputs and expert paths')
        preflight(args)
    stages={'paired':paired,'cases':cases,'trajectories':trajectories,'expert':expert,'replay':replay,'validate':validate,'report':report}
    if args.stage=='all':
        for fn in stages.values(): fn(args.output.resolve())
    elif args.stage in stages: stages[args.stage](args.output.resolve())
    print('Completed',args.stage,args.output,flush=True)


if __name__=='__main__':
    main()
