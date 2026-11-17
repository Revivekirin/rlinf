"""CPU-only repeated-rollout planning, storage and resume contracts."""

import argparse
import json
import sys
import tempfile
import types
import unittest
import weakref
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/analysis'))
sys.path.insert(0,str(ROOT/'scripts/evaluate'))
from nominal_storage import write_episode
from nominal_validate import validate_episode
from nominal_behavior import read, save
from phase3a_core import protocol_hash, sha, REVISION
from phase3b import seal
import phase3b_pilot as pilot


class RepeatedRolloutTests(unittest.TestCase):
    def fixture(self, root, tasks=(3,8), trials=range(5), policies=('sft','grpo'), seeds=None):
        phase3a=root/'phase3a'; phase3a.mkdir(); (phase3a/'validation.json').write_text('{}\n')
        mapping=dict(schema_version=1,trials_compared=5,tasks=[])
        records={p:[] for p in policies}
        for task in tasks:
            mr=dict(suite='libero_10',subset_task_id=task,task_name=f'task-{task}',
                    libero_130_task_id=120+task,problem_folder='folder',bddl_file='task.bddl',
                    init_states_file='states',trial_sha256=[f'state-{task}-{i}' for i in range(5)])
            mapping['tasks'].append(mr)
            for trial in trials:
                eid=f'libero_10-t{task:02d}-i{trial:02d}'
                for policy in policies:
                    records[policy].append(dict(episode_id=eid,task_id=task,trial_id=trial,
                        task_name=mr['task_name'],initial_state_sha256=mr['trial_sha256'][trial],
                        language=f'instruction-{task}'))
        mapping_path=root/'mapping.json'; save(mapping_path,mapping)
        protocols={p:dict(checkpoint_revision=REVISION[p],checkpoint_repo=pilot.REPEAT_REPOS[p],
            checkpoint_path='/server/checkpoint/'+p,lora_applied=False,
            unnorm_key='libero_130_no_noops_trajall',mapping_sha256=protocol_hash(mapping),
            model_config=dict(model_path='/server/checkpoint/'+p,is_lora=False,lora_path=None))
            for p in policies}
        info=dict(manifest=dict(mapping=str(mapping_path),protocols=protocols),records=records)
        self.fixture_info=info
        args=argparse.Namespace(phase3a=phase3a,mapping=mapping_path,output=root/'plan',suite='libero_10',
            task_ids=list(tasks),trial_ids=list(trials),policies=list(policies),sampling_seeds=seeds,
            repeats=None,temperature=1.6,top_p=1.0,top_k=-1)
        with patch.object(pilot,'verify_phase3a',return_value=info): pilot.repeat_plan(args)
        return args,read(args.output/'manifest.json')

    def run_root(self, root, args, pl):
        out=root/'run'; out.mkdir(); (out/'entries').mkdir(); (out/'jobs').mkdir()
        save(out/'manifest.json',dict(plan=str(args.output.resolve()),plan_sha256=sha(args.output/'validation.json'),
            experiment_hash=pl['experiment_hash']))
        return out

    def add_episode(self, out, pl, job, *, corrupt=False, entry=True, seed_action=True):
        dest=out/'jobs'/job['job_id']; dest.mkdir()
        save(dest/'manifest.json',dict(protocol=pl['protocols'][job['policy']],protocol_hash=job['protocol_hash']))
        metadata=dict(complete=True,episode_id=job['episode_id'],run_id=out.name,
            repeated_job_id=job['job_id'],policy=job['policy'],suite=job['suite'],task_id=job['task_id'],
            trial_id=job['trial_id'],task_name=job['task_name'],libero_130_task_id=job['libero_130_task_id'],
            initial_state_sha256=job['initial_state_sha256'],policy_seed=job['policy_seed'],simulator_seed=0,
            checkpoint_repo=pl['protocols'][job['policy']]['checkpoint_repo'],
            checkpoint_revision=pl['protocols'][job['policy']]['checkpoint_revision'],lora_applied=False,
            sampling_mode='stochastic',temperature=1.6,language=job['language'],
            protocol_hash=job['protocol_hash'],episode_length=2,num_queries=1,success=False)
        states=dict(robot0_eef_pos=np.zeros((3,3)),robot0_eef_quat=np.tile([0.,0,0,1],(3,1)),
                    robot0_gripper_qpos=np.zeros((3,2)))
        decoded=np.zeros((1,8,7)); executed=np.zeros((2,7)); executed[:,6]=1
        actions=dict(raw_action_tokens=np.full((1,8,7),job['policy_seed']%100 if seed_action else 0,dtype=np.int32),
            decoded_action_chunks=decoded,executed_actions=executed,executed_query_index=np.zeros(2,dtype=np.int32),
            executed_action_slot=np.arange(2,dtype=np.int16),simulator_step=np.arange(2,dtype=np.int32))
        obs=dict(query_step=np.array([0]),main_image_raw=np.zeros((1,2,2,3),dtype=np.uint8),
            main_image_policy=np.zeros((1,2,2,3),dtype=np.uint8),pixel_values=np.zeros((1,3,2,2)),
            input_ids=np.array([[1,2]]),attention_mask=np.ones((1,2),dtype=bool))
        path=write_episode(dest,metadata,states,actions,obs)
        if corrupt:
            with (path/'actions.npz').open('wb') as f: np.savez(f,**dict(actions,executed_actions=np.full((2,7),np.nan)))
        if entry:
            save(out/'entries'/(job['job_id']+'.json'),dict(job_id=job['job_id'],episode_path=str(path.resolve()),
                protocol_hash=job['protocol_hash'],hashes={n:sha(path/n) for n in pilot.REPEAT_ARTIFACTS},provenance='fixture'))
        return path

    def test_default_plan_and_legacy_seed_compatibility(self):
        with tempfile.TemporaryDirectory() as td:
            args,pl=self.fixture(Path(td))
            self.assertEqual(pl['expected'],100)
            self.assertEqual(len({j['job_id'] for j in pl['jobs']}),100)
            self.assertEqual(len(set(pl['protocol_hashes'].values())),2)
            for job in pl['jobs']:
                self.assertEqual(job['initial_state_sha256'],f"state-{job['task_id']}-{job['trial_id']}")
            j=next(j for j in pl['jobs'] if j['task_id']==8 and j['trial_id']==0 and j['policy']=='grpo')
            self.assertEqual(j['policy_seed'],pilot.seed_for(8,0,1))
            self.assertEqual(j['policy_seed'],next(x['policy_seed'] for x in pl['jobs']
                if x['episode_id']==j['episode_id'] and x['policy']=='sft'))
            self.assertNotEqual(j['job_id'],next(x['job_id'] for x in pl['jobs']
                if x['episode_id']==j['episode_id'] and x['policy']=='sft'))

    def test_explicit_seeds_and_protocol_rejection(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); args,pl=self.fixture(root,tasks=(8,),trials=(0,),seeds=[3101,3102])
            self.assertEqual(pl['expected'],4)
            self.assertEqual({j['policy_seed'] for j in pl['jobs']},{3101,3102})
            args.output=root/'bad-plan'; args.top_p=.9
            with patch.object(pilot,'verify_phase3a',return_value=self.fixture_info):
                with self.assertRaises(ValueError): pilot.repeat_plan(args)
            args.top_p=1.0; args.top_k=5
            with patch.object(pilot,'verify_phase3a',return_value=self.fixture_info):
                with self.assertRaises(ValueError): pilot.repeat_plan(args)

    def test_partial_complete_resume_and_corruption(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); args,pl=self.fixture(root,tasks=(8,),trials=(0,),seeds=[3101,3102])
            out=self.run_root(root,args,pl); first=pl['jobs'][0]
            path=self.add_episode(out,pl,first)
            self.assertEqual(validate_episode(path,first['protocol_hash'])['episode_id'],first['episode_id'])
            state=pilot.repeat_validate_state(out,pl,partial=True)
            self.assertEqual((state['status'],state['completed_jobs'],len(state['pending_job_ids'])),('PARTIAL_VALID',1,3))
            self.assertEqual(pilot.repeat_validate_state(out,pl)['status'],'FAILED')
            with self.assertRaises(ValueError): pilot.repeat_validate_state(out,pl,partial=True,selected=[pl['jobs'][1]['job_id']])
            run_args=argparse.Namespace(plan=args.output,audit=root/'audit',output=out,research_root=ROOT,smoke=False,resume=True)
            with patch.object(pilot,'repeat_preflight'),patch('phase3b_distribution.load_model',side_effect=RuntimeError('GPU must not load')):
                with self.assertRaises(RuntimeError): pilot.repeat_run(run_args)
            self.assertTrue((out/'entries'/(first['job_id']+'.json')).exists())
            for job in pl['jobs'][1:]: self.add_episode(out,pl,job)
            state=pilot.repeat_validate_state(out,pl)
            self.assertEqual((state['status'],state['completed_jobs']),('COMPLETE_VALID',4))
            before=len(list((out/'entries').glob('*.json')))
            with patch.object(pilot,'repeat_preflight'),patch('phase3b_distribution.load_model') as load:
                pilot.repeat_run(run_args); load.assert_not_called()
            self.assertEqual(len(list((out/'entries').glob('*.json'))),before)
            entry_path=out/'entries'/(first['job_id']+'.json')
            entry=read(entry_path); entry['protocol_hash']='wrong'; entry_path.write_text(json.dumps(entry))
            with self.assertRaises(ValueError): pilot.repeat_validate_state(out,pl,partial=True)

    def test_interrupted_and_corrupted_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); args,pl=self.fixture(root,tasks=(8,),trials=(0,),policies=('grpo',),seeds=[3101,3102])
            out=self.run_root(root,args,pl)
            partial=out/'jobs'/pl['jobs'][0]['job_id']; partial.mkdir()
            self.assertEqual(pilot.repeat_validate_state(out,pl,partial=True)['completed_jobs'],0)
            with self.assertRaises(ValueError):
                run_args=argparse.Namespace(plan=args.output,audit=root/'audit',output=out,research_root=ROOT,smoke=False,resume=True)
                with patch.object(pilot,'repeat_preflight'): pilot.repeat_run(run_args)
            self.assertTrue((out/'failures'/(pl['jobs'][0]['job_id']+'.json')).is_file())
            path=self.add_episode(out,pl,pl['jobs'][1],corrupt=True)
            with self.assertRaises((ValueError,AssertionError)): pilot.repeat_validate_state(out,pl,partial=True)

    def test_policy_switch_releases_model_and_core(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); args,pl=self.fixture(root,tasks=(8,),trials=(0,),seeds=[3101])
            out=self.run_root(root,args,pl)
            run_args=argparse.Namespace(plan=args.output,audit=root/'audit',output=out,
                                        research_root=ROOT,smoke=False,resume=True)
            refs=[]
            class FakeModel: pass
            def load_model(_root,_phase3a,policy):
                if refs:
                    self.assertTrue(all(ref() is None for ref in refs[-1]),
                                    'Previous policy still retains model or core')
                model=FakeModel(); core=FakeModel(); model.core=core
                refs.append((weakref.ref(model),weakref.ref(core)))
                return model,core,pl['protocols'][policy]['model_config']
            def collect(_args,_pl,job,_model,_cfg):
                return self.add_episode(out,pl,job,entry=False)
            fake_torch=types.SimpleNamespace(cuda=types.SimpleNamespace(empty_cache=lambda: None))
            with patch.object(pilot,'repeat_preflight'),patch('phase3b_distribution.load_model',new=load_model),\
                 patch.object(pilot,'collect_repeat_job',new=collect),patch.dict(sys.modules,{'torch':fake_torch}):
                pilot.repeat_run(run_args)
            self.assertEqual(pilot.repeat_validate_state(out,pl)['status'],'COMPLETE_VALID')

    def test_recovery_and_same_seed_replay_fixture(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); args,pl=self.fixture(root,tasks=(8,),trials=(0,),policies=('grpo',),seeds=[3101])
            job=pl['jobs'][0]; out=self.run_root(root,args,pl)
            first=self.add_episode(out,pl,job,entry=False)
            self.assertEqual(pilot.repeat_validate_state(out,pl,partial=True)['recoverable_job_ids'],[job['job_id']])
            run_args=argparse.Namespace(plan=args.output,audit=root/'audit',output=out,research_root=ROOT,smoke=False,resume=True)
            with patch.object(pilot,'repeat_preflight'),patch('phase3b_distribution.load_model') as load:
                pilot.repeat_run(run_args); load.assert_not_called()
            self.assertEqual(pilot.repeat_validate_state(out,pl)['status'],'COMPLETE_VALID')
            second_root=root/'replay'; second_root.mkdir(); second=self.run_root(second_root,args,pl)
            replay=self.add_episode(second,pl,job)
            for name in ('states.npz','actions.npz','observations.npz'):
                with np.load(first/name,allow_pickle=False) as a,np.load(replay/name,allow_pickle=False) as b:
                    self.assertEqual(a.files,b.files)
                    for key in a.files: np.testing.assert_array_equal(a[key],b[key])

    def test_corrupted_state_and_cross_policy_initial_mismatch(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); args,pl=self.fixture(root,tasks=(8,),trials=(0,),seeds=[3101])
            out=self.run_root(root,args,pl)
            first=self.add_episode(out,pl,pl['jobs'][0])
            second=self.add_episode(out,pl,pl['jobs'][1])
            self.assertEqual(pilot.repeat_validate_state(out,pl)['status'],'COMPLETE_VALID')
            with np.load(second/'states.npz',allow_pickle=False) as z: states={k:z[k].copy() for k in z.files}
            states['robot0_eef_pos'][0,0]=.01
            with (second/'states.npz').open('wb') as f: np.savez_compressed(f,**states)
            entry_path=out/'entries'/(pl['jobs'][1]['job_id']+'.json')
            entry=read(entry_path); entry['hashes']['states.npz']=sha(second/'states.npz')
            entry_path.write_text(json.dumps(entry))
            with self.assertRaises(ValueError): pilot.repeat_validate_state(out,pl,partial=True)
            states['robot0_eef_pos'][0,0]=np.nan
            with (second/'states.npz').open('wb') as f: np.savez_compressed(f,**states)
            entry['hashes']['states.npz']=sha(second/'states.npz'); entry_path.write_text(json.dumps(entry))
            with self.assertRaises((ValueError,AssertionError)): pilot.repeat_validate_state(out,pl,partial=True)


if __name__=='__main__': unittest.main()
