"""CPU tests: measured motion boundaries, full-support divergence, immutable provenance and seed grid."""
import sys
import tempfile
import unittest
import argparse
import csv
import json
import shutil
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/analysis'))
from phase3b import motion_metrics,stationary_runs,fresh,seal,verify_seal,sources
from phase3b_distribution import distribution_metrics, summarize
from nominal_behavior import save
from phase3a_core import sha, protocol_hash, REVISION
from phase3b_pilot import seed_for, validate as validate_pilot
from phase3b_sft_sampling import summarize as summarize_sft_sampling

class MotionTests(unittest.TestCase):
    def test_threshold_and_fraction_boundary(self):
        x=np.array([[0.,0,0],[0,0,0],[.0001,0,0]])
        m,d,_,_=motion_metrics(x)
        self.assertEqual(m['stationary_steps'],1) # exactly 0.1 mm is moving
        self.assertEqual(m['stationary_fraction'],.5)
        self.assertTrue(m['stagnation'])
        self.assertFalse(m['legacy_strict_gt_half'])
    def test_runs_include_boundary_segments(self):
        starts,lengths=stationary_runs([1,1,0,1,0,1,1,1])
        np.testing.assert_array_equal(starts,[0,3,5]); np.testing.assert_array_equal(lengths,[2,1,3])
    def test_motion_not_equal_endpoint_displacement(self):
        x=np.array([[0.,0,0],[1,0,0],[0,0,0]])
        m,_,_,_=motion_metrics(x); self.assertEqual(m['path_length'],2); self.assertEqual(m['net_displacement'],0); self.assertFalse(m['stagnation'])
    def test_bad_trajectory_rejected(self):
        for x in (np.ones((1,3)),np.ones((3,2)),np.full((3,3),np.nan)):
            with self.assertRaises(ValueError): motion_metrics(x)
    def test_zero_motion_all_stationary(self):
        m,_,_,_=motion_metrics(np.zeros((521,3)))
        self.assertEqual(m['stationary_longest_run'],520); self.assertEqual(m['stationary_runs'],1)

class ProbabilityTests(unittest.TestCase):
    def test_uniform_full_256_bins(self):
        m=distribution_metrics(np.zeros((56,256)),np.zeros((56,256)))
        np.testing.assert_allclose(m['entropy_sft'],np.log(256))
        np.testing.assert_allclose(m['top10_sft'],10/256)
        np.testing.assert_allclose(m['kl_sft_grpo'],0,atol=1e-14)
        np.testing.assert_allclose(m['js'],0,atol=1e-14)
    def test_known_bernoulli_kl(self):
        p=np.array([[.8,.2]]); q=np.array([[.4,.6]])
        r=distribution_metrics(np.log(p),np.log(q))
        self.assertAlmostEqual(r['kl_sft_grpo'][0],.8*np.log(2)+.2*np.log(1/3))
        self.assertAlmostEqual(r['kl_grpo_sft'][0],.4*np.log(.5)+.6*np.log(3))
        self.assertTrue(r['greedy_disagreement'][0])
    def test_js_symmetric_extreme_logits_finite(self):
        a=np.array([[1000.,-1000.]]); b=-a
        r=distribution_metrics(a,b); rr=distribution_metrics(b,a)
        self.assertAlmostEqual(r['js'][0],np.log(2)); np.testing.assert_allclose(r['js'],rr['js'])
        self.assertTrue(np.isfinite(r['kl_sft_grpo']).all())
    def test_temperature_changes_mass_not_argmax(self):
        a=np.array([[4.,0.,-4.]]); b=a[:,::-1]
        r=distribution_metrics(a,b,1); rr=distribution_metrics(a,b,1.6)
        self.assertGreater(rr['entropy_sft'][0],r['entropy_sft'][0]); np.testing.assert_array_equal(r['greedy_disagreement'],rr['greedy_disagreement'])
    def test_different_support_or_nan_rejected(self):
        with self.assertRaises(ValueError): distribution_metrics(np.zeros((1,2)),np.zeros((1,3)))
        with self.assertRaises(ValueError): distribution_metrics(np.array([[np.nan,0]]),np.zeros((1,2)))

class ProvenanceTests(unittest.TestCase):
    def test_seal_detects_changes_and_missing(self):
        with tempfile.TemporaryDirectory() as td:
            p=fresh(Path(td)/'run'); (p/'data.txt').write_text('a'); seal(p,dict(status='PASS')); verify_seal(p)
            (p/'data.txt').write_text('b')
            with self.assertRaises(ValueError): verify_seal(p)
    def test_no_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(ValueError): fresh(Path(td))
    def test_pilot_grid_unique_nonoverlap(self):
        seeds=[seed_for(t,i,k) for t in (8,3) for i in range(5) for k in range(1,6)]
        old={200000+100*t+i for t in (8,3) for i in range(5)}
        self.assertEqual(len(seeds),50); self.assertEqual(len(set(seeds)),50); self.assertFalse(set(seeds)&old)
        self.assertNotEqual(seed_for(8,0,1),seed_for(8,0,2))
    def test_pilot_b_not_accepted(self):
        with self.assertRaises(ValueError): seed_for(8,0,6)

class SummaryTests(unittest.TestCase):
    def test_sft_sampling_paired_gains_and_losses(self):
        rows = [
            dict(suite='libero_goal', outcome='gained', deterministic_success=False, stochastic_success=True,
                 deterministic_stagnant=True, stochastic_stagnant=False),
            dict(suite='libero_goal', outcome='lost', deterministic_success=True, stochastic_success=False,
                 deterministic_stagnant=False, stochastic_stagnant=True),
            dict(suite='libero_10', outcome='both_success', deterministic_success=True, stochastic_success=True,
                 deterministic_stagnant=False, stochastic_stagnant=False),
        ]
        total = summarize_sft_sampling(rows, ())[0]
        self.assertEqual((total['gained'], total['lost']), (1, 1))
        self.assertEqual((total['deterministic_success'], total['stochastic_success']), (2, 2))
        self.assertEqual((total['deterministic_stagnant'], total['stochastic_stagnant']), (1, 1))
        self.assertEqual(len(summarize_sft_sampling(rows, ('suite',))), 2)

    def test_two_worker_common_observation_join(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); pre=fresh(root/'pre'); out=root/'output'
            entry=dict(observation_id='test',source_policy='sft',source_episode='episode',category='gained',query_index=0,simulator_step=0,
                       file_sha256='same-input',source_metadata=dict(initial_state_sha256='state',language='test action'),sft_stagnation=True,grpo_stagnation=False)
            save(pre/'manifest.json',dict(observations=[entry])); seal(pre,dict(status='PASS'))
            for policy in ('sft','grpo'):
                worker=fresh(out/policy); arrays=dict(logits=np.zeros((56,256)),action_tokens=np.zeros((8,7),dtype=int),decoded_actions=np.zeros((8,7)),normalized_actions=np.zeros((8,7)))
                if policy=='grpo': arrays['logits'][:,0]=1
                np.savez(worker/'test.npz',**arrays)
                record=dict(observation_id='test',file='test.npz',input_checksum='same-input',processed_tensor_hashes=dict(input_ids='same',attention_mask='same',pixel_values='same'))
                save(worker/'manifest.json',dict(mode='smoke',preflight_sha256=sha(pre/'validation.json'),source_hashes=sources(),action_token_ids=list(range(256)),bin_centers=[0],records=[record],action_norm_stats={}))
                seal(worker,dict(status='PASS'))
            summarize(argparse.Namespace(preflight=pre,output=out))
            verify_seal(out)
            with (out/'token_metrics.csv').open() as f: rows=list(csv.DictReader(f))
            self.assertEqual(len(rows),112)
            self.assertTrue(all(float(r['js'])>0 for r in rows))
            self.assertTrue(json.loads((out/'manifest.json').read_text())['processed_inputs_equal'])

class PilotSmokeTests(unittest.TestCase):
    def fixture(self, root, mismatch=False):
        sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/evaluate'))
        from nominal_storage import write_episode
        eid='libero_10-t08-i00'; protocol={'synthetic_test_only':True}; ph=protocol_hash(protocol)
        m=dict(complete=True,episode_id=eid,episode_length=2,num_queries=1,protocol_hash=ph,policy_seed=200800,initial_state_sha256='state',checkpoint_revision=REVISION['grpo'],sampling_mode='stochastic',temperature=1.6,lora_applied=False,simulator_seed=0,task_id=8,trial_id=0,task_name='test',language='test',success=False)
        states=dict(robot0_eef_pos=np.zeros((3,3)),robot0_eef_quat=np.tile([0.,0,0,1],(3,1)))
        executed=np.zeros((2,7)); executed[:,6]=1
        actions=dict(executed_actions=executed,decoded_action_chunks=np.zeros((1,8,7)),raw_action_tokens=np.zeros((1,8,7),dtype=int),executed_query_index=np.zeros(2,dtype=int),executed_action_slot=np.arange(2),simulator_step=np.arange(2))
        obs=dict(query_step=np.array([0]),main_image_policy=np.zeros((1,2,2,3),dtype=np.uint8),main_image_raw=np.zeros((1,2,2,3),dtype=np.uint8),pixel_values=np.zeros((1,3,2,2)),input_ids=np.array([[1,29871]]),attention_mask=np.ones((1,2),dtype=bool))
        ref=write_episode(root/'reference',m,states,actions,obs)
        p3=fresh(root/'p3'); save(p3/'manifest.json',dict(source_files={str(ref/n):sha(ref/n) for n in ('metadata.json','states.npz','actions.npz','observations.npz')}))
        jobs=[]; out=fresh(root/'smoke'); (out/'entries').mkdir()
        for k in (1,2):
            seed=seed_for(8,0,k); job=dict(job_id=f'{eid}-s{seed}',episode_id=eid,policy_seed=seed,initial_state_sha256='state',task_id=8,trial_id=0,task_name='test',reference=dict(source_dir=str(ref)))
            jobs.append(job); run=out/'new'/str(k); run.mkdir(parents=True); save(run/'manifest.json',dict(protocol=protocol,protocol_hash=ph))
            act=dict(actions,raw_action_tokens=np.full((1,8,7),k,dtype=int)); ob={key:v.copy() for key,v in obs.items()}
            if mismatch and k==2: ob['main_image_policy'][0,0,0,0]=1
            dest=write_episode(run,dict(m,policy_seed=seed),states,act,ob)
            save(out/'entries'/(job['job_id']+'.json'),dict(job_id=job['job_id'],episode_path=str(dest),protocol_hash=ph,provenance='new',hashes={n:sha(dest/n) for n in ('metadata.json','states.npz','actions.npz','observations.npz')}))
        plan=fresh(root/'plan'); save(plan/'manifest.json',dict(jobs=jobs,phase3a=str(p3),rng_control='synthetic test')); seal(plan,dict(status='PASS'))
        save(out/'manifest.json',dict(mode='smoke',plan=str(plan),plan_sha256=sha(plan/'validation.json')))
        return out
    def test_distinct_seeds_same_observation_validate(self):
        with tempfile.TemporaryDirectory() as td:
            out=self.fixture(Path(td)); validate_pilot(argparse.Namespace(output=out)); v=verify_seal(out)
            self.assertTrue(v['seed_divergence_verified']); self.assertEqual(v['episodes'],2)
    def test_same_state_observation_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            out=self.fixture(Path(td),mismatch=True)
            with self.assertRaises(ValueError): validate_pilot(argparse.Namespace(output=out))
    def test_missing_seed_rejected(self):
        with tempfile.TemporaryDirectory() as td:
            out=self.fixture(Path(td)); next((out/'entries').glob('*.json')).unlink()
            with self.assertRaises(ValueError): validate_pilot(argparse.Namespace(output=out))

if __name__=='__main__': unittest.main()
