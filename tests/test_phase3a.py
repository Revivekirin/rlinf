"""Adversarial saved-data tests. Fixtures are synthetic, never research results."""
import copy
import importlib.util
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest
import argparse
import numpy as np

sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/analysis'))
import phase3a_core as c
import nominal_behavior as app


def fixtures():
    mapping={}; records={l:[] for l in c.LABELS}
    for s in c.SUITES:
        for t in range(10):
            mapping[(s,t)]=dict(task_name=f'{s}_{t}',libero_130_task_id=c.SUITES.index(s)*10+t,
                problem_folder=s,bddl_file=f'{t}.bddl',init_states_file=f'{t}.pruned',trial_sha256=[str(i)*64 for i in range(5)])
            for i in range(5):
                row=dict(suite=s,task_id=t,trial_id=i,task_name=f'{s}_{t}',libero_130_task_id=c.SUITES.index(s)*10+t,
                    initial_state_sha256=str(i)*64,simulator_seed=0,policy_seed=100000*c.SUITES.index(s)+100*t+i,
                    episode_id=f'{s}-t{t:02d}-i{i:02d}')
                for l in c.LABELS: records[l].append(dict(row))
    return mapping,records


class CoreTests(unittest.TestCase):
    def test_categories_and_exact_test(self):
        self.assertEqual([c.category(a,b) for a,b in ((True,True),(False,True),(True,False),(False,False))],list(c.CATEGORIES))
        rows=[dict(category='gained',left_success=False,right_success=True,suite='x',task_id=i) for i in range(5)]
        result=c.outcome_stats(rows,seed=8,iterations=100)
        self.assertAlmostEqual(result['mcnemar_exact_two_sided_p'],.0625)
        self.assertEqual(result['difference_pp'],100)
        self.assertEqual(result,c.outcome_stats(rows,seed=8,iterations=100))

    def test_gained_lost_invariant_and_clustered_dependence(self):
        rows=[]
        for task, (a,b) in enumerate(((False,True),(True,False),(True,True))):
            for trial in range(5):
                rows.append(dict(category=c.category(a,b),left_success=a,right_success=b,suite='x',task_id=task))
        r=c.outcome_stats(rows,iterations=2000)
        self.assertEqual(r['gained']-r['lost'],sum(int(x['right_success'])-int(x['left_success']) for x in rows))
        self.assertEqual(sum(r[k] for k in c.CATEGORIES),15)
        self.assertGreater(r['task_cluster_bootstrap_ci95_pp'][1]-r['task_cluster_bootstrap_ci95_pp'][0],
                           r['paired_bootstrap_ci95_pp'][1]-r['paired_bootstrap_ci95_pp'][0])

    def test_identity_failures(self):
        mapping,records=fixtures()
        self.assertEqual(len(c.match_records(records,mapping)['sft']),150)
        for mutation in ('missing','duplicate','task','hash','seed'):
            changed=copy.deepcopy(records)
            if mutation=='missing': changed['grpo'].pop()
            elif mutation=='duplicate': changed['grpo'][-1]=changed['grpo'][0]
            elif mutation=='task': changed['grpo'][0]['libero_130_task_id']=100
            elif mutation=='hash': changed['grpo'][0]['initial_state_sha256']='wrong'
            else: changed['grpo'][0]['policy_seed']=999
            with self.subTest(mutation=mutation),self.assertRaises(ValueError): c.match_records(changed,mapping)

    def test_canonical_and_optional_objects(self):
        states={'robot0_eef_pos':np.zeros((3,3)),'robot0_eef_quat':np.tile([0.,0,0,1],(3,1)),
                'robot0_gripper_qpos':np.zeros((3,2))}
        actions={'executed_actions':np.zeros((2,7))}
        self.assertEqual(c.canonical(states,actions,{'episode_length':2})['objects'],{})
        for change in ('shape','nan','gripper','frame','quaternion','units'):
            st=copy.deepcopy(states); frame='simulator_world'; units='m'
            if change=='shape': st['robot0_eef_pos']=np.zeros((2,3))
            elif change=='nan': st['robot0_eef_pos'][0,0]=np.nan
            elif change=='gripper': st['robot0_gripper_qpos']=np.zeros((2,2))
            elif change=='quaternion': st['robot0_eef_quat'][:]=0
            elif change=='units': units='cm'
            else: frame='robot_base'
            with self.subTest(change=change),self.assertRaises(ValueError): c.canonical(st,actions,{'episode_length':2},frame=frame,units=units)

    def test_selection_empty_categories_and_diversity(self):
        rows=[dict(category='gained',task_id=t,trial_id=i,episode_id=f'{t}-{i}') for t in range(3) for i in range(5)]
        chosen=c.choose_cases(rows)
        self.assertEqual(len(chosen),3); self.assertEqual(len({r['task_id'] for r in chosen}),3)
        self.assertEqual(chosen,c.choose_cases(list(reversed(rows))))

    def test_query_alignment(self):
        self.assertIn(0,c.select_queries(np.arange(4)*8,np.r_[np.zeros(9),np.ones(20)]))
        with self.assertRaises(ValueError): c.select_queries(np.array([0,9]),np.zeros(10))

    def test_checksum_includes_dtype_shape_and_values(self):
        a=np.arange(6,dtype=np.float32)
        for b in (a.astype(np.float64),a.reshape(2,3),a+1): self.assertNotEqual(c.array_hash(a),c.array_hash(b))

    def test_replay_readback_tamper_missing_reference_and_duplicates(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'source.npz'; target=root/'replay_bank/observations/o.npz'
            target.parent.mkdir(parents=True)
            a=np.arange(12,dtype=np.uint8).reshape(2,2,3)
            np.savez_compressed(source,query_step=np.array([0]),main_image_raw=a[None])
            np.savez_compressed(target,main_image_raw=a)
            entry=dict(observation_id='o',source=str(source),source_sha256=c.sha(source),query_index=0,
                simulator_step=0,file=str(target.relative_to(root)),file_sha256=c.sha(target),
                arrays={'main_image_raw':dict(sha256=c.array_hash(a),shape=list(a.shape),dtype=str(a.dtype))})
            bank=dict(observations=[entry],origin_policy_counts={'sft':1})
            app.save(root/'replay_bank/index.json',bank)
            self.assertEqual(app.validate_bank(root)['status'],'PASS')
            for kind in ('duplicate','missing','step','hash'):
                broken=copy.deepcopy(bank)
                if kind=='duplicate': broken['observations'].append(entry)
                elif kind=='missing': broken['observations'][0]['source']=str(root/'missing.npz')
                elif kind=='step': broken['observations'][0]['simulator_step']=1
                else: broken['observations'][0]['arrays']['main_image_raw']['sha256']='bad'
                (root/'replay_bank/index.json').write_text(json.dumps(broken))
                with self.subTest(kind=kind),self.assertRaises(ValueError): app.validate_bank(root)
            self.assertEqual(c.sha(source),entry['source_sha256'])
            (root/'replay_bank/index.json').write_text(json.dumps(bank))
            np.savez_compressed(source,query_step=np.array([0]),main_image_raw=(a+1)[None])
            with self.assertRaisesRegex(ValueError,'Source observation changed'):
                app.validate_bank(root)

    def test_reject_overwrite(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/'x.json'; app.save(p,{'v':1})
            with self.assertRaises(FileExistsError): app.save(p,{'v':2})
            self.assertEqual(app.read(p),{'v':1})


@unittest.skipUnless(os.environ.get('PHASE3A_EXPERT_ROOT'),'Set PHASE3A_EXPERT_ROOT for real legacy regression')
class LegacyTests(unittest.TestCase):
    def test_known_paths_resampling_rms_orientation(self):
        a,v=c.legacy(Path(os.environ['PHASE3A_EXPERT_ROOT']))
        x=np.array([[0.,0,0],[1,0,0],[1,1,0]])
        self.assertEqual(a.path_statistics(x)['path_length'],2)
        self.assertAlmostEqual(a.path_statistics(x)['net_displacement'],np.sqrt(2))
        self.assertEqual(a.align_path(x).shape,(32,3))
        np.testing.assert_allclose(v.variation([x,x+[0,0,2]])['world_pairs'],[2])
        np.testing.assert_allclose(a.quat2axisangle(np.array([0.,0,np.sin(.5),np.cos(.5)])),[0,0,1])
        self.assertNotIn('torch',sys.modules)
        self.assertNotIn('robosuite',sys.modules)
        self.assertNotIn('libero',sys.modules)


if __name__=='__main__': unittest.main()
