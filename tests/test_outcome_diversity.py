"""CPU regressions for R3 outcome-conditioned repeated-rollout analysis."""
import argparse
import contextlib
import csv
import io
import json
from pathlib import Path
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/analysis'))
sys.path.insert(0,str(ROOT/'scripts/evaluate'))
sys.path.insert(0,str(ROOT/'tests'))
from scripts.common.io import read
from scripts.common.integrity import sha, verify_seal
import outcome_diversity as r3
import phase3b_pilot as pilot
import test_repeated_rollout as fixture_module


class Metric:
    def variation(self, paths):
        aligned=[]
        for x in paths:
            u=np.linspace(0,1,len(x)); q=np.linspace(0,1,32)
            aligned.append(np.column_stack([np.interp(q,u,x[:,k]) for k in range(3)]))
        return {'world_pairs':[float(np.sqrt(np.mean(np.sum((aligned[0]-aligned[1])**2,axis=1))))]}


class OutcomeDiversityTests(unittest.TestCase):
    def test_category_symmetry_and_undefined(self):
        for a in (False,True):
            for b in (False,True): self.assertEqual(r3.category(a,b),r3.category(b,a))
        episodes=[]; pairs=[]
        for t in (3,8):
            for i in range(5):
                eid=f'libero_10-t{t:02d}-i{i:02d}'
                for policy in ('sft','grpo'):
                    items=[]
                    for seed in range(5):
                        row=dict(job_id=f'{eid}-{policy}-{seed}',episode_id=eid,task_id=t,trial_id=i,policy=policy,
                                 policy_seed=seed,success=(policy=='grpo' or seed==0),steps=seed+2,initial_state_sha256=eid)
                        episodes.append(row); items.append(row)
                    for a in range(5):
                        for b in range(a+1,5):
                            x,y=items[a],items[b]
                            pairs.append(dict(episode_id=eid,pair_type=policy,job_a=x['job_id'],job_b=y['job_id'],
                                              success_a=x['success'],success_b=y['success'],trajectory_rms_32_m=float(a+b+1)))
                left=[r for r in episodes if r['episode_id']==eid and r['policy']=='sft']
                right=[r for r in episodes if r['episode_id']==eid and r['policy']=='grpo']
                for x in left:
                    for y in right:
                        pairs.append(dict(episode_id=eid,pair_type='cross',job_a=x['job_id'],job_b=y['job_id'],
                                          success_a=x['success'],success_b=y['success'],trajectory_rms_32_m=1.0))
        state,decomp=r3.decompose(pairs,episodes)
        self.assertEqual((len(pairs),len(state),len(decomp)),(450,20,20))
        s=next(r for r in state if r['policy']=='sft')
        self.assertEqual((s['n_SS_pairs'],s['n_SF_pairs'],s['n_FF_pairs']),(0,4,6))
        self.assertIsNone(s['D_SS_m']); self.assertEqual(s['SS_status'],'UNDEFINED')
        self.assertAlmostEqual(s['D_all_m'],sum(d['C_'+c+'_m'] for c in r3.CATEGORIES for d in decomp if d['episode_id']==s['episode_id'] and d['policy']=='sft'))
        bad=[dict(p) for p in pairs]; bad.append(dict(bad[0]))
        with self.assertRaisesRegex(ValueError,'duplicate pair'): r3.decompose(bad,episodes)
        bad=[dict(p) for p in pairs]; bad[0]['episode_id']='libero_10-t08-i04'
        with self.assertRaisesRegex(ValueError,'Different initial states'): r3.decompose(bad,episodes)
        bad=[dict(p) for p in pairs]; bad.pop()
        with self.assertRaisesRegex(ValueError,'Cross-policy pair grid'): r3.decompose(bad,episodes)

    def test_original_metric_and_full_saved_output(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); fixture=fixture_module.RepeatedRolloutTests()
            args,plan=fixture.fixture(root); phase3a=root/'phase3a'
            from scripts.common.io import save
            save(phase3a/'manifest.json',{'expert_root':str(root/'expert')})
            run=fixture.run_root(root,args,plan)
            for job in plan['jobs']:
                path=fixture.add_episode(run,plan,job)
                t=job['task_id']; i=job['trial_id']; seed=(job['policy_seed']//1000000)-1
                index=(0 if t==3 else 5)+i
                success=(seed<2 if index<3 else seed<1 if index<6 else False) if job['policy']=='sft' else (seed<4 if index<8 else seed<3)
                m=read(path/'metadata.json'); m['success']=success; (path/'metadata.json').write_text(json.dumps(m))
                with np.load(path/'states.npz',allow_pickle=False) as z: states={k:z[k].copy() for k in z.files}
                states['robot0_eef_pos'][1:]=[[.01*(seed+1)+.004*index,.005*(seed+1),.001],
                                             [.02*(seed+1)+.006*index,.007*(seed+1),.002]]
                with (path/'states.npz').open('wb') as f: np.savez_compressed(f,**states)
                ep=run/'entries'/(job['job_id']+'.json'); entry=read(ep)
                for name in ('metadata.json','states.npz'): entry['hashes'][name]=sha(path/name)
                ep.write_text(json.dumps(entry))
            metric=Metric(); analysis=types.SimpleNamespace(path_statistics=lambda xyz:{'path_length':float(np.sum(np.linalg.norm(np.diff(xyz,axis=0),axis=1)))})
            old=root/'old'; stream=io.StringIO()
            with patch.object(pilot,'legacy',return_value=(analysis,metric)),contextlib.redirect_stdout(stream):
                pilot.repeat_diversity(argparse.Namespace(pilot=run,output=old))
            with (old/'pairwise_diversity.csv').open() as f: oldpair=list(csv.DictReader(f))
            refs={p:float(np.mean([float(r['trajectory_rms_32_m']) for r in oldpair if r['pair_type']==p])) for p in ('sft','grpo','cross')}
            output=root/'r3'
            with patch.object(r3,'REFERENCE',refs),patch.object(r3,'legacy',return_value=(analysis,metric)):
                r3.compute(run,old,output)
            wrong=types.SimpleNamespace(variation=lambda paths:{'world_pairs':[metric.variation(paths)['world_pairs'][0]+.01]})
            with self.assertRaisesRegex(ValueError,'Original RMS regression'):
                r3.inspect_inputs(run,old,metric=wrong)
            self.assertEqual(r3.validate_output(output)['figures'],30)
            with (output/'per_state_outcome_diversity.csv').open() as f: rows=list(csv.DictReader(f))
            self.assertEqual(sum(r['SS_status']=='DEFINED' for r in rows if r['policy']=='sft'),3)
            self.assertEqual(sum(r['SS_status']=='DEFINED' for r in rows if r['policy']=='grpo'),10)
            with (output/'pairwise_labeled.csv').open() as f:
                self.assertEqual(len(list(csv.DictReader(f))),450)
            self.assertEqual(r3.regenerate_figures(output)['status'],'PASS')
            self.assertEqual(verify_seal(output)['status'],'PASS')
            changed=output/'figures'/'paired_rms.png'; changed.write_bytes(b'bad')
            with self.assertRaises(ValueError): r3.validate_output(output)


if __name__=='__main__': unittest.main()
