"""CPU contracts for saved-token distribution and command analysis."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/analysis')); sys.path.insert(0,str(ROOT))
from scripts.common.action_schema import *
from scripts.common.integrity import seal,verify_seal
from scripts.common.io import save
from token_action_core import episode_means,initial_states,summarize,clean,check_analysis_copy,load_record
from token_action_analysis import check_outcomes
from phase3b_distribution import distribution_metrics
from phase3b_generality import task_cluster_ci


def fixture():
    m=dict(bins=16,token_start=100,token_to_bin_index=np.clip(16-np.arange(16)-1,0,14).tolist(),action_norm_stats=dict(q01=[-1]*7,q99=[1]*7,mask=[True]*6+[False]))
    ids=np.arange(56).reshape(8,7)%16; logits=np.zeros((56,16)); logits[np.arange(56),ids.ravel()]=3
    n=token_values(m)[ids]
    return m,dict(logits=logits,action_tokens=ids+100,normalized_actions=n,decoded_actions=decode(n,m['action_norm_stats']))


def row(ep='a',value=1,**kw):
    return dict(origin='sft',episode_id=ep,suite='goal',task_id=0,trial_id=0,outcome='both_fail',source_success=False,
                temperature=1.,group='translation',scope='predicted',delta_entropy=value,**kw)


class TokenActionTests(unittest.TestCase):
    def test_saved_schema_and_decoding(self):
        m,z=fixture(); self.assertTrue(validate_saved(z,m))
        for key in z:
            bad=copy.deepcopy(z); bad[key]=bad[key][:-1]
            with self.assertRaises(ValueError): validate_saved(bad,m)
    def test_nonfinite_and_argmax_rejected(self):
        m,z=fixture(); bad=copy.deepcopy(z); bad['logits'][0,0]=np.nan
        with self.assertRaises(ValueError): validate_saved(bad,m)
        z['action_tokens'][0,0]+=1
        with self.assertRaises(ValueError): validate_saved(z,m)
    def test_slot_mapping(self):
        slots=slot_indices(); self.assertEqual(len(slots),56); self.assertEqual(slots[10],(10,1,3,'rx')); self.assertEqual(slots[-1],(55,7,6,'gripper'))
    def test_clipped_token_alias(self):
        m,_=fixture(); vals=token_values(m); self.assertEqual(vals[0],vals[1]); self.assertTrue(np.all(np.diff(vals)<=0))
        m['token_to_bin_index'][0]=0
        with self.assertRaises(ValueError): token_values(m)
    def test_entropy_kl_js_known_values(self):
        p=np.array([[.1,.2,.7]]); q=np.array([[.3,.3,.4]]); d=distribution_metrics(np.log(p),np.log(q))
        self.assertAlmostEqual(d['entropy_sft'][0],-(p*np.log(p)).sum())
        self.assertAlmostEqual(d['kl_sft_grpo'][0],(p*np.log(p/q)).sum())
        mid=(p+q)/2; self.assertAlmostEqual(d['js'][0],.5*((p*np.log(p/mid)).sum()+(q*np.log(q/mid)).sum()))
        self.assertAlmostEqual(d['kl_grpo_sft'][0],(q*np.log(q/p)).sum())
    def test_temperature_and_topk(self):
        logits=np.arange(16,dtype=float)[None,:]; d=distribution_metrics(logits,logits); high=distribution_metrics(logits,logits,1.6)
        p=np.exp(logits-logits.max()); p/=p.sum()
        for k in (1,5,10): self.assertAlmostEqual(d['top'+str(k)+'_sft'][0],p[0,-k:].sum())
        self.assertGreater(high['entropy_sft'][0],d['entropy_sft'][0]); self.assertEqual(d['js'][0],0)
    def test_greedy_disagreement(self):
        d=distribution_metrics(np.array([[3,1],[1,3]]),np.array([[1,3],[1,3]])); np.testing.assert_array_equal(d['greedy_disagreement'],[True,False])
    def test_translation_norm(self):
        v=vector_metrics([3,4,0],[0,0,5]); self.assertEqual(v['magnitude_sft'],5); self.assertEqual(v['magnitude_delta'],0); self.assertAlmostEqual(float(v['difference_l2']),np.sqrt(50)); self.assertEqual(v['cosine'],0)
    def test_near_zero(self):
        v=vector_metrics([[0,0,0],[.01,0,0],[1,0,0]],[[1,0,0]]*3,.01)
        self.assertTrue(np.isnan(v['cosine'][:2]).all()); self.assertEqual(v['cosine'][2],1); self.assertIsNone(clean(v['cosine'][0]))
    def test_rotation_scope(self):
        self.assertIn('UNKNOWN',contract()['rotation']); self.assertIn('no SO(3)',contract()['rotation'])
    def test_gripper_sign(self):
        np.testing.assert_array_equal(prepared_gripper([-.9,0,.49,.5,.51,1]),[1,1,1,0,-1,-1])
    def test_equal_episode_weight(self):
        ep=episode_means([row('a',1)]*10+[row('b',3)],['delta_entropy'])
        s=summarize(ep,['delta_entropy']); allrow=next(r for r in s if r['origin']=='all')
        self.assertEqual(allrow['mean'],2); self.assertEqual(allrow['task_macro_mean'],2)
    def test_task_macro_separate(self):
        rows=[row('a',0),row('b',0),dict(row('c',6),task_id=1)]
        s=next(r for r in summarize(rows,['delta_entropy']) if r['origin']=='all')
        self.assertEqual(s['mean'],2); self.assertEqual(s['task_macro_mean'],3)
    def test_alignment_rejected(self):
        with self.assertRaises(ValueError): episode_means([row(),dict(row(),outcome='gained')],['delta_entropy'])
    def test_initial_state_pairing(self):
        ep=[row(value=1),dict(row(value=3),origin='grpo')]
        self.assertEqual(initial_states(ep,['delta_entropy'])[0]['delta_entropy'],2)
    def test_outcomes(self):
        e=dict(source_policy='sft',source_success=False,outcome='gained',source_episode='a',suite='x',task_id=0,trial_id=0,initial_state_sha256='abc')
        check_outcomes([e,dict(e,source_policy='grpo',source_success=True)])
        with self.assertRaises(ValueError): check_outcomes([dict(e,source_success=True)])
    def test_bootstrap_reproducible(self):
        rows=[dict(row(str(i),i),task_id=i) for i in range(4)]
        a=task_cluster_ci(rows,iterations=50); b=task_cluster_ci(rows,iterations=50)
        self.assertEqual(a,b); self.assertEqual(a['clusters'],4); self.assertLessEqual(a['ci95'][0],1.5)
    def test_missing_record(self):
        with tempfile.TemporaryDirectory() as t:
            with self.assertRaises(FileNotFoundError): load_record(Path(t),'sft',{'observation_id':'missing'}, {})
    def test_seal_tamper(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); (p/'a').write_text('a'); seal(p,dict(status='PASS')); verify_seal(p); (p/'a').write_text('changed')
            with self.assertRaises(ValueError): verify_seal(p)
    def test_missing_figures_not_numeric(self):
        with tempfile.TemporaryDirectory() as t:
            p=Path(t); (p/'figures').mkdir(); (p/'figures/a.png').write_bytes(b'pic'); (p/'metrics.json').write_text('{}'); seal(p,dict(status='PASS'))
            (p/'figures/a.png').unlink(); self.assertEqual(check_analysis_copy(p)['status'],'PARTIAL_COPY_FIGURES_MISSING')
            (p/'metrics.json').unlink()
            with self.assertRaises(ValueError): check_analysis_copy(p)

if __name__=='__main__': unittest.main()
