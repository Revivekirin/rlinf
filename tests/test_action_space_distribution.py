"""CPU tests for exact decoded scalar transport and clustered summaries."""
from pathlib import Path
import sys
import tempfile
import unittest
import numpy as np
from scipy.stats import wasserstein_distance,spearmanr
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/analysis')); sys.path.insert(0,str(ROOT))
from action_space_core import *
from scripts.common.action_schema import validate_saved
from scripts.common.integrity import seal,verify_seal
from phase3b_distribution import distribution_metrics
from token_action_core import episode_means
from action_space_distribution import aggregate_summary


def fixture():
    m=dict(bins=16,token_start=100,token_to_bin_index=np.clip(16-np.arange(16)-1,0,14).tolist(),action_norm_stats=dict(q01=[-2,-1,-.5,-.1,-.2,-.3,0],q99=[2,1,.5,.1,.2,.3,1],mask=[True]*6+[False]))
    ids=np.arange(56).reshape(8,7)%16; z=np.zeros((56,16)); z[np.arange(56),ids.ravel()]=3
    normalized=token_values(m)[ids]; decoded=decode(normalized,m['action_norm_stats'])
    return m,dict(logits=z,action_tokens=ids+100,decoded_actions=decoded,normalized_actions=normalized)


def rows():
    return [dict(origin='sft',episode_id=str(i),suite='suite',task_id=i//2,trial_id=i%2,outcome='both_fail',source_success=False,temperature=1.,group='translation',scope='predicted',x=float(i),y=float(i*i)) for i in range(8)]


class ActionSpaceTests(unittest.TestCase):
    def test_shape_and_finite(self):
        m,z=fixture(); validate_saved(z,m); z['logits'][0,0]=np.inf
        with self.assertRaises(ValueError): validate_saved(z,m)
        with self.assertRaises(ValueError): softmax(np.ones((56,16))*np.nan,1)
        m,z=fixture(); z['logits']=z['logits'][:-1]
        with self.assertRaises(ValueError): validate_saved(z,m)
    def test_decoder_support_and_alias(self):
        m,z=fixture(); support=action_support(m)
        self.assertEqual(support.shape,(7,16)); np.testing.assert_array_equal(support[:,0],support[:,1])
        for c,d in np.ndindex(8,7): self.assertAlmostEqual(support[d,z['action_tokens'][c,d]-100],z['decoded_actions'][c,d])
    def test_probability_temperature(self):
        z=np.array([[1.,2.,3.]]); p=softmax(z,1); np.testing.assert_allclose(p,np.exp(z)/np.exp(z).sum(-1,keepdims=True)); self.assertGreater(entropy(softmax(z,1.6)),entropy(p))
        for t in (0,-1,np.nan):
            with self.assertRaises(ValueError): softmax(z,t)
    def test_moments(self):
        mu,var=moments(np.array([[.25,.75]]),np.array([-1.,3.])); self.assertEqual(mu[0],2); self.assertEqual(var[0],3)
    def test_w1_point_mass(self):
        self.assertEqual(wasserstein1([1.,0],[0,1.],[-2.,3.]),5)
    def test_w1_identical(self):
        p=np.array([[.1,.2,.7]]); self.assertEqual(wasserstein1(p,p,[-1,0,2])[0],0)
    def test_w1_nonmonotonic_duplicate_scipy(self):
        rng=np.random.default_rng(5); x=np.array([2.,-1,2.,.25,0.])
        for _ in range(20):
            p=rng.dirichlet(np.ones(5)); q=rng.dirichlet(np.ones(5)); self.assertAlmostEqual(float(wasserstein1(p,q,x)),wasserstein_distance(x,x,p,q),places=12)
    def test_w1_native_all_dimensions(self):
        m,_=fixture(); x=action_support(m); rng=np.random.default_rng(4); p=softmax(rng.normal(size=(8,7,16)),1.6); q=softmax(rng.normal(size=(8,7,16)),1.6); w=wasserstein1(p,q,x)
        for c,d in np.ndindex(8,7): self.assertAlmostEqual(w[c,d],wasserstein_distance(x[d],x[d],p[c,d],q[c,d]),places=12)
    def test_probability_invalid(self):
        with self.assertRaises(ValueError): moments(np.array([.2,.3]),np.array([0.,1.]))
        with self.assertRaises(ValueError): wasserstein1([1,0],[0,1],[0,np.inf])
    def test_same_argmax_nonzero_js(self):
        p=np.array([[.6,.3,.1]]); q=np.array([[.5,.25,.25]]); d=distribution_metrics(np.log(p),np.log(q)); self.assertFalse(d['greedy_disagreement'][0]); self.assertGreater(d['js'][0],0); self.assertGreater(wasserstein1(p,q,[-1,0,1])[0],0)
    def test_changed_argmax_small_js(self):
        p=np.array([[.5001,.4999]]); q=p[:,::-1]; d=distribution_metrics(np.log(p),np.log(q)); self.assertTrue(d['greedy_disagreement'][0]); self.assertLess(d['js'][0],1e-6)
    def test_entropy_variance_can_differ(self):
        # Lower entropy can coexist with higher action-value variance.
        p=np.array([[0,.5,.5,0]]); q=np.array([[.9,0,0,.1]]); x=np.array([-10.,0.,1.,10.])
        self.assertLess(entropy(q)[0],entropy(p)[0]); self.assertGreater(moments(q,x)[1][0],moments(p,x)[1][0])
    def test_gripper_binary(self):
        p=np.array([[.25,.25,.5]]); x=np.array([0.,.49,1.]); d=gripper_distribution(p,x)
        self.assertEqual(d['p_open'][0],.5); self.assertEqual(d['p_close'][0],.5); self.assertEqual(d['p_other'][0],0); self.assertAlmostEqual(d['binary_entropy'][0],np.log(2)); self.assertEqual(d['threshold_mass'][0],.25)
    def test_gripper_other_never_renormalized(self):
        d=gripper_distribution(np.array([[.2,.3,.5]]),np.array([0.,.5,1.])); self.assertEqual(d['p_other'][0],.3); self.assertTrue(np.isnan(d['binary_entropy'][0])); self.assertAlmostEqual(float((d['p_open']+d['p_close']+d['p_other'])[0]),1)
    def test_binary_entropy_extreme(self):
        self.assertEqual(entropy([[1.,0.]])[0],0); self.assertAlmostEqual(entropy([[.5,.5]])[0],np.log(2))
    def test_weighted_spearman_matches_scipy(self):
        x=np.array([1,1,2,3,4]); y=np.array([1,3,2,4,2]); w=np.array([1,3,2,1,4]); self.assertAlmostEqual(weighted_spearman(x,y,w),spearmanr(np.repeat(x,w),np.repeat(y,w)).statistic)
    def test_bootstrap_reproducible(self):
        r=rows(); a=correlation(r,'x','y',20); b=correlation(r,'x','y',20); self.assertEqual(a,b); self.assertAlmostEqual(a['spearman'],1)
        ta,ca=task_bootstrap(r,20); tb,cb=task_bootstrap(r,20); np.testing.assert_array_equal(ca,cb); np.testing.assert_array_equal(ta,tb)
    def test_quantile_bins_reproducible(self):
        r=rows(); a=quantile_bins(r,'x','y',20); b=quantile_bins(r,'x','y',20); self.assertEqual(a,b); self.assertEqual(sum(k['n_points'] for k in a),8)
    def test_equal_episode_weighting(self):
        r=rows()[:2]; r[0]['y']=0; r[1]['y']=10
        duplicated=[r[0]]*10+[r[1]]; ep=episode_means(duplicated,['y']); summary=aggregate_summary(ep,['y'],20); self.assertEqual(next(v['mean'] for v in summary if v['origin']=='all'),5)
        self.assertAlmostEqual(np.average([v['y'] for v in duplicated],weights=episode_weights(duplicated)),5)
    def test_chunk_pooled(self):
        rng=np.random.default_rng(0); a=rng.normal(size=(20,8,7)); np.testing.assert_allclose(a.mean((1,2)),a.mean(2).mean(1))
    def test_seal(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp); (p/'a').write_text('a'); seal(p,dict(status='PASS')); verify_seal(p); (p/'a').write_text('b')
            with self.assertRaises(ValueError): verify_seal(p)

if __name__=='__main__': unittest.main()
