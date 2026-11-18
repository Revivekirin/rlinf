"""CPU contracts for full nominal common-observation generality."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'scripts/analysis'))
from phase3b_generality import (TEMPERATURES, _bank_readback, _check_worker_record, build_bank, worker,
    _plot_inputs, aggregate_episodes, aggregate_groups, figures, observation_metric_rows,
    report, task_cluster_ci, uniform_queries)
from phase3b_distribution import capture_native_logits
from phase3a_core import array_hash, category, sha
from scripts.common.io import csv_save, read, save
from scripts.common.integrity import seal


class GeneralityTests(unittest.TestCase):
    def test_uniform_selection_and_ids(self):
        self.assertEqual(uniform_queries(1),[0])
        self.assertEqual(uniform_queries(7),list(range(7)))
        self.assertEqual(uniform_queries(65),[0,7,14,21,28,35,42,49,56,64])
        self.assertEqual(len(set(uniform_queries(65))),10)

    def test_metrics_temperature_and_equal_episode(self):
        e=dict(observation_id='sft-libero_goal-t00-i00-q0000',source_episode='libero_goal-t00-i00',
               suite='libero_goal',task_id=0,trial_id=0,source_policy='sft',
               outcome=category(False,True),source_success=False,query_index=0,simulator_step=0)
        a=np.zeros((56,4)); b=np.tile([3.,0.,0.,0.],(56,1))
        rows=observation_metric_rows(e,a,b)
        self.assertEqual(len(rows),2)
        self.assertLess(rows[0]['delta_entropy'],0)
        self.assertLess(abs(rows[1]['delta_entropy']),abs(rows[0]['delta_entropy']))
        self.assertGreater(rows[0]['js'],0)
        self.assertGreaterEqual(rows[0]['kl_sft_grpo'],0)
        self.assertGreaterEqual(rows[0]['kl_grpo_sft'],0)
        other=dict(rows[0],observation_id='second',query_index=1,delta_entropy=1.,
                   entropy_sft=1.,entropy_grpo=2.)
        episodes=aggregate_episodes([rows[0],other])
        self.assertEqual(len(episodes),1)
        self.assertAlmostEqual(episodes[0]['delta_entropy'],(rows[0]['delta_entropy']+1)/2)
        self.assertEqual(episodes[0]['n_observations'],2)

    def test_task_cluster_bootstrap_and_grouping(self):
        rows=[]
        for suite in ('libero_goal','libero_spatial','libero_10'):
            for task in range(2):
                for origin in ('sft','grpo'):
                    for trial in range(2):
                        rows.append(dict(suite=suite,task_id=task,origin=origin,
                            episode_id=f'{suite}-t{task:02d}-i{trial:02d}',outcome='both_fail',
                            temperature=1.6,n_observations=1,delta_entropy=float(task-1),
                            entropy_sft=2.,entropy_grpo=float(task+1),delta_top1=.1,js=.2,
                            greedy_disagreement=.3))
        a=task_cluster_ci(rows,seed=31,iterations=100)
        self.assertEqual(a,task_cluster_ci(rows,seed=31,iterations=100))
        self.assertEqual(a['clusters'],6)
        grouped=aggregate_groups(rows)
        overall_tasks=[r for r in grouped['tasks'] if r['origin']=='all']
        self.assertEqual(len(overall_tasks),6)
        self.assertTrue(all(r['source_episodes']==4 for r in overall_tasks))

    def test_bank_raw_readback_and_worker_checksum(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); source=root/'source'; source.mkdir(); bank=root/'run/bank/observations'; bank.mkdir(parents=True)
            raw=np.array([[[1.,2.]],[[3.,4.]]])
            np.savez_compressed(source/'observations.npz',query_step=np.array([0,8]),pixel_values=raw)
            save(source/'metadata.json',dict(episode_id='case'))
            selected=raw[1]; np.savez_compressed(bank/'one.npz',pixel_values=selected)
            entry=dict(observation_id='one',source_policy='sft',source_episode='case',query_index=1,
                simulator_step=8,source=str(source/'observations.npz'),source_sha256=sha(source/'observations.npz'),
                source_metadata_sha256=sha(source/'metadata.json'),file='observations/one.npz',
                file_sha256=sha(bank/'one.npz'),arrays=dict(pixel_values=dict(sha256=array_hash(selected),
                    shape=list(selected.shape),dtype=str(selected.dtype))))
            save(root/'run/bank/manifest.json',dict(observations=[entry]))
            self.assertEqual(_bank_readback(root/'run')['observations'],1)
            np.savez_compressed(source/'observations.npz',query_step=np.array([0,8]),pixel_values=raw+1)
            with self.assertRaises(ValueError): _bank_readback(root/'run')
            work=root/'worker/records/one'; work.mkdir(parents=True)
            logits=np.zeros((56,4)); actions=np.zeros((8,7))
            np.savez_compressed(work/'logits.npz',logits=logits,action_tokens=np.zeros((8,7),dtype=int),
                                decoded_actions=actions,normalized_actions=actions)
            save(work/'record.json',dict(observation_id='one',model='sft',input_checksum=entry['file_sha256'],
                 bank_validation_sha256='bank',file_sha256=sha(work/'logits.npz')))
            manifest=dict(bank_validation_sha256='bank',bins=4,token_start=0)
            self.assertEqual(_check_worker_record(root/'worker',entry,'sft',manifest)['observation_id'],'one')
            (work/'logits.npz').write_bytes(b'changed')
            with self.assertRaises(ValueError): _check_worker_record(root/'worker',entry,'sft',manifest)

    def test_shared_native_capture_contract_on_cpu_mock(self):
        import torch
        from types import SimpleNamespace
        class LanguageModel:
            hook=None
            def register_forward_hook(self,hook):
                self.hook=hook
                class Handle:
                    def remove(inner): self.hook=None
                return Handle()
        language=LanguageModel()
        core=SimpleNamespace(vocab_size=4,config=SimpleNamespace(n_action_bins=4),
            action_dim=7,num_action_chunks=8,language_model=language)
        class Model:
            def predict_action_batch(self,**kwargs):
                logits=torch.zeros((1,57,4)); logits[:,:,3]=2
                if language.hook is not None:
                    language.hook(language,(),SimpleNamespace(logits=logits))
                fi=dict(action_tokens=torch.full((1,8,7),3),input_ids=torch.ones((1,2),dtype=torch.long),
                    attention_mask=torch.ones((1,2),dtype=torch.long),pixel_values=torch.ones((1,3,2,2)))
                return torch.zeros((1,8,7)),dict(forward_inputs=fi)
        bank=dict(main_image_policy=np.zeros((2,2,3),dtype=np.uint8),
                  proprio_policy=np.zeros(8,dtype=np.float32))
        logits,actions,fi,lo,hi=capture_native_logits(Model(),core,bank,'task',3101,True)
        self.assertEqual(logits.shape,(56,4))
        self.assertEqual(actions.shape,(8,7))
        self.assertEqual((lo,hi),(0,4))
        self.assertTrue(np.array_equal(fi['action_tokens'],np.full((8,7),3)))

    def test_full_primary_bank_cpu_fixture(self):
        import argparse
        from phase3a_core import SUITES
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); phase3a=root/'phase3a'; (phase3a/'paired').mkdir(parents=True)
            audit=root/'audit'; audit.mkdir(); save(audit/'validation.json',dict(status='PASS'))
            save(phase3a/'validation.json',dict(status='PASS'))
            mapping=[]; pairs=[]; records={'sft':[],'grpo':[]}
            for suite in SUITES:
                for task in range(10):
                    states=[f'{suite}-t{task:02d}-state-{i}' for i in range(5)]
                    mapping.append(dict(suite=suite,subset_task_id=task,trial_sha256=states))
                    for trial in range(5):
                        eid=f'{suite}-t{task:02d}-i{trial:02d}'
                        sft_success=trial%2==0; grpo_success=trial%3==0
                        pairs.append(dict(comparison='primary',episode_id=eid,suite=suite,task_id=task,
                            trial_id=trial,initial_state_sha256=states[trial],
                            category=category(sft_success,grpo_success)))
                        for origin,success in (('sft',sft_success),('grpo',grpo_success)):
                            source=root/origin/eid; source.mkdir(parents=True)
                            save(source/'metadata.json',dict(episode_id=eid))
                            np.savez_compressed(source/'observations.npz',query_step=np.array([0]),
                                main_image_policy=np.zeros((1,2,2,3),dtype=np.uint8),
                                proprio_policy=np.zeros((1,8),dtype=np.float32),
                                pixel_values=np.zeros((1,3,2,2),dtype=np.float32),
                                input_ids=np.zeros((1,1,2),dtype=np.int64),
                                attention_mask=np.ones((1,1,2),dtype=np.int64))
                            records[origin].append(dict(episode_id=eid,suite=suite,task_id=task,
                                trial_id=trial,initial_state_sha256=states[trial],source_dir=str(source),
                                num_queries=1,language='put the object',success=success))
            save(root/'mapping.json',dict(tasks=mapping))
            save(phase3a/'paired/pairs.json',pairs)
            info=dict(records=records,manifest=dict(mapping=str(root/'mapping.json'),source_files={},
                protocols={m:dict(upstream_commit='pinned') for m in ('sft','grpo')}))
            args=argparse.Namespace(origins=['sft','grpo'],max_queries=10,phase3a=phase3a,audit=audit,
                                    research_root=ROOT,output=root/'run')
            with patch('phase3b_generality.verify_phase3a',return_value=info), \
                 patch('phase3b_generality.require_server_audit'), \
                 patch('phase3b_generality.nominal_inventory',return_value={'fixture':dict(completed_episode_dirs=150)}):
                build_bank(args)
            bank=read(root/'run/bank/manifest.json')
            self.assertEqual(len(bank['source_episodes']),300)
            self.assertEqual(len(bank['observations']),300)
            self.assertEqual(len({e['observation_id'] for e in bank['observations']}),300)
            self.assertEqual(_bank_readback(root/'run')['status'],'PASS')

    def test_worker_chunk_resume_checks_committed_logits(self):
        import argparse
        from types import SimpleNamespace
        from phase3b_generality import STAGE_SOURCE
        with tempfile.TemporaryDirectory() as d:
            run=Path(d); bank=run/'bank'; (bank/'observations').mkdir(parents=True)
            entries=[]
            for i in range(2):
                oid=f'observation-{i}'
                arrays=dict(main_image_policy=np.zeros((2,2,3),dtype=np.uint8),
                    proprio_policy=np.zeros(8,dtype=np.float32),
                    pixel_values=np.zeros((3,2,2),dtype=np.float32),
                    input_ids=np.zeros((1,2),dtype=np.int64),
                    attention_mask=np.ones((1,2),dtype=np.int64))
                path=bank/'observations'/(oid+'.npz'); np.savez_compressed(path,**arrays)
                entries.append(dict(observation_id=oid,file='observations/'+oid+'.npz',
                    file_sha256=sha(path),language='task',arrays={k:dict(sha256=array_hash(v)) for k,v in arrays.items()}))
            save(bank/'manifest.json',dict(phase3a=str(run/'phase3a'),audit=str(run/'audit'),
                observations=entries,stage_source_sha256=sha(STAGE_SOURCE)))
            seal(bank,dict(status='PASS',observations=2))
            save(run/'run_manifest.json',dict(bank_validation_sha256=sha(bank/'validation.json'),
                stage_source_sha256=sha(STAGE_SOURCE)))
            class Core:
                vocab_size=4
                config=SimpleNamespace(n_action_bins=4)
                action_dim=7
                num_action_chunks=8
                bin_centers=np.arange(4)
                def _get_action_stats(self): return {}
            calls=[]
            def capture(model,core,raw,language,seed,check_equivalence=False):
                calls.append(seed)
                logits=np.zeros((56,4)); logits[:,3]=1
                fi=dict(action_tokens=np.full((8,7),3),input_ids=raw['input_ids'],
                        attention_mask=raw['attention_mask'],pixel_values=raw['pixel_values'])
                return logits,np.zeros((8,7)),fi,0,4
            args=argparse.Namespace(run=run,model='sft',smoke=False,resume=False,max_new=1,
                                    research_root=ROOT)
            with patch('phase3b_generality.require_server_audit'), \
                 patch('phase3b_distribution.load_model',return_value=(object(),Core(),dict(model_path='mock'))), \
                 patch('phase3b_generality.capture_native_logits',capture), \
                 patch('torch.cuda.mem_get_info',return_value=(1000000,2000000)):
                worker(args)
                self.assertFalse((run/'workers/sft/validation.json').exists())
                args.resume=True; args.max_new=0
                worker(args)
                self.assertEqual(calls,[3101,3102])
                worker(args)
                self.assertEqual(calls,[3101,3102])
                (run/'workers/sft/records/observation-0/logits.npz').write_bytes(b'corrupt')
                with self.assertRaises(ValueError): worker(args)

    def test_figure_coverage_png_pdf_and_report(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d); analysis=root/'analysis'; analysis.mkdir()
            obs=[]
            for suite in ('libero_goal','libero_spatial','libero_10'):
                for task in range(10):
                    for trial in range(5):
                        for origin in ('sft','grpo'):
                            e=dict(observation_id=f'{origin}-{suite}-t{task:02d}-i{trial:02d}-q0000',
                                source_episode=f'{suite}-t{task:02d}-i{trial:02d}',suite=suite,
                                task_id=task,trial_id=trial,source_policy=origin,outcome='both_fail',
                                source_success=False,query_index=0,simulator_step=0)
                            x=float(task)/10+float(trial)/100
                            a=np.tile([x,0.,0.,0.],(56,1)); b=np.tile([x+.2,0.,0.,0.],(56,1))
                            for row in observation_metric_rows(e,a,b):
                                row['processed_inputs_equal']=True
                                obs.append(row)
            episodes=aggregate_episodes(obs); groups=aggregate_groups(episodes)
            csv_save(analysis/'observations.csv',obs); csv_save(analysis/'episodes.csv',episodes)
            for name,rows in groups.items(): csv_save(analysis/(name+'.csv'),rows)
            overall={}
            for t in TEMPERATURES:
                rr=[r for r in episodes if r['temperature']==t]
                overall[str(t)]=dict(mean_delta_entropy=float(np.mean([r['delta_entropy'] for r in rr])),
                    median_delta_entropy=float(np.median([r['delta_entropy'] for r in rr])),
                    p25_delta_entropy=float(np.quantile([r['delta_entropy'] for r in rr],.25)),
                    p75_delta_entropy=float(np.quantile([r['delta_entropy'] for r in rr],.75)),
                    fraction_negative=float(np.mean([r['delta_entropy']<0 for r in rr])),
                    episode_fraction_positive=float(np.mean([r['delta_entropy']>0 for r in rr])),
                    observation_fraction_positive=float(np.mean([r['delta_entropy']>0 for r in rr])),
                    observation_count=len(rr),
                    macro_task_mean_delta_entropy=0.,mean_delta_top1=0.,mean_js=0.,
                    task_cluster_bootstrap=task_cluster_ci(rr,iterations=20))
            save(analysis/'metrics.json',dict(scope='primary',source_episodes=300,observations=300,
                 processed_inputs_equal=300,action_normalization_equal=True,overall=overall))
            import argparse
            self.assertEqual(len(_plot_inputs(analysis,1.6)[0]),300)
            figures(argparse.Namespace(run=root))
            index=read(analysis/'figure_index.json')
            self.assertEqual(len(index['stems']),12)
            self.assertTrue(all((analysis/'figures'/(stem+'.'+ext)).is_file()
                for stem in index['stems'] for ext in ('png','pdf')))
            report(argparse.Namespace(run=root))
            self.assertEqual(read(analysis/'validation.json')['status'],'PASS')


if __name__=='__main__': unittest.main()
