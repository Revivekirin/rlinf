"""Build SYNTHETIC test-only saved rollouts. Never a research input generator."""
from pathlib import Path
import json
import shutil
import sys
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts/analysis'))
import phase3a_core as c
import nominal_behavior as app


def build(root, expert_root):
    root=Path(root); root.mkdir(parents=True,exist_ok=False)
    (root/'SYNTHETIC_TEST_ONLY').write_text('Synthetic unit/integration fixtures, not collected robot data.\n')
    evaluate=root/'scripts/evaluate'; evaluate.mkdir(parents=True)
    source=Path(__file__).resolve().parents[1]/'scripts/evaluate'
    for name in ('nominal_validate.py','nominal_storage.py','nominal_collect.py'):
        shutil.copyfile(source/name,evaluate/name)
    site=root/'upstream/RLinf/.venv-test/lib/python3.12/site-packages'
    robot=site/'robosuite/robots'; robot.mkdir(parents=True)
    shutil.copyfile(expert_root/'experiments/dataset_analysis/evidence_sources/robosuite_robosuite_robots_single_arm.py',robot/'single_arm.py')
    objects=site/'libero/libero/envs';objects.mkdir(parents=True)
    shutil.copyfile(expert_root/'vendor/LIBERO/libero/libero/envs/bddl_base_domain.py',objects/'bddl_base_domain.py')
    reference=json.loads((expert_root/'outputs/stage1_review/dataset_analysis/analysis.json').read_text())
    tasks={suite:sorted(reference['suites'][alias]['tasks']['train']) for suite,alias in zip(c.SUITES,('goal','spatial','long'))}
    mapping={'schema_version':1,'trials_compared':5,'tasks':[]}
    for suite in c.SUITES:
        for t in range(10):
            mapping['tasks'].append(dict(suite=suite,subset_task_id=t,libero_130_task_id=c.SUITES.index(suite)*10+t,
                task_name=tasks[suite][t],problem_folder=suite,bddl_file=f'{t}.bddl',init_states_file=f'{t}.pruned',
                trial_sha256=[str(i)*64 for i in range(5)]))
    mp=root/'mapping.json';app.save(mp,mapping)
    counts={'sft':(30,27,7),'grpo':(46,48,36),'stochastic':(49,49,48)}
    for label in c.LABELS:
        run=root/label;run.mkdir()
        p=dict(condition=c.CONDITIONS[label],checkpoint_repo='RLinf/RLinf-OpenVLAOFT-LIBERO-130'+('-Base-Lora' if label=='sft' else ''),
            checkpoint_revision=c.REVISION[label],checkpoint_path='/TEST_ONLY/'+label,
            collector_sha256=c.sha(evaluate/'nominal_collect.py'),mapping_sha256=c.protocol_hash(mapping),
            sampling_mode='stochastic' if label=='stochastic' else 'deterministic',temperature=1.6 if label=='stochastic' else 1.,
            do_sample=label=='stochastic',lora_applied=False,horizon=520,action_chunks=8,trials_per_task=5,
            seed=0,num_steps_wait=15,unnorm_key='libero_130_no_noops_trajall',
            upstream_commit='e05f71e80d14ab4b55e5027ee739cfbcc6b12b9d',model_config={'model_path':'/TEST_ONLY/'+label})
        ph=c.protocol_hash(p);app.save(run/'manifest.json',dict(protocol=p,protocol_hash=ph))
        for row in mapping['tasks']:
            suite=row['suite'];t=row['subset_task_id']
            for i in range(5):
                success=t*5+i<counts[label][c.SUITES.index(suite)]
                T=9 if success else 520;Q=(T+7)//8;eid=f'{suite}-t{t:02d}-i{i:02d}'
                dest=run/'rollouts'/eid;dest.mkdir(parents=True)
                m=dict(complete=True,episode_id=eid,run_id=label,protocol_hash=ph,suite=suite,task_id=t,
                    libero_130_task_id=row['libero_130_task_id'],task_name=row['task_name'],trial_id=i,
                    initial_state_sha256=row['trial_sha256'][i],simulator_seed=0,
                    policy_seed=100000*c.SUITES.index(suite)+100*t+i,
                    success=success,episode_length=T,num_queries=Q,termination_reason='success' if success else 'horizon',
                    language='SYNTHETIC TEST ONLY',wait_steps=15)
                for k in ('checkpoint_repo','checkpoint_revision','collector_sha256','lora_applied','sampling_mode','temperature','horizon','action_chunks','unnorm_key','upstream_commit'):m[k]=p[k]
                app.save(dest/'metadata.json',m)
                x=np.stack((np.linspace(0,.2,T+1),np.sin(np.linspace(0,3,T+1))*.1,np.ones(T+1)),axis=1)
                np.savez_compressed(dest/'states.npz',robot0_eef_pos=x,robot0_eef_quat=np.tile([0.,0,0,1],(T+1,1)),
                    robot0_gripper_qpos=np.ones((T+1,2))*.01,object_1_pos=np.tile([.2,0,1],(T+1,1)))
                decoded=np.zeros((Q,8,7));decoded[:,:,6]=1
                actions=np.zeros((T,7));actions[:,6]=-1
                np.savez_compressed(dest/'actions.npz',raw_action_tokens=np.full((Q,8,7),31800,dtype=np.int64),
                    decoded_action_chunks=decoded,executed_actions=actions,executed_query_index=np.arange(T)//8,
                    executed_action_slot=np.arange(T)%8,simulator_step=np.arange(T))
                np.savez_compressed(dest/'observations.npz',query_step=np.arange(Q)*8,
                    main_image_raw=np.zeros((Q,2,2,3),dtype=np.uint8),main_image_policy=np.zeros((Q,2,2,3),dtype=np.uint8),
                    pixel_values=np.zeros((Q,6,2,2)),proprio_policy=np.zeros((Q,8)),
                    input_ids=np.zeros((Q,128),dtype=np.int64),attention_mask=np.ones((Q,128),dtype=bool))
    return root


if __name__=='__main__':
    build(Path(sys.argv[1]),Path(sys.argv[2]))
