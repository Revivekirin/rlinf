"""Read-only paired SFT deterministic/stochastic outcome and EEF analysis."""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from phase3b import fresh, motion_metrics, seal, sha, sources
from phase3a_core import require
from nominal_behavior import csv_save, read, save

SUITES = ('libero_goal', 'libero_spatial', 'libero_10')
SFT_REVISION = 'a4ea91b3897648f3c1395e5221edb40b8f53d1a4'
STOCHASTIC_COLLECTOR_SHA = 'cbd26f446dd5dc19db5e737b618ac8787563b2292a588dff98c4be6cdae6d7f0'


def protocol_of(run):
    manifest = read(run / 'manifest.json')
    protocol = manifest['protocol']
    actual = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    require(actual == manifest['protocol_hash'], f'Protocol hash mismatch: {run}')
    return protocol, manifest['protocol_hash']


def summarize(rows, keys):
    grouped = collections.defaultdict(list)
    for row in rows:
        grouped[tuple(row[k] for k in keys)].append(row)
    result = []
    for group, entries in sorted(grouped.items()):
        result.append(dict(zip(keys, group), n=len(entries),
            deterministic_success=sum(r['deterministic_success'] for r in entries),
            stochastic_success=sum(r['stochastic_success'] for r in entries),
            gained=sum(r['outcome'] == 'gained' for r in entries),
            lost=sum(r['outcome'] == 'lost' for r in entries),
            both_success=sum(r['outcome'] == 'both_success' for r in entries),
            both_fail=sum(r['outcome'] == 'both_fail' for r in entries),
            deterministic_stagnant=sum(r['deterministic_stagnant'] for r in entries),
            stochastic_stagnant=sum(r['stochastic_stagnant'] for r in entries)))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--research-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    require(__debug__, 'PYTHONOPTIMIZE must be unset')
    root = args.research_root.resolve()
    det = root / 'outputs/nominal/nominal-sft_base_only_deterministic-v1'
    stoch = root / 'outputs/nominal/nominal-sft_base_only_stochastic-v1'
    p_det, h_det = protocol_of(det)
    p_stoch, h_stoch = protocol_of(stoch)
    require(p_det['condition'] == 'sft_base_only_deterministic' and p_stoch['condition'] == 'sft_base_only_stochastic', 'Wrong SFT conditions')
    require(p_det['do_sample'] is False and p_stoch['do_sample'] is True and p_stoch['temperature'] == 1.6, 'Wrong sampling protocol')
    require(p_det['lora_applied'] is False and p_stoch['lora_applied'] is False, 'Unexpected second adapter')
    require(p_det['checkpoint_revision'] == p_stoch['checkpoint_revision'] == SFT_REVISION, 'Wrong SFT revision')
    require(p_stoch['collector_sha256'] == STOCHASTIC_COLLECTOR_SHA, 'Stochastic collector differs from validated run')
    allowed = {'condition', 'sampling_mode', 'do_sample', 'temperature', 'collector_sha256'}
    differences = {key: [p_det.get(key), p_stoch.get(key)] for key in p_det.keys() | p_stoch.keys() if p_det.get(key) != p_stoch.get(key)}
    require(set(differences) <= allowed, f'Additional protocol differences: {differences}')
    require(p_det['trials_per_task'] == p_stoch['trials_per_task'] == 5, 'Expected five initial states')
    require(sha(root / 'scripts/evaluate/nominal_collect.py') == STOCHASTIC_COLLECTOR_SHA, 'Live collector changed')

    sys.path.insert(0, str(root / 'scripts/evaluate'))
    from nominal_storage import episode_id
    from nominal_validate import validate_episode

    expected = {episode_id(s, t, i) for s in SUITES for t in range(10) for i in range(5)}
    output = fresh(args.output)
    input_hashes = {}
    for run in (det, stoch):
        found = {p.name for p in (run / 'rollouts').iterdir() if p.is_dir() and not p.name.startswith('.partial-')}
        require(found == expected, f'Missing or unexpected episodes in {run}: {sorted(expected ^ found)}')
        input_hashes[str(run / 'manifest.json')] = sha(run / 'manifest.json')
        input_hashes[str(run / 'summary.json')] = sha(run / 'summary.json')
        summary = read(run / 'summary.json')
        require(summary['expected'] == summary['completed'] == 150 and not summary['missing_episode_ids'], f'Incomplete summary: {run}')

    pairs = []
    sensitivity = []
    for eid in sorted(expected):
        measurements = {}
        for label, run, protocol_hash in (('deterministic', det, h_det), ('stochastic', stoch, h_stoch)):
            source = run / 'rollouts' / eid
            metadata = validate_episode(source, protocol_hash)
            for filename in ('metadata.json', 'states.npz', 'actions.npz', 'observations.npz'):
                path = source / filename
                input_hashes[str(path)] = sha(path)
            with np.load(source / 'states.npz', allow_pickle=False) as state:
                xyz = state['robot0_eef_pos'].copy()
                initial_pose = {key: state[key][0].copy() for key in ('robot0_eef_pos', 'robot0_eef_quat', 'robot0_gripper_qpos')}
            metric, displacement, _, _ = motion_metrics(xyz)
            with np.load(source / 'actions.npz', allow_pickle=False) as action:
                command_norm = np.linalg.norm(action['executed_actions'][:, :3], axis=1)
            measurements[label] = dict(metadata=metadata, metric=metric, displacement=displacement, initial_pose=initial_pose,
                mean_command_norm=float(command_norm.mean()), stationary_nonzero_commands=int(np.count_nonzero((displacement < 1e-4) & (command_norm > 0))))
        a, b = measurements['deterministic'], measurements['stochastic']
        ma, mb = a['metadata'], b['metadata']
        for key in ('episode_id', 'suite', 'task_id', 'trial_id', 'task_name', 'initial_state_sha256', 'simulator_seed', 'policy_seed', 'checkpoint_revision'):
            require(ma[key] == mb[key], f'Paired identity mismatch: {eid}/{key}')
        for key in a['initial_pose']:
            require(np.array_equal(a['initial_pose'][key], b['initial_pose'][key]), f'Different initial robot pose: {eid}/{key}')
        left, right = bool(ma['success']), bool(mb['success'])
        outcome = 'both_success' if left and right else 'gained' if right else 'lost' if left else 'both_fail'
        row = dict(episode_id=eid, suite=ma['suite'], task_id=ma['task_id'], trial_id=ma['trial_id'], outcome=outcome,
            deterministic_success=left, stochastic_success=right,
            deterministic_stagnant=a['metric']['stagnation'], stochastic_stagnant=b['metric']['stagnation'],
            deterministic_stationary_fraction=a['metric']['stationary_fraction'], stochastic_stationary_fraction=b['metric']['stationary_fraction'],
            deterministic_path_length=a['metric']['path_length'], stochastic_path_length=b['metric']['path_length'],
            deterministic_mean_command_norm=a['mean_command_norm'], stochastic_mean_command_norm=b['mean_command_norm'],
            deterministic_stationary_nonzero_commands=a['stationary_nonzero_commands'], stochastic_stationary_nonzero_commands=b['stationary_nonzero_commands'])
        pairs.append(row)
        for label, measure in measurements.items():
            for threshold_mm in (.05, .1, .2, .5):
                fraction = float((measure['displacement'] < threshold_mm / 1000).mean())
                for cutoff in (.25, .5, .75):
                    sensitivity.append(dict(episode_id=eid, policy=label, threshold_mm=threshold_mm, fraction_cutoff=cutoff,
                        stationary_fraction=fraction, stagnant=fraction >= cutoff))

    for label, run in (('deterministic', det), ('stochastic', stoch)):
        expected_success = sum(r[f'{label}_success'] for r in pairs)
        require(read(run / 'summary.json')['successes'] == expected_success, f'Summary success mismatch: {run}')
    overall = summarize(pairs, ())[0]
    require(overall['deterministic_success'] == 64 and overall['stochastic_success'] == 94, 'Observed SFT totals differ from prior validated results')
    csv_save(output / 'paired_episodes.csv', pairs)
    csv_save(output / 'suite_summary.csv', summarize(pairs, ('suite',)))
    csv_save(output / 'task_summary.csv', summarize(pairs, ('suite', 'task_id')))
    csv_save(output / 'outcome_summary.csv', summarize(pairs, ('outcome',)))
    csv_save(output / 'threshold_sensitivity.csv', sensitivity)
    save(output / 'overall_summary.json', overall)
    save(output / 'manifest.json', dict(conditions=['sft_base_only_deterministic', 'sft_base_only_stochastic'],
        runs=[str(det), str(stoch)], protocol_differences=differences, input_hashes=input_hashes,
        analysis_source_hashes={**sources(), str(Path(__file__)): sha(__file__)},
        definition='step EEF displacement < 0.0001 m for >= 50% of steps; descriptive only',
        interpretation='Same checkpoint and initial states; changed sampling/temperature. New success does not establish a new skill or causal mechanism.'))
    summary = (f"# SFT deterministic → stochastic\n\nSuccess: {overall['deterministic_success']}/150 → {overall['stochastic_success']}/150 "
        f"(net +{overall['stochastic_success'] - overall['deterministic_success']} episodes).\n\n"
        f"Paired gained {overall['gained']}, lost {overall['lost']}, both success {overall['both_success']}, both fail {overall['both_fail']}.\n\n"
        f"Stagnant: {overall['deterministic_stagnant']}/150 → {overall['stochastic_stagnant']}/150.\n\n"
        'Raw arrays and protocols were revalidated; command magnitudes are native action units, not world displacement.\n')
    (output / 'summary.md').write_text(summary)
    for filename, digest in input_hashes.items():
        require(sha(filename) == digest, f'Input changed during analysis: {filename}')
    seal(output, dict(status='PASS',episodes=150,pairs=150,raw_sources_verified=True))
    print(summary)


if __name__ == '__main__':
    main()
