"""Read-only regression checks against the validated Phase 3B artifacts, when present."""

import collections
import csv
import json
import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/analysis'))
from phase3b import motion_metrics, verify_seal
from phase3b_distribution import distribution_metrics

RUN = 'phase3b-20261117T111837Z-2822692'
BASE = ROOT / 'outputs/analysis'
P3A = BASE / 'nominal_behavior/phase3a-20261109T082746Z-1415246'
STAG = BASE / 'stagnation' / RUN
SFT = BASE / 'sft_sampling' / RUN
DIST = BASE / 'policy_distribution' / RUN


def rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


class SavedRegressionTests(unittest.TestCase):
    def setUp(self):
        if not all((p / 'validation.json').is_file() for p in (STAG, SFT, DIST)):
            self.skipTest('Validated Phase 3B artifacts are unavailable')

    def test_seals_and_saved_outcomes(self):
        for path in (STAG, SFT, DIST):
            self.assertEqual(verify_seal(path)['status'], 'PASS')
        sft = json.loads((SFT / 'overall_summary.json').read_text())
        self.assertEqual({k: sft[k] for k in ('deterministic_success', 'stochastic_success',
                         'deterministic_stagnant', 'stochastic_stagnant', 'gained', 'lost',
                         'both_success', 'both_fail')},
                         dict(deterministic_success=64, stochastic_success=94,
                              deterministic_stagnant=61, stochastic_stagnant=0,
                              gained=43, lost=13, both_success=51, both_fail=43))
        paired = rows(SFT / 'paired_episodes.csv')
        self.assertEqual(len(paired), 150)
        self.assertEqual(collections.Counter(r['outcome'] for r in paired),
                         dict(gained=43, lost=13, both_success=51, both_fail=43))
        overall = {r['comparison']: r for r in rows(STAG / 'overall_summary.csv')}
        self.assertEqual({k: int(overall['primary'][k]) for k in
                         ('left_stagnant', 'right_stagnant', 'both_stagnant', 'resolved', 'appeared')},
                         dict(left_stagnant=61, right_stagnant=8, both_stagnant=5,
                              resolved=56, appeared=3))
        self.assertEqual({k: int(overall['secondary'][k]) for k in
                         ('left_stagnant', 'right_stagnant', 'resolved', 'appeared')},
                         dict(left_stagnant=8, right_stagnant=0, resolved=8, appeared=0))

    def test_stagnation_and_threshold_rows(self):
        if not (P3A / 'trajectories/canonical').is_dir():
            self.skipTest('Phase 3A canonical trajectories are unavailable')
        saved = rows(STAG / 'per_episode.csv')
        threshold = {(r['policy'], r['episode_id'], float(r['threshold_mm']),
                      float(r['fraction_cutoff'])): r for r in rows(STAG / 'threshold_per_episode.csv')}
        self.assertEqual(len(saved), 450)
        for row in saved:
            path = P3A / 'trajectories/canonical' / row['policy'] / (row['episode_id'] + '.npz')
            with np.load(path, allow_pickle=False) as z:
                metric, displacement, _, _ = motion_metrics(z['xyz'])
            for key in ('steps', 'stationary_steps', 'stationary_runs', 'stationary_longest_run'):
                self.assertEqual(int(row[key]), metric[key], (row['policy'], row['episode_id'], key))
            for key in ('path_length', 'stationary_fraction', 'net_displacement'):
                self.assertAlmostEqual(float(row[key]), metric[key], places=12)
            self.assertEqual(row['stagnation'], str(metric['stagnation']))
            for mm in (.05, .1, .2, .5):
                fraction = float((displacement < mm / 1000).mean())
                for cutoff in (.25, .5, .75):
                    old = threshold[(row['policy'], row['episode_id'], mm, cutoff)]
                    self.assertAlmostEqual(float(old['stationary_fraction']), fraction, places=12)
                    self.assertEqual(old['stagnation'], str(fraction >= cutoff))

    def test_distribution_token_and_summary(self):
        observations = rows(DIST / 'observations.csv')
        self.assertEqual(len(observations), 260)
        self.assertEqual(len({(r['origin'], r['episode_id']) for r in observations}), 18)
        tokens = rows(DIST / 'token_metrics.csv')
        self.assertEqual(len(tokens), 260 * 2 * 56)
        by_observation = collections.defaultdict(list)
        for row in tokens:
            by_observation[(row['observation_id'], float(row['temperature']))].append(row)
        action_rows = collections.defaultdict(list)
        for row in rows(DIST / 'action_metrics.csv'):
            action_rows[(row['observation_id'], row['space'])].append(row)
        action_groups = collections.defaultdict(lambda: collections.defaultdict(list))
        keys = ('entropy_sft', 'entropy_grpo', 'top1_sft', 'top1_grpo',
                'greedy_disagreement', 'kl_sft_grpo', 'kl_grpo_sft', 'js',
                'top5_sft', 'top5_grpo', 'top10_sft', 'top10_grpo')
        grouped = collections.defaultdict(lambda: collections.defaultdict(list))
        for obs in observations:
            oid = obs['observation_id']
            with np.load(DIST / 'sft' / (oid + '.npz'), allow_pickle=False) as a, \
                 np.load(DIST / 'grpo' / (oid + '.npz'), allow_pickle=False) as b:
                for space, key in (('native_decoded', 'decoded_actions'),
                                   ('normalized_bins', 'normalized_actions')):
                    delta = b[key] - a[key]
                    old_actions = action_rows[(oid, space)]
                    self.assertEqual(len(old_actions), 8)
                    for row in old_actions:
                        slot = int(row['chunk_slot'])
                        current_action = dict(
                            translation_l2=float(np.linalg.norm(delta[slot, :3])),
                            rotation_l2=float(np.linalg.norm(delta[slot, 3:6])),
                            gripper_abs=float(abs(delta[slot, 6])),
                            executed_gripper_disagreement=float(
                                np.sign(2*a['decoded_actions'][slot, 6]-1) !=
                                np.sign(2*b['decoded_actions'][slot, 6]-1)))
                        for metric, value in current_action.items():
                            saved = float(row[metric]) if metric != 'executed_gripper_disagreement' \
                                else float(row[metric] == 'True')
                            self.assertAlmostEqual(saved, value, places=11)
                        group = (obs['origin'], obs['outcome'], space)
                        action_groups[group][obs['episode_id']].append(current_action)
                for temp in (1.0, 1.6):
                    current = distribution_metrics(a['logits'], b['logits'], temp)
                    old = by_observation[(oid, temp)]
                    self.assertEqual(len(old), 56)
                    for row in old:
                        pos = int(row['position'])
                        for key in keys:
                            self.assertAlmostEqual(float(row[key]), float(current[key][pos]), places=11)
                    group = (obs['origin'], obs['outcome'], temp)
                    grouped[group][obs['episode_id']].append({k: float(np.mean(current[k])) for k in keys})
        for row in rows(DIST / 'group_summary.csv'):
            episodes = grouped[(row['origin'], row['outcome'], float(row['temperature']))]
            self.assertEqual(int(row['episodes']), len(episodes))
            self.assertEqual(int(row['observations']), sum(map(len, episodes.values())))
            for key in keys:
                value = float(np.mean([np.mean([r[key] for r in er]) for er in episodes.values()]))
                self.assertAlmostEqual(float(row[key]), value, places=11)
        for row in rows(DIST / 'action_group_summary.csv'):
            episodes = action_groups[(row['origin'], row['outcome'], row['space'])]
            self.assertEqual(int(row['episodes']), len(episodes))
            for key in ('translation_l2', 'rotation_l2', 'gripper_abs',
                        'executed_gripper_disagreement'):
                value = float(np.mean([np.mean([r[key] for r in er]) for er in episodes.values()]))
                self.assertAlmostEqual(float(row[key]), value, places=11)


if __name__ == '__main__':
    unittest.main()
