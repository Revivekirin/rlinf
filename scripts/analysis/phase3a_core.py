"""CPU-only contracts and statistics for saved nominal rollouts."""
from __future__ import annotations
import ast
import collections
import hashlib
import importlib
import json
import math
import sys
from pathlib import Path
import numpy as np

SUITES = ('libero_goal', 'libero_spatial', 'libero_10')
LABELS = ('sft', 'grpo', 'stochastic')
COMPARISONS = (('primary', 'sft', 'grpo', 66),
               ('secondary', 'grpo', 'stochastic', 16),
               ('additional', 'sft', 'stochastic', 82))
CATEGORIES = ('both_success', 'gained', 'lost', 'both_fail')
REVISION = {'sft': 'a4ea91b3897648f3c1395e5221edb40b8f53d1a4',
            'grpo': '473954241c8148bc9a1bf9ed547a74523e4a7d41',
            'stochastic': '473954241c8148bc9a1bf9ed547a74523e4a7d41'}
CONDITIONS = dict(zip(LABELS, ('sft_base_only_deterministic', 'grpo_deterministic', 'grpo_stochastic')))


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(a):
    a = np.ascontiguousarray(a)
    require(not a.dtype.hasobject, 'Object array forbidden')
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(repr(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def protocol_hash(p):
    return hashlib.sha256(json.dumps(p, sort_keys=True).encode()).hexdigest()


def legacy(root):
    """Import original metrics without creating bytecode in the reference repo."""
    sys.dont_write_bytecode = True
    root = Path(root).resolve()
    require((root/'experiments/dataset_analysis/analysis.py').is_file(), 'Missing legacy analysis source')
    sys.path.insert(0, str(root))
    a = importlib.import_module('experiments.dataset_analysis.analysis')
    v = importlib.import_module('experiments.dataset_analysis.task_plots')
    require(Path(a.__file__).resolve().is_relative_to(root), 'Wrong legacy module loaded')
    # Reuse the documented converter without importing robosuite/numba or a simulator.
    source = root/'experiments/dataset_analysis/evidence_sources/robosuite_robosuite_utils_transform_utils.py'
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'quat2axisangle')
    node.decorator_list = []
    scope = {'np': np, 'math': math}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(source), 'exec'), scope)
    a.quat2axisangle = scope['quat2axisangle']
    return a, v


def category(a, b):
    require(type(a) is bool and type(b) is bool, 'Outcome must be bool')
    return ('both_success' if b else 'lost') if a else ('gained' if b else 'both_fail')


def identity(m, mapping):
    s, t, i = m['suite'], m['task_id'], m['trial_id']
    require(s in SUITES and type(t) is int and 0 <= t < 10 and type(i) is int and 0 <= i < 5,
            'Invalid suite/task/trial')
    r = mapping[(s, t)]
    require(m['task_name'] == r['task_name'] and m['libero_130_task_id'] == r['libero_130_task_id'], 'Task ID/name mismatch')
    require(m['initial_state_sha256'] == r['trial_sha256'][i], 'Initial-state hash mismatch')
    require(m['episode_id'] == f'{s}-t{t:02d}-i{i:02d}', 'Episode ID mismatch')
    return (s, t, m['task_name'], r['problem_folder'], r['bddl_file'],
            r['init_states_file'], i, m['initial_state_sha256'], m['simulator_seed'])


def match_records(records, mapping):
    indices = {}
    for label in LABELS:
        index = {}
        for m in records[label]:
            key = identity(m, mapping)
            require(key not in index, f'Duplicate identity in {label}: {key}')
            index[key] = m
        require(len(index) == 150, f'{label}: expected 150 identities')
        indices[label] = index
    require(set(indices['sft']) == set(indices['grpo']) == set(indices['stochastic']), 'Missing/mismatched pairs')
    for key in indices['sft']:
        require(len({indices[l][key]['policy_seed'] for l in LABELS}) == 1, 'Policy seed mismatch')
    return indices


def outcome_stats(rows, seed=3101, iterations=10000):
    require(bool(rows) and iterations > 0, 'Empty statistics / invalid iterations')
    counts = collections.Counter(r['category'] for r in rows)
    gained, lost = counts['gained'], counts['lost']
    discordant = gained + lost
    p = min(1., 2 * sum(math.comb(discordant, j) for j in range(min(gained,lost)+1)) / 2**discordant) if discordant else 1.
    d = np.array([int(r['right_success'])-int(r['left_success']) for r in rows], dtype=float)
    rng = np.random.default_rng(seed)
    means = []
    for start in range(0, iterations, 500):
        n = min(500, iterations-start)
        means.extend(d[rng.integers(len(d), size=(n,len(d)))].mean(1))
    groups = collections.defaultdict(list)
    for r, x in zip(rows, d):
        groups[(r['suite'],r['task_id'])].append(x)
    # Resample whole tasks, stratified by suite; retain all five trials per task.
    strata = collections.defaultdict(list)
    for (suite, task), values in sorted(groups.items()):
        strata[suite].append(values)
    clustered = np.zeros(iterations)
    for values in strata.values():
        sizes = np.array([len(v) for v in values]); sums = np.array([sum(v) for v in values])
        require(len(set(sizes)) == 1, 'Unequal task sizes unsupported for cluster bootstrap')
        sampled = rng.integers(len(values), size=(iterations,len(values)))
        clustered += sums[sampled].sum(1) / len(rows)
    return dict(n=len(rows), **{k:counts[k] for k in CATEGORIES},
                difference_pp=float(100*d.mean()), mcnemar_exact_two_sided_p=p,
                paired_bootstrap_ci95_pp=(100*np.quantile(means,[.025,.975])).tolist(),
                task_cluster_bootstrap_ci95_pp=(100*np.quantile(clustered,[.025,.975])).tolist(),
                bootstrap_seed=seed, iterations=iterations, cluster_count=len(groups),
                cluster_scheme='whole tasks resampled within each suite; trials kept together',
                scope='Descriptive sensitivity for these fixed tasks/states; not a randomly sampled task population. No causal or mechanism claim.')


def canonical(states, actions, metadata, frame='simulator_world', units='m'):
    require(frame == 'simulator_world' and units == 'm', 'Coordinate frame/units mismatch')
    T = metadata['episode_length']
    x = states['robot0_eef_pos']; q = states['robot0_eef_quat']; g = states['robot0_gripper_qpos']
    a = actions['executed_actions']
    require(x.shape == (T+1,3) and q.shape == (T+1,4), 'Invalid EEF shape')
    require(g.shape == (T+1,2) and a.shape == (T,7), 'Gripper/action alignment')
    require(all(np.isfinite(v).all() for v in (x,q,g,a)), 'Nonfinite trajectory')
    require(np.allclose(np.linalg.norm(q,axis=1),1,atol=1e-3), 'Invalid quaternion norm')
    objects = {k[:-4]:v for k,v in states.items() if k.endswith('_pos') and not k.startswith('robot') and '_to_' not in k and v.shape == (T+1,3)}
    events = np.flatnonzero(np.diff(a[:,6]) != 0)+1
    return dict(xyz=x, orientation=q, orientation_representation='quaternion_xyzw', orientation_units='dimensionless',
                gripper=g, command=a[:,6], objects=objects,
                object_quaternions={k:states[k+'_quat'] for k in objects if k+'_quat' in states}, events=events,
                frame=frame, units=units, time='simulator step; T+1 states, T actions; no wall clock')


def choose_cases(rows, count=3):
    selected = []
    for cat in CATEGORIES:
        pool = [r for r in rows if r['category']==cat]
        used_tasks, used_trials = set(), set()
        for _ in range(min(count,len(pool))):
            # Coverage first; stable hash breaks ties without using success magnitude or visual preference.
            row = min(pool, key=lambda r:(r['task_id'] in used_tasks, r['trial_id'] in used_trials,
                hashlib.sha256(r['episode_id'].encode()).hexdigest()))
            selected.append(row); pool.remove(row)
            used_tasks.add(row['task_id']); used_trials.add(row['trial_id'])
    return selected


def select_queries(query_steps, command):
    require(len(query_steps)>0 and np.array_equal(query_steps, np.arange(len(query_steps))*8), 'Query-step alignment')
    chosen = collections.defaultdict(set)
    for fraction, reason in ((0,'initial'),(.2,'early'),(.5,'middle'),(.85,'late')):
        chosen[round(fraction*(len(query_steps)-1))].add(reason)
    for step in np.flatnonzero(np.diff(command)!=0)+1:
        q = int(np.searchsorted(query_steps,step,side='right')-1)
        for i in (q-1,q,q+1):
            if 0 <= i < len(query_steps): chosen[i].add('around_gripper_command_transition')
    return {i:sorted(v) for i,v in sorted(chosen.items())}
