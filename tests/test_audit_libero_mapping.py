import unittest
from dataclasses import dataclass

import numpy as np

from scripts.diagnostics.audit_libero_mapping import audit, state_hash


@dataclass
class Task:
    name: str
    problem_folder: str
    bddl_file: str
    init_states_file: str
    language: str


class Benchmark:
    def __init__(self, tasks, states):
        self.tasks = tasks
        self.states = states

    def get_num_tasks(self):
        return len(self.tasks)

    def get_task(self, task_id):
        return self.tasks[task_id]

    def get_task_init_states(self, task_id):
        return self.states[task_id]


class MappingTests(unittest.TestCase):
    def make_factory(self):
        names = ("libero_goal", "libero_spatial", "libero_10")
        all_tasks = [
            Task(f"task-{i}", f"suite-{i//10}", f"{i}.bddl", f"{i}.init", f"do {i}")
            for i in range(130)
        ]
        all_states = [[np.array([i, trial]) for trial in range(5)] for i in range(130)]
        datasets = {"libero_130": Benchmark(all_tasks, all_states)}
        for group, name in enumerate(names):
            start = group * 10
            datasets[name] = Benchmark(
                all_tasks[start : start + 10], all_states[start : start + 10]
            )
        return lambda name: lambda: datasets[name]

    def test_identity_and_state_match(self):
        result = audit(self.make_factory())
        self.assertEqual(len(result["tasks"]), 30)
        self.assertEqual(result["tasks"][10]["libero_130_task_id"], 10)

    def test_state_mismatch_fails(self):
        factory = self.make_factory()
        factory("libero_goal")().states[0] = [np.array([999, x]) for x in range(5)]
        with self.assertRaisesRegex(ValueError, "init-state contents differ"):
            audit(factory)

    def test_hash_includes_shape(self):
        self.assertNotEqual(
            state_hash(np.array([[1, 2]])), state_hash(np.array([[1], [2]]))
        )


if __name__ == "__main__":
    unittest.main()
