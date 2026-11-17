"""Golden contracts captured from the pre-extraction implementations."""

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/analysis'))
sys.path.insert(0, str(ROOT / 'scripts/evaluate'))

from nominal_behavior import read, save, csv_save
from nominal_storage import episode_id
from phase3a_core import array_hash, identity, protocol_hash, sha
from phase3b import seal, verify_seal


class CommonContractTests(unittest.TestCase):
    def test_protocol_and_array_hash_golden(self):
        self.assertEqual(protocol_hash({'b': 2.5, 'a': 1}),
                         '9a65554d3c3de4f574f3f7177c7e9090d3486ba145a6087fb9834cf1e74bde9e')
        self.assertEqual(array_hash(np.array([1, 256], dtype='<i2')),
                         'ebc7f06ec7e30f919a9b27b24aea9ea44215876c4746b3f55a92fb0ed4a1f4f6')
        with self.assertRaises(ValueError):
            array_hash(np.array([object()], dtype=object))

    def test_io_bytes_order_and_overwrite(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            save(root / 'nested' / 'data.json', {'b': 2, 'a': 1})
            self.assertEqual((root / 'nested' / 'data.json').read_bytes(),
                             b'{\n  "a": 1,\n  "b": 2\n}\n')
            self.assertEqual(read(root / 'nested' / 'data.json'), {'a': 1, 'b': 2})
            with self.assertRaises(FileExistsError):
                save(root / 'nested' / 'data.json', {})
            with self.assertRaises(ValueError):
                save(root / 'nan.json', {'x': float('nan')})
            csv_save(root / 'rows.csv', [{'b': [1, 2], 'a': 3}, {'b': {'x': 1}, 'a': 4}])
            self.assertEqual((root / 'rows.csv').read_bytes(),
                             b'b,a\r\n"[1, 2]",3\r\n"{""x"": 1}",4\r\n')

    def test_episode_and_identity(self):
        eid = episode_id('libero_10', 3, 4)
        self.assertEqual(eid, 'libero_10-t03-i04')
        row = dict(suite='libero_10', task_id=3, trial_id=4, task_name='task',
                   libero_130_task_id=42, initial_state_sha256='state', episode_id=eid,
                   simulator_seed=0)
        mapping = {('libero_10', 3): dict(task_name='task', libero_130_task_id=42,
                    trial_sha256=['other'] * 4 + ['state'], problem_folder='p',
                    bddl_file='b', init_states_file='i')}
        self.assertEqual(identity(row, mapping),
                         ('libero_10', 3, 'task', 'p', 'b', 'i', 4, 'state', 0))
        with self.assertRaises(ValueError):
            identity(dict(row, initial_state_sha256='changed'), mapping)

    def test_seal_bytes_and_tamper_detection(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / 'data.txt').write_bytes(b'payload\n')
            sealed = seal(root, {'status': 'PASS'})
            self.assertEqual(sealed, {'status': 'PASS', 'output_hashes': {
                'data.txt': 'd4e4877bac978b7952f0d544fc52ebff5411d351d129f1f056fa43f11da9af2b'}})
            self.assertEqual(verify_seal(root), sealed)
            self.assertNotIn('validation.json', sealed['output_hashes'])
            self.assertEqual(json.loads((root / 'validation.json').read_text()), sealed)
            (root / 'data.txt').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                verify_seal(root)
            (root / 'data.txt').unlink()
            with self.assertRaises(FileNotFoundError):
                verify_seal(root)


if __name__ == '__main__':
    unittest.main()
