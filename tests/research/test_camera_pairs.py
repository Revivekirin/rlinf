import copy
from pathlib import Path
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np

from scripts.research.core.artifacts import write_json
from scripts.research.data.camera_pairs import OBSERVATIONS, PAIRS, TASKS, load_arrays, plan, save_arrays, validate_pair
from scripts.research.data.validation import verify_dataset
from scripts.research.data.simulator_state import require_same_state, stage_plus_package, update_camera
from scripts.research.metrics.action import differences
from scripts.research.models.checkpoint import AuditBlocked

ROOT = Path(__file__).resolve().parents[2]


class CameraPairTests(unittest.TestCase):
    def test_camera_update_uses_native_camera_kinematics_and_reports_state_changes(self):
        raw_model, raw_data = object(), object()
        sim = SimpleNamespace(
            model=SimpleNamespace(_model=raw_model, cam_pos=np.zeros((2, 3))),
            data=SimpleNamespace(_data=raw_data),
        )
        calls = []
        update_camera(sim, 1, {"cam_pos": [1, 2, 3]}, lambda *args: calls.append(args))
        self.assertEqual(calls, [(raw_model, raw_data)])
        np.testing.assert_array_equal(sim.model.cam_pos, [[0, 0, 0], [1, 2, 3]])
        frozen = {"state_time": np.array(0.1), "state_qpos": np.zeros(2)}
        require_same_state(frozen, frozen, "s00", "canonical", "camera update")
        with self.assertRaisesRegex(ValueError, "s00/plus-1 during observation rendering: state_qpos"):
            require_same_state(frozen, dict(frozen, state_qpos=np.ones(2)),
                               "s00", "plus-1", "observation rendering")

    def test_staged_plus_sources_resolve_sibling_asset_paths(self):
        plus = ROOT / "upstream/LIBERO-plus/libero/libero"
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            assets = root / "installed-assets"
            scene = assets / "scenes/libero_tabletop_base_style.xml"
            with self.assertRaisesRegex(AuditBlocked, "Required PLUS scene is missing"):
                stage_plus_package(plus, root / "staging", assets)
            scene.parent.mkdir(parents=True)
            scene.write_text("<mujoco/>")
            package = stage_plus_package(plus, root / "staging", assets)
            source = Path("envs/bddl_base_domain.py")
            self.assertEqual((package / source).read_bytes(), (plus / source).read_bytes())
            sibling = (package / source).resolve().parent / "../assets/scenes" / scene.name
            self.assertEqual(sibling.resolve(), scene.resolve())
            self.assertEqual(sibling.read_text(), "<mujoco/>")

    def setUp(self):
        self.left = {
            "state_id": "state", "task": "task", "language": "instruction",
            "controller": {"type": "OSC_POSE"}, "source": "saved-trial",
            "model_sha256": "model", "camera_id": "canonical", "camera": {"position": [0, 0, 0]},
        }
        self.right = dict(self.left, camera_id="plus-3", camera={"position": [0, 1, 0]})
        self.a = {f"state_{key}": np.zeros(2) for key in ("flat", "time", "qpos", "qvel", "xpos", "xquat")}
        self.a.update(main_image_policy=np.zeros((256, 256, 3), dtype=np.uint8), proprio_policy=np.zeros(8))
        self.b = copy.deepcopy(self.a)
        self.b["main_image_policy"][0, 0] = 1

    def test_pinned_source_plan(self):
        plus = ROOT / "upstream/LIBERO-plus/libero/libero"
        specs = plan(plus)
        self.assertEqual(len(specs), len(TASKS))
        self.assertEqual(sum(len(spec.variants) * 3 for spec in specs), PAIRS)
        for spec in specs:
            self.assertTrue((plus / "bddl_files" / spec.suite / (spec.task + ".bddl")).is_file())
            self.assertTrue((plus / "init_files" / spec.suite / (spec.task + ".pruned_init")).is_file())
            self.assertEqual(len({row["difficulty_level"] for row in spec.variants}), 3)
            self.assertTrue(all(row["name"].endswith("_initstate_0") for row in spec.variants))

    def test_aligned_pair_and_immutable_round_trip(self):
        validate_pair(self.left, self.right, self.a, self.b)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            path = root / "observation.npz"
            record = save_arrays(path, self.a)
            for name, value in load_arrays(root, record).items():
                self.assertEqual(value.tobytes(), self.a[name].tobytes())
            with self.assertRaises(FileExistsError):
                save_arrays(path, self.b)
            with path.open("ab") as stream:
                stream.write(b"tampered")
            with self.assertRaisesRegex(ValueError, "hash mismatch"):
                load_arrays(root, record)

    def test_rejects_step_or_changed_pose_or_instruction(self):
        for field in ("state_time", "state_qpos", "state_xquat"):
            changed = copy.deepcopy(self.b)
            changed[field][0] = 1e-12
            with self.assertRaisesRegex(ValueError, "Physical state mismatch"):
                validate_pair(self.left, self.right, self.a, changed)
        with self.assertRaisesRegex(ValueError, "instruction"):
            validate_pair(self.left, dict(self.right, language="different"), self.a, self.b)
        with self.assertRaisesRegex(ValueError, "identical RGB"):
            validate_pair(self.left, self.right, self.a, self.a)

    def test_action_command_metrics_and_gripper_threshold(self):
        a = np.zeros((8, 7))
        b = a.copy()
        b[0, [0, 1, 3, 6]] = [3, 4, 2, 1]
        result = differences(a, b, a, b)
        self.assertEqual(result["translation_command_l2"][0], 5)
        self.assertEqual(result["rotation_vector_command_l2"][0], 2)
        self.assertTrue(result["gripper_binary_disagreement"][0])
        self.assertEqual(result["token_disagreement_count"], 4)

    def test_offline_dataset_checks_all_target_pairs(self):
        specs = plan(ROOT / "upstream/LIBERO-plus/libero/libero")
        versions = {"schema": 1, "plus_commit": "plus", "rlinf_commit": "rlinf"}
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            write_json(path / "plan.json", dict(
                versions, cases=[vars(spec) for spec in specs],
                expected_observations=OBSERVATIONS, expected_pairs=PAIRS,
            ))
            observations, pairs = {}, []
            for spec in specs:
                task = {"suite": spec.suite, "original_id": spec.task_id, "name": spec.task}
                for trial in range(3):
                    state = f"{spec.suite}-t{spec.task_id:02d}-s{trial:02d}"
                    for view, variant in enumerate((None, *spec.variants)):
                        label = f"plus-{variant['id']}" if variant else "canonical"
                        key = f"{state}-{label}"
                        arrays = dict(self.a, main_image_policy=np.full((256, 256, 3), view, dtype=np.uint8))
                        camera = {"position_m": [view, 0, 0], "rotation_world_from_camera": np.eye(3).tolist(),
                                  "fovy_deg": 90, "focal_px": 128, "principal_point_px": [128, 128]}
                        observations[key] = dict(
                            self.left, state_id=state, trial=trial, task=task, observation_id=key,
                            camera_id=label, camera=camera, variant=variant,
                            controller={"config": {"control_delta": True}},
                            data=save_arrays(path / f"{key}.npz", arrays),
                        )
                        if variant:
                            pairs.append({"pair_id": key, "id": f"{state}-canonical", "ood": key,
                                          "physical_atol": 0, "physical_rtol": 0,
                                          "kind": "same_state_replay_of_plus_camera"})
            write_json(path / "dataset.json", dict(versions, observations=observations, pairs=pairs))
            self.assertEqual(len(verify_dataset(path)["pairs"]), PAIRS)
            (path / observations[key]["data"]["file"]).unlink()
            with self.assertRaises(FileNotFoundError):
                verify_dataset(path)
