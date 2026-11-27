"""Pinned PLUS camera geometry replayed in one canonical simulator per task."""

from contextlib import closing
import hashlib
from importlib.machinery import ModuleSpec
import importlib.util
import os
from pathlib import Path
from shutil import copyfile
import subprocess
import sys
from types import ModuleType

import numpy as np

from scripts.common.integrity import array_hash, sha
from scripts.research.core.artifacts import write_json
from scripts.research.models.checkpoint import AuditBlocked
from .camera_pairs import OBSERVATIONS, PAIRS, SCHEMA, TRIALS, plan, save_arrays, validate_pair

PLUS_COMMIT = "4976dc30028e805ff8094b55501d532c48fec182"
CAMERA_FIELDS = ("cam_pos", "cam_quat", "cam_fovy")
STATE_FIELDS = ("time", "qpos", "qvel", "act", "ctrl", "mocap_pos", "mocap_quat",
                "qfrc_applied", "xfrc_applied", "qacc_warmstart", "eq_active")


def revision(path, expected):
    cmd = ["git", "--no-optional-locks", "-C", str(path)]
    head = subprocess.check_output(cmd + ["rev-parse", "HEAD"], text=True).strip()
    status = subprocess.check_output(cmd + ["status", "--porcelain", "--untracked-files=no"], text=True)
    if head != expected or status:
        raise ValueError(f"Pinned source mismatch: {path}")


def stage_plus_package(plus, staging, assets):
    """Place pinned Python sources next to the installed read-only assets."""
    scene = assets / "scenes/libero_tabletop_base_style.xml"
    if not scene.is_file():
        raise AuditBlocked(f"Required PLUS scene is missing: {scene}")
    package_root = staging / "libero"
    for source in plus.rglob("*.py"):
        target = package_root / source.relative_to(plus)
        target.parent.mkdir(parents=True, exist_ok=True)
        copyfile(source, target)
    (package_root / "assets").symlink_to(assets.resolve(), target_is_directory=True)
    return package_root


def simulator(plus, out, assets, staging):
    try:
        import robosuite
    except (ImportError, OSError) as exc:
        raise AuditBlocked(f"Simulator unavailable: {exc}") from exc

    if assets is None:
        installed = importlib.util.find_spec("liberoplus")
        if installed and installed.submodule_search_locations:
            assets = Path(next(iter(installed.submodule_search_locations))) / "liberoplus/assets"
    if assets is None or not assets.is_dir():
        raise AuditBlocked("PLUS assets unavailable; use --assets with the installed PLUS asset directory")

    # The pinned checkout uses libero.libero absolute imports. Isolate its alias
    # and path configuration inside this process and this run directory.
    if "libero" in sys.modules:
        raise ValueError("Cannot load pinned PLUS after another LIBERO package")
    package_root = stage_plus_package(plus, staging, assets)
    config = out / "libero-config"
    config.mkdir()
    write_json(config / "config.yaml", {
        "benchmark_root": str(package_root), "bddl_files": str(plus / "bddl_files"),
        "init_states": str(plus / "init_files"), "assets": str(assets.resolve()),
        "datasets": str(out),
    })
    os.environ["LIBERO_CONFIG_PATH"] = str(config)
    os.environ["LIBERO_TYPE"] = "standard"  # RLinf image utilities resolve our pinned alias.
    package = ModuleType("libero")
    package.__path__ = [str(package_root.parent)]
    package.__spec__ = ModuleSpec("libero", loader=None, is_package=True)
    package.__spec__.submodule_search_locations = package.__path__
    sys.modules["libero"] = package
    try:
        from libero.libero.envs import OffScreenRenderEnv
        from libero.libero.envs.bddl_utils import get_problem_info
        from rlinf.envs.sim.libero.utils import get_libero_image, quat2axisangle
        import torch
    except (ImportError, OSError) as exc:
        raise AuditBlocked(f"Simulator dependencies unavailable: {exc}") from exc
    return OffScreenRenderEnv, get_problem_info, get_libero_image, quat2axisangle, robosuite, torch


def camera_config(sim):
    index = sim.model.camera_name2id("agentview")
    if int(sim.model.cam_bodyid[index]) != 0 or int(sim.model.cam_mode[index]) != 0:
        raise ValueError("Agentview must be a fixed world camera")
    settings = {key: np.asarray(getattr(sim.model, key)[index]).copy() for key in CAMERA_FIELDS}
    return index, settings


def model_fingerprint(sim, index):
    digest = hashlib.sha256()
    model = getattr(sim.model, "_model", sim.model)
    keys = set(dir(model)) | {"body_pos", "geom_pos", *CAMERA_FIELDS}
    for key in sorted(n for n in keys if n.startswith((
        "body_", "geom_", "jnt_", "dof_", "cam_", "actuator_", "light_", "mat_",
    ))):
        value = getattr(model, key)
        if not isinstance(value, np.ndarray) or value.dtype.kind not in "biuf":
            continue
        if key in CAMERA_FIELDS:
            value = np.delete(value, index, axis=0)
        digest.update(key.encode())
        digest.update(array_hash(value).encode())
    return digest.hexdigest()


def snapshot(env):
    data = env.sim.data
    return {"state_flat": np.asarray(env.get_sim_state()).copy(),
            "state_xpos": np.asarray(data.body_xpos).copy(),
            "state_xquat": np.asarray(data.body_xquat).copy(), **{
        f"state_{key}": np.asarray(getattr(data, key)).copy()
        for key in STATE_FIELDS if hasattr(data, key)
    }}


def update_camera(sim, index, settings, camera_kinematics):
    for key, value in settings.items():
        getattr(sim.model, key)[index] = value
    camera_kinematics(getattr(sim.model, "_model", sim.model),
                      getattr(sim.data, "_data", sim.data))


def require_same_state(expected, actual, state_id, camera_id, phase):
    changed = sorted(key for key in expected.keys() | actual.keys()
                     if key not in expected or key not in actual
                     or expected[key].dtype != actual[key].dtype
                     or not np.array_equal(expected[key], actual[key]))
    if changed:
        raise ValueError(f"Physical state changed for {state_id}/{camera_id} during {phase}: "
                         + ", ".join(changed))


def collect(root, out, assets, staging):
    from scripts.research.cli.audit import UPSTREAM_PIN

    checkout = root / "upstream/LIBERO-plus"
    plus = checkout / "libero/libero"
    revision(checkout, PLUS_COMMIT)
    revision(root / "upstream/RLinf", UPSTREAM_PIN)
    specifications = plan(plus)
    write_json(out / "plan.json", {
        "schema": SCHEMA, "cases": [vars(spec) for spec in specifications],
        "expected_observations": OBSERVATIONS, "expected_pairs": PAIRS,
        "plus_commit": PLUS_COMMIT, "rlinf_commit": UPSTREAM_PIN,
        "classification_sha256": sha(plus / "benchmark/task_classification.json"),
    })
    create, problem_info, image, quat_axis, suite, torch = simulator(plus, out, assets, staging)
    from mujoco import mj_camlight

    controller_config = suite.load_controller_config(default_controller="OSC_POSE")
    if controller_config.get("control_delta") is not True:
        raise ValueError("Expected delta OSC_POSE convention")
    controller = {"name": "OSC_POSE", "config": controller_config, "frequency_hz": 20,
                  "ordering": ["dx", "dy", "dz", "dax", "day", "daz", "gripper"]}
    # Normalize numpy values (some robosuite versions expose numpy bounds).
    controller["config"] = {key: value.tolist() if isinstance(value, np.ndarray) else value
                            for key, value in controller_config.items()}
    observations, pairs = {}, []
    for spec in specifications:
        bddl = plus / "bddl_files" / spec.suite / f"{spec.task}.bddl"
        init = plus / "init_files" / spec.suite / f"{spec.task}.pruned_init"
        options = dict(camera_heights=256, camera_widths=256, hard_reset=False)
        language = problem_info(str(bddl))["language_instruction"]
        targets = {}
        for variant in spec.variants:
            variant_path = bddl.with_name(variant["name"] + ".bddl")
            with closing(create(bddl_file_name=str(variant_path), **options)) as geometry:
                if geometry.language_instruction != language:
                    raise ValueError("Camera variant changed the instruction")
                _, targets[variant["id"]] = camera_config(geometry.sim)
        # Files come from the clean, pinned checkout. Some releases use numpy
        # pickles that the newer torch weights-only loader cannot deserialize.
        initial = torch.load(init, map_location="cpu", weights_only=False)
        if len(initial) < TRIALS:
            raise ValueError(f"Insufficient saved initial states: {init}")
        with closing(create(bddl_file_name=str(bddl), **options)) as env:
            env.seed(0)
            index, nominal = camera_config(env.sim)
            for trial in range(TRIALS):
                state_id = f"{spec.suite}-t{spec.task_id:02d}-s{trial:02d}"
                env.reset()
                env.set_init_state(np.asarray(initial[trial]))
                for _ in range(15):
                    env.step(np.zeros(7))
                # Synchronize derived poses with the final settled qpos before freezing.
                env.sim.forward()
                frozen = snapshot(env)
                active_controller = env.robots[0].controller
                model_hash = model_fingerprint(env.sim, index)
                previous = None
                for variant in (None, *spec.variants):
                    label = "canonical" if variant is None else f"plus-{variant['id']}"
                    settings = nominal if variant is None else targets[variant["id"]]
                    update_camera(env.sim, index, settings, mj_camlight)
                    require_same_state(frozen, snapshot(env), state_id, label, "camera update")
                    env._update_observables(force=True)
                    raw = env.env._get_observations()
                    current = snapshot(env)
                    require_same_state(frozen, current, state_id, label, "observation rendering")
                    if model_fingerprint(env.sim, index) != model_hash or env.robots[0].controller is not active_controller:
                        raise ValueError(f"Model properties or controller changed during camera rendering: {state_id}")
                    obs_id = f"{state_id}-{label}"
                    fovy = float(settings["cam_fovy"])
                    focal = 128 / np.tan(np.radians(fovy) / 2)
                    camera = {"position_m": env.sim.data.get_camera_xpos("agentview").tolist(),
                              "rotation_world_from_camera": env.sim.data.get_camera_xmat("agentview").reshape(3, 3).tolist(),
                              "fovy_deg": fovy, "focal_px": float(focal), "principal_point_px": [128, 128],
                              "calibration_convention": "MuJoCo camera axes; centered square sensor"}
                    record = {
                        "observation_id": obs_id, "state_id": state_id, "camera_id": label,
                        "task": {"suite": spec.suite, "original_id": spec.task_id, "name": spec.task},
                        "trial": trial, "language": language, "camera": camera, "variant": variant,
                        "model_sha256": model_hash, "controller": controller,
                        "source": {"bddl": bddl.relative_to(plus).as_posix(),
                                   "init": init.relative_to(plus).as_posix(),
                                   "bddl_sha256": sha(bddl), "init_sha256": sha(init),
                                   "selected_init_sha256": array_hash(np.asarray(initial[trial])),
                                   "settling_steps": 15, "seed": 0},
                    }
                    arrays = dict(current, main_image_policy=np.ascontiguousarray(image(raw)),
                                  proprio_policy=np.concatenate((raw["robot0_eef_pos"],
                                      quat_axis(raw["robot0_eef_quat"].copy()), raw["robot0_gripper_qpos"])))
                    if variant is None:
                        previous = record, arrays
                    else:
                        validate_pair(previous[0], record, previous[1], arrays)
                        pairs.append({"pair_id": obs_id, "id": previous[0]["observation_id"],
                                      "ood": obs_id, "physical_atol": 0, "physical_rtol": 0,
                                      "kind": "same_state_replay_of_plus_camera"})
                    record["data"] = save_arrays(out / f"{obs_id}.npz", arrays)
                    observations[obs_id] = record
                update_camera(env.sim, index, nominal, mj_camlight)
    result = {"schema": SCHEMA, "observations": observations, "pairs": pairs,
              "plus_commit": PLUS_COMMIT, "rlinf_commit": UPSTREAM_PIN,
              "simulator": {"robosuite": suite.__version__, "mujoco": __import__("mujoco").__version__},
              "preprocessing": "RLinf get_libero_image (180-degree rotation), then native R0 processor"}
    write_json(out / "dataset.json", result)
