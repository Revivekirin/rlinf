"""Paired response change and per-boundary functional differences."""

import numpy as np


def activation_difference(reference, candidate):
    if reference.shape != candidate.shape or not np.isfinite(candidate).all():
        raise ValueError("Invalid activation comparison")
    delta = candidate.astype(np.float64) - reference
    return {"l2": float(np.linalg.norm(delta.ravel())),
            "max_abs": float(np.max(np.abs(delta), initial=0)), "elements": delta.size}


def pair_change(base_id, base_ood, edited_id, edited_ood):
    delta = (edited_ood - edited_id) - (base_ood - base_id)
    if delta.shape != (8, 7) or not np.isfinite(delta).all():
        raise ValueError("Invalid camera action response")
    return {"camera_response_delta": delta.tolist(),
            "translation_l2": np.linalg.norm(delta[:, :3], axis=1).tolist(),
            "rotation_vector_l2": np.linalg.norm(delta[:, 3:6], axis=1).tolist(),
            "gripper_abs": np.abs(delta[:, 6]).tolist(),
            "units": "native decoded OSC commands"}
