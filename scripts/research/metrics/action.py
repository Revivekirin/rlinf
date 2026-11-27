"""Descriptive differences in native LIBERO delta OSC_POSE commands."""

import numpy as np


def differences(left, right, left_tokens, right_tokens):
    if any(x.shape != (8, 7) for x in (left, right, left_tokens, right_tokens)):
        raise ValueError("Invalid action chunk")
    delta = right.astype(np.float64) - left
    return {
        "action_delta_ood_minus_id": delta.tolist(),
        "translation_command_l2": np.linalg.norm(delta[:, :3], axis=1).tolist(),
        "rotation_vector_command_l2": np.linalg.norm(delta[:, 3:6], axis=1).tolist(),
        "gripper_command_abs": np.abs(delta[:, 6]).tolist(),
        "gripper_binary_disagreement": (np.sign(2 * left[:, 6] - 1)
                                        != np.sign(2 * right[:, 6] - 1)).tolist(),
        "token_disagreement_count": int(np.count_nonzero(left_tokens != right_tokens)),
        "units": "native decoded controller commands; no executed pose displacement",
    }
