"""Adapter for the base-only native policy used in the Phase 3B reference."""

import inspect
from pathlib import Path

from scripts.common.integrity import sha

from .checkpoint import AuditBlocked
from .modules import ModuleRegistry

MODEL_SOURCE = Path("rlinf/models/embodiment/openvla_oft/rlinf/openvla_oft_action_model.py")
MODEL_SHA = "22a5b7864bc125bb8ca757b1ed14e901972eea96ab243e9b5d663f61b3f219e9"


def baseline_config(protocol: dict, pin: dict, path: Path) -> dict:
    if (protocol["checkpoint_repo"], protocol["checkpoint_revision"]) != (
        pin["repo_id"], pin["revision"]
    ):
        raise ValueError("Phase 3A checkpoint identity disagrees with the audit pin")
    cfg = protocol["model_config"].copy()
    expected = {
        "model_type": "openvla_oft", "implement_version": "rlinf", "is_lora": False,
        "lora_path": None, "action_dim": 7, "num_action_chunks": 8,
        "num_images_in_input": 1, "use_proprio": False, "add_value_head": False,
    }
    if protocol["lora_applied"] is not False or any(
        key not in cfg or cfg[key] != value for key, value in expected.items()
    ):
        raise ValueError("Unexpected native Phase 3A baseline configuration")
    cfg["model_path"] = str(path)
    return cfg


class RLinfOFTAdapter:
    def __init__(self, model):
        self.model = model
        self.registry = ModuleRegistry(model)

    @classmethod
    def load(cls, upstream: Path, config: dict):
        try:
            import torch
        except ModuleNotFoundError as exc:
            raise AuditBlocked(f"Native environment unavailable: {exc}") from exc
        if not torch.cuda.is_available():
            raise AuditBlocked("CUDA unavailable for the pinned native FlashAttention policy")

        try:
            from omegaconf import OmegaConf
            from rlinf.models import get_model
            from rlinf.models.embodiment.openvla_oft.rlinf.openvla_oft_action_model import (
                OpenVLAOFTForRLActionPrediction,
            )
        except (ImportError, OSError) as exc:
            raise AuditBlocked(f"Native environment unavailable: {exc}") from exc
        if Path(inspect.getfile(get_model)).resolve() != upstream / "rlinf/models/__init__.py":
            raise ValueError("RLinf loader comes from a different checkout")
        source = Path(inspect.getfile(OpenVLAOFTForRLActionPrediction)).resolve()
        if source != upstream / MODEL_SOURCE or sha(source) != MODEL_SHA:
            raise ValueError("Native action model source differs from Phase 3B")
        model = get_model(OmegaConf.create(config)).to("cuda").eval().requires_grad_(False)
        if type(model) is not OpenVLAOFTForRLActionPrediction:
            raise ValueError("Unexpected model class or wrapper")
        return cls(model)

    def predict(self, observations, *, do_sample=False, temperature=1.0, top_k=-1):
        import torch

        # The native policy adds an image dimension by reassigning dictionary keys.
        with torch.inference_mode():
            return self.model.predict_action_batch(
                env_obs=dict(observations), do_sample=do_sample, temperature=temperature,
                top_k=top_k, top_p=1.0, calculate_logprobs=False, calculate_values=False,
            )

    def smoke(self, config: dict) -> dict:
        import torch

        model = self.model
        vision = model.vision_backbone
        if (model.action_dim, model.num_action_chunks, vision.get_num_images_in_input()) != (
            config["action_dim"], config["num_action_chunks"], config["num_images_in_input"]
        ) or vision.get_num_patches() <= 0:
            raise ValueError("Unexpected runtime action or image contract")

        required = {"vision", "vision.primary", "projector", "language_model",
                    "transformer", "action.logits"}
        if getattr(model.config, "use_fused_vision_backbone", False):
            required.add("vision.fused")
        missing = required - self.registry.modules.keys()
        if missing:
            raise ValueError(f"Unavailable required runtime modules: {sorted(missing)}")
        if self.registry.block_count != model.config.text_config.num_hidden_layers:
            raise ValueError("Runtime block count differs from configuration")
        _, head = self.registry.modules["action.logits"]
        if not 0 < model.config.n_action_bins <= model.vocab_size <= head.out_features:
            raise ValueError("Native action vocabulary is outside the LM output")

        height, width = config["image_size"]
        actions, result = self.predict({
            "main_images": torch.zeros(1, height, width, 3, dtype=torch.uint8),
            "states": torch.zeros(1, 8),
            "task_descriptions": ["pick up the object"],
        })
        tokens = result["forward_inputs"]["action_tokens"]
        lo, hi = model.vocab_size - model.config.n_action_bins, model.vocab_size
        expected = (1, model.num_action_chunks, model.action_dim)
        if (actions.shape != expected or tokens.shape != expected
                or not torch.isfinite(actions).all()
                or not ((tokens >= lo) & (tokens < hi)).all()):
            raise ValueError("Native policy returned invalid action shapes or values")
        tensors = {
            name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in result["forward_inputs"].items()
        }
        return {
            "input": "single synthetic zero RGB frame and fixed text prompt",
            "decoded_actions": {"shape": list(actions.shape), "dtype": str(actions.dtype)},
            "forward_inputs": tensors, "action_token_range": [lo, hi],
            "num_image_patches": vision.get_num_patches(),
            "action_path": "shared language model vocabulary projection",
        }
