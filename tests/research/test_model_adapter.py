import unittest
from pathlib import Path

import torch
from torch import nn

from scripts.research.models.rlinf_oft import RLinfOFTAdapter, baseline_config


class FakeLM(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Linear(2, 2), nn.Linear(2, 2)])
        self.lm_head = nn.Linear(2, 3)

    def get_output_embeddings(self):
        return self.lm_head


class FakePolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.vision_backbone = nn.Module()
        self.vision_backbone.featurizer = nn.Linear(2, 2)
        self.projector = nn.Linear(2, 2)
        self.language_model = FakeLM()

    def predict_action_batch(self, *, env_obs, **kwargs):
        env_obs["main_images"] = env_obs["main_images"].unsqueeze(1)
        return env_obs, kwargs, torch.is_inference_mode_enabled()


class AdapterTests(unittest.TestCase):
    def test_registry_resolves_objects_and_parameter_metadata(self):
        model = FakePolicy().requires_grad_(False)
        registry = RLinfOFTAdapter(model).registry
        self.assertIs(registry.modules["transformer.1"][1], model.language_model.model.layers[1])
        self.assertIs(registry.modules["action.logits"][1], model.language_model.lm_head)
        summary = registry.summarize()
        self.assertIn("vision.fused", summary["unavailable"])
        self.assertNotIn("vision.fused", summary["modules"])
        output = summary["modules"]["action.logits"]
        self.assertEqual(output["parameter_count"], 9)
        self.assertEqual(output["parameters"][0], {
            "path": "language_model.lm_head.weight", "shape": [3, 2],
            "dtype": "torch.float32", "numel": 6, "requires_grad": False,
        })

    def test_wrong_lm_output_identity_is_unavailable(self):
        model = FakePolicy()
        model.language_model.get_output_embeddings = lambda: model.projector
        registry = RLinfOFTAdapter(model).registry
        self.assertIn("action.logits", registry.unavailable)
        self.assertNotIn("action.logits", registry.modules)

    def test_predict_delegates_and_preserves_observation_mapping(self):
        adapter = RLinfOFTAdapter(FakePolicy())
        obs = {"main_images": torch.zeros(1, 2, 3, 3)}
        forwarded, kwargs, inference_mode = adapter.predict(obs, do_sample=True, temperature=1.6)
        self.assertTrue(inference_mode)
        self.assertEqual(kwargs, {
            "do_sample": True, "temperature": 1.6, "top_k": -1, "top_p": 1.0,
            "calculate_logprobs": False, "calculate_values": False,
        })
        self.assertEqual(forwarded["main_images"].ndim, 5)
        self.assertEqual(obs["main_images"].ndim, 4)

    def test_baseline_config_rejects_adapter_and_wrong_revision(self):
        protocol = {
            "checkpoint_repo": "org/model", "checkpoint_revision": "pin", "lora_applied": False,
            "model_config": {
                "model_type": "openvla_oft", "implement_version": "rlinf",
                "model_path": "original", "is_lora": False, "lora_path": None,
                "action_dim": 7, "num_action_chunks": 8, "num_images_in_input": 1,
                "use_proprio": False, "add_value_head": False,
            },
        }
        pin = {"repo_id": "org/model", "revision": "pin"}
        self.assertEqual(baseline_config(protocol, pin, Path("override"))["model_path"], "override")
        self.assertEqual(protocol["model_config"]["model_path"], "original")
        with self.assertRaises(ValueError):
            baseline_config(protocol, dict(pin, revision="other"), Path("override"))
        protocol["model_config"]["is_lora"] = True
        with self.assertRaises(ValueError):
            baseline_config(protocol, pin, Path("override"))
