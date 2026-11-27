import unittest
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from scripts.research.capture.equivalence import evaluate
from scripts.research.capture.hooks import HookManager
from scripts.research.models.modules import ModuleRegistry
from scripts.research.models.rlinf_oft import RLinfOFTAdapter


class Vision(nn.Identity):
    def get_num_patches(self):
        return 2

    def get_num_images_in_input(self):
        return 1


class FakePolicy(nn.Module):
    vocab_size = 5
    action_dim = 7
    num_action_chunks = 8

    def __init__(self):
        super().__init__()
        self.config = type("Config", (), {"n_action_bins": 2})()
        self.vision_backbone = Vision()
        self.projector = nn.Identity()
        self.language_model = nn.Module()
        self.language_model.model = nn.Module()
        self.language_model.model.layers = nn.ModuleList([nn.Identity() for _ in range(3)])
        self.language_model.lm_head = nn.Identity()
        self.language_model.get_output_embeddings = lambda: self.language_model.lm_head

    def predict_action_batch(self, *, env_obs, **kwargs):
        pixels = env_obs["main_images"]
        self.projector(self.vision_backbone(pixels))
        logits = torch.arange(5, dtype=torch.float32).expand(1, 61, 5).clone()
        for layer in self.language_model.model.layers:
            logits = layer(logits)
        logits = self.language_model.lm_head(logits)
        return torch.zeros(1, 8, 7), {"forward_inputs": {
            "input_ids": torch.zeros(1, 3, dtype=torch.long),
            "attention_mask": torch.ones(1, 3, dtype=torch.bool),
            "pixel_values": pixels,
            "action_tokens": logits[:, 4:60].argmax(-1).reshape(1, 8, 7),
        }}


class HookTests(unittest.TestCase):
    def test_repeated_capture_keeps_output_and_leaves_existing_hook(self):
        model = FakePolicy()
        registry = ModuleRegistry(model)
        manager = HookManager(registry, ["projector"], input_identity="observation")
        module = model.projector
        preexisting = module.register_forward_hook(lambda *args: None)
        self.addCleanup(preexisting.remove)
        data = torch.ones(1, 2, 3)
        for _ in range(2):
            with manager:
                self.assertIs(module(data), data)
                self.assertIs(module(data), data)
            self.assertEqual(list(module._forward_hooks), [preexisting.id])
            self.assertEqual([r["forward_call_index"] for r in manager.records], [0, 1])
            self.assertEqual(manager.records[0]["shape"], [1, 2, 3])
            self.assertEqual(manager.values, {})

    def test_cleanup_after_body_or_registration_exception(self):
        model = FakePolicy()
        manager = HookManager(ModuleRegistry(model), ["vision", "projector"], input_identity="x")
        with self.assertRaisesRegex(RuntimeError, "body"):
            with manager:
                raise RuntimeError("body")
        self.assertFalse(model.vision_backbone._forward_hooks)
        with patch.object(model.projector, "register_forward_hook", side_effect=RuntimeError("register")):
            with self.assertRaisesRegex(RuntimeError, "register"):
                with manager:
                    self.fail("partial registration must fail")
        self.assertFalse(model.vision_backbone._forward_hooks)
        with manager:
            self.assertTrue(model.projector._forward_hooks)
        self.assertFalse(model.projector._forward_hooks)

    def test_selected_capture_is_independent_of_source_tensor(self):
        model = FakePolicy()
        manager = HookManager(ModuleRegistry(model), ["projector"], input_identity="x",
                              capture={"projector": lambda tensor: tensor[:, :1]})
        data = torch.ones(1, 2)
        with manager:
            self.assertIs(model.projector(data), data)
            data.zero_()
        torch.testing.assert_close(manager.values["projector"][0], torch.ones(1, 1))

    def test_equivalence_and_logit_divergence(self):
        model = FakePolicy()
        bank = {
            "main_image_policy": np.zeros((2, 2, 3), dtype=np.uint8),
            "proprio_policy": np.zeros(8, dtype=np.float32),
            "input_ids": np.zeros(3, dtype=np.int64),
        }

        def reference(model, core, bank, language, seed, check_equivalence):
            actions, info = model.predict_action_batch(env_obs={
                "main_images": torch.from_numpy(bank["main_image_policy"]).unsqueeze(0),
            })
            logits = np.tile(np.array([3, 4], dtype=np.float32), (56, 1))
            return logits, actions.numpy()[0], {k: v.numpy()[0] for k, v in info["forward_inputs"].items()}, 3, 5

        result, logits = evaluate(RLinfOFTAdapter(model), bank, "sample", "obs", reference)
        self.assertTrue(result["passed"])
        self.assertEqual(logits["hooked"].shape, (56, 2))
        self.assertEqual(len(result["hook_calls"]), 2)

        def divergent(*args, **kwargs):
            logits, actions, inputs, lo, hi = reference(*args, **kwargs)
            return logits + 1, actions, inputs, lo, hi

        result, _ = evaluate(RLinfOFTAdapter(model), bank, "sample", "obs", divergent)
        self.assertFalse(result["passed"])
        self.assertEqual(result["checks"]["action_logits_0"]["phase3b"]["max_absolute_delta"], 1)
