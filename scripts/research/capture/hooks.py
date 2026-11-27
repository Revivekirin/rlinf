"""Scoped, read-only output capture for registry-resolved modules."""

from collections import Counter

import torch


def output_tensor(output):
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output and isinstance(output[0], torch.Tensor):
        return output[0]
    raise ValueError(f"Unsupported hook output {type(output).__qualname__}")


class HookManager:
    def __init__(self, registry, selectors, *, input_identity, token_boundaries=None, capture=None):
        selectors = tuple(selectors)
        if len(set(selectors)) != len(selectors):
            raise ValueError("Duplicate hook selectors")
        self.targets = {key: registry.modules[key] for key in selectors}
        self.input_identity = input_identity
        self.token_boundaries = token_boundaries or {}
        self.capture = capture or {}
        self.records = []
        self.values = {}
        self._calls = Counter()
        self._handles = []
        self._active = False

    def _hook(self, key, path):
        def capture_output(module, inputs, output):
            tensor = output_tensor(output).detach()
            call = self._calls[key]
            self._calls[key] += 1
            self.records.append({
                "selector": key, "module_path": path, "module_type": type(module).__qualname__,
                "capture_method": "forward_hook", "input_identity": self.input_identity,
                "forward_call_index": call, "token_boundaries": self.token_boundaries.get(key),
                "shape": list(tensor.shape), "dtype": str(tensor.dtype),
            })
            if key in self.capture:
                selected = self.capture[key](tensor)
                self.values.setdefault(key, []).append(selected.to(device="cpu", copy=True))
        return capture_output

    def __enter__(self):
        if self._active:
            raise RuntimeError("HookManager is already active")
        self._active = True
        self.records.clear()
        self.values.clear()
        self._calls.clear()
        try:
            for key, (path, module) in self.targets.items():
                self._handles.append(module.register_forward_hook(self._hook(key, path)))
        except BaseException:
            self.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, trace):
        for handle in reversed(self._handles):
            handle.remove()
        self._handles.clear()
        self._active = False
