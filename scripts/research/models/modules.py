"""Resolve the native OpenVLA-OFT boundaries on a loaded Llama policy."""


class ModuleRegistry:
    def __init__(self, core):
        self.modules = {}
        self.unavailable = {}

        def add(logical_name, path):
            try:
                module = core.get_submodule(path)
            except AttributeError:
                self.unavailable[logical_name] = path
            else:
                self.modules[logical_name] = (path, module)
                return module

        add("vision", "vision_backbone")
        add("vision.primary", "vision_backbone.featurizer")
        add("vision.fused", "vision_backbone.fused_featurizer")
        add("projector", "projector")
        add("language_model", "language_model")
        blocks = add("transformer", "language_model.model.layers")
        self.block_count = len(blocks) if blocks is not None else 0
        for index in range(self.block_count):
            add(f"transformer.{index}", f"language_model.model.layers.{index}")
        output = add("action.logits", "language_model.lm_head")
        if output is not None and output is not core.language_model.get_output_embeddings():
            self.modules.pop("action.logits")
            self.unavailable["action.logits"] = "LM output module identity mismatch"

    def summarize(self):
        def entry(path, module):
            params = [
                {"path": f"{path}.{name}", "shape": list(param.shape),
                 "dtype": str(param.dtype), "numel": param.numel(),
                 "requires_grad": param.requires_grad}
                for name, param in module.named_parameters()
            ]
            return {
                "path": path, "type": type(module).__qualname__,
                "parameter_count": sum(param["numel"] for param in params),
                "parameters": params,
            }

        return {
            "modules": {key: entry(*value) for key, value in self.modules.items()},
            "unavailable": self.unavailable.copy(),
        }
