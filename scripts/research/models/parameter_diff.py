"""Align authenticated checkpoint keys to the verified runtime parameter names."""

from dataclasses import dataclass
import math

import numpy as np

from .tensor_shards import FLOAT_DTYPES

ARCHITECTURE_KEYS = (
    "model_type", "arch_specifier", "llm_backbone_id", "hf_llm_id",
    "vision_backbone_id", "use_fused_vision_backbone", "timm_model_ids",
    "timm_override_act_layers", "image_sizes", "n_action_bins",
    "pad_to_multiple_of", "text_config",
)


def runtime_parameters(registry, vision_ids):
    modules = registry["modules"]
    # The RLinf Prismatic source uses the first and second timm IDs for these
    # paths. Preserve the configured IDs alongside any short labels.
    labels = {"vision.primary": "vision.primary", "vision.fused": "vision.fused"}
    for logical, prefix, label in (
        ("vision.primary", "vit_large_patch14_reg4_dinov2.", "vision.dinov2"),
        ("vision.fused", "vit_so400m_patch14_siglip_", "vision.siglip"),
    ):
        index = 0 if logical == "vision.primary" else 1
        if len(vision_ids) > index and vision_ids[index].startswith(prefix):
            labels[logical] = label

    result = {}
    ranges = [name for name in modules if name.startswith("transformer.")]
    priority = [*ranges, "action.logits", *labels, "projector", "vision", "language_model"]
    for logical in priority:
        if logical not in modules:
            continue
        module = modules[logical]
        seen = set()
        total = 0
        for param in module["parameters"]:
            name = param["path"]
            shape = tuple(param["shape"])
            if (not name.startswith(module["path"] + ".") or name in seen
                    or math.prod(shape) != param["numel"]):
                raise ValueError(f"Invalid runtime parameter evidence for {name}")
            seen.add(name)
            total += param["numel"]
            value = (labels.get(logical, logical), shape, param["dtype"])
            previous = result.setdefault(name, value)
            if previous[1:] != value[1:]:
                raise ValueError(f"Conflicting runtime parameter evidence for {name}")
        if total != module["parameter_count"]:
            raise ValueError(f"Invalid parameter total for {logical}")
    return result


@dataclass
class Reduction:
    count: int = 0
    changed: int = 0
    delta2: float = 0.0
    sft2: float = 0.0
    grpo2: float = 0.0
    cross: float = 0.0
    max_abs: float = 0.0

    def merge(self, other):
        self.count += other.count
        self.changed += other.changed
        self.delta2 += other.delta2
        self.sft2 += other.sft2
        self.grpo2 += other.grpo2
        self.cross += other.cross
        self.max_abs = max(self.max_abs, other.max_abs)

    def metrics(self):
        denom = math.sqrt(self.sft2) * math.sqrt(self.grpo2)
        return {
            "parameter_count": self.count,
            "changed_count": self.changed,
            "changed_fraction": self.changed / self.count if self.count else None,
            "absolute_l2": math.sqrt(self.delta2),
            "relative_l2": math.sqrt(self.delta2 / self.sft2) if self.sft2 else None,
            "max_absolute_delta": self.max_abs,
            "cosine_similarity": max(-1.0, min(1.0, self.cross / denom)) if denom else None,
        }


def reduce_pair(sft, grpo, chunk_elements=262_144):
    result = Reduction()
    if not sft.numel:
        return result
    left, right = sft.map(), grpo.map()
    for start in range(0, sft.numel, chunk_elements):
        end = start + chunk_elements
        a, b = sft.decode(left[start:end]), grpo.decode(right[start:end])
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            raise ValueError("Nonfinite checkpoint values")
        delta = b - a
        result.merge(Reduction(
            a.size, int(np.count_nonzero(delta)), float(np.sum(delta * delta)),
            float(np.sum(a * a)), float(np.sum(b * b)), float(np.sum(a * b)),
            float(np.max(np.abs(delta))),
        ))
    if not all(math.isfinite(x) for x in (result.delta2, result.sft2, result.grpo2, result.cross)):
        raise ValueError("Nonfinite checkpoint reduction")
    return result


def compare(sft_runtime, grpo_runtime, sft, grpo, *, dry_run=False):
    rows = []
    totals = {}
    complete = True
    for name in sorted(sft_runtime.keys() | grpo_runtime.keys() | sft.keys() | grpo.keys()):
        a, b = sft.get(name), grpo.get(name)
        left, right = sft_runtime.get(name), grpo_runtime.get(name)
        module = (left or right or ("unmapped",))[0]
        row = {
            "canonical_name": name if left or right else None, "checkpoint_key": name, "module": module,
            "sft_shape": a.shape if a else None, "grpo_shape": b.shape if b else None,
            "sft_storage_dtype": a.dtype if a else None, "grpo_storage_dtype": b.dtype if b else None,
            "sft_count": a.numel if a else None, "grpo_count": b.numel if b else None,
        }
        rows.append(row)
        if left is None and right is None:
            row.update(status="unmapped" if a and b else "extra_sft" if a else "extra_grpo")
            continue
        if a is None or b is None:
            row.update(status="missing_both" if not (a or b) else "missing_grpo" if a else "missing_sft")
            complete = False
            continue
        if left != right or a.shape != left[1] or a.shape != b.shape:
            row.update(status="unsafe", reason="runtime mapping or shape mismatch")
        elif a.dtype != b.dtype or a.dtype not in FLOAT_DTYPES:
            row.update(status="unsafe", reason="storage dtype mismatch or unsupported dtype")
        elif dry_run:
            row.update(status="aligned", parameter_count=a.numel)
            continue
        else:
            try:
                metrics = reduce_pair(a, b)
            except ValueError as exc:
                row.update(status="unsafe", reason=str(exc))
            else:
                row.update(status="changed" if metrics.changed else "identical", **metrics.metrics())
                totals.setdefault(module, Reduction()).merge(metrics)
                if module.startswith("vision.") or module == "vision":
                    totals.setdefault("vision.total", Reduction()).merge(metrics)
                if module.startswith("transformer."):
                    totals.setdefault("transformer.total", Reduction()).merge(metrics)
                totals.setdefault("policy.total", Reduction()).merge(metrics)
                continue
        complete = False
    return rows, {key: value.metrics() for key, value in totals.items()}, complete
