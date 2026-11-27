"""Audited parameter partition and reversible streamed intervention."""

from collections import defaultdict
from contextlib import contextmanager
import math

from scripts.research.models.tensor_shards import FLOAT_DTYPES


def groups_from_audit(comparison, runtime):
    groups, parents, excluded = {}, defaultdict(list), []
    seen = set()
    for row in comparison["tensors"]:
        name = row["canonical_name"]
        if name not in runtime:
            excluded.append({"tensor": row["checkpoint_key"], "status": "UNSAFE",
                             "reason": f"Outside audited native policy: {row['status']}",
                             "stored_elements": row["grpo_count"] or row["sft_count"] or 0})
            continue
        if name in seen:
            raise ValueError(f"Duplicate audited parameter: {name}")
        seen.add(name)
        region, shape, _ = runtime[name]
        owner = name.rsplit(".", 1)[0]
        group = groups.setdefault(owner, {"region": region, "parameters": [], "elements": 0,
                                          "changed": 0, "status": "NO_UPDATE"})
        group["parameters"].append(name)
        count = math.prod(shape)
        group["elements"] += count
        if (row["status"] not in {"changed", "identical"} or row["checkpoint_key"] != name
                or row["sft_shape"] != list(shape) or row["grpo_shape"] != list(shape)
                or row["sft_storage_dtype"] != row["grpo_storage_dtype"]
                or row["sft_storage_dtype"] not in FLOAT_DTYPES or row["parameter_count"] != count):
            group["status"] = "UNSAFE"
        else:
            group["changed"] += row["changed_count"]
    if seen != runtime.keys():
        raise ValueError("R0 tensor inventory does not cover the runtime parameter names")
    for owner, group in groups.items():
        if group["status"] != "UNSAFE" and group["changed"]:
            group["status"] = "READY"
        region = group["region"]
        parent = "transformer" if region.startswith("transformer.") else region
        parents[parent].append(owner)
    return {"leaves": groups, "parents": dict(parents), "excluded": excluded,
            "runtime_elements": sum(g["elements"] for g in groups.values())}


def live_parameters(model, runtime):
    parameters = dict(model.named_parameters(remove_duplicate=False))
    by_storage = defaultdict(list)
    for name, param in parameters.items():
        by_storage[(str(param.device), param.untyped_storage().data_ptr())].append(name)
    aliases = [names for names in by_storage.values() if len(names) > 1]
    if aliases:
        raise ValueError(f"UNSAFE shared storage needs canonical alias correspondence: {aliases}")
    if parameters.keys() != runtime.keys():
        raise ValueError("Loaded parameter names differ from audited runtime")
    for name, param in parameters.items():
        _, shape, dtype = runtime[name]
        if tuple(param.shape) != shape or str(param.dtype) != dtype or not param.is_contiguous():
            raise ValueError(f"Loaded tensor does not match audited correspondence: {name}")
    return parameters


def cast_interpolation(sft, grpo, alpha, dtype):
    if alpha not in (0, 0.5, 1):
        raise ValueError("Unsupported interpolation alpha")
    if alpha in (0, 1):
        return (sft if alpha == 0 else grpo).to(dtype)
    return (sft + alpha * (grpo - sft)).to(dtype)


@contextmanager
def edited(parameters, selected, sft, grpo, alpha, chunk=262144):
    import torch
    from scripts.research.models.tensor_shards import FLOAT_DTYPES

    if alpha not in (0, 0.5, 1) or not selected or len(set(selected)) != len(selected):
        raise ValueError("Invalid intervention condition")
    versions = {key: param._version for key, param in parameters.items() if key not in selected}

    def chunks(name):
        param = parameters[name].view(-1)
        a, b = sft[name], grpo[name]
        if (a.shape != b.shape or a.shape != tuple(parameters[name].shape)
                or a.dtype != b.dtype or a.dtype not in FLOAT_DTYPES or not param.is_floating_point()):
            raise ValueError(f"UNSAFE shard correspondence: {name}")
        left, right = a.map(), b.map()
        for offset in range(0, param.numel(), chunk):
            sl = slice(offset, offset + chunk)
            x = torch.from_numpy(a.decode(left[sl])).float()
            y = torch.from_numpy(b.decode(right[sl])).float()
            if not torch.isfinite(x).all() or not torch.isfinite(y).all():
                raise ValueError(f"Nonfinite tensor: {name}")
            yield param[sl], x, y

    def assert_baseline():
        for name in selected:
            for value, _, y in chunks(name):
                if not torch.equal(value, y.to(device=value.device, dtype=value.dtype)):
                    raise ValueError(f"GRPO parameter verification failed: {name}")

    stats = {"elements": 0, "effective_changed": 0, "effective_l2": 0.0,
             "effective_relative_l2": None, "restored": False, "non_selected_unchanged": False}
    delta2 = base2 = 0.0
    with torch.no_grad():
        assert_baseline()
        try:
            for name in selected:
                for value, x, y in chunks(name):
                    baseline = y.to(value.dtype)
                    replacement = cast_interpolation(x, y, alpha, value.dtype)
                    delta = replacement.double() - baseline.double()
                    stats["elements"] += delta.numel()
                    stats["effective_changed"] += int(torch.count_nonzero(delta))
                    delta2 += float(delta.square().sum())
                    base2 += float(baseline.double().square().sum())
                    if alpha != 1:
                        value.copy_(replacement)
                        if not torch.equal(value, replacement.to(value.device)):
                            raise ValueError(f"Selected parameter replacement failed: {name}")
            stats["effective_l2"] = math.sqrt(delta2)
            stats["effective_relative_l2"] = math.sqrt(delta2 / base2) if base2 else None
            yield stats
        finally:
            for name in selected:
                for value, _, y in chunks(name):
                    if alpha != 1:
                        value.copy_(y)
            assert_baseline()
            stats["restored"] = True
            stats["non_selected_unchanged"] = all(parameters[key]._version == version
                                                    for key, version in versions.items())
            if not stats["non_selected_unchanged"]:
                raise ValueError("Unexpected mutation outside selected parameters")
