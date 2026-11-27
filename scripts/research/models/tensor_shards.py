"""Read indexed safetensors on CPU in bounded, contiguous slices."""

from dataclasses import dataclass
import json
import math
from pathlib import Path
import struct

import numpy as np


# Safetensors stores little-endian data. BF16 shares the high bits of IEEE F32.
FLOAT_DTYPES = {"BF16": "<u2", "F16": "<f2", "F32": "<f4", "F64": "<f8"}
OTHER_WIDTHS = {
    "BOOL": 1, "I8": 1, "U8": 1, "I16": 2, "U16": 2, "I32": 4,
    "U32": 4, "I64": 8, "U64": 8, "F8_E4M3": 1, "F8_E5M2": 1,
}


@dataclass(frozen=True)
class Tensor:
    shard: Path
    offset: int
    shape: tuple
    dtype: str

    @property
    def numel(self):
        return math.prod(self.shape)

    def map(self):
        return np.memmap(self.shard, mode="r", dtype=FLOAT_DTYPES[self.dtype],
                         shape=(self.numel,), offset=self.offset)

    def decode(self, values):
        if self.dtype == "BF16":
            values = (values.astype(np.uint32) << 16).view(np.float32)
        return values.astype(np.float64)


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate safetensors or index key: {key}")
        result[key] = value
    return result


def shard_tensors(path):
    with path.open("rb") as stream:
        count, = struct.unpack("<Q", stream.read(8))
        if not 0 < count <= 100_000_000:
            raise ValueError(f"Invalid safetensors header length in {path}")
        header = json.loads(stream.read(count), object_pairs_hook=unique_object)
    header.pop("__metadata__", None)
    data_offset = 8 + count
    end_of_data = path.stat().st_size - data_offset
    last_end = 0
    for name, entry in sorted(header.items(), key=lambda item: item[1]["data_offsets"]):
        start, end = entry["data_offsets"]
        dtype, shape = entry["dtype"], tuple(entry["shape"])
        width = np.dtype(FLOAT_DTYPES[dtype]).itemsize if dtype in FLOAT_DTYPES else OTHER_WIDTHS[dtype]
        if (any(type(dim) is not int or dim < 0 for dim in shape)
                or start != last_end or end > end_of_data
                or end - start != math.prod(shape) * width):
            raise ValueError(f"Invalid safetensors bounds in {path} for {name}")
        last_end = end
        yield name, Tensor(path, data_offset + start, shape, dtype)
    if last_end != end_of_data:
        raise ValueError(f"Unindexed bytes in safetensors shard {path}")


def indexed_tensors(path, expected_shards):
    index = path / "model.safetensors.index.json"
    weights = json.loads(index.read_text(), object_pairs_hook=unique_object)["weight_map"]
    if set(weights.values()) != set(expected_shards):
        raise ValueError(f"Unexpected indexed shards in {path}")
    tensors = {}
    for shard in sorted(expected_shards):
        for name, tensor in shard_tensors(path / shard):
            if name in tensors or weights.get(name) != shard:
                raise ValueError(f"Shard/index disagreement for {name} in {path}")
            tensors[name] = tensor
    if tensors.keys() != weights.keys():
        raise ValueError(f"Index references absent tensors in {path}")
    return tensors
