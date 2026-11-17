"""File/array hashes and the existing Phase 3B output seal."""

import hashlib
from pathlib import Path

import numpy as np

from .io import read, save


def require(ok, message):
    if not ok:
        raise ValueError(message)


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def array_hash(a):
    a = np.ascontiguousarray(a)
    require(not a.dtype.hasobject, 'Object array forbidden')
    h = hashlib.sha256()
    h.update(str(a.dtype).encode()); h.update(repr(a.shape).encode()); h.update(a.tobytes())
    return h.hexdigest()


def seal(out, details):
    details = dict(details, output_hashes={str(p.relative_to(out)): sha(p) for p in sorted(out.rglob('*')) if p.is_file()})
    save(out/'validation.json', details)
    return details


def verify_seal(out):
    v = read(out/'validation.json')
    require(v['status']=='PASS', f'Validation did not pass: {out}')
    for name, digest in v['output_hashes'].items():
        require(sha(out/name)==digest, f'Changed derived output: {name}')
    return v
