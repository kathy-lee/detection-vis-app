"""Byte-level digests of dataset samples, shared by the capture script and the golden test.

A sample is the dict one __getitem__ returns. Arrays are digested over their raw bytes in
C order together with dtype and shape, so a transposed axis, a dtype change or a single
flipped bit all fail the comparison. Strings (image_path) are kept as-is.
"""
import hashlib
import json

import numpy as np


def digest_array(arr):
    a = np.ascontiguousarray(np.asarray(arr))
    return {"sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "dtype": str(a.dtype),
            "shape": list(a.shape)}


def _plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    return str(value)


def digest_object(value):
    """Nested lists/dicts (CRUW's per-frame obj_infos) digested over their canonical JSON:
    an object-dtype array would hash the element POINTERS, which differ per process."""
    text = json.dumps(value, sort_keys=True, default=_plain)
    return {"sha256": hashlib.sha256(text.encode()).hexdigest(), "dtype": "json"}


def digest_sample(sample):
    out = {}
    for key, value in sample.items():
        if isinstance(value, str):
            out[key] = value
        elif isinstance(value, (list, dict)) and np.asarray(value).dtype == object:
            out[key] = digest_object(value)
        else:
            out[key] = digest_array(value)
    return out
