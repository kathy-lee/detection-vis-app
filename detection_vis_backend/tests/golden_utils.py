"""Byte-level digests of dataset samples, shared by the capture script and the golden test.

A sample is the dict one __getitem__ returns. Arrays are digested over their raw bytes in
C order together with dtype and shape, so a transposed axis, a dtype change or a single
flipped bit all fail the comparison. Strings (image_path) are kept as-is.
"""
import hashlib

import numpy as np


def digest_array(arr):
    a = np.ascontiguousarray(np.asarray(arr))
    return {"sha256": hashlib.sha256(a.tobytes()).hexdigest(),
            "dtype": str(a.dtype),
            "shape": list(a.shape)}


def digest_sample(sample):
    out = {}
    for key, value in sample.items():
        if isinstance(value, str):
            out[key] = value
        else:
            out[key] = digest_array(value)
    return out
