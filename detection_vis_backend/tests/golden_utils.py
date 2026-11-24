"""Byte-level digests of dataset samples, shared by the capture script and the golden test.

A sample is the dict one __getitem__ returns. Arrays are digested over their raw bytes in
C order together with dtype and shape, so a transposed axis, a dtype change or a single
flipped bit all fail the comparison. Strings (image_path) are kept relative to the dataset
root, as the fixtures' own paths are, so a fixture is the same on every machine.
"""
import hashlib
import json

import numpy as np

from detection_vis_backend.paths import dataset_path, relative_to_dataset_root, tmp_root


def load_golden(path):
    """A golden fixture with its root_path and config joined to this machine's dataset
    root, and tmp_rootdir set to this machine's TMP_ROOTDIR."""
    with open(path) as f:
        golden = json.load(f)
    for key in ("root_path", "config"):
        if golden.get(key):
            golden[key] = dataset_path(golden[key])
    if "root_path" in golden:
        golden["tmp_rootdir"] = tmp_root()
    return golden


def load_train_sample(path):
    """A train() request fixture (datafiles, features, configs) with each datafile's path
    and config joined to this machine's dataset root."""
    with open(path) as f:
        sample = json.load(f)
    for datafile in sample["datafiles"]:
        for key in ("path", "config"):
            if datafile.get(key):
                datafile[key] = dataset_path(datafile[key])
    return sample


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
            out[key] = relative_to_dataset_root(value)
        elif isinstance(value, (list, dict)) and np.asarray(value).dtype == object:
            out[key] = digest_object(value)
        else:
            out[key] = digest_array(value)
    return out
