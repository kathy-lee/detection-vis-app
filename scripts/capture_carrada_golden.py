#!/usr/bin/env python
"""Capture or verify golden sample digests for every network that reads CARRADA.

Run ONCE on the unrefactored code to write the fixture, then with --verify after every
change that touches sample construction. The fixture holds SHA-256 digests of every
array in every sample, so equality is byte-level, not metric-level.

    python scripts/capture_carrada_golden.py            # write the fixture
    python scripts/capture_carrada_golden.py --verify   # rebuild and compare

One sequence (2019-09-16-13-06-41, an official val sequence) is read through all three
splits: the split decides whether flips (RECORD family, DAROD) and noise
(RadarCrossAttention) are applied. Those draw from numpy's global RNG, so every sample is
captured under a fixed numpy seed and replayed under the same seed.

The fixture was captured at commit 6dc18f0, where CARRADA still built samples in
prepare_for_train/__getitem__; build() handles both that API and the adapters, so the
same script captures on the old code and verifies on the new.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

os.environ.setdefault("MODEL_ROOTDIR", "/home/kangle/dataset/trained_models")
os.environ.setdefault("DATASET_ROOTDIR", "/home/kangle/dataset")
os.environ.setdefault("TMP_ROOTDIR", "/home/kangle/dataset/tmp")
os.environ.setdefault("LOGURU_LEVEL", "ERROR")
os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost/postgres")

from detection_vis_backend.tests.golden_utils import digest_sample  # noqa: E402

FIXTURE = os.path.join(REPO_ROOT, "detection_vis_backend", "tests", "fixtures", "carrada_golden.json")
DATASET = "CARRADA"
SEQUENCE = "2019-09-16-13-06-41"
# (model, features, train-config variant): the variant each campaign row was run with.
CASES = [
    ("RECORD", ["RD"], "_reference"), ("RECORD", ["RA"], "_reference"),
    ("RECORDNoLstm", ["RD"], ""), ("RECORDNoLstm", ["RA"], ""),
    ("RECORDNoLstmMulti", ["RD"], ""), ("RECORDNoLstmMulti", ["RA"], ""),
    ("MVRECORD", ["RD", "RA", "AD"], "_reference"),
    ("DAROD", ["RD"], ""),
    ("RadarCrossAttention", ["RD", "RA", "AD"], "_reference"),
]
SPLITS = ("train", "val", "test")
SEEDS = [0, 1, 2]


def case_id(model, features):
    return f"{model}/{'+'.join(features)}"


def pick_indices(n):
    return {"first": [0, 1], "middle": [n // 2], "last": [n - 1]}


def build(model, features, variant, splittype, file_row):
    """(indexable sample source, model_cfg, train_cfg) on the CURRENT code: an adapter,
    or on the pre-refactor code a configured dataset copy."""
    from smoke_sweep import resolve_configs
    from detection_vis_backend.datasets.dataset import DatasetFactory
    model_cfg, train_cfg = resolve_configs(model, DATASET, features, variant)
    factory = DatasetFactory()
    factory.get_shared_instance(DATASET, file_row["id"],
                                parse_args=(file_row["id"], file_row["path"],
                                            file_row["name"], file_row["config"]))
    ds = factory.get_instance(DATASET, file_row["id"])
    ds.prepare_for_train(features, train_cfg, model_cfg, splittype=splittype)
    if hasattr(ds, "datasamples_length"):          # pre-refactor CARRADA builds samples itself
        return ds, model_cfg, train_cfg
    from detection_vis_backend.datasets.adapters import make_sample_adapter
    return make_sample_adapter(model, ds, features, model_cfg, train_cfg, splittype=splittype), model_cfg, train_cfg


def _sample(ds, i, seed):
    np.random.seed(seed)
    return digest_sample(ds[i])


def capture(file_row):
    fixture = {"dataset": DATASET, "file_id": file_row["id"], "file_name": file_row["name"],
               "root_path": file_row["path"], "config": file_row["config"],
               "tmp_rootdir": os.environ["TMP_ROOTDIR"], "seeds": SEEDS, "cases": []}
    for model, features, variant in CASES:
        case = {"id": case_id(model, features), "model": model, "features": features,
                "variant": variant, "n": {}}
        for split in SPLITS:
            ds, model_cfg, train_cfg = build(model, features, variant, split, file_row)
            case["model_cfg"], case["train_cfg"] = model_cfg, train_cfg
            n = len(ds)
            case["n"][split] = n
            indices = sorted({i for g in pick_indices(n).values() for i in g})
            case[split] = {str(s): {str(i): _sample(ds, i, s) for i in indices} for s in SEEDS}
            print(f"  {case['id']} {split} ({n} samples) ok", flush=True)
        fixture["cases"].append(case)
    return fixture


def diff(expected, actual, where, errors):
    for key, exp in expected.items():
        act = actual.get(key)
        if act != exp:
            errors.append(f"{where}[{key!r}]: expected {exp} got {act}")
    extra = set(actual) - set(expected)
    if extra:
        errors.append(f"{where}: unexpected keys {sorted(extra)}")


def verify(file_row):
    with open(FIXTURE) as f:
        fixture = json.load(f)
    errors = []
    for case in fixture["cases"]:
        for split in SPLITS:
            ds, _, _ = build(case["model"], case["features"], case["variant"], split, file_row)
            if len(ds) != case["n"][split]:
                errors.append(f"{case['id']} {split}: len {len(ds)} != {case['n'][split]}")
            for seed, per_index in case[split].items():
                for i, expected in per_index.items():
                    diff(expected, _sample(ds, int(i), int(seed)), f"{case['id']} {split}[seed {seed}][{i}]", errors)
    return errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    from smoke_sweep import fetch_datafiles
    file_row = next(r for r in fetch_datafiles()[DATASET] if r["name"] == SEQUENCE)
    if args.verify:
        errors = verify(file_row)
        for e in errors[:40]:
            print("MISMATCH", e)
        print("verify:", "OK" if not errors else f"{len(errors)} mismatches")
        return 1 if errors else 0
    fixture = capture(file_row)
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    with open(FIXTURE, "w") as f:
        json.dump(fixture, f, indent=1, default=str)
    print(f"wrote {FIXTURE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
