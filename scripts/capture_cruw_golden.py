#!/usr/bin/env python
"""Capture or verify golden sample digests for every network that reads CRUW.

Run ONCE on the unrefactored code to write the fixture, then with --verify after every
change that touches sample construction. The fixture holds SHA-256 digests of every
array in every sample, so equality is byte-level, not metric-level.

    python scripts/capture_cruw_golden.py            # write the fixture
    python scripts/capture_cruw_golden.py --verify   # rebuild and compare

One sequence (2019_04_09_BMS1001, 897 frames) is windowed through all three splits:
train and val at train_stride, test at test_stride. The single-chirp paths draw the chirp
with Python's `random`, so every sample is captured under a fixed `random` seed and
replayed under the same seed; the multi-chirp paths ignore the seed.

The fixture was captured at commit b8f7e75, where CRUW still built samples in
prepare_for_train/__getitem__; build() handles both that API and the adapters, so the
same script captures on the old code and verifies on the new.
"""
import argparse
import json
import os
import random
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

os.environ.setdefault("MODEL_ROOTDIR", "/home/kangle/dataset/trained_models")
os.environ.setdefault("DATASET_ROOTDIR", "/home/kangle/dataset")
os.environ.setdefault("TMP_ROOTDIR", "/home/kangle/dataset/tmp")
os.environ.setdefault("LOGURU_LEVEL", "ERROR")
os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost/postgres")

from detection_vis_backend.tests.golden_utils import digest_sample  # noqa: E402
from smoke_sweep import RODNET_VARIANTS  # noqa: E402

FIXTURE = os.path.join(REPO_ROOT, "detection_vis_backend", "tests", "fixtures", "cruw_golden.json")
DATASET = "CRUW"
SEQUENCE = "2019_04_09_BMS1001"
# (model, features, train-config variant). The RODNet campaign rows used the reference
# config; RECORD has no variant. Neither changes how a sample is built except through
# win_size / strides, which the fixture embeds verbatim.
CASES = ([(m, ["RA"], "_reference") for m in RODNET_VARIANTS]
         + [(m, ["RA"], "") for m in ("RECORD", "RECORDNoLstm", "RECORDNoLstmMulti")])
SPLITS = ("train", "val", "test")
CHIRP_SEEDS = [0, 1, 2]


def pick_indices(n_windows):
    """A small, self-describing set of windows: the first two, the middle, the last."""
    return {"first": [0, 1], "middle": [n_windows // 2], "last": [n_windows - 1]}


def build(model, features, variant, splittype, file_row):
    """Return (indexable sample source, model_cfg, train_cfg) for one case on the CURRENT
    code: a sample adapter, or on the pre-refactor code a configured dataset copy."""
    from smoke_sweep import resolve_configs
    from detection_vis_backend.datasets.dataset import DatasetFactory
    model_cfg, train_cfg = resolve_configs(model, DATASET, features, variant)
    factory = DatasetFactory()
    factory.get_shared_instance(DATASET, file_row["id"],
                                parse_args=(file_row["id"], file_row["path"],
                                            file_row["name"], file_row["config"]))
    ds = factory.get_instance(DATASET, file_row["id"])
    ds.prepare_for_train(features, train_cfg, model_cfg, splittype=splittype)
    if hasattr(ds, "datasamples_length"):          # pre-refactor CRUW: windows itself
        return ds, model_cfg, train_cfg
    from detection_vis_backend.datasets.adapters import make_sample_adapter
    adapter = make_sample_adapter(model, ds, features, model_cfg, train_cfg, splittype=splittype)
    return adapter, model_cfg, train_cfg


def _sample(ds, i, seed):
    random.seed(seed)
    return digest_sample(ds[i])


def capture(file_row):
    fixture = {"dataset": DATASET, "file_id": file_row["id"], "file_name": file_row["name"],
               "root_path": file_row["path"], "config": file_row["config"],
               "tmp_rootdir": os.environ["TMP_ROOTDIR"],
               "chirp_seeds": CHIRP_SEEDS, "chosen": {}, "cases": []}
    for model, features, variant in CASES:
        case = {"model": model, "features": features, "variant": variant, "n_windows": {}}
        for split in SPLITS:
            ds, model_cfg, train_cfg = build(model, features, variant, split, file_row)
            case["model_cfg"], case["train_cfg"] = model_cfg, train_cfg
            n = len(ds)
            case["n_windows"][split] = n
            chosen = pick_indices(n)
            fixture["chosen"].setdefault(split, chosen)
            indices = sorted({i for group in chosen.values() for i in group})
            case[split] = {}
            for seed in CHIRP_SEEDS:
                case[split][str(seed)] = {str(i): _sample(ds, i, seed) for i in indices}
            print(f"  {model} {split} ({n} windows) ok", flush=True)
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
            if len(ds) != case["n_windows"][split]:
                errors.append(f"{case['model']} {split}: len {len(ds)} != {case['n_windows'][split]}")
            for seed, per_index in case[split].items():
                for i, expected in per_index.items():
                    diff(expected, _sample(ds, int(i), int(seed)),
                         f"{case['model']} {split}[seed {seed}][{i}]", errors)
    return errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    from smoke_sweep import fetch_datafiles
    file_row = next(r for r in fetch_datafiles()[DATASET] if r["name"] == SEQUENCE)
    if args.verify:
        errors = verify(file_row)
        for e in errors:
            print("MISMATCH", e)
        print("verify:", "OK" if not errors else f"{len(errors)} mismatches")
        return 1 if errors else 0
    fixture = capture(file_row)
    os.makedirs(os.path.dirname(FIXTURE), exist_ok=True)
    with open(FIXTURE, "w") as f:
        json.dump(fixture, f, indent=1, default=str)
    print(f"wrote {FIXTURE}: windows {fixture['chosen']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
