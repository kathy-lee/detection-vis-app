#!/usr/bin/env python
"""Capture or verify golden sample digests for RADDet and DAROD on RADDetDataset.

Run ONCE on the unrefactored code to write the fixture, then with --verify after every
change that touches sample construction. The fixture holds SHA-256 digests of every
array in every sample, so equality is byte-level, not metric-level.

    python scripts/capture_raddet_golden.py            # write the fixture
    python scripts/capture_raddet_golden.py --verify   # rebuild and compare

Frames are taken from the OFFICIAL VAL split (no augmentation on that path). One extra
train-path DAROD sample is recorded under a fixed numpy seed to pin the flip augmentation.
"""
import argparse
import json
import os
import pickle
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

FIXTURE = os.path.join(REPO_ROOT, "detection_vis_backend", "tests", "fixtures", "raddet_golden.json")
DATASET = "RADDetDataset"
# (model, features, train-config variant). RADDet's campaign row used the reference config;
# DAROD has a single nested config. Neither variant changes how a sample is BUILT except
# through transformations / data_mean / data_std, which the fixture embeds verbatim.
CASES = [("RADDet", ["RAD"], "_reference"), ("DAROD", ["RD"], "")]
TRAIN_SEED = 123


def pick_indices(root_path):
    """A small, self-describing set of val frames: the first three, the last, the first
    with a DAROD wrap-around box (x1 < 0 or x2 >= 64 after the RD projection) and the
    first with no objects, when such frames exist."""
    with open(os.path.join(root_path, "original_split_info.json")) as f:
        val_names = sorted(json.load(f)["val"])
    gt_dir = os.path.join(root_path, "gt")
    names = sorted(os.path.splitext(n)[0] for n in os.listdir(gt_dir))
    val_idx = [names.index(n) for n in val_names if n in names]
    chosen = {"first": val_idx[:3], "last": [val_idx[-1]], "wrap": [], "empty": []}
    for i in val_idx:
        with open(os.path.join(gt_dir, names[i] + ".pickle"), "rb") as f:
            gt = pickle.load(f)
        if not chosen["empty"] and len(gt["classes"]) == 0:
            chosen["empty"].append(i)
        if not chosen["wrap"]:
            for box in gt["boxes"]:
                xc, w = box[2], box[5]
                if int(xc - w / 2) < 0 or int(xc + w / 2) >= 64:
                    chosen["wrap"].append(i)
                    break
        if chosen["empty"] and chosen["wrap"]:
            break
    return chosen


def build(model, features, variant, splittype, file_row):
    """Return (adapter, model_cfg, train_cfg) for one case on the CURRENT code.
    Before 2026-10-15 this called RADDetDataset.prepare_for_train; the fixture was
    captured through that path and the adapters must keep matching it."""
    from smoke_sweep import resolve_configs
    from detection_vis_backend.datasets.dataset import DatasetFactory
    from detection_vis_backend.datasets.adapters import make_sample_adapter
    model_cfg, train_cfg = resolve_configs(model, DATASET, features, variant)
    factory = DatasetFactory()
    parser = factory.get_shared_instance(DATASET, file_row["id"],
                                         parse_args=(file_row["id"], file_row["path"],
                                                     file_row["name"], file_row["config"]))
    adapter = make_sample_adapter(model, parser, features, model_cfg, train_cfg, splittype=splittype)
    return adapter, model_cfg, train_cfg


def capture(file_row):
    chosen = pick_indices(file_row["path"])
    indices = sorted({i for group in chosen.values() for i in group})
    fixture = {"dataset": DATASET, "file_id": file_row["id"], "file_name": file_row["name"],
               "root_path": file_row["path"], "config": file_row["config"],
               "tmp_rootdir": os.environ["TMP_ROOTDIR"],
               "chosen": chosen, "train_seed": TRAIN_SEED, "cases": []}
    for model, features, variant in CASES:
        ds, model_cfg, train_cfg = build(model, features, variant, "val", file_row)
        case = {"model": model, "features": features, "variant": variant,
                "model_cfg": model_cfg, "train_cfg": train_cfg,
                "n_frames": len(ds), "val": {}}
        for i in indices:
            case["val"][str(i)] = digest_sample(ds[i])
            print(f"  {model} val[{i}] ok", flush=True)
        if model == "DAROD":
            ds_train, _, _ = build(model, features, variant, "train", file_row)
            case["train_seeded"] = {}
            for i in indices:
                np.random.seed(TRAIN_SEED)
                case["train_seeded"][str(i)] = digest_sample(ds_train[i])
            print(f"  {model} train (seed {TRAIN_SEED}) ok", flush=True)
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
        ds, _, _ = build(case["model"], case["features"], case["variant"], "val", file_row)
        if len(ds) != case["n_frames"]:
            errors.append(f"{case['model']}: len {len(ds)} != {case['n_frames']}")
        for i, expected in case["val"].items():
            diff(expected, digest_sample(ds[int(i)]), f"{case['model']} val[{i}]", errors)
        if "train_seeded" in case:
            ds_train, _, _ = build(case["model"], case["features"], case["variant"], "train", file_row)
            for i, expected in case["train_seeded"].items():
                np.random.seed(fixture["train_seed"])
                diff(expected, digest_sample(ds_train[int(i)]),
                     f"{case['model']} train_seeded[{i}]", errors)
    return errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--verify", action="store_true")
    args = p.parse_args()
    from smoke_sweep import fetch_datafiles
    file_row = fetch_datafiles()[DATASET][0]
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
    print(f"wrote {FIXTURE}: frames {sorted(fixture['chosen'].items())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
