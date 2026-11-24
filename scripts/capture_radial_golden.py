#!/usr/bin/env python
"""Capture or verify golden sample digests for FFTRadNet on RADIal.

Run ONCE on the unrefactored code to write the fixture, then with --verify after every
change that touches sample construction. The fixture holds SHA-256 digests of every
array in every sample, so equality is byte-level, not metric-level.

    python scripts/capture_radial_golden.py            # write the fixture
    python scripts/capture_radial_golden.py --verify   # rebuild and compare

Frames: the first, the last, the first with no object (labels row -1), the first marked
Difficult, and the first two of the valeo test recordings. The fixture also holds the
train/val/test frame indices of the valeo sequence split ("original" splitmode).

The fixture was captured at commit e25f3ff plus the uncommitted SSMRadNet loss, where
RADIal still built samples in prepare_for_train/__getitem__; build() handles both that
API and the adapters, so the same script captures on the old code and verifies on the new.
"""
import argparse
import json
import os
import sys

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

from detection_vis_backend.paths import load_env  # noqa: E402

load_env()
os.environ.setdefault("LOGURU_LEVEL", "ERROR")
os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost/postgres")

from detection_vis_backend.tests.golden_utils import digest_sample  # noqa: E402

FIXTURE = os.path.join(REPO_ROOT, "detection_vis_backend", "tests", "fixtures", "radial_golden.json")
DATASET = "RADIal"
MODEL, FEATURES = "FFTRadNet", ["RD"]
# valeo's sequence split, as train.py's "original" splitmode has always used it.
VALEO_SEQUENCES = {'val': ['RECORD@2020-11-22_12.49.56', 'RECORD@2020-11-22_12.11.49',
                           'RECORD@2020-11-22_12.28.47', 'RECORD@2020-11-21_14.25.06'],
                   'test': ['RECORD@2020-11-22_12.45.05', 'RECORD@2020-11-22_12.25.47',
                            'RECORD@2020-11-22_12.03.47', 'RECORD@2020-11-22_12.54.38']}


def build(file_row):
    """(indexable sample source, parser) on the CURRENT code: a sample adapter, or on the
    pre-refactor code the configured dataset copy (which is its own parser)."""
    from smoke_sweep import resolve_configs
    from detection_vis_backend.datasets.dataset import DatasetFactory
    model_cfg, train_cfg = resolve_configs(MODEL, DATASET, FEATURES)
    factory = DatasetFactory()
    factory.get_shared_instance(DATASET, file_row["id"],
                                parse_args=(file_row["id"], file_row["path"],
                                            file_row["name"], file_row["config"]))
    parser = factory.get_instance(DATASET, file_row["id"])
    if hasattr(parser, "prepare_for_train"):          # pre-refactor RADIal
        parser.prepare_for_train(FEATURES, train_cfg, model_cfg)
        return parser, parser
    from detection_vis_backend.datasets.adapters import make_sample_adapter
    return make_sample_adapter(MODEL, parser, FEATURES, model_cfg, train_cfg), parser


def split_ids(parser):
    """train.py's "original" RADIal split at capture time, over the parser's frames."""
    labels = parser.labels
    key_to_index = {s: i for i, s in enumerate(parser.sample_keys)}
    out = {}
    for split in ('val', 'test'):
        keys = np.unique(np.concatenate([labels[labels[:, 14] == seq, 0] for seq in VALEO_SEQUENCES[split]]))
        out[split] = [key_to_index[k] for k in keys]
    out['train'] = np.setdiff1d(np.arange(len(parser)), np.concatenate([out['val'], out['test']])).tolist()
    return out


def pick_indices(parser, splits):
    labels = parser.labels
    key_to_index = {s: i for i, s in enumerate(parser.sample_keys)}
    empty = next(key_to_index[r[0]] for r in labels if r[1] == -1)
    difficult = next(key_to_index[r[0]] for r in labels if r[-1] == 1)
    return {"first": [0], "last": [len(parser) - 1], "empty": [empty],
            "difficult": [difficult], "test": splits["test"][:2]}


def capture(file_row):
    ds, parser = build(file_row)
    splits = split_ids(parser)
    chosen = pick_indices(parser, splits)
    fixture = {"dataset": DATASET, "file_id": file_row["id"], "model": MODEL, "features": FEATURES,
               "n_frames": len(ds), "chosen": chosen, "splits": splits, "samples": {}}
    for i in sorted({i for group in chosen.values() for i in group}):
        fixture["samples"][str(i)] = digest_sample(ds[i])
        print(f"  {MODEL} [{i}] ok", flush=True)
    return fixture


def verify(file_row):
    with open(FIXTURE) as f:
        fixture = json.load(f)
    ds, parser = build(file_row)
    errors = []
    if len(ds) != fixture["n_frames"]:
        errors.append(f"len {len(ds)} != {fixture['n_frames']}")
    for i, expected in fixture["samples"].items():
        actual = digest_sample(ds[int(i)])
        if actual != expected:
            errors.append(f"[{i}]: expected {expected} got {actual}")
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
    print(f"wrote {FIXTURE}: frames {sorted(fixture['chosen'].items())}, "
          f"split sizes {[(k, len(v)) for k, v in fixture['splits'].items()]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
