"""Byte-level regression: FFTRadNet's RADIal adapter must rebuild exactly the samples RADIal
built before the move to sample adapters (fixture captured by scripts/capture_radial_golden.py
on 2026-10-27, unmigrated code). Needs the dataset and the DB row; skipped elsewhere.
"""
import json
import os
import sys

import pytest

from detection_vis_backend.tests.golden_utils import digest_sample

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "radial_golden.json")
ROOT = "/home/kangle/dataset/radial/RADIal-labeled"
with open(FIXTURE) as _f:
    GOLDEN = json.load(_f)

pytestmark = pytest.mark.skipif(not os.path.isdir(ROOT), reason=f"RADIal not at {ROOT}")


@pytest.fixture(scope="module")
def parser():
    from detection_vis_backend.datasets.dataset import RADIal
    ds = RADIal()
    ds.parse(GOLDEN["file_id"], os.path.dirname(ROOT), os.path.basename(ROOT), "")
    return ds


def test_samples_are_byte_identical(parser):
    sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))
    from smoke_sweep import resolve_configs
    from detection_vis_backend.datasets.adapters import make_sample_adapter
    model_cfg, train_cfg = resolve_configs(GOLDEN["model"], "RADIal", GOLDEN["features"])
    adapter = make_sample_adapter(GOLDEN["model"], parser, GOLDEN["features"], model_cfg, train_cfg)
    assert len(adapter) == GOLDEN["n_frames"]
    for i, expected in GOLDEN["samples"].items():
        assert digest_sample(adapter[int(i)]) == expected, f"frame {i}"


def test_original_split_ids_are_unchanged(parser):
    from detection_vis_backend.datasets.adapters.radial import radial_sequence_split
    splits = radial_sequence_split(parser)
    for name in ("train", "val", "test"):
        assert list(splits[name]) == GOLDEN["splits"][name], name
