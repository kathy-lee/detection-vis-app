"""Byte-level regression: the adapters must rebuild exactly the samples RADDetDataset
built before the refactor (fixture captured by scripts/capture_raddet_golden.py on
2026-10-15, unrefactored code). Needs the dataset on disk; skipped elsewhere.
"""
import json
import os

import numpy as np
import pytest

from detection_vis_backend.tests.golden_utils import digest_sample

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "raddet_golden.json")
with open(FIXTURE) as _f:
    GOLDEN = json.load(_f)

pytestmark = pytest.mark.skipif(not os.path.isdir(GOLDEN["root_path"]),
                                reason=f"RADDet dataset not at {GOLDEN['root_path']}")


@pytest.fixture(scope="module")
def parser():
    # __init__ computes the cache dir from TMP_ROOTDIR, so set it before constructing.
    os.environ["TMP_ROOTDIR"] = GOLDEN["tmp_rootdir"]
    from detection_vis_backend.datasets.dataset import RADDetDataset
    ds = RADDetDataset()
    ds.parse(GOLDEN["file_id"], GOLDEN["root_path"], GOLDEN["file_name"], GOLDEN["config"])
    return ds


def _assert_same(expected, actual, where):
    assert set(actual) == set(expected), f"{where}: keys {sorted(actual)} != {sorted(expected)}"
    for key, exp in expected.items():
        assert actual[key] == exp, f"{where}[{key!r}]: {actual[key]} != {exp}"


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=[c["model"] for c in GOLDEN["cases"]])
def test_val_samples_are_byte_identical(parser, case):
    from detection_vis_backend.datasets.adapters import make_sample_adapter
    adapter = make_sample_adapter(case["model"], parser, case["features"],
                                  case["model_cfg"], case["train_cfg"], splittype="val")
    assert len(adapter) == case["n_frames"]
    for i, expected in case["val"].items():
        _assert_same(expected, digest_sample(adapter[int(i)]), f"{case['model']} val[{i}]")


@pytest.mark.parametrize("case", [c for c in GOLDEN["cases"] if "train_seeded" in c],
                         ids=lambda c: c["model"])
def test_train_path_with_fixed_seed_is_byte_identical(parser, case):
    from detection_vis_backend.datasets.adapters import make_sample_adapter
    adapter = make_sample_adapter(case["model"], parser, case["features"],
                                  case["model_cfg"], case["train_cfg"], splittype="train")
    for i, expected in case["train_seeded"].items():
        np.random.seed(GOLDEN["train_seed"])
        _assert_same(expected, digest_sample(adapter[int(i)]), f"{case['model']} train[{i}]")
