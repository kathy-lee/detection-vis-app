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
    # __init__ computes the cache dir from TMP_ROOTDIR, so set it before constructing, and
    # restore it afterwards so later tests do not write caches under the dataset disk.
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TMP_ROOTDIR", GOLDEN["tmp_rootdir"])
        from detection_vis_backend.datasets.dataset import RADDetDataset
        ds = RADDetDataset()
        ds.parse(GOLDEN["file_id"], GOLDEN["root_path"], GOLDEN["file_name"], GOLDEN["config"])
        yield ds


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
def test_train_path_with_fixed_seeds_is_byte_identical(parser, case):
    """One seed per augmentation branch (hflip, vflip, none), so the flips themselves and
    the single draw per sample are compared against the pre-refactor output."""
    from detection_vis_backend.datasets.adapters import make_sample_adapter
    adapter = make_sample_adapter(case["model"], parser, case["features"],
                                  case["model_cfg"], case["train_cfg"], splittype="train")
    assert [int(s) for s in case["train_seeded"]] == GOLDEN["train_seeds"]
    for seed, per_frame in case["train_seeded"].items():
        for i, expected in per_frame.items():
            np.random.seed(int(seed))
            _assert_same(expected, digest_sample(adapter[int(i)]),
                         f"{case['model']} train[seed {seed}][{i}]")


@pytest.mark.parametrize("case", [c for c in GOLDEN["cases"] if "train_seeded" in c],
                         ids=lambda c: c["model"])
def test_fixture_seeds_cover_three_distinct_augmentations(case):
    """Guards the fixture itself: if every seed produced the same bytes, no flip was
    ever recorded and the train-path test would prove nothing."""
    feature = case["features"][0]
    for i in next(iter(case["train_seeded"].values())):
        digests = {per_frame[i][feature]["sha256"] for per_frame in case["train_seeded"].values()}
        assert len(digests) == 3, f"frame {i}: seeds did not produce hflip, vflip and no-flip"


def test_val_path_draws_no_random_numbers(parser):
    """A val adapter must not consume the global RNG: the draw order of a training run
    is part of its result."""
    from detection_vis_backend.datasets.adapters import make_sample_adapter
    case = next(c for c in GOLDEN["cases"] if c["model"] == "DAROD")
    adapter = make_sample_adapter("DAROD", parser, case["features"],
                                  case["model_cfg"], case["train_cfg"], splittype="val")
    np.random.seed(0)
    before = np.random.get_state()[1].copy()
    adapter[int(next(iter(case["val"])))]
    assert np.array_equal(np.random.get_state()[1], before)
