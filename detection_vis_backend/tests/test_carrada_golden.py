"""Byte-level regression: the CARRADA adapters must rebuild exactly the samples CARRADA
built before the refactor (fixture captured by scripts/capture_carrada_golden.py on
2026-10-22, unrefactored code, commit 6dc18f0). Needs the dataset on disk."""
import json
import os

import numpy as np
import pytest

from detection_vis_backend.tests.golden_utils import digest_sample

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "carrada_golden.json")
with open(FIXTURE) as _f:
    GOLDEN = json.load(_f)

pytestmark = pytest.mark.skipif(not os.path.isdir(GOLDEN["root_path"]),
                                reason=f"CARRADA dataset not at {GOLDEN['root_path']}")
SPLITS = ("train", "val", "test")


@pytest.fixture(scope="module")
def parser():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TMP_ROOTDIR", GOLDEN["tmp_rootdir"])
        from detection_vis_backend.datasets.dataset import DatasetFactory
        factory = DatasetFactory()
        factory.get_shared_instance("CARRADA", GOLDEN["file_id"],
                                    parse_args=(GOLDEN["file_id"], GOLDEN["root_path"],
                                                GOLDEN["file_name"], GOLDEN["config"]))
        yield factory.get_instance("CARRADA", GOLDEN["file_id"])


def _assert_same(expected, actual, where):
    assert set(actual) == set(expected), f"{where}: keys {sorted(actual)} != {sorted(expected)}"
    for key, exp in expected.items():
        assert actual[key] == exp, f"{where}[{key!r}]: {actual[key]} != {exp}"


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["id"])
def test_samples_are_byte_identical(parser, case, split):
    from detection_vis_backend.datasets.adapters import make_sample_adapter
    adapter = make_sample_adapter(case["model"], parser, case["features"],
                                  case["model_cfg"], case["train_cfg"], splittype=split)
    assert len(adapter) == case["n"][split]
    for seed, per_index in case[split].items():
        for i, expected in per_index.items():
            np.random.seed(int(seed))
            _assert_same(expected, digest_sample(adapter[int(i)]), f"{case['id']} {split}[seed {seed}][{i}]")


def test_fixture_seeds_cover_distinct_augmentations_on_train():
    """Guards the fixture: if no seed changed a train sample, the flips were never recorded."""
    for case in GOLDEN["cases"]:
        key = case["features"][0]
        digests = {per["1"][key]["sha256"] for per in case["train"].values()}
        assert len(digests) > 1, case["id"]
