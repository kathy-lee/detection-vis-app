"""Byte-level regression: the CRUW adapters must rebuild exactly the windows CRUW built
before the refactor (fixture captured by scripts/capture_cruw_golden.py on 2026-10-19,
unrefactored code, commit b8f7e75). Needs the dataset on disk; skipped elsewhere.
"""
import json
import os
import random

import pytest

from detection_vis_backend.tests.golden_utils import digest_sample

FIXTURE = os.path.join(os.path.dirname(__file__), "fixtures", "cruw_golden.json")
with open(FIXTURE) as _f:
    GOLDEN = json.load(_f)

pytestmark = pytest.mark.skipif(not os.path.isdir(GOLDEN["root_path"]),
                                reason=f"CRUW dataset not at {GOLDEN['root_path']}")
SPLITS = ("train", "val", "test")


@pytest.fixture(scope="module")
def parser():
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("TMP_ROOTDIR", GOLDEN["tmp_rootdir"])
        from detection_vis_backend.datasets.dataset import DatasetFactory
        factory = DatasetFactory()
        factory.get_shared_instance("CRUW", GOLDEN["file_id"],
                                    parse_args=(GOLDEN["file_id"], GOLDEN["root_path"],
                                                GOLDEN["file_name"], GOLDEN["config"]))
        yield factory.get_instance("CRUW", GOLDEN["file_id"])


def _assert_same(expected, actual, where):
    assert set(actual) == set(expected), f"{where}: keys {sorted(actual)} != {sorted(expected)}"
    for key, exp in expected.items():
        assert actual[key] == exp, f"{where}[{key!r}]: {actual[key]} != {exp}"


@pytest.mark.parametrize("split", SPLITS)
@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["model"])
def test_windows_are_byte_identical(parser, case, split):
    from detection_vis_backend.datasets.adapters import make_sample_adapter
    adapter = make_sample_adapter(case["model"], parser, case["features"],
                                  case["model_cfg"], case["train_cfg"], splittype=split)
    assert len(adapter) == case["n_windows"][split]
    for seed, per_index in case[split].items():
        for i, expected in per_index.items():
            random.seed(int(seed))
            _assert_same(expected, digest_sample(adapter[int(i)]),
                         f"{case['model']} {split}[seed {seed}][{i}]")


@pytest.mark.parametrize("model", ["RODNet_CDC", "RODNet_HG", "RODNet_HGwI"])
def test_fixture_seeds_cover_distinct_chirps_on_the_single_chirp_paths(model):
    """Guards the fixture itself: if every seed produced the same bytes, the chirp draw
    was never recorded and the seeded replay would prove nothing."""
    case = next(c for c in GOLDEN["cases"] if c["model"] == model)
    digests = {per_index["0"]["RA"]["sha256"] for per_index in case["val"].values()}
    assert len(digests) == len(GOLDEN["chirp_seeds"])
