"""scripts/tfftradnet_input_statistics.py: per-channel input statistics of the T-FFTRadNet
modes that upstream's configs leave without them, computed as upstream's
print_dataset_statistics.py computes them."""
import importlib
import json
import os
import sys

import numpy as np
import pytest

from detection_vis_backend.paths import dataset_path

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATISTICS = os.path.join(REPO_ROOT, "detection_vis_backend", "datasets", "adapters", "tfftradnet_statistics.json")
REAL_RADDET = dataset_path("RADDet_reform_dataset")


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(os.path.join(REPO_ROOT, "scripts"))
    sys.modules.pop("tfftradnet_input_statistics", None)
    yield importlib.import_module("tfftradnet_input_statistics")
    sys.modules.pop("tfftradnet_input_statistics", None)


def test_channel_statistics_is_the_mean_of_per_frame_moments(script):
    a = np.stack([np.full((4, 4), 1.0), np.arange(16.0).reshape(4, 4)])       # channels: const 1, 0..15
    b = np.stack([np.full((4, 4), 3.0), np.zeros((4, 4))])
    mean, std = script.channel_statistics([a, b])
    np.testing.assert_allclose(mean, [2.0, 7.5 / 2])
    np.testing.assert_allclose(std, [0.0, np.arange(16.0).std() / 2])


def test_raddet_train_frames_follow_the_official_list(script, tmp_path):
    (tmp_path / "gt").mkdir()
    for n in ("000001", "000002", "000003"):
        (tmp_path / "gt" / f"{n}.pickle").write_bytes(b"")
    (tmp_path / "original_split_info.json").write_text(json.dumps({"train": ["000003", "000001"], "val": ["000002"]}))

    class Parser:
        root_path = str(tmp_path)
        anno_filenames = [str(tmp_path / "gt" / f"{n}.pickle") for n in ("000001", "000002", "000003")]

    assert script.raddet_train_frames(Parser()) == [0, 2]


@pytest.mark.skipif(not os.path.isdir(os.path.join(REAL_RADDET, "ADC")), reason="RADDet ADC absent")
def test_upstream_statistics_are_reproduced_on_raddet_rd(script):
    # Measured 2026-11-24: 50 frames are 4-9 % off upstream, 500 within 2 %, 2000 within 1 %.
    from detection_vis_backend.datasets.adapters.tfftradnet_common import load_tfftradnet_statistics
    mean, std = script.compute("RADDetDataset", "RD", n_frames=500)
    upstream = np.array(load_tfftradnet_statistics("RADDetDataset", "RD")["input_std"])
    assert np.abs(np.array(std) / upstream - 1).max() < 0.03


def test_statistics_json_has_no_null_entry():
    with open(STATISTICS) as f:
        stats = json.load(f)
    missing = [f"{d} {m}" for d, modes in stats.items() if not d.startswith("_")
               for m, v in modes.items() if v is None]
    assert not missing, f"not computed yet: {missing}"
