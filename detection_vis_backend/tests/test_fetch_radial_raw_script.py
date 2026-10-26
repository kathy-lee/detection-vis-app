"""scripts/fetch_radial_raw.py downloads only recordings that still owe benchmark frames,
and deletes only what its own zip unpacked."""
import importlib
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

LABELS = pd.DataFrame({
    "numSample": [18, 37, 38, 90],
    "dataset": ["RECORD@a", "RECORD@b", "RECORD@b", "RECORD@c"],
    "index": [26, 5, 6, 3],
})


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(os.path.join(REPO_ROOT, "scripts"))
    sys.modules.pop("fetch_radial_raw", None)
    yield importlib.import_module("fetch_radial_raw")
    sys.modules.pop("fetch_radial_raw", None)


def test_pending_skips_finished_recordings_and_those_without_benchmark_frames(script, tmp_path):
    np.save(tmp_path / "adc_000018.npy", np.zeros(1))      # RECORD@a: done
    np.save(tmp_path / "adc_000037.npy", np.zeros(1))      # RECORD@b: half done
    listing = [("labels_CVPR.csv", "id0"), ("RECORD@a.zip", "id1"), ("RECORD@b.zip", "id2"),
               ("RECORD@c.zip", "id3"), ("RECORD@z.zip", "id4")]
    assert script.pending(listing, LABELS, str(tmp_path)) == [
        ("RECORD@b", "id2", [(38, 6)]),
        ("RECORD@c", "id3", [(90, 3)]),
    ]


def _touch(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"")


@pytest.mark.parametrize("layout", ["RECORD@b/RECORD@b_radar_ch0.bin", "RECORD@b_radar_ch0.bin"])
def test_recording_dir_is_found_whether_the_zip_nests_a_folder_or_not(script, tmp_path, layout):
    _touch(tmp_path / layout)
    assert script.recording_dir(str(tmp_path), "RECORD@b") == os.path.dirname(str(tmp_path / layout))


def test_recording_dir_rejects_a_zip_without_the_recordings_radar_files(script, tmp_path):
    _touch(tmp_path / "RECORD@q" / "RECORD@q_radar_ch0.bin")
    with pytest.raises(FileNotFoundError):
        script.recording_dir(str(tmp_path), "RECORD@b")
