"""SSMRadNet's RADIal adapter: raw ADC frames decoded as the authors' convert_radar_cube,
served only for the frames whose adc file exists, with FFTRadNet's labels."""
import importlib
import os
import shutil
import sys

import numpy as np
import pandas as pd
import pytest
from PIL import Image

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
COLUMNS = ["numSample", "x1_pix", "y1_pix", "x2_pix", "y2_pix", "laser_X_m", "laser_Y_m", "laser_Z_m",
           "radar_X_m", "radar_Y_m", "radar_R_m", "radar_A_deg", "radar_D_mps", "radar_P_db",
           "dataset", "index", "Difficult"]
MODEL_CFG = {"segmentation_head": True}


def _raw(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(-2000, 2000, size=(1048576, 4), dtype=np.int16)


@pytest.fixture
def radial_root(tmp_path):
    """A RADIal-labeled tree: 3 frames over 2 recordings (frame 30 has no object),
    adc files for frames 10 and 30 only."""
    root = tmp_path / "RADIal-labeled"
    for d in ("camera", "laser_PCL", "radar_FFT", "radar_PCL", "radar_Freespace", "radar_ADC"):
        (root / d).mkdir(parents=True)
    rows = [[10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20.0, 3.0, 1.0, 50, "RECORD@A", 0, 0],
            [20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 35.0, -10.0, 2.0, 50, "RECORD@A", 1, 0],
            [30, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, "RECORD@B", 0, 0]]
    pd.DataFrame(rows, columns=COLUMNS).to_csv(root / "labels.csv", index=False)
    for n in (10, 20, 30):
        mask = np.zeros((512, 900), dtype=np.uint8)
        mask[100:300, 300:600] = 255
        Image.fromarray(mask).save(root / "radar_Freespace" / f"freespace_{n:06d}.png")
        # FFTRadNet's samples read the RD spectrum; zeros suffice for comparing labels.
        np.save(root / "radar_FFT" / f"fft_{n:06d}.npy", np.zeros((512, 256, 16), dtype=np.complex64))
    for n in (10, 30):
        np.save(root / "radar_ADC" / f"adc_{n:06d}.npy", _raw(n))
    return root


@pytest.fixture
def parser(radial_root):
    from detection_vis_backend.datasets.dataset import RADIal
    p = RADIal()
    p.parse(1, str(radial_root.parent), radial_root.name, "")
    return p


def _adapter(model, parser):
    from detection_vis_backend.datasets.adapters import make_sample_adapter
    return make_sample_adapter(model, parser, ["ADC"] if model == "SSMRadNet" else ["RD"], MODEL_CFG, {})


def test_decode_matches_the_extraction_script(monkeypatch):
    from detection_vis_backend.datasets.adapters.radial import decode_radial_adc
    monkeypatch.syspath_prepend(os.path.join(REPO_ROOT, "scripts"))
    sys.modules.pop("extract_radial_adc", None)
    script = importlib.import_module("extract_radial_adc")
    raw = _raw(0)
    cube = script.decode(raw)
    out = decode_radial_adc(raw)
    assert out.dtype == np.float32 and out.shape == (512, 256, 32)
    np.testing.assert_array_equal(out, np.concatenate([cube.real, cube.imag], axis=2).astype(np.float32))


def test_only_frames_with_an_adc_file_are_served(parser):
    adapter = _adapter("SSMRadNet", parser)
    assert len(adapter) == 2
    assert [parser.sample_keys[i] for i in adapter.frames] == [10, 30]


def test_sample_has_the_model_input_shape_and_labels(parser):
    ssm = _adapter("SSMRadNet", parser)
    fft = _adapter("FFTRadNet", parser)
    sample = ssm[0]
    assert sample["ADC"].shape == (512, 256, 32) and sample["ADC"].dtype == np.float32
    assert sample["encoded_label"].shape == (3, 128, 224)
    assert sample["seg_label"].shape == (256, 224)
    np.testing.assert_array_equal(sample["encoded_label"], fft[0]["encoded_label"])
    np.testing.assert_array_equal(sample["box_label"], fft[0]["box_label"])
    # Served frame 1 is parser frame 2 (numSample 30), the one without objects.
    np.testing.assert_array_equal(ssm[1]["encoded_label"], fft[2]["encoded_label"])


def test_missing_adc_folder_raises(radial_root, parser):
    shutil.rmtree(radial_root / "radar_ADC")
    with pytest.raises(FileNotFoundError, match="radar_ADC"):
        _adapter("SSMRadNet", parser)


def test_empty_adc_folder_raises(radial_root, parser):
    for f in (radial_root / "radar_ADC").iterdir():
        f.unlink()
    with pytest.raises(FileNotFoundError, match="radar_ADC"):
        _adapter("SSMRadNet", parser)


def test_fftradnet_still_sees_every_frame(parser):
    assert len(_adapter("FFTRadNet", parser)) == 3
