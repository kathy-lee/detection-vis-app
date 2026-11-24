"""T-FFTRadNet's RADIal adapter: each input mode built as the reference repo
(jgiroux8/T_FFTRadNet 68bc041) RadIal/dataset/dataset.py builds it, with FFTRadNet's labels
and freespace map."""
import os

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from detection_vis_backend.paths import dataset_path

COLUMNS = ["numSample", "x1_pix", "y1_pix", "x2_pix", "y2_pix", "laser_X_m", "laser_Y_m", "laser_Z_m",
           "radar_X_m", "radar_Y_m", "radar_R_m", "radar_A_deg", "radar_D_mps", "radar_P_db",
           "dataset", "index", "Difficult"]
REAL_RADIAL = dataset_path("radial", "RADIal-labeled")
FEATURE = {"RD": "RD", "RD_shift_as_published": "RD", "RD_shift": "RD", "RD_no_window": "RD", "ADC": "ADC"}


def _raw(seed):
    rng = np.random.default_rng(seed)
    return rng.integers(-2000, 2000, size=(1048576, 4), dtype=np.int16)


@pytest.fixture
def radial_root(tmp_path):
    """A RADIal-labeled tree: 3 frames over 2 recordings, adc files for frames 10 and 30 only."""
    root = tmp_path / "RADIal-labeled"
    for d in ("camera", "laser_PCL", "radar_FFT", "radar_PCL", "radar_Freespace", "radar_ADC"):
        (root / d).mkdir(parents=True)
    rows = [[10, 1, 2, 3, 4, 5, 6, 7, 8, 9, 20.0, 3.0, 1.0, 50, "RECORD@A", 0, 0],
            [20, 1, 2, 3, 4, 5, 6, 7, 8, 9, 35.0, -10.0, 2.0, 50, "RECORD@A", 1, 0],
            [30, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, "RECORD@B", 0, 0]]
    pd.DataFrame(rows, columns=COLUMNS).to_csv(root / "labels.csv", index=False)
    rng = np.random.default_rng(0)
    for n in (10, 20, 30):
        mask = np.zeros((512, 900), dtype=np.uint8)
        mask[100:300, 300:600] = 255
        Image.fromarray(mask).save(root / "radar_Freespace" / f"freespace_{n:06d}.png")
        fft = (rng.standard_normal((512, 256, 16)) + 1j * rng.standard_normal((512, 256, 16))) * 3e4
        np.save(root / "radar_FFT" / f"fft_{n:06d}.npy", fft.astype(np.complex64))
    for n in (10, 30):
        np.save(root / "radar_ADC" / f"adc_{n:06d}.npy", _raw(n))
    return root


@pytest.fixture
def parser(radial_root):
    from detection_vis_backend.datasets.dataset import RADIal
    p = RADIal()
    p.parse(1, str(radial_root.parent), radial_root.name, "")
    return p


def _adapter(parser, mode, features=None):
    from detection_vis_backend.datasets.adapters import make_sample_adapter
    cfg = {"dataset": "RADIal", "input_mode": mode, "segmentation_head": True}
    return make_sample_adapter("T_FFTRadNet", parser, features or [FEATURE[mode]], cfg, {})


def _stats(mode):
    from detection_vis_backend.datasets.adapters.tfftradnet_common import load_tfftradnet_statistics
    s = load_tfftradnet_statistics("RADIal", mode)
    return np.array(s["input_mean"])[:, None, None], np.array(s["input_std"])[:, None, None]


def _reference_rd(raw, window=True, shift=False):
    """The reference's Custom_RD from its decoded ADC: DC removal, Hamming windows, range then
    Doppler FFT, real || imag (with the Doppler fftshift only when asked)."""
    from detection_vis_backend.datasets.adapters.radial import radial_complex_adc
    adc = radial_complex_adc(raw).astype(np.complex128)
    w_r = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(512) / 511)
    w_d = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(256) / 255)
    if not window:
        w_r, w_d = np.ones(512), np.ones(256)
    rd = np.fft.fft(np.fft.fft(adc * w_r[:, None, None], 512, axis=0) * w_d[None, :, None], 256, axis=1)
    if shift:
        rd = np.fft.fftshift(rd, axes=1)
    return np.concatenate([rd.real, rd.imag], axis=2).transpose(2, 0, 1)


def test_rd_mode_equals_fftradnet_input(parser):
    from detection_vis_backend.datasets.adapters import make_sample_adapter
    ours = _adapter(parser, "RD")[1]
    theirs = make_sample_adapter("FFTRadNet", parser, ["RD"], {"segmentation_head": True}, {})[1]
    assert set(ours) == {"RD", "encoded_label", "box_label", "seg_label"}
    assert ours["RD"].shape == (32, 512, 256)
    np.testing.assert_allclose(ours["RD"], theirs["RD"], rtol=1e-6)
    for key in ("encoded_label", "box_label", "seg_label"):
        np.testing.assert_array_equal(ours[key], theirs[key])


def test_complex_adc_is_decoded_and_dc_free():
    from detection_vis_backend.datasets.adapters.radial import decode_radial_adc, radial_complex_adc
    raw = _raw(10)
    adc = radial_complex_adc(raw)
    assert adc.shape == (512, 256, 16) and adc.dtype == np.complex64
    decoded = decode_radial_adc(raw)
    cube = decoded[..., :16] + 1j * decoded[..., 16:]
    np.testing.assert_allclose(adc, cube - cube.mean(axis=(0, 1)), rtol=1e-5, atol=1e-2)


def test_rd_shift_as_published_equals_unshifted_rd_from_adc(parser):
    mean, std = _stats("RD_shift_as_published")
    got = _adapter(parser, "RD_shift_as_published")[0]["RD"]
    np.testing.assert_allclose(got, (_reference_rd(_raw(10)) - mean) / std, rtol=1e-4, atol=1e-5)


def test_rd_shift_applies_the_doppler_shift(parser):
    mean, std = _stats("RD_shift")
    got = _adapter(parser, "RD_shift")[0]["RD"]
    np.testing.assert_allclose(got, (_reference_rd(_raw(10), shift=True) - mean) / std, rtol=1e-4, atol=1e-5)


def test_adc_mode_is_dc_free_complex_stacked(parser):
    from detection_vis_backend.datasets.adapters.radial import radial_complex_adc
    sample = _adapter(parser, "ADC")[0]
    assert set(sample) == {"ADC", "encoded_label", "box_label", "seg_label"}
    adc = radial_complex_adc(_raw(10))
    assert sample["ADC"].dtype == np.float32 and sample["ADC"].shape == (32, 512, 256)
    np.testing.assert_array_equal(sample["ADC"], np.concatenate([adc.real, adc.imag], axis=2).transpose(2, 0, 1))
    assert np.abs(sample["ADC"].mean(axis=(1, 2))).max() < 1e-2


@pytest.mark.parametrize("mode", ["RD_shift_as_published", "RD_shift", "ADC"])
def test_adc_modes_serve_only_frames_with_adc(parser, mode):
    adapter = _adapter(parser, mode)
    assert len(adapter) == 2
    assert [adapter.dataset_frame(i) for i in range(2)] == [0, 2]
    assert adapter.has_frame(2) and not adapter.has_frame(1)
    assert adapter.sample_of_frame(2) == 1
    with pytest.raises(ValueError, match="no raw ADC"):
        adapter.sample_of_frame(1)


@pytest.mark.parametrize("mode", ["RD", "ADC"])
def test_adapter_pickles_for_fresh_worker_processes(parser, mode):
    # DataLoader workers started fresh (forkserver) receive the dataset pickled.
    import pickle
    clone = pickle.loads(pickle.dumps(_adapter(parser, mode)))
    assert len(clone) == len(_adapter(parser, mode))
    if mode == "ADC":
        assert clone.sample_of_frame(2) == 1


def test_rd_mode_serves_every_frame(parser):
    adapter = _adapter(parser, "RD")
    assert len(adapter) == 3 and not hasattr(adapter, "sample_of_frame")


def test_feature_mode_mismatch_raises(parser):
    with pytest.raises(ValueError, match=r"'ADC'.*'RD'|'RD'.*'ADC'"):
        _adapter(parser, "RD", features=["ADC"])


def test_rd_no_window_drops_the_windows(parser):
    mean, std = _stats("RD_no_window")
    got = _adapter(parser, "RD_no_window")[0]["RD"]
    np.testing.assert_allclose(got, (_reference_rd(_raw(10), window=False) - mean) / std, rtol=1e-4, atol=1e-5)


def test_missing_statistics_raise_naming_the_script(monkeypatch, tmp_path):
    import json
    from detection_vis_backend.datasets.adapters import tfftradnet_common
    f = tmp_path / "stats.json"
    f.write_text(json.dumps({"RADIal": {"RD_no_window": None}}))
    monkeypatch.setattr(tfftradnet_common, "STATISTICS_FILE", str(f))
    with pytest.raises(ValueError, match="tfftradnet_input_statistics"):
        tfftradnet_common.load_tfftradnet_statistics("RADIal", "RD_no_window")


@pytest.mark.skipif(not os.path.isdir(os.path.join(REAL_RADIAL, "radar_ADC")), reason="RADIal ADC absent")
def test_real_frame_rd_from_adc_matches_stored_radar_fft():
    from detection_vis_backend.datasets.adapters.radial import radial_complex_adc, radial_rd_from_adc
    name = sorted(os.listdir(os.path.join(REAL_RADIAL, "radar_ADC")))[0]
    num = name[len("adc_"):-len(".npy")]
    rd = radial_rd_from_adc(radial_complex_adc(np.load(os.path.join(REAL_RADIAL, "radar_ADC", name))),
                            window=True, shift=False)
    ref = np.load(os.path.join(REAL_RADIAL, "radar_FFT", f"fft_{num}.npy"))
    ref = np.concatenate([ref.real, ref.imag], axis=2)
    assert np.abs(rd - ref).max() / np.abs(ref).max() < 1e-5
