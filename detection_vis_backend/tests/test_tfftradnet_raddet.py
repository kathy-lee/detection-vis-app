"""T-FFTRadNet's RADDet adapter: point labels on a range-angle grid and each input mode
built as the reference repo (jgiroux8/T_FFTRadNet 68bc041) RADDet/dataset/dataset.py builds
it (plus the stored-RAD variant)."""
import os
import pickle

import numpy as np
import pytest

from detection_vis_backend.paths import dataset_path

REAL_RADDET = dataset_path("RADDet_reform_dataset")
FEATURE = {"RAD": "RAD", "RAD_stored": "RAD", "RD": "RD", "RD_no_window": "RD", "ADC": "ADC"}
HAMMING_R = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(256) / 255)
HAMMING_D = 0.54 - 0.46 * np.cos(2 * np.pi * np.arange(64) / 63)


def _adc(seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((256, 64, 4, 2)) + 1j * rng.standard_normal((256, 64, 4, 2))


class FakeRADDet:
    """Just what the adapter reads: annotations, ADC by frame and the stored RAD cube."""

    def __init__(self, n=2):
        self.n = n
        self.adc = {i: _adc(i) for i in range(n)}
        rng = np.random.default_rng(9)
        self.rad = {i: (rng.standard_normal((256, 256, 64)) + 1j * rng.standard_normal((256, 256, 64))).astype(np.complex64)
                    for i in range(n)}

    def __len__(self):
        return self.n

    def annotation(self, i):
        return {"classes": ["car", "person"], "boxes": [np.zeros(6), np.zeros(6)],
                "cart_boxes": [np.array([128.0, 256.0, 10, 10]), np.array([200.0, 300.0, 10, 10])]}

    def get_ADC(self, i):
        return self.adc[i]

    def get_RAD(self, i):
        return self.rad[i]


FakeRADDet.__name__ = "RADDetDataset"   # the adapter registry keys on the parser class name


def _adapter(mode, features=None, parser=None):
    from detection_vis_backend.datasets.adapters import make_sample_adapter
    cfg = {"dataset": "RADDetDataset", "input_mode": mode, "segmentation_head": False}
    return make_sample_adapter("T_FFTRadNet", parser or FakeRADDet(), features or [FEATURE[mode]], cfg, {})


def _stats(mode):
    from detection_vis_backend.datasets.adapters.tfftradnet_common import load_tfftradnet_statistics
    s = load_tfftradnet_statistics("RADDetDataset", mode)
    return np.array(s["input_mean"]), np.array(s["input_std"])


# ---- labels --------------------------------------------------------------------------

def _gt(cart_centres, classes):
    return {"classes": classes, "boxes": [np.zeros(6)] * len(classes),
            "cart_boxes": [np.array([r, c, 10, 10]) for r, c in cart_centres]}


def test_point_labels_follow_reference_geometry():
    from detection_vis_backend.datasets.adapters.raddet import raddet_point_labels
    pts = raddet_point_labels(_gt([(128.0, 256.0), (0.0, 356.0)], ["car", "person"]))
    # (128, 256): y = 50 - 25 = 25 m ahead, x = 0 -> R 25, phi 0. (0, 356): y 50, x 19.53125.
    np.testing.assert_allclose(pts[0], [25.0, 0.0, 2], atol=1e-5)
    np.testing.assert_allclose(pts[1], [np.hypot(19.53125, 50), np.degrees(np.arctan(19.53125 / 50)), 4], rtol=1e-6)
    assert pts.dtype == np.float32


def test_classes_are_the_references_alphabetical_order():
    from detection_vis_backend.datasets.adapters.raddet import RADDET_CLASSES
    assert RADDET_CLASSES == ["bicycle", "bus", "car", "motorcycle", "person", "truck"]


def test_point_labels_stop_after_31_boxes():
    from detection_vis_backend.datasets.adapters.raddet import raddet_point_labels
    assert raddet_point_labels(_gt([(100.0, 256.0)] * 40, ["car"] * 40)).shape == (31, 3)


def test_no_boxes_gives_an_empty_label():
    from detection_vis_backend.datasets.adapters.raddet import encode_raddet_label, raddet_point_labels
    pts = raddet_point_labels(_gt([], []))
    assert pts.shape == (0, 3)
    assert not encode_raddet_label(pts).any()


def test_encoder_marks_a_3x3_patch_with_the_class():
    from detection_vis_backend.datasets.adapters.raddet import encode_raddet_label
    m = encode_raddet_label(np.array([[25.0, 0.0, 2]], dtype=np.float32))
    assert m.shape == (9, 64, 128)
    rows, cols = np.nonzero(m[0])
    assert len(rows) == 9 and np.ptp(rows) == 2 and np.ptp(cols) == 2
    assert (m[3 + 2][rows, cols] == 1).all() and m[3:].sum() == 9


def test_encoder_clips_at_the_first_azimuth_bin():
    from detection_vis_backend.datasets.adapters.raddet import encode_raddet_label
    m = encode_raddet_label(np.array([[20.0, -89.9, 0]], dtype=np.float32))   # angle bin 0
    assert np.nonzero(m[0])[1].tolist() == [0, 1] * 3


def test_encoder_drops_a_point_in_the_last_azimuth_bin_as_the_reference():
    # The reference's encoder has no branch for angle bin 127 (it tests >= 128), so such a
    # point gets no target; kept, as the published weights were trained on it.
    from detection_vis_backend.datasets.adapters.raddet import encode_raddet_label
    assert not encode_raddet_label(np.array([[20.0, 89.9, 0]], dtype=np.float32)).any()


def test_decode_inverts_encode():
    from detection_vis_backend.datasets.adapters.raddet import decode_raddet_map, encode_raddet_label
    m = encode_raddet_label(np.array([[25.3, 7.4, 4]], dtype=np.float32))
    centre = [p for p in decode_raddet_map(m, 0.5)]
    assert len(centre) == 9
    for r, a, conf, cls in centre:
        assert conf == 1 and cls == 4
        assert abs(r - 25.3) < 1e-4 and abs(a - 7.4) < 1e-4


# ---- input modes --------------------------------------------------------------------

def _reference_rad(adc):
    """Inline copy of the reference's RADDet/dataset/dataset.py:109-121, numpy.fft for mkl_fft."""
    adc = adc.reshape(256, 64, 8)
    rf = np.fft.fft(adc * HAMMING_R[:, None, None], 256, axis=0)
    rd = np.fft.fftshift(np.fft.fft(rf * HAMMING_D[None, :, None], 64, axis=1), axes=1)
    rad = np.concatenate([rd, np.zeros((256, 64, 248)).astype("complex64")], axis=2)
    rad = np.fft.fftshift(np.fft.fft(rad, axis=2), axes=2)
    return np.abs(np.transpose(rad, axes=[0, 2, 1]))


def _reference_rd(adc, window=True):
    """Inline copy of the reference's RADDet/dataset/dataset.py:86-100."""
    adc = adc.reshape(256, 64, 8)
    adc = adc - np.mean(adc, axis=(0, 1))
    w_r, w_d = (HAMMING_R, HAMMING_D) if window else (np.ones(256), np.ones(64))
    rf = np.fft.fft(adc * w_r[:, None, None], 256, axis=0)
    rd = np.fft.fftshift(np.fft.fft(rf * w_d[None, :, None], 64, axis=1), axes=1)
    return np.concatenate([rd.real, rd.imag], axis=2)


def test_rad_mode_matches_reference_formula():
    parser = FakeRADDet()
    mean, std = _stats("RAD")
    got = _adapter("RAD", parser=parser)[0]
    assert set(got) == {"RAD", "encoded_label", "class_map", "box_label"}
    assert got["RAD"].shape == (64, 256, 256) and got["RAD"].dtype == np.float32
    np.testing.assert_allclose(got["RAD"], ((_reference_rad(parser.adc[0]) - mean) / std).transpose(2, 0, 1),
                               rtol=1e-4, atol=1e-5)


def test_rd_mode_matches_reference_formula():
    parser = FakeRADDet()
    mean, std = _stats("RD")
    got = _adapter("RD", parser=parser)[0]["RD"]
    assert got.shape == (16, 256, 64)
    np.testing.assert_allclose(got, ((_reference_rd(parser.adc[0]) - mean) / std).transpose(2, 0, 1),
                               rtol=1e-4, atol=1e-5)


def test_raddet_rd_no_window_drops_the_windows():
    from detection_vis_backend.datasets.adapters.raddet import raddet_rd_reference
    adc = _adc(3)
    np.testing.assert_allclose(raddet_rd_reference(adc, window=False),
                               _reference_rd(adc, window=False).transpose(2, 0, 1), rtol=1e-4, atol=1e-4)


def test_adc_mode_is_raw_reshape_stacked():
    parser = FakeRADDet()
    got = _adapter("ADC", parser=parser)[0]["ADC"]
    flat = parser.adc[0].reshape(256, 64, 8)
    assert got.shape == (16, 256, 64) and got.dtype == np.float32
    np.testing.assert_allclose(got, np.concatenate([flat.real, flat.imag], axis=2).transpose(2, 0, 1), rtol=1e-6)


def test_rad_stored_reverses_range_and_angle():
    from detection_vis_backend.datasets.adapters.raddet import raddet_rad_stored
    cube = np.zeros((256, 256, 64), np.complex64)
    cube[10, 30, 5] = 3 - 4j
    out = raddet_rad_stored(cube)
    assert out.shape == (64, 256, 256)
    assert out[5, 245, 225] == 5 and np.count_nonzero(out) == 1


def test_labels_are_served_with_the_input():
    s = _adapter("RD")[1]
    assert s["encoded_label"].shape == (3, 64, 128) and s["class_map"].shape == (6, 64, 128)
    np.testing.assert_allclose(s["box_label"][:, 2], [2, 4])


@pytest.mark.parametrize("mode,features", [("RAD", ["RD"]), ("ADC", ["RAD"]), ("RD", ["ADC"])])
def test_feature_mode_mismatch_raises(mode, features):
    with pytest.raises(ValueError, match="input_mode"):
        _adapter(mode, features=features)


def test_adapter_pickles_for_fresh_worker_processes():
    clone = pickle.loads(pickle.dumps(_adapter("RAD")))
    assert len(clone) == 2


def test_get_adc_names_the_missing_file(tmp_path):
    from detection_vis_backend.datasets.dataset import RADDetDataset
    ds = RADDetDataset()
    ds.root_path = str(tmp_path)
    ds.anno_filenames = [str(tmp_path / "gt" / "000026.pickle")]
    with pytest.raises(FileNotFoundError, match="000026.npy"):
        ds.get_ADC(0)


def test_get_adc_reads_the_frame_by_number(tmp_path):
    from detection_vis_backend.datasets.dataset import RADDetDataset
    (tmp_path / "ADC").mkdir()
    np.save(tmp_path / "ADC" / "000026.npy", _adc(1))
    ds = RADDetDataset()
    ds.root_path = str(tmp_path)
    ds.anno_filenames = [str(tmp_path / "gt" / "000026.pickle")]
    np.testing.assert_array_equal(ds.get_ADC(0), _adc(1))
    assert ds.get_feature("ADC", 0).shape == (256, 64, 4, 2)


@pytest.mark.skipif(not os.path.isdir(os.path.join(REAL_RADDET, "ADC")), reason="RADDet ADC absent")
def test_rad_stored_correlates_with_the_correct_order_cube():
    # The recovered recipe of the stored cubes (docs/raddet-stored-rad-from-adc.md), without
    # its final reversal: what RAD_stored should equal once it reverses range and angle.
    from detection_vis_backend.datasets.adapters.raddet import raddet_rad_stored
    raw = np.load(os.path.join(REAL_RADDET, "ADC", "000026.npy"))
    adc = raw.transpose(0, 1, 3, 2).reshape(256, 64, 8)
    adc = adc - adc.mean(axis=0, keepdims=True)
    x = np.fft.fft(adc * np.hanning(256)[:, None, None], axis=0)
    x = np.fft.fftshift(np.fft.fft(x, axis=1), axes=1)
    x = np.fft.fftshift(np.fft.fft(x, 256, axis=2), axes=2).transpose(0, 2, 1)
    ref = np.abs(x).transpose(2, 0, 1).ravel()
    got = raddet_rad_stored(np.load(os.path.join(REAL_RADDET, "RAD", "000026.npy"))).ravel()
    a, b = got - got.mean(), ref - ref.mean()
    assert a @ b / np.sqrt((a @ a) * (b @ b)) > 0.9999
