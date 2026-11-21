"""scripts/extract_radial_adc.py stores each labelled RADIal frame's raw ADC streams, and
refuses a frame whose streams do not reproduce the ready-to-use radar_FFT."""
import importlib
import os
import sys

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@pytest.fixture
def script(monkeypatch):
    monkeypatch.syspath_prepend(os.path.join(REPO_ROOT, "scripts"))
    sys.modules.pop("extract_radial_adc", None)
    yield importlib.import_module("extract_radial_adc")
    sys.modules.pop("extract_radial_adc", None)


def _streams(seed=0):
    """Four int16 chip streams of the RADIal frame size, I/Q interleaved."""
    rng = np.random.default_rng(seed)
    return [rng.integers(-2000, 2000, size=512 * 256 * 4 * 2, dtype=np.int16) for _ in range(4)]


def test_decode_orders_chips_3_0_1_2_and_interleaves_iq(script):
    chans = [np.zeros(512 * 256 * 4 * 2, dtype=np.int16) for _ in range(4)]
    for c in range(4):
        chans[c][0] = 10 * (c + 1)      # I of sample 0, chirp 0, rx 0
        chans[c][1] = -(c + 1)          # Q of the same sample
    cube = script.decode(np.stack(chans, axis=1))
    assert cube.shape == (512, 256, 16)
    assert [cube[0, 0, 4 * k] for k in range(4)] == [40 - 4j, 10 - 1j, 20 - 2j, 30 - 3j]


def test_rd_spectrum_of_stored_streams_reproduces_the_reference(script):
    raw = np.stack(_streams(), axis=1)
    ref = script.rd_spectrum(script.decode(raw)).astype(np.complex64)
    assert script.relative_error(raw, ref) < script.MAX_REL_ERR


def test_wrong_frame_does_not_match_the_reference(script):
    ref = script.rd_spectrum(script.decode(np.stack(_streams(0), axis=1))).astype(np.complex64)
    assert script.relative_error(np.stack(_streams(1), axis=1), ref) > script.MAX_REL_ERR


def test_labelled_frames_lists_each_sample_once_for_its_recording(script):
    labels = pd.DataFrame({
        "numSample": [18, 18, 37, 90],
        "dataset": ["RECORD@a", "RECORD@a", "RECORD@a", "RECORD@b"],
        "index": [26, 26, 51, 3],
    })
    assert script.labelled_frames(labels, "RECORD@a") == [(18, 26), (37, 51)]
    assert script.labelled_frames(labels, "RECORD@c") == []
