"""CRUW as a parser and its sample adapters, exercised on synthetic sequences -- no DB."""
import json
import os
import pickle

import numpy as np
import pytest

SENSOR_CFG = "/home/kangle/dataset/CRUW/sensor_config_rod2021.json"
needs_sensor_cfg = pytest.mark.skipif(not os.path.exists(SENSOR_CFG),
                                      reason=f"CRUW sensor config not at {SENSOR_CFG}")

N_FRAME = 20
N_CHIRPS = 4


def _write_sequence(tmp_path, with_anno=True):
    """A pickle shaped like CRUW.parse writes, plus the .npy chirps it points to."""
    rng = np.random.default_rng(0)
    radar_dir = tmp_path / "RADAR_RA_H"
    radar_dir.mkdir()
    radar_paths, image_paths = [], []
    for f in range(N_FRAME):
        chirps = []
        for c in range(N_CHIRPS):
            p = radar_dir / f"{f:06d}_{c:04d}.npy"
            np.save(p, rng.standard_normal((128, 128, 2)).astype(np.float32))
            chirps.append(str(p))
        radar_paths.append(chirps)
        image_paths.append(str(tmp_path / f"{f:010d}.jpg"))
    anno = None
    if with_anno:
        anno = {"metadata": [{"frame_id": f, "rad_h": {"obj_info": {"categories": ["car"] if f % 3 == 0 else [],
                                                                     "center_ids": [[10 + f, 40]] if f % 3 == 0 else []}}}
                             for f in range(N_FRAME)],
                "confmaps": rng.random((N_FRAME, 4, 128, 128)).astype(np.float32)}
    pkl = tmp_path / "seq.pkl"
    with open(pkl, "wb") as fh:
        pickle.dump({"data_root": str(tmp_path), "seq_name": "seq", "n_frame": N_FRAME,
                     "image_paths": image_paths, "radar_paths": radar_paths, "anno": anno}, fh)
    return pkl


def _parser(tmp_path, with_anno=True):
    from detection_vis_backend.datasets.dataset import CRUW
    pkl = _write_sequence(tmp_path, with_anno)
    ds = CRUW()
    ds.pkl_path = str(pkl)
    ds.root_path = str(tmp_path)
    ds.seq_name = "seq"
    ds.feature_path = tmp_path
    with open(SENSOR_CFG) as fh:
        ds.sensor_cfg = json.load(fh)
    ds._request_scoped = True
    return ds


@pytest.fixture
def cruw_parser(tmp_path):
    return _parser(tmp_path)


@pytest.fixture
def cruw_parser_no_anno(tmp_path):
    return _parser(tmp_path, with_anno=False)


@needs_sensor_cfg
def test_parser_len_is_frames_and_views_load_without_prepare(cruw_parser):
    assert len(cruw_parser) == N_FRAME
    assert cruw_parser.annotation(3) is cruw_parser.obj_infos[3]
    assert cruw_parser.confmap(3).shape == (3, 128, 128)
    assert cruw_parser.radar_chirp(3, 1).shape == (128, 128, 2)


@needs_sensor_cfg
def test_parser_without_annotations_serves_frames_only(cruw_parser_no_anno):
    assert len(cruw_parser_no_anno) == N_FRAME
    assert cruw_parser_no_anno.confmaps == [] and cruw_parser_no_anno.obj_infos == []


def expected_dtype(parser):
    # the sidecar keeps the pickle's dtype (float64 on disk today): identical bytes, so the
    # goldens captured before the refactor still hold
    with open(parser.pkl_path, "rb") as fh:
        return pickle.load(fh)["anno"]["confmaps"].dtype


@needs_sensor_cfg
def test_sidecar_is_rebuilt_when_the_pickle_is_newer(cruw_parser, tmp_path):
    """Regenerating a sequence (delete the pickle, parse again) must not pair the new
    annotations with the old confidence maps left in the sidecar."""
    cruw_parser._load_details()
    sidecar = tmp_path / "seq_confmaps.npy"
    old_first = np.asarray(cruw_parser.confmaps[0]).copy()
    with open(cruw_parser.pkl_path, "rb") as fh:
        details = pickle.load(fh)
    details["anno"]["confmaps"] = details["anno"]["confmaps"] + 1.0
    import time
    time.sleep(0.01)
    with open(cruw_parser.pkl_path, "wb") as fh:
        pickle.dump(details, fh)
    os.utime(cruw_parser.pkl_path, None)
    from detection_vis_backend.datasets.dataset import CRUW
    fresh = CRUW()
    for attr in ("pkl_path", "root_path", "seq_name", "feature_path", "sensor_cfg", "_request_scoped"):
        setattr(fresh, attr, getattr(cruw_parser, attr))
    fresh._load_details()
    assert sidecar.stat().st_mtime_ns >= os.stat(cruw_parser.pkl_path).st_mtime_ns
    assert np.allclose(np.asarray(fresh.confmaps[0]), old_first + 1.0)


@needs_sensor_cfg
def test_parser_memory_maps_the_confidence_maps(cruw_parser, tmp_path):
    """The pre-rendered maps are float64 and ~0.5 GB per sequence; held in RAM by the
    trainer and every loader worker they fill the host (20 GB for ROD2021). Serve them
    from a .npy beside the pickle through a memory map instead: same bytes, one shared
    copy, paged in on demand."""
    cruw_parser._load_details()
    sidecar = tmp_path / "seq_confmaps.npy"
    assert sidecar.exists()
    assert isinstance(cruw_parser.confmaps, np.memmap) and cruw_parser.confmaps.dtype == expected_dtype(cruw_parser)
    with open(cruw_parser.pkl_path, "rb") as fh:
        expected = pickle.load(fh)["anno"]["confmaps"]
    assert np.array_equal(np.asarray(cruw_parser.confmaps), expected)
    assert np.array_equal(cruw_parser.confmap(7), expected[7][:3])
    # a second instance reuses the sidecar rather than rewriting it
    before = sidecar.stat().st_mtime_ns
    from detection_vis_backend.datasets.dataset import CRUW
    other = CRUW()
    for attr in ("pkl_path", "root_path", "seq_name", "feature_path", "sensor_cfg", "_request_scoped"):
        setattr(other, attr, getattr(cruw_parser, attr))
    other._load_details()
    assert sidecar.stat().st_mtime_ns == before and isinstance(other.confmaps, np.memmap)


# ---------------------------------------------------------------------------------------
# Adapters on an in-memory parser (the registry keys on the class NAME, so the fake
# shares it).
# ---------------------------------------------------------------------------------------
import random

import torch

from detection_vis_backend.datasets.adapters import make_sample_adapter, _REGISTRY, MIGRATED

RODNET_VARIANTS = ["RODNet_CDC", "RODNet_CDCv2", "RODNet_HG", "RODNet_HGv2", "RODNet_HGwI", "RODNet_HGwIv2"]
TRAIN_CFG = {"win_size": 4, "train_step": 1, "train_stride": 2, "test_step": 1, "test_stride": 3}


class FakeCRUW:
    def __init__(self, n_frame=20, anno=True, chirps=None, tmp_path=None):
        self.n_frame = n_frame
        self.chirp_ids = [0, 64, 128, 192]
        self.n_class = 3
        self.seq_name = "fake"
        self.feature_path = tmp_path
        rng = np.random.default_rng(1)
        # chirps: frame -> chirp -> (128, 128, 2); chirp c of frame f is filled with f + c/10
        self._chirps = chirps or {(f, c): np.full((128, 128, 2), f + c / 10, dtype=np.float32)
                                  for f in range(n_frame) for c in range(4)}
        self.radar_paths = [[f"/nowhere/{f:06d}_{c:04d}.npy" for c in range(4)] for f in range(n_frame)]
        self.image_paths = [f"/img/{f}.jpg" for f in range(n_frame)]
        self.obj_infos = [{"frame_id": f} for f in range(n_frame)] if anno else []
        self.confmaps = rng.random((n_frame, 4, 128, 128)).astype(np.float32) if anno else []

    def __len__(self):
        return self.n_frame

    def annotation(self, i):
        return self.obj_infos[i]

    def confmap(self, i):
        return self.confmaps[i][:self.n_class]

    def radar_chirp(self, i, c):
        if (i, c) not in self._chirps:
            raise FileNotFoundError(self.radar_paths[i][c])
        return self._chirps[(i, c)]


FakeCRUW.__name__ = "CRUW"


def _adapter(model, split="train", model_cfg=None, train_cfg=None, **kw):
    return make_sample_adapter(model, FakeCRUW(**kw), ["RA"], model_cfg or {"class": model},
                               train_cfg or TRAIN_CFG, split)


def test_registry_fans_out_the_eleven_cruw_models():
    for m in RODNET_VARIANTS + ["RadarFormer_hrformer2d", "RECORD", "RECORDNoLstm", "RECORDNoLstmMulti", "T_RODNet"]:
        assert (m, "CRUW") in _REGISTRY, m
    assert MIGRATED == frozenset({"RADDetDataset", "CRUW"})


@pytest.mark.parametrize("n_frame,split,expected", [(20, "train", 9), (20, "test", 6), (4, "train", 1)])
def test_window_count_matches_the_old_arithmetic(n_frame, split, expected):
    assert len(_adapter("RODNet_CDC", split, n_frame=n_frame)) == expected


def test_window_count_is_zero_for_a_sequence_shorter_than_a_window():
    assert len(_adapter("RODNet_CDC", n_frame=3)) == 0


def test_rodnet_single_chirp_sample_layout():
    s = _adapter("RODNet_CDC", "val")[1]
    assert set(s) == {"status", "seq_name", "image_paths", "start_frame", "end_frame", "RA", "obj_infos", "gt_mask"}
    assert isinstance(s["RA"], np.ndarray) and s["RA"].dtype == np.float32 and s["RA"].shape == (2, 4, 128, 128)
    assert s["gt_mask"].shape == (3, 4, 128, 128)
    assert (s["start_frame"], s["end_frame"]) == (2, 5) and s["image_paths"] == [f"/img/{f}.jpg" for f in (2, 3, 4, 5)]
    assert np.allclose(np.floor(s["RA"][0, :, 0, 0]), [2, 3, 4, 5])      # frames of the window


def test_rodnet_mnet_sample_layout():
    s = _adapter("RODNet_CDCv2", "val", model_cfg={"class": "RODNet_CDCv2", "mnet_cfg": [4, 32]})[0]
    assert s["RA"].shape == (2, 4, 4, 128, 128)
    assert np.allclose(s["RA"][0, 0, :, 0, 0], [0, 0.1, 0.2, 0.3], atol=1e-6)   # chirp axis


def test_record_sample_layouts():
    s = _adapter("RECORD", "val")[0]
    assert isinstance(s["RA"], torch.Tensor) and s["RA"].shape == (8, 4, 128, 128) and s["gt_mask"].shape == (3, 128, 128)
    multi = _adapter("RECORDNoLstmMulti", "val")[0]
    assert multi["RA"].shape == (32, 128, 128)
    single = _adapter("RECORDNoLstm", "val", train_cfg={**TRAIN_CFG, "win_size": 1})[0]
    assert single["RA"].shape == (8, 128, 128)


def test_rodnet_adapter_draws_a_chirp_on_every_split():
    a = _adapter("RODNet_CDC", "val")
    random.seed(0); first = a[0]["RA"][0, 0, 0, 0]
    seen = {first}
    for seed in range(1, 6):
        random.seed(seed); seen.add(a[0]["RA"][0, 0, 0, 0])
    assert len(seen) > 1


TRODNET_CFG = {**TRAIN_CFG, "win_size": 16}


def test_trodnet_adapter_uses_chirp_zero_off_train():
    for split in ("val", "test"):
        a = _adapter("T_RODNet", split, train_cfg=TRODNET_CFG)
        for seed in range(5):
            random.seed(seed)
            assert a[0]["RA"][0, 0, 0, 0] == 0.0, split          # chirp 0 of frame 0
    a = _adapter("T_RODNet", "train", train_cfg=TRODNET_CFG)
    seen = set()
    for seed in range(8):
        random.seed(seed); seen.add(float(a[0]["RA"][0, 0, 0, 0]))
    assert len(seen) > 1


def test_trodnet_adapter_asserts_its_fixed_input_shape():
    s = _adapter("T_RODNet", "train", train_cfg=TRODNET_CFG)[0]
    assert s["RA"].shape == (2, 16, 128, 128)
    with pytest.raises(AssertionError):
        _adapter("T_RODNet", "train", train_cfg={**TRAIN_CFG, "win_size": 4})[0]


def test_adapter_serves_windows_without_ground_truth():
    s = _adapter("RODNet_CDC", "val", anno=False)[0]
    assert "RA" in s and "gt_mask" not in s and "obj_infos" not in s
    r = _adapter("RECORD", "val", anno=False)[0]
    assert "RA" in r and "gt_mask" not in r


def test_bad_npy_sets_status_false_and_logs(tmp_path):
    chirps = {(f, c): np.zeros((128, 128, 2), np.float32) for f in range(20) for c in range(4)}
    for c in range(4):
        del chirps[(2, c)]
    a = make_sample_adapter("RODNet_CDC", FakeCRUW(chirps=chirps, tmp_path=tmp_path), ["RA"],
                            {"class": "RODNet_CDC"}, TRAIN_CFG, "val")
    random.seed(0)
    s = a[0]
    assert s["status"] is False and "RA" not in s
    assert any(p.name.startswith("loadnpyfail-") for p in tmp_path.iterdir())
