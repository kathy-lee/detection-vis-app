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
