import os
import pytest
import numpy as np

from detection_vis_backend.datasets.dataset import (
    RaDICaL, 
    RADIal, RADIalRaw,
    CRUW, 
    CARRADA, 
    RADDetDataset, 
    Astyx, 
    UWCR,
)
from detection_vis_backend.datasets.view_of_delft import ViewOfDelft
from detection_vis_backend.datasets.nuscenes import NuScenes

from data import crud
from data.database import get_session_local
from sqlalchemy.orm import Session
from loguru import logger



@pytest.fixture(scope="function")
def db_session():
    db = get_session_local()
    try:
        yield db
    finally:
        db.close()


@pytest.fixture(autouse=True, scope='session')
def set_env_vars():
    os.environ['DATABASE_URL'] = "postgresql://postgres:postgres@localhost/postgres"


def test_CARRADA_keeps_only_its_own_sequence_annotations(db_session: Session):
    """annotations_frame_oriented.json holds ALL 30 sequences (37.7MB on disk, 331MB as
    Python objects), but an instance only ever reads its own sequence. parse() must keep
    just self.annos = full[self.seq_name], or every cached CARRADA instance pins ~345MB
    RSS and the 'original' splitmode (30 instances alive in one ConcatDataset) needs ~10GB.
    """
    datafile = crud.get_datafile(db_session, 50)
    assert datafile is not None and datafile.parse == "CARRADA"
    inst = CARRADA()
    inst.parse(50, datafile.path, datafile.name, datafile.config)

    # keyed by frame id directly, not by sequence name
    assert datafile.name not in inst.annos, "annos still holds the whole 30-sequence dict"
    assert "000066" in inst.annos, "expected frame keys at the top level"
    # and get_label still works against the sliced shape
    gt = inst.get_label("RA", 66)
    assert isinstance(gt, list)


def test_ViewOfDelft(db_session: Session):
    dataset_inst = ViewOfDelft()
    file_id = 110
    datafile = crud.get_datafile(db_session, file_id)
    assert datafile is not None
    dataset_inst.parse(file_id, datafile.path, datafile.name, datafile.config)
    
    id = 0

    # get image and its label
    image_path = dataset_inst.get_feature("image", id, for_visualize=True)
    gt_label = dataset_inst.get_label("image", id)
    assert os.path.exists(image_path), f"Image file path {image_path} does not exist."
    assert isinstance(gt_label, list), f"Image gt label should be list type."

    # get lidar point cloud and its label
    lidar_pc = dataset_inst.get_feature("lidarPC", id, for_visualize=True)
    gt_label = dataset_inst.get_label("lidarPC", id)
    assert isinstance(lidar_pc, (np.ndarray, np.generic)), "Lidar point cloud data in View-of-Delft is numpy array type."
    assert len(lidar_pc.shape) == 2, "Lidar point cloud data in View-of-Delft is a 2-dimensional array."
    assert lidar_pc.shape[1] == 4, "Lidar point cloud data in View-of-Delft is with N*4 shape."

    # get radar point cloud and its label
    radar_pc = dataset_inst.get_feature("radarPC", id, for_visualize=True)
    gt_label = dataset_inst.get_label("radarPC", id)
    assert isinstance(radar_pc, (np.ndarray, np.generic)), "Radar point cloud data in View-of-Delft is numpy array type."
    assert len(radar_pc.shape) == 2, "Radar point cloud data in View-of-Delft is a 2-dimensional array."
    assert radar_pc.shape[1] == 7, "Radar point cloud data in View-of-Delft is with N*7 shape."


def test_ViewOfDelft_getitem_for_radarpillar(db_session: Session):
    """RadarPillar training support: __getitem__
    yields pillar tensors + radar-frame gt boxes for the configured radar variant."""
    # mirrors RadarPillar_model_config.json (Task 7); inlined so this test does not
    # depend on task execution order
    model_cfg = {
        "point_cloud_range": [0.0, -25.6, -3.0, 51.2, 25.6, 2.0],
        "voxel_size": [0.16, 0.16, 5.0],
        "max_points_per_pillar": 32,
        "max_pillars": 16000,
        "point_features": ["x", "y", "z", "rcs", "v_r", "v_r_comp", "time"],
        "head_stride": 2,
        "dir_offset": 0.78539,
        "anchor_rotations": [0.0, 1.57],
        "anchors": {
            "Car":        {"sizes": [3.9, 1.6, 1.56], "z_bottom": -1.78, "match_threshold": 0.6,  "unmatch_threshold": 0.45},
            "Pedestrian": {"sizes": [0.8, 0.6, 1.73], "z_bottom": -0.6,  "match_threshold": 0.5,  "unmatch_threshold": 0.35},
            "Cyclist":    {"sizes": [1.76, 0.6, 1.73], "z_bottom": -0.6, "match_threshold": 0.5,  "unmatch_threshold": 0.35},
        },
    }
    datafile = crud.get_datafile(db_session, 110)
    assert datafile is not None
    inst = ViewOfDelft()
    inst.parse(110, datafile.path, datafile.name, datafile.config)
    inst._request_scoped = True   # simulate the factory's request copy
    train_cfg_stub = {
        "radar_variant": "radar_5frames",
        "fov_points_only": True,
        "augmentation": {"random_world_flip": True, "random_world_rotation": True,
                         "random_world_scaling": True},
    }
    inst.prepare_for_train(["radarPC"], train_cfg_stub, model_cfg, splittype="val")

    # val (and None) splittype must not augment: repeated reads are identical
    assert inst.transformations is None
    s1, s2 = inst[0], inst[0]
    assert np.array_equal(s1["pillars"], s2["pillars"])

    # a train-split copy carries the augmentor
    inst_train = ViewOfDelft()
    inst_train.parse(110, datafile.path, datafile.name, datafile.config)
    inst_train._request_scoped = True
    inst_train.prepare_for_train(["radarPC"], train_cfg_stub, model_cfg, splittype="train")
    assert inst_train.transformations is not None

    assert len(inst) == inst.frame_sync
    sample = inst[0]
    P, K, D = sample["pillars"].shape
    assert K == model_cfg["max_points_per_pillar"] and D == len(model_cfg["point_features"]) + 6
    assert sample["coords"].shape == (P, 2)
    assert sample["gt_boxes"].ndim == 2 and sample["gt_boxes"].shape[1] == 7
    assert set(np.unique(sample["gt_labels"])).issubset({0, 1, 2})
    # every kept gt center inside the configured range (the spec's integration check)
    x0, y0, z0, x1, y1, z1 = model_cfg["point_cloud_range"]
    found_gt = False
    for i in (0, 50, 100, 200):
        s = inst[i]
        if len(s["gt_boxes"]):
            found_gt = True
            assert (s["gt_boxes"][:, 0] >= x0).all() and (s["gt_boxes"][:, 0] <= x1).all()
            assert (s["gt_boxes"][:, 1] >= y0).all() and (s["gt_boxes"][:, 1] <= y1).all()
    assert found_gt, "expected at least one of the probed frames to contain objects"


@pytest.mark.skip(reason="NuScenes mini dataset not on this machine (/home/kangle/dataset/nuscences_v1.0_mini)")
def test_NuScenes(db_session: Session):
    dataset_inst = NuScenes()
    file_id = 111
    datafile = crud.get_datafile(db_session, file_id)
    assert datafile is not None
    dataset_inst.parse(file_id, datafile.path, datafile.name, datafile.config)

    id = 0

    # get image and its label
    image_path = dataset_inst.get_feature("image", id, for_visualize=True)
    assert os.path.exists(image_path), f"Image file path {image_path} does not exist."
    gt_label = dataset_inst.get_label("image", id)
    assert isinstance(gt_label, list), f"Image gt label should be list type."
    assert len(gt_label[0]) == 17, f"Image gt label should be 17 dimension."

    # get lidar point cloud and its label
    lidar_pc = dataset_inst.get_feature("lidarPC", id, for_visualize=True)
    assert isinstance(lidar_pc, (np.ndarray, np.generic)), "Lidar point cloud data in NuScenes is numpy array type."
    assert len(lidar_pc.shape) == 2, "Lidar point cloud data in NuScenes is a 2-dimensional array."
    assert lidar_pc.shape[1] == 4, "Lidar point cloud data in NuScenes is with N*4 shape."
    gt_label = dataset_inst.get_label("lidarPC", id)
    assert isinstance(gt_label, list), f"Lidar gt label should be list type."
    assert len(gt_label[0]) == 25, f"Lidar gt label should be 25 dimension."

    # get radar point cloud and its label
    radar_pc = dataset_inst.get_feature("radarPC", id, for_visualize=True)
    assert isinstance(radar_pc, (np.ndarray, np.generic)), "Radar point cloud data in NuScenes is numpy array type."
    assert len(radar_pc.shape) == 2, "Radar point cloud data in NuScenes is a 2-dimensional array."
    assert radar_pc.shape[1] == 18, "Radar point cloud data in NuScenes is with N*18 shape."
    gt_label = dataset_inst.get_label("radarPC", id)
    assert isinstance(gt_label, list), f"Radar gt label should be list type."
    assert len(gt_label[0]) == 25, f"Radar gt label should be 25 dimension."