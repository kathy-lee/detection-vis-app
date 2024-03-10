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
    gt_label = dataset_inst.get_label("lidarPC", id)
    assert isinstance(lidar_pc, (np.ndarray, np.generic)), "Lidar point cloud data in NuScenes is numpy array type."
    assert len(lidar_pc.shape) == 2, "Lidar point cloud data in NuScenes is a 2-dimensional array."
    assert lidar_pc.shape[1] == 4, "Lidar point cloud data in NuScenes is with N*4 shape."

    # get radar point cloud and its label
    radar_pc = dataset_inst.get_feature("radarPC", id, for_visualize=True)
    gt_label = dataset_inst.get_label("radarPC", id)
    assert isinstance(radar_pc, (np.ndarray, np.generic)), "Radar point cloud data in NuScenes is numpy array type."
    assert len(radar_pc.shape) == 2, "Radar point cloud data in NuScenes is a 2-dimensional array."
    assert radar_pc.shape[1] == 18, "Radar point cloud data in NuScenes is with N*18 shape."