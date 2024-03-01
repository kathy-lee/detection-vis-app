import os
import json
import pytest

from fastapi.testclient import TestClient
from detection_vis_backend.app import app
from data.models import Directory, File, MLmodel

@pytest.fixture(autouse=True, scope='session')
def set_env_vars():
    print("Setting environment variables")
    os.environ['DATABASE_URL'] = "postgresql://postgres:postgres@localhost/postgres"
    os.environ['BACKEND_SERVICE'] = "localhost"
    os.environ['TMP_ROOTDIR'] = "/home/kangle/dataset/tmp"
    os.environ['MODEL_ROOTDIR'] = "/home/kangle/dataset/trained_models"


client = TestClient(app)
backend_service = os.getenv('BACKEND_SERVICE', 'localhost')

def test_environment_variable():
    assert os.getenv("DATABASE_URL") is not None, "DATABASE_URL is not set"

def test_read_datasets():
    response = client.get(f"http://{backend_service}:8001/datasets")
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, list), "Result should be a list"
    assert all(isinstance(item, dict) and 'id' in item and 'name' in item for item in result), "Each item should be a dictionary representation of Directory model"


def test_read_datasetfiles():
    dataset_id = 0
    response = client.get(f"http://{backend_service}:8001/dataset/{dataset_id}")
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, list), "Result should be a list"
    assert all(isinstance(item, dict) and 'id' in item and 'name' in item for item in result), "Each item should be a dictionary representation of File model"


def test_parse_data():
    file_id = 6
    response = client.get(f"http://{backend_service}:8001/parse/{file_id}")
    assert response.status_code == 204


def test_check_sync():
    file_id = 6
    response = client.get(f"http://{backend_service}:8001/parse/{file_id}")
    assert response.status_code == 204
    response = client.get(f"http://{backend_service}:8001/sync_check/{file_id}")
    assert response.status_code == 200
    sync_frames = response.json()
    assert sync_frames == 189


def test_get_feature_size():
    file_id = 6
    feature_name = "ADC"
    response = client.get(f"http://{backend_service}:8001/parse/{file_id}")
    assert response.status_code == 204
    response = client.get(f"http://{backend_service}:8001/feature/{file_id}/{feature_name}/size")
    assert response.status_code == 200
    feature_size = response.json()
    assert feature_size == 189


def test_get_feature():
    file_id = 6
    feature_name = "RD"
    id = 0
    response = client.get(f"http://{backend_service}:8001/parse/{file_id}")
    assert response.status_code == 204
    response = client.get(f"http://{backend_service}:8001/feature/{file_id}/{feature_name}/{id}")
    assert response.status_code == 200
    feature_dict = response.json()
    assert "serialized_feature" in feature_dict.keys()
    assert "gt_label" in feature_dict.keys()
    if feature_name in ('image', 'depth_image'):
        assert isinstance(feature_dict["serialized_feature"], str), "Result should be a string"
    else:
        assert isinstance(feature_dict["serialized_feature"], list), "Result should be a list"
    assert isinstance(feature_dict["gt_label"], list) 


def test_get_feature_video():
    file_id = 5
    features = ["image", "radarPC", "RA", "RD"]
    param = {"feature_list": features}
    response = client.get(f"http://{backend_service}:8001/parse/{file_id}")
    assert response.status_code == 204
    response = client.get(f"http://{backend_service}:8001/video/{file_id}", params=param)
    assert response.status_code == 200
    video_path = response.json()
    assert os.path.isfile(video_path), f"Given path {video_path} is not a file"
    assert os.path.getsize(video_path) > 0, f"File {video_path} is empty"
    assert os.path.basename(video_path) == '_'.join(features) + '.mp4', f"File name doesn't follow correct pattern"


def test_read_models():
    response = client.get(f"http://{backend_service}:8001/models")
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, list), "Result should be a list"
    assert all(isinstance(item, dict) and 'id' in item and 'name' in item for item in result), "Each item should be a dictionary representation of MLmodel model"


def test_read_model():
    model_id = 123
    response = client.get(f"http://{backend_service}:8001/model/{model_id}")
    assert response.status_code == 200
    model_dict = response.json()
    assert isinstance(model_dict, dict), "Result should be a dictionary"          
    assert all(key in model_dict for key in ["datafiles", "features", "model_config", "train_config"])


def test_train_model():
    sample_config = json.load("sample_for_train_test.json")
    params = {
                "datafiles_chosen": sample_config["datafiles"], 
                "features_chosen": sample_config["features"], 
                "mlmodel_configs": sample_config["model_config"], 
                "train_configs": sample_config["train_config"]
            }
    response = client.post(f"http://{backend_service}:8001/train", json=params)
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, dict) and "model_name" in result.keys()


def test_predict():
    model_id = 123
    checkpoint_id = 0
    file_id = 53
    split_type = "train"
    sample_id = 0
    params = {"checkpoint_id": checkpoint_id, "sample_id": sample_id, "file_id": file_id, "split_type": split_type}
    response = client.get(f"http://{backend_service}:8001/predict/{model_id}", params=params)  
    assert response.status_code == 200
    prediction = response.json()
    assert all(isinstance(value, list) for value in prediction.values()), "Prediction objects should be list type."



def test_retrain_model():
    sample_config = json.load("sample_for_train_test.json")
    model_id = sample_config["model_id"]
    params = {
                "datafiles_chosen": sample_config["datafiles"], 
                "features_chosen": sample_config["features"], 
                "mlmodel_configs": sample_config["model_config"], 
                "train_configs": sample_config["train_config"]
            }
    response = client.post(f"http://{backend_service}:8001/retrain/{model_id}", json=params)
    assert response.status_code == 200
    result = response.json()
    assert isinstance(result, dict) and "model_name" in result.keys()