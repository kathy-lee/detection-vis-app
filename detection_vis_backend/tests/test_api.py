import os
import json
import pytest

from fastapi.testclient import TestClient
from detection_vis_backend.app import app
from data.models import Directory, File, MLmodel

@pytest.fixture(autouse=True, scope='session')
def set_env_vars():
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


def _discover_readable_model():
    """Pick a real model from the live DB instead of hardcoding an id.

    The old tests pinned model_id=123, which rotted the moment that model was
    deleted (the DB now holds 133+). Walk /models newest-first and return the
    first whose train_info resolves, together with its info payload.
    """
    models = client.get(f"http://{backend_service}:8001/models").json()
    for model in sorted(models, key=lambda m: m["id"], reverse=True):
        response = client.get(f"http://{backend_service}:8001/model/{model['id']}")
        if response.status_code == 200:
            return model["id"], response.json()
    pytest.skip("no model in the DB has readable train_info on disk")


def test_read_model():
    # model_id = 123   # rotted: hardcoded id no longer in the DB
    model_id, model_dict = _discover_readable_model()
    assert isinstance(model_dict, dict), "Result should be a dictionary"
    assert all(key in model_dict for key in ["datafiles", "features", "model_config", "train_config"])


# def test_train_model():
#     sample_config = json.load("sample_for_train_test.json")
#     params = {
#                 "datafiles_chosen": sample_config["datafiles"], 
#                 "features_chosen": sample_config["features"], 
#                 "mlmodel_configs": sample_config["model_config"], 
#                 "train_configs": sample_config["train_config"]
#             }
#     response = client.post(f"http://{backend_service}:8001/train", json=params)
#     assert response.status_code == 200
#     result = response.json()
#     assert isinstance(result, dict) and "model_name" in result.keys()


def test_predict():
    # model_id = 123; file_id = 53   # rotted: hardcoded pair no longer in the DB
    # Derive the file from the model's own training data: /predict looks file_id up in
    # the model's datafiles, so an arbitrary file_id crashes with a mismatched model.
    model_id, model_info = _discover_readable_model()
    datafiles = model_info.get("datafiles") or []
    assert datafiles, f"model {model_id} has no datafiles in train_info"
    file_id = datafiles[0]["id"]

    params = {"checkpoint_id": 0, "sample_id": 0, "file_id": file_id, "split_type": "train"}
    response = client.get(f"http://{backend_service}:8001/predict/{model_id}", params=params)
    assert response.status_code == 200, f"model {model_id} / file {file_id}: {response.text[:200]}"
    prediction = response.json()
    assert isinstance(prediction, dict), "Prediction should be a dict keyed by feature"
    # Detection models return lists of boxes/points; segmentation models (RECORD/MVRECORD
    # on CARRADA) return mask dicts. Both are valid shapes for this endpoint.
    assert all(isinstance(value, (list, dict)) for value in prediction.values())



# def test_retrain_model():
#     sample_config = json.load("sample_for_train_test.json")
#     model_id = sample_config["model_id"]
#     params = {
#                 "datafiles_chosen": sample_config["datafiles"],
#                 "features_chosen": sample_config["features"],
#                 "mlmodel_configs": sample_config["model_config"],
#                 "train_configs": sample_config["train_config"]
#             }
#     response = client.post(f"http://{backend_service}:8001/retrain/{model_id}", json=params)
#     assert response.status_code == 200
#     result = response.json()
#     assert isinstance(result, dict) and "model_name" in result.keys()


# --- job-queue endpoint tests (fast; fake queue, mocked training — no GPU/data/DB) ---
from unittest.mock import MagicMock
from contextlib import contextmanager


@pytest.fixture
def mock_training(monkeypatch):
    """Make run_training/run_retraining return fast without GPU/data/DB."""
    from detection_vis_backend.jobs import tasks
    monkeypatch.setattr(tasks, "_parse_datafiles", lambda datafiles: None)
    monkeypatch.setattr(tasks, "train", lambda *a, **k: "FFTRadNet___TEST")

    @contextmanager
    def _s():
        yield MagicMock()
    monkeypatch.setattr(tasks, "_session", _s)
    monkeypatch.setattr(tasks.crud, "add_model", lambda db, mlmodel: None)
    monkeypatch.setattr(tasks, "_resolve_last_checkpoint", lambda model_id: "/ckpt.pth")


def _sample():
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "sample_for_train_test.json")) as f:
        return json.load(f)


def test_train_returns_job_id_and_job_finishes(fake_queue, mock_training):
    s = _sample()
    resp = client.post("/train", json={
        "datafiles_chosen": s["datafiles"], "features_chosen": s["features"],
        "mlmodel_configs": s["model_config"], "train_configs": s["train_config"],
    })
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert job_id

    jresp = client.get(f"/jobs/{job_id}")
    assert jresp.status_code == 200
    body = jresp.json()
    assert body["status"] == "finished"
    assert body["model_name"] == "FFTRadNet___TEST"


def test_retrain_returns_job_id(fake_queue, mock_training):
    s = _sample()
    model_id = s["model_id"]
    resp = client.post(f"/retrain/{model_id}", json={
        "datafiles_chosen": s["datafiles"], "features_chosen": s["features"],
        "mlmodel_configs": s["model_config"], "train_configs": s["train_config"],
    })
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]
    assert job_id

    body = client.get(f"/jobs/{job_id}").json()
    assert body["status"] == "finished"
    assert body["model_name"] == "FFTRadNet___TEST"


def test_get_jobs_unknown_id_404(fake_queue):
    resp = client.get("/jobs/does-not-exist")
    assert resp.status_code == 404


def test_read_model_missing_returns_404_not_unbound_local(monkeypatch):
    """Regression test: GET /model/{id} 500'd with UnboundLocalError for a nonexistent model.

    read_model's `except ValueError` / `except Exception` handlers only logged, so control fell
    through to `return {"datafiles": parameters[...]}` while the failing try had never bound
    `parameters`. The client got an opaque 500 instead of a 404, and the real cause ("Model not
    found") was buried in the backend log.

    Deliberately does not rely on live DB contents: the stale test_read_model pins model_id=123,
    which no longer exists, so it cannot distinguish "bug fixed" from "model missing".
    """
    import detection_vis_backend.app as app_module

    app_module.app.dependency_overrides[app_module.get_db] = lambda: None
    monkeypatch.setattr(app_module.crud, "get_model", lambda db, id: None)
    try:
        response = client.get("/model/999999")
    finally:
        app_module.app.dependency_overrides.clear()

    assert response.status_code == 404
    assert response.json()["detail"] == "Model not found"


def test_delete_started_job_returns_409(fake_queue, mock_training):
    # is_async=False jobs finish immediately, so they are not "queued" → cannot cancel
    s = _sample()
    job_id = client.post("/train", json={
        "datafiles_chosen": s["datafiles"], "features_chosen": s["features"],
        "mlmodel_configs": s["model_config"], "train_configs": s["train_config"],
    }).json()["job_id"]
    resp = client.delete(f"/jobs/{job_id}")
    assert resp.status_code == 409