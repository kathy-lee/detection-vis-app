import os
import pytest

from detection_vis_backend.jobs import queue as job_queue


def test_get_queue_uses_gpu_name_and_no_timeout(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    q = job_queue.get_queue()
    assert q.name == "gpu"
    assert q._default_timeout == -1


from contextlib import contextmanager
from unittest.mock import MagicMock

from detection_vis_backend.jobs import tasks


class _FakeJob:
    def __init__(self):
        self.meta = {}

    def save_meta(self):
        pass


@contextmanager
def _dummy_session():
    yield MagicMock()


def test_run_training_writes_progress_and_adds_model(monkeypatch):
    fake_job = _FakeJob()
    monkeypatch.setattr(tasks, "get_current_job", lambda: fake_job)
    monkeypatch.setattr(tasks, "_parse_datafiles", lambda datafiles: None)
    monkeypatch.setattr(tasks, "_session", _dummy_session)

    def fake_train(datafiles, features, model_config, train_config, pretrained=None, progress_cb=None):
        progress_cb(1, 2, 0.5, "train")
        progress_cb(2, 2, 0.3, "train")
        return "FFTRadNet___X"
    monkeypatch.setattr(tasks, "train", fake_train)

    added = {}
    monkeypatch.setattr(tasks.crud, "add_model",
                        lambda db, mlmodel: added.update(name=mlmodel.name, parent_id=mlmodel.parent_id))

    result = tasks.run_training([{"id": 1, "parse": "CRUW"}], ["RA"], {"class": "FFTRadNet"}, {"num_epochs": 2})

    assert result == {"model_name": "FFTRadNet___X"}
    assert fake_job.meta["progress"] == {"epoch": 2, "total_epochs": 2, "loss": 0.3, "phase": "train"}
    assert added == {"name": "FFTRadNet___X", "parent_id": None}


def test_run_training_failure_skips_add_model(monkeypatch):
    monkeypatch.setattr(tasks, "get_current_job", lambda: None)
    monkeypatch.setattr(tasks, "_parse_datafiles", lambda datafiles: None)
    monkeypatch.setattr(tasks, "_session", _dummy_session)

    def boom(*a, **k):
        raise RuntimeError("train failed")
    monkeypatch.setattr(tasks, "train", boom)

    called = {"add": False}
    monkeypatch.setattr(tasks.crud, "add_model", lambda db, mlmodel: called.__setitem__("add", True))

    with pytest.raises(RuntimeError):
        tasks.run_training([{"id": 1, "parse": "CRUW"}], ["RA"], {"class": "FFTRadNet"}, {"num_epochs": 1})

    assert called["add"] is False


import json


def test_resolve_last_checkpoint_matches_zero_padded(monkeypatch, tmp_path):
    model_dir = tmp_path / "FFTRadNet___Y"
    model_dir.mkdir()
    (model_dir / "train_info.txt").write_text(json.dumps({"train_config": {"num_epochs": 10}}))
    (model_dir / "FFTRadNet_epoch09_loss_0.1_AP_0.5_AR_0.5_IOU_0.5.pth").write_text("x")
    monkeypatch.setenv("MODEL_ROOTDIR", str(tmp_path))

    fake_model = type("M", (), {"name": "FFTRadNet___Y"})()
    monkeypatch.setattr(tasks, "_session", _dummy_session)
    monkeypatch.setattr(tasks.crud, "get_model", lambda db, mid: fake_model)

    ckpt = tasks._resolve_last_checkpoint(7)
    assert ckpt.endswith("FFTRadNet_epoch09_loss_0.1_AP_0.5_AR_0.5_IOU_0.5.pth")


def test_run_retraining_sets_parent_id(monkeypatch):
    fake_job = _FakeJob()
    monkeypatch.setattr(tasks, "get_current_job", lambda: fake_job)
    monkeypatch.setattr(tasks, "_parse_datafiles", lambda datafiles: None)
    monkeypatch.setattr(tasks, "_session", _dummy_session)
    monkeypatch.setattr(tasks, "_resolve_last_checkpoint", lambda model_id: "/ckpt.pth")

    seen = {}
    def fake_train(datafiles, features, model_config, train_config, pretrained=None, progress_cb=None):
        seen["pretrained"] = pretrained
        return "FFTRadNet___Z"
    monkeypatch.setattr(tasks, "train", fake_train)

    added = {}
    monkeypatch.setattr(tasks.crud, "add_model",
                        lambda db, mlmodel: added.update(name=mlmodel.name, parent_id=mlmodel.parent_id))

    result = tasks.run_retraining(7, [{"id": 1, "parse": "CRUW"}], ["RA"], {"class": "FFTRadNet"}, {"num_epochs": 2})

    assert result == {"model_name": "FFTRadNet___Z"}
    assert seen["pretrained"] == "/ckpt.pth"
    assert added == {"name": "FFTRadNet___Z", "parent_id": 7}
