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


def test_worker_exposes_main():
    from detection_vis_backend.jobs import worker
    assert callable(worker.main)


from datetime import datetime, timezone


class _FakeRegistryJob:
    def __init__(self, jid, status, ended_at):
        self.id = jid
        self._status = status
        self.ended_at = ended_at
        self.enqueued_at = None
        self.started_at = None
        self.meta = {}
        self.args = [[], [], {}, {}]
        self.func_name = "detection_vis_backend.jobs.tasks.run_training"
        self.exc_info = "Traceback (most recent call last):\nTypeError: boom"

    def get_status(self):
        return self._status

    def return_value(self):
        return {"model_name": f"model-{self.id}"}


def _dt(day, hour):
    return datetime(2026, 7, day, hour, 0, tzinfo=timezone.utc)


def test_recent_jobs_are_ordered_by_end_time_not_by_status(monkeypatch):
    """Regression test: a job that just failed did not appear on the Jobs page.

    list_jobs built "recent" by concatenating registries -- all finished, THEN all failed:
        list(reversed(finished.get_job_ids()))[:10] + list(reversed(failed.get_job_ids()))[:10]
    so ordering was by CATEGORY, not time, despite the comment saying "most recent first".
    A fresh failure always rendered below every past success, and since the UI hides "recent"
    in a collapsed expander, failures sank out of sight as the finished list grew.

    Pins the invariant: the newest job comes first, whatever its status.
    """
    import detection_vis_backend.app as app

    # the failure is the NEWEST job, but finished jobs outnumber it -- the exact shape of the bug
    jobs = {
        "fail-new": _FakeRegistryJob("fail-new", "failed", _dt(15, 7)),
        "fin-old": _FakeRegistryJob("fin-old", "finished", _dt(13, 7)),
        "fin-mid": _FakeRegistryJob("fin-mid", "finished", _dt(14, 18)),
        "fin-newer": _FakeRegistryJob("fin-newer", "finished", _dt(15, 1)),
    }

    monkeypatch.setattr(app.job_queue, "get_redis", lambda: object())
    monkeypatch.setattr(app, "Queue", lambda *a, **k: type("Q", (), {"job_ids": []})())
    monkeypatch.setattr(app, "StartedJobRegistry", lambda *a, **k: type("R", (), {"get_job_ids": lambda self: []})())
    monkeypatch.setattr(app, "FinishedJobRegistry",
                        lambda *a, **k: type("R", (), {"get_job_ids": lambda self: ["fin-old", "fin-mid", "fin-newer"]})())
    monkeypatch.setattr(app, "FailedJobRegistry",
                        lambda *a, **k: type("R", (), {"get_job_ids": lambda self: ["fail-new"]})())
    monkeypatch.setattr(app.Job, "fetch", staticmethod(lambda jid, connection=None: jobs[jid]))

    recent = app.list_jobs()["recent"]

    assert [j["job_id"] for j in recent] == ["fail-new", "fin-newer", "fin-mid", "fin-old"]
    # the thing the user actually needs: the failure is visible at the top, not buried
    assert recent[0]["status"] == "failed"
    assert recent[0]["error"] == "TypeError: boom"
