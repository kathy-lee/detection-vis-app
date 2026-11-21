import os

import pytest
from loguru import logger
from fakeredis import FakeStrictRedis
from rq import Queue


def pytest_configure(config):
    # Next to this file, wherever pytest is started from (a relative path made a nested
    # detection_vis_backend/detection_vis_backend/tests/ when run from the backend folder).
    logger.add(os.path.join(os.path.dirname(os.path.abspath(__file__)), "test_logs.log"),
               format="{time} {level} {message}", level="INFO")


@pytest.fixture
def fake_queue(monkeypatch):
    """RQ queue that runs jobs inline on a fake Redis — no worker/Redis/GPU."""
    conn = FakeStrictRedis()
    q = Queue("gpu", connection=conn, is_async=False, default_timeout=-1)
    monkeypatch.setattr("detection_vis_backend.jobs.queue.get_queue", lambda: q)
    monkeypatch.setattr("detection_vis_backend.jobs.queue.get_redis", lambda: conn)
    return q
