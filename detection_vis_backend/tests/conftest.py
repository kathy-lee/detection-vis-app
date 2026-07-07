import pytest
from loguru import logger
from fakeredis import FakeStrictRedis
from rq import Queue


def pytest_configure(config):
    logger.add("./detection_vis_backend/tests/test_logs.log",
               format="{time} {level} {message}", level="INFO")


@pytest.fixture
def fake_queue(monkeypatch):
    """RQ queue that runs jobs inline on a fake Redis — no worker/Redis/GPU."""
    conn = FakeStrictRedis()
    q = Queue("gpu", connection=conn, is_async=False, default_timeout=-1)
    monkeypatch.setattr("detection_vis_backend.jobs.queue.get_queue", lambda: q)
    monkeypatch.setattr("detection_vis_backend.jobs.queue.get_redis", lambda: conn)
    return q
