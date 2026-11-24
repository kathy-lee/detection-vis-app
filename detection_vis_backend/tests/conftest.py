import os

import pytest
from loguru import logger
from fakeredis import FakeStrictRedis
from rq import Queue

from detection_vis_backend.paths import load_env

# This machine's DATASET_ROOTDIR, MODEL_ROOTDIR and TMP_ROOTDIR, from the repo's .env,
# for every test and the code under test (variables already set in the shell win).
load_env()


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
