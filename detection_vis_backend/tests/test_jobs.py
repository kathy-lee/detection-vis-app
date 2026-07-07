import os
import pytest

from detection_vis_backend.jobs import queue as job_queue


def test_get_queue_uses_gpu_name_and_no_timeout(monkeypatch):
    monkeypatch.setenv("REDIS_URL", "redis://localhost:6379/0")
    q = job_queue.get_queue()
    assert q.name == "gpu"
    assert q._default_timeout == -1
