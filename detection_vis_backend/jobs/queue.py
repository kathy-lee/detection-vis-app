import os

from redis import Redis
from rq import Queue

QUEUE_NAME = "gpu"


def get_redis():
    return Redis.from_url(os.getenv("REDIS_URL", "redis://localhost:6379/0"))


def get_queue():
    return Queue(QUEUE_NAME, connection=get_redis(), default_timeout=-1)
