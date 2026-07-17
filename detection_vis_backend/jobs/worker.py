from rq import Worker
from rq.registry import StartedJobRegistry
from rq.utils import current_timestamp

from detection_vis_backend.jobs.queue import QUEUE_NAME, get_redis


def _fail_orphaned_jobs(connection):
    """Reconcile phantom 'running' jobs left by a killed session.

    The worker lives alongside the backend (run.sh) and there is only ever one
    worker, so a freshly started worker means nothing can legitimately be
    running yet. Any job still in the StartedJobRegistry is therefore a leftover
    from a session that died abruptly (terminal/window closed -> SIGHUP, host
    reboot, or a crash) without RQ getting to fail it. Force-expire the whole
    registry so those jobs move to the FailedJobRegistry and the Jobs page shows
    them as failed instead of a phantom 'running'.

    NOTE: safe to fail *all* started jobs only because of the single-worker
    setup. If multiple workers are ever run concurrently, this must instead
    check each job against the set of currently-alive workers.
    """
    reg = StartedJobRegistry(QUEUE_NAME, connection=connection)
    # A timestamp far in the future expires every entry, regardless of heartbeat.
    reg.cleanup(timestamp=current_timestamp() + 10 ** 9)


def main():
    connection = get_redis()
    _fail_orphaned_jobs(connection)
    worker = Worker([QUEUE_NAME], connection=connection)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
