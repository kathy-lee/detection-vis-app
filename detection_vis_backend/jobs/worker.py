from rq import Worker

from detection_vis_backend.jobs.queue import QUEUE_NAME, get_redis


def main():
    connection = get_redis()
    worker = Worker([QUEUE_NAME], connection=connection)
    worker.work(with_scheduler=True)


if __name__ == "__main__":
    main()
