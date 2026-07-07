from contextlib import contextmanager

from rq import get_current_job

from data import crud, schemas
from data.database import get_session_local
from detection_vis_backend.datasets.dataset import DatasetFactory
from detection_vis_backend.train.train import train


@contextmanager
def _session():
    db = get_session_local()
    try:
        yield db
    finally:
        db.close()


def _parse_datafiles(datafiles):
    """Parse each chosen datafile so train()'s dataloaders can read features.
    Folded in from the UI's former pre-train parse loop."""
    factory = DatasetFactory()
    for f in datafiles:
        inst = factory.get_instance(f["parse"], f["id"])
        inst.parse(f["id"], f["path"], f["name"], f["config"])


def _make_cb(job):
    def cb(epoch, total_epochs, loss, phase):
        if job is not None:
            job.meta["progress"] = {
                "epoch": epoch, "total_epochs": total_epochs,
                "loss": loss, "phase": phase,
            }
            job.save_meta()
    return cb


def run_training(datafiles, features, model_config, train_config):
    job = get_current_job()
    _parse_datafiles(datafiles)
    exp_name = train(datafiles, features, model_config, train_config,
                     progress_cb=_make_cb(job))
    with _session() as db:
        crud.add_model(db, schemas.MLModelCreate(name=exp_name, description="info"))
    return {"model_name": exp_name}
