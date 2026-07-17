import json
import os
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
        # inst = factory.get_instance(f["parse"], f["id"])
        # inst.parse(f["id"], f["path"], f["name"], f["config"])
        # parse() writes file state that must persist, so it runs on the shared instance;
        # get_shared_instance also makes it run once per file_id instead of on every call.
        factory.get_shared_instance(
            f["parse"], f["id"],
            parse_args=(f["id"], f["path"], f["name"], f["config"]))


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


def _resolve_last_checkpoint(model_id):
    with _session() as db:
        model = crud.get_model(db, model_id)
    model_rootdir = os.getenv("MODEL_ROOTDIR")
    model_dir = os.path.join(model_rootdir, model.name)
    with open(os.path.join(model_dir, "train_info.txt")) as f:
        parameters = json.load(f)
    pattern = "epoch{:02d}".format(parameters["train_config"]["num_epochs"] - 1)
    for fname in os.listdir(model_dir):
        if pattern in fname and fname.endswith(".pth"):
            return os.path.join(model_dir, fname)
    raise FileNotFoundError(f"No final checkpoint ({pattern}) for model {model.name}")


def run_retraining(model_id, datafiles, features, model_config, train_config):
    job = get_current_job()
    checkpoint = _resolve_last_checkpoint(model_id)
    _parse_datafiles(datafiles)
    exp_name = train(datafiles, features, model_config, train_config,
                     pretrained=checkpoint, progress_cb=_make_cb(job))
    with _session() as db:
        crud.add_model(db, schemas.MLModelCreate(name=exp_name, description="info", parent_id=model_id))
    return {"model_name": exp_name}
