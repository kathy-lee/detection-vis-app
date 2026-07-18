"""Train RadarPillar on View-of-Delft from the command line.

Same code path as the train page (RadarPillar + ViewOfDelft + split
"original"), without the job queue: parses the datafile, then calls the
backend's train() with the shipped model/train configs (80 epochs, 5-frame
radar, per-epoch official VoD devkit evaluation).

Usage:
    .venv/bin/python scripts/train_radarpillar_vod.py           # full run (config's 80 epochs)
    SMOKE=1 .venv/bin/python scripts/train_radarpillar_vod.py   # 1-epoch smoke test

Assumes the backend services are reachable (Postgres with the datafile
registry) and the VoD dataset path registered under file id 110.
"""
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
os.environ.setdefault('MODEL_ROOTDIR', '/home/kangle/dataset/trained_models')
os.environ.setdefault('DATABASE_URL', 'postgresql://postgres:postgres@localhost/postgres')

from data.database import get_session_local
from data import crud
from detection_vis_backend.datasets.dataset import DatasetFactory
from detection_vis_backend.train.train import train

VOD_FILE_ID = 110

db = get_session_local()
datafile = crud.get_datafile(db, VOD_FILE_ID)
db.close()
DatasetFactory().get_shared_instance(
    "ViewOfDelft", VOD_FILE_ID,
    parse_args=(VOD_FILE_ID, datafile.path, datafile.name, datafile.config))

with open(os.path.join(REPO_ROOT, 'detection_vis_backend/networks/RadarPillar_model_config.json')) as f:
    model_config = json.load(f)
model_config['class'] = 'RadarPillar'
with open(os.path.join(REPO_ROOT, 'detection_vis_backend/train/RadarPillar_train_config.json')) as f:
    train_config = json.load(f)
train_config['dataloader']['splitmode'] = 'original'
if os.environ.get('SMOKE', '0') == '1':
    train_config['num_epochs'] = 1

datafiles = [{'parse': 'ViewOfDelft', 'id': VOD_FILE_ID,
              'name': datafile.name, 'path': datafile.path}]
exp = train(datafiles, ['radarPC'], model_config, train_config)
print('EXP:', exp)
