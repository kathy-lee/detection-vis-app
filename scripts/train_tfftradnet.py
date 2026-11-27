"""Train T-FFTRadNet from the command line, as the Train page would.

The model and train configs are built the way the Train page builds them (the dataset
block of T_FFTRadNet_model_config.json; the train config's default, then the dataset or
per-feature block), with the official split; then the backend's train() runs without the
job queue. The model row is not added to the database: the script prints the experiment
name to register.

Usage:
    .venv/bin/python scripts/train_tfftradnet.py RADIal RD
    .venv/bin/python scripts/train_tfftradnet.py RADDetDataset ADC
    .venv/bin/python scripts/train_tfftradnet.py RADIal RD --input-mode RD_shift
    SMOKE=1 ...   # one epoch

Assumes Postgres with the datafile registry is reachable.

The script didn't register the model in the app. To add its row, run this in the prompt; 
it is the same call the app's training job makes:
uv run python -c "import os; from detection_vis_backend.paths import load_env; load_env(); os.environ.setdefault('DATABASE_URL','postgresql://postgres:postgres@localhost/postgres'); from data import crud, schemas; from data.database import get_session_local; db=get_session_local(); crud.add_model(db, schemas.MLModelCreate(name=<EXP_NAME>, description='info')); db.close()"
"""
import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
from detection_vis_backend.paths import load_env  # noqa: E402

load_env()
os.environ.setdefault('DATABASE_URL', 'postgresql://postgres:postgres@localhost/postgres')

from data import models  # noqa: E402
from data.database import get_session_local  # noqa: E402
from detection_vis_backend.datasets.dataset import DatasetFactory  # noqa: E402
from detection_vis_backend.train.train import train  # noqa: E402


def configs(dataset, feature, input_mode):
    with open(os.path.join(REPO_ROOT, 'detection_vis_backend/networks/T_FFTRadNet_model_config.json')) as f:
        model_config = {k: v for k, v in json.load(f)[dataset].items() if not k.startswith('_')}
    model_config['class'] = 'T_FFTRadNet'
    # The dataset block's input_mode serves one feature only (RADDet's is RAD), so without
    # an explicit mode the network reads the feature it was given.
    model_config['input_mode'] = input_mode or feature
    with open(os.path.join(REPO_ROOT, 'detection_vis_backend/train/T_FFTRadNet_train_config.json')) as f:
        data = json.load(f)
    train_config = data['default'].copy()
    block = data.get(dataset, {})
    train_config.update(block[feature] if feature in block else block)
    train_config['dataloader'] = dict(train_config['dataloader'], splitmode='original')
    if os.environ.get('SMOKE', '0') == '1':
        train_config['num_epochs'] = 1
    return model_config, train_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('dataset', choices=['RADIal', 'RADDetDataset'])
    ap.add_argument('feature', choices=['RD', 'ADC', 'RAD'])
    ap.add_argument('--input-mode', help="defaults to the feature; e.g. RD_shift for RADIal RD")
    args = ap.parse_args()

    db = get_session_local()
    rows = db.query(models.File).filter_by(parse=args.dataset).all()
    db.close()
    if len(rows) != 1:
        sys.exit(f"expected one {args.dataset} datafile, found {[r.id for r in rows]}")
    row = rows[0]
    DatasetFactory().get_shared_instance(args.dataset, row.id,
                                         parse_args=(row.id, row.path, row.name, row.config))

    model_config, train_config = configs(args.dataset, args.feature, args.input_mode)
    print('MODEL CONFIG:', json.dumps(model_config), flush=True)
    print('TRAIN CONFIG:', json.dumps(train_config), flush=True)
    datafiles = [{'parse': args.dataset, 'id': row.id, 'name': row.name, 'path': row.path,
                  'config': row.config}]
    exp = train(datafiles, [args.feature], model_config, train_config)
    print('EXP:', exp, flush=True)


if __name__ == '__main__':
    main()
