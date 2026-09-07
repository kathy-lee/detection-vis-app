import torch
import random
import numpy as np
import pkbar
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import os
import json
import pandas as pd


from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from torch.utils.data import ConcatDataset, DataLoader, random_split, Subset
from contextlib import contextmanager
from loguru import logger

import sys
sys.path.insert(0, '/home/kangle/projects/detection-vis-app')

from detection_vis_backend.datasets.dataset import DatasetFactory
from detection_vis_backend.networks.network import NetworkFactory
from detection_vis_backend.train.utils import FFTRadNet_collate, default_collate, DAROD_collate, radarpillar_collate
from detection_vis_backend.train.evaluate import eval_func_dict
from data import crud, schemas
from data.database import get_session_local

@contextmanager
def get_db():
    # data.database.SessionLocal is a module-level None that is never assigned -- the
    # live API is get_session_local(), as jobs/tasks.py already uses. Importing the
    # stale name made get_db() raise "'NoneType' object is not callable" on every call,
    # and its only caller is the CARRADA branch of the "original" splitmode
    # (CreateDataLoaders, which resolves sequences by name), so that split has never
    # worked -- including the official 18/6/6 split the published DAROD schedule uses.
    db = get_session_local()
    try:
        yield db
    finally:
        db.close()

@contextmanager
def function_specific_log_to_file(file_name, level="INFO"):
    log_id = logger.add(file_name, level=level)
    try:
        yield
    finally:
        logger.remove(log_id)

def CreateDataLoaders(datafiles: list, features: list, model_config: dict, train_config: dict, output_path: str):
    """
        Filter out data items with empty annotations when the network demands(for example when use CARRADA dataset 
        to train DAROD model).
    """
    collate_func = {
        'FFTRadNet': FFTRadNet_collate,
        'RODNet_CDC': default_collate,
        'RODNet_CDCv2': default_collate,
        'RODNet_HG': default_collate,
        'RODNet_HGv2': default_collate,
        'RODNet_HGwI': default_collate,
        'RODNet_HGwIv2': default_collate,
        'RadarFormer_hrformer2d': default_collate,
        'RECORD': default_collate,
        'RECORDNoLstm': default_collate,
        'RECORDNoLstmMulti': default_collate,
        'MVRECORD': default_collate,
        'RADDet': default_collate,
        'DAROD': DAROD_collate,
        'RAMP_CNN': default_collate,
        'RadarCrossAttention': default_collate,
        'RadarPillar': radarpillar_collate,
        'PointPillar': radarpillar_collate
    }
    logger.info("Preparing train/val/test dataloaders...")
    dataset_factory = DatasetFactory()
    dataset_type = datafiles[0]["parse"]

    if model_config['class'] == "DAROD":
        skip_empty_label = True
    else:
        skip_empty_label = False

    datasets = dict.fromkeys(["train", "val", "test"])
    split_samples_info = dict(train=[], val=[], test=[])
    if train_config["dataloader"]["splitmode"] != "original":
        if train_config['dataloader']['splitmode'] == 'sequence':
            assert len(datafiles) > 1
            assert 'train' in train_config['dataloader']['split_sequence']
            assert 'val' in train_config['dataloader']['split_sequence']

            for split_type in train_config['dataloader']['split_sequence'].keys():
                # The UI's per-file radios can leave a bucket empty. An empty train/val
                # is nonsense -- fail with the split's name instead of the opaque
                # "datasets should not be an empty iterable" assert from ConcatDataset([]).
                # An empty test is a legitimate choice (e.g. two files: one train, one
                # val): skip it here, and train() skips the test evaluation.
                if not train_config['dataloader']['split_sequence'][split_type]:
                    if split_type in ('train', 'val'):
                        raise ValueError(
                            f"Sequence split: no files assigned to '{split_type}'. "
                            "Assign at least one file to 'train' and one to 'val'.")
                    logger.info(f"Sequence split: no files assigned to '{split_type}'; skipping it.")
                    continue  # datasets[split_type] stays None
                samples_info = []
                valid_ids = []
                counter = 0
                dataset_inst_list = []
                for file in train_config['dataloader']['split_sequence'][split_type]:
                    dataset_inst = dataset_factory.get_instance(file['parse'], file['id'])
                    dataset_inst.prepare_for_train(features, train_config, model_config, splittype=split_type)
                    dataset_inst_list.append(dataset_inst)
                    for i in range(len(dataset_inst)):
                        samples_info.append([file['parse'], file['name'], file['id'], i])
                        counter += 1
                        if skip_empty_label and len(dataset_inst[i]['label']) == 0:
                            continue
                        valid_ids.append(counter-1)       
                concat_dataset = ConcatDataset(dataset_inst_list)
                # Extract dataset with valid(non-empty) labels
                datasets[split_type] = Subset(concat_dataset, valid_ids)
                split_samples_info[split_type] = [samples_info[id] for id in valid_ids]
                if skip_empty_label:
                    logger.debug(f"Before and After filtering samples with empty annotations: {split_type}: {counter} -> {len(valid_ids)}")  
        elif train_config['dataloader']['splitmode'] == 'random':
            samples_info = []
            valid_ids = []
            counter = 0
            dataset_inst_list = []
            for file in datafiles:
                dataset_inst = dataset_factory.get_instance(file['parse'], file['id'])
                dataset_inst.prepare_for_train(features, train_config, model_config)
                logger.debug(f"file {file['id']} prepare for train, length: {len(dataset_inst)} data samples.")
                dataset_inst_list.append(dataset_inst)
                for i in range(len(dataset_inst)):
                    logger.debug(i)
                    if len(datafiles) == 1:
                        samples_info.append(i)
                    else:
                        samples_info.append([file['parse'], file['name'], file['id'], i])
                    counter += 1
                    if skip_empty_label and len(dataset_inst[i]['label']) == 0:
                        continue
                    valid_ids.append(counter-1) 
            logger.debug(f"concat {len(datafiles)} dataset(s).")
            dataset = ConcatDataset(dataset_inst_list)
            split = np.array(train_config['dataloader']['split_random'])
            n_samples = len(dataset)
            n_train, n_val = int(split[0] * n_samples), int(split[1] * n_samples)
            n_test = n_samples - n_train - n_val
            train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(train_config['seed']))
            train_ids = train_dataset.indices
            val_ids = val_dataset.indices
            test_ids = test_dataset.indices
            logger.debug("random split the concated dataset.")
            
            if skip_empty_label:
                logger.debug(f"Before filtering samples with empty annotations: train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
                valid_ids_set = set(valid_ids)
                train_ids = [id for id in train_ids if id in valid_ids_set]
                val_ids = [id for id in val_ids if id in valid_ids_set]
                test_ids = [id for id in test_ids if id in valid_ids_set]
                datasets['train'] = Subset(dataset, train_ids)
                datasets['val'] = Subset(dataset, val_ids)
                datasets['test'] = Subset(dataset, test_ids)
                logger.debug(f"After filtering samples with empty annotations: train: {len(datasets['train'])}, val: {len(datasets['val'])}, test: {len(datasets['test'])}") 
            else:
                datasets['train'] = train_dataset
                datasets['val'] = val_dataset
                datasets['test'] = test_dataset

            split_samples_info['train'] = [samples_info[id] for id in train_ids]
            split_samples_info['val'] = [samples_info[id] for id in val_ids]
            split_samples_info['test'] = [samples_info[id] for id in test_ids]           
        else:
            ValueError("Split type not supported. Please choose following: 'sequence', 'random', 'original(some of the datasets has a built-in split).")        
    else:
        if dataset_type == "RADIal":
            # RADIal dataset is partly labeled
            Sequences = {'val':['RECORD@2020-11-22_12.49.56', 'RECORD@2020-11-22_12.11.49',
                                       'RECORD@2020-11-22_12.28.47','RECORD@2020-11-21_14.25.06'],
                        'test':['RECORD@2020-11-22_12.45.05','RECORD@2020-11-22_12.25.47',
                                'RECORD@2020-11-22_12.03.47','RECORD@2020-11-22_12.54.38']}
            dataset = dataset_factory.get_instance(dataset_type, datafiles[0]['id'])
            dataset.prepare_for_train(features, train_config, model_config)
            labels = dataset.labels
            dict_index_to_keys = {s:i for i,s in enumerate(dataset.sample_keys)}
            Val_indexes = []
            for seq in Sequences['val']:
                idx = np.where(labels[:,14]==seq)[0]
                Val_indexes.append(labels[idx,0])
            Val_indexes = np.unique(np.concatenate(Val_indexes))

            Test_indexes = []
            for seq in Sequences['test']:
                idx = np.where(labels[:,14]==seq)[0]
                Test_indexes.append(labels[idx,0])
            Test_indexes = np.unique(np.concatenate(Test_indexes))

            split_samples_info['val'] = [dict_index_to_keys[k] for k in Val_indexes]
            split_samples_info['test'] = [dict_index_to_keys[k] for k in Test_indexes]
            split_samples_info['train'] = np.setdiff1d(np.arange(len(dataset)), np.concatenate([split_samples_info['val'], split_samples_info['test']]))  
            datasets["train"] = Subset(dataset, split_samples_info['train'])
            datasets["val"] = Subset(dataset, split_samples_info['val'])
            datasets["test"] = Subset(dataset, split_samples_info['test'])
        elif dataset_type == "CARRADA":
            Sequences = {'train': ['2019-09-16-12-52-12', '2019-09-16-12-55-51', '2019-09-16-12-58-42', '2019-09-16-13-03-38', 
                                   '2019-09-16-13-11-12', '2019-09-16-13-14-29', '2019-09-16-13-20-20', '2019-09-16-13-25-35', 
                                   '2020-02-28-12-12-16', '2020-02-28-12-16-05', '2020-02-28-12-22-05', '2020-02-28-13-05-44', 
                                   '2020-02-28-13-06-53', '2020-02-28-13-09-58', '2020-02-28-13-10-51', '2020-02-28-13-12-42', 
                                   '2020-02-28-13-13-43', '2020-02-28-13-15-36'],
                        'val': ['2019-09-16-13-06-41', '2019-09-16-13-23-22', '2020-02-28-12-17-57', '2020-02-28-12-20-22', 
                               '2020-02-28-13-07-38', '2020-02-28-13-11-45'],
                        'test': ['2019-09-16-13-13-01', '2019-09-16-13-18-33', '2020-02-28-12-13-54', '2020-02-28-12-23-30', 
                                 '2020-02-28-13-08-51', '2020-02-28-13-14-35']}
            for split_type in Sequences.keys():
                samples_info = []
                valid_ids = []
                counter = 0
                dataset_inst_list = []
                for file_name in Sequences[split_type]:
                    with get_db() as db:
                        datafile = crud.get_datafile_by_name(db, file_name)
                    # dataset_inst = dataset_factory.get_instance('CARRADA', datafile.id)
                    # dataset_inst.parse(datafile.id, datafile.path, file_name, datafile.config)
                    # Parse on the shared instance (writes must persist, and parse-once
                    # skips the redundant re-parse when _parse_datafiles already ran);
                    # configure a request copy so this model's config cannot leak.
                    dataset_factory.get_shared_instance(
                        'CARRADA', datafile.id,
                        parse_args=(datafile.id, datafile.path, file_name, datafile.config))
                    dataset_inst = dataset_factory.get_instance('CARRADA', datafile.id)
                    dataset_inst.prepare_for_train(features, train_config, model_config, splittype=split_type)
                    dataset_inst_list.append(dataset_inst)
                    for i in range(len(dataset_inst)):
                        samples_info.append(['CARRADA', file_name, datafile.id, i])
                        counter += 1
                        if skip_empty_label and len(dataset_inst[i]['label']) == 0:
                            continue
                        valid_ids.append(counter-1)       
                concat_dataset = ConcatDataset(dataset_inst_list)
                # Extract dataset with valid(non-empty) labels
                datasets[split_type] = Subset(concat_dataset, valid_ids)
                split_samples_info[split_type] = [samples_info[id] for id in valid_ids]
                if skip_empty_label:
                    logger.debug(f"Before and After filtering samples with empty annotations: {split_type}: {counter} -> {len(valid_ids)}") 
        elif dataset_type == "RADDetDataset":
            # One request copy PER SPLIT. Augmentation is train-only and is decided from
            # instance state (splittype), so a single shared object would augment val and
            # test too -- exactly what the ViewOfDelft branch below already avoids.
            dataset = dataset_factory.get_instance(dataset_type, datafiles[0]['id'])
            dataset.prepare_for_train(features, train_config, model_config, splittype='train')
            sample_ids = [os.path.splitext(os.path.basename(f))[0] for f in dataset.anno_filenames]
            with open(os.path.join(dataset.root_path, "original_split_info.json"), "r") as f:
                dataset.original_split = json.load(f)
            split_samples_info['val'] = [sample_ids.index(i) for i in dataset.original_split['val'] if i in sample_ids]
            split_samples_info['test'] = [sample_ids.index(i) for i in dataset.original_split['test'] if i in sample_ids]
            split_samples_info['train'] = np.setdiff1d(np.arange(len(dataset)), np.concatenate([split_samples_info['val'], split_samples_info['test']]))
            for split_type in ('train', 'val', 'test'):
                if split_type == 'train':
                    split_ds = dataset
                else:
                    split_ds = dataset_factory.get_instance(dataset_type, datafiles[0]['id'])
                    split_ds.prepare_for_train(features, train_config, model_config,
                                               splittype=split_type)
                datasets[split_type] = Subset(split_ds, split_samples_info[split_type])
        elif dataset_type == "ViewOfDelft":
            # Official split from ImageSets (5,139 train / 1,296 val). One request
            # copy PER SPLIT (slice 3.5): the train copy carries the augmentor, the
            # val copy is clean (augmentation is train-only, and the reference's
            # shuffle_points is train-only too). label_2 holds exactly the train+val
            # frames; the official test split ships no labels, so the test bucket
            # stays empty and train() skips the test evaluation (2026-07-16 handling).
            for split_type, txt_name in (('train', 'train.txt'), ('val', 'val.txt')):
                dataset = dataset_factory.get_instance(dataset_type, datafiles[0]['id'])
                dataset.prepare_for_train(features, train_config, model_config, splittype=split_type)
                frame_names = [os.path.basename(f)[:-4] for f in dataset.label_filenames]
                name_to_idx = {n: i for i, n in enumerate(frame_names)}
                imagesets_dir = os.path.join(dataset.root_path, 'radar', 'ImageSets')
                with open(os.path.join(imagesets_dir, txt_name)) as f:
                    ids = [name_to_idx[line.strip()] for line in f if line.strip() in name_to_idx]
                split_samples_info[split_type] = ids
                datasets[split_type] = Subset(dataset, ids)
            datasets['test'] = None
        else:
            raise ValueError(f"{dataset_type} Dataset doesn't provide default data split.")
    
    train_loader = DataLoader(datasets['train'], 
                        batch_size=train_config['dataloader']['train']['batch_size'], 
                        shuffle=True,
                        num_workers=train_config['dataloader']['train']['num_workers'],
                        pin_memory=True,
                        collate_fn=collate_func[model_config['class']])
    val_loader =  DataLoader(datasets['val'], 
                        batch_size=train_config['dataloader']['val']['batch_size'], 
                        shuffle=False,
                        num_workers=train_config['dataloader']['val']['num_workers'],
                        pin_memory=True,
                        collate_fn=collate_func[model_config['class']])
    # test_loader =  DataLoader(datasets['test'], ...)
    # A sequence split may legitimately have no test files; then there is no test loader
    # and train() skips the final test evaluation.
    if datasets['test'] is None:
        test_loader = None
    else:
        test_loader = DataLoader(datasets['test'],
                        batch_size=train_config['dataloader']['test']['batch_size'],
                        shuffle=False,
                        num_workers=train_config['dataloader']['test']['num_workers'],
                        pin_memory=True,
                        collate_fn=collate_func[model_config['class']])
    
    if isinstance(split_samples_info['train'][0], (int, np.integer)):
        header = ['original sample id in the file']
    else:
        header = ['dataset_name', 'file_name', 'file_id', 'original sample id in the file']
    for split_type in split_samples_info.keys():
        df = pd.DataFrame(split_samples_info[split_type], columns=header)
        df.to_csv(os.path.join(output_path, f'{split_type}_sample_ids.csv'), index_label='sample id')

    logger.info('===========  Dataset  ==================:')
    logger.info(f"      Split Mode: {train_config['dataloader']['splitmode']}")
    logger.info(f"      Training:   {len(datasets['train'])}")
    logger.info(f"      Validation: {len(datasets['val'])}")
    logger.info(f"      Test:       {len(datasets['test']) if datasets['test'] is not None else 0}")
    return train_loader, val_loader, test_loader


def train(datafiles: list, features: list, model_config: dict, train_config: dict, pretrained: str=None, progress_cb=None):
    
    # Setup random seed
    torch.manual_seed(train_config['seed'])
    np.random.seed(train_config['seed'])
    random.seed(train_config['seed'])
    torch.cuda.manual_seed(train_config['seed'])

    # check if all datafiles from the same dataset
    dataset_type = datafiles[0]["parse"]

    # create experiment model name: <model type>___<dataset>___<timestamp>
    curr_date = datetime.now()
    exp_name = model_config['class'] + '_' + dataset_type + '_' + curr_date.strftime('%b-%d-%Y_%H:%M:%S')

    # Initialize tensorboard
    output_root = Path(os.getenv('MODEL_ROOTDIR'))
    (output_root / exp_name).mkdir(parents=True, exist_ok=True)
    output_dir = output_root / exp_name
    writer = SummaryWriter(output_dir)

    with function_specific_log_to_file(os.path.join(output_dir, "train_log.log")): 
        train_loader, val_loader, test_loader = CreateDataLoaders(datafiles, features, model_config, train_config, output_dir)

        # save model lineage info
        train_info_path = os.path.join(output_dir, 'train_info.txt')
        with open(train_info_path, 'w') as f:
            json.dump({"datafiles": datafiles, "features": features, "model_config": model_config, "train_config": train_config}, f)

        # save the evaluation of val dataset and test dataset
        val_eval_path = os.path.join(output_dir, "val_eval.csv")
        test_eval_path = os.path.join(output_dir, "test_eval.csv")
        df_val_eval = pd.DataFrame(columns=['Epoch', 'loss', 'mAP', 'mAR', 'mIoU'])

        # set device
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        #device = torch.device('cpu')
        # if device == 'cuda:0':
        #     torch.cuda.empty_cache()
        
        network_factory = NetworkFactory()
        model_type = model_config['class']
        model_config = model_config.copy()
        model_config.pop('class', None)
        net = network_factory.get_instance(model_type, model_config)
        logger.info(f'Network initialized: {exp_name}')
        net.to(device)
        net.init_lossfunc(train_config, device)

        # Optimizer
        lr = float(train_config['optimizer']['lr'])
        num_epochs = int(train_config['num_epochs'])
        # gamma = float(train_config['lr_scheduler']['gamma'])  # moved into the branches:
        # the RadarPillar config has no 'lr_scheduler' key (it uses 'scheduler'/OneCycleLR)
        if train_config['optimizer'].get('type') == 'AdamW':
            optimizer = optim.AdamW(filter(lambda p: p.requires_grad, net.parameters()),
                                    lr=lr, weight_decay=float(train_config['optimizer'].get('weight_decay', 0.01)))
        else:
            optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)

        # RadarPillar reference recipe (fthbng77/RadarPillar kitti_pointpillar.yaml:145-164,
        # adam_onecycle): OneCycleLR stepped per iteration, MOMS [0.95, 0.85] -> torch
        # max_momentum/base_momentum. Other models keep the 'lr_scheduler' path unchanged.
        step_scheduler_per_iter = False
        # Set by the 'plateau' branch: the name of the validation metric the scheduler is
        # stepped on, AFTER validation. None for every schedule that steps on the clock.
        plateau_monitor = None
        if train_config.get('scheduler', {}).get('type') == 'OneCycleLR':
            sched_cfg = train_config['scheduler']
            scheduler = lr_scheduler.OneCycleLR(
                optimizer, max_lr=float(sched_cfg['max_lr']),
                total_steps=num_epochs * len(train_loader),
                pct_start=float(sched_cfg.get('pct_start', 0.4)),
                div_factor=float(sched_cfg.get('div_factor', 10)),
                base_momentum=0.85, max_momentum=0.95)
            step_scheduler_per_iter = True
        elif train_config['lr_scheduler']['type'] == 'cyclic':
            # RAMP-CNN reference recipe (Xiangyu-Gao/Radar-multiple-perspective-object-
            # detection, config.py): CyclicLR(base_lr=1e-6, max_lr=1e-5,
            # step_size_up/down=430) with IS_Steplr=False. CyclicLR is a per-BATCH
            # schedule by construction -- step_size_up counts optimizer steps, not epochs
            # -- so it steps per iteration like OneCycleLR above.
            #
            # cycle_momentum MUST be False: it defaults to True and then writes a
            # 'momentum' key into the optimizer's param_groups, which Adam does not have
            # (it raises at construction). The reference uses Adam, so there is no
            # momentum to cycle.
            #
            # The reference's OWN StepLR branch, taken when IS_Steplr=True, decays with
            # gamma 0.1 -- not the 0.9 that RAMP_CNN_train_config.json's stand-in used.
            # Neither is a substitute for the cyclic schedule, which oscillates rather
            # than decays; that stand-in is why this branch exists.
            sched_cfg = train_config['lr_scheduler']
            step_up = int(sched_cfg.get('step_size_up', 430))
            scheduler = lr_scheduler.CyclicLR(
                optimizer,
                base_lr=float(sched_cfg['base_lr']),
                max_lr=float(sched_cfg['max_lr']),
                step_size_up=step_up,
                step_size_down=int(sched_cfg.get('step_size_down', step_up)),
                cycle_momentum=False)
            step_scheduler_per_iter = True
        elif train_config['lr_scheduler']['type'] == 'decay_to_floor':
            # RADDet reference recipe (ZhangAoCanada/RADDet train.py:166-178). Its loop is
            # step-based, not epoch-based, and the rule is
            #     lr <- lr_end + decay * (lr - lr_end)      every `decay_gap` steps,
            #                                               after `warmup_steps` steps
            # i.e. GEOMETRIC DECAY OF (lr - lr_end), so lr approaches lr_end as a FLOOR.
            # Neither StepLR nor ExponentialLR can express that: both scale lr itself and
            # therefore decay to zero. Measured at epoch 100 (batch 3): reference 1.72e-05,
            # StepLR(10, 0.9) 3.49e-05, ExponentialLR(0.9) 2.66e-09 -- four orders out.
            # So this is a LambdaLR on the step count instead, stepped per iteration.
            #
            # Off-by-one matters and is deliberate: the reference increments global_steps
            # BEFORE testing, and 60000 % 3000 == 0, so the first decay lands ON step
            # 60000. Hence `+ 1` in the exponent.
            #
            # NOTE the thresholds are counted in OPTIMIZER STEPS, so they only mean what
            # the reference meant if the batch size matches (3). At batch 4 the same 60000
            # steps would cover 32.8 epochs instead of 24.6.
            sched_cfg = train_config['lr_scheduler']
            lr_end = float(sched_cfg['lr_end'])
            decay = float(sched_cfg['decay'])
            decay_gap = int(sched_cfg['decay_gap'])
            warmup_steps = int(sched_cfg['warmup_steps'])

            def _decay_to_floor(step):
                # LambdaLR multiplies the BASE lr by this factor.
                if step < warmup_steps:
                    return 1.0
                n = (step - warmup_steps) // decay_gap + 1
                return (lr_end + (lr - lr_end) * (decay ** n)) / lr

            scheduler = lr_scheduler.LambdaLR(optimizer, _decay_to_floor)
            step_scheduler_per_iter = True
        elif train_config['lr_scheduler']['type'] == 'plateau':
            # RadarCrossAttention reference recipe (ravikothari510/crossattention_radar_
            # detector train.py:157-158): ReduceLROnPlateau(patience=4, factor=0.1), stepped
            # once per epoch on the VALIDATION metric -- there, the mean over val batches of
            # the sum-MSE between sigmoid(pred_mask) and the Gaussian target, which the
            # evaluator returns as 'val_mse'. A metric-driven schedule cannot step where the
            # clock-driven ones do (before validation), so `plateau_monitor` defers it to
            # after the evaluator has run. 'loss' monitors the evaluator's val loss.
            #
            # In this torch (2.1) ReduceLROnPlateau is not an LRScheduler subclass and has
            # no get_last_lr(); the lr history below reads the optimizer instead.
            sched_cfg = train_config['lr_scheduler']
            scheduler = lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode=sched_cfg.get('mode', 'min'),
                factor=float(sched_cfg.get('factor', 0.1)),
                patience=int(sched_cfg.get('patience', 4)))
            plateau_monitor = sched_cfg.get('monitor', 'loss')
        elif train_config['lr_scheduler']['type'] == 'step':
            gamma = float(train_config['lr_scheduler']['gamma'])
            step_size = int(train_config['lr_scheduler']['step_size'])
            scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        elif train_config['lr_scheduler']['type'] == 'exp':
            gamma = float(train_config['lr_scheduler']['gamma'])
            scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
        else:
            raise ValueError

        logger.info('===========  Optimizer  ==================:')
        logger.info(f'      LR: {lr}')
        logger.info(f'      num_epochs: {num_epochs}')
        logger.info('')

        # Mixed precision, config-gated (`"amp": true`), off by default so every existing
        # recipe is bit-for-bit unchanged. The RadarCrossAttention reference trains under
        # torch.cuda.amp.autocast with a GradScaler (train.py:62,107-110,165); measured
        # here at its batch 16: 1.41 GiB / 0.27 s per iteration against 2.72 GiB / 0.60 s
        # in fp32. Loss functions built on binary_cross_entropy_with_logits stay in fp32
        # under autocast, so the sum-reduced losses these models use do not overflow.
        use_amp = bool(train_config.get('amp', False))
        if use_amp and device.type != 'cuda':
            logger.warning("amp requested but no CUDA device -- training in fp32")
            use_amp = False
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

        # Train
        startEpoch = 0
        global_step = 0
        history = {'train_loss':[],'val_loss':[],'lr':[],'mAP':[],'mAR':[],'mIoU':[]}
        best_mAP = 0
        # Optional config-driven early stopping, e.g.
        # {"monitor": "mAP", "mode": "max", "patience": 20} -- monitor "loss" reads
        # val_loss. Absent from the config means no early stopping (full schedule).
        early_stop = train_config.get('early_stop')
        # Optional config-driven checkpoint selection for the FINAL test evaluation,
        # e.g. {"monitor": "mAP", "mode": "max"}. The RECORD reference tests
        # trainer.test(..., ckpt_path='best'), i.e. the best epoch by its monitored
        # metric, not the weights left at loop exit -- with early stopping those differ
        # by up to `patience` epochs. Absent from the config keeps the final weights,
        # which is what other families' references do (DAROD's eval.py restores
        # latest_checkpoint).
        test_checkpoint = train_config.get('test_checkpoint')
        best_state = None
        best_score = None
        best_state_epoch = None

        if pretrained:
            logger.info('===========  Resume training  ==================:')
            dict = torch.load(pretrained)
            net.load_state_dict(dict['net_state_dict'])
            optimizer.load_state_dict(dict['optimizer'])
            scheduler.load_state_dict(dict['scheduler'])
            startEpoch = dict['epoch']+1
            history = dict['history']
            global_step = dict['global_step']
            if 'scaler' in dict:
                scaler.load_state_dict(dict['scaler'])
            logger.info(f'       ... Start at epoch: {startEpoch}')


        for epoch in range(startEpoch, num_epochs):
            kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)
            logger.info(f'Epoch {epoch+1}/{num_epochs}')
            # report the epoch as running so the UI shows progress during (not only after) it
            if progress_cb is not None:
                progress_cb(epoch + 1, num_epochs, None, 'running')
            ###################
            ## Training loop ##
            ###################
            net.train()
            running_loss = 0.0

            # Gradient accumulation: `accum` micro-batches of the dataloader's batch_size
            # compose one effective batch of accum*batch_size. Mirrors PyTorch Lightning's
            # accumulate_grad_batches, which the reference CARRADA configs set -- including
            # its loss scaling: PL divides the step loss by the accumulation factor before
            # backward, so the accumulated gradient is the MEAN over the effective batch,
            # not its sum. Needed for MVRECORD, which OOMs at the published batch 8 on a
            # 24 GB card and must run batch 4 x accum 2 to keep the published effective
            # batch. accum=1 (the default) reproduces the previous behaviour exactly.
            accum = max(1, int(train_config.get('accumulate_grad', 1) or 1))
            optimizer.zero_grad()

            for iter, data in enumerate(train_loader):
                for key in data:
                    if isinstance(data[key], torch.Tensor):
                        if model_type in ('RadarPillar', 'PointPillar'):
                            # pillar coords / gt_labels are integer tensors (scatter
                            # indexing, class ids) -- .float() would corrupt them
                            data[key] = data[key].to(device)
                        else:
                            data[key] = data[key].to(device).float()

                if model_type in ('RadarPillar', 'PointPillar'):
                    inputs = data          # the collated dict IS the network input
                elif len(features) == 1:
                    inputs = data[features[0]]
                else:
                    inputs = {feature: data[feature] for feature in features}
                logger.debug('get input')
                # Actual sample count of THIS batch -- the final batch is short whenever
                # the split is not a multiple of batch_size (drop_last is unset), and the
                # configured value would over-weight it in the epoch mean.
                if model_type in ('RadarPillar', 'PointPillar'):
                    n_samples = len(data['frame_ids'])
                elif len(features) == 1:
                    n_samples = inputs.shape[0]
                else:
                    n_samples = inputs[features[0]].shape[0]

                # The loss is computed inside autocast as well, as the reference does; with
                # use_amp False the context is a no-op.
                with torch.set_grad_enabled(True), torch.autocast(device_type='cuda', enabled=use_amp):
                    outputs = net(inputs)
                    logger.debug(f'get net output')
                    # pop out feature data from data dict, keep only label info
                    for feature in features:
                        data.pop(feature, None)   # RadarPillar batches have no feature-named key
                    loss = net.get_loss(outputs, data, train_config)
                logger.info(f"Train sample {iter} loss: {loss}")
                scaler.scale(loss / accum).backward()

                # Step on every accum-th micro-batch, and on the last batch of the epoch so
                # a trailing partial group is not silently dropped (PL behaves the same).
                # Clipping belongs at the step boundary, on the fully accumulated gradient.
                if (iter + 1) % accum == 0 or (iter + 1) == len(train_loader):
                    if train_config.get('grad_norm_clip'):
                        # Gradients must be unscaled before their norm means anything.
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(net.parameters(), float(train_config['grad_norm_clip']))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                    if step_scheduler_per_iter:
                        scheduler.step()

                running_loss += loss.item() * n_samples   # unscaled loss, actual batch
                kbar.update(iter, values=[("loss", loss.item())])
                global_step += 1

            if not step_scheduler_per_iter and plateau_monitor is None:
                scheduler.step()

            history['train_loss'].append(running_loss / len(train_loader.dataset))
            # ReduceLROnPlateau has no get_last_lr() in this torch; the optimizer is the
            # authority either way (every scheduler writes into its param_groups).
            history['lr'].append(optimizer.param_groups[0]['lr'])

            ######################
            ## validation phase ##
            ######################
            logger.info(f'=========== Validation of Val data ===========')
            if callable(eval_func_dict[model_type, dataset_type]):
                eval = eval_func_dict[model_type, dataset_type](net, val_loader, train_config, features, output_dir, 'val', device)
                logger.info('Validation of Val data finished.')
            else:
                raise ValueError(f"Evaluation of {model_type} model with {dataset_type} dataset not supported. ")
            
            history['val_loss'].append(eval['loss'])
            history['mAP'].append(eval['mAP'])
            history['mAR'].append(eval['mAR'])
            history['mIoU'].append(eval['mIoU'])

            # Evaluators may report more than the four shared columns -- MVRECORD returns
            # per-view mIoU, which is the form the reference and the paper report. Carry
            # whatever else came back so it reaches history and val_eval.csv; the test path
            # already writes every returned key (from_dict below).
            extras = {k: v for k, v in eval.items() if k not in ('loss', 'mAP', 'mAR', 'mIoU')}
            for k, v in extras.items():
                # Pad when a series starts late (a run resumed from a checkpoint written
                # before the key existed) so index == epoch stays true for every series --
                # test_checkpoint below uses argmax(series) AS an epoch index.
                history.setdefault(k, [None] * (len(history['mAP']) - 1)).append(v)

            if plateau_monitor is not None:
                # Deferred from the clock-driven position above: step on this epoch's
                # validation metric. The lr recorded in history for this epoch is the one
                # the epoch actually trained with; a reduction applies from the next.
                metric = eval['loss'] if plateau_monitor == 'loss' else eval[plateau_monitor]
                scheduler.step(metric)
                if optimizer.param_groups[0]['lr'] != history['lr'][-1]:
                    logger.info(f"ReduceLROnPlateau: {plateau_monitor} plateaued, lr "
                                f"{history['lr'][-1]:.3g} -> {optimizer.param_groups[0]['lr']:.3g}")

            new_row = pd.Series({'Epoch': epoch, 'loss': eval['loss'], 'mAP': eval['mAP'], 'mAR': eval['mAR'], 'mIoU': eval['mIoU'], **extras})
            df_val_eval = pd.concat([df_val_eval, pd.DataFrame([new_row])], ignore_index=True)
            
            kbar.add(1, values=[("val_loss", eval['loss']),("mAP", eval['mAP']),("mAR", eval['mAR']),("mIoU", eval['mIoU'])])

            if progress_cb is not None:
                progress_cb(epoch + 1, num_epochs, float(eval['loss']), 'train')

            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('Loss/test', eval['loss'], global_step)
            writer.add_scalar('Metrics/mAP', eval['mAP'], global_step)
            writer.add_scalar('Metrics/mAR', eval['mAR'], global_step)
            writer.add_scalar('Metrics/mIoU', eval['mIoU'], global_step)
            
            # Saving all checkpoint as the best checkpoint for multi-task is a balance between both --> up to the user to decide
            name_output_file = model_type + '_epoch{:02d}_loss_{:.4f}_AP_{:.4f}_AR_{:.4f}_IOU_{:.4f}.pth'.format(epoch, eval['loss'],eval['mAP'],eval['mAR'],eval['mIoU'])
            filename = output_dir / name_output_file
            
            checkpoint={}
            checkpoint['net_state_dict'] = net.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            checkpoint['scheduler'] = scheduler.state_dict()
            checkpoint['epoch'] = epoch
            checkpoint['history'] = history
            checkpoint['global_step'] = global_step
            checkpoint['scaler'] = scaler.state_dict()
            
            torch.save(checkpoint,filename)
            logger.info(f'Saved checkpoint of epoch {epoch}.')

            if test_checkpoint:
                score = eval['loss'] if test_checkpoint['monitor'] == 'loss' else eval[test_checkpoint['monitor']]
                improved = (best_score is None
                            or (score < best_score if test_checkpoint.get('mode', 'max') == 'min' else score > best_score))
                if improved:
                    best_score = score
                    best_state_epoch = epoch
                    best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}

            if early_stop:
                # Derived from history rather than a running counter so a resumed run
                # (history restored from the checkpoint) keeps the same stopping point.
                series = history['val_loss'] if early_stop['monitor'] == 'loss' else history[early_stop['monitor']]
                best_idx = int(np.argmin(series)) if early_stop.get('mode', 'max') == 'min' else int(np.argmax(series))
                since_best = len(series) - 1 - best_idx
                if since_best >= early_stop['patience']:
                    logger.info(f"Early stopping at epoch {epoch}: no {early_stop['monitor']} improvement "
                                f"for {since_best} epochs (best at epoch {best_idx}, "
                                f"patience {early_stop['patience']}).")
                    break

        df_val_eval.to_csv(val_eval_path, index=False)

        if test_checkpoint:
            monitor = test_checkpoint['monitor']
            mode = test_checkpoint.get('mode', 'max')
            series = history['val_loss'] if monitor == 'loss' else history[monitor]
            best_idx = (int(np.argmin(series)) if mode == 'min' else int(np.argmax(series))) if series else None
            if best_idx is not None and best_state is not None and best_idx == best_state_epoch:
                net.load_state_dict(best_state)
                logger.info(f'Restored best checkpoint for testing: epoch {best_idx} '
                            f'({monitor}={series[best_idx]:.4f}).')
            elif best_idx is not None:
                # Resumed run: the best epoch predates this segment, so those weights
                # exist only on disk, in the directory the resume checkpoint came from.
                import glob
                matches = sorted(glob.glob(os.path.join(os.path.dirname(pretrained),
                                                        f'{model_type}_epoch{best_idx:02d}_*.pth'))) if pretrained else []
                if matches:
                    net.load_state_dict(torch.load(matches[0], map_location=device)['net_state_dict'])
                    logger.info(f'Restored best checkpoint for testing from {matches[0]} '
                                f'(epoch {best_idx}, {monitor}={series[best_idx]:.4f}).')
                else:
                    logger.warning(f'Best epoch by {monitor} is {best_idx}, whose checkpoint was not found '
                                   f'(resumed run); testing the final weights instead. The reported test score '
                                   f'is NOT the reference-faithful best-checkpoint number.')

        logger.info(f'=========== Evaluation on Test data ===========')
        if test_loader is None:
            # Sequence split with no files assigned to 'test' -- a legitimate choice.
            logger.info('No files assigned to the test split; skipping test evaluation.')
        elif callable(eval_func_dict[model_type, dataset_type]):
            eval = eval_func_dict[model_type, dataset_type](net, test_loader, train_config, features, output_dir, 'test', device)
            logger.info('Evaluation on Test data finished.')
            df_test_val = pd.DataFrame.from_dict(eval, orient='index').transpose()
            df_test_val.to_csv(test_eval_path, index=False)
        else:
            raise ValueError(f"Evaluation of {model_type} model with {dataset_type} dataset not supported. ")
    return exp_name


# class TrainModelFlow(FlowSpec):
#     datafiles_str = Parameter('datafiles',
#                           help="Chosen data files",
#                           required=True,
#                           type=str)
#     features_str = Parameter('features',
#                         help="Chosen features",
#                         required=True,
#                         type=str)
#     model_config_str = Parameter('model_config', 
#                              help="Model configurations",
#                              required=True,
#                              type=str)
#     train_config_str = Parameter('train_config', 
#                              help="Train configurations",
#                              required=True,
#                              type=str)

#     @step
#     def start(self):
#         # save flow run id info
#         with open('modelflow_info.txt', 'w') as f:
#             f.write(f"RUN_ID: {current.run_id}\n")

#         logger.info("Training begins.")
#         print("########################### Training begins #############################")
#         self.datafiles = json.loads(self.datafiles_str)
#         self.features = json.loads(self.features_str)
#         self.model_config = json.loads(self.model_config_str)
#         self.train_config = json.loads(self.train_config_str)
#         print(self.datafiles)
#         print(self.features)
#         print(self.model_config)
#         print(self.train_config)
#         self.next(self.train_model)

#     @step
#     def train_model(self):
    
#         for file in self.datafiles:
#             dataset_factory = DatasetFactory()
#             dataset_inst = dataset_factory.get_instance(file['parse'], file['id'])
#             dataset_inst.parse(file['path'], file['name'], file['config'])
#             # specify the features as train input data type
#             dataset_inst.set_features(self.features)
#             train_loader, val_loader, test_loader, train_ids, val_ids, test_ids = CreateDataLoaders(dataset_inst, self.train_config)


#         # Setup random seed
#         torch.manual_seed(self.train_config['seed'])
#         np.random.seed(self.train_config['seed'])
#         random.seed(self.train_config['seed'])
#         torch.cuda.manual_seed(self.train_config['seed'])

#         # create experiment model name
#         curr_date = datetime.now()
#         exp_name = self.model_config['type'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
#         print(exp_name)

#         # save model path(also model name)
#         with open("modelflow_info.txt", 'a') as f:
#             f.write(f"EXP_NAME: {exp_name}\n")

#         # Initialize tensorboard
#         output_folder = Path(os.getenv('MODEL_ROOTDIR'))
#         output_folder.mkdir(parents=True, exist_ok=True)
#         (output_folder / exp_name).mkdir(parents=True, exist_ok=True)
#         writer = SummaryWriter(output_folder / exp_name)

#         # save sample split info
#         split_info_path = os.path.join(output_folder, exp_name, 'samples_split.txt')
#         with open(split_info_path, 'w') as f:
#             f.write(f"TRAIN_SAMPLE_IDS: {','.join(map(str, train_ids))}\n")
#             f.write(f"VAL_SAMPLE_IDS: {','.join(map(str, val_ids))}\n")
#             f.write(f"TEST_SAMPLE_IDS: {','.join(map(str, test_ids))}\n")

#         # save the evaluation of val dataset and test dataset
#         val_eval_path = os.path.join(output_folder, exp_name, "val_eval.csv")
#         test_eval_path = os.path.join(output_folder, exp_name, "test_eval.csv")
#         df_val_eval = pd.DataFrame(columns=['loss', 'mAP', 'mAR', 'mIoU'])

#         # set device
#         device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
#         network_factory = NetworkFactory()
#         model_type = self.model_config['type']
#         model_config = self.model_config.copy()
#         model_config.pop('type', None)
#         net = network_factory.get_instance(model_type, model_config)
#         net.to('cuda')

#         # Optimizer
#         lr = float(self.train_config['lr'])
#         step_size = int(self.train_config['step_size'])
#         gamma = float(self.train_config['gamma'])
#         optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
#         scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
#         num_epochs=int(self.train_config['num_epochs'])

#         print('===========  Optimizer  ==================:')
#         print('      LR:', lr)
#         print('      step_size:', step_size)
#         print('      gamma:', gamma)
#         print('      num_epochs:', num_epochs)
#         print('')

#         # Train
#         startEpoch = 0
#         global_step = 0
#         history = {'train_loss':[],'val_loss':[],'lr':[],'mAP':[],'mAR':[],'mIoU':[]}
#         best_mAP = 0

#         freespace_loss = nn.BCEWithLogitsLoss(reduction='mean')


#         # if resume:
#         #     print('===========  Resume training  ==================:')
#         #     dict = torch.load(resume)
#         #     net.load_state_dict(dict['net_state_dict'])
#         #     optimizer.load_state_dict(dict['optimizer'])
#         #     scheduler.load_state_dict(dict['scheduler'])
#         #     startEpoch = dict['epoch']+1
#         #     history = dict['history']
#         #     global_step = dict['global_step']
#         #     print('       ... Start at epoch:',startEpoch)


#         for epoch in range(startEpoch,num_epochs):
#             kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)
#             print(f'Epoch {epoch+1}/{num_epochs}')
#             ###################
#             ## Training loop ##
#             ###################
#             net.train()
#             running_loss = 0.0

#             for i, data in enumerate(train_loader):
#                 inputs = data[0].to('cuda').float()
#                 label_map = data[1].to('cuda').float()
#                 if(self.model_config['segmentation_head']=='True'):
#                     seg_map_label = data[2].to('cuda').double()

#                 # reset the gradient
#                 optimizer.zero_grad()
                
#                 # forward pass, enable to track our gradient
#                 with torch.set_grad_enabled(True):
#                     outputs = net(inputs)

#                 classif_loss,reg_loss = pixor_loss(outputs['Detection'], label_map, self.train_config['losses'])           
                
#                 prediction = outputs['Segmentation'].contiguous().flatten()
#                 label = seg_map_label.contiguous().flatten()        
#                 loss_seg = freespace_loss(prediction, label)
#                 loss_seg *= inputs.size(0)

#                 classif_loss *= self.train_config['losses']['weight'][0]
#                 reg_loss *= self.train_config['losses']['weight'][1]
#                 loss_seg *=self.train_config['losses']['weight'][2]


#                 loss = classif_loss + reg_loss + loss_seg

#                 writer.add_scalar('Loss/train', loss.item(), global_step)
#                 writer.add_scalar('Loss/train_clc', classif_loss.item(), global_step)
#                 writer.add_scalar('Loss/train_reg', reg_loss.item(), global_step)
#                 writer.add_scalar('Loss/train_freespace', loss_seg.item(), global_step)

#                 # backprop
#                 loss.backward()
#                 optimizer.step()

#                 # statistics
#                 running_loss += loss.item() * inputs.size(0)
            
#                 kbar.update(i, values=[("loss", loss.item()), ("class", classif_loss.item()), ("reg", reg_loss.item()),("freeSpace", loss_seg.item())])
#                 print(f'Step {i+1}/{len(train_loader)} - loss: {loss.item()}, class: {classif_loss.item()}, reg: {reg_loss.item()}, freeSpace: {loss_seg.item()}')

#                 global_step += 1


#             scheduler.step()

#             history['train_loss'].append(running_loss / len(train_loader.dataset))
#             history['lr'].append(scheduler.get_last_lr()[0])

            
#             ######################
#             ## validation phase ##
#             ######################
#             eval = run_evaluation(net, val_loader, check_perf=(epoch>=10), detection_loss=pixor_loss, 
#                                         segmentation_loss=freespace_loss, losses_params=self.train_config['losses'], device=device)
                
#             history['val_loss'].append(eval['loss'])
#             history['mAP'].append(eval['mAP'])
#             history['mAR'].append(eval['mAR'])
#             history['mIoU'].append(eval['mIoU'])

#             new_row = pd.Series({'loss': eval['loss'], 'mAP': eval['mAP'], 'mAR': eval['mAR'], 'mIoU': eval['mIoU']})
#             df_val_eval = pd.concat([df_val_eval, pd.DataFrame([new_row])], ignore_index=True)
            
#             kbar.add(1, values=[("val_loss", eval['loss']),("mAP", eval['mAP']),("mAR", eval['mAR']),("mIoU", eval['mIoU'])])


#             writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
#             writer.add_scalar('Loss/test', eval['loss'], global_step)
#             writer.add_scalar('Metrics/mAP', eval['mAP'], global_step)
#             writer.add_scalar('Metrics/mAR', eval['mAR'], global_step)
#             writer.add_scalar('Metrics/mIoU', eval['mIoU'], global_step)

#             # Saving all checkpoint as the best checkpoint for multi-task is a balance between both --> up to the user to decide
#             name_output_file = model_type + '_epoch{:02d}_loss_{:.4f}_AP_{:.4f}_AR_{:.4f}_IOU_{:.4f}.pth'.format(epoch, eval['loss'],eval['mAP'],eval['mAR'],eval['mIoU'])
#             filename = output_folder / exp_name / name_output_file

#             checkpoint={}
#             checkpoint['net_state_dict'] = net.state_dict()
#             checkpoint['optimizer'] = optimizer.state_dict()
#             checkpoint['scheduler'] = scheduler.state_dict()
#             checkpoint['epoch'] = epoch
#             checkpoint['history'] = history
#             checkpoint['global_step'] = global_step

#             torch.save(checkpoint,filename)
            
#             print('')

#         df_val_eval.to_csv(val_eval_path)
#         print("########################### Training ends sucessfully #############################")

#         print("########################### Evaluation begins #############################")
#         run_FullEvaluation(net, test_loader, test_eval_path, device=device)
#         print("########################### Evaluation ends sucessfully #############################")

#         self.next(self.end)

#     @step
#     def end(self):
#         print("TrainModelFlow ends.")

        


# if __name__ == '__main__':
#     TrainModelFlow()