import torch
import random
import numpy as np
import pkbar
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
import os
import logging
import json
import pandas as pd


from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from torch.utils.data import ConcatDataset, DataLoader, random_split, Subset
from contextlib import contextmanager

import sys
sys.path.insert(0, '/home/kangle/projects/detection-vis-app')

from detection_vis_backend.datasets.dataset import DatasetFactory
from detection_vis_backend.networks.network import NetworkFactory
from detection_vis_backend.train.utils import FFTRadNet_collate, default_collate, DAROD_collate 
from detection_vis_backend.train.evaluate import FFTRadNet_val_evaluation, FFTRadNet_test_evaluation, RODNet_evaluation, RECORD_CRUW_evaluation, RECORD_CARRADA_evaluation, MVRECORD_CARRADA_evaluation, RADDet_evaluation, DAROD_evaluation, RAMP_CNN_evaluation, RadarCrossAttention_evaluation
from data import crud, schemas
from data.database import SessionLocal

@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()  

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
        'RadarCrossAttention': default_collate
    }  
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
                    print(f"Before and After filtering samples with empty annotations: {split_type}: {counter} -> {len(valid_ids)}")  
        elif train_config['dataloader']['splitmode'] == 'random':
            samples_info = []
            valid_ids = []
            counter = 0
            dataset_inst_list = []
            for file in datafiles:
                dataset_inst = dataset_factory.get_instance(file['parse'], file['id'])
                dataset_inst.prepare_for_train(features, train_config, model_config)
                dataset_inst_list.append(dataset_inst)
                for i in range(len(dataset_inst)):
                    if len(datafiles) == 1:
                        samples_info.append(i)
                    else:
                        samples_info.append([file['parse'], file['name'], file['id'], i])
                    counter += 1
                    if skip_empty_label and len(dataset_inst[i]['label']) == 0:
                        continue
                    valid_ids.append(counter-1) 
            dataset = ConcatDataset(dataset_inst_list)
            split = np.array(train_config['dataloader']['split_random'])
            n_samples = len(dataset)
            n_train, n_val = int(split[0] * n_samples), int(split[1] * n_samples)
            n_test = n_samples - n_train - n_val
            train_dataset, val_dataset, test_dataset = random_split(dataset, [n_train, n_val, n_test], generator=torch.Generator().manual_seed(train_config['seed']))
            train_ids = train_dataset.indices
            val_ids = val_dataset.indices
            test_ids = test_dataset.indices
            
            if skip_empty_label:
                print(f"Before filtering samples with empty annotations: train: {len(train_dataset)}, val: {len(val_dataset)}, test: {len(test_dataset)}")
                valid_ids_set = set(valid_ids)
                train_ids = [id for id in train_ids if id in valid_ids_set]
                val_ids = [id for id in val_ids if id in valid_ids_set]
                test_ids = [id for id in test_ids if id in valid_ids_set]
                datasets['train'] = Subset(dataset, train_ids)
                datasets['val'] = Subset(dataset, val_ids)
                datasets['test'] = Subset(dataset, test_ids)
                print(f"After filtering samples with empty annotations: train: {len(datasets['train'])}, val: {len(datasets['val'])}, test: {len(datasets['test'])}") 
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
                    dataset_inst = dataset_factory.get_instance('CARRADA', datafile.id)
                    dataset_inst.parse(datafile.id, datafile.path, file_name, datafile.config)
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
                    print(f"Before and After filtering samples with empty annotations: {split_type}: {counter} -> {len(valid_ids)}") 
        elif dataset_type == "RADDetDataset":
            dataset = dataset_factory.get_instance(dataset_type, datafiles[0]['id'])
            dataset.prepare_for_train(features, train_config, model_config)
            sample_ids = [os.path.splitext(os.path.basename(f))[0] for f in dataset.anno_filenames]
            with open(os.path.join(dataset.root_path, "original_split_info.json"), "r") as f:
                dataset.original_split = json.load(f)
            split_samples_info['val'] = [sample_ids.index(i) for i in dataset.original_split['val'] if i in sample_ids]
            split_samples_info['test'] = [sample_ids.index(i) for i in dataset.original_split['test'] if i in sample_ids]
            split_samples_info['train'] = np.setdiff1d(np.arange(len(dataset)), np.concatenate([split_samples_info['val'], split_samples_info['test']]))  
            datasets['train'] = Subset(dataset, split_samples_info['train'])
            datasets['val'] = Subset(dataset, split_samples_info['val'])
            datasets['test'] = Subset(dataset, split_samples_info['test'])   
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
    test_loader =  DataLoader(datasets['test'], 
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

    print('===========  Dataset  ==================:')
    print('      Split Mode:', train_config['dataloader']['splitmode'])
    print('      Training:', len(datasets['train']))
    print('      Validation:', len(datasets['val']))
    print('      Test:', len(datasets['test']))
    return train_loader, val_loader, test_loader


RODNet_subtypes = ["RODNet_CDC", "RODNet_CDCv2", "RODNet_HG", "RODNet_HGv2", "RODNet_HGwI", "RODNet_HGwIv2"]
RECORD_subtypes = ["RECORD", "RECORDNoLstm", "RECORDNoLstmMulti"]
network_types = ["FFRRadNet", "MVRECORD", "DAROD", "RADDet", "RAMP_CNN", "RadarCrossAttention" ] + RODNet_subtypes + RECORD_subtypes
dataset_types = ["RADIal", "CRUW", "CARRADA", "RADDetDataset", "UWCR"]
eval_func_dict = {(i, j): None for i in network_types for j in dataset_types}
eval_func_dict["FFRRadNet"]["RADIal"] = FFTRadNet_val_evaluation
for i in RODNet_subtypes:
    eval_func_dict[i]["CRUW"] = RODNet_evaluation
for i in RECORD_subtypes:
    eval_func_dict[i]["CRUW"] = RECORD_CRUW_evaluation
    eval_func_dict[i]["CARRADA"] = RECORD_CARRADA_evaluation
eval_func_dict["MVRECORD"]["CARRADA"] = MVRECORD_CARRADA_evaluation
eval_func_dict["RADDet"]["RADDetDataset"] = RADDet_evaluation
eval_func_dict["DAROD"]["RADDetDataset"] = DAROD_evaluation
eval_func_dict["RAMP_CNN"]["UWCR"] = RAMP_CNN_evaluation
eval_func_dict["RadarCrossAttention"]["CARRADA"] = RadarCrossAttention_evaluation


def train(datafiles: list, features: list, model_config: dict, train_config: dict, pretrained: str=None):    
    # Setup random seed
    torch.manual_seed(train_config['seed'])
    np.random.seed(train_config['seed'])
    random.seed(train_config['seed'])
    torch.cuda.manual_seed(train_config['seed'])

    # create experiment model name
    curr_date = datetime.now()
    exp_name = model_config['class'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')

    # check if all datafiles from the same dataset
    dataset_type = datafiles[0]["parse"]

    # Initialize tensorboard
    output_root = Path(os.getenv('MODEL_ROOTDIR'))
    (output_root / exp_name).mkdir(parents=True, exist_ok=True)
    output_dir = output_root / exp_name
    writer = SummaryWriter(output_dir)

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
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    #device = torch.device('cpu')
    
    network_factory = NetworkFactory()
    model_type = model_config['class']
    model_config = model_config.copy()
    model_config.pop('class', None)
    net = network_factory.get_instance(model_type, model_config)
    print(f'Network initialized: {exp_name}')
    net.to(device)
    net.init_lossfunc(train_config['losses'])

    # Optimizer
    lr = float(train_config['optimizer']['lr'])
    gamma = float(train_config['lr_scheduler']['gamma'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    if train_config['lr_scheduler']['type'] == 'step':
        step_size = int(train_config['lr_scheduler']['step_size'])
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    elif train_config['lr_scheduler']['type'] == 'exp':
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    else:
        raise ValueError
    num_epochs=int(train_config['num_epochs'])

    print('===========  Optimizer  ==================:')
    print('      LR:', lr)
    print('      num_epochs:', num_epochs)
    print('')

    # Train
    startEpoch = 0
    global_step = 0
    history = {'train_loss':[],'val_loss':[],'lr':[],'mAP':[],'mAR':[],'mIoU':[]}
    best_mAP = 0

    if pretrained:
        print('===========  Resume training  ==================:')
        dict = torch.load(pretrained)
        net.load_state_dict(dict['net_state_dict'])
        optimizer.load_state_dict(dict['optimizer'])
        scheduler.load_state_dict(dict['scheduler'])
        startEpoch = dict['epoch']+1
        history = dict['history']
        global_step = dict['global_step']
        print('       ... Start at epoch:',startEpoch)


    for epoch in range(startEpoch, num_epochs):
        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)
        print(f'Epoch {epoch+1}/{num_epochs}')
        ###################
        ## Training loop ##
        ###################
        net.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            for key in data:
                if isinstance(data[key], str) or isinstance(data[key], list):
                    continue
                data[key] = data[key].to(device).float()
            if len(features) == 1:
                inputs = data[features[0]]
            else:
                #inputs = tuple(data[feature] for feature in features)
                inputs = {feature: data[feature] for feature in features}

            optimizer.zero_grad()
            with torch.set_grad_enabled(True):
                outputs = net(inputs)

            # pop out feature data from data dict, keep only label info
            for feature in features:
                data.pop(feature)
            loss = net.get_loss(outputs, data, train_config)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * train_config['dataloader']['train']['batch_size']
            # kbar.update(i, values=[("loss", loss.item()), ("class", classif_loss.item()), ("reg", reg_loss.item()),("freeSpace", loss_seg.item())])
            # print(f'Step {i+1}/{len(train_loader)} - loss: {loss.item()}, class: {classif_loss.item()}, reg: {reg_loss.item()}, freeSpace: {loss_seg.item()}')
            kbar.update(i, values=[("loss", loss.item())])
            print(f'Step {i+1}/{len(train_loader)} - loss: {loss.item()}')
            global_step += 1

        scheduler.step()

        history['train_loss'].append(running_loss / len(train_loader.dataset))
        history['lr'].append(scheduler.get_last_lr()[0])

        ######################
        ## validation phase ##
        ######################
        print(f'=========== Validation of Val data ===========')
        if callable(eval_func_dict[model_type][dataset_type]):
            eval = eval_func_dict[model_type][dataset_type](net, val_loader, train_config, features, output_dir, device)
        else:
            raise ValueError(f"Evaluation of {model_type} model with {dataset_type} dataset not supported. ")
            
        history['val_loss'].append(eval['loss'])
        history['mAP'].append(eval['mAP'])
        history['mAR'].append(eval['mAR'])
        history['mIoU'].append(eval['mIoU'])

        new_row = pd.Series({'Epoch': epoch, 'loss': eval['loss'], 'mAP': eval['mAP'], 'mAR': eval['mAR'], 'mIoU': eval['mIoU']})
        df_val_eval = pd.concat([df_val_eval, pd.DataFrame([new_row])], ignore_index=True)

        kbar.add(1, values=[("val_loss", eval['loss']),("mAP", eval['mAP']),("mAR", eval['mAR']),("mIoU", eval['mIoU'])])

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

        torch.save(checkpoint,filename)

    df_val_eval.to_csv(val_eval_path, index=False)

    print(f'=========== Evaluation of Test data ===========')
    if callable(eval_func_dict[model_type][dataset_type]):
        eval = eval_func_dict[model_type][dataset_type](net, test_loader, train_config, features, output_dir, device)
    else:
        raise ValueError(f"Evaluation of {model_type} model with {dataset_type} dataset not supported. ")
    
    df_test_val = pd.DataFrame.from_dict(eval, orient='index').transpose()
    df_test_val.to_csv(test_eval_path, index=False)       
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

#         logging.info("Training begins.")
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