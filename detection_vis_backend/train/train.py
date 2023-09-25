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
import pickle
import time


from metaflow import FlowSpec, Parameter, step, current
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from torch.optim import lr_scheduler
from torch.utils.data import Dataset
from tqdm import tqdm

import sys
sys.path.insert(0, '/home/kangle/projects/detection-vis-app')

from detection_vis_backend.datasets.dataset import DatasetFactory
from detection_vis_backend.networks.network import NetworkFactory
from detection_vis_backend.train.utils import CreateDataLoaders, pixor_loss
from detection_vis_backend.train.evaluate import run_evaluation, run_FullEvaluation, RODNet_evaluation


class TrainDataset(Dataset):
    def __init__(self, datasets, model_cfg, train_cfg, features):
        """
        Args:
            datasets (list): A list of dataset instances.
        """
        if not isinstance(datasets, list):
            raise TypeError("The provided dataset instaces must be a list!")

        if not datasets:
            raise ValueError("The provided dataset instances list is empty!")
        
        if not all(type(item) == type(datasets[0]) for item in datasets):
            raise ValueError("All dataset instaces must be of the same type.")

        self.datasets = datasets
        self.dataclass = type(datasets[0]).__name__
        self.dataset_names = [dataset.seq_name for dataset in datasets]
        self.dataset_lengths = [len(dataset) for dataset in datasets]
        self.cumulative_lengths = np.cumsum(self.dataset_lengths)
        self.model_cfg = model_cfg
        self.train_cfg = train_cfg
        self.features = features
        self.data_root = self.datasets[0].root_path

        if self.model_cfg['class'] in ('RODNet', 'RECORD'):
            with open(self.datasets[0].config, 'r') as file:
                self.sensor_cfg = json.load(file)

            # parameters settings
            self.n_class = 3 # dataset.object_cfg.n_class
            self.win_size = train_cfg['win_size']

            # if split == 'train' or split == 'valid':
            #     self.step = train_cfg['train_step']
            #     self.stride = train_cfg['train_stride']
            # else:
            #     self.step = train_cfg['test_step']
            #     self.stride = train_cfg['test_stride']
            self.step = train_cfg['train_step']
            self.stride = train_cfg['train_stride']

            self.is_random_chirp = True
            self.noise_channel = False

            # Dataloader for MNet
            if 'mnet_cfg' in model_cfg:
                in_chirps, out_channels = model_cfg['mnet_cfg']
                self.n_chirps = in_chirps
            elif self.model_cfg['class'] == 'RECORD':
                self.n_chirps = 4
            else:
                self.n_chirps = 1

            self.chirp_ids = self.sensor_cfg['radar_cfg']['chirp_ids']

            # dataset initialization
            self.image_paths = []
            self.radar_paths = []
            self.obj_infos = []
            self.confmaps = []
            self.n_data = 0
            self.index_mapping = []

            # if subset is not None:
            #     self.data_files = [subset + '.pkl']
            # else:
            #     # self.data_files = list_pkl_filenames(config_dict['dataset_cfg'], split)

            # self.data_files = sorted(os.listdir(self.pkl_path))
            # self.seq_names = [name.split('.')[0] for name in self.data_files]
            # self.n_seq = len(self.seq_names)
            self.data_files = [os.path.join(dataset.pkl_path, dataset.seq_name + '.pkl') for dataset in self.datasets]
            self.seq_names = [dataset.seq_name for dataset in self.datasets]

            for seq_id, data_file in enumerate(tqdm(self.data_files)):
                data_details = pickle.load(open(data_file, 'rb'))
                # if split == 'train' or split == 'valid':
                #     assert data_details['anno'] is not None
                n_frame = data_details['n_frame']
                self.image_paths.append(data_details['image_paths'])
                self.radar_paths.append(data_details['radar_paths'])
                n_data_in_seq = (n_frame - (self.win_size * self.step - 1)) // self.stride + (
                    1 if (n_frame - (self.win_size * self.step - 1)) % self.stride > 0 else 0)
                ## Added for dataset iterate with win size, step, stride
                self.dataset_lengths[seq_id] = n_data_in_seq
                ## End
                self.n_data += n_data_in_seq
                for data_id in range(n_data_in_seq):
                    self.index_mapping.append([seq_id, data_id * self.stride])
                if data_details['anno'] is not None:
                    self.obj_infos.append(data_details['anno']['metadata'])
                    self.confmaps.append(data_details['anno']['confmaps'])
            ## Added for dataset iterate with win size, step, stride
            self.cumulative_lengths = np.cumsum(self.dataset_lengths)
            ## End
            #print(f"##################### index_mapping: {len(self.index_mapping)}, {self.dataset_lengths}, {self.index_mapping[0]}")
            return

    def __len__(self):
        if self.model_cfg['class'] == 'RODNet':
            return self.n_data
        else:
            return sum(self.dataset_lengths)

    def __getitem__(self, index):
        # # Determine which dataset the index belongs to
        # if idx < 0:
        #     if -idx > len(self):
        #         raise ValueError("absolute value of index should not exceed dataset length")
        #     idx = len(self) + idx
        # dataset_idx = next(i for i, cumulative_length in enumerate(self.cumulative_lengths) if idx < cumulative_length)
        # if dataset_idx > 0:
        #     idx = idx - self.cumulative_lengths[dataset_idx - 1]
        # return self.datasets[dataset_idx].get_data(idx) 
    
        if self.model_cfg['class'] == 'FFTRadNet':
            # (radar_FFT, segmap,out_label,box_labels,image)
            # Determine which dataset the index belongs to
            if idx < 0:
                if -idx > len(self):
                    raise ValueError("absolute value of index should not exceed dataset length")
                idx = len(self) + idx
            dataset_idx = next(i for i, cumulative_length in enumerate(self.cumulative_lengths) if idx < cumulative_length)
            if dataset_idx > 0:
                idx = idx - self.cumulative_lengths[dataset_idx - 1]
            return self.datasets[dataset_idx](idx) 
        
            # data_dict = {}
            # for f in self.features:
            #     data_dict[f] = self.datasets[dataset_idx].get_feature(f, idx)
            # return data_dict
        elif self.model_cfg['class'] in ('RODNet', 'RECORD'):
            seq_id, data_id = self.index_mapping[index]
            seq_name = self.seq_names[seq_id]
            image_paths = self.image_paths[seq_id]
            radar_paths = self.radar_paths[seq_id]
            if len(self.confmaps) != 0:
                this_seq_obj_info = self.obj_infos[seq_id]
                this_seq_confmap = self.confmaps[seq_id]

            data_dict = dict(
                status=True,
                seq_names=seq_name,
                image_paths=[]
            )

            if self.is_random_chirp:
                chirp_id = random.randint(0, len(self.chirp_ids) - 1)
            else:
                chirp_id = 0

            # Dataloader for MNet
            if 'mnet_cfg' in self.model_cfg:
                chirp_id = self.chirp_ids

            radar_configs = self.sensor_cfg['radar_cfg']
            ramap_rsize = radar_configs['ramap_rsize']
            ramap_asize = radar_configs['ramap_asize']

            # Load radar data
            try:
                if radar_configs['data_type'] == 'RI' or radar_configs['data_type'] == 'AP':  # drop this format
                    radar_npy_win = np.zeros((self.win_size, ramap_rsize, ramap_asize, 2), dtype=np.float32)
                    for idx, frameid in enumerate(
                            range(data_id, data_id + self.win_size * self.step, self.step)):
                        radar_npy_win[idx, :, :, :] = np.load(radar_paths[frameid])
                        data_dict['image_paths'].append(image_paths[frameid])
                elif radar_configs['data_type'] == 'RISEP' or radar_configs['data_type'] == 'APSEP':
                    if isinstance(chirp_id, int):
                        radar_npy_win = np.zeros((self.win_size, ramap_rsize, ramap_asize, 2), dtype=np.float32)
                        for idx, frameid in enumerate(
                                range(data_id, data_id + self.win_size * self.step, self.step)):
                            radar_npy_win[idx, :, :, :] = np.load(radar_paths[frameid][chirp_id])
                            data_dict['image_paths'].append(image_paths[frameid])
                    elif isinstance(chirp_id, list):
                        radar_npy_win = np.zeros((self.win_size, self.n_chirps, ramap_rsize, ramap_asize, 2), dtype=np.float32)
                        for idx, frameid in enumerate(
                                range(data_id, data_id + self.win_size * self.step, self.step)):
                            for cid, c in enumerate(chirp_id):
                                npy_path = radar_paths[frameid][c]
                                radar_npy_win[idx, cid, :, :, :] = np.load(npy_path)
                            data_dict['image_paths'].append(image_paths[frameid])
                    else:
                        raise TypeError
                elif radar_configs['data_type'] == 'ROD2021':
                    if isinstance(chirp_id, int):
                        radar_npy_win = np.zeros((self.win_size, ramap_rsize, ramap_asize, 2), dtype=np.float32)
                        for idx, frameid in enumerate(
                                range(data_id, data_id + self.win_size * self.step, self.step)):
                            radar_npy_win[idx, :, :, :] = np.load(radar_paths[frameid][chirp_id])
                            data_dict['image_paths'].append(image_paths[frameid])
                    elif isinstance(chirp_id, list):
                        radar_npy_win = np.zeros((self.win_size, self.n_chirps, ramap_rsize, ramap_asize, 2), dtype=np.float32)
                        for idx, frameid in enumerate(
                                range(data_id, data_id + self.win_size * self.step, self.step)):
                            for cid, c in enumerate(chirp_id):
                                npy_path = radar_paths[frameid][cid]
                                radar_npy_win[idx, cid, :, :, :] = np.load(npy_path)
                            data_dict['image_paths'].append(image_paths[frameid])
                    else:
                        raise TypeError
                else:
                    raise NotImplementedError

                data_dict['start_frame'] = data_id
                data_dict['end_frame'] = data_id + self.win_size * self.step - 1
                #print(f"############################ {data_dict['start_frame']} ~~~  {data_dict['end_frame']}")
            except:
                # in case load npy fail
                data_dict['status'] = False
                if not os.path.exists('./tmp'):
                    os.makedirs('./tmp')
                log_name = 'loadnpyfail-' + time.strftime("%Y%m%d-%H%M%S") + '.txt'
                with open(os.path.join('./tmp', log_name), 'w') as f_log:
                    f_log.write('npy path: ' + radar_paths[frameid][chirp_id] + \
                                '\nframe indices: %d:%d:%d' % (data_id, data_id + self.win_size * self.step, self.step))
                return data_dict

            # Dataloader for MNet
            if 'mnet_cfg' in self.model_cfg:
                radar_npy_win = np.transpose(radar_npy_win, (4, 0, 1, 2, 3))
                assert radar_npy_win.shape == (
                    2, self.win_size, self.n_chirps, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])
            else:
                radar_npy_win = np.transpose(radar_npy_win, (3, 0, 1, 2))
                assert radar_npy_win.shape == (2, self.win_size, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

            data_dict['radar_data'] = radar_npy_win

            # Load annotations
            if len(self.confmaps) != 0:
                confmap_gt = this_seq_confmap[data_id:data_id + self.win_size * self.step:self.step]
                confmap_gt = np.transpose(confmap_gt, (1, 0, 2, 3))
                obj_info = this_seq_obj_info[data_id:data_id + self.win_size * self.step:self.step]
                if self.noise_channel:
                    assert confmap_gt.shape == \
                        (self.n_class + 1, self.win_size, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])
                else:
                    confmap_gt = confmap_gt[:self.n_class]
                    assert confmap_gt.shape == \
                        (self.n_class, self.win_size, radar_configs['ramap_rsize'], radar_configs['ramap_asize'])

                data_dict['anno'] = dict(
                    obj_infos=obj_info,
                    confmaps=confmap_gt,
                )
            else:
                data_dict['anno'] = None

            return data_dict

    

def train(datafiles: list, features: list, model_config: dict, train_config: dict, pretrained: str=None):
    # # old single data file mode
    # for file in datafiles:
    #     dataset_factory = DatasetFactory()
    #     dataset_inst = dataset_factory.get_instance(file['parse'], file['id'])
    #     dataset_inst.parse(file['path'], file['name'], file['config'])
    #     # specify the features as train input data type
    #     dataset_inst.set_features(features)
    
    # new multiple data files mode
    dataset_inst_list = []
    for file in datafiles:
        dataset_factory = DatasetFactory()
        dataset_inst = dataset_factory.get_instance(file['parse'], file['id'])
        dataset_inst_list.append(dataset_inst)
    integrated_dataset = TrainDataset(dataset_inst_list, model_config, train_config, features)

    train_loader, val_loader, test_loader, train_ids, val_ids, test_ids = CreateDataLoaders(integrated_dataset, train_config['dataloader'], train_config['seed'])

    # Setup random seed
    torch.manual_seed(train_config['seed'])
    np.random.seed(train_config['seed'])
    random.seed(train_config['seed'])
    torch.cuda.manual_seed(train_config['seed'])

    # create experiment model name
    curr_date = datetime.now()
    exp_name = model_config['type'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
    print(exp_name)

    # save model path(also model name)
    with open("exp_info.txt", 'w') as f:
        f.write(exp_name)

    # Initialize tensorboard
    output_folder = Path(os.getenv('MODEL_ROOTDIR'))
    output_folder.mkdir(parents=True, exist_ok=True)
    (output_folder / exp_name).mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(output_folder / exp_name)

    # save model lineage info
    train_info_path = os.path.join(output_folder, exp_name, 'train_info.txt')
    with open(train_info_path, 'w') as f:
        json.dump({"datafiles": datafiles, "features": features, "model_config": model_config, "train_config": train_config}, f)
    # # read model lineage info
    # with open('data.json', 'r') as file:
    #     data = json.load(file)
    #     A_from_file = data["list_of_dicts"]
    #     B_from_file = data["single_dict"]


    # save sample split info
    split_info_path = os.path.join(output_folder, exp_name, 'samples_split.txt')
    with open(split_info_path, 'w') as f:
        f.write(f"TRAIN_SAMPLE_IDS: {','.join(map(str, train_ids))}\n")
        f.write(f"VAL_SAMPLE_IDS: {','.join(map(str, val_ids))}\n")
        f.write(f"TEST_SAMPLE_IDS: {','.join(map(str, test_ids))}\n")

    # save the evaluation of val dataset and test dataset
    val_eval_path = os.path.join(output_folder, exp_name, "val_eval.csv")
    test_eval_path = os.path.join(output_folder, exp_name, "test_eval.csv")
    df_val_eval = pd.DataFrame(columns=['Epoch', 'loss', 'mAP', 'mAR', 'mIoU'])

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    network_factory = NetworkFactory()
    model_type = model_config['class']
    model_config = model_config.copy()
    model_config.pop('class', None)
    print(model_type)
    print(model_config)
    net = network_factory.get_instance(model_type, model_config)
    print('network created')
    net.to(device)

    # Optimizer
    lr = float(train_config['optimizer']['lr'])
    step_size = int(train_config['lr_scheduler']['step_size'])
    gamma = float(train_config['lr_scheduler']['gamma'])
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    num_epochs=int(train_config['num_epochs'])

    print('===========  Optimizer  ==================:')
    print('      LR:', lr)
    print('      step_size:', step_size)
    print('      gamma:', gamma)
    print('      num_epochs:', num_epochs)
    print('')

    # Train
    startEpoch = 0
    global_step = 0
    history = {'train_loss':[],'val_loss':[],'lr':[],'mAP':[],'mAR':[],'mIoU':[]}
    best_mAP = 0

    freespace_loss = nn.BCEWithLogitsLoss(reduction='mean')

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


    for epoch in range(startEpoch,num_epochs):
        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)
        print(f'Epoch {epoch+1}/{num_epochs}')
        ###################
        ## Training loop ##
        ###################
        net.train()
        running_loss = 0.0

        for i, data in enumerate(train_loader):
            if model_type == "FFTRadNet":
                inputs = data[0].to(device).float()
                label_map = data[1].to(device).float()
                if(model_config['segmentation_head']=='True'):
                    seg_map_label = data[2].to(device).double()
            elif model_type == "RODNet":
                inputs = data['radar_data'].to(device).float()
                confmap_gt = data['anno']['confmaps'].to(device).float()

            # reset the gradient
            optimizer.zero_grad()
            
            # forward pass, enable to track our gradient
            with torch.set_grad_enabled(True):
                outputs = net(inputs)

            if model_type == "FFTRadNet":
                classif_loss,reg_loss = pixor_loss(outputs['Detection'], label_map, train_config['losses'])           
                
                prediction = outputs['Segmentation'].contiguous().flatten()
                label = seg_map_label.contiguous().flatten()        
                loss_seg = freespace_loss(prediction, label)
                loss_seg *= inputs.size(0)

                classif_loss *= train_config['losses']['weight'][0]
                reg_loss *= train_config['losses']['weight'][1]
                loss_seg *= train_config['losses']['weight'][2]


                loss = classif_loss + reg_loss + loss_seg

                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Loss/train_clc', classif_loss.item(), global_step)
                writer.add_scalar('Loss/train_reg', reg_loss.item(), global_step)
                writer.add_scalar('Loss/train_freespace', loss_seg.item(), global_step)
            elif model_type == "RODNet":
                criterion = nn.BCELoss()
                if model_config['stacked_num'] is not None:
                    for i in range(model_config['stacked_num']):
                        loss_cur = criterion(outputs[i], confmap_gt)
                        loss += loss_cur
                else:
                    loss = criterion(outputs, confmap_gt)

            # backprop
            loss.backward()
            optimizer.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
        
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
        if model_type == "FFTRadNet":
            eval = run_evaluation(net,val_loader, check_perf=(epoch>=10), detection_loss=pixor_loss, 
                                    segmentation_loss=freespace_loss, losses_params=train_config['losses'],
                                    device=device)
        elif model_type == "RODNet":
            eval = RODNet_evaluation(net, val_loader, integrated_dataset.data_root, os.path.join(output_folder, exp_name), 
                                     train_config, model_config, integrated_dataset.sensor_cfg['radar_cfg'], device)
            
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
        filename = output_folder / exp_name / name_output_file

        checkpoint={}
        checkpoint['net_state_dict'] = net.state_dict()
        checkpoint['optimizer'] = optimizer.state_dict()
        checkpoint['scheduler'] = scheduler.state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['history'] = history
        checkpoint['global_step'] = global_step

        torch.save(checkpoint,filename)

    df_val_eval.to_csv(val_eval_path, index=False)
    run_FullEvaluation(net, test_loader, test_eval_path, device=device)

    return 


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