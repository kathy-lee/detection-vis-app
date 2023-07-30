# import torch
# import random
# import numpy as np
# import pkbar
# import torch.optim as optim
# import torch.nn.functional as F
# import torch.nn as nn
# import os
import logging
import subprocess
import json

from metaflow import FlowSpec, Parameter, step
from metaflow import Metaflow, FlowSpec, Parameter, step, IncludeFile
from metaflow.cli_args import cli_args
# from pathlib import Path
# from datetime import datetime
# from torch.utils.tensorboard import SummaryWriter
# from torch.optim import lr_scheduler

# from .utils import CreateDataLoaders, run_evaluation, pixor_loss
# from data import crud
# from data.database import SessionLocal
# from detection_vis_backend.datasets.dataset import DatasetFactory
# from ..networks.network import NetworkFactory



# # Dependency
# def get_db():
#     db = SessionLocal()
#     try:
#         yield db
#     finally:
#         db.close()


class TrainModelFlow(FlowSpec):
    datafiles_str = Parameter('datafiles',
                          help="IDs of chosen data files",
                          required=True,
                          type=str)
    features_str = Parameter('features',
                        help="Chosen features",
                        required=True,
                        type=str)
    model_config_str = Parameter('model_config', 
                             help="Path to the model configuration file.",
                             required=True,
                             type=str)
    train_config_str = Parameter('train_config', 
                             help="Path to the train configuration file.",
                             required=True,
                             type=str)

    @step
    def start(self):
        logging.info("Training begins.")
        self.datafiles = json.loads(self.datafiles_str)
        self.features = json.loads(self.features_str)
        self.model_config = json.loads(self.model_config_str)
        self.train_config = json.loads(self.train_config_str)
        print(self.datafiles)
        print(self.features)
        print(self.model_config)
        print(self.train_config)
        self.next(self.train_model)

    @step
    def train_model(self):
        # Create data loaders 
        train_loaders = []
        val_loaders = []
        test_loaders = []
    #     # with get_db() as db:
    #     #     for file in self.datafiles:
    #     #         datafile = crud.get_datafile(db, file["id"])
    #     #         dataset_factory = DatasetFactory()
    #     #         dataset_inst = dataset_factory.get_instance(datafile.parse, file["id"])
    #     #         # specify the features as train input data type
    #     #         #dataset_inst.set_feature(self.features)
    #     #         train_loader, val_loader, test_loader = CreateDataLoaders(dataset_inst, self.train_config)
    #     #         train_loaders.append(train_loader)
    #     #         val_loaders.append(val_loader)
    #     #         test_loaders.append(test_loader)
            
    #     # # Setup random seed
    #     # torch.manual_seed(self.train_config['seed'])
    #     # np.random.seed(self.train_config['seed'])
    #     # random.seed(self.train_config['seed'])
    #     # torch.cuda.manual_seed(self.train_config['seed'])

    #     # # create experiment model name
    #     # curr_date = datetime.now()
    #     # exp_name = self.model_config['type'] + self.datafiles['parse'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
    #     # print(exp_name)

    #     # # Initialize tensorboard
    #     # output_folder = Path("/home/kangle/dataset/trained_models")
    #     # output_folder.mkdir(parents=True, exist_ok=True)
    #     # (output_folder / exp_name).mkdir(parents=True, exist_ok=True)
    #     # writer = SummaryWriter(output_folder / exp_name)

    #     # # save model path
    #     # self.model_path = os.path.join(output_folder, exp_name)

    #     # # set device
    #     # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
    #     # network_factory = NetworkFactory()
    #     # net = network_factory.get_instance(self.model_config['type'])
    #     # net.to('cuda')

    #     # # Optimizer
    #     # lr = float(self.train_config['optimizer']['lr'])
    #     # step_size = int(self.train_config['lr_scheduler']['step_size'])
    #     # gamma = float(self.train_config['lr_scheduler']['gamma'])
    #     # optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
    #     # scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    #     # num_epochs=int(self.train_config['num_epochs'])


    #     # print('===========  Optimizer  ==================:')
    #     # print('      LR:', lr)
    #     # print('      step_size:', step_size)
    #     # print('      gamma:', gamma)
    #     # print('      num_epochs:', num_epochs)
    #     # print('')

    #     # # Train
    #     # startEpoch = 0
    #     # global_step = 0
    #     # history = {'train_loss':[],'val_loss':[],'lr':[],'mAP':[],'mAR':[],'mIoU':[]}
    #     # best_mAP = 0

    #     # freespace_loss = nn.BCEWithLogitsLoss(reduction='mean')


    #     # # if resume:
    #     # #     print('===========  Resume training  ==================:')
    #     # #     dict = torch.load(resume)
    #     # #     net.load_state_dict(dict['net_state_dict'])
    #     # #     optimizer.load_state_dict(dict['optimizer'])
    #     # #     scheduler.load_state_dict(dict['scheduler'])
    #     # #     startEpoch = dict['epoch']+1
    #     # #     history = dict['history']
    #     # #     global_step = dict['global_step']
    #     # #     print('       ... Start at epoch:',startEpoch)


    #     # for epoch in range(startEpoch,num_epochs):
    #     #     batch_count = sum(len(loader) for loader in train_loaders)
    #     #     kbar = pkbar.Kbar(target=batch_count, epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)
            
    #     #     ###################
    #     #     ## Training loop ##
    #     #     ###################
    #     #     net.train()
    #     #     running_loss = 0.0
            
    #     #     length_total = 0
    #     #     for train_loader in train_loaders:
    #     #         length_total += len(train_loader.dataset)

    #     #         for i, data in enumerate(train_loader):
    #     #             inputs = data[0].to('cuda').float()
    #     #             label_map = data[1].to('cuda').float()
    #     #             if(self.model_config['SegmentationHead']=='True'):
    #     #                 seg_map_label = data[2].to('cuda').double()

    #     #             # reset the gradient
    #     #             optimizer.zero_grad()
                    
    #     #             # forward pass, enable to track our gradient
    #     #             with torch.set_grad_enabled(True):
    #     #                 outputs = net(inputs)


    #     #             classif_loss,reg_loss = pixor_loss(outputs['Detection'], label_map, self.train_config['losses'])           
                    
    #     #             prediction = outputs['Segmentation'].contiguous().flatten()
    #     #             label = seg_map_label.contiguous().flatten()        
    #     #             loss_seg = freespace_loss(prediction, label)
    #     #             loss_seg *= inputs.size(0)

    #     #             classif_loss *= self.train_config['losses']['weight'][0]
    #     #             reg_loss *= self.train_config['losses']['weight'][1]
    #     #             loss_seg *=self.train_config['losses']['weight'][2]


    #     #             loss = classif_loss + reg_loss + loss_seg

    #     #             writer.add_scalar('Loss/train', loss.item(), global_step)
    #     #             writer.add_scalar('Loss/train_clc', classif_loss.item(), global_step)
    #     #             writer.add_scalar('Loss/train_reg', reg_loss.item(), global_step)
    #     #             writer.add_scalar('Loss/train_freespace', loss_seg.item(), global_step)

    #     #             # backprop
    #     #             loss.backward()
    #     #             optimizer.step()

    #     #             # statistics
    #     #             running_loss += loss.item() * inputs.size(0)
                
    #     #             kbar.update(i, values=[("loss", loss.item()), ("class", classif_loss.item()), ("reg", reg_loss.item()),("freeSpace", loss_seg.item())])
    
    #     #             global_step += 1


    #     #     scheduler.step()

    #     #     history['train_loss'].append(running_loss / length_total)
    #     #     history['lr'].append(scheduler.get_last_lr()[0])

            
    #     #     ######################
    #     #     ## validation phase ##
    #     #     ######################
    #     #     eval = {}
    #     #     for val_loader in val_loaders: 
    #     #         eval_single = run_evaluation(net,val_loader,enc,check_perf=(epoch>=10),
    #     #                                 detection_loss=pixor_loss,segmentation_loss=freespace_loss,
    #     #                                 losses_params=self.train_config['losses'])
    #     #         eval['loss'] += eval_single['loss']
    #     #         eval['mAP'] += eval_single['mAP']
    #     #         eval['mAR'] += eval_single['mAR']
    #     #         eval['mIoU'] += eval_single['mIoU']
            
    #     #     eval['mAP'] /= len(val_loaders)
    #     #     eval['mAR'] /= len(val_loaders)
    #     #     eval['mIoU'] /= len(val_loaders)
                

    #     #     history['val_loss'].append(eval['loss'])
    #     #     history['mAP'].append(eval['mAP'])
    #     #     history['mAR'].append(eval['mAR'])
    #     #     history['mIoU'].append(eval['mIoU'])

    #     #     kbar.add(1, values=[("val_loss", eval['loss']),("mAP", eval['mAP']),("mAR", eval['mAR']),("mIoU", eval['mIoU'])])


    #     #     writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
    #     #     writer.add_scalar('Loss/test', eval['loss'], global_step)
    #     #     writer.add_scalar('Metrics/mAP', eval['mAP'], global_step)
    #     #     writer.add_scalar('Metrics/mAR', eval['mAR'], global_step)
    #     #     writer.add_scalar('Metrics/mIoU', eval['mIoU'], global_step)

    #     #     # Saving all checkpoint as the best checkpoint for multi-task is a balance between both --> up to the user to decide
    #     #     name_output_file = self.model_config['type'] + self.datafiles['parse'] +'_epoch{:02d}_loss_{:.4f}_AP_{:.4f}_AR_{:.4f}_IOU_{:.4f}.pth'.format(epoch, eval['loss'],eval['mAP'],eval['mAR'],eval['mIoU'])
    #     #     filename = output_folder / exp_name / name_output_file

    #     #     checkpoint={}
    #     #     checkpoint['net_state_dict'] = net.state_dict()
    #     #     checkpoint['optimizer'] = optimizer.state_dict()
    #     #     checkpoint['scheduler'] = scheduler.state_dict()
    #     #     checkpoint['epoch'] = epoch
    #     #     checkpoint['history'] = history
    #     #     checkpoint['global_step'] = global_step

    #     #     torch.save(checkpoint,filename)
            
    #     #     print('')

        self.next(self.end)

    @step
    def end(self):

        print("Training ends sucessfully.")

    # def run_flow(self):
    #     from metaflow import get_metadata
    #     print("Using metaflow version %s" % Metaflow().version)

    #     print("Using metadata provider: %s" % get_metadata())
    #     self.datastore(None, mode='local')
    #     self.environment(None, mode='conda')

    #     self.run(id='TrainModelFlow',
    #              metadata=None,
    #              datastore=None,
    #              environment=None,
    #              logger=logging.getLogger(),
    #              graph_learnings=None,
    #              termination_value=None,
    #              attempt=0)


if __name__ == '__main__':
    TrainModelFlow()