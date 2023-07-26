import torch
import random
import numpy as np
import pkbar
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn

from metaflow import FlowSpec, step, Parameter
from pathlib import Path
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
from model.FFTRadNet import FFTRadNet
from dataset.dataset import RADIal
from dataset.encoder import ra_encoder
from dataset.dataloader import CreateDataLoaders
from torch.optim import lr_scheduler
from loss import pixor_loss
from utils.evaluation import run_evaluation
from utils import TrainDataset




class TrainModelFlow(FlowSpec):

    datafiles = Parameter('datafiles',
                          help="IDs of chosen data files",
                          required=True,
                          type=list)
    features= Parameter('features',
                        help="Chosen features",
                        required=True,
                        type=list)
    model_config = Parameter('model_config', 
                             help="Path to the model configuration file.",
                             required=True,
                             type=dict)
    train_config = Parameter('train_config', 
                             help="Path to the train configuration file.",
                             required=True,
                             type=dict)

    @step
    def start(self):
        

        self.next(self.split_data)

    @step
    def split_data(self):
        
        self.next(self.train_model)

    @step
    def train_model(self):
        # Get input data for training
        if len(self.datafiles) == 1: 
            datafile = crud.get_datafile(db, file["id"])
            dataset_factory = DatasetFactory()
            dataset_inst = dataset_factory.get_instance(datafile.parse, file["id"])
            train_data = dataset_inst
        else:
            for file in self.datafiles:
                datafile = crud.get_datafile(db, file["id"])
                dataset_factory = DatasetFactory()
                dataset_inst = dataset_factory.get_instance(datafile.parse, file["id"])
                for feature in self.features:
                    function_dict = {
                        'RAD': dataset_inst.get_RAD,
                        'RD': dataset_inst.get_RD,
                        'RA': dataset_inst.get_RA,
                        'spectrogram': dataset_inst.get_spectrogram,
                        'radarPC': dataset_inst.get_radarpointcloud,
                        'lidarPC': dataset_inst.get_lidarpointcloud,
                        'image': dataset_inst.get_image,
                        'depth_image': dataset_inst.get_depthimage,
                    }
                    feature_data = function_dict[feature]()
                    train_data[feature].append(feature_data)
    
        # Load the dataset
        # enc = ra_encoder(geometry = config['dataset']['geometry'], 
        #                     statistics = config['dataset']['statistics'],
        #                     regression_layer = 2)
        
        # dataset = RADIal(root_dir = config['dataset']['root_dir'],
        #                     statistics= config['dataset']['statistics'],
        #                     encoder=enc.encode,
        #                     difficult=True)

        train_loader, val_loader, test_loader = CreateDataLoaders(train_data, self.train_config)

        # Setup random seed
        torch.manual_seed(self.train_config['seed'])
        np.random.seed(self.train_config['seed'])
        random.seed(self.train_config['seed'])
        torch.cuda.manual_seed(self.train_config['seed'])

        # create experiment model name
        curr_date = datetime.now()
        exp_name = self.model_config['type'] + self.datafiles['parse'] + '___' + curr_date.strftime('%b-%d-%Y___%H:%M:%S')
        print(exp_name)

        # Initialize tensorboard
        output_folder = Path("/home/kangle/dataset/trained_models")
        output_folder.mkdir(parents=True, exist_ok=True)
        (output_folder / exp_name).mkdir(parents=True, exist_ok=True)
        writer = SummaryWriter(output_folder / exp_name)

        # set device
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        
        network_factory = NetworkFactory()
        net = network_factory.get_instance(self.model_config['type'])
        net.to('cuda')

        # Optimizer
        lr = float(self.train_config['optimizer']['lr'])
        step_size = int(self.train_config['lr_scheduler']['step_size'])
        gamma = float(self.train_config['lr_scheduler']['gamma'])
        optimizer = optim.Adam(filter(lambda p: p.requires_grad, net.parameters()), lr=lr)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
        num_epochs=int(self.train_config['num_epochs'])


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


        if resume:
            print('===========  Resume training  ==================:')
            dict = torch.load(resume)
            net.load_state_dict(dict['net_state_dict'])
            optimizer.load_state_dict(dict['optimizer'])
            scheduler.load_state_dict(dict['scheduler'])
            startEpoch = dict['epoch']+1
            history = dict['history']
            global_step = dict['global_step']

            print('       ... Start at epoch:',startEpoch)


        for epoch in range(startEpoch,num_epochs):
            
            kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=20, always_stateful=False)
            
            ###################
            ## Training loop ##
            ###################
            net.train()
            running_loss = 0.0
            
            for i, data in enumerate(train_loader):
                inputs = data[0].to('cuda').float()
                label_map = data[1].to('cuda').float()
                if(self.model_config['SegmentationHead']=='True'):
                    seg_map_label = data[2].to('cuda').double()

                # reset the gradient
                optimizer.zero_grad()
                
                # forward pass, enable to track our gradient
                with torch.set_grad_enabled(True):
                    outputs = net(inputs)


                classif_loss,reg_loss = pixor_loss(outputs['Detection'], label_map, self.train_config['losses'])           
                
                prediction = outputs['Segmentation'].contiguous().flatten()
                label = seg_map_label.contiguous().flatten()        
                loss_seg = freespace_loss(prediction, label)
                loss_seg *= inputs.size(0)

                classif_loss *= self.train_config['losses']['weight'][0]
                reg_loss *= self.train_config['losses']['weight'][1]
                loss_seg *=self.train_config['losses']['weight'][2]


                loss = classif_loss + reg_loss + loss_seg

                writer.add_scalar('Loss/train', loss.item(), global_step)
                writer.add_scalar('Loss/train_clc', classif_loss.item(), global_step)
                writer.add_scalar('Loss/train_reg', reg_loss.item(), global_step)
                writer.add_scalar('Loss/train_freespace', loss_seg.item(), global_step)

                # backprop
                loss.backward()
                optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
            
                kbar.update(i, values=[("loss", loss.item()), ("class", classif_loss.item()), ("reg", reg_loss.item()),("freeSpace", loss_seg.item())])

                
                global_step += 1


            scheduler.step()

            history['train_loss'].append(running_loss / len(train_loader.dataset))
            history['lr'].append(scheduler.get_last_lr()[0])

            
            ######################
            ## validation phase ##
            ######################

            eval = run_evaluation(net,val_loader,enc,check_perf=(epoch>=10),
                                    detection_loss=pixor_loss,segmentation_loss=freespace_loss,
                                    losses_params=self.train_config['losses'])

            history['val_loss'].append(eval['loss'])
            history['mAP'].append(eval['mAP'])
            history['mAR'].append(eval['mAR'])
            history['mIoU'].append(eval['mIoU'])

            kbar.add(1, values=[("val_loss", eval['loss']),("mAP", eval['mAP']),("mAR", eval['mAR']),("mIoU", eval['mIoU'])])


            writer.add_scalar('learning_rate', optimizer.param_groups[0]['lr'], global_step)
            writer.add_scalar('Loss/test', eval['loss'], global_step)
            writer.add_scalar('Metrics/mAP', eval['mAP'], global_step)
            writer.add_scalar('Metrics/mAR', eval['mAR'], global_step)
            writer.add_scalar('Metrics/mIoU', eval['mIoU'], global_step)

            # Saving all checkpoint as the best checkpoint for multi-task is a balance between both --> up to the user to decide
            name_output_file = self.model_config['type'] + self.datafiles['parse'] +'_epoch{:02d}_loss_{:.4f}_AP_{:.4f}_AR_{:.4f}_IOU_{:.4f}.pth'.format(epoch, eval['loss'],eval['mAP'],eval['mAR'],eval['mIoU'])
            filename = output_folder / exp_name / name_output_file

            checkpoint={}
            checkpoint['net_state_dict'] = net.state_dict()
            checkpoint['optimizer'] = optimizer.state_dict()
            checkpoint['scheduler'] = scheduler.state_dict()
            checkpoint['epoch'] = epoch
            checkpoint['history'] = history
            checkpoint['global_step'] = global_step

            torch.save(checkpoint,filename)
            
            print('')

            self.next(self.evaluate_model)

    @step
    def evaluate_model(self):
        self.accuracy = self.model.score(self.test.drop(['target'], axis=1), self.test['target'])
        print('Model Accuracy:', self.accuracy)
        self.next(self.end)

    @step
    def end(self):
        print("Model Training Complete.")
        with open("model.pkl", "wb") as f:
            pickle.dump(self.model, f)
        print("Model Saved.")



