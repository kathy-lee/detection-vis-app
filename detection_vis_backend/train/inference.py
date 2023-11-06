import numpy as np
import pandas as pd
import pkbar
import torch
import time
import os
import math
import json
import cv2
import polarTransform
import torch.nn as nn

from shapely.geometry import Polygon
from scipy.stats import hmean
from sklearn.metrics import confusion_matrix
from PIL import Image

from detection_vis_backend.datasets.dataset import DatasetFactory
from detection_vis_backend.networks.network import NetworkFactory
from detection_vis_backend.datasets.utils import confmap2ra
from detection_vis_backend.train.utils import post_process_single_frame, get_class_name, worldToImage, decode, process_predictions_FFT


def infer(model, checkpoint_id, sample_id, file_id, split_type):
    model_rootdir = os.getenv('MODEL_ROOTDIR')
    parameter_path = os.path.join(model_rootdir, model, "train_info.txt")
    with open(parameter_path, 'r') as f:
        parameters = json.load(f)
    if not parameters:
        raise ValueError("Parameters are empty")

    # Get input data and groundtruth label info
    dataset_factory = DatasetFactory()
    dataset_type = parameters["datafiles"][0]["parse"]
    dataset_inst = dataset_factory.get_instance(dataset_type, file_id)
    #dataset_inst.parse(file_id, parameters["datafiles"][0]["path"], parameters["datafiles"][0]["name"], parameters["datafiles"][0]["config"])
    dataset_inst.prepare_for_train(parameters["features"], parameters["train_config"], parameters["model_config"], split_type)
    data = dataset_inst[sample_id]
    gt_labels = dataset_inst.get_label(parameters["features"], sample_id)

    # Initialize the model
    model_config = parameters["model_config"]
    network_factory = NetworkFactory()
    model_type = model_config['class']
    model_config.pop('class', None)
    net = network_factory.get_instance(model_type, model_config)

    # set device
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    # Load the model checkpoint
    model_rootdir = os.getenv('MODEL_ROOTDIR')
    model_path = os.path.join(model_rootdir, model)
    checkpoint = [file for file in os.listdir(model_path) if f"epoch{checkpoint_id:02}" in file][0]
    dict = torch.load(os.path.join(model_path, checkpoint), map_location=device)
    net.load_state_dict(dict['net_state_dict'])  
    net.to(device)
    
    # Prediction
    net.eval()
    with torch.set_grad_enabled(False):
        if model_type == "FFTRadNet":
            # input_data: [radar_FFT, segmap,out_label,box_labels,image]
            input = torch.tensor(data[0]).unsqueeze(0).to(device).float()
            output = net(input)
            pred_image = display_inference_FFTRadNet(data[4], data[0], output, data[3])
            feature_show_pred = "image"
        elif model_type in ("RODNet_CDC", "RODNet_CDCv2", "RODNet_HG", "RODNet_HGv2", "RODNet_HGwI", "RODNet_HGwIv2", "RadarFormer_hrformer2d"):
            input = torch.tensor(data['radar_data']).unsqueeze(0).to(device).float()
            output = net(input)
            if 'stacked_num' in model_config:
                confmap_pred = output[-1].cpu().detach().numpy()  # (1, 4, 32, 128, 128)
            else:
                confmap_pred = output.cpu().detach().numpy()
            pred_objs = post_process_single_frame(confmap_pred[0,:,0,:,:], parameters["train_config"], dataset_inst.n_class, dataset_inst.rng_grid, dataset_inst.agl_grid) #[B, win_size, max_dets, 4]
            # filter invalid predictions
            mask = pred_objs[:, 0] != -1 
            pred_objs = pred_objs[mask]
            # limit conf value
            mask = pred_objs[:, -1] > 1  
            pred_objs[mask, -1] = 1
            # reorganize the items: [row_id, col_id, cls_id, conf_value]
            pred_objs = pred_objs[:, [1, 2, 0, 3]]
            feature_show_pred = "RA"
        # elif model_type in ("RECORD", "RECORDNoLstm", "RECORDNoLstmMulti") and dataset_type == "CRUW":
        #     input = torch.tensor(data['radar']).unsqueeze(0).to(device).float()
        #     output = net(input)
        #     confmap_pred = output[0].cpu().detach().numpy()
        #     pred_image = infer_CRUW(data['image_paths'][0], input, confmap_pred, gt_labels, parameters["train_config"])
        # elif model_type == "RECORD" and dataset_type == "CARRADA":
        #     input = torch.tensor(data['radar']).unsqueeze(0).to(device).float()
        #     output = net(input)
        #     pred_image = display_inference_CARRADA(data['image_path'], input, output, gt_labels)
        # elif model_type == "MVRECORD":
        #     input = (torch.tensor(data['rd_matrix']).unsqueeze(0).to(device).float(), 
        #              torch.tensor(data['ra_matrix']).unsqueeze(0).to(device).float(), 
        #              torch.tensor(data['ad_matrix']).unsqueeze(0).to(device).float())
        #     output = net(input)
        #     pred_image = display_inference_CARRADA(data['image_path'], input, output, gt_labels)
        # elif model_type == "RADDet" or model_type == "DAROD":
        #     input = torch.tensor(data['radar']).unsqueeze(0).to(device).float()
        #     output = net(input)
        #     pred_image = display_inference_RADDetDataset(data['image_path'], input, output, gt_labels)
        else:
            raise ValueError("Inference of the chosen model type is not supported")

    return pred_objs, feature_show_pred


def display_inference_FFTRadNet(image, input, model_outputs, obj_labels, train_config=None):
    # Model outputs
    pred_obj = model_outputs['Detection'].detach().cpu().numpy().copy()[0]
    out_seg = torch.sigmoid(model_outputs['Segmentation']).detach().cpu().numpy().copy()[0,0]
    print(f"pred_obj before decode: {pred_obj.shape}")
    # Decode the output detection map
    pred_obj = decode(pred_obj,0.05)
    print(f"pred_obj after decode: {type(pred_obj)}, {len(pred_obj)}")
    pred_obj = np.asarray(pred_obj)

    # process prediction: polar to cartesian, NMS...
    if(len(pred_obj)>0):
        pred_obj = process_predictions_FFT(pred_obj,confidence_threshold=0.2)

    ## FFT
    FFT = np.abs(input[...,:16]+input[...,16:]*1j).mean(axis=2)
    PowerSpectrum = np.log10(FFT)
    # rescale
    PowerSpectrum = (PowerSpectrum -PowerSpectrum.min())/(PowerSpectrum.max()-PowerSpectrum.min())*255
    PowerSpectrum = cv2.cvtColor(PowerSpectrum.astype('uint8'),cv2.COLOR_GRAY2BGR)
    ## Image
    for box in pred_obj:
        box = box[1:]
        u1,v1 = worldToImage(-box[2],box[1],0)
        u2,v2 = worldToImage(-box[0],box[1],1.6)

        u1 = int(u1/2)
        v1 = int(v1/2)
        u2 = int(u2/2)
        v2 = int(v2/2)

        image = cv2.rectangle(image, (u1,v1), (u2,v2), (0, 0, 255), 3)
    
    for box in obj_labels:
        box = box[6:]
        box = [int(x/2) for x in box]
        image = cv2.rectangle(image, (box[0],box[1]), (box[2],box[3]), (255, 0, 0), 2)

    RA_cartesian,_=polarTransform.convertToCartesianImage(np.moveaxis(out_seg,0,1),useMultiThreading=True,
        initialAngle=0, finalAngle=np.pi,order=0,hasColor=False)
    # Make a crop on the angle axis
    RA_cartesian = RA_cartesian[:,256-100:256+100]
    
    RA_cartesian = np.asarray((RA_cartesian*255).astype('uint8'))
    RA_cartesian = cv2.cvtColor(RA_cartesian, cv2.COLOR_GRAY2BGR)
    RA_cartesian = cv2.resize(RA_cartesian,dsize=(400,512))
    RA_cartesian=cv2.flip(RA_cartesian,flipCode=-1)
    return np.hstack((PowerSpectrum,image[:512],RA_cartesian))
    




def display_inference_CARRADA(image_path, model_input, model_output, gt_labels, train_config):
    return None
   

def display_inference_RADDetDataset(image_path, model_input, model_output, gt_labels, train_config):
    # _, pred = boxDecoder(model_output, train_config['input_size'], train_config['anchor_boxes'], model_config['num_class'], train_config['yolohead_xyz_scales'][0], device)
    # predicitons = yoloheadToPredictions(pred[0], conf_threshold=train_config["confidence_threshold"])
    # det = nms(predicitons, train_config["nms_iou3d_threshold"], train_config["input_size"], sigma=0.3, method="nms")
    return None