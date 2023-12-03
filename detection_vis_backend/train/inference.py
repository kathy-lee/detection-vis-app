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
from detection_vis_backend.train.utils import post_process_single_frame, get_class_name, worldToImage, decode, process_predictions_FFT, yoloheadToPredictions, boxDecoder, nms, create_default, peaks_detect, distribute, update_peaks, association, pol2cord, orent


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
            pred_obj = output['Detection'].detach().cpu().numpy().copy()[0]
            pred_obj = decode(pred_obj,0.05)
            pred_obj = np.asarray(pred_obj)
            # process prediction: polar to cartesian, NMS...
            if(len(pred_obj)>0):
                pred_obj = process_predictions_FFT(pred_obj,confidence_threshold=0.2)
            pred_objs = []
            for box in pred_obj:
                box = box[1:]
                u1,v1 = worldToImage(-box[2],box[1],0)
                u2,v2 = worldToImage(-box[0],box[1],1.6)
                u1 = int(u1/2)
                v1 = int(v1/2)
                u2 = int(u2/2)
                v2 = int(v2/2)
                pred_objs.append([u1, v1, u2, v2])
            #feature_show_pred = "image"
            pred_objs = {"image": pred_objs}
        elif model_type in ("RODNet_CDC", "RODNet_CDCv2", "RODNet_HG", "RODNet_HGv2", "RODNet_HGwI", "RODNet_HGwIv2", "RadarFormer_hrformer2d"):
            input = torch.tensor(data['radar_data']).unsqueeze(0).to(device).float()
            output = net(input)
            if 'stacked_num' in model_config:
                confmap_pred = output[-1].detach().cpu().numpy()  # (1, 4, 32, 128, 128)
            else:
                confmap_pred = output.detach().cpu().numpy()
            pred_objs = post_process_single_frame(confmap_pred[0,:,0,:,:], parameters["train_config"], dataset_inst.n_class, dataset_inst.rng_grid, dataset_inst.agl_grid) #[B, win_size, max_dets, 4]
            # filter invalid predictions
            mask = pred_objs[:, 0] != -1 
            pred_objs = pred_objs[mask]
            # limit conf value
            mask = pred_objs[:, -1] > 1  
            pred_objs[mask, -1] = 1
            # reorganize the items: [row_id, col_id, cls_id, conf_value]
            pred_objs = pred_objs[:, [1, 2, 0, 3]]
            #feature_show_pred = "RA"
            pred_objs = {"RA": pred_objs.tolist()}
        elif model_type in ("RECORD", "RECORDNoLstm", "RECORDNoLstmMulti") and dataset_type == "CRUW":
            input = torch.tensor(data['radar_data']).unsqueeze(0).to(device).float()
            output = net(input)
            confmap_pred = output[0].detach().cpu().numpy()
            pred_objs = post_process_single_frame(confmap_pred, parameters["train_config"], dataset_inst.n_class, dataset_inst.rng_grid, dataset_inst.agl_grid) #[B, win_size, max_dets, 4]
            # filter invalid predictions
            mask = pred_objs[:, 0] != -1 
            pred_objs = pred_objs[mask]
            # limit conf value
            mask = pred_objs[:, -1] > 1  
            pred_objs[mask, -1] = 1
            # reorganize the items: [row_id, col_id, cls_id, conf_value]
            pred_objs = pred_objs[:, [1, 2, 0, 3]]
            #feature_show_pred = "RA"
            pred_objs = {"RA": pred_objs.tolist()}
        elif model_type == "RECORD" and dataset_type == "CARRADA":
            input = torch.tensor(data['radar']).unsqueeze(0).to(device).float()
            output = net(input)
            confmap_pred = output[0].detach().cpu().numpy()
            feature = parameters["features"][0]
            if feature == "RA":
                pred_objs = post_process_single_frame(confmap_pred, parameters["train_config"], dataset_inst.n_class, dataset_inst.rng_grid, dataset_inst.agl_grid) #[B, win_size, max_dets, 4]
            elif feature == "RD":
                pred_objs = post_process_single_frame(confmap_pred, parameters["train_config"], dataset_inst.n_class, dataset_inst.rng_grid, dataset_inst.dpl_grid) #[B, win_size, max_dets, 4]
            else:
                raise ValueError("Feature type not supported in inference.")
            # filter invalid predictions
            mask = pred_objs[:, 0] != -1 
            pred_objs = pred_objs[mask]
            # limit conf value
            mask = pred_objs[:, -1] > 1  
            pred_objs[mask, -1] = 1
            # reorganize the items: [row_id, col_id, cls_id, conf_value]
            pred_objs = pred_objs[:, [1, 2, 0, 3]]
            pred_objs = {feature: pred_objs.tolist()}
        elif model_type == "MVRECORD":
            input = (torch.tensor(data['rd_matrix']).unsqueeze(0).to(device).float(), 
                     torch.tensor(data['ra_matrix']).unsqueeze(0).to(device).float(), 
                     torch.tensor(data['ad_matrix']).unsqueeze(0).to(device).float())
            output = net(input)
            confmap_pred_ra = output['ra'].detach().cpu().numpy()
            confmap_pred_rd = output['rd'].detach().cpu().numpy()
            pred_objs_ra = post_process_single_frame(confmap_pred_ra, parameters["train_config"], dataset_inst.n_class, dataset_inst.rng_grid, dataset_inst.agl_grid) #[B, win_size, max_dets, 4]
            pred_objs_rd = post_process_single_frame(confmap_pred_rd, parameters["train_config"], dataset_inst.n_class, dataset_inst.rng_grid, dataset_inst.dpl_grid) #[B, win_size, max_dets, 4]
            pred_objs = {"RA": pred_objs_ra, "RD": pred_objs_rd}
            # filter invalid predictions
            mask = pred_objs["RA"][:, 0] != -1 
            pred_objs["RA"] = pred_objs["RA"][mask]
            # limit conf value
            mask = pred_objs["RA"][:, -1] > 1  
            pred_objs["RA"][mask, -1] = 1
            # reorganize the items: [row_id, col_id, cls_id, conf_value]
            pred_objs["RA"] = pred_objs["RA"][:, [1, 2, 0, 3]]
            # filter invalid predictions
            mask = pred_objs["RD"][:, 0] != -1 
            pred_objs["RD"] = pred_objs["RD"][mask]
            # limit conf value
            mask = pred_objs["RD"][:, -1] > 1  
            pred_objs["RD"][mask, -1] = 1
            # reorganize the items: [row_id, col_id, cls_id, conf_value]
            pred_objs["RD"] = pred_objs["RD"][:, [1, 2, 0, 3]]
            pred_objs = {"RA": pred_objs_ra.tolist(), "RD": pred_objs_rd.tolist()}
        elif model_type == "RADDet":
            input = torch.tensor(data['radar']).unsqueeze(0).to(device).float()
            output = net(input)
            train_config = parameters["train_config"]
            _, pred = boxDecoder(output, train_config['input_size'], train_config['anchor_boxes'], model_config['n_class'], train_config['yolohead_xyz_scales'][0], device)
            pred = pred.detach().cpu()
            predicitons = yoloheadToPredictions(pred[0], conf_threshold=train_config["confidence_threshold"])
            pred_objs = nms(predicitons, train_config["nms_iou3d_threshold"], train_config["input_size"], sigma=0.3, method="nms")
            rd_objs = []
            ra_objs = []
            for i in range(len(pred_objs)):
                bbox3d = pred_objs[i, :6]
                cls = int(pred_objs[i, 7])
                rd_objs.append([bbox3d[0], bbox3d[2], bbox3d[3], bbox3d[5], cls])
                ra_objs.append([bbox3d[0], bbox3d[1], bbox3d[3], bbox3d[4], cls])
            pred_objs = {"RD": rd_objs, "RA": ra_objs}
        elif model_type == "DAROD":
            input = torch.tensor(data['radar']).unsqueeze(0).to(device).float()
            output = net(input)
            pred_boxes, pred_labels, pred_scores = output["decoder_output"]
            pred_boxes = pred_boxes.detach().cpu().numpy()[0].astype(float)
            pred_labels = pred_labels.detach().cpu().numpy()[0].astype(float)
            pred_scores = pred_scores.detach().cpu().numpy()[0].astype(float)
            pred_labels = pred_labels - 1
            pred_objs = [list(pred_boxes[i]) + [pred_labels[i], pred_scores[i]] for i in range(len(pred_boxes))]
            pred_objs = {"RD": pred_objs}
        elif model_type == "RAMP_CNN":
            input = (data['ra_matrix'].to(device).float(), data['rv_matrix'].to(device).float(), data['va_matrix'].to(device).float())
            output = net(input)
            confmap_pred = output['confmap_pred'].cpu().detach().numpy() 
            pred_objs = post_process_single_frame(confmap_pred, parameters["train_config"], dataset_inst.n_class, dataset_inst.rng_grid, dataset_inst.agl_grid) #[B, win_size, max_dets, 4]
            # filter invalid predictions
            mask = pred_objs[:, 0] != -1 
            pred_objs = pred_objs[mask]
            # limit conf value
            mask = pred_objs[:, -1] > 1  
            pred_objs[mask, -1] = 1
            # reorganize the items: [row_id, col_id, cls_id, conf_value]
            pred_objs = pred_objs[:, [1, 2, 0, 3]]
            #feature_show_pred = "RA"
            pred_objs = {"RA": pred_objs.tolist()}
        elif model_type == "RadarCrossAttention":
            input = (data['ra_matrix'].to(device).float(), data['rd_matrix'].to(device).float(), data['ad_matrix'].to(device).float())  
            output = net(input)
            pred_map = torch.sigmoid(output["pred_mask"])
            pred_c = 8 * (torch.sigmoid(output["pred_center"]) - 0.5)
            pred_o = output["pred_orent"]
            mask, peak_cls = create_default(pred_map.size(), kernel_window=(3,5))
            pred_intent, pred_idx = peaks_detect(pred_map, mask, peak_cls, heat_thresh=0.6)
            pred_idx= distribute(pred_idx,device)
            pred_idx = update_peaks(pred_c, pred_idx)
            pred_idx, pred_intent= association(pred_intent, pred_idx, device)
            ra_objs = []
            for cnt, cord in enumerate(pred_idx):
                p_cls, p_r, p_c = int(cord[1]), cord[2], cord[3]
                ang = (p_c*57.29*2/256 - 57.29)/np.pi
                ra_objs.append([50*p_r*np.cos(ang), 50*p_r*np.sin(ang), p_cls, pred_intent[cnt], 0]) # [row_id, col_id, cls_id, conf_value, heading]
                print(f"Class: {p_cls}, Confidence: {pred_intent[cnt]},\
                    Range: {50*p_r/256}, Angle(deg): {p_c*57.29*2/256 - 57.29},\
                    Heading: {orent(pred_o[p_r,::], int(p_r//4), int(p_c//4), pred=True)}")  
            pred_objs = {"RA": ra_objs.tolist()}
        else:
            raise ValueError("Inference of the chosen model type is not supported")

    return pred_objs


def display_inference_FFTRadNet(image, input, model_outputs, obj_labels, train_config=None):
    # Usage: #pred_image = display_inference_FFTRadNet(data[4], data[0], output, data[3])
    
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
