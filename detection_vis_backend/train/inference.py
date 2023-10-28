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

from detection_vis_backend.datasets.utils import confmap2ra
from detection_vis_backend.train.utils import post_process_single_frame, get_class_name, worldToImage, decode, process_predictions_FFT


def display_inference_CRUW(image_path, RAmap, model_output, gt_labels, train_config):
    root_path = "/home/kangle/dataset/CRUW"
    with open(os.path.join(root_path, 'sensor_config_rod2021.json'), 'r') as file:
        sensor_cfg = json.load(file)
    radar_cfg = sensor_cfg['radar_cfg']
    n_class = 3 # dataset.object_cfg.n_class
    classes = ["pedestrian", "cyclist", "car"]  # dataset.object_cfg.classes
    rng_grid = confmap2ra(radar_cfg, name='range')
    agl_grid = confmap2ra(radar_cfg, name='angle')
    res_final = post_process_single_frame(model_output, train_config, n_class, rng_grid, agl_grid) #[B, win_size, max_dets, 4]
    
    img_data = np.asarray(Image.open(image_path))
    if img_data.shape[0] > 864:
        img_data = img_data[:img_data.shape[0] // 5 * 4, :, :]

    confmap_pred = np.transpose(model_output, (1, 2, 0))
    confmap_pred[confmap_pred < 0] = 0
    confmap_pred[confmap_pred > 1] = 1

    # Draw predictions on confmap_pred
    max_dets, _ = res_final.shape
    for d in range(max_dets):
        cla_id = int(res_final[d, 0])
        if cla_id == -1:
            continue
        row_id = res_final[d, 1]
        col_id = res_final[d, 2]
        conf = res_final[d, 3]
        conf = 1.0 if conf > 1 else conf
        cla_str = get_class_name(cla_id, classes)
        cv2.circle(confmap_pred, (col_id, row_id)) #plt.scatter(col_id, row_id, s=10, c='white')
        text = cla_str + '\n%.2f' % conf
        cv2.putText(confmap_pred, text, (col_id + 5, row_id)) #plt.text(col_id + 5, row_id, text, color='white', fontsize=10)
    
    confmap_gt = np.transpose(confmap_gt, (1, 2, 0))
    # Draw gt_labels
    for obj in gt_labels:
        cv2.circle(confmap_gt, (obj[1],obj[0]))
        cv2.putText(confmap_gt, obj[2], (obj[1] + 2, obj[0] + 2))
    return np.hstack(img_data, RAmap, confmap_gt, confmap_pred)


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