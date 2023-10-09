import numpy as np
import pandas as pd
import pkbar
import torch
import time
import os
import math
import json
import torch.nn as nn


from shapely.geometry import Polygon
from scipy.stats import hmean
from sklearn.metrics import confusion_matrix

from detection_vis_backend.train.utils import pixor_loss, decode, get_metrics, boxDecoder, lossYolo
from detection_vis_backend.datasets.utils import confmap2ra, get_class_id, iou3d




def FFTRadNet_val_evaluation(net, loader, check_perf=False, losses_params=None, device='cpu'):
    net.eval()
    running_loss = 0.0
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)
    metrics = Metrics()
    metrics.reset()
    criterion_det = pixor_loss if losses_params['detection_loss'] == 'PixorLoss' else None
    criterion_seg = nn.BCEWithLogitsLoss(reduction='mean') if losses_params['segmentation_loss'] == 'BCEWithLogitsLoss' else nn.BCELoss()

    for i, data in enumerate(loader):
        # input, out_label,segmap,labels
        inputs = data[0].to(device).float()
        label_map = data[1].to(device).float()
        seg_map_label = data[2].to(device).double()

        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        if(criterion_det!=None and criterion_seg!=None):
            classif_loss,reg_loss = criterion_det(outputs['Detection'], label_map,losses_params)           
            prediction = outputs['Segmentation'].contiguous().flatten()
            label = seg_map_label.contiguous().flatten()        
            loss_seg = criterion_seg(prediction, label)
            loss_seg *= inputs.size(0)
                
            classif_loss *= losses_params['weight'][0]
            reg_loss *= losses_params['weight'][1]
            loss_seg *=losses_params['weight'][2]

            loss = classif_loss + reg_loss + loss_seg

            # statistics
            running_loss += loss.item() * inputs.size(0)

        if(check_perf):
            out_obj = outputs['Detection'].detach().cpu().numpy().copy()
            labels = data[3]

            out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
            label_freespace = seg_map_label.detach().cpu().numpy().copy()

            for pred_obj,pred_map,true_obj,true_map in zip(out_obj,out_seg,labels,label_freespace):
                metrics.update(pred_map[0],true_map,np.asarray(decode(pred_obj,0.05)),true_obj,threshold=0.2,range_min=5,range_max=100) 
                
        kbar.update(i)
        
    mAP,mAR, mIoU = metrics.GetMetrics()

    return {'loss':running_loss, 'mAP':mAP, 'mAR':mAR, 'mIoU':mIoU}


def FFTRadNet_test_evaluation(net, loader, device='cpu', iou_threshold=0.5):

    net.eval()
    
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    print('Generating Predictions...')
    predictions = {'prediction':{'objects':[],'freespace':[]},'label':{'objects':[],'freespace':[]}}
    for i, data in enumerate(loader):

        # input, out_label,segmap,labels
        inputs = data[0].to(device).float()

        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        out_obj = outputs['Detection'].detach().cpu().numpy().copy()
        out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
        
        labels_object = data[3]
        label_freespace = data[2].numpy().copy()
            
        for pred_obj,pred_map,true_obj,true_map in zip(out_obj,out_seg,labels_object,label_freespace):
            
            predictions['prediction']['objects'].append( np.asarray(decode(pred_obj,0.05)))
            predictions['label']['objects'].append(true_obj)

            predictions['prediction']['freespace'].append(pred_map[0])
            predictions['label']['freespace'].append(true_map)
                

        kbar.update(i)
        print(f'Step {i+1}/{len(loader)}')
        
    mAP, mAR, F1_score = GetFullMetrics(predictions['prediction']['objects'],predictions['label']['objects'],range_min=5,range_max=100,IOU_threshold=0.5)

    mIoU = []
    for i in range(len(predictions['prediction']['freespace'])):
        # 0 to 124 means 0 to 50m
        pred = predictions['prediction']['freespace'][i][:124].reshape(-1)>=0.5
        label = predictions['label']['freespace'][i][:124].reshape(-1)
        
        intersection = np.abs(pred*label).sum()
        union = np.sum(label) + np.sum(pred) -intersection
        iou = intersection /union
        mIoU.append(iou)

    mIoU = np.asarray(mIoU).mean()
    print('------- Freespace Scores ------------')
    print('  mIoU',mIoU*100,'%')
    return {'mAP': [mAP], 'mAR':[mAR], 'F1_score': [F1_score], 'mIoU': [mIoU*100]}


def GetFullMetrics(predictions,object_labels,range_min=5,range_max=100,IOU_threshold=0.5):
    perfs = {}
    precision = []
    recall = []
    iou_threshold = []
    RangeError = []
    AngleError = []

    out = []

    for threshold in np.arange(0.1,0.96,0.1):
        print(f"begin the iteration of threshold = {threshold}")
        iou_threshold.append(threshold)

        TP = 0
        FP = 0
        FN = 0
        NbDet = 0
        NbGT = 0
        NBFrame = 0
        range_error=0
        angle_error=0
        nbObjects = 0

        for frame_id in range(len(predictions)):
            pred= predictions[frame_id]
            labels = object_labels[frame_id]

            # get final bounding box predictions
            Object_predictions = []
            ground_truth_box_corners = []           
            
            if(len(pred)>0):
                Object_predictions = process_predictions_FFT(pred,confidence_threshold=threshold)

            if(len(Object_predictions)>0):
                max_distance_predictions = (Object_predictions[:,2]+Object_predictions[:,4])/2
                ids = np.where((max_distance_predictions>=range_min) & (max_distance_predictions <= range_max) )
                Object_predictions = Object_predictions[ids]

            NbDet += len(Object_predictions)

            if(len(labels)>0):
                ids = np.where((labels[:,0]>=range_min) & (labels[:,0] <= range_max))
                labels = labels[ids]

            if(len(labels)>0):
                ground_truth_box_corners = np.asarray(RA_to_cartesian_box(labels))
                NbGT += ground_truth_box_corners.shape[0]

            # valid predictions and labels exist for the currently inspected point cloud
            if len(ground_truth_box_corners)>0 and len(Object_predictions)>0:

                used_gt = np.zeros(len(ground_truth_box_corners))
                for pid, prediction in enumerate(Object_predictions):
                    iou = bbox_iou(prediction[1:], ground_truth_box_corners)
                    ids = np.where(iou>=IOU_threshold)[0]

                    
                    if(len(ids)>0):
                        TP += 1
                        used_gt[ids]=1

                        # cummulate errors
                        range_error += np.sum(np.abs(ground_truth_box_corners[ids,-2] - prediction[-2]))
                        angle_error += np.sum(np.abs(ground_truth_box_corners[ids,-1] - prediction[-1]))
                        nbObjects+=len(ids)
                    else:
                        FP+=1
                FN += np.sum(used_gt==0)


            elif(len(ground_truth_box_corners)==0):
                FP += len(Object_predictions)
            elif(len(Object_predictions)==0):
                FN += len(ground_truth_box_corners)
                
        if(TP!=0):
            precision.append( TP / (TP+FP)) # When there is a detection, how much I m sure
            recall.append(TP / (TP+FN))
        else:
            precision.append( 0) # When there is a detection, how much I m sure
            recall.append(0)

        if nbObjects > 0:
            RangeError.append(range_error/nbObjects)
            AngleError.append(angle_error/nbObjects)
        

    perfs['precision']=precision
    perfs['recall']=recall

    F1_score = (np.mean(precision)*np.mean(recall))/((np.mean(precision) + np.mean(recall))/2)
    print('------- Detection Scores ------------')
    print('  mAP:',np.mean(perfs['precision']))
    print('  mAR:',np.mean(perfs['recall']))
    print('  F1 score:',F1_score)

    print('------- Regression Errors------------')
    print('  Range Error:',np.mean(RangeError),'m')
    print('  Angle Error:',np.mean(AngleError),'degree')

    mAP = np.mean(perfs['precision'])
    mAR = np.mean(perfs['recall'])
    return mAP, mAR, F1_score

def GetDetMetrics(predictions,object_labels,threshold=0.2,range_min=5,range_max=70,IOU_threshold=0.2):

    TP = 0
    FP = 0
    FN = 0
    NbDet=0
    NbGT=0
   
    # get final bounding box predictions
    Object_predictions = []
    ground_truth_box_corners = []    
    labels=[]       

    if(len(predictions)>0):
        Object_predictions = process_predictions_FFT(predictions,confidence_threshold=threshold)

    if(len(Object_predictions)>0):
        max_distance_predictions = (Object_predictions[:,2]+Object_predictions[:,4])/2
        ids = np.where((max_distance_predictions>=range_min) & (max_distance_predictions <= range_max) )
        Object_predictions = Object_predictions[ids]

    NbDet = len(Object_predictions)
 
    if(len(object_labels)>0):
        ids = np.where((object_labels[:,0]>=range_min) & (object_labels[:,0] <= range_max))
        labels = object_labels[ids]
    if(len(labels)>0):
        ground_truth_box_corners = np.asarray(RA_to_cartesian_box(labels))
        NbGT = len(ground_truth_box_corners)

    # valid predictions and labels exist for the currently inspected point cloud
    if NbDet>0 and NbGT>0:

        used_gt = np.zeros(len(ground_truth_box_corners))

        for pid, prediction in enumerate(Object_predictions):
            iou = bbox_iou(prediction[1:], ground_truth_box_corners)
            ids = np.where(iou>=IOU_threshold)[0]

            if(len(ids)>0):
                TP += 1
                used_gt[ids]=1
            else:
                FP+=1
        FN += np.sum(used_gt==0)

    elif(NbGT==0):
        FP += NbDet
    elif(NbDet==0):
        FN += NbGT
        
    return TP,FP,FN


def GetSegMetrics(PredMap,label_map):

    # Segmentation
    pred = PredMap.reshape(-1)>=0.5
    label = label_map.reshape(-1)

    intersection = np.abs(pred*label).sum()
    union = np.sum(label) + np.sum(pred) -intersection
    iou = intersection /union

    return iou

class Metrics():
    def __init__(self,):
        
        self.iou = []
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.precision = 0
        self.recall = 0
        self.mIoU =0

    def update(self,PredMap,label_map,ObjectPred,Objectlabels,threshold=0.2,range_min=5,range_max=70):

        if(len(PredMap)>0):
            pred = PredMap.reshape(-1)>=0.5
            label = label_map.reshape(-1)

            intersection = np.abs(pred*label).sum()
            union = np.sum(label) + np.sum(pred) -intersection
            self.iou.append(intersection /union)

        TP,FP,FN = GetDetMetrics(ObjectPred,Objectlabels,threshold=0.2,range_min=range_min,range_max=range_max)

        self.TP += TP
        self.FP += FP
        self.FN += FN

    def reset(self,):
        self.iou = []
        self.TP = 0
        self.FP = 0
        self.FN = 0
        self.precision = 0
        self.mIoU =0

    def GetMetrics(self,):
        
        if(self.TP+self.FP!=0):
            self.precision = self.TP / (self.TP+self.FP)
        if(self.TP+self.FN!=0):
            self.recall = self.TP / (self.TP+self.FN)

        if(len(self.iou)>0):
            self.mIoU = np.asarray(self.iou).mean()

        return self.precision,self.recall,self.mIoU 


def RA_to_cartesian_box(data):
    L = 4
    W = 1.8
    
    boxes = []
    for i in range(len(data)):
        
        x = np.sin(np.radians(data[i][1])) * data[i][0]
        y = np.cos(np.radians(data[i][1])) * data[i][0]

        boxes.append([x - W/2,y,x + W/2,y, x + W/2,y+L,x - W/2,y+L])
              
    return boxes

def perform_nms(valid_class_predictions, valid_box_predictions, nms_threshold):

    # sort the detections such that the entry with the maximum confidence score is at the top
    sorted_indices = np.argsort(valid_class_predictions)[::-1]
    sorted_box_predictions = valid_box_predictions[sorted_indices]
    sorted_class_predictions = valid_class_predictions[sorted_indices]

    for i in range(sorted_box_predictions.shape[0]):
        # get the IOUs of all boxes with the currently most certain bounding box
        try:
            ious = np.zeros((sorted_box_predictions.shape[0]))
            ious[i + 1:] = bbox_iou(sorted_box_predictions[i, :], sorted_box_predictions[i + 1:, :])
        except ValueError:
            break
        except IndexError:
            break

        # eliminate all detections which have IoU > threshold
        overlap_mask = np.where(ious < nms_threshold, True, False)
        sorted_box_predictions = sorted_box_predictions[overlap_mask]
        sorted_class_predictions = sorted_class_predictions[overlap_mask]

    return sorted_class_predictions, sorted_box_predictions


def bbox_iou(box1, boxes):

    # currently inspected box
    box1 = box1.reshape((4,2))
    rect_1 = Polygon([(box1[0, 0], box1[0, 1]), (box1[1, 0], box1[1, 1]), (box1[2, 0], box1[2, 1]),
                      (box1[3, 0], box1[3, 1])])
    area_1 = rect_1.area

    # IoU of box1 with each of the boxes in "boxes"
    ious = np.zeros(boxes.shape[0])
    for box_id in range(boxes.shape[0]):
        box2 = boxes[box_id]
        box2 = box2.reshape((4,2))
        rect_2 = Polygon([(box2[0, 0], box2[0, 1]), (box2[1, 0], box2[1, 1]), (box2[2, 0], box2[2, 1]),
                          (box2[3, 0], box2[3, 1])])
        area_2 = rect_2.area

        # get intersection of both bounding boxes
        inter_area = rect_1.intersection(rect_2).area

        # compute IoU of the two bounding boxes
        iou = inter_area / (area_1 + area_2 - inter_area)

        ious[box_id] = iou

    return ious

def process_predictions_FFT(batch_predictions, confidence_threshold=0.1, nms_threshold=0.05):

    # process targets and perform NMS for each prediction in batch
    final_batch_predictions = None  # store final bounding box predictions
    
    point_cloud_reg_predictions = RA_to_cartesian_box(batch_predictions)
    point_cloud_reg_predictions = np.asarray(point_cloud_reg_predictions)
    point_cloud_class_predictions = batch_predictions[:,-1]

    # get valid detections
    validity_mask = np.where(point_cloud_class_predictions > confidence_threshold, True, False)
    
    valid_box_predictions = point_cloud_reg_predictions[validity_mask]
    valid_class_predictions = point_cloud_class_predictions[validity_mask]

    # perform Non-Maximum Suppression
    final_class_predictions, final_box_predictions = perform_nms(valid_class_predictions, valid_box_predictions, nms_threshold)

    # concatenate point_cloud_id, confidence score and bounding box prediction | shape: [N_FINAL, 1+1+8]
    final_Object_predictions = np.hstack((final_class_predictions[:, np.newaxis], final_box_predictions))
    return final_Object_predictions


class ConfmapStack:
    def __init__(self, confmap_shape):
        self.confmap = np.zeros(confmap_shape)
        self.count = 0
        self.next = None
        self.ready = False

    def append(self, confmap):
        self.confmap = (self.confmap * self.count + confmap) / (self.count + 1)
        self.count += 1

    def setNext(self, _genconfmap):
        self.next = _genconfmap


def get_class_name(class_id, classes):
    n_class = len(classes)
    if 0 <= class_id < n_class:
        class_name = classes[class_id]
    elif class_id == -1000:
        class_name = '__background'
    else:
        raise ValueError("Class ID is not defined")
    return class_name


def pol2cart_ramap(rho, phi):
    """
    Transform from polar to cart under RAMap coordinates
    :param rho: distance to origin
    :param phi: angle (rad) under RAMap coordinates
    :return: x, y
    """
    x = rho * np.sin(phi)
    y = rho * np.cos(phi)
    return x, y


def get_ols_btw_objects(obj1, obj2):
    classes = ["pedestrian", "cyclist", "car"] 
    object_sizes = {
        "pedestrian": 0.5,
        "cyclist": 1.0,
        "car": 3.0
    }

    if obj1['class_id'] != obj2['class_id']:
        print('Error: Computing OLS between different classes!')
        raise TypeError("OLS can only be compute between objects with same class.  ")
    if obj1['score'] < obj2['score']:
        raise TypeError("Confidence score of obj1 should not be smaller than obj2. "
                        "obj1['score'] = %s, obj2['score'] = %s" % (obj1['score'], obj2['score']))

    classid = obj1['class_id']
    class_str = get_class_name(classid, classes)
    rng1 = obj1['range']
    agl1 = obj1['angle']
    rng2 = obj2['range']
    agl2 = obj2['angle']
    x1, y1 = pol2cart_ramap(rng1, agl1)
    x2, y2 = pol2cart_ramap(rng2, agl2)
    dx = x1 - x2
    dy = y1 - y2
    s_square = x1 ** 2 + y1 ** 2
    kappa = object_sizes[class_str] / 100  # TODO: tune kappa
    e = (dx ** 2 + dy ** 2) / 2 / (s_square * kappa)
    ols = math.exp(-e)
    return ols


def lnms(obj_dicts_in_class, train_cfg):
    """
    Location-based NMS
    :param obj_dicts_in_class:
    :param config_dict:
    :return:
    """
    detect_mat = - np.ones((train_cfg['max_dets'], 4))
    cur_det_id = 0
    # sort peaks by confidence score
    inds = np.argsort([-d['score'] for d in obj_dicts_in_class], kind='mergesort')
    dts = [obj_dicts_in_class[i] for i in inds]
    while len(dts) != 0:
        if cur_det_id >= train_cfg['max_dets']:
            break
        p_star = dts[0]
        detect_mat[cur_det_id, 0] = p_star['class_id']
        detect_mat[cur_det_id, 1] = p_star['range_id']
        detect_mat[cur_det_id, 2] = p_star['angle_id']
        detect_mat[cur_det_id, 3] = p_star['score']
        cur_det_id += 1
        del dts[0]
        for pid, pi in enumerate(dts):
            ols = get_ols_btw_objects(p_star, pi)
            if ols > train_cfg['ols_thres']:
                del dts[pid]

    return detect_mat


def detect_peaks(image, threshold=0.3):
    peaks_row = []
    peaks_col = []
    height, width = image.shape
    for h in range(1, height - 1):
        for w in range(2, width - 2):
            area = image[h - 1:h + 2, w - 2:w + 3]
            center = image[h, w]
            flag = np.where(area >= center)
            if flag[0].shape[0] == 1 and center > threshold:
                peaks_row.append(h)
                peaks_col.append(w)

    return peaks_row, peaks_col


def post_process_single_frame(confmaps, train_cfg, n_class, rng_grid, agl_grid):
    """
    Post-processing for RODNet
    :param confmaps: predicted confidence map [B, n_class, win_size, ramap_r, ramap_a]
    :param search_size: search other detections within this window (resolution of our system)
    :param peak_thres: peak threshold
    :return: [B, win_size, max_dets, 4]
    """
    max_dets = train_cfg['max_dets']
    peak_thres = train_cfg['peak_thres']

    class_size, height, width = confmaps.shape

    if class_size != n_class:
        raise TypeError("Wrong class number setting. ")

    res_final = - np.ones((max_dets, 4))

    detect_mat = []
    for c in range(class_size):
        obj_dicts_in_class = []
        confmap = confmaps[c, :, :]
        rowids, colids = detect_peaks(confmap, threshold=peak_thres)

        for ridx, aidx in zip(rowids, colids):
            rng = rng_grid[ridx]
            agl = agl_grid[aidx]
            conf = confmap[ridx, aidx]
            obj_dict = dict(
                frame_id=None,
                range=rng,
                angle=agl,
                range_id=ridx,
                angle_id=aidx,
                class_id=c,
                score=conf,
            )
            obj_dicts_in_class.append(obj_dict)

        detect_mat_in_class = lnms(obj_dicts_in_class, train_cfg)
        detect_mat.append(detect_mat_in_class)

    detect_mat = np.array(detect_mat)
    detect_mat = np.reshape(detect_mat, (class_size * max_dets, 4))
    detect_mat = detect_mat[detect_mat[:, 3].argsort(kind='mergesort')[::-1]]
    res_final[:, :] = detect_mat[:max_dets]

    return res_final


def write_dets_results_single_frame(res, data_id, save_path, classes):
    max_dets, _ = res.shape
    with open(save_path, 'a+') as f:
        for d in range(max_dets):
            cla_id = int(res[d, 0])
            if cla_id == -1:
                continue
            row_id = res[d, 1]
            col_id = res[d, 2]
            conf = res[d, 3]
            f.write("%d %s %d %d %.4f\n" % (data_id, get_class_name(cla_id, classes), row_id, col_id, conf))


def read_gt_txt(txt_path, n_frame, n_class, classes):
    with open(txt_path, 'r') as f:
        data = f.readlines()
    dets = [None] * n_frame
    for line in data:
        frame_id, r, a, class_name = line.rstrip().split()
        frame_id = int(frame_id)
        r = float(r)
        a = float(a)
        class_id = classes.index(class_name)
        obj_dict = dict(
            frame_id=frame_id,
            range=r,
            angle=a,
            class_name=class_name,
            class_id=class_id
        )
        if dets[frame_id] is None:
            dets[frame_id] = [obj_dict]
        else:
            dets[frame_id].append(obj_dict)

    gts = {(i, j): [] for i in range(n_frame) for j in range(n_class)}
    id = 1
    for frameid, obj_info in enumerate(dets):
        # for each frame
        if obj_info is None:
            continue
        for obj_dict in obj_info:
            rng = obj_dict['range']
            agl = obj_dict['angle']
            class_id = obj_dict['class_id']
            if rng > 25 or rng < 1:
                continue
            if agl > math.radians(60) or agl < math.radians(-60):
                continue
            obj_dict_gt = obj_dict.copy()
            obj_dict_gt['id'] = id
            obj_dict_gt['score'] = 1.0
            gts[frameid, class_id].append(obj_dict_gt)
            id += 1

    return gts


def read_rodnet_res(filename, n_frame, n_class, classes, rng_grid, agl_grid):
    with open(filename, 'r') as df:
        data = df.readlines()
    if len(data) == 0:
        return None

    dts = {(i, j): [] for i in range(n_frame) for j in range(n_class)}

    for id, line in enumerate(data):
        if line is not None:
            line = line.rstrip().split()
            frameid, class_str, ridx, aidx, conf = line
            frameid = int(frameid)
            classid = get_class_id(class_str, classes)
            ridx = int(ridx)
            aidx = int(aidx)
            conf = float(conf)
            if conf > 1:
                conf = 1
            rng = rng_grid[ridx]
            agl = agl_grid[aidx]
            if rng > 25 or rng < 1:
                continue
            if agl > math.radians(60) or agl < math.radians(-60):
                continue
            obj_dict = dict(
                id=id + 1,
                frame_id=frameid,
                range=rng,
                angle=agl,
                range_id=ridx,
                angle_id=aidx,
                class_id=classid,
                score=conf
            )
            dts[frameid, classid].append(obj_dict)

    return dts


def evaluate_img(gts_dict, dts_dict, imgId, catId, olss_dict, olsThrs, recThrs, classes, log=False):
    gts = gts_dict[imgId, catId]
    dts = dts_dict[imgId, catId]
    if len(gts) == 0 and len(dts) == 0:
        return None

    if log:
        olss_flatten = np.ravel(olss_dict[imgId, catId])
        print("Frame %d: %10s %s" % (imgId, classes[catId], list(olss_flatten)))

    dtind = np.argsort([-d['score'] for d in dts], kind='mergesort')
    dts = [dts[i] for i in dtind]
    olss = olss_dict[imgId, catId]

    T = len(olsThrs)
    G = len(gts)
    D = len(dts)
    gtm = np.zeros((T, G))
    dtm = np.zeros((T, D))

    if not len(olss) == 0:
        for tind, t in enumerate(olsThrs):
            for dind, d in enumerate(dts):
                # information about best match so far (m=-1 -> unmatched)
                iou = min([t, 1 - 1e-10])
                m = -1
                for gind, g in enumerate(gts):
                    # if this gt already matched, continue
                    if gtm[tind, gind] > 0:
                        continue
                    if olss[dind, gind] < iou:
                        continue
                    # if match successful and best so far, store appropriately
                    iou = olss[dind, gind]
                    m = gind
                # if match made store id of match for both dt and gt
                if m == -1:
                    # no gt matched
                    continue
                dtm[tind, dind] = gts[m]['id']
                gtm[tind, m] = d['id']
    # store results for given image and category
    return {
        'image_id': imgId,
        'category_id': catId,
        'dtIds': [d['id'] for d in dts],
        'gtIds': [g['id'] for g in gts],
        'dtMatches': dtm,
        'gtMatches': gtm,
        'dtScores': [d['score'] for d in dts],
    }


def evaluate_rodnet_seq(res_path, gt_path, n_frame, n_class, classes, rng_grid, agl_grid, olsThrs, recThrs):

    gt_dets = read_gt_txt(gt_path, n_frame, n_class, classes)
    sub_dets = read_rodnet_res(res_path, n_frame, n_class, classes, rng_grid, agl_grid)

    olss_all = {(imgId, catId): compute_ols_dts_gts(gt_dets, sub_dets, imgId, catId) \
                for imgId in range(n_frame)
                for catId in range(3)}

    evalImgs = [evaluate_img(gt_dets, sub_dets, imgId, catId, olss_all, olsThrs, recThrs, classes)
                for imgId in range(n_frame)
                for catId in range(3)]

    return evalImgs


def compute_ols_dts_gts(gts_dict, dts_dict, imgId, catId):
    """Compute OLS between detections and gts for a category in a frame."""
    gts = gts_dict[imgId, catId]
    dts = dts_dict[imgId, catId]
    inds = np.argsort([-d['score'] for d in dts], kind='mergesort')
    dts = [dts[i] for i in inds]
    if len(gts) == 0 or len(dts) == 0:
        return []
    olss = np.zeros((len(dts), len(gts)))
    # compute oks between each detection and ground truth object
    for j, gt in enumerate(gts):
        for i, dt in enumerate(dts):
            olss[i, j] = get_ols_btw_objects(gt, dt)
    return olss


def accumulate(evalImgs, n_frame, olsThrs, recThrs, n_class, classes, log=True):
    T = len(olsThrs)
    R = len(recThrs)
    K = n_class
    precision = -np.ones((T, R, K))  # -1 for the precision of absent categories
    recall = -np.ones((T, K))
    scores = -np.ones((T, R, K))
    n_objects = np.zeros((K,))

    for classid in range(n_class):
        E = [evalImgs[i * n_class + classid] for i in range(n_frame)]
        E = [e for e in E if not e is None]
        if len(E) == 0:
            continue

        dtScores = np.concatenate([e['dtScores'] for e in E])
        # different sorting method generates slightly different results.
        # mergesort is used to be consistent as Matlab implementation.
        inds = np.argsort(-dtScores, kind='mergesort')
        dtScoresSorted = dtScores[inds]

        dtm = np.concatenate([e['dtMatches'] for e in E], axis=1)[:, inds]
        gtm = np.concatenate([e['gtMatches'] for e in E], axis=1)
        nd = dtm.shape[1]  # number of detections
        ng = gtm.shape[1]  # number of ground truth
        n_objects[classid] = ng

        if log:
            print("%10s: %4d dets, %4d gts" % (classes[classid], dtm.shape[1], gtm.shape[1]))

        tps = np.array(dtm, dtype=bool)
        fps = np.logical_not(dtm)
        tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
        fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)

        for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
            tp = np.array(tp)
            fp = np.array(fp)
            rc = tp / (ng + np.spacing(1))
            pr = tp / (fp + tp + np.spacing(1))
            q = np.zeros((R,))
            ss = np.zeros((R,))

            if nd:
                recall[t, classid] = rc[-1]
            else:
                recall[t, classid] = 0

            # numpy is slow without cython optimization for accessing elements
            # use python array gets significant speed improvement
            pr = pr.tolist()
            q = q.tolist()

            for i in range(nd - 1, 0, -1):
                if pr[i] > pr[i - 1]:
                    pr[i - 1] = pr[i]

            inds = np.searchsorted(rc, recThrs, side='left')
            try:
                for ri, pi in enumerate(inds):
                    q[ri] = pr[pi]
                    ss[ri] = dtScoresSorted[pi]
            except:
                pass
            precision[t, :, classid] = np.array(q)
            scores[t, :, classid] = np.array(ss)

    eval = {
        'counts': [T, R, K],
        'object_counts': n_objects,
        'precision': precision,
        'recall': recall,
        'scores': scores,
    }
    return eval


def summarize(eval, olsThrs, recThrs, n_class, gl=True):
    def _summarize(eval=eval, ap=1, olsThr=None):
        object_counts = eval['object_counts']
        n_objects = np.sum(object_counts)
        if ap == 1:
            # dimension of precision: [TxRxK]
            s = eval['precision']
            # IoU
            if olsThr is not None:
                t = np.where(olsThr == olsThrs)[0]
                s = s[t]
            s = s[:, :, :]
        else:
            # dimension of recall: [TxK]
            s = eval['recall']
            if olsThr is not None:
                t = np.where(olsThr == olsThrs)[0]
                s = s[t]
            s = s[:, :]
        # mean_s = np.mean(s[s>-1])
        mean_s = 0
        for classid in range(n_class):
            if ap == 1:
                s_class = s[:, :, classid]
                if len(s_class[s_class > -1]) == 0:
                    pass
                else:
                    mean_s += object_counts[classid] / n_objects * np.mean(s_class[s_class > -1])
            else:
                s_class = s[:, classid]
                if len(s_class[s_class > -1]) == 0:
                    pass
                else:
                    mean_s += object_counts[classid] / n_objects * np.mean(s_class[s_class > -1])
        return mean_s

    def _summarizeKps():
        stats = np.zeros((12,))
        stats[0] = _summarize(ap=1)
        stats[1] = _summarize(ap=1, olsThr=.5)
        stats[2] = _summarize(ap=1, olsThr=.6)
        stats[3] = _summarize(ap=1, olsThr=.7)
        stats[4] = _summarize(ap=1, olsThr=.8)
        stats[5] = _summarize(ap=1, olsThr=.9)
        stats[6] = _summarize(ap=0)
        stats[7] = _summarize(ap=0, olsThr=.5)
        stats[8] = _summarize(ap=0, olsThr=.6)
        stats[9] = _summarize(ap=0, olsThr=.7)
        stats[10] = _summarize(ap=0, olsThr=.8)
        stats[11] = _summarize(ap=0, olsThr=.9)
        return stats

    def _summarizeKps_cur():
        stats = np.zeros((2,))
        stats[0] = _summarize(ap=1)
        stats[1] = _summarize(ap=0)
        return stats

    if gl:
        summarize = _summarizeKps
    else:
        summarize = _summarizeKps_cur

    stats = summarize()
    return stats


def validate(net, dataloader, criterion, device):
    net.eval()
    running_loss = 0.0
    
    kbar = pkbar.Kbar(target=len(dataloader), width=20, always_stateful=False)
    for i, data_dict in enumerate(dataloader):
        data = data_dict['radar_data'].to(device).float()
        confmap_gt = data['anno']['confmaps'].to(device).float()
        with torch.set_grad_enabled(False):
            confmap_pred = net(data)
        loss = criterion(confmap_pred, confmap_gt)
        print(f'val loss of sample {i}: {loss}')
        running_loss += loss.item() * data.size(0)
        kbar.update(i)
    return


def RODNet_evaluation(net, dataloader, save_dir, train_cfg, model_cfg, device):
    root_path = "/home/kangle/dataset/CRUW"
    with open(os.path.join(root_path, 'sensor_config_rod2021.json'), 'r') as file:
        sensor_cfg = json.load(file)
    radar_cfg = sensor_cfg['radar_cfg']
    n_class = 3 # dataset.object_cfg.n_class
    classes = ["pedestrian", "cyclist", "car"]  # dataset.object_cfg.classes
    rng_grid = confmap2ra(radar_cfg, name='range')
    agl_grid = confmap2ra(radar_cfg, name='angle')

    net.eval()

    # 1.Generate network output (confmaps) and post-process them to the form of detection predictions
    confmap_shape = (3, 128, 128) # (n_class, radar_cfg['ramap_rsize'], radar_cfg['ramap_asize'])
    init_genConfmap = ConfmapStack(confmap_shape)
    iter_ = init_genConfmap
    for i in range(train_cfg['win_size'] - 1):
        while iter_.next is not None:
            iter_ = iter_.next
        iter_.next = ConfmapStack(confmap_shape)

    total_time = 0
    total_count = 0
    load_tic = time.time()
    sequences = []
    for iter, data_dict in enumerate(dataloader):
        load_time = time.time() - load_tic
        data = data_dict['radar_data'].to(device).float()
        try:
            image_paths = data_dict['image_paths'][0]
        except:
            print('warning: fail to load RGB images, will not visualize results')
            image_paths = None
        seq_name = data_dict['seq_names'][0]
        if seq_name not in sequences:
            sequences.append(seq_name)

        # if not args.demo:
        confmap_gt = data_dict['anno']['confmaps']
        obj_info = data_dict['anno']['obj_infos']
        # else:
        #     confmap_gt = None
        #     obj_info = None
        
        # Currently only support batch_size set to 1 for val evaluation
        start_frame_id = data_dict['start_frame'].item()
        end_frame_id = data_dict['end_frame'].item()

        tic = time.time()
        #confmap_pred = net(data.float().cuda())
        with torch.set_grad_enabled(False):
            confmap_pred = net(data)

        if model_cfg['stacked_num'] is not None:
            confmap_pred = confmap_pred[-1].cpu().detach().numpy()  # (1, 4, 32, 128, 128)
        else:
            confmap_pred = confmap_pred.cpu().detach().numpy()

        infer_time = time.time() - tic
        total_time += infer_time

        iter_ = init_genConfmap
        for i in range(confmap_pred.shape[2]):
            if iter_.next is None and i != confmap_pred.shape[2] - 1:
                iter_.next = ConfmapStack(confmap_shape)
            iter_.append(confmap_pred[0, :, i, :, :])
            iter_ = iter_.next

        process_tic = time.time()
        save_path = os.path.join(save_dir, seq_name + '_rod_res.txt')
        for i in range(train_cfg['train_stride']): # test_stride
            total_count += 1
            res_final = post_process_single_frame(init_genConfmap.confmap, train_cfg, n_class, rng_grid, agl_grid)
            cur_frame_id = start_frame_id + i
            write_dets_results_single_frame(res_final, cur_frame_id, save_path, classes)
            # confmap_pred_0 = init_genConfmap.confmap
            # res_final_0 = res_final
            # if image_paths is not None:
            #     img_path = image_paths[i]
            #     radar_input = chirp_amp(data.numpy()[0, :, i, :, :], radar_configs['data_type'])
            #     fig_name = os.path.join(test_res_dir, seq_name, 'rod_viz', '%010d.jpg' % (cur_frame_id))
            #     if confmap_gt is not None:
            #         confmap_gt_0 = confmap_gt[0, :, i, :, :]
            #         visualize_test_img(fig_name, img_path, radar_input, confmap_pred_0, confmap_gt_0, res_final_0,
            #                             dataset, sybl=sybl)
            #     else:
            #         visualize_test_img_wo_gt(fig_name, img_path, radar_input, confmap_pred_0, res_final_0,
            #                                     dataset, sybl=sybl)
            init_genConfmap = init_genConfmap.next

        if iter == len(dataloader) - 1:
            offset = train_cfg['train_stride'] # test_stride
            cur_frame_id = start_frame_id + offset
            while init_genConfmap is not None:
                total_count += 1
                res_final = post_process_single_frame(init_genConfmap.confmap, train_cfg, n_class, rng_grid, agl_grid)
                write_dets_results_single_frame(res_final, cur_frame_id, save_path, classes)
                # confmap_pred_0 = init_genConfmap.confmap
                # res_final_0 = res_final
                # if image_paths is not None:
                #     img_path = image_paths[offset]
                #     radar_input = chirp_amp(data.numpy()[0, :, offset, :, :], radar_configs['data_type'])
                #     fig_name = os.path.join(test_res_dir, seq_name, 'rod_viz', '%010d.jpg' % (cur_frame_id))
                #     if confmap_gt is not None:
                #         confmap_gt_0 = confmap_gt[0, :, offset, :, :]
                #         visualize_test_img(fig_name, img_path, radar_input, confmap_pred_0, confmap_gt_0,
                #                             res_final_0,
                #                             dataset, sybl=sybl)
                #     else:
                #         visualize_test_img_wo_gt(fig_name, img_path, radar_input, confmap_pred_0, res_final_0,
                #                                     dataset, sybl=sybl)
                init_genConfmap = init_genConfmap.next
                offset += 1
                cur_frame_id += 1
        print(f"Sample {iter}/{len(dataloader)} finished")

        if init_genConfmap is None:
            init_genConfmap = ConfmapStack(confmap_shape)

        proc_time = time.time() - process_tic
        print("Testing %s: frame %4d to %4d | Load time: %.4f | Inference time: %.4f | Process time: %.4f" %
                (seq_name, start_frame_id, end_frame_id, load_time, infer_time, proc_time))

        load_tic = time.time()
    print("ave time: %f" % (total_time / total_count))

    # 2.Evaluation the detection predictions with Ground-truth annotations
    evalImgs_all = []
    n_frames_all = 0

    for seq_name in sequences:
        gt_path = os.path.join(root_path, 'TRAIN_RAD_H_ANNO', seq_name + '.txt')
        res_path = os.path.join(save_dir, seq_name + '_rod_res.txt')
        n_frame = len(os.listdir(os.path.join(root_path, 'TRAIN_CAM_0', seq_name, 'IMAGES_0')))

        olsThrs = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1), endpoint=True), decimals=2)
        recThrs = np.around(np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01) + 1), endpoint=True), decimals=2)
        evalImgs = evaluate_rodnet_seq(res_path, gt_path, n_frame, n_class, classes, rng_grid, agl_grid, olsThrs, recThrs)
        
        eval = accumulate(evalImgs, n_frame, olsThrs, recThrs, n_class, classes, log=False)
        stats = summarize(eval, olsThrs, recThrs, n_class, gl=False)
        print("%s | AP_total: %.4f | AR_total: %.4f" % (seq_name.upper(), stats[0] * 100, stats[1] * 100))

        n_frames_all += n_frame
        evalImgs_all.extend(evalImgs)

    eval = accumulate(evalImgs_all, n_frames_all, olsThrs, recThrs, n_class, classes, log=False)
    stats = summarize(eval, olsThrs, recThrs, n_class, gl=False)
    print("%s | AP_total: %.4f | AR_total: %.4f" % ('Overall'.ljust(18), stats[0] * 100, stats[1] * 100))
    return {'loss':0, 'mAP':stats[0], 'mAR':stats[1], 'mIoU':0}


def RECORD_CRUW_evaluation(net, dataloader, save_dir, train_cfg, model_cfg, device, model_type):
    root_path = "/home/kangle/dataset/CRUW"
    with open(os.path.join(root_path, 'sensor_config_rod2021.json'), 'r') as file:
        sensor_cfg = json.load(file)
    radar_cfg = sensor_cfg['radar_cfg']
    n_class = 3 # dataset.object_cfg.n_class
    classes = ["pedestrian", "cyclist", "car"]  # dataset.object_cfg.classes
    rng_grid = confmap2ra(radar_cfg, name='range')
    agl_grid = confmap2ra(radar_cfg, name='angle')

    net.eval()
    sequences = []
    for iter, data_dict in enumerate(dataloader):
        print(f"Sample {iter}")
        
        ra_maps = data_dict['radar_data'].to(device).float()
        confmap_gts = data_dict['anno']['confmaps'].float()
        image_paths = data_dict['image_paths']
        seq_name = data_dict['seq_names'][0]
        if seq_name not in sequences:
            sequences.append(seq_name)
        save_path = os.path.join(save_dir, seq_name + '_record_res.txt')

        if confmap_gts is not None:
            start_frame_name = image_paths[0][0].split('/')[-1].split('.')[0]
            frame_name = image_paths[0][-1].split('/')[-1].split('.')[0]
            frame_id = int(frame_name)
        else:
            start_frame_name = image_paths[0][0][0].split('/')[-1].split('.')[0].split('_')[0]
            frame_name = image_paths[0][-1][0].split('/')[-1].split('.')[0].split('_')[0]
            frame_id = int(frame_name)

        if frame_id == train_cfg['win_size']-1 and model_type not in ('RECORDNoLstmMulti', 'RECORDNoLstm'):
            for tmp_frame_id in range(frame_id):
                print("Eval frame", tmp_frame_id)
                tmp_ra_maps = ra_maps[:, :, :tmp_frame_id+1]
                with torch.set_grad_enabled(False):
                    confmap_pred = net(tmp_ra_maps)
                res_final = post_process_single_frame(confmap_pred[0].cpu(), train_cfg, n_class, rng_grid, agl_grid)
                write_dets_results_single_frame(res_final, tmp_frame_id, save_path, classes)

        with torch.set_grad_enabled(False):
            confmap_pred = net(ra_maps)
            
        # Write results
        res_final = post_process_single_frame(confmap_pred[0].cpu(), train_cfg, n_class, rng_grid, agl_grid)
        write_dets_results_single_frame(res_final, frame_id, save_path, classes)
    print(f'record_res.txt file(s) for {sequences} created')

    # 2.Evaluation the detection predictions with Ground-truth annotations
    evalImgs_all = []
    n_frames_all = 0
    for seq_name in sequences:
        res_path = os.path.join(save_dir, seq_name + '_record_res.txt')
        # with open(res_path, 'r') as f:
        #     content = f.read().strip()
        # if not content:
        #     print(f"No objects detected in {seq_name}.")
        #     continue

        gt_path = os.path.join(root_path, 'TRAIN_RAD_H_ANNO', seq_name + '.txt')
        n_frame = len(os.listdir(os.path.join(root_path, 'TRAIN_CAM_0', seq_name, 'IMAGES_0')))

        olsThrs = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1), endpoint=True), decimals=2)
        recThrs = np.around(np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01) + 1), endpoint=True), decimals=2)
        evalImgs = evaluate_rodnet_seq(res_path, gt_path, n_frame, n_class, classes, rng_grid, agl_grid, olsThrs, recThrs)

        eval = accumulate(evalImgs, n_frame, olsThrs, recThrs, n_class, classes, log=False)
        stats = summarize(eval, olsThrs, recThrs, n_class, gl=False)
        print("%s | AP_total: %.4f | AR_total: %.4f" % (seq_name.upper(), stats[0] * 100, stats[1] * 100))

        n_frames_all += n_frame
        evalImgs_all.extend(evalImgs)
        
    eval = accumulate(evalImgs_all, n_frames_all, olsThrs, recThrs, n_class, classes, log=False)
    stats = summarize(eval, olsThrs, recThrs, n_class, gl=False)
    print("%s | AP_total: %.4f | AR_total: %.4f" % ('Overall'.ljust(18), stats[0] * 100, stats[1] * 100))
    return {'loss':0, 'mAP':stats[0], 'mAR':stats[1], 'mIoU':0}


class Evaluator:
    """Class to evaluate a model with quantitative metrics
    using a ground truth mask and a predicted mask.

    PARAMETERS
    ----------
    num_class: int
    """

    def __init__(self, num_class):
        self.num_class = num_class
        self.confusion_matrix = np.zeros((self.num_class,) * 2)

    def get_pixel_prec_class(self, harmonic_mean=False):
        """Pixel Precision"""
        prec_by_class = np.diag(self.confusion_matrix) / np.nansum(self.confusion_matrix, axis=0)
        prec_by_class = np.nan_to_num(prec_by_class)
        if harmonic_mean:
            prec = hmean(prec_by_class)
        else:
            prec = np.mean(prec_by_class)
        return prec, prec_by_class

    def get_pixel_recall_class(self, harmonic_mean=False):
        """Pixel Recall"""
        recall_by_class = np.diag(self.confusion_matrix) / np.nansum(self.confusion_matrix, axis=1)
        recall_by_class = np.nan_to_num(recall_by_class)
        if harmonic_mean:
            recall = hmean(recall_by_class)
        else:
            recall = np.mean(recall_by_class)
        return recall, recall_by_class

    def get_pixel_acc_class(self, harmonic_mean=False):
        """Pixel Accuracy"""
        acc_by_class = np.diag(self.confusion_matrix).sum() / (np.nansum(self.confusion_matrix, axis=1)
                                                               + np.nansum(self.confusion_matrix, axis=0)
                                                               + np.diag(self.confusion_matrix).sum()
                                                               - 2*np.diag(self.confusion_matrix))
        acc_by_class = np.nan_to_num(acc_by_class)
        if harmonic_mean:
            acc = hmean(acc_by_class)
        else:
            acc = np.mean(acc_by_class)
        return acc, acc_by_class

    def get_miou_class(self, harmonic_mean=False):
        """Mean Intersection over Union"""
        miou_by_class = np.diag(self.confusion_matrix) / (np.nansum(self.confusion_matrix, axis=1)
                                                          + np.nansum(self.confusion_matrix, axis=0)
                                                          - np.diag(self.confusion_matrix))
        miou_by_class = np.nan_to_num(miou_by_class)
        if harmonic_mean:
            miou = hmean(miou_by_class)
        else:
            miou = np.mean(miou_by_class)
        return miou, miou_by_class

    def get_dice_class(self, harmonic_mean=False):
        """Dice"""
        _, prec_by_class = self.get_pixel_prec_class()
        _, recall_by_class = self.get_pixel_recall_class()
        # Add epsilon term to avoid /0
        dice_by_class = 2*prec_by_class*recall_by_class/(prec_by_class + recall_by_class + 1e-8)
        if harmonic_mean:
            dice = hmean(dice_by_class)
        else:
            dice = np.mean(dice_by_class)
        return dice, dice_by_class

    def _generate_matrix(self, labels, predictions):
        matrix = confusion_matrix(labels.flatten(), predictions.flatten(),
                                  labels=list(range(self.num_class)))
        return matrix

    def add_batch(self, labels, predictions):
        """Method to add ground truth and predicted masks by batch
        and update the global confusion matrix (entire dataset)

        PARAMETERS
        ----------
        labels: torch tensor or numpy array
            Ground truth masks
        predictions: torch tensor or numpy array
            Predicted masks
        """
        assert labels.shape == predictions.shape
        self.confusion_matrix += self._generate_matrix(labels, predictions)

    def reset(self):
        """Method to reset the confusion matrix"""
        self.confusion_matrix = np.zeros((self.num_class,) * 2)


def RECORD_CARRADA_evaluation(net, dataloader, features, criterion, device):
    net.eval()
    metrics = Evaluator(4) # number of classes
    kbar = pkbar.Kbar(target=len(dataloader), width=20, always_stateful=False)
    running_loss = 0.0
    for iter, data in enumerate(dataloader):
        print(f"Sample {iter}")
        if features == ['RD']:
            input = data['rd_matrix'].to(device).float()
            label = data['rd_mask'].to(device).float()
        elif features == ['RA']:
            input = data['ra_matrix'].to(device).float()
            label = data['ra_mask'].to(device).float()
     
        with torch.set_grad_enabled(False):
            outputs = net(input)

        losses = [c(outputs, torch.argmax(label, axis=1)) for c in criterion]
        loss = torch.mean(torch.stack(losses))
        running_loss += loss.item() * input.size(0)
        metrics.add_batch(torch.argmax(label, axis=1).cpu(), torch.argmax(outputs, axis=1).cpu())
        kbar.update(iter)

    metrics_dict = get_metrics(metrics)

    mAP = sum(metrics_dict['prec_by_class']) / len(metrics_dict['prec_by_class'])
    mAR = sum(metrics_dict['recall_by_class']) / len(metrics_dict['recall_by_class'])
    return {'loss':running_loss, 'mAP':mAP, 'mAR':mAR, 'mIoU':metrics_dict['miou']}
   

def MVRECORD_CARRADA_evaluation(net, dataloader, features, rd_criterion, ra_criterion, device):
    net.eval()
    rd_metrics = Evaluator(4) # number of classes
    ra_metrics = Evaluator(4) # number of classes
    kbar = pkbar.Kbar(target=len(dataloader), width=20, always_stateful=False)
    running_loss = 0.0
    for iter, data in enumerate(dataloader):
        print(f"Sample {iter}")
        input = (data['rd_matrix'].to(device).float(), data['ra_matrix'].to(device).float(), data['ad_matrix'].to(device).float())
        label = {'rd': data['rd_mask'].to(device).float(), 'ra': data['ra_mask'].to(device).float()}
            
        with torch.set_grad_enabled(False):
            outputs = net(input)

        rd_losses = [c(outputs['rd'], torch.argmax(label['rd'], axis=1)) for c in rd_criterion]
        rd_loss = torch.mean(torch.stack(rd_losses))
        
        ra_losses = [c(outputs['ra'], torch.argmax(label['ra'], axis=1)) for c in ra_criterion]
        ra_loss = torch.mean(torch.stack(ra_losses))

        loss = torch.mean(rd_loss + ra_loss)
        running_loss += loss.item() * input[0].size(0)

        rd_metrics.add_batch(torch.argmax(label['rd'], axis=1).cpu(), torch.argmax(outputs['rd'], axis=1).cpu())
        ra_metrics.add_batch(torch.argmax(label['ra'], axis=1).cpu(), torch.argmax(outputs['ra'], axis=1).cpu())
        kbar.update(iter)

    metrics_dict = dict()
    metrics_dict['range_doppler'] = get_metrics(rd_metrics)
    metrics_dict['range_angle'] = get_metrics(ra_metrics)

    metrics_dict['global_acc'] = (1/2)*(metrics_dict['range_doppler']['acc'] + metrics_dict['range_angle']['acc'])
    metrics_dict['global_prec'] = (1/2)*(metrics_dict['range_doppler']['prec'] + metrics_dict['range_angle']['prec'])
    metrics_dict['global_dice'] = (1/2)*(metrics_dict['range_doppler']['dice'] + metrics_dict['range_angle']['dice'])

    mAP = sum(metrics_dict['range_doppler']['prec_by_class']) / len(metrics_dict['range_doppler']['prec_by_class']) / 2 + \
            sum(metrics_dict['range_angle']['prec_by_class']) / len(metrics_dict['range_angle']['prec_by_class']) / 2 
    mAR = sum(metrics_dict['range_doppler']['recall_by_class']) / len(metrics_dict['range_doppler']['recall_by_class']) / 2 + \
            sum(metrics_dict['range_angle']['recall_by_class']) / len(metrics_dict['range_angle']['recall_by_class']) / 2
    return {'loss': running_loss, 
            'mAP': mAP, 
            'mAR': mAR, 
            'mIoU': (metrics_dict['range_doppler']['miou'] + metrics_dict['range_doppler']['miou'])/2}


def yoloheadToPredictions(yolohead_output, conf_threshold=0.5):
    """ Transfer YOLO HEAD output to [:, 8], where 8 means
    [x, y, z, w, h, d, score, class_index]"""
    prediction = yolohead_output.reshape(-1, yolohead_output.shape[-1])
    prediction_class = np.argmax(prediction[:, 7:], axis=-1)
    predictions = np.concatenate([prediction[:, :7], \
                    np.expand_dims(prediction_class, axis=-1)], axis=-1)
    conf_mask = (predictions[:, 6] >= conf_threshold)
    predictions = predictions[conf_mask]
    return predictions


def iou2d(box_xywh_1, box_xywh_2):
    """ Numpy version of 3D bounding box IOU calculation 
    Args:
        box_xywh_1        ->      box1 [x, y, w, h]
        box_xywh_2        ->      box2 [x, y, w, h]"""
    assert box_xywh_1.shape[-1] == 4
    assert box_xywh_2.shape[-1] == 4
    ### areas of both boxes
    box1_area = box_xywh_1[..., 2] * box_xywh_1[..., 3]
    box2_area = box_xywh_2[..., 2] * box_xywh_2[..., 3]
    ### find the intersection box
    box1_min = box_xywh_1[..., :2] - box_xywh_1[..., 2:] * 0.5
    box1_max = box_xywh_1[..., :2] + box_xywh_1[..., 2:] * 0.5
    box2_min = box_xywh_2[..., :2] - box_xywh_2[..., 2:] * 0.5
    box2_max = box_xywh_2[..., :2] + box_xywh_2[..., 2:] * 0.5

    left_top = np.maximum(box1_min, box2_min)
    bottom_right = np.minimum(box1_max, box2_max)
    ### get intersection area
    intersection = np.maximum(bottom_right - left_top, 0.0)
    intersection_area = intersection[..., 0] * intersection[..., 1]
    ### get union area
    union_area = box1_area + box2_area - intersection_area
    ### get iou
    iou = np.nan_to_num(intersection_area / (union_area + 1e-10))
    return iou


def nms(bboxes, iou_threshold, input_size, sigma=0.3, method='nms'):
    """ Bboxes format [x, y, z, w, h, d, score, class_index] """
    """ Implemented the same way as YOLOv4 """ 
    assert method in ['nms', 'soft-nms']
    if len(bboxes) == 0:
        best_bboxes = np.zeros([0, 8])
    else:
        all_pred_classes = list(set(bboxes[:, 7]))
        unique_classes = list(np.unique(all_pred_classes))
        best_bboxes = []
        for cls in unique_classes:
            cls_mask = (bboxes[:, 7] == cls)
            cls_bboxes = bboxes[cls_mask]
            ### NOTE: start looping over boxes to find the best one ###
            while len(cls_bboxes) > 0:
                max_ind = np.argmax(cls_bboxes[:, 6])
                best_bbox = cls_bboxes[max_ind]
                best_bboxes.append(best_bbox)
                cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
                iou = iou3d(best_bbox[np.newaxis, :6], cls_bboxes[:, :6], \
                            input_size)
                weight = np.ones((len(iou),), dtype=np.float32)
                if method == 'nms':
                    iou_mask = iou > iou_threshold
                    weight[iou_mask] = 0.0
                if method == 'soft-nms':
                    weight = np.exp(-(1.0 * iou ** 2 / sigma))
                cls_bboxes[:, 6] = cls_bboxes[:, 6] * weight
                score_mask = cls_bboxes[:, 6] > 0.
                cls_bboxes = cls_bboxes[score_mask]
        if len(best_bboxes) != 0:
            best_bboxes = np.array(best_bboxes)
        else:
            best_bboxes = np.zeros([0, 8])
    return best_bboxes


def getTruePositive(pred, gt, input_size, iou_threshold=0.5, mode="3D"):
    """ output tp (true positive) with size [num_pred, ] """
    assert mode in ["3D", "2D"]
    tp = np.zeros(len(pred))
    detected_gt_boxes = []
    for i in range(len(pred)):
        current_pred = pred[i]
        if mode == "3D":
            current_pred_box = current_pred[:6]
            current_pred_score = current_pred[6]
            current_pred_class = current_pred[7]
            gt_box = gt[..., :6]
            gt_class = gt[..., 6]
        else:
            current_pred_box = current_pred[:4]
            current_pred_score = current_pred[4]
            current_pred_class = current_pred[5]
            gt_box = gt[..., :4]
            gt_class = gt[..., 4]

        if len(detected_gt_boxes) == len(gt): break
        
        if mode == "3D":
            iou = iou3d(current_pred_box[np.newaxis, ...], gt_box, input_size)
        else:
            iou = iou2d(current_pred_box[np.newaxis, ...], gt_box)
        iou_max_idx = np.argmax(iou)
        iou_max = iou[iou_max_idx]
        if iou_max >= iou_threshold and iou_max_idx not in detected_gt_boxes:
            tp[i] = 1.
            detected_gt_boxes.append(iou_max_idx)
    fp = 1. - tp
    return tp, fp


def computeAP(tp, fp, num_gt_class):
    """ Compute Average Precision """
    tp_cumsum = np.cumsum(tp).astype(np.float32)
    fp_cumsum = np.cumsum(fp).astype(np.float32)
    recall = tp_cumsum / (num_gt_class + 1e-16)
    precision = tp_cumsum / (tp_cumsum + fp_cumsum)
    ########## NOTE: the following is under the reference of the repo ###########
    recall = np.insert(recall, 0, 0.0)
    recall = np.append(recall, 1.0)
    precision = np.insert(precision, 0, 0.0)
    precision = np.append(precision, 0.0)
    mrec = recall.copy()
    mpre = precision.copy()

    for i in range(len(mpre)-2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i+1])

    i_list = []
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i-1]:
            i_list.append(i) # if it was matlab would be i + 1

    ap = 0.0
    for i in i_list:
        ap += ((mrec[i]-mrec[i-1])*mpre[i])
    return ap, mrec, mpre


def mAP(predictions, gts, input_size, ap_each_class, tp_iou_threshold=0.5, mode="3D"):
    """ Main function for calculating mAP 
    Args:
        predictions         ->      [num_pred, 6 + score + class]
        gts                 ->      [num_gt, 6 + class]"""
    gts = gts[gts[..., :6].any(axis=-1) > 0]
    all_gt_classes = np.unique(gts[:, 6])
    ap_all = []
    # ap_all_classes = np.zeros(num_all_classes).astype(np.float32)
    for class_i in all_gt_classes:
        ### NOTE: get the prediction per class and sort it ###
        pred_class = predictions[predictions[..., 7] == class_i]
        pred_class = pred_class[np.argsort(pred_class[..., 6])[::-1]]
        ### NOTE: get the ground truth per class ###
        gt_class = gts[gts[..., 6] == class_i]
        tp, fp = getTruePositive(pred_class, gt_class, input_size, \
                                iou_threshold=tp_iou_threshold, mode=mode)
        ap, mrecall, mprecision = computeAP(tp, fp, len(gt_class))
        ap_all.append(ap)
        ap_each_class[int(class_i)].append(ap)
    mean_ap = np.mean(ap_all)
    return mean_ap, ap_each_class


def RADDet_evaluation(net, dataloader, batch_size, model_config, train_config, device):
    net.eval()
    kbar = pkbar.Kbar(target=len(dataloader), width=20, always_stateful=False)
    running_loss = 0.0
    mean_ap_test = 0.0
    ap_all_class_test = []
    ap_all_class = []
    for class_id in range(model_config['num_class']):
        ap_all_class.append([])
    with torch.set_grad_enabled(False):
        for iter, data in enumerate(dataloader):
            inputs = data['radar_data'].to(device).float().permute(0, 3, 1, 2)
            label = data['label'].to(device).float()
            raw_boxes = data['boxes'].to(device).float()
            outputs = net(inputs)
            pred_raw, pred = boxDecoder(outputs, train_config['input_size'], train_config['anchor_boxes'], model_config['num_class'], train_config['yolohead_xyz_scales'][0], device)
            box_loss, conf_loss, category_loss = lossYolo(pred_raw, pred, label, raw_boxes[..., :6], train_config['input_size'], train_config['focal_loss_iou_threshold'])
            box_loss *= 1e-1
            loss = box_loss + conf_loss + category_loss
            running_loss += loss.item() * inputs.size(0)
            pred = pred.detach().cpu().numpy()
            raw_boxes = raw_boxes.detach().cpu().numpy()
            for batch_id in range(raw_boxes.shape[0]):
                raw_boxes_frame = raw_boxes[batch_id]
                pred_frame = pred[batch_id]
                predicitons = yoloheadToPredictions(pred_frame, \
                                    conf_threshold=train_config["confidence_threshold"])
                nms_pred = nms(predicitons, train_config["nms_iou3d_threshold"], \
                                train_config["input_size"], sigma=0.3, method="nms")
                mean_ap, ap_all_class = mAP(nms_pred, raw_boxes_frame, \
                                        train_config["input_size"], ap_all_class, \
                                        tp_iou_threshold=train_config["mAP_iou3d_threshold"])
                mean_ap_test += mean_ap
            kbar.update(iter)
    for ap_class_i in ap_all_class:
        if len(ap_class_i) == 0:
            class_ap = 0.
        else:
            class_ap = np.mean(ap_class_i)
        ap_all_class_test.append(class_ap)
    mean_ap_test /= batch_size * len(dataloader)
    print("-------> ap: %.6f"%(mean_ap_test))
    return {'loss': running_loss, 
            'mAP': mean_ap_test, 
            'mAR': 0.0, 
            'mIoU': 0.0}