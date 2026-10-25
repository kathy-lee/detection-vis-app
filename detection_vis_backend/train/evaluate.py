import numpy as np
import pandas as pd
import pkbar
import torch
import time
import os
import math
import json
import cv2
import torch.nn as nn


from shapely.geometry import Polygon
from scipy.stats import hmean
from sklearn.metrics import confusion_matrix
from copy import deepcopy
from loguru import logger


from detection_vis_backend.train.utils import decode, RA_to_cartesian_box, bbox_iou, get_class_name, get_metrics, process_predictions_FFT, post_process_single_frame, get_ols_btw_objects, yoloheadToPredictions, nms, create_default, peaks_detect, distribute, update_peaks, association, pol2cord, orent, distance
from detection_vis_backend.datasets.utils import confmap2ra, get_class_id, iou3d
from detection_vis_backend.networks.transrad import decode_predictions as transrad_decode, nms_class_then_location as transrad_nms



def FFTRadNet_val_evaluation(net, loader, train_config, check_perf=True, device='cpu'):
    """Per-epoch validation, a port of the reference's run_evaluation
    (RADIal FFTRadNet utils/evaluation.py:6-64).

    ONE confidence threshold (0.2) through the Metrics class, not the 0.1..0.9 sweep of
    GetFullMetrics -- that sweep is run_FullEvaluation, which the reference invokes once
    from 3-Evaluation.py on the TEST split to produce the paper's Table 2. Running it
    every epoch costs 9x and reports a quantity the reference never reports per-epoch;
    it is also what made this pair's validation unfinishable before the metric helpers
    were vectorized. Dispatch lives in FFTRadNet_evaluation below.

    check_perf defaults True: the reference passes check_perf=(epoch>=0), i.e. always.
    """
    net.eval()
    running_loss = 0.0
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)
    metrics = Metrics()
    metrics.reset()

    for i, data in enumerate(loader):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device).float()

        inputs = data['RD']
        with torch.set_grad_enabled(False):
            outputs = net(inputs)
    
        data.pop('RD')
        loss = net.get_loss(outputs, data, train_config)
        running_loss += loss.item() * inputs.size(0)

        if(check_perf):
            out_obj = outputs['Detection'].detach().cpu().numpy().copy()
            label_obj = data['box_label']

            out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
            label_freespace = data['seg_label'].detach().cpu().numpy().copy()

            for pred_obj,pred_map,true_obj,true_map in zip(out_obj,out_seg,label_obj,label_freespace):
                metrics.update(pred_map[0],true_map,np.asarray(decode(pred_obj,0.05)),true_obj,threshold=0.2,range_min=5,range_max=100) 
                
        kbar.update(i)
        
    mAP,mAR, mIoU = metrics.GetMetrics()
    # The reference returns the raw running sum; normalized here so val_loss is
    # comparable across splits and runs, as every other pair in this repo reports it.
    return {'loss':running_loss / len(loader.dataset), 'mAP':mAP, 'mAR':mAR, 'mIoU':mIoU}


def FFTRadNet_evaluation(net, loader, train_config, features, output_path, eval_type, device='cpu'):
    """Split the reference's two evaluations the way the reference does.

    val  -> run_evaluation      : single threshold 0.2, cheap, every epoch.
    test -> run_FullEvaluation  : GetFullMetrics sweeping 0.1..0.9 plus the freespace
                                  mIoU over [0, 50] m -- the paper's Table 2 / Table 3.
    """
    if eval_type == "val":
        return FFTRadNet_val_evaluation(net, loader, train_config, check_perf=True, device=device)
    return FFTRadNet_full_evaluation(net, loader, train_config, features, output_path, eval_type, device)


def FFTRadNet_full_evaluation(net, loader, train_config, features, output_path, eval_type, device='cpu'):
    net.eval()
    running_loss = 0.0
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    predictions = {'prediction':{'objects':[],'freespace':[]},'label':{'objects':[],'freespace':[]}}
    for iter, data in enumerate(loader):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device).float()
        
        inputs = data['RD'].to(device).float()
        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        if eval_type == "val":
            loss = net.get_loss(outputs, data, train_config)
            running_loss += loss.item() * inputs.size(0)
            logger.info(f"Val sample {iter} loss: {loss}")

        out_obj = outputs['Detection'].detach().cpu().numpy().copy()
        out_seg = torch.sigmoid(outputs['Segmentation']).detach().cpu().numpy().copy()
        
        labels_obj = data['box_label']
        label_freespace = data['seg_label'].detach().cpu().numpy().copy()
            
        for pred_obj,pred_map,true_obj,true_map in zip(out_obj,out_seg,labels_obj,label_freespace):
            predictions['prediction']['objects'].append( np.asarray(decode(pred_obj,0.05)))
            predictions['label']['objects'].append(true_obj)

            predictions['prediction']['freespace'].append(pred_map[0])
            predictions['label']['freespace'].append(true_map)   

        if eval_type == "val":
            kbar.update(iter, values=[("loss", loss.item())])
        else:
            kbar.update(iter)
  
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
    logger.info('------- Freespace Scores ------------')
    logger.info(f'  mIoU: {mIoU*100} %')

    result = {'mAP': mAP, 'mAR': mAR, 'F1_score': F1_score, 'mIoU': mIoU}
    if eval_type == 'val':
        result.update({'loss': running_loss / len(loader.dataset)})
    return result


def GetFullMetrics(predictions,object_labels,range_min=5,range_max=100,IOU_threshold=0.5):
    perfs = {}
    precision = []
    recall = []
    iou_threshold = []
    RangeError = []
    AngleError = []

    # The reference sweeps 0.1..0.9 (utils/metrics.py:120); this loop started at 0.2 from
    # 2023-08 to 2026-09, which drops the lowest-precision point from the mean and so
    # overstated mAP relative to the published protocol.
    for threshold in np.arange(0.1,0.96,0.1):
        logger.info(f"\nEvaluation with threshold = {threshold}")
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
            precision.append(0) 
            recall.append(0)

        if nbObjects > 0:
            RangeError.append(range_error/nbObjects)
            AngleError.append(angle_error/nbObjects)
        
    perfs['precision']=precision
    perfs['recall']=recall

    mAP_ = float(np.mean(precision))
    mAR_ = float(np.mean(recall))
    # A model that detects nothing at any threshold gives 0/0 here. The reference returns
    # nan; report 0 instead, because a nan is indistinguishable from a broken metric --
    # the same confusion the RadarCrossAttention evaluator's absent-class nan caused.
    F1_score = (mAP_ * mAR_) / ((mAP_ + mAR_) / 2) if (mAP_ + mAR_) > 0 else 0.0
    logger.info('------- Detection Scores ------------')
    logger.info(f"  mAP: {mAP_}")
    logger.info(f"  mAR: {mAR_}")
    logger.info(f"  F1 score: {F1_score}")

    logger.info('------- Regression Errors------------')
    # Empty when no threshold produced a true positive; np.mean([]) is nan with a warning.
    logger.info(f"  Range Error: {np.mean(RangeError) if RangeError else float('nan')} m")
    logger.info(f"  Angle Error: {np.mean(AngleError) if AngleError else float('nan')} degree")

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

        TP,FP,FN = GetDetMetrics(ObjectPred,Objectlabels,threshold=threshold,range_min=range_min,range_max=range_max)

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


def evaluate_img(gts_dict, dts_dict, imgId, catId, olss_dict, olsThrs, recThrs, classes):
    gts = gts_dict[imgId, catId]
    dts = dts_dict[imgId, catId]
    if len(gts) == 0 and len(dts) == 0:
        return None

    olss_flatten = np.ravel(olss_dict[imgId, catId])
    logger.debug("Frame %d: %10s %s" % (imgId, classes[catId], list(olss_flatten)))

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


def evaluate_rodnet_seq(res_path, gt_path, n_frame, n_class, classes, rng_grid, agl_grid, olsThrs, recThrs, frame_ids=None):

    gt_dets = read_gt_txt(gt_path, n_frame, n_class, classes)
    sub_dets = read_rodnet_res(res_path, n_frame, n_class, classes, rng_grid, agl_grid)
    if frame_ids is not None:
        # Score only these frames, on both sides of the match. Callers whose
        # loader does not cover every frame (RECORD's val loader strides by
        # train_stride but the model predicts one frame per window) pass the
        # covered set here; without the filter, every uncovered frame's GT
        # counts as misses.
        gt_dets = {key: (objs if key[0] in frame_ids else []) for key, objs in gt_dets.items()}
        if sub_dets:
            sub_dets = {key: (objs if key[0] in frame_ids else []) for key, objs in sub_dets.items()}
    if sub_dets:
        olss_all = {(imgId, catId): compute_ols_dts_gts(gt_dets, sub_dets, imgId, catId) \
                    for imgId in range(n_frame)
                    for catId in range(3)}

        evalImgs = [evaluate_img(gt_dets, sub_dets, imgId, catId, olss_all, olsThrs, recThrs, classes)
                    for imgId in range(n_frame)
                    for catId in range(3)]
    else:
        evalImgs = None
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


def accumulate(evalImgs, n_frame, olsThrs, recThrs, n_class, classes):
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

        logger.debug("%10s: %4d dets, %4d gts" % (classes[classid], dtm.shape[1], gtm.shape[1]))

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


def RODNet_evaluation(net, dataloader, train_config, features, save_dir, eval_type, device):
    root_path = "/home/kangle/dataset/CRUW"
    with open(os.path.join(root_path, 'sensor_config_rod2021.json'), 'r') as file:
        sensor_cfg = json.load(file)
    radar_cfg = sensor_cfg['radar_cfg']
    n_class = 3 
    classes = ["pedestrian", "cyclist", "car"]  
    rng_grid = confmap2ra(radar_cfg, name='range')
    agl_grid = confmap2ra(radar_cfg, name='angle')

    net.eval()
    running_loss = 0.0
    kbar = pkbar.Kbar(target=len(dataloader), width=20, always_stateful=False)
    batch_size = train_config['dataloader'][eval_type]['batch_size']
    # 1.Generate network output (confmaps) and post-process them to the form of detection predictions
    confmap_shape = (3, 128, 128) # (n_class, radar_cfg['ramap_rsize'], radar_cfg['ramap_asize'])
    init_genConfmap = ConfmapStack(confmap_shape)
    iter_ = init_genConfmap
    for i in range(train_config['win_size'] - 1):
        while iter_.next is not None:
            iter_ = iter_.next
        iter_.next = ConfmapStack(confmap_shape)

    total_time = 0
    total_count = 0
    load_tic = time.time()
    sequences = []
    # The val loader windows at train_stride, the test loader at test_stride
    # (prepare_for_train mirrors the reference CRDataset's split logic), and the
    # pop count below must match the loader's advance -- with a mismatch, part of
    # the frames are never written and score as misses.
    stride = train_config['test_stride'] if eval_type == 'test' else train_config['train_stride']
    for iter, data in enumerate(dataloader):
        load_time = time.time() - load_tic

        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device).float()
        
        inputs = data['RA']
        tic = time.time()
        with torch.set_grad_enabled(False):
            outputs = net(inputs)
        
        if hasattr(net, 'stacked_num'):
            confmap_pred = outputs[-1].cpu().detach().numpy()  # (1, 4, 32, 128, 128)
        else:
            confmap_pred = outputs.cpu().detach().numpy()

        infer_time = time.time() - tic
        total_time += infer_time

        if eval_type == "val":
            loss = net.get_loss(outputs, data, train_config)
            # get_loss reduces over the ACTUAL batch, so weight by that, not by the
            # configured batch_size: the final batch is short whenever the split size
            # is not a multiple of it, and would otherwise be over-weighted.
            running_loss += loss.item() * inputs.shape[0]
            logger.info(f"Val sample {iter} loss: {loss}")
        
        seq_name = data['seq_name'][0]
        if seq_name not in sequences:
            # Flush the previous sequence's tail before starting the next one. The
            # reference builds one dataset per sequence and resets its stack in
            # between; our single loader spans all sequences, so without this the
            # last win_size - stride frames of each sequence are never written and
            # the next sequence's first windows average against the stale tail.
            if sequences:
                offset = prev_start_frame + stride
                while init_genConfmap is not None:
                    total_count += 1
                    res_final = post_process_single_frame(init_genConfmap.confmap, train_config, n_class, rng_grid, agl_grid)
                    write_dets_results_single_frame(res_final, offset, save_path, classes)
                    init_genConfmap = init_genConfmap.next
                    offset += 1
                init_genConfmap = ConfmapStack(confmap_shape)
            sequences.append(seq_name)
            save_path = os.path.join(save_dir, seq_name + '_rod_res.txt')
            # write_dets_results_single_frame appends, and val runs every epoch into the
            # same exp dir -- without this, epoch N is scored against the union of all
            # detections from epochs <= N (duplicates count as false positives, so mAP
            # decays monotonically while mAR climbs).
            if os.path.exists(save_path):
                os.remove(save_path)
        # Currently only support batch_size set to 1 for val evaluation
        start_frame_id = data['start_frame'].item()
        end_frame_id = data['end_frame'].item()
        
        iter_ = init_genConfmap
        for i in range(confmap_pred.shape[2]):
            if iter_.next is None and i != confmap_pred.shape[2] - 1:
                iter_.next = ConfmapStack(confmap_shape)
            iter_.append(confmap_pred[0, :, i, :, :])
            iter_ = iter_.next
        
        process_tic = time.time()
        
        for i in range(stride):
            total_count += 1
            res_final = post_process_single_frame(init_genConfmap.confmap, train_config, n_class, rng_grid, agl_grid)
            cur_frame_id = start_frame_id + i
            write_dets_results_single_frame(res_final, cur_frame_id, save_path, classes)
            init_genConfmap = init_genConfmap.next
        prev_start_frame = start_frame_id

        if iter == len(dataloader) - 1:
            offset = stride
            cur_frame_id = start_frame_id + offset
            while init_genConfmap is not None:
                total_count += 1
                res_final = post_process_single_frame(init_genConfmap.confmap, train_config, n_class, rng_grid, agl_grid)
                write_dets_results_single_frame(res_final, cur_frame_id, save_path, classes)
                init_genConfmap = init_genConfmap.next
                offset += 1
                cur_frame_id += 1
        
        if eval_type == "val":
            kbar.update(iter, values=[("loss", loss.item())])
        else:
            kbar.update(iter)

        if init_genConfmap is None:
            init_genConfmap = ConfmapStack(confmap_shape)

        proc_time = time.time() - process_tic
        logger.info("Testing %s: frame %4d to %4d | Load time: %.4f | Inference time: %.4f | Process time: %.4f" %
                (seq_name, start_frame_id, end_frame_id, load_time, infer_time, proc_time))

        load_tic = time.time()
    logger.info("Ave infer time: %f" % (total_time / total_count))

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
        
        if evalImgs:
            eval = accumulate(evalImgs, n_frame, olsThrs, recThrs, n_class, classes)
            stats = summarize(eval, olsThrs, recThrs, n_class, gl=False)
            logger.info("%s | AP_total: %.4f | AR_total: %.4f" % (seq_name.upper(), stats[0] * 100, stats[1] * 100))
            n_frames_all += n_frame
            evalImgs_all.extend(evalImgs)
        else:
            logger.info(f"{seq_name}_rod_res.txt is empty.")
            
    if evalImgs_all:    
        eval = accumulate(evalImgs_all, n_frames_all, olsThrs, recThrs, n_class, classes)
        stats = summarize(eval, olsThrs, recThrs, n_class, gl=False)
        logger.info("%s | AP_total: %.4f | AR_total: %.4f" % ('Overall'.ljust(18), stats[0] * 100, stats[1] * 100))
        result = {'mAP': stats[0], 'mAR': stats[1], 'F1_score': 0, 'mIoU': 0}
    else:
        result = {'mAP': 0, 'mAR': 0, 'F1_score': 0, 'mIoU': 0}

    if eval_type == 'val':
        result.update({'loss': running_loss / len(dataloader.dataset)})
    return result


def RECORD_CRUW_evaluation(net, dataloader, train_config, features, save_dir, eval_type, device):
    root_path = "/home/kangle/dataset/CRUW"
    with open(os.path.join(root_path, 'sensor_config_rod2021.json'), 'r') as file:
        sensor_cfg = json.load(file)
    radar_cfg = sensor_cfg['radar_cfg']
    n_class = 3 # dataset.object_cfg.n_class
    classes = ["pedestrian", "cyclist", "car"]  # dataset.object_cfg.classes
    rng_grid = confmap2ra(radar_cfg, name='range')
    agl_grid = confmap2ra(radar_cfg, name='angle')

    net.eval()
    running_loss = 0.0
    kbar = pkbar.Kbar(target=len(dataloader), width=20, always_stateful=False)
    batch_size = train_config['dataloader'][eval_type]['batch_size']
    sequences = []
    # The model predicts one frame per window, so frame coverage equals the
    # loader's stride: complete at test_stride 1, but only every train_stride-th
    # frame on the val loader. Track what was actually written so the val score
    # is computed over exactly those frames (an unbiased uniform sample) instead
    # of counting the uncovered frames' GT as misses. The NoLstm variants take a
    # fixed-size window, so predicting the skipped frames is not an option.
    frames_written = {}
    for iter, data in enumerate(dataloader):
        seq_name = data['seq_name'][0]
        if seq_name not in sequences:
            sequences.append(seq_name)
            frames_written[seq_name] = set()
            save_path = os.path.join(save_dir, seq_name + '_record_res.txt')
            # Same append-across-epochs trap as RODNet_evaluation above.
            if os.path.exists(save_path):
                os.remove(save_path)

        for key in data:
            logger.debug(f"data contains key: {key}")
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device).float()
                logger.debug(f"Put {key} feature to GPU: {data[key]}")

        inputs = data['RA']
        logger.debug('get input data')

        frame_id = int(data['end_frame'].item())
        if frame_id == train_config['win_size']-1 and type(net).__name__ not in ('RECORDNoLstm', 'RECORDNoLstmMulti'):
            for tmp_frame_id in range(frame_id):
                tmp_ra_maps = inputs[:, :, :tmp_frame_id+1]
                with torch.set_grad_enabled(False):
                    confmap_pred = net(tmp_ra_maps)
                res_final = post_process_single_frame(confmap_pred[0].cpu(), train_config, n_class, rng_grid, agl_grid)
                write_dets_results_single_frame(res_final, tmp_frame_id, save_path, classes)
                frames_written[seq_name].add(tmp_frame_id)

        with torch.set_grad_enabled(False):
            confmap_pred = net(inputs)
        logger.debug('get network output')

        res_final = post_process_single_frame(confmap_pred[0].cpu(), train_config, n_class, rng_grid, agl_grid)
        write_dets_results_single_frame(res_final, frame_id, save_path, classes)
        frames_written[seq_name].add(frame_id)

        if eval_type == "val":
            loss = net.get_loss(confmap_pred, data, train_config)
            running_loss += loss.item() * inputs.shape[0]   # actual batch, not configured
            logger.info(f"Val sample {iter} loss: {loss}")

        if eval_type == "val":
            kbar.update(iter, values=[("loss", loss.item())])
        else:
            kbar.update(iter)
    logger.info(f'record_res.txt file(s) for {sequences} created')

    # 2.Evaluation the detection predictions with Ground-truth annotations
    evalImgs_all = []
    n_frames_all = 0
    for seq_name in sequences:
        res_path = os.path.join(save_dir, seq_name + '_record_res.txt')
        gt_path = os.path.join(root_path, 'TRAIN_RAD_H_ANNO', seq_name + '.txt')
        n_frame = len(os.listdir(os.path.join(root_path, 'TRAIN_CAM_0', seq_name, 'IMAGES_0')))

        olsThrs = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1), endpoint=True), decimals=2)
        recThrs = np.around(np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01) + 1), endpoint=True), decimals=2)
        # Val-only frame filter; the test path (stride 1, full coverage) keeps
        # the reference-faithful scoring over all frames.
        frame_filter = frames_written[seq_name] if eval_type == 'val' else None
        evalImgs = evaluate_rodnet_seq(res_path, gt_path, n_frame, n_class, classes, rng_grid, agl_grid, olsThrs, recThrs, frame_ids=frame_filter)

        if evalImgs:
            eval = accumulate(evalImgs, n_frame, olsThrs, recThrs, n_class, classes)
            stats = summarize(eval, olsThrs, recThrs, n_class, gl=False)
            logger.info("%s | AP_total: %.4f | AR_total: %.4f" % (seq_name.upper(), stats[0] * 100, stats[1] * 100))
            n_frames_all += n_frame
            evalImgs_all.extend(evalImgs)
        else:
            logger.info(f"{seq_name}_rod_res.txt is empty.")

    if evalImgs_all:    
        eval = accumulate(evalImgs_all, n_frames_all, olsThrs, recThrs, n_class, classes)
        stats = summarize(eval, olsThrs, recThrs, n_class, gl=False)
        logger.info("%s | AP_total: %.4f | AR_total: %.4f" % ('Overall'.ljust(18), stats[0] * 100, stats[1] * 100))
        result = {'mAP': stats[0], 'mAR': stats[1], 'F1_score': 0, 'mIoU': 0}
    else:
        result = {'mAP': 0, 'mAR': 0, 'F1_score': 0, 'mIoU': 0}

    if eval_type == 'val':
        result.update({'loss': running_loss / len(dataloader.dataset)})
    return result


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


def RECORD_CARRADA_evaluation(net, dataloader, train_config, features, output_dir, eval_type, device):
    net.eval()
    metrics = Evaluator(net.n_class) 
    kbar = pkbar.Kbar(target=len(dataloader), width=20, always_stateful=False)
    running_loss = 0.0
    batch_size = train_config['dataloader'][eval_type]['batch_size']
    for iter, data in enumerate(dataloader):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device).float()
        
        inputs = data[features[0]]
        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        if eval_type == "val":
            loss = net.get_loss(outputs, data, train_config)
            running_loss += loss.item() * inputs.shape[0]   # actual batch, not configured
            logger.info(f"Val sample {iter} loss: {loss}")
        
        metrics.add_batch(torch.argmax(data['gt_mask'], axis=1).cpu(), torch.argmax(outputs, axis=1).cpu())
        if eval_type == "val":
            kbar.update(iter, values=[("loss", loss.item())])
        else:
            kbar.update(iter)

    metrics_dict = get_metrics(metrics)

    mAP = sum(metrics_dict['prec_by_class']) / len(metrics_dict['prec_by_class'])
    mAR = sum(metrics_dict['recall_by_class']) / len(metrics_dict['recall_by_class'])
    result = {'mAP':mAP, 'mAR':mAR, 'mIoU': metrics_dict['miou']}
    if eval_type == 'val':
        result.update({'loss': running_loss / len(dataloader.dataset)})
    return  result
   

def T_RODNet_CARRADA_evaluation(net, dataloader, train_config, features, output_dir, eval_type, device):
    """RA semantic segmentation as the T-RODNet paper reports it on CARRADA (Table II:
    per-class IoU and Dice, their means). The body predicts every frame of the 4-frame
    window; the window's last frame is the one scored, as for RECORD."""
    net.eval()
    metrics = Evaluator(net.n_class)
    kbar = pkbar.Kbar(target=len(dataloader), width=20, always_stateful=False)
    running_loss = 0.0
    for iter, data in enumerate(dataloader):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device).float()

        inputs = data[features[0]]
        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        if eval_type == "val":
            loss = net.get_loss(outputs, data, train_config)
            running_loss += loss.item() * inputs.shape[0]
            kbar.update(iter, values=[("loss", loss.item())])
        else:
            kbar.update(iter)
        metrics.add_batch(torch.argmax(data['gt_mask'][:, :, -1], axis=1).cpu(),
                          torch.argmax(outputs[:, :, -1], axis=1).cpu())

    metrics_dict = get_metrics(metrics)
    result = {'mAP': metrics_dict['prec'], 'mAR': metrics_dict['recall'],
              'mIoU': metrics_dict['miou'], 'mDice': metrics_dict['dice'],
              'iou_by_class': metrics_dict['miou_by_class'], 'dice_by_class': metrics_dict['dice_by_class']}
    logger.info(f"{eval_type}: mIoU {result['mIoU'] * 100:.2f} %, mDice {result['mDice'] * 100:.2f} %, "
                f"IoU by class {[round(v * 100, 1) for v in result['iou_by_class']]}, "
                f"Dice by class {[round(v * 100, 1) for v in result['dice_by_class']]}")
    if eval_type == 'val':
        result.update({'loss': running_loss / len(dataloader.dataset)})
    return result


def MVRECORD_CARRADA_evaluation(net, dataloader, train_config, features, output_dir, eval_type, device):
    net.eval()
    kbar = pkbar.Kbar(target=len(dataloader), width=20, always_stateful=False)
    running_loss = 0.0
    batch_size = train_config['dataloader'][eval_type]['batch_size']
    rd_metrics = Evaluator(net.n_class) 
    ra_metrics = Evaluator(net.n_class) 
    for iter, data in enumerate(dataloader):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device).float()
        
        inputs = {feature: data[feature] for feature in features}
        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        if eval_type == "val":
            loss = net.get_loss(outputs, data, train_config)
            # inputs is a dict here; any view carries the sample axis.
            running_loss += loss.item() * data[features[0]].shape[0]
            logger.info(f"Val sample {iter} loss: {loss}")

        rd_metrics.add_batch(torch.argmax(data['rd_mask'], axis=1).cpu(), torch.argmax(outputs['rd'], axis=1).cpu())
        ra_metrics.add_batch(torch.argmax(data['ra_mask'], axis=1).cpu(), torch.argmax(outputs['ra'], axis=1).cpu())
        if eval_type == "val":
            kbar.update(iter, values=[("loss", loss.item())])
        else:
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
    # The reference keeps the two views apart: carrada_trainer.py logs 'rd_miou' and
    # 'ra_miou' separately (lines 191/195 val, 322/326 test) and builds global_ aggregates
    # for acc/prec/dice ONLY -- never for mIoU. The paper does the same, reporting 63.2 RD
    # against 44.5 RA. An average of those two matches nothing published, so both views are
    # carried here; 'mIoU' remains for train()'s history/CSV contract.
    # (Until 2026-08-26 this line read range_doppler twice, so 'mIoU' was the RD score alone.)
    rd_miou = metrics_dict['range_doppler']['miou']
    ra_miou = metrics_dict['range_angle']['miou']
    result = {'mAP': mAP, 'mAR': mAR, 'mIoU': (rd_miou + ra_miou)/2,
              'mIoU_rd': rd_miou, 'mIoU_ra': ra_miou}
    if eval_type == 'val':
        result.update({'loss': running_loss / len(dataloader.dataset)})
    return  result


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


def RADDet_evaluation(net, dataloader, train_config, features, output_dir, eval_type, device):
    net.eval()
    n_class = net.n_class
    kbar = pkbar.Kbar(target=len(dataloader), width=20, always_stateful=False)
    running_loss = 0.0
    batch_size = train_config['dataloader'][eval_type]['batch_size']

    # The reference splits this the same way: its MODEL block carries a single
    # mAP_iou3d_threshold (0.3) used during training/validation, while its EVALUATE block
    # carries a LIST [0.1, 0.3, 0.5, 0.7] for the final test -- which is what the paper
    # reports (Table I/II: AP@0.1 0.764, @0.3 0.563, @0.5 0.251, @0.7 0.059). So the
    # scalar stays the primary and drives 'mAP' (the checkpoint-selection contract with
    # train()); the optional list adds the other thresholds as extra keys, which
    # train() carries into val_eval.csv and test_eval.csv unchanged.
    primary_thr = train_config["mAP_iou3d_threshold"]
    thresholds = [primary_thr]
    for thr in (train_config.get("mAP_iou3d_threshold_list") or []):
        if thr not in thresholds:
            thresholds.append(thr)
    class_names = getattr(net, 'class_names', None) or \
        ["person", "bicycle", "car", "motorcycle", "bus", "truck"][:n_class]
    ap_by_thr = {thr: [[] for _ in range(n_class)] for thr in thresholds}
    total_by_thr = {thr: 0.0 for thr in thresholds}
    mean_ap_test = 0.0

    for iter, data in enumerate(dataloader):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device).float()

        inputs = data["RAD"]
        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        if eval_type == "val":
            loss = net.get_loss(outputs, data, train_config)
            running_loss += loss.item() * inputs.shape[0]   # actual batch, not configured
            logger.info(f"Val sample {iter} loss: {loss}")

        pred = outputs['pred'].detach().cpu().numpy()
        raw_boxes = data['boxes'].detach().cpu().numpy()
        for batch_id in range(raw_boxes.shape[0]):
            raw_boxes_frame = raw_boxes[batch_id]
            pred_frame = pred[batch_id]
            predicitons = yoloheadToPredictions(pred_frame, \
                                conf_threshold=train_config["confidence_threshold"])
            nms_pred = nms(predicitons, train_config["nms_iou3d_threshold"], \
                            net.input_size, sigma=0.3, method="nms")
            # The IoU threshold only affects TP/FP MATCHING, never the predictions, so all
            # thresholds share this one forward pass + NMS. Same structure as the
            # reference's evaluate.py:118-146, which keeps one accumulator per threshold
            # and calls mAP() once per threshold with the same nms_pred.
            for thr in thresholds:
                mean_ap, ap_by_thr[thr] = mAP(nms_pred, raw_boxes_frame, \
                                        net.input_size, ap_by_thr[thr], \
                                        tp_iou_threshold=thr)
                total_by_thr[thr] += mean_ap
            mean_ap_test = total_by_thr[primary_thr]

        if eval_type == "val":
            kbar.update(iter, values=[("loss", loss.item())])
        else:
            kbar.update(iter)

    # mean_ap_test accumulates once per FRAME (the batch_id loop above), so normalise by
    # the frame count. `batch_size * len(dataloader)` overestimates it whenever the split
    # is not a multiple of batch_size, which would drag mAP down.
    n_frames = len(dataloader.dataset)
    result = {'mAP': total_by_thr[primary_thr] / n_frames, 'mAR': 0.0, 'mIoU': 0.0}
    for thr in thresholds:
        # Per-class AP was computed here before and then thrown away; the paper reports it
        # (Table I) and every other closed row in the campaign is argued per class.
        per_class = [float(np.mean(c)) if len(c) else 0.0 for c in ap_by_thr[thr]]
        if thr != primary_thr:
            result[f'mAP@{thr}'] = total_by_thr[thr] / n_frames
        for name, ap in zip(class_names, per_class):
            result[f'AP@{thr}_{name}'] = ap
    if eval_type == 'val':
        result.update({'loss': running_loss / len(dataloader.dataset)})
    return  result


def TransRAD_evaluation(net, dataloader, train_config, features, output_dir, eval_type, device):
    """3D RAD-box mAP, the paper's Table I protocol: decode, class NMS then Location-Aware
    NMS, RADDet's per-class AP at every IoU threshold from the one forward pass. Keys
    follow RADDet_evaluation so the baselines row reads the same way."""
    net.eval()
    n_class = net.n_class
    kbar = pkbar.Kbar(target=len(dataloader), width=20, always_stateful=False)
    running_loss = 0.0
    eval_size = train_config["eval_input_size"]
    primary_thr = train_config["mAP_iou3d_threshold"]
    thresholds = [primary_thr] + [t for t in (train_config.get("mAP_iou3d_threshold_list") or []) if t != primary_thr]
    class_names = getattr(net, 'class_names', None) or ["person", "bicycle", "car", "motorcycle", "bus", "truck"][:n_class]
    ap_by_thr = {thr: [[] for _ in range(n_class)] for thr in thresholds}
    total_by_thr = {thr: 0.0 for thr in thresholds}

    for iter, data in enumerate(dataloader):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device).float()
        inputs = data["RAD"]
        with torch.set_grad_enabled(False):
            outputs = net(inputs)
        if eval_type == "val":
            loss = net.get_loss(outputs, data, train_config)
            running_loss += loss.item() * inputs.shape[0]
            logger.info(f"Val sample {iter} loss: {loss}")

        rows = transrad_decode(outputs, n_class, eval_size)
        raw_boxes = data['boxes'].detach().cpu().numpy()
        for batch_id in range(raw_boxes.shape[0]):
            if not raw_boxes[batch_id][:, :6].any():
                # No ground truth in this frame (the adapter yields (0, 7) boxes and the
                # collate pads with zeros): mAP() would average an empty list to NaN. It
                # counts as a frame scoring 0, RADDet's convention for absent classes.
                continue
            nms_pred = transrad_nms(rows[batch_id], eval_size, train_config["confidence_threshold"],
                                    train_config["nms_iou3d_threshold"], train_config["la_nms_iou3d_threshold"])
            for thr in thresholds:
                mean_ap, ap_by_thr[thr] = mAP(nms_pred, raw_boxes[batch_id], eval_size, ap_by_thr[thr], tp_iou_threshold=thr)
                total_by_thr[thr] += mean_ap
        if eval_type == "val":
            kbar.update(iter, values=[("loss", loss.item())])
        else:
            kbar.update(iter)

    # mean AP accumulates once per FRAME, so normalise by the frame count (RADDet_evaluation
    # does the same; batch_size * len(dataloader) would over-count a ragged last batch).
    n_frames = len(dataloader.dataset)
    result = {'mAP': total_by_thr[primary_thr] / n_frames, 'mAR': 0.0, 'mIoU': 0.0}
    for thr in thresholds:
        per_class = [float(np.mean(c)) if len(c) else 0.0 for c in ap_by_thr[thr]]
        if thr != primary_thr:
            result[f'mAP@{thr}'] = total_by_thr[thr] / n_frames
        for name, ap in zip(class_names, per_class):
            result[f'AP@{thr}_{name}'] = ap
    if eval_type == 'val':
        result.update({'loss': running_loss / n_frames})
    return result


class RunningAverage():
    def __init__(self):
        self.total = 0
        self.count = 0
    
    def __call__(self, value):
        self.total += value
        self.count += 1
        return self.total / self.count
    
    def reset(self):
        self.total = 0
        self.count = 0
    
    def result(self):
        return self.total / self.count if self.count != 0 else 0


def accumulate_tp_fp(pred_boxes, pred_labels, pred_scores, gt_boxes, gt_classes, tp_dict, iou_thresholds):
    """
    Count the number of false positive and true positive for a single image

    :param pred_boxes: the predicted boxes for the image i
    :param pred_labels: the predicted labels for the image i
    :param pred_scores: the predicted scores for the image i
    :param gt_boxes: the ground truth boxes for the image i
    :param gt_classes: the ground truth labels for the image i
    :param tp_dict: dictionary containing accumulated statistics on the test set
    :param iou_thresholds: threshold to use for tp detections

    :return: the updated ap_dict
    """
    # Remove background detections
    keep_idx = pred_boxes[..., :].any(axis=-1) > 0
    pred_boxes = pred_boxes[keep_idx]
    pred_labels = pred_labels[keep_idx]
    pred_scores = pred_scores[keep_idx]
    detected_gts = [[] for _ in range(len(iou_thresholds))]
    # Count number of GT for each class
    for i in range(len(gt_classes)):
        gt_temp = gt_classes[i]
        if gt_temp == -2:
            continue
        tp_dict[gt_temp]["total_gt"] += 1
    for i in range(len(pred_boxes)):
        pred_box = pred_boxes[i]
        pred_label = int(pred_labels[i])
        pred_score = pred_scores[i]
        iou = iou2d_corner(pred_box, gt_boxes)
        max_idx_iou = np.argmax(iou)
        gt_box = gt_boxes[max_idx_iou]
        gt_label = gt_classes[max_idx_iou]
        for iou_idx in range(len(iou_thresholds)):
            tp_dict[pred_label]["scores"][iou_idx].append(pred_score)
            tp_dict[pred_label]["tp"][iou_idx].append(0)
            tp_dict[pred_label]["fp"][iou_idx].append(0)
            if pred_label == gt_label and iou[max_idx_iou] >= iou_thresholds[iou_idx] and \
                    list(gt_box) not in detected_gts[iou_idx]:
                tp_dict[pred_label]["tp"][iou_idx][-1] = 1
                detected_gts[iou_idx].append(list(gt_box))
            else:
                tp_dict[pred_label]["fp"][iou_idx][-1] = 1

    return tp_dict


def iou2d_corner(box_xywh_1, box_xywh_2):
    """
    Numpy version of 3D bounding box IOU calculation
    :param box_xywh_1: [x1, y1, x2, y2]
    :param box_xywh_2: [x1, y1, x2, y2]
    :return:
    """
    assert box_xywh_1.shape[-1] == 4
    assert box_xywh_2.shape[-1] == 4
    box1_w = box_xywh_1[..., 2] - box_xywh_1[..., 0]
    box1_h = box_xywh_1[..., 3] - box_xywh_1[..., 1]
    #
    box2_w = box_xywh_2[..., 2] - box_xywh_2[..., 0]
    box2_h = box_xywh_2[..., 3] - box_xywh_2[..., 1]
    ### areas of both boxes
    box1_area = box1_h * box1_w
    box2_area = box2_h * box2_w
    ### find the intersection box
    box1_min = [box_xywh_1[..., 0], box_xywh_1[..., 1]]
    box1_max = [box_xywh_1[..., 2], box_xywh_1[..., 3]]
    box2_min = [box_xywh_2[..., 0], box_xywh_2[..., 1]]
    box2_max = [box_xywh_2[..., 2], box_xywh_2[..., 3]]

    x_top = np.maximum(box1_min[0], box2_min[0])
    y_top = np.maximum(box1_min[1], box2_min[1])
    x_bottom = np.minimum(box1_max[0], box2_max[0])
    y_bottom = np.minimum(box1_max[1], box2_max[1])

    intersection_area = np.maximum(x_bottom - x_top, 0) * np.maximum(y_bottom - y_top, 0)
    ### get union area
    union_area = box1_area + box2_area - intersection_area
    ### get iou
    iou = np.nan_to_num(intersection_area / (union_area + 1e-10))
    return iou


def compute_ap(recall, precision):
    """
    Compute the average precision, given the recall and precision curves.
    Code originally from https://github.com/rbgirshick/py-faster-rcnn.
    :param recall: The recall curve (list).
    :param precision: The precision curve (list).
    :return: The average precision as computed in py-faster-rcnn.
    """
    # correct AP calculation
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))

    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap


def compute_ap_class(tp, fp, scores, total_gt):
    """
    Compute the AP given tp, fp, scores and number of ground truth of a class
    See: https://github.com/keshik6/KITTI-2d-object-detection for some part of this code.
    :param tp: true positives list
    :param fp: false positives list
    :param scores: scores list
    :param total_gt: number of total GT in the dataset
    :return: average precision of a class
    """
    # Array manipulation
    tp = np.array(tp)
    fp = np.array(fp)
    scores = np.array(scores)
    # Sort detection by scores
    indices = np.argsort(-scores)
    fp = fp[indices]
    tp = tp[indices]
    # compute false positives and true positives
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    # compute precision/recall
    recall = tp / total_gt
    precision = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    # compute ap
    average_precision = compute_ap(recall, precision)
    if len(precision) == 0:
        mean_pre = 0.0
    else:
        mean_pre = np.mean(precision)
    if len(recall) == 0:
        mean_rec = 0.0
    else:
        mean_rec = np.mean(recall)
    return average_precision, mean_pre, mean_rec


def compute_f1(precision, recall):
    """
    Compute F1 score
    :param precision:
    :param recall:
    :return: F1-score
    """
    return 2 * (precision * recall) / (precision + recall + 1e-16)


def AP(tp_dict, n_classes, iou_th=[0.5]):
    """
    Create a dictionary containing ap per class

    :param tp_dict:
    :param n_classes:
    :return: A dictionary with the AP for each class
    """
    ap_dict = dict()
    valid_class_ids = []
    # Ignore 0 class which is BG
    for class_id in range(n_classes):
        tp, fp, scores, total_gt = tp_dict[class_id]["tp"], tp_dict[class_id]["fp"], tp_dict[class_id]["scores"], \
            tp_dict[class_id]["total_gt"]
        logger.debug(f"class {class_id}: {total_gt}")
        # Added begins
        if total_gt == 0:
            continue
        else:
            valid_class_ids.append(class_id)
        # Added ends
        ap_dict[class_id] = [[] for _ in range(len(iou_th))]
        ap_dict[class_id] = {
            "AP": [0.0 for _ in range(len(iou_th))],
            "precision": [0.0 for _ in range(len(iou_th))],
            "recall": [0.0 for _ in range(len(iou_th))],
            "F1": [0.0 for _ in range(len(iou_th))]
        }
        for iou_idx in range(len(iou_th)):
            ap_dict[class_id]["AP"][iou_idx], ap_dict[class_id]["precision"][iou_idx], ap_dict[class_id]["recall"][iou_idx] \
                = compute_ap_class(tp[iou_idx], fp[iou_idx], scores[iou_idx], total_gt)
            ap_dict[class_id]["F1"][iou_idx] = compute_f1(ap_dict[class_id]["precision"][iou_idx],
                                                          ap_dict[class_id]["recall"][iou_idx])
    logger.debug(valid_class_ids)
    ap_dict["mean"] = {
        "AP": [np.mean([ap_dict[class_id]["AP"][iou_th] for class_id in valid_class_ids])
               for iou_th in range(len(ap_dict[valid_class_ids[0]]["AP"]))],
        "precision": [np.mean([ap_dict[class_id]["precision"][iou_th] for class_id in valid_class_ids])
                      for iou_th in range(len(ap_dict[valid_class_ids[0]]["precision"]))],
        "recall": [np.mean([ap_dict[class_id]["recall"][iou_th] for class_id in valid_class_ids])
                   for iou_th in range(len(ap_dict[valid_class_ids[0]]["recall"]))],
        "F1": [np.mean([ap_dict[class_id]["F1"][iou_th] for class_id in valid_class_ids])
               for iou_th in range(len(ap_dict[valid_class_ids[0]]["F1"]))]
    }
    return ap_dict


def DAROD_evaluation(net, dataloader, train_config, features, output_dir, eval_type, device, iou_thresholds=[0.5]):
    net.eval()
    n_class = net.n_class
    kbar = pkbar.Kbar(target=len(dataloader), width=20, always_stateful=False)
    running_loss = 0.0
    batch_size = train_config['dataloader'][eval_type]['batch_size']

    n_class = n_class - 1 # Ignore 0 class which is BG
    tp_dict = dict()
    for class_id in range(n_class):
        tp_dict[class_id] = {
            "tp": [[] for _ in range(len(iou_thresholds))],
            "fp": [[] for _ in range(len(iou_thresholds))],
            "scores": [[] for _ in range(len(iou_thresholds))],
            "total_gt": 0
        }
    
    for iter, data in enumerate(dataloader):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device).float()
        
        inputs = data['RD']
        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        if eval_type == "val":
            loss = net.get_loss(outputs, data, train_config)
            running_loss += loss.item() * inputs.shape[0]   # actual batch, not configured
            logger.info(f"Val sample {iter} loss: {loss}")

        if outputs["decoder_output"] is not None:
            pred_boxes, pred_labels, pred_scores = outputs["decoder_output"]
            pred_labels = pred_labels - 1
            gt_labels = data['label'] - 1
            for batch_id in range(pred_boxes.shape[0]):
                tp_dict = accumulate_tp_fp(pred_boxes.cpu().numpy()[batch_id], pred_labels.cpu().numpy()[batch_id],
                                                pred_scores.cpu().numpy()[batch_id], data['boxes'].cpu().numpy()[batch_id],
                                                gt_labels.cpu().numpy()[batch_id], tp_dict,
                                                iou_thresholds=iou_thresholds)
        if eval_type == "val":
            kbar.update(iter, values=[("loss", loss.item())])
        else:
            kbar.update(iter)

    ap_dict = AP(tp_dict, n_class, iou_thresholds)
    mAP = np.mean(ap_dict["mean"]["AP"])
    mAR = np.mean(ap_dict["mean"]["recall"])
    result = {'mAP': mAP, 'mAR': mAR, 'mIoU': 0.0}
    if eval_type == 'val':
        result.update({'loss': running_loss / len(dataloader.dataset)})
    return  result


def evaluate_rampcnn_seq(res_objs, gt_objs, n_frame, n_class, classes, rng_grid, agl_grid, olsThrs, recThrs):
    olss_all = {(imgId, catId): compute_ols_dts_gts(gt_objs, res_objs, imgId, catId) \
                for imgId in range(n_frame)
                for catId in range(3)}

    evalImgs = [evaluate_img(gt_objs, res_objs, imgId, catId, olss_all, olsThrs, recThrs, classes)
                for imgId in range(n_frame)
                for catId in range(3)]
    return evalImgs


def RAMP_CNN_evaluation(net, dataloader, train_config, features, output_dir, eval_type, device):

    net.eval()
    n_class = net.n_class
    kbar = pkbar.Kbar(target=len(dataloader), width=20, always_stateful=False)
    running_loss = 0.0
    batch_size = train_config['dataloader'][eval_type]['batch_size']

    # 1.Generate network output (confmaps) and post-process them to the form of detection predictions
    classes = ["pedestrian", "cyclist", "car"] 
    with open('/home/kangle/dataset/UWCR/Automotive/radar_config.json', 'r') as file:
        radar_cfg = json.load(file)
    rng_grid = confmap2ra(radar_cfg, name='range')
    agl_grid = confmap2ra(radar_cfg, name='angle')
    
    confmap_shape = (n_class, radar_cfg['ramap_rsize'], radar_cfg['ramap_asize'])
    init_genConfmap = ConfmapStack(confmap_shape)
    iter_ = init_genConfmap
    for i in range(train_config['win_size'] - 1):
        while iter_.next is not None:
            iter_ = iter_.next
        iter_.next = ConfmapStack(confmap_shape)
    
    total_time = 0
    total_count = 0
    load_tic = time.time()
    n_frame = len(dataloader)
    # sequences = []
    res_objs = {(i, j): [] for i in range(n_frame) for j in range(n_class)}
    gt_objs = {(i, j): [] for i in range(n_frame) for j in range(n_class)}
    id_gt = 1
    id_res = 1
    for iter, data in enumerate(dataloader):
        # get gt info
        for i in range(train_config['train_stride']):
            if data['gt_label'][0][i]:
                for obj in data['gt_label'][0][i]:
                    ridx, aidx, category = obj
                    rng = rng_grid[ridx]
                    agl = agl_grid[aidx]
                    classid = get_class_id(category, classes)
                    if classid == -1000: 
                        continue
                    gt_obj = {'id': id_gt, 'frameid': iter, 'range': rng, 'angle': agl, 'ridx': ridx, 'aidx': aidx,
                                    'class_id': classid, 'score': 1.0}
                    gt_objs[iter, classid].append(gt_obj)
                    id_gt += 1
            # TODO: Handle the case of empty annotation

        load_time = time.time() - load_tic
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device).float()
        
        inputs = {feature: data[feature] for feature in ['RD', 'RA', 'AD']}
                
        # seq_name = data_dict['seq_name']
        # if seq_name not in sequences:
        #     sequences.append(seq_name)
        
        # # Currently only support batch_size set to 1 for val evaluation
        # start_frame_id = data_dict['start_frame'].item()
        # end_frame_id = data_dict['end_frame'].item()
        
        tic = time.time()
        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        if eval_type == "val":
            loss = net.get_loss(outputs, data, train_config)
            # inputs is a dict here; any view carries the sample axis.
            running_loss += loss.item() * data['RD'].shape[0]
            logger.info(f"Val sample {iter} loss: {loss}")

        confmap_pred = outputs['confmap_pred'][-1].cpu().detach().numpy()  
        confmap_pred = np.expand_dims(confmap_pred, axis=0)
        infer_time = time.time() - tic
        total_time += infer_time
        
        iter_ = init_genConfmap
        for i in range(confmap_pred.shape[2]):
            if iter_.next is None and i != confmap_pred.shape[2] - 1:
                iter_.next = ConfmapStack(confmap_shape)
            iter_.append(confmap_pred[0, :, i, :, :])
            iter_ = iter_.next

        process_tic = time.time()
        # save_path = os.path.join(save_dir, seq_name + '_rampcnn_res.txt')
        for i in range(train_config['train_stride']): # test_stride
            total_count += 1
            res_final = post_process_single_frame(init_genConfmap.confmap, train_config, n_class, rng_grid, agl_grid)
            # cur_frame_id = start_frame_id + i
            # write_dets_results_single_frame(res_final, cur_frame_id, save_path, classes)
            for obj in res_final:
                class_id, ridx, aidx, conf = obj
                if class_id == -1:
                    continue
                # res_final is a float array, so these unpack as numpy floats and
                # rng_grid[ridx] raises IndexError. The line only runs once a detection
                # survives post-processing, so it stayed latent for as long as the model
                # detected nothing -- the run "succeeded" with mAP 0 and crashed the
                # moment the model started working (epoch 3 of a 20-epoch run).
                class_id, ridx, aidx = int(class_id), int(ridx), int(aidx)
                rng = rng_grid[ridx]
                agl = agl_grid[aidx]
                res_obj = {'id': id_res, 'frameid': iter, 'range': rng, 'angle': agl, 'ridx': ridx, 'aidx': aidx,
                            'class_id': class_id, 'score': conf}
                # class_id, not classid: `classid` is a leaked loop variable from the
                # ground-truth loop above (the class of the last GT object seen), so every
                # detection was filed under the wrong class bucket -- and under a stale one
                # on frames with no GT at all. accumulate() then scored each class against
                # another class's detections, which pins mAP/mAR at 0 however well the
                # model does. The dict value on the line above already had it right.
                res_objs[iter, class_id].append(res_obj)
                id_res += 1
            init_genConfmap = init_genConfmap.next

        if iter == len(dataloader) - 1:
            # offset = train_cfg['train_stride'] # test_stride
            # cur_frame_id = start_frame_id + offset
            while init_genConfmap is not None:
                total_count += 1
                res_final = post_process_single_frame(init_genConfmap.confmap, train_config, n_class, rng_grid, agl_grid)
                # write_dets_results_single_frame(res_final, cur_frame_id, save_path, classes)
                for obj in res_final:
                    class_id, ridx, aidx, conf = obj
                    if class_id == -1:
                        continue
                    class_id, ridx, aidx = int(class_id), int(ridx), int(aidx)  # see above
                    rng = rng_grid[ridx]
                    agl = agl_grid[aidx]
                    res_obj = {'id': id_res, 'frameid': iter, 'range': rng, 'angle': agl, 'ridx': ridx, 'aidx': aidx,
                                'class_id': class_id, 'score': conf}
                    res_objs[iter, class_id].append(res_obj)   # see the note above
                    id_res += 1
                init_genConfmap = init_genConfmap.next
                # offset += 1
                # cur_frame_id += 1
        if eval_type == "val":
            kbar.update(iter, values=[("loss", loss.item())])
        else:
            kbar.update(iter)

        if init_genConfmap is None:
            init_genConfmap = ConfmapStack(confmap_shape)

        # proc_time = time.time() - process_tic
        # logger.info("Testing %s: frame %4d to %4d | Load time: %.4f | Inference time: %.4f | Process time: %.4f" %
        #         (seq_name, start_frame_id, end_frame_id, load_time, infer_time, proc_time))

        load_tic = time.time()
    logger.info("Average Infer time: {:.4f} s".format(total_time / total_count))

    # 2.Evaluation the detection predictions with Ground-truth annotations
    olsThrs = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1), endpoint=True), decimals=2)
    recThrs = np.around(np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01) + 1), endpoint=True), decimals=2)
    evalImgs = evaluate_rampcnn_seq(res_objs, gt_objs, n_frame, n_class, classes, rng_grid, agl_grid, olsThrs, recThrs)
    eval = accumulate(evalImgs, n_frame, olsThrs, recThrs, n_class, classes)
    stats = summarize(eval, olsThrs, recThrs, n_class, gl=False)
    logger.info("AP_total: {:.4f} | AR_total: {:.4f}".format(stats[0] * 100, stats[1] * 100))
    result = {'mAP': stats[0], 'mAR': stats[1], 'mIoU': 0}
    if eval_type == 'val':
        result.update({'loss': running_loss / len(dataloader.dataset)})
    return result


RCA_CLASS_NAMES = ('ped', 'cyc', 'car')   # bi_var_gauss channel order: cls 1/2/3 -> 0/1/2


def _rca_class_ap(cls, tag):
    """Per-class AP from the accumulated confidence/TP lists, as the reference computes
    it (Utils/val_utils.py:131-145): rank by confidence, trapezoid over the P/R curve.

    Returns (aps, mAP): one AP per class (nan where the split has no ground truth for
    it) and the mean over the classes that do occur.
    """
    aps = []
    for idx in range(len(cls)):
        categ = cls[f"{idx}"]
        GT = categ['grd_no']
        if GT == 0:
            # No ground truth for this class in this split: recall is undefined, so
            # TP/GT gave 0/0 -> nan -> np.trapz -> nan, and the nan then propagated into
            # the 3-class mean, reporting mAP as nan for a model that was detecting
            # perfectly well. Seen in the Phase 1 smoke sweep, whose 3-file sequence split
            # of CARRADA contains no pedestrian or cyclist frames: classes 0 and 1 had
            # GT 0 while class 2 scored AP 0.418 at recall 0.93.
            # Exclude the class instead, which is also what COCO-style mAP does with
            # absent categories.
            categ['AP'] = float('nan')
            aps.append(float('nan'))
            logger.info(f"{tag} {RCA_CLASS_NAMES[idx]} | no ground truth in this split, excluded from mAP")
            continue
        index = np.argsort(-np.asarray(categ['Conf']))
        P, R, TP = [], [], 0
        for cnt, i in enumerate(index):
            TP += categ['TP'][i]
            P = np.append(P, TP / (cnt + 1))
            R = np.append(R, TP / GT)
        categ['P'] = P
        categ['R'] = R
        # No predictions at all for a class that does occur is AP 0, not a crash.
        categ['AP'] = float(np.trapz(P, R)) if len(P) else 0.0
        n_pred = len(categ['TP'])
        n_tp = float(np.sum(categ['TP']))
        logger.info(f"{tag} {RCA_CLASS_NAMES[idx]} | AP {categ['AP']:.4f}  "
                    f"P {n_tp / n_pred if n_pred else float('nan'):.3f}  R {n_tp / GT:.3f}  "
                    f"GT {GT}  preds {n_pred}")
        aps.append(categ['AP'])
    # Average over the classes that actually occur, not a hardcoded 3 -- see the GT == 0
    # branch above. A single absent class used to make the whole metric nan.
    present = [aps[i] for i in range(len(cls)) if cls[f"{i}"]['grd_no'] > 0]
    mAP = float(np.mean(present)) if present else 0.0
    if len(present) < len(cls):
        logger.warning(f"{tag}: mAP averaged over {len(present)} of {len(cls)} classes; the rest "
                       f"have no ground truth in this split and cannot be scored.")
    return aps, mAP


def RadarCrossAttention_evaluation(net, dataloader, train_config, features, output_dir, eval_type, device,
                                   thresholds=(2, 1)):
    """AP at distance thresholds, per class, as the reference's evaluation() scores it.

    Scored at BOTH 2 m and 1 m in one pass (the paper's Table I reports both; the
    reference scores one threshold per invocation). 'mAP' is the 2 m headline; the 1 m
    mean and the six per-class APs come back as extras. For 'val' it also returns
    'val_mse', the reference's checkpoint-selection and ReduceLROnPlateau metric
    (Utils/val_utils.py:44-46): the sum-MSE between sigmoid(pred_mask) and the Gaussian
    target per batch, averaged over batches -- batch-size dependent by construction, so
    the val batch size should match the reference's 16 for the numbers to be comparable.
    """
    net.eval()
    kbar = pkbar.Kbar(target=len(dataloader), width=20, always_stateful=False)
    running_loss = 0.0
    running_mse = 0.0
    n_batches = 0
    batch_size = train_config['dataloader'][eval_type]['batch_size']

    temp_dict = {'grd_no':0,'Conf':[],'O_P':[],'O_T':[],'TP':[],'P':[],'R':[],'AP':[],'asc':{'gt_r':[],'gt_a':[],'pd_r':[],'pd_a':[]}}
    cls_by_thresh = {t: {str(i): deepcopy(temp_dict) for i in range(3)} for t in thresholds}

    prev_size = 0

    for iter, data in enumerate(dataloader):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device).float()
        
        inputs = {feature: data[feature] for feature in ['RD', 'RA', 'AD']}
        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        if eval_type == "val":
            loss = net.get_loss(outputs, data, train_config)
            # inputs is a dict here; any view carries the sample axis.
            running_loss += loss.item() * data['RD'].shape[0]
            logger.info(f"Val sample {iter} loss: {loss}")

        if data['RA'].shape[0] != prev_size:
            mask, peak_cls = create_default(data["gt_mask"].size(), kernel_window=(3,5))
            mask = mask.to(device=device)
            peak_cls = peak_cls.to(device=device)
        
        prev_size = data['RA'].shape[0]

        pred_map = torch.sigmoid(outputs["pred_mask"])
        # Centre-offset and heading are optional heads: the network returns the int 0 for
        # a disabled one (as the reference does) and the dataset then carries no
        # gt_orent_map. metrics_center/validation test torch.is_tensor for these, where
        # the reference's `if pred_c != 0` raises on any real tensor.
        pred_c = (8 * (torch.sigmoid(outputs["pred_center"]) - 0.5)
                  if torch.is_tensor(outputs["pred_center"]) else 0)
        tr_o = data.get("gt_orent_map", 0)
        pred_o = outputs["pred_orent"]

        running_mse += torch.nn.functional.mse_loss(pred_map, data["gt_mask"], reduction='sum').item()
        n_batches += 1

        # Peak detection, centre-offset and NMS do not depend on the threshold, so they
        # run once; validation() rebinds rather than mutates its index tensors, so the
        # same associated peaks can be scored at each threshold independently.
        grd_idx, pred_idx, pred_intent = _detect_and_associate(
            data["gt_mask"], pred_map, pred_c, mask, peak_cls, device)
        for t in thresholds:
            cls_by_thresh[t] = validation(grd_idx, pred_idx, pred_intent, tr_o, pred_o,
                                          cls_by_thresh[t], dist_thresh=t, device=device)
        if eval_type == "val":
            kbar.update(iter, values=[("loss", loss.item())])
        else:
            kbar.update(iter)
    logger.info("Val data loop finished.")

    aps, maps = {}, {}
    for t in thresholds:
        aps[t], maps[t] = _rca_class_ap(cls_by_thresh[t], f"AP@{t}m")
    
    # 'mAP' is the first threshold (2 m, the paper's headline); everything else rides
    # along as extras, which train() carries into history and val_eval.csv.
    headline = thresholds[0]
    result = {'mAP': maps[headline], 'mAR': 0, 'mIoU': 0}
    for t in thresholds:
        if t != headline:
            result[f'mAP_{t}m'] = maps[t]
        for name, ap in zip(RCA_CLASS_NAMES, aps[t]):
            result[f'AP{t}m_{name}'] = ap
    if eval_type == 'val':
        result.update({'loss': running_loss / len(dataloader.dataset),
                       'val_mse': running_mse / max(n_batches, 1)})
    return result


def _detect_and_associate(grd_map, pred_map, pred_c, mask, peaks_cls, device):
    """The threshold-independent half of the reference's metrics_center: peaks of the
    target (0.8) and of the prediction (0.1), centre-offset shift, 2 m NMS."""
    grd_intent, grd_idx = peaks_detect(grd_map, mask, peaks_cls, heat_thresh=0.8)
    grd_idx = distribute(grd_idx, device)
    pred_intent, pred_idx = peaks_detect(pred_map, mask, peaks_cls, heat_thresh=0.1)
    pred_idx = distribute(pred_idx, device)
    # The reference tests `if pred_c != 0`, which raises on a real tensor; the intent is
    # "is the centre-offset head on" (the network returns the int 0 when it is not).
    if torch.is_tensor(pred_c):
        pred_idx = update_peaks(pred_c, pred_idx)
    pred_idx, pred_intent = association(pred_intent, pred_idx, device)
    return grd_idx, pred_idx, pred_intent


def metrics_center(grd_map, pred_map, pred_c, mask,
                  peaks_cls, tr_o, pred_o, cls,
                  thresh, device):
    """The reference's metrics_center, unchanged in behaviour; the evaluator itself calls
    the two halves separately so one association serves both thresholds."""
    grd_idx, pred_idx, pred_intent = _detect_and_associate(grd_map, pred_map, pred_c, mask, peaks_cls, device)
    return validation(grd_idx, pred_idx, pred_intent, tr_o, pred_o, cls, dist_thresh=thresh, device=device)


def validation(grd_idx, pred_idx, pred_int, tr_o, pr_o, cls, dist_thresh=2, device='cpu'):
    for grd_cord in grd_idx:
        g_fr, g_cls = grd_cord[0], int(grd_cord[1])
        cls[f"{g_cls}"]['grd_no'] +=1

    while len(pred_int) != 0:
        cord = pred_idx[0]
        p_r, p_c= cord[2], cord[3]
        p_fr, p_cls = cord[0], int(cord[1])
        cls[f"{p_cls}"]['Conf'] = np.append(cls[f"{p_cls}"]['Conf'], pred_int[0].to('cpu').numpy())
        x1, y1 = pol2cord(p_r, p_c)
        dist = torch.Tensor().to(device =device)
        pred_int = pred_int[1:]
        pred_idx = pred_idx[1:]
        
        for grd_cord in grd_idx:
            g_fr, g_cls = grd_cord[0], int(grd_cord[1])
            if g_fr==p_fr and g_cls==p_cls:
                g_r, g_c = grd_cord[2], grd_cord[3]
                x2, y2 = pol2cord(g_r, g_c)
                dist = torch.cat([dist, torch.Tensor([distance(x1, y1, x2, y2)]).to(device=device)])
            else :
                dist = torch.cat([dist,torch.Tensor([100]).to(device=device)]) # dummy distance

        if len(dist)!=0:
            if torch.min(dist) < dist_thresh:
                min_idx = torch.argmin(dist)
                t_r = int(grd_idx[min_idx][2])
                grd_idx = torch.cat([grd_idx[0:min_idx], grd_idx[1+min_idx:]])
                cls[f"{p_cls}"]['TP'] = np.append(cls[f"{p_cls}"]['TP'], 1)
                
                p_fr = int(p_fr)
                # Heading is scored only when the head is on AND the split carries the
                # heading target; both are the int 0 otherwise (see metrics_center).
                if torch.is_tensor(pr_o) and torch.is_tensor(tr_o):
                        
                    tro_id = torch.nonzero(tr_o[p_fr, 1, ::], as_tuple=True)
                    if len(tro_id[0])>0 :
                        tro_min = torch.argmin(torch.abs(tro_id[0]-(t_r//4)))
                    else:
                        tro_id = torch.nonzero(tr_o[p_fr, 0, ::], as_tuple=True)
                        tro_min = torch.argmin(torch.abs(tro_id[0] - t_r//4))
    
                    cls[f"{p_cls}"]['O_T'].append(orent(tr_o[p_fr,::], tro_id[0][tro_min], tro_id[1][tro_min]))
                    cls[f"{p_cls}"]['O_P'].append(orent(pr_o[p_fr,::], int(p_r//4), int(p_c//4), pred=True))
            else:
                cls[f"{p_cls}"]['TP'] = np.append(cls[f"{p_cls}"]['TP'],0)
        else:
            cls[f"{p_cls}"]['TP'] = np.append(cls[f"{p_cls}"]['TP'],0)
    return cls



def RadarPillar_evaluation(net, dataloader, train_config, features, output_dir, eval_type, device):
    """Official VoD devkit evaluation (KITTI-style 3D AP, R11, rotated IoU;
    tudelft-iv/view-of-delft-dataset, copied under train/vod_evaluation).

    Radar-frame detections are converted back to KITTI camera frame -- the exact
    inverse of ViewOfDelft._gt_boxes_radar, mirroring the reference
    generate_prediction_dicts (fthbng77/RadarPillar vod_dataset.py:300-375):
    bottom-center through R0_ext @ V2C_ext, dims written h w l, ry = -yaw - pi/2,
    alpha = -arctan2(-y, x) + ry. The projected image bbox is required: the
    devkit ignores detections whose bbox height is under 40 px
    (kitti_official_evaluate.clean_data). Every val frame gets a file, even if
    empty -- the devkit derives the frame list from the prediction directory.
    Returned mAP/AP values are in PERCENT (Table I convention, e.g. 46.7),
    unlike the other models' 0-1 fractions.
    """
    from detection_vis_backend.networks.radarpillar import postprocess, box_corners_3d
    from detection_vis_backend.train.vod_evaluation import Evaluation

    net.eval()
    running_loss = 0.0
    kbar = pkbar.Kbar(target=len(dataloader), width=20, always_stateful=False)
    batch_size = train_config['dataloader'][eval_type]['batch_size']

    # Unwrap Subset / random_split / ConcatDataset nesting to the ViewOfDelft
    # instance (frame_ids index into it; calib + label paths live on it).
    dataset = dataloader.dataset
    while not hasattr(dataset, 'label_filenames'):
        if hasattr(dataset, 'dataset'):
            dataset = dataset.dataset
        elif hasattr(dataset, 'datasets'):
            dataset = dataset.datasets[0]
        else:
            raise ValueError("RadarPillar_evaluation: no ViewOfDelft dataset inside the loader")

    img = cv2.imread(dataset.image_filenames[0])
    img_h, img_w = img.shape[:2]

    pred_dir = os.path.join(output_dir, f'vod_eval_{eval_type}')
    os.makedirs(pred_dir, exist_ok=True)
    for stale in os.listdir(pred_dir):          # fresh prediction set every epoch
        os.remove(os.path.join(pred_dir, stale))

    class_names = ["Car", "Pedestrian", "Cyclist"]
    model_cfg = dict(net.model_cfg)

    for iter, data in enumerate(dataloader):
        for key in data:
            if isinstance(data[key], torch.Tensor):
                data[key] = data[key].to(device)   # dtype-preserving: coords/labels stay int
        with torch.set_grad_enabled(False):
            outputs = net(data)
        if eval_type == 'val':
            loss = net.get_loss(outputs, data, train_config)
            # Point-cloud batch: 'pillars'/'coords' are concatenated across samples, so
            # their leading axis is not the sample count -- frame_ids is one per sample.
            running_loss += loss.item() * len(data['frame_ids'])

        results = postprocess(outputs, net.net.anchors, model_cfg, train_config)
        for b, res in enumerate(results):
            frame_idx = data['frame_ids'][b]
            frame_name = os.path.basename(dataset.label_filenames[frame_idx])[:-4]
            intrinsic, R0_ext, V2C_ext = dataset._calib_radar(frame_idx)
            cam_from_radar = R0_ext @ V2C_ext
            lines = []
            for k in range(len(res['boxes'])):
                x, y, z, l, w, h, yaw = [float(v) for v in res['boxes'][k]]
                bottom = cam_from_radar @ np.array([x, y, z - h / 2.0, 1.0], dtype=np.float32)
                ry = -yaw - np.pi / 2
                alpha = -np.arctan2(-y, x) + ry
                corners = box_corners_3d(np.array([x, y, z, l, w, h, yaw]))
                cam = (cam_from_radar @ np.concatenate([corners, np.ones((8, 1))], axis=1).T).T[:, :3]
                if cam[:, 2].min() < 0.1:
                    continue                     # corner at/behind the image plane: unprojectable
                uvw = (intrinsic @ np.concatenate([cam, np.ones((8, 1))], axis=1).T).T
                us, vs = uvw[:, 0] / uvw[:, 2], uvw[:, 1] / uvw[:, 2]
                u0, v0 = max(float(us.min()), 0.0), max(float(vs.min()), 0.0)
                u1 = min(float(us.max()), img_w - 1.0)
                v1 = min(float(vs.max()), img_h - 1.0)
                lines.append('%s -1 -1 %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f %.4f'
                             % (class_names[int(res['labels'][k])], alpha, u0, v0, u1, v1,
                                h, w, l, bottom[0], bottom[1], bottom[2], ry, float(res['scores'][k])))
            with open(os.path.join(pred_dir, frame_name + '.txt'), 'w') as f:
                f.write('\n'.join(lines) + ('\n' if lines else ''))

        if eval_type == 'val':
            kbar.update(iter, values=[("loss", loss.item())])
        else:
            kbar.update(iter)

    gt_dir = os.path.join(dataset.root_path, 'radar', 'training', 'label_2')
    vod_result = Evaluation(test_annotation_file=gt_dir).evaluate(
        result_path=pred_dir, current_class=[0, 1, 2])
    ea, roi = vod_result['entire_area'], vod_result['roi']
    mAP = (ea['Car_3d_all'] + ea['Pedestrian_3d_all'] + ea['Cyclist_3d_all']) / 3
    mAP_roi = (roi['Car_3d_all'] + roi['Pedestrian_3d_all'] + roi['Cyclist_3d_all']) / 3
    logger.info('------- VoD official metric (3D AP, percent) ------------')
    logger.info(f"  Entire area: Car {ea['Car_3d_all']:.2f} | Ped {ea['Pedestrian_3d_all']:.2f} | "
                f"Cyc {ea['Cyclist_3d_all']:.2f} | mAP {mAP:.2f}")
    logger.info(f"  Driving corridor: Car {roi['Car_3d_all']:.2f} | Ped {roi['Pedestrian_3d_all']:.2f} | "
                f"Cyc {roi['Cyclist_3d_all']:.2f} | mAP {mAP_roi:.2f}")

    result = {'mAP': mAP, 'mAR': 0.0, 'mIoU': 0.0}
    if eval_type == 'val':
        result.update({'loss': running_loss / len(dataloader.dataset)})
    return result


RODNet_subtypes = ["RODNet_CDC", "RODNet_CDCv2", "RODNet_HG", "RODNet_HGv2", "RODNet_HGwI", "RODNet_HGwIv2", "T_RODNet"]
RECORD_subtypes = ["RECORD", "RECORDNoLstm", "RECORDNoLstmMulti"]
network_types = ["FFTRadNet", "MVRECORD", "DAROD", "RADDet", "TransRAD", "RAMP_CNN", "RadarCrossAttention", "RadarPillar", "PointPillar" ] + RODNet_subtypes + RECORD_subtypes
dataset_types = ["RADIal", "CRUW", "CARRADA", "RADDetDataset", "UWCR", "ViewOfDelft"]
eval_func_dict = {(i, j): None for i in network_types for j in dataset_types}
eval_func_dict["FFTRadNet", "RADIal"] = FFTRadNet_evaluation
for i in RODNet_subtypes:
    eval_func_dict[i, "CRUW"] = RODNet_evaluation
for i in RECORD_subtypes:
    eval_func_dict[i, "CRUW"] = RECORD_CRUW_evaluation
    eval_func_dict[i, "CARRADA"] = RECORD_CARRADA_evaluation
eval_func_dict["MVRECORD", "CARRADA"] = MVRECORD_CARRADA_evaluation
eval_func_dict["T_RODNet", "CARRADA"] = T_RODNet_CARRADA_evaluation
eval_func_dict["RADDet", "RADDetDataset"] = RADDet_evaluation
eval_func_dict["DAROD", "RADDetDataset"] = DAROD_evaluation
eval_func_dict["TransRAD", "RADDetDataset"] = TransRAD_evaluation
eval_func_dict["DAROD", "CARRADA"] = DAROD_evaluation
eval_func_dict["RAMP_CNN", "UWCR"] = RAMP_CNN_evaluation
eval_func_dict["RadarCrossAttention", "CARRADA"] = RadarCrossAttention_evaluation
eval_func_dict["RadarPillar", "ViewOfDelft"] = RadarPillar_evaluation
eval_func_dict["PointPillar", "ViewOfDelft"] = RadarPillar_evaluation