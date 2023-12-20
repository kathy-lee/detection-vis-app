import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import cv2
import polarTransform
import re
import json
import math


from shapely.geometry import Polygon
from typing import Optional
from loguru import logger


# def FFTRadNet_collate(batch):
#     images = []
#     FFTs = []
#     segmaps = []
#     labels = []
#     encoded_label = []

#     for radar_FFT, segmap, out_label, box_labels in batch:
#         FFTs.append(torch.tensor(radar_FFT))
#         segmaps.append(torch.tensor(segmap))
#         encoded_label.append(torch.tensor(out_label))
#         labels.append(torch.from_numpy(box_labels))    
#     return torch.stack(FFTs), torch.stack(encoded_label),torch.stack(segmaps),labels

def FFTRadNet_collate(batch):
    rd = [torch.tensor(item['RD']) for item in batch]
    rd = torch.stack(rd, 0)
    encoded_label = [torch.tensor(item['encoded_label']) for item in batch]
    encoded_label = torch.stack(encoded_label, 0)
    box_label = [torch.from_numpy(item['box_label']) for item in batch]
    if 'seg_label' in batch[0]:
        seg_label = [torch.tensor(item['seg_label']) for item in batch]
        seg_label = torch.stack(seg_label, 0)
        return {'RD': rd, 'encoded_label': encoded_label, 'box_label': box_label, 'seg_label': seg_label}
    else:
        return {'RD': rd, 'encoded_label': encoded_label, 'box_label': box_label}


def DAROD_collate(batch):
    # raw labels from CARRADA dataset: 1,2,3
    # raw labels from RADDet dataset: 1,2,3,4,5,6
    radar = [torch.tensor(item['RD'].copy()) for item in batch]
    gt_labels = [torch.tensor(item['label']) for item in batch]
    [ item.update({'boxes': np.array(item['boxes'])}) for item in batch ]
    gt_boxes = [torch.tensor(item['boxes'].reshape(item['boxes'].shape[0], -1)) for item in batch]
    
    radar = torch.stack(radar, 0)
    max_boxes = max([box.shape[0] for box in gt_boxes])
    # Initialize tensors for bboxes and labels filled with appropriate padding values
    padded_bboxes = torch.zeros(len(batch), max_boxes, 4)
    padded_labels = torch.full((len(batch), max_boxes), fill_value=-1, dtype=torch.long)  
    
    for idx, (box, label) in enumerate(zip(gt_boxes, gt_labels)):
        padded_bboxes[idx, :box.shape[0]] = box
        padded_labels[idx, :label.shape[0]] = label
    logger.debug(f"padding: {gt_labels} -> {padded_labels}")
    return {'RD': radar, 'label': padded_labels, 'boxes': padded_bboxes}


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")

    elem = batch[0]
    elem_type = type(elem)
    if elem is None:
        return None
    elif isinstance(elem, torch.Tensor):
        return torch.stack(batch, 0)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, bool):
        return all(batch)
    elif isinstance(elem, int): # int_classes
        return torch.tensor(batch)
    elif isinstance(elem, str): # string_classes
        return batch
    elif isinstance(elem, dict): # container_abcs.Mapping
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, (list, tuple)): # container_abcs.Sequence
        # transposed = zip(*batch)
        # return [cr_collate(samples) for samples in transposed]
        return batch
    else:
        raise TypeError(default_collate_err_msg_format.format(elem_type))


def decode(map,threshold):
        geometry = {
            "ranges": [512,896,1],
            "resolution": [0.201171875,0.2],
            "size": 3
        }
        statistics = {
            "input_mean":[-2.6244e-03, -2.1335e-01,  1.8789e-02, -1.4427e+00, -3.7618e-01,
                1.3594e+00, -2.2987e-01,  1.2244e-01,  1.7359e+00, -6.5345e-01,
                3.7976e-01,  5.5521e+00,  7.7462e-01, -1.5589e+00, -7.2473e-01,
                1.5182e+00, -3.7189e-01, -8.8332e-02, -1.6194e-01,  1.0984e+00,
                9.9929e-01, -1.0495e+00,  1.9972e+00,  9.2869e-01,  1.8991e+00,
               -2.3772e-01,  2.0000e+00,  7.7737e-01,  1.3239e+00,  1.1817e+00,
               -6.9696e-01,  4.4288e-01],
            "input_std":[20775.3809, 23085.5000, 23017.6387, 14548.6357, 32133.5547, 28838.8047,
                27195.8945, 33103.7148, 32181.5273, 35022.1797, 31259.1895, 36684.6133,
                33552.9258, 25958.7539, 29532.6230, 32646.8984, 20728.3320, 23160.8828,
                23069.0449, 14915.9053, 32149.6172, 28958.5840, 27210.8652, 33005.6602,
                31905.9336, 35124.9180, 31258.4316, 31086.0273, 33628.5352, 25950.2363,
                29445.2598, 32885.7422],
            "reg_mean":[0.4048094369863972,0.3997392847799934],
            "reg_std":[0.6968599580482511,0.6942950877813826]
        }
        regression_layer = 2
        INPUT_DIM = (geometry['ranges'][0],geometry['ranges'][1],geometry['ranges'][2])
        OUTPUT_DIM = (regression_layer + 1, INPUT_DIM[0] // 4 , INPUT_DIM[1] // 4 )
        
        range_bins,angle_bins = np.where(map[0,:,:]>=threshold)

        coordinates = []

        for range_bin,angle_bin in zip(range_bins,angle_bins):
            R = range_bin*4*geometry['resolution'][0] + map[1,range_bin,angle_bin] * statistics['reg_std'][0] + statistics['reg_mean'][0]
            A = (angle_bin-OUTPUT_DIM[2]/2)*4*geometry['resolution'][1] + map[2,range_bin,angle_bin] * statistics['reg_std'][1] + statistics['reg_mean'][1]
            C = map[0,range_bin,angle_bin]
        
            coordinates.append([R,A,C])
       
        return coordinates


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
            ious[i + 1:] = bbox_iou(sorted_box_predictions[i, :8], sorted_box_predictions[i + 1:, :8])
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
    final_class_predictions, final_box_predictions = perform_nms(valid_class_predictions, valid_box_predictions,nms_threshold)

    # concatenate point_cloud_id, confidence score and bounding box prediction | shape: [N_FINAL, 1+1+8]
    final_point_cloud_predictions = np.hstack((final_class_predictions[:, np.newaxis],
                                               final_box_predictions))

    return final_point_cloud_predictions


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
        logger.error('Error: Computing OLS between different classes!')
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


def get_class_name(class_id, classes):
    n_class = len(classes)
    if 0 <= class_id < n_class:
        class_name = classes[class_id]
    elif class_id == -1000:
        class_name = '__background'
    else:
        raise ValueError("Class ID is not defined")
    return class_name


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


def worldToImage(x,y,z):
    # Camera parameters
    camera_matrix = np.array([[1.84541929e+03, 0.00000000e+00, 8.55802458e+02],
                    [0.00000000e+00 , 1.78869210e+03 , 6.07342667e+02],[0.,0.,1]])
    dist_coeffs = np.array([2.51771602e-01,-1.32561698e+01,4.33607564e-03,-6.94637533e-03,5.95513933e+01])
    rvecs = np.array([1.61803058, 0.03365624,-0.04003127])
    tvecs = np.array([0.09138029,1.38369885,1.43674736])
    ImageWidth = 1920
    ImageHeight = 1080

    world_points = np.array([[x,y,z]],dtype = 'float32')
    rotation_matrix = cv2.Rodrigues(rvecs)[0]
    imgpts, _ = cv2.projectPoints(world_points, rotation_matrix, tvecs, camera_matrix, dist_coeffs)

    u = int(min(max(0,imgpts[0][0][0]),ImageWidth-1))
    v = int(min(max(0,imgpts[0][0][1]),ImageHeight-1))
    return u, v


def custom_one_hot(labels, num_classes):
    r"""
    torch.nn.functional.one_hot() could not handle negative class labels, but tf.one_hot() can: 
    When a negative value is encountered, it results in an all-zero vector in the one-hot encoded output. 
    This function implements the same as tf.one_hot(). It can also handle 2D input tensor.
    """
    # Handle the negative values by increasing them by num_classes
    labels = torch.where(labels < 0, labels + num_classes, labels)
    
    # Apply one hot encoding
    one_hot = F.one_hot(labels, num_classes=num_classes + 1)  # One additional class for the negative values
    
    # If we have the additional "negative" class, remove it (it will be the last class)
    if one_hot.size(-1) > num_classes:
        one_hot = one_hot[..., :-1]
    
    return one_hot
    

def get_metrics(metrics):
    """Structure the metric results
    @param metrics: contains statistics recorded during inference
    @return: metrics values
    """
    metrics_values = dict()
    acc, acc_by_class = metrics.get_pixel_acc_class()  # harmonic_mean=True)
    prec, prec_by_class = metrics.get_pixel_prec_class()
    recall, recall_by_class = metrics.get_pixel_recall_class()  # harmonic_mean=True)
    miou, miou_by_class = metrics.get_miou_class()  # harmonic_mean=True)
    dice, dice_by_class = metrics.get_dice_class()
    metrics_values['acc'] = acc
    metrics_values['acc_by_class'] = acc_by_class.tolist()
    metrics_values['prec'] = prec
    metrics_values['prec_by_class'] = prec_by_class.tolist()
    metrics_values['recall'] = recall
    metrics_values['recall_by_class'] = recall_by_class.tolist()
    metrics_values['miou'] = miou
    metrics_values['miou_by_class'] = miou_by_class.tolist()
    metrics_values['dice'] = dice
    metrics_values['dice_by_class'] = dice_by_class.tolist()
    return metrics_values


def iou3d(box_xyzwhd_1, box_xyzwhd_2, input_size):
    """ PyTorch version of 3D bounding box IOU calculation 
    Args:
        box_xyzwhd_1        ->      box1 [x, y, z, w, h, d]
        box_xyzwhd_2        ->      box2 [x, y, z, w, h, d]"""
    assert box_xyzwhd_1.shape[-1] == 6
    assert box_xyzwhd_2.shape[-1] == 6
    fft_shift_implement = torch.tensor([0, 0, input_size[2]/2], dtype=box_xyzwhd_1.dtype, device=box_xyzwhd_1.device)
    
    ### areas of both boxes
    box1_area = box_xyzwhd_1[..., 3] * box_xyzwhd_1[..., 4] * box_xyzwhd_1[..., 5]
    box2_area = box_xyzwhd_2[..., 3] * box_xyzwhd_2[..., 4] * box_xyzwhd_2[..., 5]
    
    box1_min = box_xyzwhd_1[..., :3] + fft_shift_implement - box_xyzwhd_1[..., 3:] * 0.5
    box1_max = box_xyzwhd_1[..., :3] + fft_shift_implement + box_xyzwhd_1[..., 3:] * 0.5
    box2_min = box_xyzwhd_2[..., :3] + fft_shift_implement - box_xyzwhd_2[..., 3:] * 0.5
    box2_max = box_xyzwhd_2[..., :3] + fft_shift_implement + box_xyzwhd_2[..., 3:] * 0.5

    left_top = torch.maximum(box1_min, box2_min)
    bottom_right = torch.minimum(box1_max, box2_max)

    ### get intersection area
    intersection = torch.maximum(bottom_right - left_top, torch.tensor(0.0, device=box_xyzwhd_1.device))
    intersection_area = intersection[..., 0] * intersection[..., 1] * intersection[..., 2]
    
    ### get union area
    union_area = box1_area + box2_area - intersection_area
    
    ### get iou
    iou = intersection_area / (union_area + 1e-10)
    return iou


def iou3d_np(box_xyzwhd_1, box_xyzwhd_2, input_size):
    """ Numpy version of 3D bounding box IOU calculation 
    Args:
        box_xyzwhd_1        ->      box1 [x, y, z, w, h, d]
        box_xyzwhd_2        ->      box2 [x, y, z, w, h, d]"""
    assert box_xyzwhd_1.shape[-1] == 6
    assert box_xyzwhd_2.shape[-1] == 6
    fft_shift_implement = np.array([0, 0, input_size[2]/2])
    ### areas of both boxes
    box1_area = box_xyzwhd_1[..., 3] * box_xyzwhd_1[..., 4] * box_xyzwhd_1[..., 5]
    box2_area = box_xyzwhd_2[..., 3] * box_xyzwhd_2[..., 4] * box_xyzwhd_2[..., 5]
    ### find the intersection box
    box1_min = box_xyzwhd_1[..., :3] + fft_shift_implement - box_xyzwhd_1[..., 3:] * 0.5
    box1_max = box_xyzwhd_1[..., :3] + fft_shift_implement + box_xyzwhd_1[..., 3:] * 0.5
    box2_min = box_xyzwhd_2[..., :3] + fft_shift_implement - box_xyzwhd_2[..., 3:] * 0.5
    box2_max = box_xyzwhd_2[..., :3] + fft_shift_implement + box_xyzwhd_2[..., 3:] * 0.5

    # box1_min = box_xyzwhd_1[..., :3] - box_xyzwhd_1[..., 3:] * 0.5
    # box1_max = box_xyzwhd_1[..., :3] + box_xyzwhd_1[..., 3:] * 0.5
    # box2_min = box_xyzwhd_2[..., :3] - box_xyzwhd_2[..., 3:] * 0.5
    # box2_max = box_xyzwhd_2[..., :3] + box_xyzwhd_2[..., 3:] * 0.5

    left_top = np.maximum(box1_min, box2_min)
    bottom_right = np.minimum(box1_max, box2_max)
    ### get intersection area
    intersection = np.maximum(bottom_right - left_top, 0.0)
    intersection_area = intersection[..., 0] * intersection[..., 1] * intersection[..., 2]
    ### get union area
    union_area = box1_area + box2_area - intersection_area
    ### get iou
    iou = np.nan_to_num(intersection_area / (union_area + 1e-10))
    return iou


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
                iou = iou3d_np(best_bbox[np.newaxis, :6], cls_bboxes[:, :6], \
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
    

def create_default(map_shape, kernel_window):
    peaks_cls = nn.MaxPool2d(kernel_size= kernel_window,stride=1,return_indices=True)

    h,w = torch.div(torch.Tensor(kernel_window), 2, rounding_mode='floor')

    mask_h = (map_shape[2]-2*h).int()
    mask_w = (map_shape[3]-2*w).int()
    mask_t = torch.zeros(mask_h,mask_w)

    for i in range(mask_t.shape[0]):
        for j in range(mask_t.shape[1]):
            mask_t[i,j]=(i+h)*map_shape[3]+ j+w
    
    mask = torch.zeros(map_shape[0],map_shape[1],mask_h,mask_w)
    for i in range(mask.shape[0]):
        for j in range(mask.shape[1]):
            mask[i,j,:,:]= mask_t           
    return mask, peaks_cls

def peaks_detect(map, mask, peaks_cls, heat_thresh):
    out = peaks_cls(map)

    peak = out[1]==mask
    peak_thresh = out[0]>heat_thresh

    idx = torch.where(peak*peak_thresh ==True)
    return out[0][idx], idx  # intensity, index

def distribute(index_t, device='cpu'):
    index = torch.tensor(()).to(device=device)
    for i in range(len(index_t[0])):
        index = torch.cat((index, torch.stack((index_t[0][i],index_t[1][i],index_t[2][i],index_t[3][i]),dim=0)))

    index = torch.reshape(index,(len(index_t[0]),4))
    return index

def update_peaks(pred_cen, pred_idx):
    for cnt,cord in enumerate(pred_idx):
        r_id = int(cord[2])
        a_id = int(cord[3])
        bat = int(cord[0])
        cord[2] += int(pred_cen[bat,0,r_id,a_id])
        cord[3] += int(pred_cen[bat,1,r_id,a_id])
        pred_idx[cnt]= cord
    return pred_idx

def association(intensity, index, device="cpu"):
    idx_list = torch.argsort(intensity)
    idx_list = torch.flip(idx_list,dims=[-1])
    out_intent =torch.Tensor().to(device=device)
    out_idx =torch.Tensor().to(device=device)

    while len(idx_list)!=0:
        out_intent=torch.cat((out_intent,torch.Tensor([intensity[idx_list[0]]]).to(device=device)))
        out_idx = torch.cat((out_idx, index[idx_list[0]]))
        p1_row , p1_col = index[idx_list[0]][2],index[idx_list[0]][3]
        frame_1, cls_1 = index[idx_list[0]][0],index[idx_list[0]][1]
        x1,y1 = pol2cord(p1_row,p1_col)
        idx_list = idx_list[1:]

        count = 0
        while len(idx_list)!=0 and count != len(idx_list):
            frame_2, cls_2 = index[idx_list[count]][0],index[idx_list[count]][1]
            if frame_2 == frame_1:
                p2_row , p2_col = index[idx_list[count]][2],index[idx_list[count]][3]
                x2,y2 = pol2cord(p2_row,p2_col)
                dist = distance(x1,y1,x2,y2)
            
                if dist < 2:
                   idx_list = torch.cat([idx_list[:count], idx_list[count+1:]])
                else:
                    count += 1
            else: 
                count += 1
    out_idx = torch.reshape(out_idx,(len(out_idx)//4,4))
    return out_idx,out_intent

def distance(x1,y1,x2,y2):
    dx = x1-x2
    dy = y1-y2
    return torch.sqrt(dx**2 + dy**2)
   
def pol2cord(rng_idx, agl_idx):
    range_array = torch.linspace(50,0,steps=256)
    w = torch.linspace(-1,1,steps=256) # angular range from -1 radian to 1 radian
    angle_array = torch.arcsin(w)
    range = range_array[int(rng_idx)]
    angle = angle_array[int(agl_idx)]
    return range*torch.sin(angle), range*torch.cos(angle)

def orent(orent_map, r, a, velo=0, pred=False):
    s_t = orent_map[0, r, a]
    c_t = orent_map[1, r, a]
    s = s_t/((s_t**2 + c_t**2)**0.5)
    c = c_t/((s_t**2 + c_t**2)**0.5)
    angle = torch.arccos(c).rad2deg()
    if s<0:
        angle= -angle

    if pred and torch.abs(angle)>150:
        w = torch.linspace(-1,1,steps=64)
        angle_array = torch.rad2deg(torch.arcsin(w))
        prj_angle = angle_array[a]
        delta = angle-prj_angle

        if velo>0:
            if delta<90 and delta>-90:pass
            else:
                angle +=180
        else:
            if delta<90 and delta>-90: 
                angle+=180
        if angle>180:
            angle = angle-360
        return angle
    return angle
