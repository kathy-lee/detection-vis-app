import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import logging
import cv2
import polarTransform
import re
import json
import math


from shapely.geometry import Polygon
from typing import Optional


def FFTRadNet_collate(batch):
    images = []
    FFTs = []
    segmaps = []
    labels = []
    encoded_label = []

    for radar_FFT, segmap,out_label,box_labels,image in batch:
        FFTs.append(torch.tensor(radar_FFT))
        segmaps.append(torch.tensor(segmap))
        encoded_label.append(torch.tensor(out_label))
        images.append(torch.tensor(image))
        labels.append(torch.from_numpy(box_labels))    
    return torch.stack(FFTs), torch.stack(encoded_label),torch.stack(segmaps),labels,torch.stack(images)

def DAROD_collate(batch):
    # raw labels from CARRADA dataset: 1,2,3
    # raw labels from RADDet dataset: 1,2,3,4,5,6
    radar = [torch.tensor(item['radar'].copy()) for item in batch]
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
    # print(f"padding: {gt_labels} -> {padded_labels}")
    return {'radar': radar, 'label': padded_labels, 'boxes': padded_bboxes}

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


class FocalLoss(nn.Module):
    """
    Focal loss class. Stabilize training by reducing the weight of easily classified background sample and focussing
    on difficult foreground detections.
    """

    def __init__(self, gamma=0, size_average=False):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.size_average = size_average

    def forward(self, prediction, target):

        # get class probability
        pt = torch.where(target == 1.0, prediction, 1-prediction)

        # compute focal loss
        loss = -1 * (1-pt)**self.gamma * torch.log(pt+1e-6)

        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()



def pixor_loss(batch_predictions, batch_labels,param):

    #########################
    #  classification loss  #
    #########################
    classification_prediction = batch_predictions[:, 0,:, :].contiguous().flatten()
    classification_label = batch_labels[:, 0,:, :].contiguous().flatten()

    if(param['classification']=='FocalLoss'):
        focal_loss = FocalLoss(gamma=2)
        classification_loss = focal_loss(classification_prediction, classification_label)
    else:
        classification_loss = F.binary_cross_entropy(classification_prediction.double(), classification_label.double(),reduction='sum')

    
    #####################
    #  Regression loss  #
    #####################

    regression_prediction = batch_predictions.permute([0, 2, 3, 1])[:, :, :, :-1]
    regression_prediction = regression_prediction.contiguous().view([regression_prediction.size(0)*
                        regression_prediction.size(1)*regression_prediction.size(2), regression_prediction.size(3)])
    regression_label = batch_labels.permute([0, 2, 3, 1])[:, :, :, :-1]
    regression_label = regression_label.contiguous().view([regression_label.size(0)*regression_label.size(1)*
                                                           regression_label.size(2), regression_label.size(3)])

    positive_mask = torch.nonzero(torch.sum(torch.abs(regression_label), dim=1))
    pos_regression_label = regression_label[positive_mask.squeeze(), :]
    pos_regression_prediction = regression_prediction[positive_mask.squeeze(), :]


    T = batch_labels[:,1:]
    P = batch_predictions[:,1:]
    M = batch_labels[:,0].unsqueeze(1)

    if(param['regression']=='SmoothL1Loss'):
        reg_loss_fct = nn.SmoothL1Loss(reduction='sum')
    else:
        reg_loss_fct = nn.L1Loss(reduction='sum')
    
    regression_loss = reg_loss_fct(P*M,T)
    NbPts = M.sum()
    if(NbPts>0):
        regression_loss/=NbPts

    return classification_loss,regression_loss


def RA_to_cartesian_box(data):
    L = 4
    W = 1.8
    
    boxes = []
    for i in range(len(data)):
        
        x = np.sin(np.radians(data[i][1])) * data[i][0]
        y = np.cos(np.radians(data[i][1])) * data[i][0]

        boxes.append([x - W/2,y,x + W/2,y, x + W/2,y+L,x - W/2,y+L,data[i][0],data[i][1]])
              
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
    print(f"shape before perform nms: {valid_class_predictions.shape}, {valid_box_predictions.shape}")
    final_class_predictions, final_box_predictions = perform_nms(valid_class_predictions, valid_box_predictions,nms_threshold)
    print(f"shape should be: {final_class_predictions.shape}, {final_box_predictions.shape}")

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

class SmoothCELoss(nn.Module):
    """
    Smooth cross entropy loss
    SCE = SmoothL1Loss() + BCELoss()
    By default reduction is mean. 
    """
    def __init__(self, alpha):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss()
        self.bce = nn.BCELoss()
        self.alpha = alpha
    
    def forward(self, input, target):
        return self.alpha * self.bce(input, target) + (1-self.alpha) * self.smooth_l1(input, target)


def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:
    r"""Converts an integer label x-D tensor to a one-hot (x+1)-D tensor.

    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, *)`,
                                where N is batch size. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.

    Returns:
        torch.Tensor: the labels in one hot tensor of shape :math:`(N, C, *)`,

    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> one_hot(labels, num_classes=3)
        tensor([[[[1.0000e+00, 1.0000e-06],
                  [1.0000e-06, 1.0000e+00]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e+00],
                  [1.0000e-06, 1.0000e-06]],
        <BLANKLINE>
                 [[1.0000e-06, 1.0000e-06],
                  [1.0000e+00, 1.0000e-06]]]])

    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    shape = labels.shape
    one_hot = torch.zeros(shape[0], num_classes, *shape[1:],
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


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


def soft_dice_loss(input: torch.Tensor, target: torch.Tensor, eps: float = 1e-8,
                   global_weight: float = 1.) -> torch.Tensor:
    r"""Function that computes Sørensen-Dice Coefficient loss.
    Arthur: add ^2 for soft formulation

    See :class:`~kornia.losses.DiceLoss` for details.
    """
    if not torch.is_tensor(input):
        raise TypeError("Input type is not a torch.Tensor. Got {}"
                        .format(type(input)))

    if not len(input.shape) == 4:
        raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                         .format(input.shape))

    if not input.shape[-2:] == target.shape[-2:]:
        raise ValueError("input and target shapes must be the same. Got: {} and {}"
                         .format(input.shape, input.shape))

    if not input.device == target.device:
        raise ValueError(
            "input and target must be in the same device. Got: {} and {}" .format(
                input.device, target.device))

    # compute softmax over the classes axis
    input_soft: torch.Tensor = F.softmax(input, dim=1)

    # create the labels one hot tensor
    target_one_hot: torch.Tensor = one_hot(target, num_classes=input.shape[1], device=input.device, dtype=input.dtype)

    # compute the actual dice score
    dims = (1, 2, 3)
    intersection = torch.sum(input_soft * target_one_hot, dims)
    cardinality = torch.sum(torch.pow(input_soft, 2) + torch.pow(target_one_hot, 2), dims)

    dice_score = 2. * intersection / (cardinality + eps)
    return global_weight*torch.mean(-dice_score + 1.)


class SoftDiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.
    Arthur: add ^2 for soft formulation

    According to [1], we compute the Sørensen-Dice Coefficient as follows:

    .. math::

        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}

    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.

    the loss, is finally computed as:

    .. math::

        \text{loss}(x, class) = 1 - \text{Dice}(x, class)

    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient

    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.

    Examples:
        >>> N = 5  # num_classes
        >>> loss = DiceLoss()
        >>> input = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(input, target)
        >>> output.backward()
    """

    def __init__(self, global_weight: float = 1.) -> None:
        super(SoftDiceLoss, self).__init__()
        self.eps: float = 1e-6
        self.global_weight = global_weight

    def forward(  # type: ignore
            self,
            input: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        return soft_dice_loss(input, target, self.eps, self.global_weight)
    

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


def boxDecoder(yolohead_output, input_size, anchors_layer, num_class, scale=1., device='cuda:0'):
    """ Decoder output from yolo head to boxes """    
    grid_size = yolohead_output.shape[1:4]
    num_anchors_layer = len(anchors_layer)
    grid_strides = torch.tensor(input_size, dtype=torch.float32).to(device) / torch.tensor(grid_size, dtype=torch.float32).to(device)
    
    reshape_size = [yolohead_output.size(0)] + list(grid_size) + [num_anchors_layer, 7+num_class]
    pred_raw = yolohead_output.view(reshape_size)
    
    raw_xyz, raw_whd, raw_conf, raw_prob = torch.split(pred_raw, (3,3,1,num_class), dim=-1)

    xyz_grid = torch.meshgrid(torch.arange(grid_size[0]).to(device), 
                              torch.arange(grid_size[1]).to(device),
                              torch.arange(grid_size[2]).to(device), indexing="ij") # Added indexing style
    xyz_grid = torch.unsqueeze(torch.stack(xyz_grid, dim=-1).to(device), dim=3)
    xyz_grid = xyz_grid.permute(1, 0, 2, 3, 4)
    xyz_grid = xyz_grid.unsqueeze(0).repeat(yolohead_output.size(0), 1, 1, 1, num_anchors_layer, 1)

    pred_xyz = ((torch.sigmoid(raw_xyz) * scale) - 0.5 * (scale - 1) + xyz_grid) * grid_strides

    # Clipping values 
    raw_whd = torch.clamp(raw_whd, 1e-12, 1e12)
    
    pred_whd = torch.exp(raw_whd) * torch.tensor(anchors_layer, dtype=torch.float32).to(device)
    pred_xyzwhd = torch.cat([pred_xyz, pred_whd], dim=-1)

    pred_conf = torch.sigmoid(raw_conf)
    pred_prob = torch.sigmoid(raw_prob)
    
    return pred_raw, torch.cat([pred_xyzwhd, pred_conf, pred_prob], dim=-1)


def extractYoloInfo(yolo_output_format_data):
    """ Extract box, objectness, class from yolo output format data """
    box = yolo_output_format_data[..., :6]
    conf = yolo_output_format_data[..., 6:7]
    category = yolo_output_format_data[..., 7:]
    return box, conf, category


def yolo1Loss(pred_box, gt_box, gt_conf, input_size, if_box_loss_scale=True):
    """ loss function for box regression (based on YOLOV1) """
    assert pred_box.shape == gt_box.shape
    if if_box_loss_scale:
        scale = 2.0 - 1.0 * gt_box[..., 3:4] * gt_box[..., 4:5] * gt_box[..., 5:6] /\
                                    (input_size[0] * input_size[1] * input_size[2])
    else:
        scale = 1.0
        
    # YOLOv1 original loss function
    giou_loss = gt_conf * scale * ((pred_box[..., :3] - gt_box[..., :3]).pow(2) + \
                    (pred_box[..., 3:].sqrt() - gt_box[..., 3:].sqrt()).pow(2))
    return giou_loss


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


def focalLoss(raw_conf, pred_conf, gt_conf, pred_box, raw_boxes, input_size, iou_loss_threshold=0.5):
    """ Calculate focal loss for objectness """
    iou = iou3d(pred_box.unsqueeze(-2), raw_boxes.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1), input_size)
    max_iou, _ = iou.max(dim=-1)
    max_iou = max_iou.unsqueeze(-1)

    gt_conf_negative = (1.0 - gt_conf) * (max_iou < iou_loss_threshold).float()
    conf_focal = (gt_conf - pred_conf).pow(2)
    alpha = 0.01

    focal_loss = conf_focal * (
        gt_conf * F.binary_cross_entropy_with_logits(raw_conf, gt_conf, reduction='none')
        +
        alpha * gt_conf_negative * F.binary_cross_entropy_with_logits(raw_conf, gt_conf, reduction='none')
    )
    return focal_loss


def categoryLoss(raw_category, pred_category, gt_category, gt_conf):
    """ Category Cross Entropy loss """
    category_loss = gt_conf * F.binary_cross_entropy_with_logits(input=raw_category, target=gt_category)
    return category_loss


def lossYolo(pred_raw, pred, label, raw_boxes, input_size, focal_loss_iou_threshold):
    """ Calculate loss function of YOLO HEAD 
    Args:
        feature_stages      ->      3 different feature stages after YOLO HEAD
                                    with shape [None, r, a, d, num_anchors, 7+num_class]
        gt_stages           ->      3 different ground truth stages 
                                    with shape [None, r, a, d, num_anchors, 7+num_class]"""
    assert len(raw_boxes.shape) == 3
    input_size = torch.tensor(input_size).float()
    assert pred_raw.shape == label.shape
    assert pred_raw.shape[0] == len(raw_boxes)
    assert pred.shape == label.shape
    assert pred.shape[0] == len(raw_boxes)
    raw_box, raw_conf, raw_category = extractYoloInfo(pred_raw)
    pred_box, pred_conf, pred_category = extractYoloInfo(pred)
    gt_box, gt_conf, gt_category = extractYoloInfo(label)
    giou_loss = yolo1Loss(pred_box, gt_box, gt_conf, input_size, \
                            if_box_loss_scale=False)
    focal_loss = focalLoss(raw_conf, pred_conf, gt_conf, pred_box, raw_boxes, \
                            input_size, focal_loss_iou_threshold)
    category_loss = categoryLoss(raw_category, pred_category, gt_category, gt_conf)
    giou_total_loss = torch.mean(torch.sum(giou_loss, dim=[1, 2, 3, 4]))
    conf_total_loss = torch.mean(torch.sum(focal_loss, dim=[1, 2, 3, 4]))
    category_total_loss = torch.mean(torch.sum(category_loss, dim=[1, 2, 3, 4]))
    return giou_total_loss, conf_total_loss, category_total_loss

