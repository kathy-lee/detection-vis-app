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


from torch.utils.data import DataLoader, random_split, Subset
from shapely.geometry import Polygon


def FFTRadNet_collate(batch):
    images = []
    FFTs = []
    segmaps = []
    labels = []
    encoded_label = []

    for radar_FFT, segmap,out_label,box_labels,image in batch:
        FFTs.append(torch.tensor(radar_FFT).permute(2,0,1))
        segmaps.append(torch.tensor(segmap))
        encoded_label.append(torch.tensor(out_label))
        images.append(torch.tensor(image))
        labels.append(torch.from_numpy(box_labels))    
    return torch.stack(FFTs), torch.stack(encoded_label),torch.stack(segmaps),labels,torch.stack(images)


def ROD_collate(batch):
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
        # out = None
        # if torch.utils.data.get_worker_info() is not None:
        #     # If we're in a background process, concatenate directly into a
        #     # shared memory tensor to avoid an extra copy
        #     numel = sum([x.numel() for x in batch])
        #     storage = elem.untyped_storage()._new_shared(numel)
        #     out = elem.new(storage)
        # return torch.stack(batch, 0, out=out)
        return torch.stack(batch, 0)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        elem = batch[0]
        if elem_type.__name__ == 'ndarray':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return ROD_collate([torch.as_tensor(b) for b in batch])
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
        return {key: ROD_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(ROD_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, (list, tuple)): # container_abcs.Sequence
        # transposed = zip(*batch)
        # return [cr_collate(samples) for samples in transposed]
        return batch

    raise TypeError(default_collate_err_msg_format.format(elem_type))


def CreateDataLoaders(dataset,config=None,seed=0):

    if dataset.model_cfg['class'] == 'FFTRadNet':
        collate_func = RADIal_collate
    elif dataset.model_cfg['class'] == 'RODNet':
        collate_func = ROD_collate

    if(config['splitmode']=='random'):
        # generated training and validation set
        # number of images used for training and validation
        n_images = dataset.__len__()

        split = np.array(config['split_random'])
        if(np.sum(split)!=1):
            raise NameError('The sum of the train/val/test split should be equal to 1')
            return

        n_train = int(config['split'][0] * n_images)
        n_val = int(config['split'][1] * n_images)
        n_test = n_images - n_train - n_val

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [n_train, n_val,n_test], generator=torch.Generator().manual_seed(seed))

        print('===========  Dataset  ==================:')
        print('      Mode:', config['splitmode'])
        print('      Train Val ratio:', config['split_random'])
        print('      Training:', len(train_dataset),' indexes...',train_dataset.indices[:3])
        print('      Validation:', len(val_dataset),' indexes...',val_dataset.indices[:3])
        print('      Test:', len(test_dataset),' indexes...',test_dataset.indices[:3])
        print('')

        # create data_loaders
        train_loader = DataLoader(train_dataset, 
                                batch_size=config['train']['batch_size'], 
                                shuffle=True,
                                num_workers=config['train']['num_workers'],
                                pin_memory=True,
                                collate_fn=collate_func)
        val_loader = DataLoader(val_dataset, 
                                batch_size=config['val']['batch_size'], 
                                shuffle=False,
                                num_workers=config['val']['num_workers'],
                                pin_memory=True,
                                collate_fn=collate_func)
        test_loader = DataLoader(test_dataset, 
                                batch_size=config['test']['batch_size'], 
                                shuffle=False,
                                num_workers=config['test']['num_workers'],
                                pin_memory=True,
                                collate_fn=collate_func)

        train_ids = train_dataset.indices
        val_ids = val_dataset.indices
        test_ids = test_dataset.indices
        return train_loader,val_loader,test_loader,train_ids,val_ids,test_ids
    elif(config['splitmode']=='sequence'):
        Sequences = config['split_sequence']
        
        if dataset.model_cfg['class'] == 'FFTRadNet':
            # Sequences = {'val':['RECORD@2020-11-22_12.49.56', 'RECORD@2020-11-22_12.11.49',
            #                            'RECORD@2020-11-22_12.28.47','RECORD@2020-11-21_14.25.06'],
            #             'test':['RECORD@2020-11-22_12.45.05','RECORD@2020-11-22_12.25.47',
            #                     'RECORD@2020-11-22_12.03.47','RECORD@2020-11-22_12.54.38']}
            
            labels = dataset.datasets[0].labels
            dict_index_to_keys = {s:i for i,s in enumerate(dataset.datasets[0].sample_keys)}

            Val_indexes = []
            for seq in Sequences['val']:
                idx = np.where(labels[:,14]==seq)[0]
                Val_indexes.append(labels[idx,0])
            Val_indexes = np.unique(np.concatenate(Val_indexes))

            Test_indexes = []
            for seq in Sequences['test']:
                idx = np.where(labels[:,14]==seq)[0]
                Test_indexes.append(labels[idx,0])
            Test_indexes = np.unique(np.concatenate(Test_indexes))

            val_ids = [dict_index_to_keys[k] for k in Val_indexes]
            test_ids = [dict_index_to_keys[k] for k in Test_indexes]
            train_ids = np.setdiff1d(np.arange(len(dataset)),np.concatenate([val_ids,test_ids]))   
        else:
            val_dataset_indices = [i for i, name in enumerate(dataset.dataset_names) if name in Sequences['val']]
            val_ids = []
            for idx in val_dataset_indices:
                start_idx = 0 if idx == 0 else dataset.cumulative_lengths[idx-1]
                end_idx = dataset.cumulative_lengths[idx]
                val_ids.extend(range(start_idx, end_idx))

            test_dataset_indices = [i for i, name in enumerate(dataset.dataset_names) if name in Sequences['test']]
            test_ids = []
            for idx in test_dataset_indices:
                start_idx = 0 if idx == 0 else dataset.cumulative_lengths[idx-1]
                end_idx = dataset.cumulative_lengths[idx]
                test_ids.extend(range(start_idx, end_idx))

            train_ids = np.setdiff1d(np.arange(len(dataset)),np.concatenate([val_ids,test_ids]))

        train_dataset = Subset(dataset,train_ids)
        val_dataset = Subset(dataset,val_ids)
        test_dataset = Subset(dataset,test_ids)

        print('===========  Dataset  ==================:')
        print('      Mode:', config['splitmode'])
        print('      Training:', len(train_dataset))
        print('      Validation:', len(val_dataset))
        print('      Test:', len(test_dataset))
        print('')

        # create data_loaders
        train_loader = DataLoader(train_dataset, 
                                batch_size=config['train']['batch_size'], 
                                shuffle=True,
                                num_workers=config['train']['num_workers'],
                                pin_memory=True,
                                collate_fn=collate_func)
        val_loader =  DataLoader(val_dataset, 
                                batch_size=config['val']['batch_size'], 
                                shuffle=False,
                                num_workers=config['val']['num_workers'],
                                pin_memory=True,
                                collate_fn=collate_func)
        test_loader =  DataLoader(test_dataset, 
                                batch_size=config['test']['batch_size'], 
                                shuffle=False,
                                num_workers=config['test']['num_workers'],
                                pin_memory=True,
                                collate_fn=collate_func)
        print(f"Train/Val/Test Dataloaders in {config['splitmode']} mode created.")
        return train_loader,val_loader,test_loader,train_ids,val_ids,test_ids
    else:      
        raise NameError(config['splitmode'], 'is not supported !')
        return


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


def DisplayHMI(image, input, model_outputs, obj_labels):

    # Model outputs
    pred_obj = model_outputs['Detection'].detach().cpu().numpy().copy()[0]
    out_seg = torch.sigmoid(model_outputs['Segmentation']).detach().cpu().numpy().copy()[0,0]
    print(f"pred_obj before decode: {pred_obj.shape}")
    # Decode the output detection map
    pred_obj = decode(pred_obj,0.05)
    print(f"pred_obj after decode: {type(pred_obj)}, {len(pred_obj)}")
    pred_obj = np.asarray(pred_obj)
    print(f"pred_obj -> np.asarry: {pred_obj.shape}")
    print(pred_obj[0,0:3])

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
