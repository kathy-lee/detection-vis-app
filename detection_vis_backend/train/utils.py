import numpy as np
import pandas as pd
import os
import torch
import torch.nn.functional as F
import torch.nn as nn
import pkbar
import logging

from torch.utils.data import Dataset, DataLoader, random_split, Subset
from typing import Final
from shapely.geometry import Polygon


# class TrainDataset(Dataset):
#     def __init__(self, data):
#         self.data = data
#         self.keys = list(self.data.keys())
#         self.length = len(self.data[self.keys[0]])

#     def __getitem__(self, index):
#         return {key: self.data[key][index] for key in self.keys}

#     def __len__(self):
#         return self.length

def CreateDataLoaders(dataset,config=None,seed=0):

    if(config['mode']=='random'):
        # generated training and validation set
        # number of images used for training and validation
        n_images = dataset.__len__()

        split = np.array(config['split'])
        if(np.sum(split)!=1):
            raise NameError('The sum of the train/val/test split should be equal to 1')
            return

        n_train = int(config['split'][0] * n_images)
        n_val = int(config['split'][1] * n_images)
        n_test = n_images - n_train - n_val

        train_dataset, val_dataset, test_dataset = random_split(
            dataset, [n_train, n_val,n_test], generator=torch.Generator().manual_seed(seed))

        print('===========  Dataset  ==================:')
        print('      Mode:', config['mode'])
        print('      Train Val ratio:', config['split'])
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
                                collate_fn=RADIal_collate)
        val_loader =  DataLoader(val_dataset, 
                                batch_size=config['val']['batch_size'], 
                                shuffle=False,
                                num_workers=config['val']['num_workers'],
                                pin_memory=True,
                                collate_fn=RADIal_collate)
        test_loader =  DataLoader(test_dataset, 
                                batch_size=config['test']['batch_size'], 
                                shuffle=False,
                                num_workers=config['test']['num_workers'],
                                pin_memory=True,
                                collate_fn=RADIal_collate)

        return train_loader,val_loader,test_loader
    elif(config['mode']=='sequence'):
        Sequences = {'Validation':['RECORD@2020-11-22_12.49.56', 'RECORD@2020-11-22_12.11.49',
                                   'RECORD@2020-11-22_12.28.47','RECORD@2020-11-21_14.25.06'],
                    'Test':['RECORD@2020-11-22_12.45.05','RECORD@2020-11-22_12.25.47',
                            'RECORD@2020-11-22_12.03.47','RECORD@2020-11-22_12.54.38']}
        
        # Added: Get labels and sample_keys 
        labels = pd.read_csv(os.path.join('/home/kangle/dataset/radial/RADIal-labeled', 'labels.csv')).to_numpy()
        unique_ids = np.unique(labels[:,0])
        label_dict = {}
        for i,ids in enumerate(unique_ids):
            sample_ids = np.where(labels[:,0]==ids)[0]
            label_dict[ids]=sample_ids
        sample_keys = list(label_dict.keys())

        dict_index_to_keys = {s:i for i,s in enumerate(sample_keys)}

        Val_indexes = []
        for seq in Sequences['Validation']:
            idx = np.where(labels[:,14]==seq)[0]
            Val_indexes.append(labels[idx,0])
        Val_indexes = np.unique(np.concatenate(Val_indexes))

        Test_indexes = []
        for seq in Sequences['Test']:
            idx = np.where(labels[:,14]==seq)[0]
            Test_indexes.append(labels[idx,0])
        Test_indexes = np.unique(np.concatenate(Test_indexes))

        val_ids = [dict_index_to_keys[k] for k in Val_indexes]
        test_ids = [dict_index_to_keys[k] for k in Test_indexes]
        # ####Experiment on fake data
        val_ids = [10,20,30,40,50,60,70,80]
        test_ids = [11,21,31,41,51,61,71,81]
        # ####
        train_ids = np.setdiff1d(np.arange(len(dataset)),np.concatenate([val_ids,test_ids]))
        logging.error(f"train ids: {train_ids}")

        train_dataset = Subset(dataset,train_ids)
        val_dataset = Subset(dataset,val_ids)
        test_dataset = Subset(dataset,test_ids)

        print('===========  Dataset  ==================:')
        print('      Mode:', config['mode'])
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
                                collate_fn=RADIal_collate)
        val_loader =  DataLoader(val_dataset, 
                                batch_size=config['val']['batch_size'], 
                                shuffle=False,
                                num_workers=config['val']['num_workers'],
                                pin_memory=True,
                                collate_fn=RADIal_collate)
        test_loader =  DataLoader(test_dataset, 
                                batch_size=config['test']['batch_size'], 
                                shuffle=False,
                                num_workers=config['test']['num_workers'],
                                pin_memory=True,
                                collate_fn=RADIal_collate)

        return train_loader,val_loader,test_loader
        
    else:      
        raise NameError(config['mode'], 'is not supported !')
        return


def RADIal_collate(batch):
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


def decode(self,map,threshold):
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
        range_bins,angle_bins = np.where(map[0,:,:]>=threshold)

        coordinates = []

        for range_bin,angle_bin in zip(range_bins,angle_bins):
            R = range_bin*4*self.geometry['resolution'][0] + map[1,range_bin,angle_bin] * statistics['reg_std'][0] + statistics['reg_mean'][0]
            A = (angle_bin-self.OUTPUT_DIM[2]/2)*4*geometry['resolution'][1] + map[2,range_bin,angle_bin] * statistics['reg_std'][1] + statistics['reg_mean'][1]
            C = map[0,range_bin,angle_bin]
        
            coordinates.append([R,A,C])
       
        return coordinates


def run_evaluation(net,loader,check_perf=False, detection_loss=None,segmentation_loss=None,losses_params=None):

    metrics = Metrics()
    metrics.reset()

    net.eval()
    running_loss = 0.0
    
    kbar = pkbar.Kbar(target=len(loader), width=20, always_stateful=False)

    for i, data in enumerate(loader):

        # input, out_label,segmap,labels
        inputs = data[0].to('cuda').float()
        label_map = data[1].to('cuda').float()
        seg_map_label = data[2].to('cuda').double()

        with torch.set_grad_enabled(False):
            outputs = net(inputs)

        if(detection_loss!=None and segmentation_loss!=None):
            classif_loss,reg_loss = detection_loss(outputs['Detection'], label_map,losses_params)           
            prediction = outputs['Segmentation'].contiguous().flatten()
            label = seg_map_label.contiguous().flatten()        
            loss_seg = segmentation_loss(prediction, label)
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

                metrics.update(pred_map[0],true_map,np.asarray(decode(pred_obj,0.05)),true_obj,
                            threshold=0.2,range_min=5,range_max=100) 
                
        kbar.update(i)
        
    mAP,mAR, mIoU = metrics.GetMetrics()

    return {'loss':running_loss, 'mAP':mAP, 'mAR':mAR, 'mIoU':mIoU}



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
    final_class_predictions, final_box_predictions = perform_nms(valid_class_predictions, valid_box_predictions,
                                                                 nms_threshold)

    
    # concatenate point_cloud_id, confidence score and bounding box prediction | shape: [N_FINAL, 1+1+8]
    final_Object_predictions = np.hstack((final_class_predictions[:, np.newaxis],
                                               final_box_predictions))


    return final_Object_predictions

def GetFullMetrics(predictions,object_labels,range_min=5,range_max=100,IOU_threshold=0.5):
    perfs = {}
    precision = []
    recall = []
    iou_threshold = []
    RangeError = []
    AngleError = []

    out = []

    for threshold in np.arange(0.1,0.96,0.1):

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

