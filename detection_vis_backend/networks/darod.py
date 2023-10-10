import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.ops import nms
from torch.nn import GroupNorm

class DARODBlock2D(nn.Module):
    def __init__(self, in_channels, filters, padding=1, kernel_size=3, num_conv=2, dilation_rate=1,
                 strides=1, activation="leaky_relu", block_norm=None,
                 pooling_size=2, pooling_strides=2, name=None):
        
        super(DARODBlock2D, self).__init__()

        self.num_conv = num_conv
        self.block_norm = block_norm
        self.activation_name = activation

        # Create first convolution, normalization, and activation
        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=filters, 
                               kernel_size=kernel_size, stride=strides,
                               padding=padding, dilation=dilation_rate)
        self.norm1 = self._get_norm(block_norm, filters, name)
        self.activation1 = self._get_activation(activation)

        # Create second convolution, normalization, and activation
        self.conv2 = nn.Conv2d(in_channels=filters, out_channels=filters, 
                               kernel_size=kernel_size, stride=strides,
                               padding=padding, dilation=dilation_rate)
        self.norm2 = self._get_norm(block_norm, filters, name)
        self.activation2 = self._get_activation(activation)

        # Optional third convolution, normalization, and activation
        if num_conv == 3:
            self.conv3 = nn.Conv2d(in_channels=filters, out_channels=filters, 
                                   kernel_size=kernel_size, stride=strides,
                                   padding=padding, dilation=dilation_rate)
            self.norm3 = self._get_norm(block_norm, filters, name)
            self.activation3 = self._get_activation(activation)

        # Max pooling
        self.maxpooling = nn.MaxPool2d(kernel_size=pooling_size, stride=pooling_strides)

    def _get_norm(self, block_norm, filters, name):
        if block_norm == "batch_norm":
            return nn.BatchNorm2d(filters)
        elif block_norm == "group_norm":
            return GroupNorm(32, filters)  # Assuming 32 groups
        elif block_norm == "layer_norm":
            return GroupNorm(1, filters)
        elif block_norm == "instance_norm":
            return nn.InstanceNorm2d(filters)
        elif block_norm is None:
            return None
        else:
            raise NotImplementedError("Unsupported normalization.")

    def _get_activation(self, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "leaky_relu":
            return nn.LeakyReLU(0.001)
        else:
            raise NotImplementedError("Unsupported activation.")

    def forward(self, x):
        # Apply first convolution, normalization, and activation
        x = self.conv1(x)
        if self.block_norm is not None:
            x = self.norm1(x)
        x = self.activation1(x)

        # Apply second convolution, normalization, and activation
        x = self.conv2(x)
        if self.block_norm is not None:
            x = self.norm2(x)
        x = self.activation2(x)

        # Optionally apply third convolution, normalization, and activation
        if self.num_conv == 3:
            x = self.conv3(x)
            if self.block_norm is not None:
                x = self.norm3(x)
            x = self.activation3(x)

        # Apply max pooling
        x = self.maxpooling(x)

        return x
    

class RoIBBox(nn.Module):
    def __init__(self, anchors, config):
        super(RoIBBox, self).__init__()
        self.config = config
        self.anchors = torch.tensor(anchors, dtype=torch.float32)

    def forward(self, rpn_bbox_deltas, rpn_labels, training=True):
        anchors = self.anchors
        pre_nms_topn = self.config["fastrcnn"]["pre_nms_topn_train"] if training else self.config["fastrcnn"]["pre_nms_topn_test"]
        post_nms_topn = self.config["fastrcnn"]["post_nms_topn_train"] if training else self.config["fastrcnn"]["post_nms_topn_test"]
        nms_iou_threshold = self.config["rpn"]["rpn_nms_iou"]
        variances = torch.tensor(self.config["rpn"]["variances"])
        total_anchors = anchors.shape[0]
        batch_size = rpn_bbox_deltas.shape[0]
        rpn_bbox_deltas = rpn_bbox_deltas.view(batch_size, total_anchors, 4)
        
        # Convert softmax in PyTorch
        rpn_labels = F.softmax(rpn_labels, dim=-1)
        rpn_labels = rpn_labels.view(batch_size, total_anchors)
        
        rpn_bbox_deltas *= variances
        # Custom utility function
        rpn_bboxes = get_bboxes_from_deltas(anchors, rpn_bbox_deltas)

        # Use PyTorch's topk() for this functionality
        _, pre_indices = torch.topk(rpn_labels, pre_nms_topn, dim=1)

        # Use PyTorch's gather() for this functionality
        pre_roi_bboxes = torch.gather(rpn_bboxes, 1, pre_indices.unsqueeze(-1).expand(-1, -1, 4))
        pre_roi_labels = torch.gather(rpn_labels, 1, pre_indices)

        # Reshaping using view in PyTorch
        pre_roi_bboxes = pre_roi_bboxes.view(batch_size, pre_nms_topn, 1, 4)
        pre_roi_labels = pre_roi_labels.view(batch_size, pre_nms_topn, 1)

        # Assuming the custom utility function is adapted for PyTorch
        roi_bboxes, roi_scores, _, _ = non_max_suppression(pre_roi_bboxes, pre_roi_labels,
                                                                      max_output_size_per_class=post_nms_topn,
                                                                      max_total_size=post_nms_topn,
                                                                      iou_threshold=nms_iou_threshold)

        return roi_bboxes.detach(), roi_scores.detach()


class RadarFeatures(nn.Module):
    """
    Extracting radar feature from RPN proposed boxes.
    This layer extracts range and Doppler values from RPN proposed
    boxes which have scores > 0.5. Otherwise, range and Doppler values
    are set to -1
    """
    def __init__(self, config):
        super(RadarFeatures, self).__init__()
        self.config = config

    def forward(self, roi_bboxes, roi_scores):
        """
        :param roi_bboxes: Bounding boxes from RPN
        :param roi_scores: Scores of the bounding boxes
        :return: radar_features
        """

        # Denormalize bounding boxes
        roi_bboxes = self.denormalize_bboxes(roi_bboxes, 
                                             height=self.config["model"]["input_size"][0],
                                             width=self.config["model"]["input_size"][1])

        # Get centers of roi boxes
        h = roi_bboxes[..., 2] - roi_bboxes[..., 0]
        w = roi_bboxes[..., 3] - roi_bboxes[..., 1]

        ctr_y = roi_bboxes[..., 0] + h / 2
        ctr_x = roi_bboxes[..., 1] + w / 2

        # Get radar feature
        range_values = self.config["data"]["range_res"] * ctr_y.float()
        doppler_values = torch.abs(32 - (self.config["data"]["doppler_res"] * ctr_x.float()))
        radar_features = torch.stack([range_values, doppler_values], dim=-1)

        # Get valid radar features
        valid_mask = (roi_scores > 0.5) & (h > 0) & (w > 0)
        radar_features = torch.where(valid_mask.unsqueeze(-1), radar_features, torch.ones_like(radar_features) * -1)

        return radar_features

    def denormalize_bboxes(self, bboxes, height, width):
        """
        Denormalize bounding boxes
        """
        bboxes[..., [0, 2]] = bboxes[..., [0, 2]] * height
        bboxes[..., [1, 3]] = bboxes[..., [1, 3]] * width
        return bboxes
    

class RoIDelta(nn.Module):
    """
    Calculating faster rcnn actual bounding box deltas and labels.
    This layer only runs in the training phase.
    """
    def __init__(self, config):
        super(RoIDelta, self).__init__()
        self.config = config

    def forward(self, roi_bboxes, gt_boxes, gt_labels):
        """
        :param roi_bboxes: ROI bounding boxes from RPN
        :param gt_boxes: Ground truth boxes
        :param gt_labels: Ground truth labels
        :return: Depending on config, either ROI box deltas and labels or expanded ROI boxes, GT boxes, and labels.
        """
        total_labels = self.config["data"]["total_labels"]
        total_pos_bboxes = int(self.config["fastrcnn"]["frcnn_boxes"] / 3)
        total_neg_bboxes = int(self.config["fastrcnn"]["frcnn_boxes"] * (2 / 3))
        variances = torch.tensor(self.config["fastrcnn"]["variances_boxes"])
        adaptive_ratio = self.config["fastrcnn"]["adaptive_ratio"]
        positive_th = self.config["fastrcnn"]["positive_th"]

        batch_size, total_bboxes = roi_bboxes.shape[0], roi_bboxes.shape[1]
        iou_map = generate_iou_map(roi_bboxes, gt_boxes)
        max_indices_each_gt_box = torch.argmax(iou_map, dim=2)
        merged_iou_map = torch.max(iou_map, dim=2).values

        pos_mask = merged_iou_map > positive_th
        pos_mask = randomly_select_xyz_mask(pos_mask, torch.tensor([total_pos_bboxes]))

        neg_mask = (merged_iou_map < positive_th) & (merged_iou_map >= 0.0)
        
        if adaptive_ratio:
            pos_count = pos_mask.int().sum(dim=-1)
            # Keep a 33%/66% ratio of positive/negative bboxes
            total_neg_bboxes = (pos_count + 1) * 3
            neg_mask = randomly_select_xyz_mask(neg_mask, total_neg_bboxes,
                                                seed=self.config["training"]["seed"])
        else:
            neg_mask = randomly_select_xyz_mask(neg_mask, torch.tensor([total_neg_bboxes], dtype=torch.int32),
                                                seed=self.config["training"]["seed"])

        gt_boxes_map = torch.gather(gt_boxes, 1, max_indices_each_gt_box.unsqueeze(-1).expand(-1,-1,4))
        expanded_gt_boxes = torch.where(pos_mask.unsqueeze(-1), gt_boxes_map, torch.zeros_like(gt_boxes_map))

        gt_labels_map = torch.gather(gt_labels, 1, max_indices_each_gt_box)
        pos_gt_labels = torch.where(pos_mask, gt_labels_map, torch.tensor(-1, dtype=torch.int32))
        neg_gt_labels = neg_mask.int()
        expanded_gt_labels = pos_gt_labels + neg_gt_labels

        roi_bbox_deltas = get_deltas_from_bboxes(roi_bboxes, expanded_gt_boxes) / variances

        roi_bbox_labels = torch.nn.functional.one_hot(expanded_gt_labels, num_classes=total_labels)
        scatter_indices = roi_bbox_labels.unsqueeze(-1).repeat(1, 1, 1, 4)
        roi_bbox_deltas = scatter_indices * roi_bbox_deltas.unsqueeze(-2)
        roi_bbox_deltas = roi_bbox_deltas.reshape(batch_size, total_bboxes * total_labels, 4)

        if self.config["fastrcnn"]["reg_loss"] == "sl1":
            return roi_bbox_deltas.detach(), roi_bbox_labels.detach()
        elif self.config["fastrcnn"]["reg_loss"] == "giou":
            expanded_roi_boxes = scatter_indices * roi_bboxes.unsqueeze(-2)
            expanded_roi_boxes = expanded_roi_boxes.reshape(batch_size, total_bboxes * total_labels, 4)
            
            expanded_gt_boxes = scatter_indices * expanded_gt_boxes.unsqueeze(-2)
            expanded_gt_boxes = expanded_gt_boxes.reshape(batch_size, total_bboxes * total_labels, 4)
            return expanded_roi_boxes.detach(), expanded_gt_boxes.detach(), roi_bbox_labels.detach()


class RoIPooling(torch.nn.Module):
    """
    Reducing all feature maps to the same size.
    Firstly cropping bounding boxes from the feature maps and then resizing it to the pooling size.
    """
    def __init__(self, config):
        super(RoIPooling, self).__init__()
        self.config = config
        self.pooling_size = self.config["fastrcnn"]["pooling_size"]
        self.roi_pool = torchvision.ops.RoIPool(output_size=(self.pooling_size, self.pooling_size), spatial_scale=1.0)

    def forward(self, feature_map, roi_bboxes):
        """
        :param feature_map: feature map
        :param roi_bboxes: roi boxes
        :return: pooled features
        """
        batch_size, total_bboxes = roi_bboxes.shape[0], roi_bboxes.shape[1]
        
        # We need to arrange bbox indices for each batch
        pooling_bbox_indices = torch.arange(batch_size).unsqueeze(1).repeat(1, total_bboxes)
        pooling_bboxes = roi_bboxes.reshape(-1, 4)
        rois = torch.cat([pooling_bbox_indices.view(-1,1).float(), pooling_bboxes], dim=1)

        # Use RoIPooling
        pooling_feature_map = self.roi_pool(feature_map, rois)
        final_pooling_feature_map = pooling_feature_map.view(batch_size, total_bboxes, pooling_feature_map.shape[1], pooling_feature_map.shape[2], pooling_feature_map.shape[3])
        return final_pooling_feature_map


def get_deltas_from_bboxes(bboxes, gt_boxes):
    """
    Calculating bounding box deltas for given bounding box and ground truth boxes.
    :param bboxes: (batch_size, total_bboxes, [y1, x1, y2, x2])
    :param gt_boxes: (batch_size, total_bboxes, [y1, x1, y2, x2])
    :return:  final_deltas = (batch_size, total_bboxes, [delta_y, delta_x, delta_h, delta_w])
    """
    bbox_width = bboxes[..., 3] - bboxes[..., 1]
    bbox_height = bboxes[..., 2] - bboxes[..., 0]
    bbox_ctr_x = bboxes[..., 1] + 0.5 * bbox_width
    bbox_ctr_y = bboxes[..., 0] + 0.5 * bbox_height
    
    gt_width = gt_boxes[..., 3] - gt_boxes[..., 1]
    gt_height = gt_boxes[..., 2] - gt_boxes[..., 0]
    gt_ctr_x = gt_boxes[..., 1] + 0.5 * gt_width
    gt_ctr_y = gt_boxes[..., 0] + 0.5 * gt_height
    
    bbox_width = torch.where(bbox_width == 0, torch.tensor(1e-3).to(bbox_width.device), bbox_width)
    bbox_height = torch.where(bbox_height == 0, torch.tensor(1e-3).to(bbox_height.device), bbox_height)
    delta_x = torch.where(gt_width == 0, torch.zeros_like(gt_width), (gt_ctr_x - bbox_ctr_x) / bbox_width)
    delta_y = torch.where(gt_height == 0, torch.zeros_like(gt_height), (gt_ctr_y - bbox_ctr_y) / bbox_height)
    delta_w = torch.where(gt_width == 0, torch.zeros_like(gt_width), torch.log(gt_width / bbox_width))
    delta_h = torch.where(gt_height == 0, torch.zeros_like(gt_height), torch.log(gt_height / bbox_height))
    
    return torch.stack([delta_y, delta_x, delta_h, delta_w], dim=-1)


def randomly_select_xyz_mask(mask, select_xyz, seed):
    """
    Selecting x, y, z number of True elements for corresponding batch and replacing others to False
    :param mask: (batch_size, [m_bool_value])
    :param select_xyz: (batch_size, [m_bool_value])
    :param seed: seed
    :return:  selected_valid_mask = (batch_size, [m_bool_value])
    """
    torch.manual_seed(seed)
    maxval = select_xyz.max().item() * 10
    random_mask = torch.randint(low=1, high=maxval, size=mask.shape, dtype=torch.int32)
    multiplied_mask = mask.int() * random_mask
    sorted_mask_indices = torch.argsort(multiplied_mask, descending=True)
    sorted_positions = torch.argsort(sorted_mask_indices)
    selected_mask = sorted_positions < select_xyz.unsqueeze(1)
    return mask & selected_mask


def generate_iou_map(bboxes, gt_boxes):
    """
    Calculating iou values for each ground truth boxes in batched manner.
    :param bboxes: (batch_size, total_bboxes, [y1, x1, y2, x2])
    :param gt_boxes: (batch_size, total_gt_boxes, [y1, x1, y2, x2])
    :return: iou_map = (batch_size, total_bboxes, total_gt_boxes)
    """
    bbox_y1, bbox_x1, bbox_y2, bbox_x2 = bboxes.split(1, dim=-1)
    gt_y1, gt_x1, gt_y2, gt_x2 = gt_boxes.split(1, dim=-1)
    # Calculate bbox and ground truth boxes areas
    gt_area = (gt_y2 - gt_y1) * (gt_x2 - gt_x1).squeeze(-1)
    bbox_area = (bbox_y2 - bbox_y1) * (bbox_x2 - bbox_x1).squeeze(-1)
    #
    x_top = torch.maximum(bbox_x1, gt_x1.transpose(1, 2))
    y_top = torch.maximum(bbox_y1, gt_y1.transpose(1, 2))
    x_bottom = torch.minimum(bbox_x2, gt_x2.transpose(1, 2))
    y_bottom = torch.minimum(bbox_y2, gt_y2.transpose(1, 2))
    ### Calculate intersection area
    intersection_area = torch.maximum(x_bottom - x_top, torch.zeros_like(x_bottom)) * torch.maximum(y_bottom - y_top, torch.zeros_like(y_bottom))
    ### Calculate union area
    union_area = (bbox_area.unsqueeze(-1) + gt_area.unsqueeze(1) - intersection_area)
    # Intersection over Union
    return intersection_area / torch.clamp(union_area, min=1e-7)


def non_max_suppression(pred_bboxes, pred_labels, iou_threshold=0.5, top_k=200):
    """
    Applying non maximum suppression using torchvision's nms function.

    :param pred_bboxes: (batch_size, total_bboxes, [y1, x1, y2, x2]), total_labels should be 1 for binary operations like in rpn
    :param pred_labels: (batch_size, total_bboxes, total_labels)
    :param iou_threshold: threshold for IOU to determine when to suppress boxes
    :param top_k: maximum number of bounding boxes to consider

    :return: list of [nms_boxes, nms_scores, nms_classes]
             where each item in the list corresponds to a batch.
    """
    batch_size = pred_bboxes.shape[0]
    nms_boxes_list = []
    nms_scores_list = []
    nms_classes_list = []

    for i in range(batch_size):
        boxes = pred_bboxes[i]
        scores = pred_labels[i]

        # Sort scores and boxes based on scores
        _, indices = scores.sort(descending=True)
        boxes = boxes[indices]
        scores = scores[indices]

        # Apply NMS
        keep_indices = nms(boxes, scores, iou_threshold)[:top_k]
        keep_boxes = boxes[keep_indices]
        keep_scores = scores[keep_indices]

        nms_boxes_list.append(keep_boxes)
        nms_scores_list.append(keep_scores)
        # Assuming that classes are just the index of the max score
        keep_classes = scores.argmax(dim=1)[keep_indices]
        nms_classes_list.append(keep_classes)

    return nms_boxes_list, nms_scores_list, nms_classes_list


def get_bboxes_from_deltas(anchors, deltas):
    """
    Calculating bounding boxes for given bounding box and delta values.
    :param anchors: (batch_size, total_bboxes, [y1, x1, y2, x2])
    :param deltas: (batch_size, total_bboxes, [delta_y, delta_x, delta_h, delta_w])
    :return:  final_boxes = (batch_size, total_bboxes, [y1, x1, y2, x2])
    """
    all_anc_width = anchors[..., 3] - anchors[..., 1]
    all_anc_height = anchors[..., 2] - anchors[..., 0]
    all_anc_ctr_x = anchors[..., 1] + 0.5 * all_anc_width
    all_anc_ctr_y = anchors[..., 0] + 0.5 * all_anc_height
    
    all_bbox_width = torch.exp(deltas[..., 3]) * all_anc_width
    all_bbox_height = torch.exp(deltas[..., 2]) * all_anc_height
    all_bbox_ctr_x = (deltas[..., 1] * all_anc_width) + all_anc_ctr_x
    all_bbox_ctr_y = (deltas[..., 0] * all_anc_height) + all_anc_ctr_y
    
    y1 = all_bbox_ctr_y - (0.5 * all_bbox_height)
    x1 = all_bbox_ctr_x - (0.5 * all_bbox_width)
    y2 = all_bbox_height + y1
    x2 = all_bbox_width + x1
    #
    return torch.stack([y1, x1, y2, x2], dim=-1)


class Decoder(nn.Module):
    """
    Generating bounding boxes and labels from faster rcnn predictions.
    First calculating the boxes from predicted deltas and label probs.
    Then applied non max suppression and selecting top_n boxes by scores.
    """
    """
    inputs:
        roi_bboxes = (batch_size, roi_bbox_size, [y1, x1, y2, x2])
        pred_deltas = (batch_size, roi_bbox_size, total_labels * [delta_y, delta_x, delta_h, delta_w])
        pred_label_probs = (batch_size, roi_bbox_size, total_labels)
    outputs:
        pred_bboxes = (batch_size, top_n, [y1, x1, y2, x2])
        pred_labels = (batch_size, top_n)
            1 to total label number
        pred_scores = (batch_size, top_n)
    """

    def __init__(self, variances, total_labels, max_total_size=100, score_threshold=0.05, iou_threshold=0.5, **kwargs):
        """

        :param variances: bbox variances
        :param total_labels: number of classes
        :param max_total_size: max number of predictions
        :param score_threshold: score threshold
        :param iou_threshold: iou threshold
        :param kwargs: other args
        """
        super(Decoder, self).__init__(**kwargs)
        self.variances = variances
        self.total_labels = total_labels
        self.max_total_size = max_total_size
        self.score_threshold = score_threshold
        self.iou_threshold = iou_threshold

    def call(self, inputs):
        """
        Make final predictions from DAROD outputs
        :param inputs: DAROD outputs (roi boxes, deltas, and probas)
        :return: final predictions (boxes, classes, scores)
        """
        roi_bboxes, pred_deltas, pred_label_probs = inputs
        batch_size = pred_deltas.shape[0]
        #
        pred_deltas = pred_deltas.view(batch_size, -1, self.total_labels, 4)
        pred_deltas *= self.variances
        #
        expanded_roi_bboxes = roi_bboxes.unsqueeze(-2).expand(-1, -1, self.total_labels, -1)
        pred_bboxes = get_bboxes_from_deltas(expanded_roi_bboxes, pred_deltas)
        pred_bboxes = torch.clamp(pred_bboxes, min=0.0, max=1.0)
        #
        pred_labels_map = pred_label_probs.argmax(dim=-1, keepdim=True)
        pred_labels = torch.where(pred_labels_map != 0, pred_label_probs, torch.zeros_like(pred_label_probs))
        #
        final_bboxes, final_scores, final_labels, _ = non_max_suppression(
            pred_bboxes, pred_labels,
            iou_threshold=self.iou_threshold,
            max_output_size_per_class=self.max_total_size,
            max_total_size=self.max_total_size,
            score_threshold=self.score_threshold)
        #
        return final_bboxes.detach(), final_labels.detach(), final_scores.detach()
