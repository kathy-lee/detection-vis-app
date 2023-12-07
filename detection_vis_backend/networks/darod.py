import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

from torchvision.ops import nms
from torch.nn import GroupNorm

from detection_vis_backend.train.utils import custom_one_hot


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
    def __init__(self, anchors, pre_nms_topn_train, pre_nms_topn_test, post_nms_topn_train, post_nms_topn_test, rpn_nms_iou, rpn_variances):
        super(RoIBBox, self).__init__()
        self.pre_nms_topn_train = pre_nms_topn_train
        self.pre_nms_topn_test = pre_nms_topn_test
        self.post_nms_topn_train = post_nms_topn_train
        self.post_nms_topn_test = post_nms_topn_test
        self.rpn_nms_iou = rpn_nms_iou
        self.rpn_variances = rpn_variances

        self.anchors = anchors.clone().detach()

    def forward(self, rpn_bbox_deltas, rpn_labels):
        anchors = self.anchors
        pre_nms_topn = self.pre_nms_topn_train if self.training else self.pre_nms_topn_test
        post_nms_topn = self.post_nms_topn_train if self.training else self.post_nms_topn_test
        nms_iou_threshold = self.rpn_nms_iou
        total_anchors = anchors.shape[0]
        batch_size = rpn_bbox_deltas.shape[0]
        rpn_bbox_deltas = rpn_bbox_deltas.view(batch_size, total_anchors, 4)
        
        # Convert softmax in PyTorch
        rpn_labels = F.softmax(rpn_labels, dim=-1)
        rpn_labels = rpn_labels.view(batch_size, total_anchors)
        variances = torch.tensor(self.rpn_variances).to(rpn_bbox_deltas.device)
        rpn_bbox_deltas *= variances.unsqueeze(0).unsqueeze(1)
        # Custom utility function
        rpn_bboxes = get_bboxes_from_deltas(anchors.to(rpn_bbox_deltas.device), rpn_bbox_deltas)
        # Use PyTorch's topk() for this functionality
        _, pre_indices = torch.topk(rpn_labels, pre_nms_topn, dim=1)

        # Use PyTorch's gather() for this functionality
        pre_roi_bboxes = torch.gather(rpn_bboxes, 1, pre_indices.unsqueeze(-1).expand(-1, -1, 4))
        pre_roi_labels = torch.gather(rpn_labels, 1, pre_indices)

        # Reshaping using view in PyTorch
        pre_roi_bboxes = pre_roi_bboxes.view(batch_size, pre_nms_topn, 1, 4)
        pre_roi_labels = pre_roi_labels.view(batch_size, pre_nms_topn, 1)
        # Assuming the custom utility function is adapted for PyTorch
        #print(f"RoIBBox before nms: {pre_roi_bboxes.shape}, {pre_roi_labels.shape}")
        roi_bboxes, roi_scores, _, _ = non_max_suppression(pre_roi_bboxes, pre_roi_labels,
                                                            max_output_size_per_class=post_nms_topn, max_total_size=post_nms_topn,
                                                            iou_threshold=nms_iou_threshold)
        #print(f"RoIBBox after nms: {roi_bboxes.shape}, {roi_scores.shape}")
        return roi_bboxes.detach(), roi_scores.detach()


class RadarFeatures(nn.Module):
    """
    Extracting radar feature from RPN proposed boxes.
    This layer extracts range and Doppler values from RPN proposed
    boxes which have scores > 0.5. Otherwise, range and Doppler values
    are set to -1
    """
    def __init__(self, input_size, range_res, doppler_res):
        super(RadarFeatures, self).__init__()
        self.input_size = input_size
        self.range_res = range_res
        self.doppler_res = doppler_res

    def forward(self, roi_bboxes, roi_scores):
        """
        :param roi_bboxes: Bounding boxes from RPN
        :param roi_scores: Scores of the bounding boxes
        :return: radar_features
        """

        # Denormalize bounding boxes
        roi_bboxes = self.denormalize_bboxes(roi_bboxes, 
                                             height=self.input_size[0],
                                             width=self.input_size[1])

        # Get centers of roi boxes
        h = roi_bboxes[..., 2] - roi_bboxes[..., 0]
        w = roi_bboxes[..., 3] - roi_bboxes[..., 1]

        ctr_y = roi_bboxes[..., 0] + h / 2
        ctr_x = roi_bboxes[..., 1] + w / 2

        # Get radar feature
        range_values = self.range_res * ctr_y.float()
        doppler_values = torch.abs(32 - (self.doppler_res * ctr_x.float()))
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


class RoIPooling(torch.nn.Module):
    """
    Reducing all feature maps to the same size.
    Firstly cropping bounding boxes from the feature maps and then resizing it to the pooling size.
    """
    def __init__(self, pooling_size):
        super(RoIPooling, self).__init__()
        self.pooling_size = pooling_size
        self.roi_pool = torchvision.ops.RoIPool(output_size=self.pooling_size, spatial_scale=1.0)

    def forward(self, feature_map, roi_bboxes):
        """
        :param feature_map: feature map
        :param roi_bboxes: roi boxes
        :return: pooled features
        """
        batch_size, total_bboxes = roi_bboxes.shape[0], roi_bboxes.shape[1]
        
        # We need to arrange bbox indices for each batch
        device = roi_bboxes.device
        pooling_bbox_indices = torch.arange(batch_size, device=device).unsqueeze(1).repeat(1, total_bboxes)
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
    device = bboxes.device
    gt_boxes = gt_boxes.to(device)

    bbox_width = bboxes[..., 3] - bboxes[..., 1]
    bbox_height = bboxes[..., 2] - bboxes[..., 0]
    bbox_ctr_x = bboxes[..., 1] + 0.5 * bbox_width
    bbox_ctr_y = bboxes[..., 0] + 0.5 * bbox_height
    
    gt_width = gt_boxes[..., 3] - gt_boxes[..., 1]
    gt_height = gt_boxes[..., 2] - gt_boxes[..., 0]
    gt_ctr_x = gt_boxes[..., 1] + 0.5 * gt_width
    gt_ctr_y = gt_boxes[..., 0] + 0.5 * gt_height

    bbox_width = torch.where(bbox_width == 0, torch.tensor(1e-3).to(device), bbox_width)
    bbox_height = torch.where(bbox_height == 0, torch.tensor(1e-3).to(device), bbox_height)

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
    device = mask.device
    select_xyz = select_xyz.to(device)
    random_mask = torch.randint(low=1, high=maxval, size=mask.shape, dtype=torch.int32, device=device)
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
    gt_area = (gt_y2 - gt_y1).squeeze(-1) * (gt_x2 - gt_x1).squeeze(-1)
    bbox_area = (bbox_y2 - bbox_y1).squeeze(-1) * (bbox_x2 - bbox_x1).squeeze(-1)
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


def non_max_suppression(boxes, scores, max_output_size_per_class, max_total_size, iou_threshold, score_threshold=float('-inf')):
    """
    Applying non maximum suppression.
    Details could be found on tensorflow documentation.
    https://www.tensorflow.org/api_docs/python/tf/image/combined_non_max_suppression

    :param pred_bboxes: (batch_size, total_bboxes, total_labels, [y1, x1, y2, x2]), total_labels should be 1 for binary operations like in rpn
    :param pred_labels: (batch_size, total_bboxes, total_labels)
    :param kwargs: other parameters
    :return: nms_boxes = (batch_size, max_detections, [y1, x1, y2, x2])
            nmsed_scores = (batch_size, max_detections)
            nmsed_classes = (batch_size, max_detections)
            valid_detections = (batch_size)
                Only the top valid_detections[i] entries in nms_boxes[i], nms_scores[i] and nms_class[i] are valid.
                The rest of the entries are zero paddings.
    """
    batch_size, num_boxes, num_classes = scores.shape
    
    all_nms_boxes = []
    all_nms_scores = []
    all_nms_classes = []
    all_valid_detections = []
    
    for b in range(batch_size):
        batch_boxes = []
        batch_scores = []
        batch_classes = []
        
        for c in range(num_classes):
            class_scores = scores[b, :, c]
            valid_inds = class_scores > score_threshold
            class_scores = class_scores[valid_inds]
            class_boxes = boxes[b, valid_inds, c, :]
            
            # Sort class_scores and keep top max_output_size_per_class
            sorted_scores, sorted_indices = class_scores.sort(descending=True)
            sorted_scores = sorted_scores[:max_output_size_per_class]
            sorted_boxes = class_boxes[sorted_indices][:max_output_size_per_class]
            
            # Apply NMS
            keep_indices = nms(sorted_boxes, sorted_scores, iou_threshold)
            keep_boxes = sorted_boxes[keep_indices]
            keep_scores = sorted_scores[keep_indices]
            
            batch_boxes.append(keep_boxes)
            batch_scores.append(keep_scores)
            batch_classes.append(torch.full((len(keep_scores),), c, dtype=torch.float32))
        
        # Concatenate results of all classes
        batch_boxes = torch.cat(batch_boxes, dim=0)
        batch_scores = torch.cat(batch_scores, dim=0)
        batch_classes = torch.cat(batch_classes, dim=0)
        
        # Keep only top `max_total_size` detections
        if len(batch_scores) > max_total_size:
            _, top_indices = batch_scores.sort(descending=True)
            top_indices = top_indices[:max_total_size]
            batch_boxes = batch_boxes[top_indices]
            batch_scores = batch_scores[top_indices]
            batch_classes = batch_classes[top_indices]
        
        all_nms_boxes.append(batch_boxes)
        all_nms_scores.append(batch_scores)
        all_nms_classes.append(batch_classes)
        all_valid_detections.append(len(batch_boxes))

    max_len = max([b.shape[0] for b in all_nms_boxes])
    
    device = all_nms_boxes[0].device
    all_nms_classes = [c.to(device) for c in all_nms_classes]
    padded_nms_boxes = [torch.cat([b, torch.zeros((max_len - b.shape[0], 4), device=device)], dim=0) for b in all_nms_boxes]
    padded_nms_scores = [torch.cat([s, torch.zeros(max_len - s.shape[0], device=device)], dim=0) for s in all_nms_scores]
    padded_nms_classes = [torch.cat([c, torch.zeros(max_len - c.shape[0], device=device)], dim=0) for c in all_nms_classes]
    # Stack these padded tensors
    all_nms_boxes = torch.stack(padded_nms_boxes)
    all_nms_scores = torch.stack(padded_nms_scores)
    all_nms_classes = torch.stack(padded_nms_classes)
    all_valid_detections = torch.tensor(all_valid_detections, device=device)
    return all_nms_boxes, all_nms_scores, all_nms_classes, all_valid_detections


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

    def forward(self, inputs):
        """
        Make final predictions from DAROD outputs
        :param inputs: DAROD outputs (roi boxes, deltas, and probas)
        :return: final predictions (boxes, classes, scores)
        """
        roi_bboxes, pred_deltas, pred_label_probs = inputs
        batch_size = pred_deltas.shape[0]
        #
        pred_deltas = pred_deltas.view(batch_size, -1, self.total_labels, 4)
        variances = torch.tensor(self.variances).to(pred_deltas.device)
        pred_deltas *= variances.unsqueeze(0).unsqueeze(1)
        #
        expanded_roi_bboxes = roi_bboxes.unsqueeze(-2).expand(-1, -1, self.total_labels, -1)
        pred_bboxes = get_bboxes_from_deltas(expanded_roi_bboxes, pred_deltas)
        pred_bboxes = torch.clamp(pred_bboxes, min=0.0, max=1.0)
        #
        pred_labels_map = pred_label_probs.argmax(dim=-1, keepdim=True)
        pred_labels = torch.where(pred_labels_map != 0, pred_label_probs, torch.zeros_like(pred_label_probs))
        #
        #print(f"Decoder before nms: {pred_bboxes.shape}, {pred_labels.shape}")
        #print(self.iou_threshold, self.max_total_size, self.score_threshold)
        final_bboxes, final_scores, final_labels, _ = non_max_suppression(
            pred_bboxes, pred_labels,
            iou_threshold=self.iou_threshold,
            max_output_size_per_class=self.max_total_size,
            max_total_size=self.max_total_size,
            score_threshold=self.score_threshold)
        #print(f"Decoder after nms: {final_bboxes.shape}, {final_scores.shape}")
        #
        return final_bboxes.detach(), final_labels.detach(), final_scores.detach()


def roi_delta(roi_bboxes, gt_boxes, gt_labels, n_class, fastrcnn_cfg, seed):
    total_labels = n_class
    total_pos_bboxes = int(fastrcnn_cfg["frcnn_boxes"] / 3)
    total_neg_bboxes = int(fastrcnn_cfg["frcnn_boxes"] * 2 / 3)
    device = roi_bboxes.device
    variances = torch.tensor(fastrcnn_cfg["variances_boxes"]).to(device)
    adaptive_ratio = fastrcnn_cfg["adaptive_ratio"]
    positive_th = fastrcnn_cfg["positive_th"]

    batch_size, total_bboxes = roi_bboxes.size(0), roi_bboxes.size(1)
    # Calculate iou values between each bbox and ground truth boxes
    iou_map = generate_iou_map(roi_bboxes, gt_boxes)
    max_indices_each_gt_box = torch.argmax(iou_map, dim=2)
    merged_iou_map, _ = torch.max(iou_map, dim=2)
    pos_mask = merged_iou_map > positive_th
    pos_mask = randomly_select_xyz_mask(pos_mask, torch.tensor([total_pos_bboxes]), seed)
    neg_mask = (merged_iou_map < positive_th) & (merged_iou_map >= 0.0)
    if adaptive_ratio:
        pos_count = torch.sum(pos_mask, dim=1)
        total_neg_bboxes = (pos_count + 1) * 3
        neg_mask = randomly_select_xyz_mask(neg_mask, total_neg_bboxes, seed)
    else:
        neg_mask = randomly_select_xyz_mask(neg_mask, torch.tensor([total_neg_bboxes]), seed)
    gt_boxes_map = torch.gather(gt_boxes, 1, max_indices_each_gt_box.unsqueeze(-1).expand(-1, -1, 4)).squeeze()
    expanded_gt_boxes = torch.where(pos_mask.unsqueeze(-1), gt_boxes_map, torch.zeros_like(gt_boxes_map))
    gt_labels_map = torch.gather(gt_labels, 1, max_indices_each_gt_box).squeeze()
    pos_gt_labels = torch.where(pos_mask, gt_labels_map, torch.tensor(-1, dtype=torch.int32))
    neg_gt_labels = neg_mask.int()
    expanded_gt_labels = pos_gt_labels + neg_gt_labels
    roi_bbox_deltas = get_deltas_from_bboxes(roi_bboxes, expanded_gt_boxes) / variances
    expanded_gt_labels = expanded_gt_labels.long()
    roi_bbox_labels = custom_one_hot(expanded_gt_labels, num_classes=total_labels)
    scatter_indices = roi_bbox_labels.unsqueeze(-1).repeat(1, 1, 1, 4)
    roi_bbox_deltas = scatter_indices * roi_bbox_deltas.unsqueeze(-2)
    roi_bbox_deltas = roi_bbox_deltas.reshape(batch_size, total_bboxes * total_labels, 4)

    if fastrcnn_cfg["reg_loss"] == "sl1":
        return roi_bbox_deltas.detach(), roi_bbox_labels.detach()
    elif fastrcnn_cfg["reg_loss"] == "giou":
        expanded_roi_boxes = scatter_indices * roi_bboxes.unsqueeze(-2)
        expanded_roi_boxes = expanded_roi_boxes.reshape(batch_size, total_bboxes * total_labels, 4)

        expanded_gt_boxes = scatter_indices * expanded_gt_boxes.unsqueeze(-2)
        expanded_gt_boxes = expanded_gt_boxes.reshape(batch_size, total_bboxes * total_labels, 4)

        return expanded_roi_boxes.detach(), expanded_gt_boxes.detach(), roi_bbox_labels.detach()
    else:
        raise ValueError("Unsupported loss type.")


def calculate_rpn_actual_outputs(anchors, gt_boxes, gt_labels, rpn_cfg, feature_map_shape, seed):
    """
    Generating one step data for training or inference. Batch operations supported.
    :param anchors: (total_anchors, [y1, x1, y2, x2]) these values in normalized format between [0, 1]
    :param gt_boxes: (batch_size, gt_box_size, [y1, x1, y2, x2]) these values in normalized format between [0, 1]
    :param gt_labels: (batch_size, gt_box_size)
    :param config: dictionary
    :return: bbox_deltas = (batch_size, total_anchors, [delta_y, delta_x, delta_h, delta_w])
             bbox_labels = (batch_size, feature_map_shape, feature_map_shape, anchor_count)
    """
    
    batch_size = gt_boxes.size(0)
    device = gt_boxes.device
    anchor_count = rpn_cfg["anchor_count"]
    total_pos_bboxes = int(rpn_cfg["rpn_boxes"] / 2)
    total_neg_bboxes = int(rpn_cfg["rpn_boxes"] / 2)
    variances = torch.tensor(rpn_cfg["variances"]).to(device)
    adaptive_ratio = rpn_cfg["adaptive_ratio"]
    postive_th = rpn_cfg["positive_th"]
    output_height, output_width = feature_map_shape
    anchors = anchors.to(device)

    iou_map = generate_iou_map(anchors, gt_boxes) 
    max_indices_each_row = torch.argmax(iou_map, dim=2)
    max_indices_each_column = torch.argmax(iou_map, dim=1)
    merged_iou_map = torch.max(iou_map, dim=2).values
    pos_mask = merged_iou_map > postive_th
    valid_indices_cond = gt_labels != -1
    valid_indices = torch.nonzero(valid_indices_cond).int()
    valid_max_indices = max_indices_each_column[valid_indices_cond]
    scatter_bbox_indices = torch.stack([valid_indices[..., 0], valid_max_indices], dim=1)

    # work only on CPU
    #max_pos_mask = torch.zeros_like(pos_mask)
    #max_pos_mask[scatter_bbox_indices.t().numpy()] = torch.ones((len(valid_indices),), dtype=torch.bool) 
    # work also on GPU 
    max_pos_mask = torch.zeros_like(pos_mask).to('cpu')
    scatter_bbox_indices = scatter_bbox_indices.to('cpu')
    valid_indices = valid_indices.to('cpu')
    max_pos_mask[scatter_bbox_indices.t().numpy()] = torch.ones((len(valid_indices),), dtype=torch.bool)
    max_pos_mask = max_pos_mask.to(device)

    pos_mask = (pos_mask | max_pos_mask) & (torch.sum(anchors, dim=-1) != 0.0)
    pos_mask = randomly_select_xyz_mask(pos_mask, torch.tensor([total_pos_bboxes]), seed)  
    pos_count = torch.sum(pos_mask.long(), dim=-1)
    # Keep a 50%/50% ratio of positive/negative samples
    if adaptive_ratio:
        neg_count = 2 * pos_count
    else:
        neg_count = (total_pos_bboxes + total_neg_bboxes) - pos_count
    neg_mask = ((merged_iou_map < 0.3) & (~pos_mask)) & (torch.sum(anchors, dim=-1) != 0.0)
    neg_mask = randomly_select_xyz_mask(neg_mask, neg_count, seed)  
    pos_labels = torch.where(pos_mask, torch.ones_like(pos_mask, dtype=torch.float32), torch.tensor(-1.0, dtype=torch.float32))
    neg_labels = neg_mask.float()
    bbox_labels = pos_labels + neg_labels
    gt_boxes_map = torch.gather(gt_boxes, 1, max_indices_each_row.unsqueeze(-1).expand(-1,-1,4))
    # Replace negative bboxes with zeros
    expanded_gt_boxes = torch.where(pos_mask.unsqueeze(-1), gt_boxes_map, torch.zeros_like(gt_boxes_map))
    bbox_deltas = get_deltas_from_bboxes(anchors, expanded_gt_boxes) / variances  
    bbox_labels = bbox_labels.view(batch_size, output_height, output_width, anchor_count)
    return bbox_deltas, bbox_labels


def darod_loss(network_output, bbox_labels, bbox_deltas, frcnn_reg_actuals, frcnn_cls_actuals):
    """
    Calculate loss function for DAROD model
    :param rpn_cls_pred: RPN classification pred
    :param rpn_delta_pred: RPN regression pred
    :param frcnn_cls_pred: FRCNN classification pred
    :param frcnn_reg_pred: FRCNN regression pred
    :param bbox_labels: bounding boxes labels
    :param bbox_deltas: bounding boxes deltas
    :param frcnn_reg_actuals: faster rcnn regression labels
    :param frcnn_cls_actuals: faster rcnn classification labels
    :return: faster rcnn loss
    """
    rpn_regression_loss = reg_loss(bbox_deltas, network_output["rpn_delta_pred"])
    rpn_classif_loss = rpn_cls_loss(bbox_labels, network_output["rpn_cls_pred"])
    frcnn_regression_loss = reg_loss(frcnn_reg_actuals, network_output["frcnn_reg_pred"])
    frcnn_classif_loss = frcnn_cls_loss(frcnn_cls_actuals, network_output["frcnn_cls_pred"])
    return rpn_regression_loss, rpn_classif_loss, frcnn_regression_loss, frcnn_classif_loss


def reg_loss(*args):
    """
    Calculating rpn / faster rcnn regression loss value.
    :param args: could be (y_true, y_pred) or ((y_true, y_pred), )
    :return: regression loss val
    """
    y_true, y_pred = args if len(args) == 2 else args[0]
    y_pred = y_pred.reshape(y_pred.shape[0], -1, 4)
    # Huber loss
    loss_for_all = F.smooth_l1_loss(y_true, y_pred, reduction='none', beta=1/9).sum(-1)
    # loss_for_all = torch.sum(loss_for_all, dim=-1)
    
    # Check for any non-zero entries in the last dimension to create a mask
    pos_cond = torch.any(y_true != 0.0, dim=-1)
    pos_mask = pos_cond.float()
    loc_loss = torch.sum(pos_mask * loss_for_all)
    total_pos_bboxes = max(1.0, pos_mask.sum().item())
    return loc_loss / total_pos_bboxes


def rpn_cls_loss(*args):
    """
    Calculating rpn class loss value.
    :param args: could be (y_true, y_pred) or ((y_true, y_pred), )
    :return: CE loss
    """
    y_true, y_pred = args if len(args) == 2 else args[0]
    # Find indices where y_true is not equal to -1
    indices = (y_true != -1).nonzero(as_tuple=True)
    
    target = y_true[indices]
    output = y_pred[indices]
    
    # Use Binary Cross Entropy with Logits as the loss function
    lf = nn.BCEWithLogitsLoss()
    return lf(output, target)


def frcnn_cls_loss(*args):
    """
    Calculating faster rcnn class loss value.
    :param args: could be (y_true, y_pred) or ((y_true, y_pred), )
    :return: CE loss
    """
    y_true, y_pred = args if len(args) == 2 else args[0]
    
    y_pred = y_pred.permute(0, 2, 1).contiguous() # Shape to (batch_size, num_of_class, N)
    
    loss_for_all = F.cross_entropy(y_pred, torch.argmax(y_true, dim=-1), reduction='none')

    cond = torch.any(y_true != 0.0, dim=-1).float()
    conf_loss = torch.sum(cond * loss_for_all)
    total_boxes = torch.maximum(torch.tensor(1.0).to(y_true.device), torch.sum(cond))

    return conf_loss / total_boxes