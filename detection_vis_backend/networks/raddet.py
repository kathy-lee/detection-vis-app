import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger

from detection_vis_backend.train.utils import iou3d


def mish_activation(x):
    """ Mish Activation Function """
    return x * torch.tanh(F.softplus(x))  


class Convolution2D(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, activation, 
                 use_activation=True, use_bias=True, bn=True):
        """
        2D convolutional layer in this research 

        Args:
            in_channels: Input channels, int
            out_channels: Output channels, int
            kernel_size: Kernel size, int
            stride: Stride, int or tuple
            padding: Padding, either 'same' or 'valid'
            activation: Activation function, either 'relu' or 'mish'
        """
        super(Convolution2D, self).__init__()

        if padding == "same":
            padding = kernel_size // 2  # This provides "same" padding for odd kernel sizes
        elif padding == "valid":
            padding = 0
        else:
            raise ValueError("Expected padding to be 'same' or 'valid', but got {}".format(padding))

        # Regularization is typically done using weight decay in the optimizer in PyTorch
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=use_bias)
        self.bn = None
        if bn:
            self.bn = nn.BatchNorm2d(out_channels)

        self.use_activation = use_activation
        self.activation = activation

    def forward(self, x):
        x = self.conv(x)
        
        if self.bn:
            x = self.bn(x)
            
        if self.use_activation:
            if self.activation == "relu":
                x = F.relu(x)
            elif self.activation == "mish":
                x = mish_activation(x)
            else:
                raise ValueError("Unsupported activation function")       
        return x


class MaxPooling2D(nn.Module):
    def __init__(self, pool_size=(2,2), strides=(2,2), padding="same"):
        """
        Max pooling layer for 2D tensors
        
        Args:
            pool_size: Size of the max pooling window, tuple
            strides: Stride of the max pooling window, tuple
            padding: Padding, either 'same' or 'valid'
        """
        super(MaxPooling2D, self).__init__()

        # Adjusting padding to be compatible with 'same' and 'valid' padding modes from TensorFlow
        if padding == 'same':
            self.padding = ((pool_size[0]-1)//2, (pool_size[1]-1)//2)
        else:  # 'valid'
            self.padding = (0, 0)

        self.pool_size = pool_size
        self.strides = strides

    def forward(self, x):
        return F.max_pool2d(x, kernel_size=self.pool_size, stride=self.strides, padding=self.padding)
    

class BasicResidualBlock(nn.Module):
    def __init__(self, input_channels, channel_expansion, strides=(1, 1), use_bias=False):
        super(BasicResidualBlock, self).__init__()

        self.channel_expansion = channel_expansion
        self.strides = strides
        self.use_bias = use_bias

        # First convolution
        self.conv1 = Convolution2D(input_channels, input_channels, 3, strides, padding='same',
                                   activation='relu', use_bias=use_bias, bn=True)
        
        # Second convolution
        self.conv2 = Convolution2D(input_channels, int(input_channels * channel_expansion), 3, (1, 1), 
                                   padding='same', activation='relu', use_bias=use_bias, bn=True)
        
        # Third convolution
        self.conv3 = Convolution2D(int(input_channels * channel_expansion), int(input_channels * channel_expansion),
                                   1, (1, 1), padding='same', activation='relu', use_bias=use_bias, bn=True)
        
        # Shortcut
        self.shortcut = Convolution2D(input_channels, int(input_channels * channel_expansion), 3, 
                                      strides, padding='same', activation='relu', use_bias=use_bias, bn=True)

    def forward(self, x):
        out1 = self.conv1(x)
        logger.debug(f"1st conv passed. input: {x.shape}, output: {out1.shape}")
        out2 = self.conv2(out1)
        logger.debug(f"2nd conv passed. output:{out2.shape}")
        out3 = self.conv3(out2)
        logger.debug(f"3rd conv passed. output:{out3.shape}")

        # Decide which path to take for the shortcut
        if any(val != 1 for val in self.strides) or self.channel_expansion != 1:
            shortcut_out = self.shortcut(x)
        else:
            shortcut_out = x

        # Combining the outputs
        out = out3 + shortcut_out
        return out


class RepeatBlock(nn.Module):
    def __init__(self, input_channels, repeat_times, all_strides=None, all_expansions=None, feature_maps_downsample=False):
        super(RepeatBlock, self).__init__()
        
        # Ensure correct parameter formats and values
        if all_strides is not None and all_expansions is not None:
            assert (isinstance(all_strides, (tuple, list)))
            assert (isinstance(all_expansions, (tuple, list)))
            assert len(all_strides) == repeat_times
            assert len(all_expansions) == repeat_times
        elif all_strides is None and all_expansions is None:
            all_strides, all_expansions = [], []
            for i in range(repeat_times):
                all_strides.append(1)
                all_expansions.append(0.5 if i % 2 == 0 else 2)

        # Build the repeated blocks
        blocks = []
        for i in range(repeat_times):
            strides = (all_strides[i], all_strides[i])
            expansion = all_expansions[i]
            blocks.append(BasicResidualBlock(input_channels, expansion, strides, use_bias=True))
            input_channels = int(input_channels * expansion)
        self.blocks = nn.Sequential(*blocks)
        
        # Optional MaxPooling2D layer
        self.feature_maps_downsample = feature_maps_downsample
        if feature_maps_downsample:
            self.maxpool = nn.MaxPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
        x = self.blocks(x)
        logger.debug(f"Blocks module passed with output shape: {x.shape}")
        if self.feature_maps_downsample:
            x = self.maxpool(x)
        return x
    

class RadarResNet3D(nn.Module):
    def __init__(self, input_channels):
        super(RadarResNet3D, self).__init__()

        # Configuration parameters
        self.block_repeat_times = [2, 4, 8, 16]
        self.channels_upsample = [False, False, True, True]
        self.feature_mp_downsample = [True, True, True, True]
        
        # Create the repeated blocks based on the configuration
        self.blocks = nn.ModuleList()
        for i, repeat_times in enumerate(self.block_repeat_times):
            if repeat_times != 1:
                all_strides = [1, 1] * int(repeat_times/2)
                all_expansions = [1, 1] * int(repeat_times/2)
            else:
                all_strides = [1] * int(repeat_times)
                all_expansions = [1] * int(repeat_times)
            if self.channels_upsample[i]:
                all_expansions[-1] *= 2

            feature_maps_downsample = self.feature_mp_downsample[i]
            block = RepeatBlock(input_channels, repeat_times, all_strides, all_expansions, feature_maps_downsample)
            input_channels = int(input_channels * all_expansions[-1])
            self.blocks.append(block)

    def forward(self, x):
        feature_stages = []
        for i, block in enumerate(self.blocks):
            x = block(x)
            if i > len(self.block_repeat_times) - 4:
                feature_stages.append(x)
            logger.debug(f"block {i} gives output shape : {x.shape}")

        # NOTE: since we are doing one-level output, only last level is used
        features = feature_stages[-1]
        logger.debug(f"RadarResNet3D gives output shape: {features.shape}")
        return features
    


class YoloHead(nn.Module):
    def __init__(self, num_anchors_layer, num_class, input_channel, last_channel):
        super(YoloHead, self).__init__()

        self.num_anchors_layer = num_anchors_layer
        self.num_class = num_class
        self.last_channel = last_channel   

        # Calculate final output channels
        self.final_output_channels = self.last_channel * self.num_anchors_layer * (self.num_class + 7)

        # Convolution Layers
        # First Conv layer
        self.conv1 = Convolution2D(input_channel, input_channel * 2, 
                                   kernel_size=3, stride=(1,1), padding='same', 
                                   activation='relu', use_activation=True, use_bias=True, bn=True)
        
        # Second Conv layer
        self.conv2 = Convolution2D(input_channel * 2, self.final_output_channels, 
                                   kernel_size=1, stride=(1,1), padding='same', 
                                   activation=None, use_activation=False, use_bias=True, bn=False)

    def forward(self, feature_map):
        # Get the required shape
        batch_size = feature_map.size(0)
        height = feature_map.size(2)
        width = feature_map.size(3)
        final_output_reshape = (batch_size, self.last_channel, self.num_anchors_layer * (self.num_class + 7), height, width)
        
        # First convolution
        conv = self.conv1(feature_map)
        logger.debug(f"The 1st conv in YoloHead gives output with shape: {conv.shape}")
        # Second convolution
        conv = self.conv2(conv)
        logger.debug(f"The 2nd conv in YoloHead gives output with shape: {conv.shape}")
        # Reshape operation
        output = conv.view(final_output_reshape)
        logger.debug(f"YoloHead final gives output with shape: {output.shape}")
        return output


def boxDecoder(yolohead_output, input_size, anchors_layer, num_class, scale=1., device='cuda:0'):
    """ Decoder output from yolo head to boxes """ 

    grid_size = yolohead_output.shape[1:4]
    num_anchors_layer = len(anchors_layer)
    grid_strides = torch.tensor(input_size, dtype=torch.float32).to(device) / torch.tensor(grid_size, dtype=torch.float32).to(device)
    reshape_size = [yolohead_output.shape[0]] + list(grid_size) + [num_anchors_layer, 7+num_class]
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


def focalLoss(raw_conf, pred_conf, gt_conf, pred_box, raw_boxes, input_size, iou_loss_threshold=0.5):
    """ Calculate focal loss for objectness """
    iou = iou3d(pred_box.unsqueeze(-2), raw_boxes.unsqueeze(1).unsqueeze(1).unsqueeze(1).unsqueeze(1), input_size)
    max_iou, _ = iou.max(dim=-1)
    max_iou = max_iou.unsqueeze(-1)

    gt_conf_negative = (1.0 - gt_conf) * (max_iou < iou_loss_threshold).float()
    conf_focal = (gt_conf - pred_conf).pow(2)
    alpha = 0.01

    focal_loss = conf_focal * (gt_conf * F.binary_cross_entropy_with_logits(raw_conf, gt_conf, reduction='none')
        + alpha * gt_conf_negative * F.binary_cross_entropy_with_logits(raw_conf, gt_conf, reduction='none'))
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