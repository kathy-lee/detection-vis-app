"""Sample adapters for RADDetDataset: how RADDet and DAROD each read the RAD cube.

Moved from RADDetDataset.__getitem__ / encodeToLabels / transform. The parser keeps
get_RAD, _get_collapsed (and its content-addressed cache), get_image, annotation and
data_stats; everything model-shaped is here.
"""
import numpy as np
from scipy.ndimage import zoom

from detection_vis_backend.datasets.adapters import SampleAdapter, register
from detection_vis_backend.datasets.utils import (
    complexTo2Channels, smoothOnehot, iou3d, flip_vertical, flip_horizontal,
)
from detection_vis_backend.networks.transrad import xyzwhd2xyzxyz, xyzxyz2xyzwhd


def random_flip(frame, gt_boxes, transformations):
    """Random flip augmentation, matching the reference's data_preprocessing
    (colindecourt/darod, darod/utils/data_utils.py): one uniform draw per sample,
    < 0.25 -> horizontal, < 0.5 -> vertical, otherwise unchanged. Never both.

    History: this used to apply EVERY entry of transformations to EVERY sample, on
    train, val AND test -- a deterministic 180 degree rotation of the whole dataset
    rather than augmentation. Both 75-epoch seeds overfit accordingly (val loss
    0.35 -> 0.47). The permanent mirroring also fed RadarFeatures wrong physics: a
    target at range r was reported to the classifier as R - r.
    """
    if not transformations:
        return frame, gt_boxes
    r = np.random.uniform(0, 1)
    if r < 0.25 and 'hflip' in transformations:
        frame, gt_boxes = flip_horizontal(frame, gt_boxes)
    elif r < 0.5 and 'vflip' in transformations:
        frame, gt_boxes = flip_vertical(frame, gt_boxes)
    return frame, gt_boxes


def encodeToLabels(gt_instances, data_stats, anchor_boxes, headoutput_shape, grid_strides, input_shape):
    """ Transfer ground truth instances into Detection Head format """
    raw_boxes_xyzwhd = np.zeros((data_stats["max_boxes_per_frame"], 7))
    gt_labels = np.zeros(list(headoutput_shape[1:4]) + \
                    [len(anchor_boxes)] + \
                    [len(data_stats["all_classes"]) + 7]) # (16, 16, 4, 6, 13)

    ### start transferring box to ground truth label format ###
    for i in range(len(gt_instances["classes"])):
        if i > data_stats["max_boxes_per_frame"]:
            continue
        class_name = gt_instances["classes"][i]
        box_xyzwhd = gt_instances["boxes"][i]
        class_id = data_stats["all_classes"].index(class_name)
        if i < data_stats["max_boxes_per_frame"]:
            raw_boxes_xyzwhd[i, :6] = box_xyzwhd
            raw_boxes_xyzwhd[i, 6] = class_id
        class_onehot = smoothOnehot(class_id, len(data_stats["all_classes"]))

        exist_positive = False

        grid_strid = grid_strides
        anchor_stage = anchor_boxes
        box_xyzwhd_scaled = box_xyzwhd[np.newaxis, :].astype(np.float32)
        box_xyzwhd_scaled[:, :3] /= grid_strid
        anchorstage_xyzwhd = np.zeros([len(anchor_stage), 6])
        anchorstage_xyzwhd[:, :3] = np.floor(box_xyzwhd_scaled[:, :3]) + 0.5
        anchorstage_xyzwhd[:, 3:] = anchor_stage.astype(np.float32)

        iou_scaled = iou3d(box_xyzwhd_scaled, anchorstage_xyzwhd, \
                                    input_shape)
        ### NOTE: 0.3 is from YOLOv4, maybe this should be different here ###
        ### it means, as long as iou is over 0.3 with an anchor, the anchor
        ### should be taken into consideration as a ground truth label
        iou_mask = iou_scaled > 0.3

        if np.any(iou_mask):
            xind, yind, zind = np.floor(np.squeeze(box_xyzwhd_scaled)[:3]).\
                                astype(np.int32)
            ### TODO: consider changing the box to raw yolohead output format ###
            gt_labels[xind, yind, zind, iou_mask, 0:6] = box_xyzwhd
            gt_labels[xind, yind, zind, iou_mask, 6:7] = 1.
            gt_labels[xind, yind, zind, iou_mask, 7:] = class_onehot
            exist_positive = True

        if not exist_positive:
            ### NOTE: this is the normal one ###
            ### it means take the anchor box with maximum iou to the raw
            ### box as the ground truth label
            anchor_ind = np.argmax(iou_scaled)
            xind, yind, zind = np.floor(np.squeeze(box_xyzwhd_scaled)[:3]).\
                                astype(np.int32)
            gt_labels[xind, yind, zind, anchor_ind, 0:6] = box_xyzwhd
            gt_labels[xind, yind, zind, anchor_ind, 6:7] = 1.
            gt_labels[xind, yind, zind, anchor_ind, 7:] = class_onehot

    has_label = False
    for label_stage in gt_labels:
        if label_stage.max() != 0:
            has_label = True
    gt_labels = np.array([np.where(gt_i == 0, 1e-16, gt_i) for gt_i in gt_labels])
    return gt_labels, has_label, raw_boxes_xyzwhd


@register
class RADDetOnRADDet(SampleAdapter):
    """Full RAD cube, global log-normalization, YOLO-encoded 3D labels."""
    MODELS, DATASET = ("RADDet",), "RADDetDataset"

    def __init__(self, parser, features, model_cfg, train_cfg, splittype=None):
        super().__init__(parser, features, model_cfg, train_cfg, splittype)
        self.data_stats = parser.data_stats
        if 'input_size' in model_cfg:
            self.input_shape = model_cfg["input_size"]
        elif 'input_size' in train_cfg:
            self.input_shape = train_cfg["input_size"]
        self.anchor_boxes = np.array(model_cfg["anchor_boxes"])
        self.headoutput_shape = [3, 16, 16, 4, 78]
        self.grid_strides = np.array(self.input_shape[:3]) / np.array(self.headoutput_shape[1:4])

    def __getitem__(self, index):
        gt_instances = self.parser.annotation(index)
        RAD_complex = self.parser.get_RAD(index)
        # Gloabl Normalization
        RAD_log = complexTo2Channels(RAD_complex)
        RAD_data = (RAD_log - self.data_stats["global_mean_log"]) / self.data_stats["global_variance_log"]
        # decode ground truth boxes to YOLO format
        gt_labels, has_label, gt_boxes = encodeToLabels(
            gt_instances, self.data_stats, self.anchor_boxes,
            self.headoutput_shape, self.grid_strides, self.input_shape)
        feature_data = RAD_data
        feature_data = np.transpose(feature_data, (2, 0, 1))
        return {"RAD": feature_data, 'label': gt_labels, 'boxes': gt_boxes,
                'image_path': self.parser.get_image(index)}


@register
class DARODOnRADDet(SampleAdapter):
    """Collapsed 2D sum-of-logs map, reference normalization chain, 2D corner boxes."""
    MODELS, DATASET = ("DAROD",), "RADDetDataset"

    def __init__(self, parser, features, model_cfg, train_cfg, splittype=None):
        super().__init__(parser, features, model_cfg, train_cfg, splittype)
        self.data_stats = parser.data_stats
        if features == ["RD"]:
            self.feature_name = "RD"                 # collapse azimuth -> (range, doppler)
        elif features == ["RA"]:
            self.feature_name = "RA"                 # collapse doppler -> (range, azimuth)
        else:
            raise ValueError(f"DAROD on RADDetDataset builds ['RD'] or ['RA'], not {features}")
        # DAROD standardizes its input: the reference divides the spectrum by the
        # per-dataset data_mean/data_std (darod/utils/data_utils.py). They live in the
        # TRAIN config because model_cfg is splatted into the network constructor
        # (network.py:38) and may only hold DAROD.__init__ arguments. None means "not
        # configured" -- __getitem__ then leaves the input untouched.
        self.data_mean = train_cfg.get("data_mean")
        self.data_std = train_cfg.get("data_std")

    def __getitem__(self, index):
        gt_instances = self.parser.annotation(index)
        feature_name = self.feature_name
        # Reference (colindecourt/darod, datasets/raddet_builder/raddet.py) builds the
        # spectrum as a SUM OF LOGS over the collapsed axis:
        #     complexTo2channels(RAD) = log10(|RAD|^2 + 1)  then  getSumDim(axis)
        # The collapsed map is cached on disk pre-standardization (parser._get_collapsed),
        # so changing data_mean/data_std does not invalidate it.
        feature_data = self.parser._get_collapsed(index, feature_name)

        # The reference normalizes the RAD cube ELEMENTWISE BEFORE collapsing
        # (datasets/raddet_builder/raddet.py:91 -> :99):
        #     RAD = (log10(|RAD|^2 + 1) - global_mean_log) / global_variance_log
        #     RD  = sum over the azimuth axis
        # which, since the cache holds the raw sum S, is (S - n*mean) / variance for
        # n the number of collapsed bins (256 azimuth for RD, 64 doppler for RA).
        # Only then does data_utils.py standardize by data_mean/data_std.
        # Measured: the reference's own trained weights score mAP 0.0031 through this
        # loader with the old scaling, against 0.4414 from its own eval.py.
        n_collapsed = 256 if feature_name == "RD" else 64   # RADDet RAD is (256, 256, 64)
        feature_data = ((feature_data - n_collapsed * self.data_stats["global_mean_log"])
                        / self.data_stats["global_variance_log"])

        # Standardize, as the reference does in darod/utils/data_utils.py:
        #     spectrums = (spectrums - data_mean) / data_std
        if self.data_mean is not None and self.data_std is not None:
            feature_data = (feature_data - self.data_mean) / self.data_std

        x_shape, y_shape = feature_data.shape[1], feature_data.shape[0]
        boxes = gt_instances["boxes"]
        classes = gt_instances["classes"]
        gt_boxes = []
        gt_labels = []
        for (box, class_) in zip(boxes, classes):
            yc, xc, h, w = box[0], box[2], box[3], box[5]
            y1, y2, x1, x2 = int(yc - h / 2), int(yc + h / 2), int(xc - w / 2), int(xc + w / 2)
            if x1 < 0:
                # Create 2 boxes
                x1 += x_shape
                box1 = [y1 / y_shape, x1 / x_shape, y2 / y_shape, x_shape / x_shape]
                box2 = [y1 / y_shape, 0 / x_shape, y2 / y_shape, x2 / x_shape]
                #
                gt_boxes.append(box1)
                gt_labels.append(class_)
                #
                gt_boxes.append(box2)
                gt_labels.append(class_)
            elif x2 >= x_shape:
                x2 -= x_shape
                box1 = [y1 / y_shape, x1 / x_shape, y2 / y_shape, x_shape / x_shape]
                box2 = [y1 / y_shape, 0 / x_shape, y2 / y_shape, x2 / x_shape]
                #
                gt_boxes.append(box1)
                gt_labels.append(class_)
                #
                gt_boxes.append(box2)
                gt_labels.append(class_)
            else:
                gt_boxes.append([y1 / y_shape, x1 / x_shape, y2 / y_shape, x2 / x_shape])
                gt_labels.append(class_)

        gt_labels = [self.data_stats["all_classes"].index(class_name)+1 for class_name in gt_labels] # Plus 1: map to 1~6
        gt_labels = np.array(gt_labels)
        gt_boxes = np.array(gt_boxes)
        # splittype None is treated as train to preserve the old default (one instance
        # for every split) rather than silently drop augmentation.
        if self.splittype in (None, 'train'):
            feature_data, gt_boxes = random_flip(feature_data, gt_boxes, self.transformations)
        feature_data = np.expand_dims(feature_data, axis=0)
        return {feature_name: feature_data, 'label': gt_labels, 'boxes': gt_boxes,
                'image_path': self.parser.get_image(index)}


def clamp_box_edges(xyzwhd, y_shape, z_shape):
    """TransRAD reference radar_dataset_generator.py clamp_box (lines 43-93): a box that
    crosses the azimuth (y) or Doppler (z) edge is clamped to whichever side holds the
    larger part. DAROD splits such boxes in two instead; TransRAD keeps one."""
    arr = xyzwhd2xyzxyz(np.asarray(xyzwhd, dtype=float))
    for i in range(arr.shape[0]):
        x1, y1, z1, x2, y2, z2 = arr[i]
        if y1 < 0:
            y1 += y_shape
            if (y_shape - y1) >= (y2 - 0):
                y1, y2 = y1, y_shape
            else:
                y1, y2 = 0, y2
        elif y2 >= y_shape:
            y2 -= y_shape
            if (y_shape - y1) >= (y2 - 0):
                y1, y2 = y1, y_shape
            else:
                y1, y2 = 0, y2
        if z1 < 0:
            z1 += z_shape
            if (z_shape - z1) >= (z2 - 0):
                z1, z2 = z1, z_shape
            else:
                z1, z2 = 0, z2
        elif z2 >= z_shape:
            z2 -= z_shape
            if (z_shape - z1) >= (z2 - 0):
                z1, z2 = z1, z_shape
            else:
                z1, z2 = 0, z2
        arr[i] = [x1, y1, z1, x2, y2, z2]
    return xyzxyz2xyzwhd(arr)


@register
class TransRADOnRADDet(SampleAdapter):
    """Log-magnitude cube, Doppler resized 64 -> 256 by nearest neighbour, Doppler as the
    channel axis; per-frame normalised boxes [cls, x, y, z, w, h, d]. Mirrors the TransRAD
    reference RadarDataset (utils/radar_dataset_generator.py) with mode 'RA',
    target_shape [256, 256, 256], interpolation 'nearest'. No augmentation on any split."""
    MODELS, DATASET = ("TransRAD",), "RADDetDataset"
    TARGET_SHAPE = (256, 256, 256)

    def __init__(self, parser, features, model_cfg, train_cfg, splittype=None):
        super().__init__(parser, features, model_cfg, train_cfg, splittype)
        if features != ["RAD"]:
            raise ValueError(f"TransRAD on RADDetDataset builds ['RAD'], not {features}")
        self.data_stats = parser.data_stats
        self.input_shape = [256, 256, 64]            # the cube on disk (range, azimuth, doppler)

    def __getitem__(self, index):
        gt = self.parser.annotation(index)
        cube = complexTo2Channels(self.parser.get_RAD(index))
        cube = (cube - self.data_stats["global_mean_log"]) / self.data_stats["global_variance_log"]

        n = len(gt["classes"])
        boxes = np.zeros((n, 7))
        for i in range(n):
            boxes[i, :6] = gt["boxes"][i]
            boxes[i, 6] = self.data_stats["all_classes"].index(gt["classes"][i])
        if n:
            boxes[:, :6] = clamp_box_edges(boxes[:, :6], self.input_shape[1], self.input_shape[2])

        label = np.zeros((n, 7), dtype=np.float32)
        if n:
            label[:, 0] = boxes[:, 6]
            label[:, 1:] = boxes[:, :6]
            label[:, [1, 4]] /= self.input_shape[0]
            label[:, [2, 5]] /= self.input_shape[1]
            label[:, [3, 6]] /= self.input_shape[2]

        scale = tuple(t / s for t, s in zip(self.TARGET_SHAPE, cube.shape))
        cube = zoom(np.asarray(cube, dtype=np.float32), scale, order=0)
        cube = np.transpose(cube, (2, 0, 1))
        return {"RAD": cube, "label": label, "boxes": boxes,
                "image_path": self.parser.get_image(index)}
