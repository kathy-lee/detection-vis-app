"""TransRAD: Retentive Vision Transformer for radar object detection on RAD cubes.

Port of https://github.com/radar-lab/TransRAD (commit 8f9b9ad, GPL-3.0; Cheng & Cao,
IEEE TRS 2025, arXiv 2501.17977), itself built on qhfan/RMT and bubbliiiing/yolov8-pytorch.
A pristine checkout lives at /home/kangle/dataset/transrad_reference/TransRAD and every
class below names the file and lines it was copied from. Deviations from the reference
are commented where they happen; there are five: SyncBatchNorm -> BatchNorm2d (same
keys, same single-GPU behaviour), feature strides fixed from the architecture instead of
a dummy forward at construction, per-step prints removed, the two timm helpers vendored,
and the unused image-dataset code left out.

Layout: the RMT backbone, the YOLOv8 neck + decoupled heads (TransRADBody), the loss
(task-aligned assignment, DFL, CIoU, focal), the decoder and the two-stage NMS
(class NMS then Location-Aware NMS).
"""
import math
import warnings

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from detection_vis_backend.train.utils import iou3d_np

REF_CHECKPOINT = "/home/kangle/dataset/transrad_reference/TransRAD/model_data/best_epoch_weights.pth"


# ---------------------------------------------------------------------------------------
# Vendored from timm (rwightman/pytorch-image-models, Apache-2.0): timm/layers/drop.py
# drop_path/DropPath and timm/layers/weight_init.py trunc_normal_. The reference imports
# them from timm; copying ~40 lines avoids a dependency on the whole package.
# ---------------------------------------------------------------------------------------
def drop_path(x, drop_prob: float = 0., training: bool = False, scale_by_keep: bool = True):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = x.new_empty(shape).bernoulli_(keep_prob)
    if keep_prob > 0.0 and scale_by_keep:
        random_tensor.div_(keep_prob)
    return x * random_tensor


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample."""

    def __init__(self, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training, self.scale_by_keep)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    def norm_cdf(x):
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("mean is more than 2 std from [a, b] in nn.init.trunc_normal_. "
                      "The distribution of values may be incorrect.", stacklevel=2)
    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)
        tensor.uniform_(2 * l - 1, 2 * u - 1)
        tensor.erfinv_()
        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)
