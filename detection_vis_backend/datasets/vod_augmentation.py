"""Train-time world augmentation for ViewOfDelft radar point clouds.

Ports of fthbng77/RadarPillar pcdet/datasets/augmentor/augmentor_utils.py
(random_flip_along_x, global_rotation, global_scaling), the transforms active
per vod_radarpillar_rot.yaml:24-32 (gt_sampling is in DISABLE_AUG_LIST there,
so no database sampler exists here). Ported to numpy (the originals depend on
OpenPCDet's torch common_utils), not copied verbatim -- correctness is pinned
by the invariant tests in test_radarpillar.py. Radar-safe by construction:
boxes are (N,7) so the reference's velocity branches (gt_boxes.shape[1] > 7)
never apply, and only point columns 0:3 (xyz) are ever transformed --
rcs/v_r/v_r_comp/time pass through untouched, per the reference docstrings.
"""
import numpy as np


def random_flip_along_x(gt_boxes, points, rng):
    # augmentor_utils.py:6-27
    if rng.random() < 0.5:
        gt_boxes[:, 1] = -gt_boxes[:, 1]
        gt_boxes[:, 6] = -gt_boxes[:, 6]
        points[:, 1] = -points[:, 1]
    return gt_boxes, points


def _rot_z(xyz, angle):
    c, s = np.cos(angle), np.sin(angle)
    out = xyz.copy()
    out[:, 0] = c * xyz[:, 0] - s * xyz[:, 1]
    out[:, 1] = s * xyz[:, 0] + c * xyz[:, 1]
    return out


def global_rotation(gt_boxes, points, rot_range, rng):
    # augmentor_utils.py:54-81 (rotate_points_along_z touches xyz only)
    angle = rng.uniform(rot_range[0], rot_range[1])
    points[:, 0:3] = _rot_z(points[:, 0:3], angle)
    gt_boxes[:, 0:3] = _rot_z(gt_boxes[:, 0:3], angle)
    gt_boxes[:, 6] += angle
    return gt_boxes, points


def global_scaling(gt_boxes, points, scale_range, rng):
    # augmentor_utils.py:84-96
    if scale_range[1] - scale_range[0] < 1e-3:
        return gt_boxes, points
    s = rng.uniform(scale_range[0], scale_range[1])
    points[:, :3] *= s
    gt_boxes[:, :6] *= s
    return gt_boxes, points


class VodAugmentor:
    """Applies the enabled transforms in the reference AUG_CONFIG_LIST order
    (flip -> rotation -> scaling)."""

    def __init__(self, cfg, seed=None):
        self.cfg = cfg or {}
        self.rng = np.random.default_rng(seed)

    def __call__(self, points, gt_boxes):
        points = np.array(points, dtype=np.float32, copy=True)
        gt_boxes = np.array(gt_boxes, dtype=np.float32, copy=True)
        if len(gt_boxes) == 0:
            # the reference transforms points and boxes jointly; without boxes
            # a world transform adds nothing
            return points, gt_boxes
        if self.cfg.get("random_world_flip", False):
            gt_boxes, points = random_flip_along_x(gt_boxes, points, self.rng)
        if self.cfg.get("random_world_rotation", False):
            gt_boxes, points = global_rotation(
                gt_boxes, points, self.cfg.get("rot_range", [-0.78539816, 0.78539816]), self.rng)
        if self.cfg.get("random_world_scaling", False):
            gt_boxes, points = global_scaling(
                gt_boxes, points, self.cfg.get("scale_range", [0.95, 1.05]), self.rng)
        return points, gt_boxes
