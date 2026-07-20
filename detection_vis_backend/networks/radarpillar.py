"""RadarPillar building blocks: a plain-PyTorch port of the PointPillars base of
RadarPillars (Musiat et al., IROS 2024, arXiv:2408.05020), targeting the
View-of-Delft radar point cloud. Design: docs/superpowers/specs/2026-07-16-radarpillar-design.md

Ported by READING the reference implementation (fthbng77/RadarPillar, an OpenPCDet
fork, Apache-2.0) -- conventions verified against these files, cited per function:
  tools/cfgs/vod_models/vod_radarpillar_rot.yaml     (ranges, anchors, thresholds)
  pcdet/utils/box_coder_utils.py                     (ResidualCoder)
  pcdet/utils/box_utils.py                           (boxes3d_nearest_bev_iou)
  pcdet/models/dense_heads/target_assigner/axis_aligned_target_assigner.py
  pcdet/models/dense_heads/target_assigner/anchor_generator.py
  pcdet/models/backbones_3d/vfe/pillar_vfe.py

Conventions:
- boxes are [x, y, z_center, l, w, h, yaw], radar frame (x fwd, y left, z up),
  yaw around +z with 0 along +x;
- anchor/prediction flatten order is (H, W, class, rotation).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from shapely.geometry import Polygon


def _bev_corners(box):
    """(7,) box -> (4,2) BEV corner array."""
    x, y, _, l, w, _, yaw = box[:7]
    dx, dy = l / 2.0, w / 2.0
    corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]], dtype=np.float64)
    c, s = np.cos(yaw), np.sin(yaw)
    rot = np.array([[c, -s], [s, c]])
    return corners @ rot.T + np.array([x, y])


def rotated_iou_bev(boxes_a, boxes_b):
    """Exact rotated BEV IoU (Na,7)x(Nb,7) -> (Na,Nb), shapely on CPU.

    NOT the assigner's metric (that is nearest_bev_iou below, matching the
    reference); exact IoU is for slice-2 NMS and evaluation. A center-distance
    gate zeroes pairs that cannot overlap before any polygon is built.
    """
    boxes_a = np.asarray(boxes_a, dtype=np.float64)
    boxes_b = np.asarray(boxes_b, dtype=np.float64)
    na, nb = len(boxes_a), len(boxes_b)
    iou = np.zeros((na, nb), dtype=np.float32)
    if na == 0 or nb == 0:
        return iou
    diag_a = np.hypot(boxes_a[:, 3], boxes_a[:, 4]) / 2.0
    diag_b = np.hypot(boxes_b[:, 3], boxes_b[:, 4]) / 2.0
    dist = np.hypot(boxes_a[:, None, 0] - boxes_b[None, :, 0],
                    boxes_a[:, None, 1] - boxes_b[None, :, 1])
    candidate = dist <= (diag_a[:, None] + diag_b[None, :])
    polys_b = {j: Polygon(_bev_corners(boxes_b[j])) for j in np.where(candidate.any(axis=0))[0]}
    for i in np.where(candidate.any(axis=1))[0]:
        poly_a = Polygon(_bev_corners(boxes_a[i]))
        area_a = poly_a.area
        for j in np.where(candidate[i])[0]:
            poly_b = polys_b[j]
            inter = poly_a.intersection(poly_b).area
            if inter > 0.0:
                iou[i, j] = inter / (area_a + poly_b.area - inter)
    return iou


def _aligned_bev_boxes(boxes):
    """Round each box's heading to the nearest multiple of pi/2 and return the
    axis-aligned BEV rectangles (N,4) [x1,y1,x2,y2].

    Port of box_utils.boxes3d_lidar_to_aligned_bev_boxes: heading within pi/4 of
    an axis keeps (l,w); otherwise they swap.
    """
    yaw = boxes[:, 6]
    # limit_period(yaw, offset=0.5, period=pi) then abs(), as in the reference
    rot = np.abs(yaw - np.floor(yaw / np.pi + 0.5) * np.pi)
    swap = rot >= np.pi / 4
    dx = np.where(swap, boxes[:, 4], boxes[:, 3])
    dy = np.where(swap, boxes[:, 3], boxes[:, 4])
    return np.stack([boxes[:, 0] - dx / 2, boxes[:, 1] - dy / 2,
                     boxes[:, 0] + dx / 2, boxes[:, 1] + dy / 2], axis=-1)


def nearest_bev_iou(boxes_a, boxes_b):
    """The assigner's IoU: axis-align both sets (nearest 90 deg), then AABB IoU.

    Port of box_utils.boxes3d_nearest_bev_iou -- what AxisAlignedTargetAssigner
    uses when MATCH_HEIGHT is False (the reference VoD config). Vectorized numpy;
    cheap even against the full 153,600-anchor grid, so target assignment needs
    no shapely and no prefilter.
    """
    boxes_a = np.asarray(boxes_a, dtype=np.float64)
    boxes_b = np.asarray(boxes_b, dtype=np.float64)
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)
    a = _aligned_bev_boxes(boxes_a)
    b = _aligned_bev_boxes(boxes_b)
    x_min = np.maximum(a[:, None, 0], b[None, :, 0])
    x_max = np.minimum(a[:, None, 2], b[None, :, 2])
    y_min = np.maximum(a[:, None, 1], b[None, :, 1])
    y_max = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x_max - x_min, 0, None) * np.clip(y_max - y_min, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    union = np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)
    return (inter / union).astype(np.float32)


def _xp(arr):
    return torch if isinstance(arr, torch.Tensor) else np


def encode_boxes(gt, anchors):
    """Residual encoding -- port of box_coder_utils.ResidualCoder.encode_torch:
    xy normalized by the anchor BEV diagonal, z by anchor height, log dims,
    plain angle difference (the sin-difference trick lives in the loss)."""
    xp = _xp(gt)
    clamp = (lambda t: torch.clamp_min(t, 1e-5)) if xp is torch else (lambda t: np.clip(t, 1e-5, None))
    la, wa, ha = clamp(anchors[..., 3]), clamp(anchors[..., 4]), clamp(anchors[..., 5])
    lg, wg, hg = clamp(gt[..., 3]), clamp(gt[..., 4]), clamp(gt[..., 5])
    d = (la ** 2 + wa ** 2) ** 0.5
    out = [
        (gt[..., 0] - anchors[..., 0]) / d,
        (gt[..., 1] - anchors[..., 1]) / d,
        (gt[..., 2] - anchors[..., 2]) / ha,
        xp.log(lg / la),
        xp.log(wg / wa),
        xp.log(hg / ha),
        gt[..., 6] - anchors[..., 6],
    ]
    return xp.stack(out, -1) if xp is torch else np.stack(out, -1).astype(np.float32)


CLASS_NAMES = ["Car", "Pedestrian", "Cyclist"]


def generate_anchors(model_cfg):
    """Anchor grid over the stride-`head_stride` BEV map.

    Flatten order (H=y, W=x, class, rotation) -- must match the head reshape in
    RadarPillarHead.forward and the target assignment; changing one requires
    changing all three (the plan's 'ordering contract'). Placement per the
    reference anchor_generator with align_center=False: endpoint-inclusive,
    stride = range/(grid-1), no half-cell offset. Config gives anchor BOTTOM
    heights (vod_radarpillar_rot.yaml anchor_bottom_heights); box z is the center.
    """
    x0, y0, _, x1, y1, _ = model_cfg["point_cloud_range"]
    vx, vy = model_cfg["voxel_size"][0], model_cfg["voxel_size"][1]
    stride = model_cfg["head_stride"]
    w = int(round((x1 - x0) / vx)) // stride
    h = int(round((y1 - y0) / vy)) // stride
    xs = np.linspace(x0, x1, w)
    ys = np.linspace(y0, y1, h)
    rotations = model_cfg["anchor_rotations"]
    per_loc = []
    per_loc_cls = []
    for ci, name in enumerate(CLASS_NAMES):
        cfg = model_cfg["anchors"][name]
        l, wd, ht = cfg["sizes"]
        for rot in rotations:
            per_loc.append((l, wd, ht, cfg["z_bottom"] + ht / 2.0, rot))
            per_loc_cls.append(ci)
    n_loc = len(per_loc)  # 6
    grid_y, grid_x = np.meshgrid(ys, xs, indexing="ij")          # (h, w)
    centers = np.stack([grid_x, grid_y], -1).reshape(-1, 2)      # (h*w, 2) H-major
    anchors = np.zeros((h * w, n_loc, 7), dtype=np.float32)
    anchors[:, :, 0:2] = centers[:, None, :]
    for k, (l, wd, ht, zc, rot) in enumerate(per_loc):
        anchors[:, k, 2] = zc
        anchors[:, k, 3] = l
        anchors[:, k, 4] = wd
        anchors[:, k, 5] = ht
        anchors[:, k, 6] = rot
    classes = np.tile(np.array(per_loc_cls, dtype=np.int64), h * w)
    return anchors.reshape(-1, 7), classes


POINT_COLUMNS = {"x": 0, "y": 1, "z": 2, "rcs": 3, "v_r": 4, "v_r_comp": 5, "time": 6}


def pillarize(points, model_cfg):
    """Group points into BEV pillars with PointPillars feature decoration.

    Per point: raw configured features + offsets to the pillar's point-mean (3)
    + offsets to the voxel center including z (3) -- the reference pillar_vfe.py
    decorates all three center axes, so D = len(point_features) + 6.
    """
    x0, y0, z0, x1, y1, z1 = model_cfg["point_cloud_range"]
    vx, vy = model_cfg["voxel_size"][0], model_cfg["voxel_size"][1]
    K = model_cfg["max_points_per_pillar"]
    feats = [POINT_COLUMNS[f] for f in model_cfg["point_features"]]
    D = len(feats) + 6
    keep = ((points[:, 0] >= x0) & (points[:, 0] < x1) &
            (points[:, 1] >= y0) & (points[:, 1] < y1) &
            (points[:, 2] >= z0) & (points[:, 2] < z1))
    pts = points[keep]
    nx = int(round((x1 - x0) / vx))
    ny = int(round((y1 - y0) / vy))
    ix = ((pts[:, 0] - x0) / vx).astype(np.int64)
    iy = ((pts[:, 1] - y0) / vy).astype(np.int64)
    # A point epsilon below the range bound passes the float64 keep-filter but the
    # float32 division can round up to exactly nx/ny (51.199997f/0.16f == 320.0f),
    # which would index off the BEV canvas in PillarScatter. The reference voxel
    # generator (spconv point_to_voxel) discards such points; do the same.
    inb = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    pts, ix, iy = pts[inb], ix[inb], iy[inb]
    key = iy * nx + ix
    uniq, inverse = np.unique(key, return_inverse=True)
    P = min(len(uniq), model_cfg["max_pillars"])
    pillars = np.zeros((P, K, D), dtype=np.float32)
    coords = np.zeros((P, 2), dtype=np.int64)
    num_points = np.zeros(P, dtype=np.int64)
    cz = z0 + model_cfg["voxel_size"][2] / 2.0   # one voxel spans the full z range
    for p in range(P):
        sel = np.where(inverse == p)[0][:K]
        n = len(sel)
        group = pts[sel]
        coords[p] = (uniq[p] % nx, uniq[p] // nx)         # (ix, iy)
        num_points[p] = n
        pillars[p, :n, :len(feats)] = group[:, feats]
        mean = group[:, :3].mean(axis=0)
        pillars[p, :n, len(feats):len(feats) + 3] = group[:, :3] - mean
        cx = x0 + (coords[p, 0] + 0.5) * vx
        cy = y0 + (coords[p, 1] + 0.5) * vy
        pillars[p, :n, len(feats) + 3] = group[:, 0] - cx
        pillars[p, :n, len(feats) + 4] = group[:, 1] - cy
        pillars[p, :n, len(feats) + 5] = group[:, 2] - cz
    return {"pillars": pillars, "coords": coords, "num_points": num_points}


def assign_targets(anchors, anchor_classes, gt_boxes, gt_labels, model_cfg):
    """Class-wise anchor->gt matching -- port of AxisAlignedTargetAssigner
    (assign_targets_single) with MATCH_HEIGHT False, so overlaps come from
    nearest_bev_iou, not exact rotated IoU.

    Returns cls_targets: -1 ignore / 0 background / 1..3 = class+1;
    box_targets: encoded residuals at positives; dir_targets: 0/1 at positives.
    Force match per the reference: ALL anchors tying a gt's max IoU become
    positive -- unless the gt overlaps nothing (empty_gt_mask guard).
    Silent-bug territory: every convention here is pinned by a test.
    """
    A = anchors.shape[0]
    cls_targets = np.zeros(A, dtype=np.int64)
    box_targets = np.zeros((A, 7), dtype=np.float32)
    dir_targets = np.zeros(A, dtype=np.int64)
    dir_offset = model_cfg.get("dir_offset", 0.78539)

    for ci, name in enumerate(CLASS_NAMES):
        cfg = model_cfg["anchors"][name]
        amask = np.where(anchor_classes == ci)[0]
        gmask = np.where(gt_labels == ci)[0]
        if len(gmask) == 0:
            continue
        ious = nearest_bev_iou(anchors[amask], gt_boxes[gmask])   # (na, ng)
        best_gt = ious.argmax(axis=1)
        best_iou = ious[np.arange(len(amask)), best_gt]
        pos = best_iou >= cfg["match_threshold"]
        ign = (~pos) & (best_iou > cfg["unmatch_threshold"])
        # force match: all anchors tying each gt's max IoU, unless that max is 0
        gt_max = ious.max(axis=0)                                  # (ng,)
        gt_max[gt_max == 0] = -1.0
        force = np.where(ious == gt_max[None, :])[0]
        pos[force] = True
        ign[force] = False
        cls_targets[amask[ign]] = -1
        pos_idx = amask[pos]
        matched = gt_boxes[gmask][best_gt[pos]]
        cls_targets[pos_idx] = ci + 1
        box_targets[pos_idx] = encode_boxes(matched, anchors[pos_idx])
        dir_targets[pos_idx] = (
            ((matched[:, 6] - dir_offset) % (2 * np.pi)) > np.pi
        ).astype(np.int64)
    return {"cls_targets": cls_targets, "box_targets": box_targets, "dir_targets": dir_targets}


def decode_boxes(deltas, anchors):
    """Inverse of encode_boxes (ResidualCoder.decode_torch)."""
    xp = _xp(deltas)
    d = (anchors[..., 3] ** 2 + anchors[..., 4] ** 2) ** 0.5
    out = [
        deltas[..., 0] * d + anchors[..., 0],
        deltas[..., 1] * d + anchors[..., 1],
        deltas[..., 2] * anchors[..., 5] + anchors[..., 2],
        xp.exp(deltas[..., 3]) * anchors[..., 3],
        xp.exp(deltas[..., 4]) * anchors[..., 4],
        xp.exp(deltas[..., 5]) * anchors[..., 5],
        deltas[..., 6] + anchors[..., 6],
    ]
    return xp.stack(out, -1) if xp is torch else np.stack(out, -1).astype(np.float32)


class PillarVFE(nn.Module):
    """Simplified PointNet over each pillar: Linear+BN+ReLU then masked max
    (reference pillar_vfe.py PFNLayer with USE_NORM, last_layer max-pool)."""
    def __init__(self, in_dim, out_dim=64):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim, bias=False)
        self.bn = nn.BatchNorm1d(out_dim)

    def forward(self, pillars, num_points):
        P, K, D = pillars.shape
        x = self.linear(pillars.view(P * K, D))
        x = F.relu(self.bn(x)).view(P, K, -1)
        mask = (torch.arange(K, device=pillars.device)[None, :] < num_points[:, None])
        x = x.masked_fill(~mask.unsqueeze(-1), float("-inf")).amax(dim=1)
        return torch.nan_to_num(x, neginf=0.0)


class PillarScatter(nn.Module):
    """Scatter pillar features onto the dense BEV canvas (PointPillarScatter)."""
    def __init__(self, nx, ny, channels):
        super().__init__()
        self.nx, self.ny, self.channels = nx, ny, channels

    def forward(self, features, coords, batch_size):
        canvas = features.new_zeros(batch_size, self.channels, self.ny, self.nx)
        canvas[coords[:, 0], :, coords[:, 2], coords[:, 1]] = features
        return canvas


class BEVBackbone(nn.Module):
    """PointPillars conv backbone with upsampled concat (BaseBEVBackbone;
    baseline channels from the reference kitti_models/pointpillar.yaml)."""
    def __init__(self, in_ch, cfg):
        super().__init__()
        self.blocks, self.ups = nn.ModuleList(), nn.ModuleList()
        ch = in_ch
        total_stride = 1
        first_stride = cfg["strides"][0]
        for n, stride, out_ch, up_ch in zip(cfg["layer_nums"], cfg["strides"],
                                            cfg["channels"], cfg["up_channels"]):
            layers = [nn.Conv2d(ch, out_ch, 3, stride=stride, padding=1, bias=False),
                      nn.BatchNorm2d(out_ch), nn.ReLU()]
            for _ in range(n):
                layers += [nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
                           nn.BatchNorm2d(out_ch), nn.ReLU()]
            self.blocks.append(nn.Sequential(*layers))
            total_stride *= stride
            up_stride = total_stride // first_stride   # upsample strides [1, 2, 4]
            self.ups.append(nn.Sequential(
                nn.ConvTranspose2d(out_ch, up_ch, up_stride, stride=up_stride, bias=False)
                if up_stride > 1 else nn.Conv2d(out_ch, up_ch, 1, bias=False),
                nn.BatchNorm2d(up_ch), nn.ReLU()))
            ch = out_ch
        self.out_channels = sum(cfg["up_channels"])

    def forward(self, x):
        outs = []
        for block, up in zip(self.blocks, self.ups):
            x = block(x)
            outs.append(up(x))
        return torch.cat(outs, dim=1)


class RadarPillarHead(nn.Module):
    """1x1-conv anchor head (AnchorHeadSingle): cls/box/dir per anchor."""
    def __init__(self, in_ch, n_anchor_per_loc=6, n_class=3):
        super().__init__()
        self.n_a, self.n_c = n_anchor_per_loc, n_class
        self.cls = nn.Conv2d(in_ch, n_anchor_per_loc * n_class, 1)
        self.box = nn.Conv2d(in_ch, n_anchor_per_loc * 7, 1)
        self.dir = nn.Conv2d(in_ch, n_anchor_per_loc * 2, 1)

    def _flat(self, t, c):
        B, _, H, W = t.shape
        # ordering contract: (H, W, anchor, channel) == generate_anchors order
        return t.permute(0, 2, 3, 1).reshape(B, H * W * self.n_a, c)

    def forward(self, x):
        return {"cls_preds": self._flat(self.cls(x), self.n_c),
                "box_preds": self._flat(self.box(x), 7),
                "dir_preds": self._flat(self.dir(x), 2)}


class RadarPillarBackboneNet(nn.Module):
    """VFE -> scatter -> backbone -> head. Config keys documented in
    RadarPillar_model_config.json; anchors generated once and kept as attributes."""
    def __init__(self, model_cfg):
        super().__init__()
        self.model_cfg = model_cfg
        x0, y0, _, x1, y1, _ = model_cfg["point_cloud_range"]
        vx, vy = model_cfg["voxel_size"][0], model_cfg["voxel_size"][1]
        self.nx, self.ny = int(round((x1 - x0) / vx)), int(round((y1 - y0) / vy))
        D = len(model_cfg["point_features"]) + 6
        C = model_cfg.get("vfe_channels", 64)
        self.vfe = PillarVFE(D, C)
        self.scatter = PillarScatter(self.nx, self.ny, C)
        self.backbone = BEVBackbone(C, model_cfg["backbone"])
        self.head = RadarPillarHead(self.backbone.out_channels)
        anchors, anchor_classes = generate_anchors(model_cfg)
        self.anchors_np, self.anchor_classes_np = anchors, anchor_classes
        self.register_buffer("anchors", torch.from_numpy(anchors), persistent=False)

    def forward(self, pillars, coords, num_points, batch_size):
        feats = self.vfe(pillars, num_points)
        bev = self.scatter(feats, coords, batch_size)
        return self.head(self.backbone(bev))


def _sigmoid_focal(logits, onehot, alpha, gamma):
    """Sigmoid focal loss, elementwise (SigmoidFocalClassificationLoss)."""
    p = torch.sigmoid(logits)
    ce = F.binary_cross_entropy_with_logits(logits, onehot, reduction="none")
    p_t = p * onehot + (1 - p) * (1 - onehot)
    a_t = alpha * onehot + (1 - alpha) * (1 - onehot)
    return a_t * (1 - p_t) ** gamma * ce


def radarpillar_loss(pred, targets, anchors_np, anchor_classes_np, loss_cfg, device):
    """Focal (cls, over cared anchors) + smooth-L1 with sin-difference heading
    (box, positives) + CE (direction bins, positives).

    Normalization follows the reference AnchorHeadTemplate: each sample's loss is
    divided by ITS OWN positive count (weights /= clamp(pos_normalizer, min=1)),
    then averaged over the batch. Weights 1.0/2.0/0.2 per vod_radarpillar_rot.yaml.
    """
    cls_t = torch.from_numpy(np.stack([t["cls_targets"] for t in targets])).to(device)
    box_t = torch.from_numpy(np.stack([t["box_targets"] for t in targets])).to(device)
    dir_t = torch.from_numpy(np.stack([t["dir_targets"] for t in targets])).to(device)
    B, A = cls_t.shape
    cared = cls_t >= 0
    pos = cls_t > 0
    n_pos = pos.sum(dim=1).clamp(min=1).float()          # (B,) per-sample normalizer

    onehot = torch.zeros(B, A, 3, device=device)
    onehot[pos] = F.one_hot((cls_t[pos] - 1).long(), 3).float()
    focal = _sigmoid_focal(pred["cls_preds"].to(device), onehot,
                           loss_cfg.get("focal_alpha", 0.25), loss_cfg.get("focal_gamma", 2.0))
    loss_cls = ((focal * cared.unsqueeze(-1)).sum(dim=[1, 2]) / n_pos).mean()

    bp, bt = pred["box_preds"].to(device).clone(), box_t.clone()
    # sin-difference heading (SECOND): replace the angle channel of pred/target
    pa, ta = bp[..., 6].clone(), bt[..., 6].clone()
    bp[..., 6] = torch.sin(pa) * torch.cos(ta)
    bt[..., 6] = torch.cos(pa) * torch.sin(ta)
    l1 = F.smooth_l1_loss(bp, bt, reduction="none").sum(-1)
    loss_box = ((l1 * pos).sum(dim=1) / n_pos).mean()

    dl = F.cross_entropy(pred["dir_preds"].to(device).view(-1, 2),
                         dir_t.view(-1), reduction="none").view(B, A)
    loss_dir = ((dl * pos).sum(dim=1) / n_pos).mean()

    loss = (loss_cfg.get("cls_weight", 1.0) * loss_cls +
            loss_cfg.get("box_weight", 2.0) * loss_box +
            loss_cfg.get("dir_weight", 0.2) * loss_dir)
    return {"loss": loss, "loss_cls": loss_cls, "loss_box": loss_box, "loss_dir": loss_dir}


def _limit_period(val, offset, period):
    return val - torch.floor(val / period + offset) * period


def decode_predictions(pred, anchors_t, model_cfg):
    """Raw head outputs -> decoded boxes + sigmoid scores, with the reference's
    direction-bin correction (anchor_head_template.generate_predicted_boxes):
    yaw = limit_period(yaw - DIR_OFFSET, DIR_LIMIT_OFFSET=0, pi) + DIR_OFFSET + pi*bin.
    """
    boxes = decode_boxes(pred["box_preds"], anchors_t.unsqueeze(0))
    dir_offset = model_cfg.get("dir_offset", 0.78539)
    dir_labels = pred["dir_preds"].argmax(dim=-1)
    period = np.pi   # 2pi / 2 bins
    dir_rot = _limit_period(boxes[..., 6] - dir_offset, 0.0, period)
    boxes = boxes.clone()
    boxes[..., 6] = dir_rot + dir_offset + period * dir_labels.to(boxes.dtype)
    return boxes, torch.sigmoid(pred["cls_preds"])


def rotated_nms(boxes, scores, iou_thresh, pre_max, post_max):
    """Greedy NMS on exact rotated BEV IoU; returns kept indices into `boxes`.
    CPU is fine: candidates are capped at pre_max=4096 and VoD scenes hold tens
    of objects."""
    order = np.argsort(-scores)[:pre_max]
    keep = []
    while len(order) and len(keep) < post_max:
        i = order[0]
        keep.append(int(i))
        if len(order) == 1:
            break
        ious = rotated_iou_bev(boxes[i:i + 1], boxes[order[1:]])[0]
        order = order[1:][ious <= iou_thresh]
    return np.array(keep, dtype=np.int64)


def postprocess(pred, anchors_t, model_cfg, train_cfg):
    """Per-class score threshold + rotated NMS, per the reference
    model_nms_utils.multi_classes_nms (MULTI_CLASSES_NMS: True)."""
    boxes_b, scores_b = decode_predictions(pred, anchors_t, model_cfg)
    score_thresh = train_cfg.get("confidence_threshold", 0.1)
    results = []
    for b in range(boxes_b.shape[0]):
        boxes = boxes_b[b].detach().cpu().numpy()
        scores = scores_b[b].detach().cpu().numpy()
        out_boxes, out_labels, out_scores = [], [], []
        for k in range(scores.shape[1]):
            mask = scores[:, k] >= score_thresh
            if not mask.any():
                continue
            kept = rotated_nms(boxes[mask], scores[mask, k],
                               train_cfg.get("nms_iou_threshold", 0.1),
                               train_cfg.get("nms_pre_max", 4096),
                               train_cfg.get("nms_post_max", 500))
            out_boxes.append(boxes[mask][kept])
            out_scores.append(scores[mask, k][kept])
            out_labels.append(np.full(len(kept), k, dtype=np.int64))
        if out_boxes:
            results.append({"boxes": np.concatenate(out_boxes),
                            "labels": np.concatenate(out_labels),
                            "scores": np.concatenate(out_scores)})
        else:
            results.append({"boxes": np.zeros((0, 7), np.float32),
                            "labels": np.zeros((0,), np.int64),
                            "scores": np.zeros((0,), np.float32)})
    return results


def box_corners_3d(box):
    """(7,) box -> (8,3) radar-frame corners, bottom 4 then top 4, yaw-rotated."""
    x, y, z, l, w, h, yaw = box[:7]
    xc = np.array([l, l, -l, -l, l, l, -l, -l]) / 2.0
    yc = np.array([w, -w, -w, w, w, -w, -w, w]) / 2.0
    zc = np.array([-h, -h, -h, -h, h, h, h, h]) / 2.0
    c, s = np.cos(yaw), np.sin(yaw)
    corners = np.stack([c * xc - s * yc + x, s * xc + c * yc + y, zc + z], axis=-1)
    return corners
