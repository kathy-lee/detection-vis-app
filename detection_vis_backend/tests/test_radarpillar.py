import numpy as np
import pytest

from detection_vis_backend.networks.radarpillar import rotated_iou_bev, nearest_bev_iou


def _box(x, y, l, w, yaw):
    return np.array([x, y, 0.0, l, w, 1.5, yaw], dtype=np.float32)


def test_rotated_iou_identical_and_disjoint():
    a = _box(5, 5, 4, 2, 0.3)
    far = _box(50, 50, 4, 2, 0.0)
    iou = rotated_iou_bev(np.stack([a, far]), np.stack([a]))
    assert iou[0, 0] == pytest.approx(1.0, abs=1e-5)
    assert iou[1, 0] == 0.0


def test_rotated_iou_crossed_pair():
    # Two 4x2 boxes at the same center, one rotated 90 deg: intersection is the
    # 2x2 center square = 4; union = 8 + 8 - 4 = 12; IoU = 1/3.
    a = _box(0, 0, 4, 2, 0.0)
    b = _box(0, 0, 4, 2, np.pi / 2)
    iou = rotated_iou_bev(np.stack([a]), np.stack([b]))
    assert iou[0, 0] == pytest.approx(1.0 / 3.0, abs=1e-4)


def test_rotated_iou_translation_invariance():
    a = _box(1, 2, 3.9, 1.6, 0.7)
    b = _box(2, 2.5, 3.9, 1.6, 1.1)
    i1 = rotated_iou_bev(np.stack([a]), np.stack([b]))[0, 0]
    a2, b2 = a.copy(), b.copy()
    a2[:2] += 17.0
    b2[:2] += 17.0
    i2 = rotated_iou_bev(np.stack([a2]), np.stack([b2]))[0, 0]
    assert i1 == pytest.approx(i2, abs=1e-5)
    assert 0.0 < i1 < 1.0


def test_nearest_bev_iou_rounds_heading_to_axis():
    """The assigner's IoU (OpenPCDet boxes3d_nearest_bev_iou, MATCH_HEIGHT False):
    headings are rounded to the nearest multiple of pi/2 BEFORE a plain AABB IoU.
    A 0.3-rad rotated box therefore scores IoU 1.0 against its unrotated twin --
    exactly where it differs from the exact rotated IoU."""
    a = _box(10, 10, 4, 2, 0.0)
    slightly_rotated = _box(10, 10, 4, 2, 0.3)          # < pi/4 -> treated as axis-aligned
    assert nearest_bev_iou(np.stack([a]), np.stack([slightly_rotated]))[0, 0] == pytest.approx(1.0)
    assert rotated_iou_bev(np.stack([a]), np.stack([slightly_rotated]))[0, 0] < 1.0


def test_nearest_bev_iou_swaps_dims_past_quarter_pi():
    # yaw ~ pi/2: (l,w) swap -> a 4x2 box becomes a 2x4 AABB; vs the 4x2 at same
    # center: intersection 2x2=4, union 12 -> 1/3
    a = _box(0, 0, 4, 2, 0.0)
    b = _box(0, 0, 4, 2, np.pi / 2 + 0.1)
    assert nearest_bev_iou(np.stack([a]), np.stack([b]))[0, 0] == pytest.approx(1.0 / 3.0, abs=1e-5)


def test_nearest_bev_iou_disjoint_and_shape():
    a = np.stack([_box(5, 5, 4, 2, 0.2), _box(40, -10, 1, 1, 0.0)])
    b = np.stack([_box(5, 5, 4, 2, 0.0)])
    iou = nearest_bev_iou(a, b)
    assert iou.shape == (2, 1)
    assert iou[1, 0] == 0.0


def test_box_coder_roundtrip():
    from detection_vis_backend.networks.radarpillar import encode_boxes, decode_boxes
    anchors = np.array([[10, -4, -1.0, 3.9, 1.6, 1.56, 0.0],
                        [30, 12, -1.0, 0.8, 0.6, 1.73, np.pi / 2]], dtype=np.float32)
    gt = np.array([[10.7, -3.5, -0.8, 4.2, 1.7, 1.5, 0.25],
                   [29.6, 12.2, -1.1, 0.9, 0.65, 1.8, 1.9]], dtype=np.float32)
    deltas = encode_boxes(gt, anchors)
    back = decode_boxes(deltas, anchors)
    np.testing.assert_allclose(back, gt, rtol=1e-5, atol=1e-5)


def test_box_coder_zero_residual_for_identical():
    from detection_vis_backend.networks.radarpillar import encode_boxes
    a = np.array([[5, 5, -1.0, 3.9, 1.6, 1.56, 0.3]], dtype=np.float32)
    np.testing.assert_allclose(encode_boxes(a.copy(), a), np.zeros((1, 7)), atol=1e-6)


# Values verified against vod_radarpillar_rot.yaml (see the plan's Source-verification
# amendments): z-range top is 2 (not 3), voxel z = 5, anchor heights are BOTTOMS.
MODEL_CFG = {
    "point_cloud_range": [0.0, -25.6, -3.0, 51.2, 25.6, 2.0],
    "voxel_size": [0.16, 0.16, 5.0],
    "head_stride": 2,
    "anchor_rotations": [0.0, 1.57],
    "anchors": {
        "Car":        {"sizes": [3.9, 1.6, 1.56], "z_bottom": -1.78, "match_threshold": 0.6,  "unmatch_threshold": 0.45},
        "Pedestrian": {"sizes": [0.8, 0.6, 1.73], "z_bottom": -0.6,  "match_threshold": 0.5,  "unmatch_threshold": 0.35},
        "Cyclist":    {"sizes": [1.76, 0.6, 1.73], "z_bottom": -0.6, "match_threshold": 0.5,  "unmatch_threshold": 0.35},
    },
}


def test_anchor_grid_count_order_and_coverage():
    from detection_vis_backend.networks.radarpillar import generate_anchors
    anchors, classes = generate_anchors(MODEL_CFG)
    h = w = int(51.2 / 0.16 / 2)  # 160
    assert anchors.shape == (h * w * 3 * 2, 7)
    assert classes.shape == (anchors.shape[0],)
    # flatten order (H, W, class, rotation): the first 6 anchors share one location
    first = anchors[:6]
    assert np.allclose(first[:, 0], first[0, 0]) and np.allclose(first[:, 1], first[0, 1])
    assert list(classes[:6]) == [0, 0, 1, 1, 2, 2]
    assert first[0, 6] == 0.0 and abs(first[1, 6] - 1.57) < 1e-6
    # align_center=False (source): anchors span the range ENDPOINT-INCLUSIVE,
    # stride = range/(grid-1); first anchor sits exactly at the range minimum
    assert anchors[:, 0].min() == pytest.approx(0.0, abs=1e-6)
    assert anchors[:, 0].max() == pytest.approx(51.2, abs=1e-4)
    assert anchors[:, 1].min() == pytest.approx(-25.6, abs=1e-4)
    # anchor z is the box CENTER = configured bottom + h/2
    assert first[0, 2] == pytest.approx(-1.78 + 1.56 / 2, abs=1e-6)   # Car -> -1.0
    assert first[2, 2] == pytest.approx(-0.6 + 1.73 / 2, abs=1e-6)    # Pedestrian -> 0.265


MODEL_CFG_DIR = dict(MODEL_CFG, dir_offset=0.78539)


def _grid():
    from detection_vis_backend.networks.radarpillar import generate_anchors
    return generate_anchors(MODEL_CFG_DIR)


def test_assignment_no_gt_is_all_background():
    from detection_vis_backend.networks.radarpillar import assign_targets
    anchors, classes = _grid()
    out = assign_targets(anchors, classes,
                         np.zeros((0, 7), dtype=np.float32),
                         np.zeros((0,), dtype=np.int64), MODEL_CFG_DIR)
    assert (out["cls_targets"] == 0).all()


def test_assignment_perfect_overlap_is_positive_with_zero_residual():
    from detection_vis_backend.networks.radarpillar import assign_targets
    anchors, classes = _grid()
    # a gt that IS one of the Car rot-0 anchors: guaranteed IoU 1.0
    car_rot0 = np.where((classes == 0) & (np.abs(anchors[:, 6]) < 1e-6))[0]
    idx = car_rot0[len(car_rot0) // 2]
    gt = anchors[idx:idx + 1].copy()
    out = assign_targets(anchors, classes, gt, np.array([0]), MODEL_CFG_DIR)
    assert out["cls_targets"][idx] == 1                     # Car -> 1
    np.testing.assert_allclose(out["box_targets"][idx], 0.0, atol=1e-5)
    # OpenPCDet dir bin = floor(((yaw - dir_offset) mod 2pi) / pi):
    # yaw 0, offset ~pi/4 -> (-pi/4) mod 2pi = 7pi/4 -> bin 1
    assert out["dir_targets"][idx] == 1


def test_assignment_respects_class_wise_matching():
    from detection_vis_backend.networks.radarpillar import assign_targets
    anchors, classes = _grid()
    car_rot0 = np.where((classes == 0) & (np.abs(anchors[:, 6]) < 1e-6))[0]
    idx = car_rot0[len(car_rot0) // 2]
    gt = anchors[idx:idx + 1].copy()
    # same box labeled Pedestrian: the Car anchor at idx must NOT be positive for it
    out = assign_targets(anchors, classes, gt, np.array([1]), MODEL_CFG_DIR)
    assert out["cls_targets"][idx] != 1
    # force-match: some pedestrian anchor overlaps this box, so at least one is positive
    assert (out["cls_targets"] == 2).sum() >= 1


def test_assignment_ignore_band():
    from detection_vis_backend.networks.radarpillar import assign_targets
    anchors, classes = _grid()
    car_rot0 = np.where((classes == 0) & (np.abs(anchors[:, 6]) < 1e-6))[0]
    idx = car_rot0[len(car_rot0) // 2]
    gt = anchors[idx:idx + 1].copy()
    gt[0, 0] += 1.3  # shift: overlapping neighbours fall between unmatch 0.45 and match 0.6
    out = assign_targets(anchors, classes, gt, np.array([0]), MODEL_CFG_DIR)
    assert (out["cls_targets"] == -1).sum() >= 1, "expected an ignore band around the gt"


PILLAR_CFG = dict(MODEL_CFG_DIR, max_points_per_pillar=32, max_pillars=16000,
                  point_features=["x", "y", "z", "rcs"])


def test_pillarize_groups_points_and_offsets():
    from detection_vis_backend.networks.radarpillar import pillarize
    # four points: two share a pillar, one is elsewhere, one is out of range
    pts = np.zeros((4, 7), dtype=np.float32)
    pts[0, :4] = [1.00, 0.00, -1.0, 5.0]
    pts[1, :4] = [1.10, 0.05, -0.5, 3.0]   # same 0.16m cell as pts[0]
    pts[2, :4] = [40.0, 10.0, 0.0, 1.0]
    pts[3, :4] = [80.0, 0.0, 0.0, 1.0]     # x beyond 51.2 -> dropped
    out = pillarize(pts, PILLAR_CFG)
    assert out["pillars"].shape[0] == 2 and out["pillars"].shape[1:] == (32, 10)
    assert set(out["num_points"].tolist()) == {2, 1}
    two = out["pillars"][out["num_points"].argmax()]
    # offsets to pillar mean: the two points' x-offsets are symmetric around 0
    assert two[0, 4] == pytest.approx(-two[1, 4], abs=1e-5)
    # padded rows stay zero
    assert np.all(two[2:] == 0.0)


def test_collate_prepends_batch_index_and_pads_gt():
    import torch
    from detection_vis_backend.train.utils import radarpillar_collate
    s0 = {"pillars": np.zeros((3, 32, 10), np.float32), "coords": np.zeros((3, 2), np.int64),
          "num_points": np.ones(3, np.int64),
          "gt_boxes": np.zeros((2, 7), np.float32), "gt_labels": np.zeros(2, np.int64),
          "frame_id": 7}
    s1 = {"pillars": np.zeros((5, 32, 10), np.float32), "coords": np.zeros((5, 2), np.int64),
          "num_points": np.ones(5, np.int64),
          "gt_boxes": np.zeros((0, 7), np.float32), "gt_labels": np.zeros(0, np.int64),
          "frame_id": 9}
    b = radarpillar_collate([s0, s1])
    assert b["pillars"].shape == (8, 32, 10) and b["coords"].shape == (8, 3)
    assert b["coords"][:3, 0].unique().tolist() == [0] and b["coords"][3:, 0].unique().tolist() == [1]
    assert b["gt_boxes"].shape == (2, 2, 7) and b["gt_labels"][1].tolist() == [-1, -1]
    assert b["frame_ids"] == [7, 9]


def test_forward_shapes():
    import torch
    from detection_vis_backend.networks.radarpillar import RadarPillarBackboneNet
    from detection_vis_backend.train.utils import radarpillar_collate
    cfg = dict(PILLAR_CFG, vfe_channels=64,
               backbone={"layer_nums": [3, 5, 5], "strides": [2, 2, 2],
                         "channels": [64, 128, 256], "up_channels": [128, 128, 128]})
    net = RadarPillarBackboneNet(cfg)
    s = {"pillars": np.random.rand(40, 32, 10).astype(np.float32),
         "coords": np.stack([np.random.randint(0, 320, 40), np.random.randint(0, 320, 40)], -1).astype(np.int64),
         "num_points": np.random.randint(1, 32, 40).astype(np.int64),
         "gt_boxes": np.zeros((1, 7), np.float32), "gt_labels": np.zeros(1, np.int64), "frame_id": 0}
    batch = radarpillar_collate([s, s])
    out = net(batch["pillars"], batch["coords"], batch["num_points"], batch_size=2)
    A = 160 * 160 * 6
    assert out["cls_preds"].shape == (2, A, 3)
    assert out["box_preds"].shape == (2, A, 7)
    assert out["dir_preds"].shape == (2, A, 2)


def _loss_cfg():
    return {"cls_weight": 1.0, "box_weight": 2.0, "dir_weight": 0.2,
            "focal_alpha": 0.25, "focal_gamma": 2.0}


def test_loss_is_finite_and_perfect_prediction_is_small():
    import torch
    from detection_vis_backend.networks.radarpillar import assign_targets, radarpillar_loss
    anchors, classes = _grid()
    gt = anchors[np.where((classes == 0) & (np.abs(anchors[:, 6]) < 1e-6))[0][100]][None].copy()
    tgt = assign_targets(anchors, classes, gt, np.array([0]), MODEL_CFG_DIR)
    A = anchors.shape[0]
    cls_preds = torch.full((1, A, 3), -12.0)   # sigmoid ~ 0 everywhere
    pos = np.where(tgt["cls_targets"] > 0)[0]
    cls_preds[0, pos, 0] = 12.0                # confident Car at positives
    box_preds = torch.from_numpy(tgt["box_targets"]).unsqueeze(0)
    dir_preds = torch.zeros((1, A, 2))
    dir_preds[0, pos, tgt["dir_targets"][pos]] = 10.0
    out = radarpillar_loss({"cls_preds": cls_preds, "box_preds": box_preds, "dir_preds": dir_preds},
                           [tgt], anchors, classes, _loss_cfg(), "cpu")
    assert torch.isfinite(out["loss"])
    assert out["loss"].item() < 0.1, "a perfect prediction must score near-zero loss"


def test_loss_penalizes_wrong_boxes_more():
    import torch
    from detection_vis_backend.networks.radarpillar import assign_targets, radarpillar_loss
    anchors, classes = _grid()
    gt = anchors[np.where((classes == 0) & (np.abs(anchors[:, 6]) < 1e-6))[0][100]][None].copy()
    tgt = assign_targets(anchors, classes, gt, np.array([0]), MODEL_CFG_DIR)
    A = anchors.shape[0]
    good = torch.from_numpy(tgt["box_targets"]).unsqueeze(0)
    bad = good + 1.0
    base = {"cls_preds": torch.full((1, A, 3), -12.0), "dir_preds": torch.zeros((1, A, 2))}
    l_good = radarpillar_loss(dict(base, box_preds=good), [tgt], anchors, classes, _loss_cfg(), "cpu")["loss_box"]
    l_bad = radarpillar_loss(dict(base, box_preds=bad), [tgt], anchors, classes, _loss_cfg(), "cpu")["loss_box"]
    assert l_bad > l_good
