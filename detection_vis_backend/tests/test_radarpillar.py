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


def test_pillarize_boundary_point_never_indexes_off_grid():
    """A point epsilon below the upper range bound passes the float64 keep-filter
    but the float32 division (51.199997f / 0.16f) rounds UP to exactly 320.0,
    yielding ix == nx. That coord later drives the BEV scatter
    (canvas[b, :, iy, ix]) and fires a CUDA device-side assert -- this killed the
    first slice-3.5 gate run at epoch 1 (augmentation re-randomizes coordinates
    every epoch, so boundary hits are re-rolled ~7.7M times per epoch).
    The reference voxel generator (spconv, used by OpenPCDet) discards points
    whose computed voxel index falls outside the grid; pillarize must do the same.
    """
    from detection_vis_backend.networks.radarpillar import pillarize
    x_edge = np.nextafter(np.float32(51.2), np.float32(0))    # 51.199997 < 51.2
    y_edge = np.nextafter(np.float32(25.6), np.float32(0))    # 25.599998 < 25.6
    pts = np.zeros((3, 7), dtype=np.float32)
    pts[0, :4] = [x_edge, 0.00, -1.0, 5.0]
    pts[1, :4] = [40.0, y_edge, 0.0, 1.0]
    pts[2, :4] = [1.0, 0.0, -1.0, 2.0]                        # ordinary in-range point
    out = pillarize(pts, PILLAR_CFG)
    nx = ny = 320
    assert (out["coords"][:, 0] < nx).all(), f"ix off grid: {out['coords'][:, 0].max()}"
    assert (out["coords"][:, 1] < ny).all(), f"iy off grid: {out['coords'][:, 1].max()}"
    assert (out["coords"] >= 0).all()


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


def test_decode_dir_correction_recovers_flipped_heading():
    """A heading in the second half-circle must be recovered by the direction bin:
    yaw_out = limit_period(yaw - off, 0, pi) + off + pi * bin (reference
    anchor_head_template.generate_predicted_boxes)."""
    import torch
    from detection_vis_backend.networks.radarpillar import encode_boxes, decode_predictions
    anchors, classes = _grid()
    idx = np.where((classes == 0) & (np.abs(anchors[:, 6]) < 1e-6))[0][100]
    gt = anchors[idx:idx + 1].copy()
    gt[0, 6] = np.pi * 0.9            # heading in the second half-circle
    deltas = encode_boxes(gt, anchors[idx:idx + 1])
    A = anchors.shape[0]
    box_preds = torch.zeros((1, A, 7))
    box_preds[0, idx] = torch.from_numpy(deltas[0])
    cls_preds = torch.full((1, A, 3), -12.0)
    cls_preds[0, idx, 0] = 12.0
    # dir bin for yaw 0.9pi, offset ~pi/4: ((0.9pi - pi/4) mod 2pi) = 0.65pi < pi -> bin 0
    dir_preds = torch.zeros((1, A, 2))
    dir_preds[0, idx, 0] = 10.0
    boxes, scores = decode_predictions(
        {"cls_preds": cls_preds, "box_preds": box_preds, "dir_preds": dir_preds},
        torch.from_numpy(anchors), MODEL_CFG_DIR)
    diff = (boxes[0, idx, 6].item() - gt[0, 6]) % (2 * np.pi)
    assert min(diff, 2 * np.pi - diff) < 1e-4
    assert scores[0, idx, 0] > 0.99   # sigmoid of 12


def test_rotated_nms_keeps_best_and_disjoint():
    from detection_vis_backend.networks.radarpillar import rotated_nms
    boxes = np.stack([_box(10, 10, 4, 2, 0.0), _box(10.2, 10, 4, 2, 0.05), _box(40, -5, 4, 2, 1.0)])
    scores = np.array([0.9, 0.6, 0.8], dtype=np.float32)
    keep = rotated_nms(boxes, scores, iou_thresh=0.1, pre_max=4096, post_max=500)
    assert list(keep) == [0, 2]   # overlapping lower-score suppressed; disjoint kept


def test_postprocess_thresholds_and_labels():
    import torch
    from detection_vis_backend.networks.radarpillar import postprocess
    anchors, classes = _grid()
    A = anchors.shape[0]
    car_idx = np.where((classes == 0) & (np.abs(anchors[:, 6]) < 1e-6))[0][100]
    ped_idx = np.where((classes == 1) & (np.abs(anchors[:, 6]) < 1e-6))[0][200]
    cls_preds = torch.full((1, A, 3), -12.0)
    cls_preds[0, car_idx, 0] = 12.0     # confident Car
    cls_preds[0, ped_idx, 1] = -1.0     # sigmoid ~0.27 Pedestrian
    pred = {"cls_preds": cls_preds, "box_preds": torch.zeros((1, A, 7)),
            "dir_preds": torch.zeros((1, A, 2))}
    tcfg = {"confidence_threshold": 0.1, "nms_iou_threshold": 0.1,
            "nms_pre_max": 4096, "nms_post_max": 500}
    out = postprocess(pred, torch.from_numpy(anchors), MODEL_CFG_DIR, tcfg)[0]
    assert 0 in out["labels"] and 1 in out["labels"]     # both above 0.1
    tcfg["confidence_threshold"] = 0.5
    out = postprocess(pred, torch.from_numpy(anchors), MODEL_CFG_DIR, tcfg)[0]
    assert set(out["labels"].tolist()) == {0}            # ped filtered at 0.5


def test_infer_readout_geometry():
    """Backend-computed display geometry (the frontend has no calib access):
    each image box row = 8 projected corners + cls + score (18 values), each BEV
    box row = 4 corners + cls + score (10 values). Uses the real VoD calib of
    file 110 and an UNTRAINED net on CPU -- geometry must be structurally valid
    regardless of prediction quality. Mirrors the infer() branch: collate ->
    net(input) -> postprocess -> radarpillar_display_geometry."""
    import os, json, torch
    os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost/postgres")
    from detection_vis_backend.datasets.view_of_delft import ViewOfDelft
    from detection_vis_backend.networks.network import NetworkFactory
    from detection_vis_backend.networks.radarpillar import postprocess
    from detection_vis_backend.train.utils import radarpillar_collate
    from detection_vis_backend.train.inference import radarpillar_display_geometry
    from data.database import get_session_local
    from data import crud

    model_cfg = json.load(open("detection_vis_backend/networks/RadarPillar_model_config.json"))
    train_cfg = json.load(open("detection_vis_backend/train/RadarPillar_train_config.json"))
    # untrained net emits ~0.5 sigmoid everywhere; keep the readout fast and non-empty
    train_cfg = dict(train_cfg, confidence_threshold=0.0, nms_pre_max=256, nms_post_max=3)

    db = get_session_local()
    datafile = crud.get_datafile(db, 110)
    db.close()
    ds = ViewOfDelft()
    ds.parse(110, datafile.path, datafile.name, datafile.config)
    ds._request_scoped = True
    ds.prepare_for_train(["radarPC"], train_cfg, model_cfg)

    net = NetworkFactory().get_instance("RadarPillar", dict(model_cfg, **{"class": "RadarPillar"}))
    net.eval()
    batch = radarpillar_collate([ds[0]])
    with torch.no_grad():
        res = postprocess(net(batch), net.net.anchors, model_cfg, train_cfg)[0]
    out = radarpillar_display_geometry(res, ds, 0)

    assert out["viz_idx"] == 0
    assert len(out["bev_boxes"]) > 0
    for row in out["bev_boxes"]:
        assert len(row) == 10
        xs, ys = row[0:8:2], row[1:8:2]
        # corners stay near the configured range (box diagonal margin)
        assert min(xs) > -6 and max(xs) < 51.2 + 6
        assert min(ys) > -25.6 - 6 and max(ys) < 25.6 + 6
    for row in out["image_boxes"]:
        assert len(row) == 18
        assert all(np.isfinite(v) for v in row)


def test_radarpillar_evaluation_writes_kitti_files_and_scores(tmp_path):
    """Adapter contract (slice 3): one KITTI prediction file per val frame with
    16-field lines (the devkit parser zeroes scores on any other width), devkit
    result keys, and the standard eval-function return shape. Untrained CPU net
    with a high confidence threshold keeps the detection set tiny."""
    import os, json, torch
    from torch.utils.data import DataLoader, Subset
    os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost/postgres")
    from detection_vis_backend.datasets.view_of_delft import ViewOfDelft
    from detection_vis_backend.networks.network import NetworkFactory
    from detection_vis_backend.train.evaluate import RadarPillar_evaluation, eval_func_dict
    from detection_vis_backend.train.utils import radarpillar_collate
    from data.database import get_session_local
    from data import crud

    assert callable(eval_func_dict["RadarPillar", "ViewOfDelft"])

    model_cfg = json.load(open("detection_vis_backend/networks/RadarPillar_model_config.json"))
    train_cfg = json.load(open("detection_vis_backend/train/RadarPillar_train_config.json"))
    # untrained net: keep detections sparse and NMS cheap
    train_cfg = dict(train_cfg, confidence_threshold=0.9, nms_pre_max=64, nms_post_max=8)

    db = get_session_local()
    datafile = crud.get_datafile(db, 110)
    db.close()
    ds = ViewOfDelft()
    ds.parse(110, datafile.path, datafile.name, datafile.config)
    ds._request_scoped = True
    ds.prepare_for_train(["radarPC"], train_cfg, model_cfg)

    net = NetworkFactory().get_instance("RadarPillar", dict(model_cfg, **{"class": "RadarPillar"}))
    net.init_lossfunc(train_cfg, "cpu")

    loader = DataLoader(Subset(ds, [0, 1, 2]), batch_size=2, collate_fn=radarpillar_collate)
    result = RadarPillar_evaluation(net, loader, train_cfg, ["radarPC"], str(tmp_path), "val", "cpu")

    assert set(result) >= {"mAP", "mAR", "mIoU", "loss"}
    pred_dir = tmp_path / "vod_eval_val"
    files = sorted(os.listdir(pred_dir))
    assert len(files) == 3, "one file per val frame, even if empty"
    for fn in files:
        assert len(os.path.splitext(fn)[0]) == 5 and fn.endswith(".txt")
        for line in (pred_dir / fn).read_text().splitlines():
            parts = line.split(" ")
            assert len(parts) == 16
            assert parts[0] in ("Car", "Pedestrian", "Cyclist")
            float(parts[15])


def test_vod_augmentation_transform_invariants():
    """Ports of fthbng77 augmentor_utils.py. The invariant that matters: a point
    inside a gt box stays inside after every world transform (points and boxes
    move together); radar feature columns (rcs, v_r, v_r_comp, time) are untouched."""
    from detection_vis_backend.datasets.vod_augmentation import (
        random_flip_along_x, global_rotation, global_scaling, VodAugmentor)

    rng = np.random.default_rng(0)
    # float32 like the real pipeline (the augmentor standardizes to float32,
    # and the identity check compares exactly)
    box = np.array([[10.0, 2.0, -0.5, 4.0, 1.8, 1.5, 0.4]], dtype=np.float32)
    # a point just inside the box + one far away
    pts = np.array([[10.5, 2.2, -0.4, 5.0, -1.0, 0.5, 0.9],
                    [30.0, -8.0, 0.0, 2.0, 0.0, 0.0, 0.0]], dtype=np.float32)

    def inside(box7, p):
        c, s = np.cos(-box7[6]), np.sin(-box7[6])
        dx, dy = p[0] - box7[0], p[1] - box7[1]
        lx, ly = c * dx - s * dy, s * dx + c * dy
        return abs(lx) <= box7[3] / 2 + 1e-6 and abs(ly) <= box7[4] / 2 + 1e-6

    for fn in (lambda b, p: random_flip_along_x(b, p, np.random.default_rng(1)),
               lambda b, p: global_rotation(b, p, [-0.785, 0.785], rng),
               lambda b, p: global_scaling(b, p, [0.95, 1.05], rng)):
        b2, p2 = fn(box.copy(), pts.copy())
        assert inside(b2[0], p2[0]), "in-box point escaped the transformed box"
        assert np.allclose(p2[:, 3:7], pts[:, 3:7]), "radar feature columns modified"

    # flip determinism: y and yaw negate together when the coin says flip
    b2, p2 = random_flip_along_x(box.copy(), pts.copy(), np.random.default_rng(1))
    if b2[0, 1] != box[0, 1]:
        assert b2[0, 1] == -box[0, 1] and b2[0, 6] == -box[0, 6] and p2[0, 1] == -pts[0, 1]

    # pipeline: disabled config is identity
    aug = VodAugmentor({"random_world_flip": False, "random_world_rotation": False,
                        "random_world_scaling": False})
    p3, b3 = aug(pts.copy(), box.copy())
    assert np.array_equal(p3, pts) and np.array_equal(b3, box)
