"""TransRAD port: backbone, neck/heads, loss, decoder, NMS, adapter, trainer additions.
Everything here runs on CPU with random tensors or the fake parser; the authors'
checkpoint is used only for key-set checks and skipped when absent."""
import os

import numpy as np
import pytest
import torch

from detection_vis_backend.networks import transrad as T

torch.manual_seed(0)
np.random.seed(0)

HAS_REF = os.path.exists(T.REF_CHECKPOINT)
needs_ref = pytest.mark.skipif(not HAS_REF, reason=f"authors' checkpoint not at {T.REF_CHECKPOINT}")


def test_drop_path_is_identity_in_eval_and_scales_in_train():
    dp = T.DropPath(0.5)
    x = torch.ones(8, 3, 2, 2)
    dp.eval()
    assert torch.equal(dp(x), x)
    dp.train()
    y = dp(x)
    kept = y[:, 0, 0, 0]
    assert set(kept.tolist()) <= {0.0, 2.0}          # survivors are rescaled by 1/(1-p)


def test_trunc_normal_stays_inside_bounds():
    t = torch.empty(10000)
    T.trunc_normal_(t, std=0.02, a=-0.04, b=0.04)
    assert t.min() >= -0.04 and t.max() <= 0.04
    assert abs(t.std().item() - 0.02) < 0.005


RMT_KW = dict(in_chans=256, out_indices=(0, 1, 2, 3), embed_dims=[32, 64, 128, 256],
              depths=[2, 2, 8, 2], num_heads=[4, 4, 8, 16], init_values=[2, 2, 2, 2],
              heads_ranges=[4, 4, 6, 6], mlp_ratios=[3, 3, 3, 3], drop_path_rate=0.1,
              norm_layer=torch.nn.LayerNorm, patch_norm=True, use_checkpoint=False,
              chunkwise_recurrents=[True, True, False, False], projection=1024,
              layerscales=[False, False, False, False], layer_init_values=1e-6, norm_eval=True)


def test_rmt_backbone_feature_shapes_at_full_size():
    net = T.RMT(**RMT_KW).eval()
    with torch.no_grad():
        f1, f2, f3 = net(torch.randn(1, 256, 256, 256))
    assert tuple(f1.shape) == (1, 64, 32, 32)
    assert tuple(f2.shape) == (1, 128, 16, 16)
    assert tuple(f3.shape) == (1, 256, 8, 8)


def test_rmt_train_mode_keeps_batchnorm_in_eval():
    """norm_eval=True: the reference freezes BatchNorm statistics during training."""
    net = T.RMT(**RMT_KW)
    net.train()
    bns = [m for m in net.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    assert bns and all(not m.training for m in bns)


@needs_ref
def test_rmt_keys_match_the_authors_backbone():
    sd = torch.load(T.REF_CHECKPOINT, map_location="cpu")
    ref = {k[len("backbone."):] for k in sd if k.startswith("backbone.")}
    ours = set(T.RMT(**RMT_KW).state_dict())
    assert ours == ref, f"missing {sorted(ref - ours)[:5]} unexpected {sorted(ours - ref)[:5]}"


BODY_KW = dict(input_size=[256, 256, 256], n_class=6)


def test_body_output_shapes_and_anchor_count():
    body = T.TransRADBody(**BODY_KW).eval()
    with torch.no_grad():
        dbox, cls, heads, anchors, strides = body(torch.randn(1, 256, 256, 256))
    assert body.no == 6 + 4 * 16 + 1 + 2
    assert [tuple(h.shape) for h in heads] == [(1, 73, 32, 32), (1, 73, 16, 16), (1, 73, 8, 8)]
    assert tuple(dbox.shape) == (1, 4, 1344) and tuple(cls.shape) == (1, 6, 1344)
    assert tuple(anchors.shape) == (2, 1344) and tuple(strides.shape) == (1, 1344)
    assert body.stride.tolist() == [8.0, 16.0, 32.0]


def test_body_strides_equal_a_dummy_forward():
    """The reference computes stride = input/feature-size with a dummy forward at
    construction; the port fixes it from the architecture and this pins the equality."""
    body = T.TransRADBody(**BODY_KW).eval()
    with torch.no_grad():
        feats = body.backbone(torch.zeros(1, 256, 256, 256))
    assert [256 / f.shape[-2] for f in feats] == body.stride.tolist()


@needs_ref
def test_body_state_dict_matches_the_authors_checkpoint_exactly():
    sd = torch.load(T.REF_CHECKPOINT, map_location="cpu")
    body = T.TransRADBody(**BODY_KW)
    assert set(body.state_dict()) == set(sd)
    assert {k: tuple(v.shape) for k, v in body.state_dict().items()} == {k: tuple(v.shape) for k, v in sd.items()}
    body.load_state_dict(sd, strict=True)


def _one_target_batch():
    # one car in frame 0, normalised xyzwhd; frame 1 empty
    return torch.tensor([[0, 2, 100 / 256, 120 / 256, 30 / 64, 10 / 256, 12 / 256, 6 / 64]], dtype=torch.float32)


def test_loss_is_finite_with_nine_components():
    body = T.TransRADBody(**BODY_KW).eval()
    crit = T.TransRADLoss(body)
    with torch.no_grad():
        preds = body(torch.randn(2, 256, 256, 256))
    total, parts = crit(preds, _one_target_batch())
    assert torch.isfinite(total) and total > 0
    assert tuple(parts.shape) == (9,) and torch.isfinite(parts).all()
    assert torch.isclose(parts.sum(), total)


def test_loss_survives_a_batch_with_no_objects():
    body = T.TransRADBody(**BODY_KW).eval()
    crit = T.TransRADLoss(body)
    with torch.no_grad():
        preds = body(torch.randn(1, 256, 256, 256))
    total, parts = crit(preds, torch.zeros(0, 8))
    assert torch.isfinite(total)
    assert parts[0] == 0 and parts[2] == 0      # no positives: box and dfl terms are zero


def test_torch_iou3d_matches_the_numpy_version_used_by_evaluation():
    from detection_vis_backend.train.utils import iou3d_np
    a = torch.tensor([[100., 120., 30., 10., 12., 6.]])
    b = torch.tensor([[102., 121., 31., 10., 12., 6.], [10., 10., 10., 4., 4., 4.]])
    size = torch.tensor([256., 256., 64.])
    got = T.torch_iou3d(a.expand(2, 6), b, size)
    want = iou3d_np(a.numpy().repeat(2, 0), b.numpy(), [256, 256, 64])
    np.testing.assert_allclose(got.numpy(), want, rtol=1e-5)


def _rows(*boxes):
    """[x,y,z,w,h,d, obj, scores over 6 classes]"""
    return np.array(boxes, dtype=np.float32)


def test_nms_drops_lower_scoring_overlap_of_another_class_and_keeps_distant_box():
    car = [100, 120, 30, 10, 12, 6, 0.9, 0, 0, 0.9, 0, 0, 0]
    truck_same_spot = [101, 121, 30, 10, 12, 6, 0.8, 0, 0, 0, 0, 0, 0.8]
    person_far = [20, 20, 10, 4, 4, 4, 0.7, 0.7, 0, 0, 0, 0, 0]
    out = T.nms_class_then_location(_rows(car, truck_same_spot, person_far), [256, 256, 64],
                                    confidence=0.3, nms_iou=0.1, la_iou=0.1)
    assert sorted(out[:, 7].tolist()) == [0.0, 2.0]        # truck removed, car and person kept
    assert out[out[:, 7] == 2][0, 6] == pytest.approx(0.9)  # score column is the class score


def test_nms_on_nothing_above_threshold_returns_empty_8_columns():
    weak = [100, 120, 30, 10, 12, 6, 0.1, 0, 0, 0.9, 0, 0, 0]
    out = T.nms_class_then_location(_rows(weak), [256, 256, 64], 0.3, 0.1, 0.1)
    assert out.shape == (0, 8)


def test_decode_recovers_a_box_planted_in_the_head_maps():
    """Plant one anchor's distribution and points so that decoding yields a known box."""
    body = T.TransRADBody(**BODY_KW).eval()
    with torch.no_grad():
        dbox, cls, heads, anchors, strides = body(torch.randn(1, 256, 256, 256))
    # pick anchor 0 of P3: cell (0,0), stride 8, anchor point (0.5, 0.5) in grid units
    reg_max, no = body.reg_max, body.no
    P3 = torch.zeros_like(heads[0])
    dist = torch.full((4, reg_max), -20.)             # one-hot-ish softmax
    for k, target in enumerate([2, 3, 2, 3]):         # l, t, r, b in grid units
        dist[k, target] = 20.
    P3[0, :4 * reg_max, 0, 0] = dist.flatten()
    P3[0, 4 * reg_max + 2, 0, 0] = 1.0                # class 2 score (the head's sigmoid already applied)
    P3[0, 4 * reg_max + 6, 0, 0] = 1.0                # objectness
    P3[0, 4 * reg_max + 7, 0, 0] = 10 / 64            # z1 (fraction of eval doppler size)
    P3[0, 4 * reg_max + 8, 0, 0] = 20 / 64            # z2
    heads = [P3, torch.zeros_like(heads[1]), torch.zeros_like(heads[2])]
    dbox = body.dfl(torch.cat([h.view(1, no, -1) for h in heads], 2)[:, :4 * reg_max])
    rows = T.decode_predictions((dbox, None, heads, anchors, strides), 6, [256, 256, 64])
    r = rows[0, 0]
    # x = (0.5 - 2 + 0.5 + 2)/2 * 8 = 4, w = (2+2)*8 = 32; y: (0.5-3 + 0.5+3)/2*8 = 4, h = 48
    np.testing.assert_allclose(r[:6], [4.0, 4.0, 15.0, 32.0, 48.0, 10.0], atol=1e-4)
    assert r[6] == pytest.approx(1.0) and r[7 + 2] == pytest.approx(1.0)


def test_decode_follows_the_body_reg_max_not_a_constant():
    """reg_max is an editable model-config key; the decoder must read the head layout
    from the maps it is given rather than assume 16."""
    body = T.TransRADBody(input_size=[256, 256, 256], n_class=6, reg_max=8).eval()
    with torch.no_grad():
        out = body(torch.randn(1, 256, 256, 256))
    rows = T.decode_predictions(out, 6, [256, 256, 64])
    assert rows.shape == (1, 1344, 13)


def test_decode_accepts_the_network_dict_output():
    body = T.TransRADBody(**BODY_KW).eval()
    with torch.no_grad():
        dbox, cls, heads, anchors, strides = body(torch.randn(1, 256, 256, 256))
    as_tuple = T.decode_predictions((dbox, cls, heads, anchors, strides), 6, [256, 256, 64])
    as_dict = T.decode_predictions({"dbox": dbox, "cls": cls, "heads": heads, "anchors": anchors, "strides": strides}, 6, [256, 256, 64])
    assert as_tuple.shape == (1, 1344, 13)
    np.testing.assert_array_equal(as_tuple, as_dict)


# ---------------------------------------------------------------------------------------
# Adapter and collate, against the fake parser shared with the other RADDet adapters.
# ---------------------------------------------------------------------------------------
from detection_vis_backend.tests.test_sample_adapters import FakeRADDet, ONE_CAR, EMPTY, WRAP, DATA_STATS  # noqa: E402
from detection_vis_backend.datasets.adapters import make_sample_adapter  # noqa: E402

TRANSRAD_MODEL_CFG = {"class": "TransRAD", "input_size": [256, 256, 256], "n_class": 6}


def test_transrad_sample_layout():
    a = make_sample_adapter("TransRAD", FakeRADDet([ONE_CAR]), ["RAD"], TRANSRAD_MODEL_CFG, {}, "val")
    s = a[0]
    assert set(s) == {"RAD", "label", "boxes", "image_path"}
    assert s["RAD"].shape == (256, 256, 256) and s["RAD"].dtype == np.float32
    assert s["label"].shape == (1, 7) and s["label"].dtype == np.float32
    assert s["boxes"].shape == (1, 7) and s["boxes"].dtype == np.float64
    np.testing.assert_allclose(s["label"][0], [2, 100 / 256, 120 / 256, 30 / 64, 10 / 256, 12 / 256, 6 / 64])
    np.testing.assert_allclose(s["boxes"][0], [100, 120, 30, 10, 12, 6, 2])


def test_transrad_resize_is_the_reference_nearest_zoom():
    from scipy.ndimage import zoom
    from detection_vis_backend.datasets.utils import complexTo2Channels
    p = FakeRADDet([ONE_CAR])
    s = make_sample_adapter("TransRAD", p, ["RAD"], TRANSRAD_MODEL_CFG, {}, "val")[0]
    cube = (complexTo2Channels(p.get_RAD(0)) - DATA_STATS["global_mean_log"]) / DATA_STATS["global_variance_log"]
    want = np.transpose(zoom(cube.astype(np.float32), (1, 1, 4), order=0), (2, 0, 1))
    np.testing.assert_array_equal(s["RAD"], want)
    # nearest neighbour: every output Doppler slice is one of the 64 input slices
    inp = np.transpose(cube.astype(np.float32), (2, 0, 1))
    assert all(any(np.array_equal(s["RAD"][k], inp[j]) for j in range(64)) for k in (0, 1, 128, 255))


def test_transrad_edge_box_is_clamped_not_split():
    a = make_sample_adapter("TransRAD", FakeRADDet([WRAP]), ["RAD"], TRANSRAD_MODEL_CFG, {}, "val")
    s = a[0]
    assert s["label"].shape == (1, 7)
    x, w = s["boxes"][0, 2], s["boxes"][0, 5]           # azimuth centre/width after clamp
    assert x - w / 2 >= 0 and x + w / 2 <= 256
    # WRAP box: x1 = 2 - 5 < 0 -> x1 += 256 -> keeps [0, 7] (the larger part), per clamp_box
    np.testing.assert_allclose([x - w / 2, x + w / 2], [0, 7])


def test_clamp_box_edges_rule_matches_reference_cases():
    # [x1, y1, z1, x2, y2, z2] after centre/size conversion; y wraps at 256, z at 64
    boxes = np.array([[10, -3, 5, 20, 4, 9],     # y1 < 0: y1 -> 253, (256-253)=3 < (4-0)=4 -> [0, 4]
                      [10, 250, 5, 20, 258, 9],  # y2 >= 256: y2 -> 2, (256-250)=6 >= 2 -> [250, 256]
                      [10, 100, 62, 20, 110, 66]], dtype=float)  # z2 >= 64: z2 -> 2, (64-62)=2 >= 2 -> [62, 64]
    xyzwhd = T.xyzxyz2xyzwhd(boxes)
    out = T.xyzxyz2xyzwhd(T.xyzwhd2xyzxyz(xyzwhd))   # round trip sanity
    np.testing.assert_allclose(out, xyzwhd)
    from detection_vis_backend.datasets.adapters.raddet import clamp_box_edges
    got = T.xyzwhd2xyzxyz(clamp_box_edges(xyzwhd, 256, 64))
    np.testing.assert_allclose(got[:, [1, 4]], [[0, 4], [250, 256], [100, 110]])
    np.testing.assert_allclose(got[:, [2, 5]], [[5, 9], [5, 9], [62, 64]])


def test_transrad_empty_frame():
    s = make_sample_adapter("TransRAD", FakeRADDet([EMPTY]), ["RAD"], TRANSRAD_MODEL_CFG, {}, "val")[0]
    assert s["label"].shape == (0, 7) and s["boxes"].shape == (0, 7)


def test_transrad_collate_concatenates_labels_with_batch_index_and_pads_boxes():
    from detection_vis_backend.train.utils import TransRAD_collate
    a = make_sample_adapter("TransRAD", FakeRADDet([ONE_CAR, EMPTY, ONE_CAR]), ["RAD"], TRANSRAD_MODEL_CFG, {}, "val")
    b = TransRAD_collate([a[0], a[1], a[2]])
    assert tuple(b["RAD"].shape) == (3, 256, 256, 256)
    assert tuple(b["label"].shape) == (2, 8) and b["label"][:, 0].tolist() == [0.0, 2.0]
    assert tuple(b["boxes"].shape) == (3, 1, 7) and b["boxes"][1].abs().sum() == 0
    assert b["image_path"] == ["/img/0.jpg", "/img/1.jpg", "/img/2.jpg"]


# ---------------------------------------------------------------------------------------
# Trainer additions: cosine-warmup schedule, weight EMA, config invariants.
# ---------------------------------------------------------------------------------------
def test_cos_warmup_matches_the_reference_schedule():
    """REF/nets/yolo_training.py yolox_warm_cos_lr with the Train.py arguments."""
    from detection_vis_backend.train.train import cos_warmup_lr
    import math as m
    lr, mn, total, wu, wu0, hold = 3e-4, 3e-6, 100, 3, 3e-5, 5
    assert cos_warmup_lr(lr, mn, total, wu, wu0, hold, 0) == pytest.approx(wu0)
    assert cos_warmup_lr(lr, mn, total, wu, wu0, hold, 3) == pytest.approx(lr)
    mid = mn + 0.5 * (lr - mn) * (1 + m.cos(m.pi * (50 - 3) / (100 - 3 - 5)))
    assert cos_warmup_lr(lr, mn, total, wu, wu0, hold, 50) == pytest.approx(mid)
    assert cos_warmup_lr(lr, mn, total, wu, wu0, hold, 99) == pytest.approx(mn)


def test_weight_ema_moves_toward_the_model_with_ramped_decay():
    from detection_vis_backend.train.train import WeightEMA
    net = torch.nn.Linear(2, 1)
    with torch.no_grad():
        net.weight.fill_(1.0); net.bias.fill_(0.0)
    ema = WeightEMA(net, decay=0.9999, tau=2000)
    with torch.no_grad():
        net.weight.fill_(3.0)
    ema.update(net)
    d = 0.9999 * (1 - np.exp(-1 / 2000))
    assert ema.updates == 1
    np.testing.assert_allclose(ema.ema.weight.detach().numpy(), 1.0 * d + 3.0 * (1 - d), rtol=1e-6)
    assert not any(p.requires_grad for p in ema.ema.parameters())


def test_existing_configs_do_not_enable_the_new_options():
    import json
    for name in ("RADDet_train_config_reference.json", "DAROD_train_config.json"):
        with open(os.path.join("detection_vis_backend", "train", name)) as f:
            cfg = json.load(f)
        node = cfg.get("RADDetDataset", cfg.get("default", cfg))
        assert "ema" not in node and "betas" not in node.get("optimizer", {})
        assert node.get("lr_scheduler", {}).get("type") != "cos_warmup"


def test_transrad_train_config_carries_the_reference_recipe():
    import json
    with open("detection_vis_backend/train/TransRAD_train_config.json") as f:
        cfg = json.load(f)
    t = dict(cfg["default"]); t.update(cfg["RADDetDataset"])
    assert t["optimizer"] == {"type": "Adam", "lr": 0.0003, "betas": [0.937, 0.999], "weight_decay": 0}
    assert t["lr_scheduler"]["type"] == "cos_warmup" and t["ema"] == {"decay": 0.9999, "tau": 2000}
    assert t["test_checkpoint"] == {"monitor": "loss", "mode": "min"} and t["grad_norm_clip"] == 10.0
    assert t["dataloader"]["train"] == {"batch_size": 4, "num_workers": 4, "drop_last": True}
    assert t["eval_input_size"] == [256, 256, 64] and t["mAP_iou3d_threshold"] == 0.3


# ---------------------------------------------------------------------------------------
# Network class, registry entries and evaluator.
# ---------------------------------------------------------------------------------------
def test_transrad_network_through_the_factory_and_loss_contract():
    import json
    from detection_vis_backend.networks.network import NetworkFactory
    with open("detection_vis_backend/networks/TransRAD_model_config.json") as f:
        cfg = json.load(f)
    net = NetworkFactory().get_instance("TransRAD", cfg).eval()
    net.init_lossfunc({}, torch.device("cpu"))
    with torch.no_grad():
        out = net(torch.randn(1, 256, 256, 256))
    assert set(out) == {"dbox", "cls", "heads", "anchors", "strides"}
    loss = net.get_loss(out, {"label": _one_target_batch()[:1], "boxes": torch.zeros(1, 1, 7)}, {})
    assert torch.isfinite(loss) and tuple(net.last_loss_components.shape) == (9,)
    assert net.n_class == 6 and net.class_names[2] == "car"


@needs_ref
def test_transrad_network_loads_the_authors_checkpoint_without_a_key_map():
    import json
    from detection_vis_backend.networks.network import NetworkFactory
    with open("detection_vis_backend/networks/TransRAD_model_config.json") as f:
        cfg = json.load(f)
    net = NetworkFactory().get_instance("TransRAD", cfg)
    sd = torch.load(T.REF_CHECKPOINT, map_location="cpu")
    net.load_state_dict(sd, strict=True)
    assert set(net.state_dict()) == set(sd)


def test_transrad_is_registered_everywhere_the_app_looks():
    import json
    from detection_vis_backend.train.evaluate import eval_func_dict, TransRAD_evaluation
    from detection_vis_backend.train.train import CreateDataLoaders
    import inspect
    assert eval_func_dict["TransRAD", "RADDetDataset"] is TransRAD_evaluation
    assert "'TransRAD': TransRAD_collate" in inspect.getsource(CreateDataLoaders)
    with open("detection_vis_backend/networks/model_features.json") as f:
        feats = json.load(f)["TransRAD"]
    assert feats["features"] == ["RAD"] and feats["datasets"] == ["RADDetDataset"] and feats["input_size"] == [256, 256, 256]


def test_transrad_configs_resolve_through_smoke_sweep():
    import sys
    sys.path.insert(0, "scripts")
    from smoke_sweep import resolve_configs
    m, t = resolve_configs("TransRAD", "RADDetDataset", ["RAD"])
    assert m["class"] == "TransRAD" and m["input_size"] == [256, 256, 256] and m["n_class"] == 6
    assert t["lr_scheduler"]["type"] == "cos_warmup" and t["eval_input_size"] == [256, 256, 64]
    assert not any(k.startswith("_") for k in t)


def test_transrad_evaluation_scores_a_perfect_prediction_at_one():
    """Feed the evaluator a fake net whose decode yields exactly the gt box."""
    from detection_vis_backend.train import evaluate as E

    class FakeNet(torch.nn.Module):
        n_class = 6
        class_names = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
        def forward(self, x):
            return {"fake": True}
        def get_loss(self, pred, data, cfg):
            return torch.tensor(0.5)
    rows = np.zeros((1, 1, 13), dtype=np.float32)
    rows[0, 0, :6] = [100, 120, 30, 10, 12, 6]; rows[0, 0, 6] = 0.9; rows[0, 0, 7 + 2] = 0.9
    batch = {"RAD": torch.zeros(1, 2, 2, 2), "label": torch.zeros(0, 8),
             "boxes": torch.tensor([[[100., 120., 30., 10., 12., 6., 2.]]]), "image_path": ["x"]}
    cfg = {"eval_input_size": [256, 256, 64], "confidence_threshold": 0.3, "nms_iou3d_threshold": 0.1,
           "la_nms_iou3d_threshold": 0.1, "mAP_iou3d_threshold": 0.3, "mAP_iou3d_threshold_list": [0.3, 0.5],
           "dataloader": {"val": {"batch_size": 1}}}

    class Loader(list):
        dataset = [0]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(E, "transrad_decode", lambda outputs, n_class, size: rows)
        res = E.TransRAD_evaluation(FakeNet(), Loader([batch]), cfg, ["RAD"], None, "val", torch.device("cpu"))
    assert res["mAP"] == pytest.approx(1.0) and res["mAP@0.5"] == pytest.approx(1.0)
    assert res["AP@0.3_car"] == pytest.approx(1.0) and res["loss"] == pytest.approx(0.5)


def test_transrad_evaluation_scores_a_frame_without_ground_truth_as_zero_not_nan():
    """The adapter yields (0, 7) boxes for an object-free frame; the evaluator must count
    it as a frame scoring 0 (RADDet's convention), not poison the mean with NaN."""
    from detection_vis_backend.train import evaluate as E

    class FakeNet(torch.nn.Module):
        n_class = 6
        class_names = ["person", "bicycle", "car", "motorcycle", "bus", "truck"]
        def forward(self, x):
            return {"fake": True}
    rows = np.zeros((2, 1, 13), dtype=np.float32)
    rows[:, 0, :6] = [100, 120, 30, 10, 12, 6]; rows[:, 0, 6] = 0.9; rows[:, 0, 7 + 2] = 0.9
    boxes = torch.zeros(2, 1, 7)
    boxes[0, 0] = torch.tensor([100., 120., 30., 10., 12., 6., 2.])   # frame 1 has no gt
    batch = {"RAD": torch.zeros(2, 2, 2, 2), "label": torch.zeros(0, 8), "boxes": boxes, "image_path": ["a", "b"]}
    cfg = {"eval_input_size": [256, 256, 64], "confidence_threshold": 0.3, "nms_iou3d_threshold": 0.1,
           "la_nms_iou3d_threshold": 0.1, "mAP_iou3d_threshold": 0.3, "mAP_iou3d_threshold_list": [0.3],
           "dataloader": {"test": {"batch_size": 2}}}

    class Loader(list):
        dataset = [0, 1]
    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(E, "transrad_decode", lambda outputs, n_class, size: rows)
        res = E.TransRAD_evaluation(FakeNet(), Loader([batch]), cfg, ["RAD"], None, "test", torch.device("cpu"))
    assert np.isfinite(res["mAP"]) and res["mAP"] == pytest.approx(0.5)
