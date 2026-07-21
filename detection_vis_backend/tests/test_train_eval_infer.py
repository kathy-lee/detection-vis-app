import os
import json
import inspect
import pytest
import numpy as np
import torch
from unittest.mock import MagicMock

from detection_vis_backend.train.train import train, CreateDataLoaders
from detection_vis_backend.train.evaluate import FFTRadNet_evaluation
from detection_vis_backend.train.inference import infer, carrada_seg_readout
from detection_vis_backend.train.utils import post_process_single_frame
from detection_vis_backend.datasets.utils import flip_horizontal, flip_vertical
from detection_vis_backend.datasets.dataset import CARRADA


def test_carrada_seg_readout_argmax():
    """CARRADA RECORD/MVRECORD inference is a segmentation readout: argmax the class channels of
    the model output and of the (one-hot) gt mask, mirroring RECORD/MVRECORD_CARRADA_evaluation.
    """
    n_class, H, W = 4, 3, 3
    logits = np.zeros((n_class, H, W))
    logits[2, 0, 0] = 5.0   # cyclist at (0,0)
    logits[3, 1, 1] = 5.0   # car at (1,1)
    gt = np.zeros((n_class, H, W))
    gt[1, 0, 0] = 1.0       # pedestrian at (0,0)
    gt[3, 2, 2] = 1.0       # car at (2,2)

    out = carrada_seg_readout(logits, gt, viz_idx=7)

    assert out["viz_idx"] == 7
    pred = np.array(out["pred_mask"])
    gtm = np.array(out["gt_mask"])
    assert pred.shape == (H, W) and gtm.shape == (H, W)
    # argmax picks the class channel; untouched pixels stay background (0)
    assert pred[0, 0] == 2 and pred[1, 1] == 3 and pred[0, 1] == 0
    assert gtm[0, 0] == 1 and gtm[2, 2] == 3 and gtm[0, 1] == 0


def test_carrada_seg_readout_accepts_label_gt():
    """gt may already be a 2-D label map (H,W) rather than one-hot; it should pass through."""
    logits = np.zeros((4, 2, 2))
    logits[1, 0, 0] = 1.0
    labels = np.array([[1, 0], [0, 3]])
    out = carrada_seg_readout(logits, labels, viz_idx=0)
    assert np.array(out["gt_mask"]).tolist() == [[1, 0], [0, 3]]


def test_post_process_requires_batchless_confmap():
    """Regression test for the MVRECORD inference crash `too many values to unpack (expected 3)`.

    MVRECORD.forward returns per-view maps of shape (B, n_class, H, W) (4-D), but the inference
    branch fed them straight into post_process_single_frame, which does
    `class_size, height, width = confmaps.shape` and therefore requires a 3-D (n_class, H, W)
    map. The 4-D array made that unpack raise. Every other branch first indexes output[0]; the
    MVRECORD branch now does the same (output['ra'][0] / output['rd'][0]).

    This pins the contract: a batched (4-D) map raises the exact reported error, while the
    batch-indexed (3-D) slice processes fine.
    """
    n_class, H, W = 4, 8, 8
    train_cfg = {"max_dets": 20, "peak_thres": 0.3, "ols_thres": 0.3}
    rng_grid = np.zeros(H)
    agl_grid = np.zeros(W)
    # all-zeros => no peaks clear peak_thres, so lnms/get_ols_btw_objects never runs and this
    # stays a pure shape/contract test, isolated from the CARRADA class-id question.
    confmap = np.zeros((n_class, H, W))

    # The bug: the un-indexed MVRECORD output is (B, n_class, H, W) and blows up the 3-way unpack.
    batched = confmap[np.newaxis, ...]  # (1, 4, 8, 8)
    with pytest.raises(ValueError, match="too many values to unpack"):
        post_process_single_frame(batched, train_cfg, n_class, rng_grid, agl_grid)

    # The fix: output[...][0] hands post_process the 3-D map it expects.
    res = post_process_single_frame(confmap, train_cfg, n_class, rng_grid, agl_grid)
    assert res.shape == (train_cfg["max_dets"], 4)


def test_flip_helpers_return_torch_convertible_arrays():
    """Regression test for the DAROD+RADDet inference crash "At least one stride in the given
    numpy array is negative, and tensors with negative strides are not currently supported".

    RADDet.transform flips via flip_horizontal/flip_vertical. Both returned raw np.fliplr/
    np.flipud *views*, which carry negative strides, and inference.py then does
    torch.tensor(data['RD']) -> torch raises. Every other flip helper in datasets/utils.py
    (HFlip/VFlip) already ended with .copy(); these two did not.

    This pins that the helpers hand back arrays torch will actually accept.
    """
    img = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    boxes = np.array([[0.1, 0.2, 0.3, 0.4]])

    for flip in (flip_horizontal, flip_vertical):
        flipped, _ = flip(img, boxes)
        assert min(flipped.strides) > 0, f"{flip.__name__} returned negative strides"
        # the operative check: this is the exact call that raised in infer()
        torch.tensor(np.expand_dims(flipped, axis=0)).unsqueeze(0).float()

    # and the flip itself is still correct (copy must not change semantics)
    assert np.array_equal(flip_horizontal(img, boxes)[0], np.fliplr(img))
    assert np.array_equal(flip_vertical(img, boxes)[0], np.flipud(img))


def test_rca_prepare_for_train_does_not_clobber_parse_annotations(tmp_path):
    """Regression test for "list indices must be integers or slices, not str" from get_label.

    parse() loads annotations_frame_oriented.json into self.annos, shaped
        annos[seq][frame] = {obj_id: {'range_angle': {'label': int, 'box': [[..],[..]]}}}
    but prepare_for_train()'s RadarCrossAttention branch loaded new_gt_anno.json -- shaped
        annos[seq][frame] = {'cls': [3.0], 'idx': [92.0, 6.0, 132.0, 26.0]}
    -- over the SAME attribute. get_label() still assumed the frame-oriented schema, so
    `obj['range_angle']` indexed a list with a string and raised. It only surfaced on frames
    that actually had objects, because `if objs:` skips empty ones. DatasetFactory caches one
    instance per file_id, so a single /predict poisoned /feature for the rest of the process.

    The RCA schema now lives in self.rca_annos. This pins that configuring for RCA leaves
    parse()'s annotations -- and therefore get_label -- intact.
    """
    seq = "2019-09-16-12-52-12"
    # parse() now keeps only its own sequence's slice of the annotation file
    # (annos[frame] = {...}, no sequence level), so build that shape directly.
    frame_oriented = {"000066": {"000000": {
        "range_angle": {"label": 3, "box": [[10, 20], [30, 40]]},
        "range_doppler": {"label": 3, "box": [[10, 20], [30, 40]]},
    }}}
    # the real new_gt_anno.json shape, verified against the dataset on disk
    rca_anno = {seq: {"tracks": [[0, 1, 2]], "000066": {"cls": [3.0], "idx": [92.0, 6.0, 132.0, 26.0]}}}
    (tmp_path / "new_gt_anno.json").write_text(json.dumps(rca_anno))

    ds = CARRADA()
    ds.root_path = str(tmp_path)
    ds.seq_name = seq
    ds.annos = frame_oriented          # as parse() would leave it
    ds.frame_sync = 3
    # prepare_for_train now guards against running on the shared cached instance
    # (state-ownership boundary; see test_dataset_state.py). This test plays the
    # role of a request copy, so mark it as one.
    ds._request_scoped = True

    ds.prepare_for_train(
        ["RA"],
        {"win_size": 1, "gauss_type": "Bivar"},
        {"class": "RadarCrossAttention", "center_offset": False, "orientation": False},
    )

    # the bug: this call raised TypeError because self.annos had been replaced
    gt = ds.get_label("RA", 66)
    assert gt == [[20, 10, 40, 30, "car"]]
    # both schemas coexist rather than one overwriting the other
    assert ds.annos is frame_oriented
    # rca_annos is likewise the per-sequence slice now
    assert ds.rca_annos["000066"] == {"cls": [3.0], "idx": [92.0, 6.0, 132.0, 26.0]}


def test_evaluate_module_calls_match_signatures():
    """Regression test for "accumulate() got an unexpected keyword argument 'log'".

    Commit 472e9fd ("replace print() with logging", 2023-12-19) turned `if log: print(...)` into
    `logger.debug(...)` and dropped the now-useless `log` param from evaluate_img AND accumulate.
    It updated evaluate_img's call sites but missed all five of accumulate's, which kept passing
    log=False -- so accumulate was uncallable for ~2.5 years. Nothing caught it because these
    eval paths only run during training of specific models; RAMP_CNN was simply the first one
    exercised.

    Rather than pin the single fixed kwarg, walk evaluate.py and assert no intra-module call
    passes a keyword its target doesn't accept, so the next partial refactor fails here loudly
    instead of waiting to surface mid-training run.
    """
    import ast

    src_path = os.path.join(os.path.dirname(__file__), "..", "train", "evaluate.py")
    tree = ast.parse(open(src_path).read())

    # module-level def name -> accepted kwarg names (None == accepts **kwargs, skip checking)
    sigs = {}
    for node in tree.body:
        if isinstance(node, ast.FunctionDef):
            if node.args.kwarg is not None:
                sigs[node.name] = None
            else:
                sigs[node.name] = {a.arg for a in node.args.args + node.args.kwonlyargs}

    violations = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Name):
            continue
        allowed = sigs.get(node.func.id, "unknown")
        if allowed in ("unknown", None):
            continue
        for kw in node.keywords:
            if kw.arg is not None and kw.arg not in allowed:
                violations.append(f"line {node.lineno}: {node.func.id}() got unexpected kwarg '{kw.arg}'")

    assert not violations, "evaluate.py calls disagree with definitions:\n" + "\n".join(violations)


def _carrada_datafile(fid):
    os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost/postgres")
    os.environ.setdefault("TMP_ROOTDIR", "/home/kangle/dataset/tmp")
    from data import crud
    from data.database import get_session_local
    db = get_session_local()
    try:
        f = crud.get_datafile(db, fid)
        return {"id": f.id, "name": f.name, "path": f.path, "parse": f.parse, "config": f.config}
    finally:
        db.close()


def _sequence_split_config(split_sequence):
    return {
        "win_size": 5, "norm_type": "local", "transformations": [],
        "dataloader": {
            "splitmode": "sequence",
            "split_sequence": split_sequence,
            "train": {"batch_size": 2, "num_workers": 0},
            "val": {"batch_size": 2, "num_workers": 0},
            "test": {"batch_size": 2, "num_workers": 0},
        },
    }


def test_sequence_split_with_empty_test_bucket(tmp_path):
    """Regression test: "AssertionError: datasets should not be an empty iterable".

    Sequence splitmode assigns whole files to splits via the UI's per-file radios. With
    two files the user filled train and val, leaving test as an empty list -- and
    CreateDataLoaders looped over ALL keys of split_sequence, building ConcatDataset([])
    for the empty bucket. An empty test split is a legitimate choice (train on one file,
    validate on the other); it must mean "no test evaluation", not a crash.
    """
    from detection_vis_backend.datasets.dataset import DatasetFactory

    f50, f22 = _carrada_datafile(50), _carrada_datafile(22)
    factory = DatasetFactory()
    for f in (f50, f22):  # the worker's _parse_datafiles does this before train()
        factory.get_shared_instance(f["parse"], f["id"],
                                    parse_args=(f["id"], f["path"], f["name"], f["config"]))

    train_config = _sequence_split_config({"train": [f50], "val": [f22], "test": []})
    train_loader, val_loader, test_loader = CreateDataLoaders(
        [f50, f22], ["RA"], {"class": "RECORD"}, train_config, str(tmp_path))

    assert len(train_loader.dataset) > 0
    assert len(val_loader.dataset) > 0
    assert test_loader is None, "empty test split should mean no test loader, not a crash"


def test_sequence_split_requires_train_and_val(tmp_path):
    """An empty train or val bucket is nonsense (nothing to train on / validate with)
    and must fail with a message naming the split, not an opaque assert five layers down.
    """
    f50, f22 = _carrada_datafile(50), _carrada_datafile(22)
    train_config = _sequence_split_config({"train": [f50, f22], "val": [], "test": []})
    with pytest.raises(ValueError, match="val"):
        CreateDataLoaders([f50, f22], ["RA"], {"class": "RECORD"}, train_config, str(tmp_path))


def test_train_accepts_progress_cb():
    params = inspect.signature(train).parameters
    assert "progress_cb" in params
    assert params["progress_cb"].default is None


@pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1",
                    reason="requires GPU + CRUW dataset; run with RUN_GPU_TESTS=1")
def test_train_calls_progress_cb_per_epoch():
    here = os.path.dirname(__file__)
    with open(os.path.join(here, "sample_for_train_test.json")) as f:
        sample = json.load(f)
    cb = MagicMock()
    train(sample["datafiles"], sample["features"], sample["model_config"],
          sample["train_config"], progress_cb=cb)
    assert cb.call_count == int(sample["train_config"]["num_epochs"])

@pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1",
                    reason="GPU + VoD data; run with RUN_GPU_TESTS=1")
def test_radarpillar_overfits_one_batch():
    """Slice-1 exit gate: a few hundred steps on ONE real
    batch must collapse the loss. Catches assignment/encoding/ordering bugs
    before any real training run."""
    import json
    import torch
    from detection_vis_backend.datasets.view_of_delft import ViewOfDelft
    from detection_vis_backend.networks.network import NetworkFactory
    from detection_vis_backend.train.utils import radarpillar_collate
    from data.database import get_session_local
    from data import crud

    os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost/postgres")
    with open("detection_vis_backend/networks/RadarPillar_model_config.json") as f:
        model_cfg = json.load(f)
    with open("detection_vis_backend/train/RadarPillar_train_config.json") as f:
        train_cfg = json.load(f)

    db = get_session_local()
    datafile = crud.get_datafile(db, 110)
    db.close()
    ds = ViewOfDelft()
    ds.parse(110, datafile.path, datafile.name, datafile.config)
    ds._request_scoped = True
    ds.prepare_for_train(["radarPC"], train_cfg, model_cfg)
    # pick 4 frames that actually contain gt
    samples = []
    for i in range(60):
        s = ds[i]
        if len(s["gt_boxes"]):
            samples.append(s)
        if len(samples) == 4:
            break
    assert len(samples) == 4, "need 4 frames with objects in the first 60"
    batch = radarpillar_collate(samples)

    device = "cuda"
    net = NetworkFactory().get_instance("RadarPillar", dict(model_cfg, **{"class": "RadarPillar"})).to(device)
    net.init_lossfunc(train_cfg, device)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    first = None
    for step in range(300):
        opt.zero_grad()
        loss = net.get_loss(net(batch), batch, train_cfg)
        if first is None:
            first = loss.item()
        loss.backward()
        opt.step()
    final = loss.item()
    assert final < 0.10 * first, f"loss did not collapse: {first:.3f} -> {final:.3f}"


@pytest.mark.skipif(os.getenv("RUN_GPU_TESTS") != "1",
                    reason="GPU + VoD data; run with RUN_GPU_TESTS=1")
def test_radarpillar_overfit_model_reproduces_gt():
    """Slice-2 exit gate: the overfit model, run through decode + direction
    correction + NMS, must reproduce its own memorized gt.

    Criteria (revised 2026-07-20, see below):
      1. every gt has a confident same-class prediction (score > 0.5);
      2. every gt's best IoU > 0.1 -- catches boxes decoded to the wrong place;
      3. per-class MEAN IoU clears the official devkit's own 3D thresholds
         (Car 0.5, Ped/Cyc 0.25 -- vod_evaluation/kitti_official_evaluate.py:694-696).

    The original criterion was `worst IoU across all ~28 gts > 0.7`, which measured
    initialization luck rather than correctness. Measured 2026-07-20 on this batch:
    the 4.8M slice-3.5 baseline -- the config that scored a REAL 52.27 mAP over 80
    epochs -- passed only 1 of 3 runs at the original 800 steps (failures at IoU 0.64
    and 0.66), and 0 of 3 at 2000 steps (0.67, 0.38, 0.57). A single 0.8x0.6m
    pedestrian, where ~15cm of offset halves the IoU, decided the outcome each time.
    Per-class means on the same runs were healthy throughout (Car 0.88-0.99,
    Ped 0.81-0.97), so the signal was there -- the statistic was wrong.

    Why these criteria still catch what this gate exists to catch: a wrong direction
    bin, swapped head ordering, or bad box encoding puts predictions nowhere near the
    gt (IoU ~= 0) for EVERY object, which fails criteria 1-3 together. What they no
    longer fail on is fine-grained regression noise on one small object.

    torch.manual_seed makes the run deterministic; without it this test was
    nondeterministic and the same config swung between IoU 0.38 and 0.67.
    """
    import json
    import torch
    from detection_vis_backend.datasets.view_of_delft import ViewOfDelft
    from detection_vis_backend.networks.network import NetworkFactory
    from detection_vis_backend.train.utils import radarpillar_collate
    from detection_vis_backend.networks.radarpillar import postprocess, rotated_iou_bev
    from data.database import get_session_local
    from data import crud

    os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost/postgres")
    with open("detection_vis_backend/networks/RadarPillar_model_config.json") as f:
        model_cfg = json.load(f)
    with open("detection_vis_backend/train/RadarPillar_train_config.json") as f:
        train_cfg = json.load(f)

    db = get_session_local()
    datafile = crud.get_datafile(db, 110)
    db.close()
    ds = ViewOfDelft()
    ds.parse(110, datafile.path, datafile.name, datafile.config)
    ds._request_scoped = True
    ds.prepare_for_train(["radarPC"], train_cfg, model_cfg)
    samples = []
    for i in range(60):
        s = ds[i]
        if len(s["gt_boxes"]):
            samples.append(s)
        if len(samples) == 4:
            break
    assert len(samples) == 4
    batch = radarpillar_collate(samples)

    device = "cuda"
    torch.manual_seed(0)     # determinism -- see docstring
    net = NetworkFactory().get_instance("RadarPillar", dict(model_cfg, **{"class": "RadarPillar"})).to(device)
    net.init_lossfunc(train_cfg, device)
    batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    # 300 steps localize near-perfectly but focal loss (alpha 0.25) raises confidence
    # slowly on a 4-sample overfit; 800 sufficed for the 4.8M slice-3.5 model. Slice 4's
    # C=32 backbone has 26x fewer parameters and needs proportionally more steps to
    # reach the same confidence -- measured 2026-07-20: 27/28 gt confident @800 (max
    # score 0.59) vs 28/28 @1600 (0.95). 2000 keeps margin above that crossover.
    for step in range(2000):
        opt.zero_grad()
        loss = net.get_loss(net(batch), batch, train_cfg)
        loss.backward()
        opt.step()

    net.eval()
    with torch.no_grad():
        results = postprocess(net(batch), net.net.anchors, model_cfg, train_cfg)

    GROSS_FLOOR = 0.1                          # criterion 2: box decoded to the right place at all
    MEAN_FLOOR = {0: 0.5, 1: 0.25, 2: 0.25}    # criterion 3: devkit's own 3D thresholds
    CLS = {0: "Car", 1: "Pedestrian", 2: "Cyclist"}
    per_class = {0: [], 1: [], 2: []}
    for b in range(len(results)):
        valid = batch["gt_labels"][b] >= 0
        gts = batch["gt_boxes"][b][valid].cpu().numpy()
        gt_labels = batch["gt_labels"][b][valid].cpu().numpy()
        res = results[b]
        for g, gl in zip(gts, gt_labels):
            same = res["boxes"][(res["labels"] == gl) & (res["scores"] > 0.5)]
            assert len(same), f"sample {b}: gt of class {CLS[int(gl)]} has no confident prediction"
            best = float(rotated_iou_bev(g[None], same)[0].max())
            assert best > GROSS_FLOOR, (
                f"sample {b}: {CLS[int(gl)]} best IoU {best:.2f} <= {GROSS_FLOOR} -- "
                f"box decoded to the wrong place, not regression noise")
            per_class[int(gl)].append(best)

    for cls_id, ious in per_class.items():
        if not ious:
            continue
        mean_iou = sum(ious) / len(ious)
        assert mean_iou > MEAN_FLOOR[cls_id], (
            f"{CLS[cls_id]}: mean IoU {mean_iou:.2f} over {len(ious)} gts "
            f"<= devkit floor {MEAN_FLOOR[cls_id]}")


def test_ViewOfDelft_original_split_loaders(tmp_path):
    """splitmode 'original' = the official ImageSets split: 5139 train / 1296 val,
    no test labels -> test bucket empty (skip-test handling applies)."""
    os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost/postgres")
    from detection_vis_backend.datasets.dataset import DatasetFactory
    from data.database import get_session_local
    from data import crud

    db = get_session_local()
    datafile = crud.get_datafile(db, 110)
    db.close()
    factory = DatasetFactory()
    factory.get_shared_instance(
        "ViewOfDelft", 110,
        parse_args=(110, datafile.path, datafile.name, datafile.config))

    model_config = json.load(open("detection_vis_backend/networks/RadarPillar_model_config.json"))
    model_config["class"] = "RadarPillar"
    train_config = json.load(open("detection_vis_backend/train/RadarPillar_train_config.json"))
    train_config["dataloader"]["splitmode"] = "original"

    datafiles = [{"parse": "ViewOfDelft", "id": 110, "name": datafile.name, "path": datafile.path}]
    train_loader, val_loader, test_loader = CreateDataLoaders(
        datafiles, ["radarPC"], model_config, train_config, str(tmp_path))

    assert len(train_loader.dataset) == 5139
    assert len(val_loader.dataset) == 1296
    assert test_loader is None

    # train subset augments, val subset must not: the two Subsets are backed by
    # DIFFERENT request copies of the dataset (slice 3.5)
    train_ds = train_loader.dataset.dataset   # Subset -> ViewOfDelft copy
    val_ds = val_loader.dataset.dataset
    assert train_ds is not val_ds
    assert train_ds.transformations is not None
    assert val_ds.transformations is None


def test_pointpillar_is_registered_alongside_radarpillar():
    """The slice-3.5 PointPillars baseline stays selectable after slice 4 retunes
    RadarPillar's config. Both names must resolve through every registry the train
    and inference paths consult, or a PointPillar run dies at collate/eval time.
    """
    from detection_vis_backend.networks.network import NetworkFactory
    from detection_vis_backend.train.evaluate import eval_func_dict, network_types

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    with open(os.path.join(root, "networks", "model_features.json")) as f:
        features = json.load(f)
    assert features["PointPillar"]["features"] == ["radarPC"]
    assert features["PointPillar"]["datasets"] == ["ViewOfDelft"]
    for name in ("networks/PointPillar_model_config.json",
                 "train/PointPillar_train_config.json"):
        assert os.path.exists(os.path.join(root, *name.split("/"))), name

    assert "PointPillar" in network_types
    assert ("PointPillar", "ViewOfDelft") in eval_func_dict
    # collate_func is a local dict inside CreateDataLoaders, so assert on its source:
    # a missing entry there means a PointPillar run dies at the first batch
    src = inspect.getsource(CreateDataLoaders)
    assert "'PointPillar': radarpillar_collate" in src

    # the baseline config is the slice-3.5 architecture, NOT the slice-4 one
    with open(os.path.join(root, "networks", "PointPillar_model_config.json")) as f:
        cfg = json.load(f)
    assert cfg["vfe_channels"] == 64
    assert cfg["backbone"]["channels"] == [64, 128, 256]
    assert cfg["backbone"]["up_channels"] == [128, 128, 128]
    assert cfg.get("velocity_decomposition", False) is False
    assert cfg.get("attention", {}).get("enabled", False) is False

    net = NetworkFactory().get_instance("PointPillar", cfg)
    assert net.net.vfe.linear.in_features == 13     # 7 raw + 6 offsets, no vx/vy


def test_radarpillar_extends_pointpillar_and_guards_name_config_mismatch():
    """RadarPillars IS PointPillars plus three config-gated mechanisms, so the class
    hierarchy says so. Each class warns when the config it gets contradicts its name,
    which is what keeps a 'baseline' number from being produced by an extras-on config.
    """
    from detection_vis_backend.networks.network import PointPillar, RadarPillar
    assert issubclass(RadarPillar, PointPillar)
    assert PointPillar.EXTRAS_EXPECTED is False and RadarPillar.EXTRAS_EXPECTED is True

    base_cfg = {
        "point_cloud_range": [0.0, -25.6, -3.0, 51.2, 25.6, 2.0],
        "voxel_size": [0.16, 0.16, 5.0], "max_points_per_pillar": 32, "max_pillars": 16000,
        "point_features": ["x", "y", "z", "rcs", "v_r", "v_r_comp", "time"],
        "vfe_channels": 32, "head_stride": 2, "dir_offset": 0.78539,
        "anchor_rotations": [0.0, 1.57],
        "anchors": {
            "Car":        {"sizes": [3.9, 1.6, 1.56], "z_bottom": -1.78, "match_threshold": 0.6, "unmatch_threshold": 0.45},
            "Pedestrian": {"sizes": [0.8, 0.6, 1.73], "z_bottom": -0.6, "match_threshold": 0.5, "unmatch_threshold": 0.35},
            "Cyclist":    {"sizes": [1.76, 0.6, 1.73], "z_bottom": -0.6, "match_threshold": 0.5, "unmatch_threshold": 0.35},
        },
        "backbone": {"layer_nums": [3, 5, 5], "strides": [2, 2, 2],
                     "channels": [32, 32, 32], "up_channels": [32, 32, 32]},
    }
    extras_on = dict(base_cfg, velocity_decomposition=True,
                     attention={"enabled": True, "embed_dim": 32, "num_heads": 1, "ffn_hidden": 32})

    # matching name/config pairs build without complaint; mismatched ones warn but still build
    assert PointPillar(**base_cfg) is not None
    assert RadarPillar(**extras_on) is not None
    assert PointPillar(**extras_on) is not None      # warns: extras on a baseline name
    assert RadarPillar(**base_cfg) is not None       # warns: no extras under the RadarPillars name
