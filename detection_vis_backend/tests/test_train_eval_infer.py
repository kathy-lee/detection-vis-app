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
    frame_oriented = {seq: {"000066": {"000000": {
        "range_angle": {"label": 3, "box": [[10, 20], [30, 40]]},
        "range_doppler": {"label": 3, "box": [[10, 20], [30, 40]]},
    }}}}
    # the real new_gt_anno.json shape, verified against the dataset on disk
    rca_anno = {seq: {"tracks": [[0, 1, 2]], "000066": {"cls": [3.0], "idx": [92.0, 6.0, 132.0, 26.0]}}}
    (tmp_path / "new_gt_anno.json").write_text(json.dumps(rca_anno))

    ds = CARRADA()
    ds.root_path = str(tmp_path)
    ds.seq_name = seq
    ds.annos = frame_oriented          # as parse() would leave it
    ds.frame_sync = 3

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
    assert ds.rca_annos[seq]["000066"] == {"cls": [3.0], "idx": [92.0, 6.0, 132.0, 26.0]}


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