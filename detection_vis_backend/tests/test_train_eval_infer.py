import os
import json
import inspect
import pytest
import numpy as np
from unittest.mock import MagicMock

from detection_vis_backend.train.train import train, CreateDataLoaders
from detection_vis_backend.train.evaluate import FFTRadNet_evaluation
from detection_vis_backend.train.inference import infer
from detection_vis_backend.train.utils import post_process_single_frame


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