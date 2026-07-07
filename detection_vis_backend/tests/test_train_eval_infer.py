import os
import json
import inspect
import pytest
from unittest.mock import MagicMock

from detection_vis_backend.train.train import train, CreateDataLoaders
from detection_vis_backend.train.evaluate import FFTRadNet_evaluation
from detection_vis_backend.train.inference import infer


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