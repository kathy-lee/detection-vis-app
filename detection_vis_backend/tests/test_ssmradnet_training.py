"""SSMRadNet on RADIal through the app's training, evaluation and inference paths: the
FFTRadNet collate, scorer and display generalized from the 'RD' input to 'ADC'."""
import json
import os

import numpy as np
import pytest
import torch

from detection_vis_backend.tests.test_radial_adc import parser, radial_root  # noqa: F401

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _sample(key, shape):
    return {key: np.random.rand(*shape).astype(np.float32),
            'encoded_label': np.zeros((3, 128, 224)), 'box_label': np.zeros((1, 10), dtype=np.float32),
            'seg_label': np.zeros((256, 224), dtype=bool)}


def test_collate_stacks_adc():
    from detection_vis_backend.train.utils import FFTRadNet_collate
    batch = FFTRadNet_collate([_sample('ADC', (512, 256, 32)) for _ in range(2)])
    assert set(batch) == {'ADC', 'encoded_label', 'box_label', 'seg_label'}
    assert batch['ADC'].shape == (2, 512, 256, 32)
    assert batch['seg_label'].shape == (2, 256, 224)


def test_collate_rd_is_unchanged():
    from detection_vis_backend.train.utils import FFTRadNet_collate
    samples = [_sample('RD', (32, 512, 256)) for _ in range(2)]
    batch = FFTRadNet_collate(samples)
    assert set(batch) == {'RD', 'encoded_label', 'box_label', 'seg_label'}
    assert torch.equal(batch['RD'], torch.stack([torch.tensor(s['RD']) for s in samples]))


def test_ssmradnet_is_registered():
    import inspect
    from detection_vis_backend.networks.network import NetworkFactory, SSMRadNet
    from detection_vis_backend.train.evaluate import eval_func_dict, FFTRadNet_evaluation
    from detection_vis_backend.train.train import CreateDataLoaders
    assert "'SSMRadNet': FFTRadNet_collate" in inspect.getsource(CreateDataLoaders)
    assert eval_func_dict["SSMRadNet", "RADIal"] is FFTRadNet_evaluation
    with open(os.path.join(BACKEND, "networks", "model_features.json")) as f:
        entry = json.load(f)["SSMRadNet"]
    assert entry["features"] == ["ADC"] and entry["datasets"] == ["RADIal"]
    with open(os.path.join(BACKEND, "networks", "SSMRadNet_model_config.json")) as f:
        assert isinstance(NetworkFactory().get_instance("SSMRadNet", json.load(f)), SSMRadNet)


def test_random_split_with_zero_test_fraction_has_no_test_frames():
    from detection_vis_backend.train.train import random_split_sizes
    # Flooring both fractions used to push the remainder into test: 568 / 142 / 1.
    assert random_split_sizes(711, [0.8, 0.2, 0.0]) == (568, 143, 0)
    assert random_split_sizes(10, [0.8, 0.2, 0.0]) == (8, 2, 0)


def test_random_split_sizes_are_unchanged_when_test_is_asked_for():
    from detection_vis_backend.train.train import random_split_sizes
    assert random_split_sizes(711, [0.7, 0.15, 0.15]) == (497, 106, 108)


def test_adam_weight_decay_defaults_to_zero():
    from detection_vis_backend.train.train import build_optimizer
    net = torch.nn.Linear(2, 2)
    for name, expected in (("FFTRadNet", 0), ("SSMRadNet", 5e-6)):
        with open(os.path.join(BACKEND, "train", f"{name}_train_config.json")) as f:
            optimizer = build_optimizer(net, json.load(f))
        assert isinstance(optimizer, torch.optim.Adam)
        assert optimizer.param_groups[0]['weight_decay'] == expected
        assert optimizer.param_groups[0]['lr'] == 1e-4


def test_ssmradnet_train_config_carries_the_paper_recipe():
    with open(os.path.join(BACKEND, "train", "SSMRadNet_train_config.json")) as f:
        cfg = json.load(f)
    assert cfg["num_epochs"] == 200
    assert cfg["dataloader"]["train"]["batch_size"] == 8
    assert cfg["dataloader"]["splitmode"] == "random" and cfg["dataloader"]["split_random"] == [0.8, 0.2, 0.0]
    assert cfg["losses"]["segmentation"] == "JaccardLoss"


def test_adapter_maps_samples_to_dataset_frames(parser):  # noqa: F811
    from detection_vis_backend.datasets.adapters import make_sample_adapter
    adapter = make_sample_adapter("SSMRadNet", parser, ["ADC"], {"segmentation_head": True}, {})
    # Parser frames 0 and 2 have an adc file, frame 1 does not.
    assert [adapter.dataset_frame(i) for i in range(len(adapter))] == [0, 2]
    assert adapter.sample_of_frame(2) == 1
    with pytest.raises(ValueError, match="radar_ADC"):
        adapter.sample_of_frame(1)


def test_original_split_without_val_frames_raises(radial_root, tmp_path):  # noqa: F811
    # The fixture's recordings are none of valeo's val/test ones, so val gets no frame.
    from detection_vis_backend.datasets.dataset import DatasetFactory
    from detection_vis_backend.train.train import CreateDataLoaders
    with open(os.path.join(BACKEND, "train", "SSMRadNet_train_config.json")) as f:
        train_config = json.load(f)
    train_config["dataloader"]["splitmode"] = "original"
    datafile = {"id": 9101, "parse": "RADIal", "name": radial_root.name, "path": str(radial_root.parent), "config": ""}
    DatasetFactory().get_shared_instance("RADIal", 9101, parse_args=(9101, datafile["path"], datafile["name"], ""))
    with pytest.raises(ValueError, match="val"):
        CreateDataLoaders([datafile], ["ADC"], {"class": "SSMRadNet", "segmentation_head": True},
                          train_config, str(tmp_path))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba kernels need CUDA")
def test_val_evaluation_reads_adc(parser):  # noqa: F811
    from torch.utils.data import DataLoader
    from detection_vis_backend.datasets.adapters import make_sample_adapter
    from detection_vis_backend.networks.network import SSMRadNet
    from detection_vis_backend.train.evaluate import FFTRadNet_val_evaluation
    from detection_vis_backend.train.utils import FFTRadNet_collate
    with open(os.path.join(BACKEND, "train", "SSMRadNet_train_config.json")) as f:
        train_config = json.load(f)
    adapter = make_sample_adapter("SSMRadNet", parser, ["ADC"], {"segmentation_head": True}, train_config)
    loader = DataLoader(adapter, batch_size=2, collate_fn=FFTRadNet_collate)
    net = SSMRadNet().cuda()
    net.init_lossfunc(train_config, "cuda")
    result = FFTRadNet_val_evaluation(net, loader, train_config, device="cuda")
    assert all(np.isfinite(result[k]) for k in ("loss", "mAP", "mAR", "mIoU"))
