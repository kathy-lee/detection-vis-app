"""T-FFTRadNet through the app's training path: collation, registration, the train config
with the paper's recipe, and one optimizer step on a synthetic RADIal tree."""
import json
import os

import numpy as np
import pytest
import torch

from detection_vis_backend.tests.test_tfftradnet_radial import parser, radial_root  # noqa: F401

BACKEND = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _load(*parts):
    with open(os.path.join(BACKEND, *parts)) as f:
        return json.load(f)


def _ui_train_config(dataset, feature):
    """The Train page's merge: default, then the dataset block or its per-feature block."""
    data = _load("train", "T_FFTRadNet_train_config.json")
    cfg = data["default"].copy()
    if dataset in data:
        cfg.update(data[dataset][feature] if feature in data[dataset] else data[dataset])
    return cfg


def _model_config(dataset):
    block = _load("networks", "T_FFTRadNet_model_config.json")[dataset]
    return {k: v for k, v in block.items() if not k.startswith("_")}


def test_collate_stacks_input_and_class_map():
    from detection_vis_backend.train.utils import TFFTRadNet_collate
    samples = [{"RAD": np.random.rand(64, 256, 256).astype(np.float32),
                "encoded_label": np.zeros((3, 64, 128), dtype=np.float32),
                "class_map": np.zeros((6, 64, 128), dtype=np.float32),
                "box_label": np.zeros((n, 3), dtype=np.float32)} for n in (1, 2)]
    batch = TFFTRadNet_collate(samples)
    assert set(batch) == {"RAD", "encoded_label", "class_map", "box_label"}
    assert batch["RAD"].shape == (2, 64, 256, 256)
    assert batch["class_map"].shape == (2, 6, 64, 128)
    assert [b.shape[0] for b in batch["box_label"]] == [1, 2]


def test_collate_keeps_seg_label_for_radial():
    from detection_vis_backend.train.utils import TFFTRadNet_collate
    samples = [{"ADC": np.zeros((32, 512, 256), dtype=np.float32),
                "encoded_label": np.zeros((3, 128, 224)), "seg_label": np.zeros((256, 224)),
                "box_label": np.zeros((1, 10), dtype=np.float32)} for _ in range(2)]
    batch = TFFTRadNet_collate(samples)
    assert set(batch) == {"ADC", "encoded_label", "seg_label", "box_label"}
    assert batch["seg_label"].shape == (2, 256, 224)


def test_t_fftradnet_is_registered_for_both_datasets():
    import inspect
    from detection_vis_backend.train.train import CreateDataLoaders
    assert "'T_FFTRadNet': TFFTRadNet_collate" in inspect.getsource(CreateDataLoaders)
    entry = _load("networks", "model_features.json")["T_FFTRadNet"]
    assert entry["features"] == ["RD", "ADC", "RAD"]
    assert entry["datasets"] == ["RADIal", "RADDetDataset"]


def test_train_config_carries_the_paper_recipe():
    for dataset, feature, batch, monitor, weight in (("RADIal", "RD", 4, "F1", [2, 100, 100]),
                                                     ("RADDetDataset", "RD", 16, "select_score", [2, 100, 100000]),
                                                     ("RADDetDataset", "ADC", 16, "select_score", [2, 100, 100000]),
                                                     ("RADDetDataset", "RAD", 4, "select_score", [2, 100, 100000])):
        cfg = _ui_train_config(dataset, feature)
        assert cfg["seed"] == 3 and cfg["num_epochs"] == 150
        assert cfg["optimizer"] == {"lr": 1e-4}
        assert cfg["lr_scheduler"] == {"type": "step", "step_size": 10, "gamma": 0.9}
        assert cfg["dataloader"]["splitmode"] == "original"
        assert cfg["dataloader"]["train"]["batch_size"] == batch, (dataset, feature)
        assert cfg["dataloader"]["val"]["batch_size"] == 4 and cfg["dataloader"]["test"]["batch_size"] == 1
        assert cfg["test_checkpoint"] == {"monitor": monitor, "mode": "max"}, (dataset, feature)
        assert cfg["losses"]["weight"] == weight, (dataset, feature)
        assert cfg["losses"]["classification"] == "FocalLoss"


@pytest.mark.parametrize("dataset", ["RADIal", "RADDetDataset"])
def test_model_config_blocks_build_the_network(dataset):
    from detection_vis_backend.networks.network import NetworkFactory, T_FFTRadNet
    assert isinstance(NetworkFactory().get_instance("T_FFTRadNet", _model_config(dataset)), T_FFTRadNet)


def test_one_training_iteration_on_synthetic_radial(parser, tmp_path, monkeypatch):  # noqa: F811
    from detection_vis_backend.datasets.dataset import DatasetFactory
    from detection_vis_backend.networks.network import T_FFTRadNet
    from detection_vis_backend.train.train import CreateDataLoaders
    monkeypatch.setattr(DatasetFactory, "get_instance", lambda self, *a, **k: parser)
    model_config = dict(_model_config("RADIal"), **{"class": "T_FFTRadNet"})
    train_config = _ui_train_config("RADIal", "RD")
    train_config["dataloader"] = dict(train_config["dataloader"], splitmode="random", split_random=[0.67, 0.33, 0.0],
                                      train={"batch_size": 2, "num_workers": 0}, val={"batch_size": 1, "num_workers": 0})
    datafiles = [{"id": 1, "name": "RADIal-labeled", "parse": "RADIal"}]
    loaders = CreateDataLoaders(datafiles, ["RD"], model_config, train_config, str(tmp_path))
    train_loader = loaders[0]
    data = next(iter(train_loader))
    data = {k: v.float() if isinstance(v, torch.Tensor) else v for k, v in data.items()}

    torch.manual_seed(0)
    net = T_FFTRadNet(**{k: v for k, v in model_config.items() if k != "class"})
    net.init_lossfunc(train_config, "cpu")
    optimizer = torch.optim.Adam(net.parameters(), lr=train_config["optimizer"]["lr"])
    before = [p.detach().clone() for p in net.parameters()]
    loss = net.get_loss(net(data.pop("RD")), data, train_config)
    loss.backward()
    optimizer.step()
    assert torch.isfinite(loss)
    assert any(not torch.equal(a, b) for a, b in zip(before, net.parameters()))
