"""The checkpoints the T-FFTRadNet authors published (linked from the README of
jgiroux8/T_FFTRadNet) load into our port unchanged, and the RADIal RD one detects as the
paper reports. The files sit outside the repository; every test skips without them."""
import os

import numpy as np
import pytest
import torch

from detection_vis_backend.paths import dataset_path
from detection_vis_backend.tests.test_tfftradnet_training import _model_config, _ui_train_config

WEIGHTS = os.path.expanduser("~/Downloads/tfftradnet_trained_models")
REAL_RADIAL = dataset_path("radial", "RADIal-labeled")

# Published file -> (dataset, input_mode). The names follow the reference repo's configs:
# RadIal RD_Shift_config.json reads "Custom_RD", the RD from ADC whose Doppler shift is discarded.
PUBLISHED = {
    "RADIal_SwinTransformer_RD.pth": ("RADIal", "RD"),
    "RADIal_SwinTransformer_RD_Shift.pth": ("RADIal", "RD_shift_as_published"),
    "RADIal_SwinTransformer_ADC.pth": ("RADIal", "ADC"),
    "RADDet_SwinTransformer_RD.pth": ("RADDetDataset", "RD"),
    "RADDet_SwinTransformer_RADCube.pth": ("RADDetDataset", "RAD"),
    "RADDet_SwinTransformer_ADC.pth": ("RADDetDataset", "ADC"),
}


def _checkpoint(name):
    path = os.path.join(WEIGHTS, name)
    if not os.path.isfile(path):
        pytest.skip(f"published weights absent: {path}")
    # The files also pickle the optimizer and scheduler, which weights_only refuses.
    return torch.load(path, map_location="cpu", weights_only=False)


def _net(dataset, mode, checkpoint):
    from detection_vis_backend.networks.network import T_FFTRadNet
    cfg = {k: v for k, v in _model_config(dataset).items() if k not in ("class", "dataset", "input_mode")}
    net = T_FFTRadNet(dataset=dataset, input_mode=mode, **cfg)
    net.load_state_dict(checkpoint["net_state_dict"], strict=True)
    return net.eval()


@pytest.mark.parametrize("name", sorted(PUBLISHED))
def test_each_published_checkpoint_loads_strictly(name):
    dataset, mode = PUBLISHED[name]
    _net(dataset, mode, _checkpoint(name))


@pytest.mark.skipif(not os.path.isdir(os.path.join(REAL_RADIAL, "radar_FFT")), reason="RADIal absent")
def test_published_radial_rd_scores_close_to_the_paper_on_test_frames():
    from detection_vis_backend.datasets.adapters import make_sample_adapter
    from detection_vis_backend.datasets.adapters.radial import radial_sequence_split
    from detection_vis_backend.datasets.dataset import RADIal
    from detection_vis_backend.train.evaluate import TFFTRadNet_radial_evaluation
    from detection_vis_backend.train.utils import TFFTRadNet_collate

    net = _net("RADIal", "RD", _checkpoint("RADIal_SwinTransformer_RD.pth"))
    parser = RADIal()
    parser.parse(1, os.path.dirname(REAL_RADIAL), os.path.basename(REAL_RADIAL), "")
    adapter = make_sample_adapter("T_FFTRadNet", parser, ["RD"], _model_config("RADIal"), {})
    test = radial_sequence_split(parser)["test"]
    frames = [int(test[i]) for i in np.linspace(0, len(test) - 1, 50).astype(int)]
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(adapter, frames), batch_size=4,
                                         collate_fn=TFFTRadNet_collate)
    train_config = _ui_train_config("RADIal", "RD")
    net.init_lossfunc(train_config, "cpu")   # the val scorer also reports the loss
    result = TFFTRadNet_radial_evaluation(net, loader, train_config, ["RD"], None, "val")
    assert result["F1"] > 0.7, result
