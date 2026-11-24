"""T-FFTRadNet (Giroux et al., ICCV 2023 BRAVO workshop) ported from the reference repo
(jgiroux8/T_FFTRadNet 68bc041): Fourier-Net, the Swin backbone, decoders and heads, and the
network class."""
import importlib.util
import os
import sys

import numpy as np
import pytest
import torch

from detection_vis_backend.paths import dataset_path

REF = dataset_path("tfftradnet_reference", "T_FFTRadNet")
needs_reference = pytest.mark.skipif(not os.path.isdir(REF), reason=f"reference clone not at {REF}")


def _complex_input(shape, seed=0):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)).astype(np.complex64)


def _timm_stand_in():
    """The reference's Swin.py imports DropPath, to_2tuple and trunc_normal_ from timm, which the
    project does not install; the same helpers come from our vendored copies."""
    import types
    from detection_vis_backend.networks import tfftradnet, transrad
    layers = types.ModuleType("timm.models.layers")
    layers.DropPath, layers.trunc_normal_, layers.to_2tuple = transrad.DropPath, transrad.trunc_normal_, tfftradnet.to_2tuple
    timm, models = types.ModuleType("timm"), types.ModuleType("timm.models")
    timm.models, models.layers = models, layers
    return {"timm": timm, "timm.models": models, "timm.models.layers": layers}


def _load_reference(folder, module):
    """A reference repo module loaded by path; its files import each other as `model.<name>`."""
    root = os.path.join(REF, folder)
    sys.path.insert(0, root)
    stand_in = {k: v for k, v in _timm_stand_in().items() if k not in sys.modules}
    sys.modules.update(stand_in)
    try:
        spec = importlib.util.spec_from_file_location(f"reference_{folder}_{module}",
                                                      os.path.join(root, "model", f"{module}.py"))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod
    finally:
        sys.path.remove(root)
        for name in [n for n in sys.modules if n == "model" or n.startswith("model.")] + list(stand_in):
            del sys.modules[name]


# ---- Fourier-Net -------------------------------------------------------------------

def test_cplxmodule_is_the_pinned_version():
    # The installed distribution; the module's own __version__ spells it "2022.06".
    from importlib.metadata import version
    assert version("cplxmodule") == "2022.6"


@pytest.mark.parametrize("ns,nc,ch", [(512, 256, 16), (256, 64, 8)])
def test_fourier_net_at_init_is_the_dft_of_the_input(ns, nc, ch):
    # Initialized to the DFT matrices, Fourier-Net equals a range FFT then a Doppler FFT
    # with the Doppler zero bin centred, before its InstanceNorm.
    from detection_vis_backend.networks.tfftradnet import FourierNet
    net = FourierNet(ns, nc, ch, doppler_shift=True).eval()
    x = _complex_input((1, ch, ns, nc))
    with torch.no_grad():
        y = net.transform(torch.from_numpy(x))
    ref = np.fft.fftshift(np.fft.fft(np.fft.fft(x, axis=2), axis=3), axes=3)
    assert y.shape == ref.shape
    assert np.abs(y.numpy() - ref).max() / np.abs(ref).max() < 1e-4


def test_no_shift_fourier_net_skips_the_doppler_shift():
    from detection_vis_backend.networks.tfftradnet import FourierNet
    net = FourierNet(256, 64, 8, doppler_shift=False).eval()
    x = _complex_input((1, 8, 256, 64), seed=1)
    with torch.no_grad():
        y = net.transform(torch.from_numpy(x))
    ref = np.fft.fft(np.fft.fft(x, axis=2), axis=3)
    assert np.abs(y.numpy() - ref).max() / np.abs(ref).max() < 1e-4


def test_fourier_net_output_is_2c_real_channels_normalized():
    from detection_vis_backend.networks.tfftradnet import FourierNet
    net = FourierNet(512, 256, 16).eval()
    with torch.no_grad():
        out = net(torch.from_numpy(_complex_input((2, 16, 512, 256), seed=2)))
    assert out.dtype == torch.float32 and out.shape == (2, 32, 512, 256)
    per_channel = out.flatten(2)
    assert per_channel.mean(-1).abs().max() < 1e-3
    assert (per_channel.std(-1) - 1).abs().max() < 1e-2


def test_nl_activation_adds_one_parameter():
    from detection_vis_backend.networks.tfftradnet import FourierNet
    linear, nl = FourierNet(256, 64, 8), FourierNet(256, 64, 8, activation="nl")
    assert "activation.m" in dict(nl.named_parameters())
    assert sum(p.numel() for p in nl.parameters()) == sum(p.numel() for p in linear.parameters()) + 1


def test_unknown_activation_raises():
    from detection_vis_backend.networks.tfftradnet import FourierNet
    with pytest.raises(ValueError, match="activation"):
        FourierNet(256, 64, 8, activation="relu")


# ---- Swin encoder, decoders, headers -----------------------------------------------

def _swin(in_chans):
    from detection_vis_backend.networks.tfftradnet import SwinTransformer
    return SwinTransformer(pretrain_img_size=None, patch_size=2, in_chans=in_chans, embed_dim=48,
                           depths=[2, 2, 6, 2], num_heads=[3, 6, 12, 24], window_size=7,
                           out_indices=(0, 1, 2, 3)).eval()


def test_swin_feature_shapes_on_radial_rd():
    with torch.no_grad():
        feats = _swin(32)(torch.randn(1, 32, 512, 256))
    assert [f.shape for f in feats] == [(1, 48, 256, 128), (1, 96, 128, 64),
                                        (1, 192, 64, 32), (1, 384, 32, 16)]


def test_radial_decoder_and_header_shapes():
    from detection_vis_backend.networks.tfftradnet import RadialDetectionHeader, RadialSwinDecoder
    with torch.no_grad():
        ra = RadialSwinDecoder().eval()(_swin(32)(torch.randn(1, 32, 512, 256)))
        det = RadialDetectionHeader(input_angle_size=224, reg_layer=2).eval()(ra)
    assert ra.shape[:2] == (1, 256)
    assert det.shape == (1, 3, 128, 224)


def test_raddet_decoder_and_header_shapes():
    from detection_vis_backend.networks.tfftradnet import RaddetDetectionHeader, RaddetSwinDecoder
    with torch.no_grad():
        ra = RaddetSwinDecoder().eval()(_swin(16)(torch.randn(1, 16, 256, 64)))
        det, cls = RaddetDetectionHeader(input_angle_size=128, reg_layer=2).eval()(ra)
    assert det.shape == (1, 3, 64, 128) and cls.shape == (1, 6, 64, 128)


def test_raddet_rad_decoder_shapes():
    from detection_vis_backend.networks.tfftradnet import RaddetDetectionHeader, RaddetSwinDecoderRAD
    with torch.no_grad():
        ra = RaddetSwinDecoderRAD().eval()(_swin(64)(torch.randn(1, 64, 256, 256)))
        det, cls = RaddetDetectionHeader(input_angle_size=128, reg_layer=2).eval()(ra)
    assert det.shape == (1, 3, 64, 128) and cls.shape == (1, 6, 64, 128)


def test_detection_channel_is_a_probability():
    from detection_vis_backend.networks.tfftradnet import RaddetDetectionHeader
    with torch.no_grad():
        det, _ = RaddetDetectionHeader().eval()(torch.randn(2, 128, 64, 128))
    assert det[:, 0].min() >= 0 and det[:, 0].max() <= 1


@needs_reference
def test_swin_matches_reference_output_for_the_same_weights():
    torch.manual_seed(0)
    theirs = _load_reference("RadIal", "Swin").SwinTransformer(
        pretrain_img_size=None, patch_size=2, in_chans=16, embed_dim=48, depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24], window_size=7, out_indices=(0, 1, 2, 3))
    theirs.eval()   # the reference's train() returns None, so eval() cannot be chained
    ours = _swin(16)
    ours.load_state_dict(theirs.state_dict(), strict=True)
    x = torch.randn(1, 16, 256, 64)
    with torch.no_grad():
        for a, b in zip(ours(x), theirs(x)):
            assert torch.equal(a, b)


@needs_reference
@pytest.mark.parametrize("folder,sizes", [("RadIal", (512, 256, 16)), ("RADDet", (256, 64, 8))])
def test_fourier_net_keys_match_reference(folder, sizes):
    from detection_vis_backend.networks.tfftradnet import FourierNet
    theirs = _load_reference(folder, "fourier_net").FFT_Net().state_dict()
    ours = FourierNet(*sizes).state_dict()
    assert set(ours) == set(theirs)
    assert all(ours[k].shape == theirs[k].shape for k in ours)


# ---- T_FFTRadNet network -----------------------------------------------------------

def _net(dataset, mode, **kw):
    from detection_vis_backend.networks.network import T_FFTRadNet
    return T_FFTRadNet(dataset=dataset, input_mode=mode, **kw)


def _count(net):
    return sum(p.numel() for p in net.parameters())


@pytest.mark.parametrize("dataset,mode,expected", [
    ("RADIal", "RD", 9.64e6), ("RADDetDataset", "RD", 7.5e6), ("RADDetDataset", "RAD", 8.22e6)])
def test_parameter_count_matches_paper_table_1(dataset, mode, expected):
    assert abs(_count(_net(dataset, mode)) - expected) / expected < 0.02


@pytest.mark.parametrize("dataset,expected", [("RADIal", 327_680), ("RADDetDataset", 69_632)])
def test_fourier_net_adds_the_paper_complex_weight_count(dataset, expected):
    # Table 1 counts complex weights once; cplxmodule stores real and imaginary parts.
    assert _count(_net(dataset, "ADC")) - _count(_net(dataset, "RD")) == 2 * expected


@pytest.mark.parametrize("dataset,mode,shape", [
    ("RADIal", "RD", (2, 32, 512, 256)), ("RADIal", "ADC", (2, 32, 512, 256)),
    ("RADDetDataset", "RD", (2, 16, 256, 64)), ("RADDetDataset", "ADC", (2, 16, 256, 64)),
    ("RADDetDataset", "RAD", (2, 64, 256, 256))])
def test_forward_output_shapes(dataset, mode, shape):
    net = _net(dataset, mode).eval()
    with torch.no_grad():
        out = net(torch.randn(*shape))
    if dataset == "RADIal":
        assert out["Detection"].shape == (2, 3, 128, 224)
        assert out["Segmentation"].shape == (2, 1, 256, 224)
        assert out["class"] == []
    else:
        assert out["Detection"].shape == (2, 3, 64, 128)
        assert out["class"].shape == (2, 6, 64, 128)
        assert out["Segmentation"] == []


@pytest.mark.parametrize("mode", ["RD", "RD_shift_as_published", "RD_shift", "RD_no_window", "ADC"])
def test_every_radial_mode_builds(mode):
    assert _net("RADIal", mode) is not None


@pytest.mark.parametrize("mode", ["RAD", "RAD_stored", "RD", "RD_no_window", "ADC"])
def test_every_raddet_mode_builds(mode):
    assert _net("RADDetDataset", mode) is not None


@pytest.mark.parametrize("dataset,mode,kw", [
    ("RADIal", "RD", {}), ("RADIal", "ADC", {}), ("RADIal", "ADC", {"fourier_activation": "nl"}),
    ("RADIal", "ADC", {"fourier_doppler_shift": False}),
    ("RADDetDataset", "RAD", {}), ("RADDetDataset", "ADC", {"fourier_activation": "nl"})])
def test_state_dict_round_trips_strictly(dataset, mode, kw):
    a, b = _net(dataset, mode, **kw), _net(dataset, mode, **kw)
    b.load_state_dict(a.state_dict(), strict=True)
    assert all(torch.equal(a.state_dict()[k], b.state_dict()[k]) for k in a.state_dict())


@pytest.mark.parametrize("dataset", ["RADIal", "RADDetDataset"])
def test_model_config_block_builds_the_network(dataset):
    # The Train page splats a dataset's block into the constructor (keys starting with '_'
    # are dropped first), so every key must be a constructor argument.
    import json
    from detection_vis_backend.networks.network import T_FFTRadNet
    path = os.path.join(os.path.dirname(__file__), "..", "networks", "T_FFTRadNet_model_config.json")
    with open(path) as f:
        block = {k: v for k, v in json.load(f)[dataset].items() if not k.startswith("_")}
    net = T_FFTRadNet(**block)
    assert (net.dataset, net.input_mode) == (dataset, "RD" if dataset == "RADIal" else "RAD")


def test_unknown_mode_raises():
    with pytest.raises(ValueError, match="RD_shift"):
        _net("RADIal", "RA")


def test_unknown_dataset_raises():
    with pytest.raises(ValueError, match="RADDetDataset"):
        _net("CARRADA", "RD")


def test_raddet_has_no_freespace_head():
    assert not hasattr(_net("RADDetDataset", "RD", segmentation_head=True), "freespace")


REFERENCE_MODEL_BLOCK = dict(patch_size=2, channels=[32, 40, 48, 56], embed_dim=48, depths=[2, 2, 6, 2],
                             num_heads=[3, 6, 12, 24], drop_rates=[0.0, 0.0, 0.0], regression_layer=2)


@needs_reference
@pytest.mark.parametrize("folder,cls,in_chans,dataset,mode", [
    ("RadIal", "FFTRadNet_ViT", 32, "RADIal", "RD"), ("RadIal", "FFTRadNet_ViT_ADC", 32, "RADIal", "ADC"),
    ("RADDet", "FFTRadNet_ViT", 16, "RADDetDataset", "RD"),
    ("RADDet", "FFTRadNet_ViT_ADC", 16, "RADDetDataset", "ADC"),
    ("RADDet", "FFTRadNet_ViT_RAD", 64, "RADDetDataset", "RAD")])
def test_state_dict_keys_and_shapes_match_reference(folder, cls, in_chans, dataset, mode):
    # RadIal's committed FFT_Net is the no-shift one; the keys are the same either way.
    reference = _load_reference(folder, "FFTRadNet_ViT")
    seg = folder == "RadIal"
    theirs = getattr(reference, cls)(in_chans=in_chans, detection_head=True, segmentation_head=seg,
                                     **REFERENCE_MODEL_BLOCK).state_dict()
    ours = _net(dataset, mode).state_dict()
    assert set(ours) == set(theirs)
    assert all(ours[k].shape == theirs[k].shape for k in ours)


def _targets(batch, dataset):
    t = {"encoded_label": torch.zeros(batch, 3, 128, 224) if dataset == "RADIal" else torch.zeros(batch, 3, 64, 128)}
    t["encoded_label"][:, 0, 10:13, 20:23] = 1
    t["encoded_label"][:, 1:, 10:13, 20:23] = 0.5
    if dataset == "RADIal":
        t["seg_label"] = (torch.rand(batch, 256, 224) > 0.5).float()
    else:
        t["class_map"] = torch.zeros(batch, 6, 64, 128)
    return t


RADIAL_LOSSES = {"losses": {"classification": "FocalLoss", "regression": "smoothL1Loss",
                            "segmentation": "BCEWithLogitsLoss", "weight": [2, 100, 100]}}
RADDET_LOSSES = {"losses": {"classification": "FocalLoss", "regression": "smoothL1Loss",
                            "weight": [2, 100, 100000]}}


def test_radial_loss_equals_fftradnet_loss():
    from detection_vis_backend.networks.network import FFTRadNet
    torch.manual_seed(0)
    pred = {"Detection": torch.cat([torch.rand(2, 1, 128, 224), torch.randn(2, 2, 128, 224)], 1),
            "Segmentation": torch.randn(2, 1, 256, 224), "class": []}
    target = _targets(2, "RADIal")
    ours = _net("RADIal", "RD")
    ours.init_lossfunc(RADIAL_LOSSES, "cpu")
    ref = FFTRadNet(mimo_layer=192, channels=[32, 40, 48, 56], blocks=[3, 6, 6, 3])
    ref.init_lossfunc(RADIAL_LOSSES, "cpu")
    assert torch.allclose(ours.get_loss(pred, target, RADIAL_LOSSES), ref.get_loss(pred, target, RADIAL_LOSSES))


def test_raddet_loss_class_term_is_zero_for_an_empty_class_map():
    torch.manual_seed(0)
    net = _net("RADDetDataset", "RD")
    net.init_lossfunc(RADDET_LOSSES, "cpu")
    det = torch.cat([torch.rand(2, 1, 64, 128), torch.randn(2, 2, 64, 128)], 1)
    target = _targets(2, "RADDetDataset")
    with_logits = net.get_loss({"Detection": det, "class": torch.randn(2, 6, 64, 128)}, target, RADDET_LOSSES)
    no_class_term = net.get_loss({"Detection": det, "class": torch.zeros(2, 6, 64, 128)}, target, RADDET_LOSSES)
    assert torch.allclose(with_logits, no_class_term)


def test_raddet_loss_weights_the_class_term():
    torch.manual_seed(0)
    net = _net("RADDetDataset", "RD")
    net.init_lossfunc(RADDET_LOSSES, "cpu")
    det = torch.cat([torch.rand(1, 1, 64, 128), torch.randn(1, 2, 64, 128)], 1)
    target = _targets(1, "RADDetDataset")
    target["class_map"][0, 2, 10:13, 20:23] = 1
    logits = torch.randn(1, 6, 64, 128)
    base = net.get_loss({"Detection": det, "class": torch.zeros(1, 6, 64, 128)}, {**target, "class_map": torch.zeros(1, 6, 64, 128)}, RADDET_LOSSES)
    total = net.get_loss({"Detection": det, "class": logits}, target, RADDET_LOSSES)
    ce = torch.nn.CrossEntropyLoss(reduction="mean")(logits, target["class_map"])
    assert torch.allclose(total - base, 100000 * ce, rtol=1e-4)
