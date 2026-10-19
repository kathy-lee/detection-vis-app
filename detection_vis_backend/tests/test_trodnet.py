"""T-RODNet: the ported body against the reference (Zhuanglong2/T-RODNet @ 39a2444)."""
import importlib.util
import os
import sys
import types

import pytest
import torch

REF_DIR = "/home/kangle/dataset/trodnet_reference/T-RODNet"
needs_ref = pytest.mark.skipif(not os.path.isdir(REF_DIR), reason=f"reference clone not at {REF_DIR}")


def test_channel_order_helpers_are_the_reference_rearranges():
    from detection_vis_backend.networks.trodnet import channels_last, channels_first
    x = torch.arange(2 * 3 * 4 * 5 * 6, dtype=torch.float32).reshape(2, 3, 4, 5, 6)
    y = channels_last(x)                                   # 'b c d h w -> b d h w c'
    assert y.shape == (2, 4, 5, 6, 3) and y[1, 2, 3, 4, 0] == x[1, 0, 2, 3, 4]
    assert torch.equal(channels_first(y), x)               # 'b d h w c -> b c d h w'


def test_window_partition_reverse_round_trip():
    from detection_vis_backend.networks.trodnet import window_partition, window_reverse
    x = torch.randn(1, 8, 64, 64, 32)
    w = window_partition(x, (4, 4, 4))
    assert w.shape == (2 * 16 * 16, 64, 32)
    assert torch.equal(window_reverse(w, (4, 4, 4), 1, 8, 64, 64), x)


def test_body_output_shape_and_finite_on_cpu():
    from detection_vis_backend.networks.trodnet import TRODNetBody
    body = TRODNetBody().eval()
    with torch.no_grad():
        out = body(torch.randn(1, 2, 16, 128, 128))
    assert out.shape == (1, 3, 16, 128, 128) and torch.isfinite(out).all()


def test_body_parameter_count_is_the_reference_classes():
    """156.8 M parameters: the paper's Table IV says 159.70 M for T-RODNet, and the
    reference's own 31.54 M comment (Nets/T_RODNet.py:1106) is thop's count of the
    modules its forward actually calls. Pinned from the reference class, which the
    key-set test below compares against directly when the clone is present."""
    from detection_vis_backend.networks.trodnet import TRODNetBody
    n = sum(p.numel() for p in TRODNetBody().parameters())
    assert n == 156_805_004, n


def _load_reference_class():
    for name in ("einops", "thop", "fvcore", "fvcore.nn", "timm", "timm.models", "timm.models.layers"):
        sys.modules.setdefault(name, types.ModuleType(name))
    from detection_vis_backend.networks import transrad
    sys.modules["timm.models.layers"].DropPath = transrad.DropPath
    sys.modules["timm.models.layers"].trunc_normal_ = transrad.trunc_normal_
    sys.modules["timm.models.layers"].to_3tuple = lambda x: x if isinstance(x, tuple) else (x, x, x)
    sys.modules["einops"].rearrange = None
    sys.modules["thop"].profile = None
    sys.modules["fvcore.nn"].FlopCountAnalysis = None
    sys.modules["fvcore.nn"].parameter_count_table = None
    spec = importlib.util.spec_from_file_location("ref_trodnet", os.path.join(REF_DIR, "Nets", "T_RODNet.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.T_RODNet


@needs_ref
def test_body_state_dict_keys_match_the_reference_class():
    from detection_vis_backend.networks.trodnet import TRODNetBody
    ref = _load_reference_class()(num_classes=3, embed_dim=64, win_size=4)
    ours = TRODNetBody()
    assert set(ours.swin_unet.state_dict()) == set(ref.swin_unet.state_dict())
    assert sum(p.numel() for p in ours.parameters()) == sum(p.numel() for p in ref.parameters())
