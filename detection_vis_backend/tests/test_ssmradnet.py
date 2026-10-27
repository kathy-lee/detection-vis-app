"""SSMRadNet (Sen et al., WACV 2026) on RADIal: the Mamba kernels it needs, and the
network ported from the authors' FFTRadNet/model/SSMRadNet.py."""
import importlib.util
import os

import pytest
import torch

UPSTREAM_MODEL = "/home/kangle/dataset/ssmradnet_reference/SSMRadNet/FFTRadNet/model/SSMRadNet.py"
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba kernels need CUDA")


def test_mamba_ssm_is_installed_for_this_torch():
    # The mamba_ssm and causal-conv1d wheels are compiled against one torch version;
    # a torch upgrade must replace them too.
    import causal_conv1d  # noqa: F401
    import mamba_ssm

    assert mamba_ssm.__version__ == "2.2.4"
    assert torch.__version__.startswith("2.1.2")


def _upstream_wrapper():
    spec = importlib.util.spec_from_file_location("upstream_ssmradnet", UPSTREAM_MODEL)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.FFTRadNet()


def test_parameter_count_matches_the_paper():
    from detection_vis_backend.networks.network import SSMRadNet

    # 306,832 parameters; the paper reports 0.31 M.
    n = sum(p.numel() for p in SSMRadNet().parameters())
    assert 300_000 <= n <= 320_000


@pytest.mark.skipif(not os.path.exists(UPSTREAM_MODEL), reason="upstream reference clone absent")
def test_state_dict_keys_match_the_upstream_wrapper():
    from detection_vis_backend.networks.network import SSMRadNet

    ours = SSMRadNet().state_dict()
    assert all(k.startswith("RADFe.") for k in ours)
    theirs = _upstream_wrapper().state_dict()
    assert set(ours) == set(theirs)
    assert all(ours[k].shape == theirs[k].shape for k in ours)


@cuda_only
def test_forward_shapes():
    from detection_vis_backend.networks.network import SSMRadNet

    net = SSMRadNet().cuda().eval()
    with torch.no_grad():
        out = net(torch.randn(2, 512, 256, 32, device="cuda"))
    assert out['Detection'].shape == (2, 3, 128, 224)
    cls = out['Detection'][:, 0]
    assert cls.min() >= 0 and cls.max() <= 1
    assert out['Segmentation'].shape == (2, 1, 256, 224)


@cuda_only
def test_batch_of_one():
    from detection_vis_backend.networks.network import SSMRadNet

    net = SSMRadNet().cuda().eval()
    with torch.no_grad():
        out = net(torch.randn(1, 512, 256, 32, device="cuda"))
    assert out['Detection'].shape == (1, 3, 128, 224)
    assert out['Segmentation'].shape == (1, 1, 256, 224)


def test_cpu_input_raises_a_clear_error():
    from detection_vis_backend.networks.network import SSMRadNet

    with pytest.raises(RuntimeError, match="CUDA"):
        SSMRadNet()(torch.zeros(1, 512, 256, 32))
