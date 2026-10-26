"""SSMRadNet (Sen et al., WACV 2026) on RADIal: the Mamba kernels it needs, and the
network ported from the authors' FFTRadNet/model/SSMRadNet.py."""
import torch


def test_mamba_ssm_is_installed_for_this_torch():
    # The mamba_ssm and causal-conv1d wheels are compiled against one torch version;
    # a torch upgrade must replace them too.
    import causal_conv1d  # noqa: F401
    import mamba_ssm

    assert mamba_ssm.__version__ == "2.2.4"
    assert torch.__version__.startswith("2.1.2")
