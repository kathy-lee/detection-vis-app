import os
import json
import copy
import pytest

from detection_vis_backend.networks.network import (
    NetworkFactory,
    FFTRadNet,
    RODNet_CDC, RODNet_CDCv2, RODNet_HG, RODNet_HGwI, RODNet_v2Base, RODNet_HGv2, RODNet_HGwIv2,
    RECORD, RECORDNoLstm, RECORDNoLstmMulti, MVRECORD,
    RADDet,
    DAROD,
    RAMP_CNN,
    RadarCrossAttention,
)

RECORD_MODEL_CONFIG = os.path.join(
    os.path.dirname(os.path.dirname(__file__)), "networks", "RECORD_model_config.json"
)


@pytest.fixture
def clean_network_factory():
    """NetworkFactory caches instances at class scope; reset it so a stale cache
    from a prior test/inference can't leak into this one."""
    NetworkFactory._instances.clear()
    yield
    NetworkFactory._instances.clear()


def test_get_instance_distinct_configs_same_class(clean_network_factory):
    """Regression test: two RECORD models with different configs must yield
    correctly-shaped instances, not a stale reuse of the first.

    Root cause of the original bug: get_instance keyed its cache on class name
    alone and ignored `params` on a cache hit. So after building RECORD with the
    CRUW config (in_channels=8), building it again with the CARRADA/RD config
    (in_channels=1) returned the cached 8-channel net, and loading a 1-channel
    CARRADA checkpoint into it failed with size mismatches at inference.
    """
    with open(RECORD_MODEL_CONFIG) as f:
        cfg = json.load(f)
    cruw_cfg = cfg["RECORD"]["CRUW"]                # in_channels=8, n_class=3
    carrada_cfg = cfg["RECORD"]["CARRADA"]["RD"]    # in_channels=1, n_class=4

    factory = NetworkFactory()

    # deepcopy per call: real call sites (inference.py, train.py) reload model_config
    # from disk each time, and __init__ mutates the config dict in place.
    cruw_net = factory.get_instance("RECORD", copy.deepcopy(cruw_cfg))
    assert cruw_net.encoder.in_conv.conv.weight.shape[1] == 8
    assert cruw_net.decoder.conv_head2.weight.shape[0] == 3

    carrada_net = factory.get_instance("RECORD", copy.deepcopy(carrada_cfg))
    # Would be 8 / 3 (the stale CRUW instance) before the fix.
    assert carrada_net.encoder.in_conv.conv.weight.shape[1] == 1
    assert carrada_net.decoder.conv_head2.weight.shape[0] == 4
    assert carrada_net is not cruw_net

    # Same class + same config (a fresh identical dict) still returns the cached instance.
    assert factory.get_instance("RECORD", copy.deepcopy(carrada_cfg)) is carrada_net