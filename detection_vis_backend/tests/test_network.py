import os
import json
import copy
import pytest
import torch

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


def test_get_instance_distinct_configs_same_class():
    """Regression test: two RECORD models with different configs must each yield a
    correctly-shaped instance, not a stale reuse of the first.

    Root cause of the original bug: get_instance cached instances keyed on class name
    alone and ignored `params` on a cache hit. So after building RECORD with the CRUW
    config (in_channels=8), building it again with the CARRADA/RD config (in_channels=1)
    returned the cached 8-channel net, and loading a 1-channel CARRADA checkpoint into it
    failed with size mismatches at inference. The factory now builds a fresh instance per
    call, so each config gets its own correctly-shaped net.
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

    # No caching: each call builds a fresh instance, even for the same config.
    assert factory.get_instance("RECORD", copy.deepcopy(carrada_cfg)) is not carrada_net


def test_load_checkpoint_ignores_criterion_keys():
    """Regression test: checkpoints save net.state_dict() after training attached the loss
    criterion as a submodule, so they carry criterion.* buffers (e.g. RECORD's weighted-CE
    `criterion.0.weight`, MVRECORD's rd/ra_criterion). Inference builds the net without a
    criterion, so a strict load rejects those keys. The fix loads with strict=False, which
    must ignore the criterion buffers while still raising on genuine shape mismatches.

    This is dataset/loss dependent: CRUW RECORD uses `bce` (parameter-less, no extra keys,
    always loaded fine), while CARRADA RECORD uses `wce_w10sdice` (a weighted CE whose
    class-weight buffer serializes as criterion.0.weight) -- which is what broke.
    """
    with open(RECORD_MODEL_CONFIG) as f:
        cfg = json.load(f)
    carrada_cfg = cfg["RECORD"]["CARRADA"]["RD"]    # n_class=4
    factory = NetworkFactory()

    # Trained net: attach the weighted-CE criterion exactly as training does, so its
    # state_dict carries criterion.0.weight.
    trained = factory.get_instance("RECORD", copy.deepcopy(carrada_cfg))
    trained.init_lossfunc(
        {"losses": {"type": "wce_w10sdice", "weight": [0.1, 0.3, 0.3, 0.3]}}, "cpu"
    )
    state_dict = trained.state_dict()
    assert any(k.startswith("criterion") for k in state_dict)  # the buffer is present

    # Inference net: no criterion attached.
    fresh = factory.get_instance("RECORD", copy.deepcopy(carrada_cfg))

    # Default strict load rejects the unexpected criterion key -- the reported bug.
    with pytest.raises(RuntimeError, match="Unexpected key"):
        fresh.load_state_dict(state_dict)

    # strict=False (the fix) loads fine, ignoring only the criterion buffer.
    result = fresh.load_state_dict(state_dict, strict=False)
    assert result.missing_keys == []
    assert result.unexpected_keys and all(k.startswith("criterion") for k in result.unexpected_keys)

    # But a genuine shape mismatch on a real network key still raises, even with strict=False,
    # so this does not re-mask the config/shape class of bug.
    bad = dict(state_dict)
    bad["encoder.in_conv.conv.weight"] = torch.zeros(32, 999, 3, 3)
    with pytest.raises(RuntimeError, match="size mismatch"):
        fresh.load_state_dict(bad, strict=False)