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
    """Import the reference file with its absent dependencies stubbed, then remove the
    stubs again: a leftover fake `einops` breaks torch.compile's import probe in later
    tests."""
    stubs = ("einops", "thop", "fvcore", "fvcore.nn", "timm", "timm.models", "timm.models.layers")
    saved = {name: sys.modules.get(name) for name in stubs}
    try:
        for name in stubs:
            sys.modules[name] = types.ModuleType(name)
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
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


@needs_ref
def test_body_state_dict_keys_match_the_reference_class():
    from detection_vis_backend.networks.trodnet import TRODNetBody
    ref = _load_reference_class()(num_classes=3, embed_dim=64, win_size=4)
    ours = TRODNetBody()
    assert set(ours.swin_unet.state_dict()) == set(ref.swin_unet.state_dict())
    assert sum(p.numel() for p in ours.parameters()) == sum(p.numel() for p in ref.parameters())


# ---------------------------------------------------------------------------------------
# Network class, configs, registry entries, cosine schedule, sweep entry.
# ---------------------------------------------------------------------------------------
import json
import inspect

from torch import nn

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TEST_SEQS = ("2019_04_09_BMS1001", "2019_04_30_MLMS001", "2019_05_23_PM1S013", "2019_09_29_ONRD005")


def _model_cfg():
    with open(os.path.join(REPO, "detection_vis_backend", "networks", "T_RODNet_model_config.json")) as f:
        return json.load(f)


def test_trodnet_network_through_the_factory_and_loss_contract():
    from detection_vis_backend.networks.network import NetworkFactory
    net = NetworkFactory().get_instance("T_RODNet", _model_cfg()).eval()
    net.init_lossfunc({"losses": {"type": "mse"}}, torch.device("cpu"))
    with torch.no_grad():
        out = net(torch.randn(1, 2, 16, 128, 128))
    assert out.shape == (1, 3, 16, 128, 128)
    loss = net.get_loss(out, {"gt_mask": torch.rand(1, 3, 16, 128, 128)}, {})
    assert torch.isfinite(loss) and net.n_class == 3 and net.class_names == ["pedestrian", "cyclist", "car"]


def test_trodnet_mse_is_mean_squared_error_not_smooth_l1():
    from detection_vis_backend.networks.network import NetworkFactory
    net = NetworkFactory().get_instance("T_RODNet", _model_cfg())
    net.init_lossfunc({"losses": {"type": "mse"}}, torch.device("cpu"))
    assert type(net.criterion) is nn.MSELoss
    with pytest.raises(ValueError):
        net.init_lossfunc({"losses": {"type": "bce"}}, torch.device("cpu"))


def test_trodnet_state_dict_keys_have_no_body_prefix():
    from detection_vis_backend.networks.network import NetworkFactory
    net = NetworkFactory().get_instance("T_RODNet", _model_cfg())
    sd = net.state_dict()
    assert all(k.startswith("swin_unet.") for k in sd)
    net.load_state_dict(sd, strict=True)


def test_trodnet_train_config_carries_the_reference_recipe():
    with open(os.path.join(REPO, "detection_vis_backend", "train", "T_RODNet_train_config.json")) as f:
        cfg = json.load(f)
    t = cfg["default"]
    assert t["optimizer"] == {"lr": 0.0001, "weight_decay": 0} and t["losses"] == {"type": "mse"}
    assert t["lr_scheduler"] == {"type": "cosine", "min_lr": 0} and t["num_epochs"] == 100 and t["seed"] == 3
    assert (t["win_size"], t["train_step"], t["train_stride"], t["test_step"], t["test_stride"]) == (16, 1, 4, 1, 8)
    assert (t["max_dets"], t["peak_thres"], t["ols_thres"]) == (20, 0.3, 0.3)
    assert t["dataloader"]["train"]["batch_size"] == 4 and t["dataloader"]["val"]["batch_size"] == 1
    assert "test_checkpoint" not in t


def test_trodnet_is_registered_everywhere_the_app_looks():
    from detection_vis_backend.train.evaluate import eval_func_dict, RODNet_evaluation
    from detection_vis_backend.train.train import CreateDataLoaders
    assert eval_func_dict["T_RODNet", "CRUW"] is RODNet_evaluation
    assert "'T_RODNet': default_collate" in inspect.getsource(CreateDataLoaders)
    with open(os.path.join(REPO, "detection_vis_backend", "networks", "model_features.json")) as f:
        feats = json.load(f)["T_RODNet"]
    assert feats["features"] == ["RA"] and feats["datasets"] == ["CRUW"] and feats["input_size"] == [16, 128, 128]


def test_cosine_branch_reproduces_the_reference_schedule():
    from detection_vis_backend.train.train import cosine_scheduler, train
    net = nn.Linear(2, 2)
    opt = torch.optim.Adam(net.parameters(), lr=1e-4)
    sched = cosine_scheduler(opt, {"type": "cosine", "min_lr": 0}, 100)
    ref_opt = torch.optim.Adam(nn.Linear(2, 2).parameters(), lr=1e-4)
    ref = torch.optim.lr_scheduler.CosineAnnealingLR(ref_opt, 100)
    src = inspect.getsource(train)
    assert "== 'cosine':" in src and "cosine_scheduler(optimizer, train_config['lr_scheduler'], num_epochs)" in src
    for _ in range(100):
        opt.step(); ref_opt.step(); sched.step(); ref.step()
        assert opt.param_groups[0]["lr"] == pytest.approx(ref_opt.param_groups[0]["lr"], rel=1e-9)
    for name in os.listdir(os.path.join(REPO, "detection_vis_backend", "train")):
        if name.endswith("_train_config.json") and not name.startswith("T_RODNet"):
            with open(os.path.join(REPO, "detection_vis_backend", "train", name)) as f:
                assert '"cosine"' not in f.read(), name


def test_trodnet_configs_and_split_resolve_through_smoke_sweep():
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from smoke_sweep import resolve_configs, build_split, run_id, RUNS, LIVE_METRICS
    m, t = resolve_configs("T_RODNet", "CRUW", ["RA"])
    assert m["class"] == "T_RODNet" and m["embed_dim"] == 64 and m["win_size"] == 4
    assert t["lr_scheduler"]["type"] == "cosine" and not any(k.startswith("_") for k in t)
    assert any(run_id(r) == "T_RODNet/CRUW" for r in RUNS) and LIVE_METRICS["T_RODNet", "CRUW"] == ("mAP", "mAR")
    rows = [{"name": f"2019_01_01_XXX{i:04d}", "id": i} for i in range(36)] + [{"name": s, "id": 100 + i} for i, s in enumerate(TEST_SEQS)]
    chosen, patch = build_split("CRUW", rows, "trodnet")
    assert patch["splitmode"] == "sequence" and len(patch["split_sequence"]["train"]) == 36
    assert [r["name"] for r in patch["split_sequence"]["val"]] == list(TEST_SEQS)
    assert patch["split_sequence"]["test"] == patch["split_sequence"]["val"]
    assert not any(r["name"] in TEST_SEQS for r in patch["split_sequence"]["train"])
    with pytest.raises(ValueError):
        build_split("CARRADA", rows, "trodnet")
    # monitor="one": the per-epoch curve on a single test sequence; the test set is unchanged
    _, one = build_split("CRUW", rows, "trodnet", monitor="one")
    assert [r["name"] for r in one["split_sequence"]["val"]] == ["2019_04_30_MLMS001"]
    assert [r["name"] for r in one["split_sequence"]["test"]] == list(TEST_SEQS)
    assert len(one["split_sequence"]["train"]) == 36


def test_rodnet_evaluation_deletes_stale_result_files_before_writing():
    from detection_vis_backend.train.evaluate import RODNet_evaluation
    src = inspect.getsource(RODNet_evaluation)
    first_write_to_current = src.index("write_dets_results_single_frame(res_final, cur_frame_id, save_path")
    assert src.index("os.remove(save_path)") < first_write_to_current


def test_scorer_check_converts_our_rows_to_the_reference_order():
    """Ours: frame class row col conf (grid indices). Theirs: frame range angle class
    conf, metres / radians to 4 decimals and the score to 2, trailing space
    (rodnet/core/post_processing/output_results.py:32)."""
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from trodnet_scorer_check import to_reference_rows
    rng_grid = [0.0, 1.5, 3.0, 4.5]
    agl_grid = [-0.5, 0.0, 0.5]
    assert to_reference_rows(["12 car 2 1 0.8731", "12 pedestrian 3 2 1.2"], rng_grid, agl_grid) == [
        "12 3.0000 0.0000 car 0.87 ", "12 4.5000 0.5000 pedestrian 1.00 "]
