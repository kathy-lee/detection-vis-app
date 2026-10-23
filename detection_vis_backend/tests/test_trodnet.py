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
        return json.load(f)["CRUW"]


def test_cruw_config_resolves_unchanged():
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from smoke_sweep import resolve_configs
    m, t = resolve_configs("T_RODNet", "CRUW", ["RA"])
    assert {k: v for k, v in m.items() if k != "class"} == {
        "n_class": 3, "embed_dim": 64, "win_size": 4, "img_size": [8, 64, 64], "patch_size": [2, 2, 2],
        "depths": [2, 2, 2], "depths_decoder": [2, 2, 2], "num_heads": [2, 4, 8], "drop_path_rate": 0.1, "ape": True}
    assert t["win_size"] == 16 and t["dataloader"]["train"]["batch_size"] == 4


def test_carrada_config_is_the_papers_recipe():
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    from smoke_sweep import resolve_configs, LIVE_METRICS
    m, t = resolve_configs("T_RODNet", "CARRADA", ["RA"])
    assert (m["n_class"], m["in_chans"], m["win_size"], m["img_size"]) == (4, 1, 4, [2, 128, 128])
    assert (m["embed_dim"], m["depths"], m["num_heads"]) == (64, [2, 2, 2], [2, 4, 8])
    assert t["win_size"] == 4 and t["num_epochs"] == 100 and t["dataloader"]["train"]["batch_size"] == 8
    assert t["transformations"] == [] and t["norm_type"] == "tvt" and t["losses"] == {"type": "mse"}
    assert t["lr_scheduler"] == {"type": "cosine", "min_lr": 0} and t["optimizer"]["lr"] == 0.0001
    assert "test_checkpoint" not in t and "early_stop" not in t
    assert LIVE_METRICS["T_RODNet", "CARRADA"] == ("mIoU", "mDice")
    from smoke_sweep import RUNS, run_id
    assert any(run_id(r) == "T_RODNet/CARRADA" for r in RUNS)


def test_trodnet_network_through_the_factory_and_loss_contract():
    from detection_vis_backend.networks.network import NetworkFactory
    net = NetworkFactory().get_instance("T_RODNet", _model_cfg()).eval()
    net.init_lossfunc({"losses": {"type": "mse"}}, torch.device("cpu"))
    with torch.no_grad():
        out = net(torch.randn(1, 2, 16, 128, 128))
    assert out.shape == (1, 3, 16, 128, 128)
    loss = net.get_loss(out, {"gt_mask": torch.rand(1, 3, 16, 128, 128)}, {})
    assert torch.isfinite(loss) and net.n_class == 3 and net.class_names == ["pedestrian", "cyclist", "car"]


def test_carrada_build_maps_a_4_frame_ra_window_to_4_class_masks():
    # CARRADA RA is 1 real channel at 256 x 256, 4 frames, 4 classes (background included):
    # the paper's 4 x 1 x 256 x 256 input, on the same body.
    from detection_vis_backend.networks.network import T_RODNet
    net = T_RODNet(n_class=4, win_size=4, img_size=(2, 128, 128), in_chans=1).eval()
    with torch.no_grad():
        out = net(torch.zeros(1, 1, 4, 256, 256))
    assert out.shape == (1, 4, 4, 256, 256)


def test_carrada_scoring_reads_the_windows_last_frame(tmp_path):
    from detection_vis_backend.train.evaluate import eval_func_dict

    def one_hot(cls_map):
        return torch.nn.functional.one_hot(cls_map, 4).permute(0, 4, 1, 2, 3).float()

    g = torch.Generator().manual_seed(0)
    batches = [torch.randint(0, 4, (2, 4, 8, 8), generator=g) for _ in range(2)]  # (B, win, H, W)

    class LastFrameOnly(nn.Module):
        n_class = 4

        def forward(self, x):
            cls = self.labels.clone()
            cls[:, :3] = (cls[:, :3] + 1) % 4   # every frame but the last is wrong
            return one_hot(cls)

    net = LastFrameOnly()
    loader = []
    for b in batches:
        loader.append({"RA": torch.zeros(2, 1, 4, 8, 8), "gt_mask": one_hot(b)})

    class Loader(list):
        dataset = range(4)

        def __iter__(self):
            for d in list.__iter__(self):
                net.labels = d["gt_mask"].argmax(1)
                yield d

    cfg = {"dataloader": {"test": {"batch_size": 2}}}
    res = eval_func_dict["T_RODNet", "CARRADA"](net, Loader(loader), cfg, ["RA"], str(tmp_path), "test", torch.device("cpu"))
    assert res["mIoU"] == pytest.approx(1.0) and res["mDice"] == pytest.approx(1.0)
    assert len(res["iou_by_class"]) == 4 and len(res["dice_by_class"]) == 4


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
    assert feats["features"] == ["RA"] and feats["datasets"] == ["CRUW", "CARRADA"]


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


def test_infer_routes_trodnet_through_the_rodnet_confmap_branch():
    """A model trained through the app must be usable for inference in the app: the
    RODNet branch handles a single confidence-map tensor, which is what T_RODNet returns."""
    from detection_vis_backend.train import inference
    src = inspect.getsource(inference.infer)
    branch = src.split('elif model_type in ("RODNet_CDC"')[1].split("elif model_type in")[0]
    assert '"T_RODNet"' in src.split('elif model_type in ("RODNet_CDC"')[1].split(":")[0]
    assert "post_process_single_frame(confmap_pred[0,:,0,:,:]" in branch


def test_trodnet_carrada_readout_shows_the_windows_last_frame():
    import numpy as np
    from detection_vis_backend.train.inference import trodnet_carrada_readout
    logits = np.zeros((4, 4, 3, 3))      # (class, frame, H, W)
    logits[2, 3, 0, 0] = 5.0             # cyclist on the last frame only
    logits[1, 0, 0, 0] = 9.0             # a pedestrian on the first frame must not show
    gt = np.zeros((4, 4, 3, 3))
    gt[0] = 1.0
    gt[3, 3, 1, 1], gt[0, 3, 1, 1] = 1.0, 0.0
    out = trodnet_carrada_readout(logits, gt, viz_idx=13)
    assert out["viz_idx"] == 13
    assert np.array(out["pred_mask"])[0, 0] == 2 and np.array(out["gt_mask"])[1, 1] == 3


def test_infer_routes_trodnet_on_carrada_to_the_segmentation_readout():
    from detection_vis_backend.train import inference
    src = inspect.getsource(inference.infer)
    seg = src.index('model_type == "T_RODNet" and dataset_type == "CARRADA"')
    assert seg < src.index('elif model_type in ("RODNet_CDC"')
    assert "trodnet_carrada_readout(" in src[seg:src.index('elif model_type in ("RODNet_CDC"')]


def test_random_splitmode_builds_adapters_for_migrated_datasets():
    """The random split is RadarCrossAttention's published protocol on CARRADA, so it has
    to serve adapter-served datasets: one adapter per file and split, never a dataset
    copy configured with prepare_for_train."""
    from detection_vis_backend.train.train import CreateDataLoaders
    src = inspect.getsource(CreateDataLoaders)
    branch = src.split("elif train_config['dataloader']['splitmode'] == 'random':")[1]
    branch = branch.split("        else:\n            ValueError(")[0]
    assert "random splitmode cannot serve" not in branch
    assert "_split_source(file, 'train')" in branch and "_split_source(file, split_type)" in branch
    helper = src.split("def _split_source(")[1].split("\n\n")[0]
    assert "if file['parse'] in MIGRATED:" in helper and "make_sample_adapter(" in helper


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
