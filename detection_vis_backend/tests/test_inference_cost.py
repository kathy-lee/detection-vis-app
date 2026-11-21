"""Inference cost of a trained model: FLOPs, per-stage latency and peak GPU memory, as
scripts/measure_inference_cost.py reports them for the baselines cost table."""
import time

import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from detection_vis_backend.train import inference_cost as ic


class _Frames(Dataset):
    def __init__(self, n):
        self.n = n

    def __len__(self):
        return self.n

    def __getitem__(self, i):
        return {"x": torch.full((4,), float(i)), "frame": i}


class _FirstKey(ic.Pipeline):
    def __init__(self):
        self.decoded = []

    def inputs(self, batch, device):
        return batch["x"].to(device).float()

    def postprocess(self, outputs, batch):
        self.decoded.append(int(batch["frame"][0]))
        return outputs.cpu().numpy()


def test_summarize_reports_the_chosen_statistic_in_ms_without_the_warm_up():
    seconds = [9.0, 9.0, 0.001, 0.002, 0.003, 0.004, 0.100]
    assert ic.summarize(seconds, "median", warmup=2) == pytest.approx(3.0)
    assert ic.summarize(seconds, "mean", warmup=2) == pytest.approx(22.0)
    assert ic.summarize(seconds, "p95", warmup=2) == pytest.approx(np.percentile([1, 2, 3, 4, 100], 95))


def test_summarize_rejects_an_unknown_statistic():
    with pytest.raises(ValueError, match="mean, median, p95"):
        ic.summarize([0.1, 0.2], "max", warmup=0)


def test_summarize_needs_frames_beyond_the_warm_up():
    with pytest.raises(ValueError, match="warm-up"):
        ic.summarize([0.1, 0.2], "median", warmup=2)


def test_count_flops_counts_one_flop_per_multiply_add():
    linear = ic.count_flops(nn.Linear(10, 20), torch.zeros(1, 10))
    assert linear["profiled"] == 10 * 20
    conv = ic.count_flops(nn.Conv2d(3, 8, 3, padding=1), torch.zeros(1, 3, 16, 16))
    assert conv["profiled"] == 8 * 16 * 16 * 3 * 3 * 3
    assert conv["total"] == conv["profiled"] and conv["mamba_scan"] == 0 and conv["uncounted"] == []


def test_count_flops_leaves_out_activations_and_normalisation():
    net = nn.Sequential(nn.Linear(10, 20), nn.ReLU(), nn.LayerNorm(20))
    assert ic.count_flops(net, torch.zeros(1, 10))["profiled"] == 200


def test_mamba_scan_flops_formula():
    # 9 per (token, channel, state) for the recurrence, 2 per (token, channel) for the
    # skip and the gate.
    assert ic.mamba_scan_flops(1, 512, 64, 16) == 9 * 512 * 64 * 16 + 2 * 512 * 64
    assert ic.mamba_scan_flops(256, 512, 64, 16) == 256 * ic.mamba_scan_flops(1, 512, 64, 16)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Mamba kernels need CUDA")
def test_count_flops_adds_the_mamba_scan_the_profiler_cannot_see():
    from mamba_ssm import Mamba
    net = Mamba(d_model=32, d_state=16, d_conv=4, expand=2).cuda()
    flops = ic.count_flops(net, torch.zeros(2, 10, 32, device="cuda"))
    assert flops["mamba_scan"] == ic.mamba_scan_flops(2, 10, 64, 16)
    assert flops["profiled"] > 0
    assert flops["total"] == flops["profiled"] + flops["mamba_scan"]


def test_measure_run_times_three_stages_per_frame():
    loader = ic.single_frame_loader(DataLoader(_Frames(8), batch_size=4, shuffle=True))
    pipeline = _FirstKey()
    raw = ic.measure_run(nn.Linear(4, 2), loader, pipeline, torch.device("cpu"))
    assert raw["frames"] == 8
    for stage in ("preprocess_s", "inference_s", "postprocess_s"):
        assert len(raw[stage]) == 8 and all(t >= 0 for t in raw[stage])
    # One frame per sample, in dataset order whatever the source loader's batching was.
    assert pipeline.decoded == list(range(8))
    # Peak memory is a CUDA allocator statistic.
    assert raw["peak_memory_bytes"] == []


def test_measure_run_stops_at_max_frames():
    loader = ic.single_frame_loader(DataLoader(_Frames(8), batch_size=1))
    raw = ic.measure_run(nn.Linear(4, 2), loader, _FirstKey(), torch.device("cpu"), max_frames=3)
    assert raw["frames"] == 3 and len(raw["inference_s"]) == 3


class _SlowDecoder(nn.Module):
    def forward(self, x):
        time.sleep(0.02)
        return x, x, x


class _DarodLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Linear(4, 4)
        self.decoder = _SlowDecoder()

    def forward(self, x):
        return {"decoder_output": self.decoder(self.body(x))}


def test_darod_decoder_time_moves_from_inference_to_postprocessing():
    net = _DarodLike()
    pipeline = ic.DarodPipeline(features=["RD"], train_config={}, split="test", net=net)
    loader = ic.single_frame_loader(DataLoader(_RdFrames(4), batch_size=1))
    raw = ic.measure_run(net, loader, pipeline, torch.device("cpu"))
    assert all(t >= 0.02 for t in raw["postprocess_s"])
    assert all(t < 0.02 for t in raw["inference_s"])


class _RdFrames(_Frames):
    def __getitem__(self, i):
        return {"RD": torch.zeros(4)}


def test_find_checkpoint_takes_the_last_epoch_unless_one_is_named(tmp_path):
    for epoch in (0, 9, 10, 100):
        (tmp_path / f"Net_epoch{epoch:02d}_loss_1.0000_AP_0.5000_AR_0.5000_IOU_0.0000.pth").touch()
    (tmp_path / "train_info.txt").touch()
    assert "_epoch100_" in ic.find_checkpoint(str(tmp_path))
    # 'epoch10' is also the start of 'epoch100'.
    assert "_epoch10_" in ic.find_checkpoint(str(tmp_path), epoch=10)
    with pytest.raises(FileNotFoundError, match="epoch 7"):
        ic.find_checkpoint(str(tmp_path), epoch=7)


def test_summarize_run_builds_the_table_row():
    record = {
        "warmup": 1,
        "flops": {"total": 2_500_000_000},
        "raw": {"preprocess_s": [9.0, 0.004, 0.006], "inference_s": [9.0, 0.010, 0.030],
                "postprocess_s": [9.0, 0.001, 0.003], "peak_memory_bytes": [3 * 2**20, 5 * 2**20, 4 * 2**20]},
    }
    row = ic.summarize_run(record, "mean")
    assert row == {"gflops": 2.5, "preprocess_ms": pytest.approx(5.0), "inference_ms": pytest.approx(20.0),
                   "postprocess_ms": pytest.approx(2.0), "peak_memory_mib": 5.0, "frames": 2}


def test_summarize_run_without_cuda_has_no_peak_memory():
    record = {"warmup": 0, "flops": {"total": 10**9},
              "raw": {"preprocess_s": [0.001], "inference_s": [0.001], "postprocess_s": [0.001],
                      "peak_memory_bytes": []}}
    assert ic.summarize_run(record, "median")["peak_memory_mib"] is None


def test_every_evaluated_pair_has_a_pipeline():
    from detection_vis_backend.train.evaluate import eval_func_dict
    missing = [pair for pair, evaluation in eval_func_dict.items()
               if evaluation is not None and evaluation not in ic.PIPELINE_FOR_EVALUATION]
    assert missing == []


def test_unwired_pair_raises():
    with pytest.raises(ValueError, match="FFTRadNet.*CRUW"):
        ic.make_pipeline("FFTRadNet", "CRUW", ["RD"], {}, "test", net=None)


def test_detection_map_pipeline_decodes_like_the_evaluation():
    from detection_vis_backend.train.utils import decode, process_predictions_FFT
    torch.manual_seed(0)
    detection = torch.zeros(1, 3, 128, 224)
    for row, col in ((20, 50), (60, 100), (100, 180)):
        detection[0, 0, row, col] = 0.9
    outputs = {"Detection": detection, "Segmentation": torch.randn(1, 1, 256, 224)}
    pipeline = ic.make_pipeline("FFTRadNet", "RADIal", ["RD"], {}, "test", net=None)
    objects, freespace = pipeline.postprocess(outputs, {})[0]
    expected = np.asarray(decode(outputs["Detection"].numpy()[0], 0.05))
    assert len(expected) == 3
    expected = process_predictions_FFT(expected, confidence_threshold=0.2)
    np.testing.assert_array_equal(objects, expected)
    np.testing.assert_array_equal(freespace, torch.sigmoid(outputs["Segmentation"]).numpy()[0, 0] >= 0.5)


def test_detection_map_pipeline_reads_adc_or_rd():
    pipeline = ic.make_pipeline("SSMRadNet", "RADIal", ["ADC"], {}, "val", net=None)
    assert pipeline.inputs({"ADC": torch.zeros(1, 2)}, "cpu").shape == (1, 2)
    assert pipeline.inputs({"RD": torch.zeros(1, 3)}, "cpu").shape == (1, 3)


def test_confmap_pipeline_averages_overlapping_windows_and_decodes_stride_frames():
    from detection_vis_backend.train.utils import post_process_single_frame
    train_config = {"win_size": 4, "test_stride": 2, "train_stride": 1, "max_dets": 20, "peak_thres": 0.3, "ols_thres": 0.3}
    rng_grid, agl_grid = np.linspace(0, 25, 16), np.linspace(-1, 1, 16)
    pipeline = ic.ConfmapPipeline(features=["RA"], train_config=train_config, split="test", net=None,
                                  n_class=3, rng_grid=rng_grid, agl_grid=agl_grid)
    rs = np.random.RandomState(0)
    first, second = rs.rand(1, 3, 4, 16, 16), rs.rand(1, 3, 4, 16, 16)
    batch = {"seq_name": ["a"]}
    assert len(pipeline.postprocess(torch.tensor(first), batch)) == 2
    decoded = pipeline.postprocess(torch.tensor(second), batch)
    # The second window starts two frames later: its first two frames overlap the first
    # window's last two, and those averages are what gets decoded.
    for k in range(2):
        averaged = (first[0, :, 2 + k] + second[0, :, k]) / 2
        np.testing.assert_array_equal(
            decoded[k], post_process_single_frame(averaged, train_config, 3, rng_grid, agl_grid))


def test_confmap_pipeline_starts_afresh_on_a_new_sequence():
    from detection_vis_backend.train.utils import post_process_single_frame
    train_config = {"win_size": 4, "test_stride": 2, "train_stride": 1, "max_dets": 20, "peak_thres": 0.3, "ols_thres": 0.3}
    rng_grid, agl_grid = np.linspace(0, 25, 16), np.linspace(-1, 1, 16)
    pipeline = ic.ConfmapPipeline(features=["RA"], train_config=train_config, split="test", net=None,
                                  n_class=3, rng_grid=rng_grid, agl_grid=agl_grid)
    rs = np.random.RandomState(1)
    first, second = rs.rand(1, 3, 4, 16, 16), rs.rand(1, 3, 4, 16, 16)
    pipeline.postprocess(torch.tensor(first), {"seq_name": ["a"]})
    decoded = pipeline.postprocess(torch.tensor(second), {"seq_name": ["b"]})
    np.testing.assert_array_equal(
        decoded[0], post_process_single_frame(second[0, :, 0], train_config, 3, rng_grid, agl_grid))


def test_confmap_pipeline_uses_the_train_stride_on_the_val_split():
    train_config = {"win_size": 4, "test_stride": 2, "train_stride": 1, "max_dets": 20, "peak_thres": 0.3, "ols_thres": 0.3}
    pipeline = ic.ConfmapPipeline(features=["RA"], train_config=train_config, split="val", net=None,
                                  n_class=3, rng_grid=np.linspace(0, 25, 16), agl_grid=np.linspace(-1, 1, 16))
    assert len(pipeline.postprocess(torch.rand(1, 3, 4, 16, 16), {"seq_name": ["a"]})) == 1


def test_segmentation_pipelines_take_the_class_argmax():
    logits = torch.randn(1, 4, 8, 8)
    record = ic.make_pipeline("RECORD", "CARRADA", ["RD"], {}, "test", net=None)
    assert torch.equal(record.postprocess(logits, {}), torch.argmax(logits, dim=1))
    assert record.inputs({"RD": torch.zeros(1, 2), "RA": torch.zeros(1, 3)}, "cpu").shape == (1, 2)

    windows = torch.randn(1, 4, 5, 8, 8)
    trodnet = ic.make_pipeline("T_RODNet", "CARRADA", ["RA"], {}, "test", net=None)
    assert torch.equal(trodnet.postprocess(windows, {}), torch.argmax(windows[:, :, -1], dim=1))

    views = {"rd": torch.randn(1, 4, 8, 8), "ra": torch.randn(1, 4, 8, 8)}
    mvrecord = ic.make_pipeline("MVRECORD", "CARRADA", ["RD", "RA", "AD"], {}, "test", net=None)
    rd, ra = mvrecord.postprocess(views, {})
    assert torch.equal(rd, torch.argmax(views["rd"], dim=1)) and torch.equal(ra, torch.argmax(views["ra"], dim=1))
    assert set(mvrecord.inputs({k: torch.zeros(1) for k in ("RD", "RA", "AD", "rd_mask")}, "cpu")) == {"RD", "RA", "AD"}


def test_cross_attention_pipeline_matches_the_evaluation_peaks():
    from detection_vis_backend.train.evaluate import _detect_and_associate
    from detection_vis_backend.train.utils import create_default
    torch.manual_seed(0)
    outputs = {"pred_mask": torch.randn(1, 3, 256, 256) - 2.0, "pred_center": torch.randn(1, 2, 256, 256)}
    gt_mask = torch.zeros(1, 3, 256, 256)
    pipeline = ic.make_pipeline("RadarCrossAttention", "CARRADA", ["RD", "RA", "AD"], {}, "test", net=None)
    pred_idx, pred_intent = pipeline.postprocess(outputs, {"gt_mask": gt_mask})

    mask, peak_cls = create_default(gt_mask.size(), kernel_window=(3, 5))
    pred_c = 8 * (torch.sigmoid(outputs["pred_center"]) - 0.5)
    _, expected_idx, expected_intent = _detect_and_associate(
        gt_mask, torch.sigmoid(outputs["pred_mask"]), pred_c, mask, peak_cls, "cpu")
    assert len(expected_idx) > 0
    assert torch.equal(pred_idx, expected_idx) and torch.equal(pred_intent, expected_intent)
