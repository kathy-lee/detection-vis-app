"""FFTRadNet must evaluate the way the reference does: cheap per epoch, full on test.

The reference (RADIal FFTRadNet) has two evaluations and uses them for different things:

  run_evaluation      utils/evaluation.py:6   -- called EVERY epoch from 1-Train.py:190,
                                                 one confidence threshold (0.2) via Metrics
  run_FullEvaluation  utils/evaluation.py:67  -- called ONCE from 3-Evaluation.py on the
                                                 TEST split, GetFullMetrics sweeping
                                                 0.1..0.9 -- the paper's Table 2

Registering the full sweep for both (as this repo did until 2026-09-08) costs 9x per
epoch and reports a quantity the reference never reports per-epoch. These tests pin the
dispatch and the two metric contracts.
"""
import numpy as np
import pytest
import torch

from detection_vis_backend.train import evaluate as ev


class _Net(torch.nn.Module):
    """Detection head with a sigmoid (as the real one has) plus a segmentation logit map."""

    def forward(self, x):
        b = x.shape[0]
        det = torch.zeros(b, 3, 128, 224)
        det[:, 0] = 0.01                      # low confidence everywhere -> few candidates
        det[:, 0, 40, 100] = 0.9              # one strong detection
        return {"Detection": det, "Segmentation": torch.zeros(b, 1, 256, 224)}

    def get_loss(self, pred, target, cfg):
        return torch.tensor(1.5)

    def eval(self):
        return self


class _Loader:
    def __init__(self, n=2, b=2):
        # box_label rows: [Range, Angle, Doppler, laser_x, laser_y, laser_z, x1,y1,x2,y2]
        box = np.array([[20.0, 0.0, 1.0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)
        self.batches = [{"RD": torch.zeros(b, 32, 512, 256),
                         "encoded_label": torch.zeros(b, 3, 128, 224),
                         "seg_label": torch.zeros(b, 256, 224),
                         "box_label": [box] * b} for _ in range(n)]
        self.dataset = list(range(n * b))

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


CFG = {"dataloader": {"val": {"batch_size": 2}, "test": {"batch_size": 2}},
       "losses": {"weight": [1, 100, 100]}}


def test_val_uses_the_cheap_single_threshold_path(monkeypatch):
    calls = []
    monkeypatch.setattr(ev, "GetFullMetrics", lambda *a, **k: calls.append("full") or (0, 0, 0))
    res = ev.FFTRadNet_evaluation(_Net(), _Loader(), CFG, ["RD"], None, "val", "cpu")
    assert calls == [], "per-epoch validation must not run the 0.1..0.9 threshold sweep"
    assert set(res) == {"loss", "mAP", "mAR", "mIoU"}
    # normalized, not the reference's raw running sum
    assert res["loss"] == pytest.approx(1.5 * 2 * 2 / 4)


def test_test_uses_the_full_threshold_sweep(monkeypatch):
    calls = []
    real = ev.GetFullMetrics
    monkeypatch.setattr(ev, "GetFullMetrics", lambda *a, **k: (calls.append("full"), real(*a, **k))[1])
    res = ev.FFTRadNet_evaluation(_Net(), _Loader(), CFG, ["RD"], None, "test", "cpu")
    assert calls == ["full"], "test evaluation must run GetFullMetrics (the paper's Table 2)"
    assert set(res) >= {"mAP", "mAR", "F1_score", "mIoU"}
    assert "loss" not in res          # no loss on the test split


def test_full_metrics_sweeps_the_reference_thresholds(monkeypatch):
    """0.1..0.9, as utils/metrics.py:120 does -- not 0.2, which dropped the
    lowest-precision point and overstated the mean."""
    seen = []
    real = ev.process_predictions_FFT
    monkeypatch.setattr(ev, "process_predictions_FFT",
                        lambda p, confidence_threshold=0.1, **k: (seen.append(round(confidence_threshold, 1)),
                                                                 real(p, confidence_threshold, **k))[1])
    preds = [np.array([[20.0, 0.0, 0.95]])]
    labels = [np.array([[20.0, 0.0, 1.0, 0, 0, 0, 0, 0, 0, 0]], dtype=np.float32)]
    ev.GetFullMetrics(preds, labels, range_min=5, range_max=100, IOU_threshold=0.5)
    assert sorted(set(seen)) == [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
