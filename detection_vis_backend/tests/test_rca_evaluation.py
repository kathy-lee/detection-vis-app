"""RadarCrossAttention evaluator: round-trip on ground truth, and the two distance thresholds.

The evaluator is a port of the reference's evaluation() (Utils/val_utils.py). Feeding it
the ground-truth Gaussian maps as if they were predictions must score AP 1.0 -- the
cheapest check that peak detection, association and the P/R integral agree with the
targets they are scored against (the campaign's standard check #5). Both heads that the
paper's headline model runs WITHOUT (heading) or that the network reports as the int 0
are exercised: the pre-2026-09-07 evaluator raised TypeError on `torch.all(0)` the moment
the heading head was off.
"""
import numpy as np
import pytest
import torch

from detection_vis_backend.datasets.utils import bi_var_gauss
from detection_vis_backend.train.evaluate import RadarCrossAttention_evaluation

RANGE_BIN_M = 50.0 / 255      # pol2cord: range_array = linspace(50, 0, 256)

# One pedestrian and one car, far enough apart that the 2 m NMS in association() cannot
# merge them (~12 m radial, ~80 angle bins). Cyclist absent on purpose: its AP must come
# back nan and be excluded from the mean rather than poison it.
ANNO = {"cls": [1.0, 3.0], "mu_r": [120.0, 60.0], "mu_a": [100.0, 180.0],
        "sigma_r": [3.0, 4.0], "sigma_a": [5.0, 8.0], "sigma_cov": [0.0, 0.2]}


class _Loader:
    def __init__(self, batches):
        self.batches = batches
        self.dataset = list(range(sum(b["RA"].shape[0] for b in batches)))

    def __len__(self):
        return len(self.batches)

    def __iter__(self):
        return iter(self.batches)


class _EchoNet(torch.nn.Module):
    """Returns the ground truth (optionally shifted along range) as logits; no optional heads."""

    def __init__(self, gt, shift_bins=0):
        super().__init__()
        self.gt = gt
        self.shift = shift_bins

    def forward(self, inputs):
        pred = torch.roll(self.gt, self.shift, dims=2) if self.shift else self.gt
        return {"pred_mask": torch.logit(pred.clamp(1e-4, 1 - 1e-4)),
                "pred_center": 0, "pred_orent": 0}

    def get_loss(self, pred, target, cfg):
        return torch.tensor(0.0)


N_FRAMES = 10
# The reference's AP is np.trapz(P, R) over the confidence-ranked curve with NO (R=0, P=1)
# origin point, so a perfect detector scores exactly 1 - 1/G for G ground truths -- a
# 1/G under-report that is negligible on a real split (G in the hundreds) but exact here.
# Pinned rather than "fixed": it is the published metric's definition.
PERFECT = 1.0 - 1.0 / N_FRAMES


def _batch(n=N_FRAMES):
    gt = torch.from_numpy(np.stack([bi_var_gauss(ANNO)] * n))          # (n, 3, 256, 256)
    return {"RA": torch.zeros(n, 1, 256, 256), "RD": torch.zeros(n, 1, 256, 64),
            "AD": torch.zeros(n, 1, 64, 256), "gt_mask": gt,
            "gt_center_map": torch.zeros(n, 2, 256, 256)}


TRAIN_CFG = {"dataloader": {"val": {"batch_size": N_FRAMES}, "test": {"batch_size": N_FRAMES}}}


def _run(shift_bins, eval_type="val"):
    batch = _batch()
    net = _EchoNet(batch["gt_mask"], shift_bins)
    return RadarCrossAttention_evaluation(net, _Loader([batch]), TRAIN_CFG, ["RD", "RA", "AD"],
                                          None, eval_type, "cpu")


def test_ground_truth_scores_perfectly_at_both_thresholds():
    res = _run(shift_bins=0)
    assert res["mAP"] == pytest.approx(PERFECT)
    assert res["mAP_1m"] == pytest.approx(PERFECT)
    assert res["AP2m_ped"] == pytest.approx(PERFECT) and res["AP2m_car"] == pytest.approx(PERFECT)
    assert res["AP1m_ped"] == pytest.approx(PERFECT) and res["AP1m_car"] == pytest.approx(PERFECT)
    # absent class: excluded, not averaged in as 0 or nan
    assert np.isnan(res["AP2m_cyc"]) and np.isnan(res["AP1m_cyc"])
    # the reference's plateau/checkpoint metric: sigmoid(logit(clamp(gt))) ~ gt. The clamp
    # floor of 1e-4 over ~2M background pixels leaves ~0.02; an all-zero map would score
    # sum(gt^2) ~ 500 on this batch.
    assert res["val_mse"] < 0.1
    assert "loss" in res


def test_thresholds_discriminate_a_localisation_error():
    """A 7-bin range shift is 1.37 m: inside 2 m, outside 1 m."""
    shift = 7
    assert 1.0 < shift * RANGE_BIN_M < 2.0
    res = _run(shift_bins=shift)
    assert res["mAP"] == pytest.approx(PERFECT)
    assert res["mAP_1m"] == pytest.approx(0.0)


def test_test_split_returns_no_val_only_keys():
    res = _run(shift_bins=0, eval_type="test")
    assert "val_mse" not in res and "loss" not in res
    assert res["mAP"] == pytest.approx(PERFECT)
