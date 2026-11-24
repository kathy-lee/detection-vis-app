"""T-FFTRadNet scoring: on RADIal FFTRadNet's scorer plus F1; on RADDet the reference repo's
(jgiroux8/T_FFTRadNet 68bc041) class-wise metrics (GetAPMetrics), per-epoch detection
metrics (Metrics / GetDetMetrics) and generalized detection metrics (GetFullMetrics)."""
import importlib.util
import io
import os
import re
from contextlib import redirect_stdout

import numpy as np
import pytest
import torch

from detection_vis_backend.paths import dataset_path

REFERENCE_METRICS = dataset_path("tfftradnet_reference", "T_FFTRadNet", "RADDet", "utils", "metrics.py")
needs_reference = pytest.mark.skipif(not os.path.isfile(REFERENCE_METRICS), reason="reference clone absent")


def _reference_metrics():
    spec = importlib.util.spec_from_file_location("reference_raddet_metrics", REFERENCE_METRICS)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fixture():
    """Decoded predictions [R, A, confidence, class] and labels [R, A, class] over 4 frames:
    hits of several classes at high and low confidence, a miss, a wrong class, a frame
    without ground truth and a frame without predictions."""
    preds = [np.array([[10.0, 5.0, 0.97, 2], [10.2, 5.1, 0.6, 2], [30.0, -20.0, 0.98, 4], [45.0, 0.0, 0.3, 0]]),
             np.array([[20.0, 10.0, 0.99, 5], [12.0, -3.0, 0.96, 2], [60.0, 0.0, 0.99, 2]]),
             np.array([[25.0, 0.0, 0.5, 1]]),
             np.zeros((0, 4))]
    labels = [np.array([[10.1, 5.0, 2], [30.0, -20.2, 4], [45.0, 0.1, 0]]),
              np.array([[20.0, 10.0, 3], [12.0, -3.0, 2]]),
              np.zeros((0, 3)),
              np.array([[8.0, 1.0, 2]])]
    return preds, labels


def _parse_reference_ap(printed):
    values = [float(v) for v in re.findall(r"(?:AP|AR|F1 score):\s*([-\d.e]+)", printed)]
    per_class = [values[i:i + 3] for i in range(0, 18, 3)]
    m_ap = float(re.search(r"^mAP:\s*([-\d.e]+)", printed, re.M).group(1))
    m_ar = float(re.search(r"^mAR:\s*([-\d.e]+)", printed, re.M).group(1))
    return per_class, m_ap, m_ar


@needs_reference
def test_class_metrics_match_the_reference_on_a_fixture():
    from detection_vis_backend.datasets.adapters.raddet import RADDET_CLASSES
    from detection_vis_backend.train.evaluate import raddet_class_metrics
    preds, labels = _fixture()
    out = io.StringIO()
    with redirect_stdout(out):
        _reference_metrics().GetAPMetrics(preds, labels, range_min=0, range_max=50, IOU_threshold=0.5)
    per_class, m_ap, m_ar = _parse_reference_ap(out.getvalue())
    ours = raddet_class_metrics(preds, labels)
    for name, theirs in zip(RADDET_CLASSES, per_class):
        np.testing.assert_allclose(ours[name], theirs, rtol=1e-6, atol=1e-7, err_msg=name)
    assert ours["mAP"] == pytest.approx(m_ap) and ours["mAR"] == pytest.approx(m_ar)
    assert ours["car"][0] > 0          # the fixture exercises a non-trivial class


@needs_reference
def test_detection_metrics_match_the_reference_on_a_fixture():
    from detection_vis_backend.train.evaluate import raddet_detection_metrics
    preds, labels = _fixture()
    metrics = _reference_metrics().Metrics()
    for p, l in zip(preds, labels):
        metrics.update(p, l, threshold=0.2, range_min=0, range_max=50)
    assert raddet_detection_metrics(preds, labels) == pytest.approx(metrics.GetMetrics())


@needs_reference
@pytest.mark.parametrize("iou", [0.1, 0.5, 0.9])
def test_full_metrics_match_the_reference_on_a_fixture(iou):
    from detection_vis_backend.train.evaluate import raddet_full_metrics
    preds, labels = _fixture()
    with redirect_stdout(io.StringIO()):
        _, p, r, f1, range_error, angle_error = _reference_metrics().GetFullMetrics(
            preds, labels, range_min=0, range_max=50, IOU_threshold=iou)
    ours = raddet_full_metrics(preds, labels, iou=iou)
    assert ours["precision"] == pytest.approx(p) and ours["recall"] == pytest.approx(r)
    assert ours["F1"] == pytest.approx(f1)
    assert ours["range_error"] == pytest.approx(range_error, nan_ok=True)
    assert ours["angle_error"] == pytest.approx(angle_error, nan_ok=True)


def test_class_ap_is_precision_at_confidence_095():
    # The reference resets its precision list inside the confidence loop, so a class's AP
    # is its precision at the last confidence, 0.95: a hit at 0.9 does not count.
    from detection_vis_backend.train.evaluate import raddet_class_metrics
    labels = [np.array([[10.0, 0.0, 2]])]
    assert raddet_class_metrics([np.array([[10.0, 0.0, 0.9, 2]])], labels)["car"][:2] == [0, 0]
    assert raddet_class_metrics([np.array([[10.0, 0.0, 0.97, 2]])], labels)["car"][:2] == [1.0, 1.0]


def test_metrics_are_zero_not_nan_without_predictions():
    from detection_vis_backend.train.evaluate import (raddet_class_metrics, raddet_detection_metrics,
                                                      raddet_full_metrics)
    preds = [np.zeros((0, 4)), np.zeros((0, 4))]
    labels = [np.array([[10.0, 0.0, 2]]), np.array([[20.0, 3.0, 4]])]
    cls = raddet_class_metrics(preds, labels)
    assert cls["mAP"] == 0 and cls["mAR"] == 0 and cls["car"] == [0, 0, 0]
    assert raddet_detection_metrics(preds, labels) == (0, 0)
    full = raddet_full_metrics(preds, labels, iou=0.5)
    assert (full["precision"], full["recall"], full["F1"]) == (0, 0, 0)


def test_metrics_are_finite_for_frames_without_ground_truth():
    from detection_vis_backend.train.evaluate import (raddet_class_metrics, raddet_detection_metrics,
                                                      raddet_full_metrics)
    preds = [np.array([[10.0, 0.0, 0.99, 2]]), np.zeros((0, 4))]
    labels = [np.zeros((0, 3)), np.zeros((0, 3))]
    cls = raddet_class_metrics(preds, labels)
    assert np.isfinite([cls["mAP"], cls["mAR"]]).all()
    assert raddet_detection_metrics(preds, labels) == (0, 0)
    full = raddet_full_metrics(preds, labels, iou=0.5)
    assert np.isfinite([full["precision"], full["recall"], full["F1"]]).all()


class _StubNet(torch.nn.Module):
    """Returns fixed maps; the loss is a constant."""

    def __init__(self, outputs):
        super().__init__()
        self.outputs = outputs

    def forward(self, x):
        return {k: v[:x.shape[0]] for k, v in self.outputs.items()}

    def get_loss(self, outputs, data, config):
        return torch.tensor(1.5)


def _raddet_batch(n):
    det = torch.zeros(n, 3, 64, 128)
    det[:, 0, 20, 64] = 0.99                        # one confident cell, class 'car'
    cls = torch.full((n, 6, 64, 128), -5.0)
    cls[:, 2] = 5.0
    data = {"RAD": torch.zeros(n, 64, 256, 256), "encoded_label": torch.zeros(n, 3, 64, 128),
            "class_map": torch.zeros(n, 6, 64, 128),
            "box_label": [torch.tensor([[20 * 4 * 0.1953125 + 0.38584477, 0.69962384, 2.0]]) for _ in range(n)]}
    return {"Detection": det, "class": cls}, data


def test_raddet_val_reports_select_score():
    from detection_vis_backend.train.evaluate import TFFTRadNet_raddet_evaluation
    outputs, data = _raddet_batch(2)
    loader = [data]

    class Loader(list):
        dataset = [0, 1]

    result = TFFTRadNet_raddet_evaluation(_StubNet(outputs), Loader(loader), {}, ["RAD"], "", "val")
    assert {"loss", "mAP", "mAR", "mIoU", "det_AP", "det_AR", "select_score"} <= set(result)
    assert result["loss"] == pytest.approx(1.5)
    assert result["det_AP"] == 1.0 and result["mAP"] == pytest.approx(1 / 6)
    assert result["select_score"] == pytest.approx(result["mAP"] + result["det_AP"])


def test_raddet_test_adds_class_and_iou_breakdown():
    from detection_vis_backend.train.evaluate import TFFTRadNet_raddet_evaluation
    outputs, data = _raddet_batch(1)
    result = TFFTRadNet_raddet_evaluation(_StubNet(outputs), [data], {}, ["RAD"], "", "test")
    assert "loss" not in result
    assert result["AP_car"] == 1.0 and result["AP_bus"] == 0
    assert {f"det_AP@{t:.1f}" for t in np.arange(0.1, 0.95, 0.1)} <= set(result)
    assert {"det_F1", "range_error", "angle_error"} <= set(result)


def test_radial_val_reports_f1():
    from detection_vis_backend.train.evaluate import TFFTRadNet_radial_evaluation
    det = torch.zeros(2, 3, 128, 224)
    det[:, 0, 40, 100] = 0.9
    outputs = {"Detection": det, "Segmentation": torch.zeros(2, 1, 256, 224)}
    data = {"RD": torch.zeros(2, 32, 512, 256), "encoded_label": torch.zeros(2, 3, 128, 224),
            "seg_label": torch.zeros(2, 256, 224), "box_label": [np.zeros((0, 10)), np.zeros((0, 10))]}

    class Loader(list):
        dataset = [0, 1]

    result = TFFTRadNet_radial_evaluation(_StubNet(outputs), Loader([data]), {}, ["RD"], "", "val")
    p, r = result["mAP"], result["mAR"]
    assert result["F1"] == (2 * p * r / (p + r) if p + r else 0.0)
    assert np.isfinite(result["F1"])


def test_registry_has_both_pairs():
    from detection_vis_backend.train.evaluate import (TFFTRadNet_radial_evaluation, TFFTRadNet_raddet_evaluation,
                                                      eval_func_dict, network_types)
    assert "T_FFTRadNet" in network_types
    assert eval_func_dict["T_FFTRadNet", "RADIal"] is TFFTRadNet_radial_evaluation
    assert eval_func_dict["T_FFTRadNet", "RADDetDataset"] is TFFTRadNet_raddet_evaluation
