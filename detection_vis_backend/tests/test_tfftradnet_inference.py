"""T-FFTRadNet in the app's inference display: RADIal detections on the camera image as
FFTRadNet's, RADDet detections as small boxes on the range-azimuth (RA) view."""
import glob
import os
import pickle

import numpy as np
import pytest
import torch

from detection_vis_backend.paths import dataset_path

REAL_RADDET = dataset_path("RADDet_reform_dataset")


class _StubNet(torch.nn.Module):
    """Records its input; returns fixed maps."""

    def __init__(self, outputs):
        super().__init__()
        self.outputs, self.seen = outputs, None

    def forward(self, x):
        self.seen = x
        return self.outputs


@pytest.mark.skipif(not os.path.isdir(os.path.join(REAL_RADDET, "gt")), reason="RADDet absent")
def test_point_pixel_lands_on_its_ground_truth_box():
    # The point labels come from cart_boxes; the RA boxes are stored separately. Both are
    # rounded box centres, so single objects can be a few pixels apart; the bulk agree.
    from detection_vis_backend.datasets.adapters.raddet import raddet_point_labels, raddet_point_to_ra_pixel
    rows, cols = [], []
    for f in sorted(glob.glob(os.path.join(REAL_RADDET, "gt", "*.pickle")))[:300]:
        with open(f, "rb") as fh:
            gt = pickle.load(fh)
        for (r, phi, _), box in zip(raddet_point_labels(gt), gt["boxes"]):
            row, col = raddet_point_to_ra_pixel(r, phi)
            rows.append(abs(row - box[0]))
            cols.append(abs(col - box[1]))
    assert len(rows) > 500
    assert np.median(rows) <= 1 and np.median(cols) <= 1
    assert np.percentile(rows, 95) <= 3 and np.percentile(cols, 95) <= 5


def test_point_pixel_geometry():
    from detection_vis_backend.datasets.adapters.raddet import raddet_point_to_ra_pixel
    assert raddet_point_to_ra_pixel(0.0, 0.0) == (255, 128)            # radar at the bottom centre
    assert raddet_point_to_ra_pixel(25.0, 0.0) == (128, 128)
    assert raddet_point_to_ra_pixel(25.0, 30.0) == (128, 192)          # sin 30 deg = 0.5
    assert raddet_point_to_ra_pixel(80.0, -90.0) == (0, 0)             # clipped to the image


def test_class_index_maps_to_the_parser_order():
    from detection_vis_backend.datasets.adapters.raddet import RADDET_CLASSES, raddet_class_to_parser
    from detection_vis_backend.datasets.dataset import RADDetDataset
    all_classes = RADDetDataset().data_stats["all_classes"]
    assert raddet_class_to_parser(RADDET_CLASSES.index("person")) == 0
    for i, name in enumerate(RADDET_CLASSES):
        assert all_classes[raddet_class_to_parser(i)] == name


def test_raddet_detections_keep_what_the_scorer_keeps():
    # Same confidence filter and NMS as the scorer's process_predictions_FFT, but the kept
    # rows stay [R, A, confidence, class] so they can be drawn with their class.
    from detection_vis_backend.train.utils import process_predictions_FFT, raddet_detections
    rng = np.random.default_rng(0)
    points = np.column_stack([rng.uniform(2, 48, 60), rng.uniform(-40, 40, 60), rng.uniform(0, 1, 60),
                              rng.integers(0, 6, 60)])
    points[1, :2] = points[0, :2] + [0.3, 0.2]                         # an overlapping pair
    kept = raddet_detections(points, threshold=0.2, nms_threshold=0.3)
    reference = process_predictions_FFT(points[:, :3], confidence_threshold=0.2, nms_threshold=0.3)
    assert kept.shape[1] == 4 and len(kept) == len(reference) < (points[:, 2] > 0.2).sum()
    np.testing.assert_allclose(np.sort(kept[:, 2]), np.sort(reference[:, 0]))
    assert set(map(tuple, kept)) <= set(map(tuple, points))


def test_radial_display_reads_the_feature_key():
    from detection_vis_backend.train.inference import detection_map_image_boxes
    det = torch.zeros(1, 3, 128, 224)
    det[0, 0, 40, 112] = 0.9
    net = _StubNet({"Detection": det, "Segmentation": torch.zeros(1, 1, 256, 224)})
    data = {"ADC": np.ones((32, 512, 256), np.float32), "RD": np.zeros((32, 512, 256), np.float32)}
    boxes = detection_map_image_boxes(net, data, "ADC", "cpu")
    assert net.seen.shape == (1, 32, 512, 256) and net.seen.sum() > 0
    assert len(boxes) == 1 and len(boxes[0]) == 4


def test_raddet_display_draws_class_boxes_on_the_ra_view():
    from detection_vis_backend.datasets.adapters.raddet import raddet_point_to_ra_pixel
    from detection_vis_backend.train.inference import tfftradnet_raddet_ra_boxes
    det = torch.zeros(1, 3, 64, 128)
    det[0, 0, 20, 64] = 0.99
    cls = torch.full((1, 6, 64, 128), -5.0)
    cls[0, 4] = 5.0                                                    # 'person' in the reference's order
    net = _StubNet({"Detection": det, "class": cls})
    boxes = tfftradnet_raddet_ra_boxes(net, {"RAD": np.zeros((64, 256, 256), np.float32)}, "RAD", "cpu")
    row, col = raddet_point_to_ra_pixel(20 * 4 * 0.1953125 + 0.38584477, 0.69962384)
    assert boxes == [[col - 3, row - 3, col + 3, row + 3, 0]]          # 'person' is the parser's class 0


def test_every_evaluated_pair_has_an_inference_cost_pipeline():
    from detection_vis_backend.train import inference_cost as ic
    from detection_vis_backend.train.evaluate import TFFTRadNet_radial_evaluation, TFFTRadNet_raddet_evaluation
    assert ic.PIPELINE_FOR_EVALUATION[TFFTRadNet_radial_evaluation] is ic.DetectionMapPipeline
    assert TFFTRadNet_raddet_evaluation in ic.PIPELINE_FOR_EVALUATION
