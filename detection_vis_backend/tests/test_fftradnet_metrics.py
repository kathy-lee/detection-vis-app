"""The vectorized FFTRadNet metric helpers must reproduce the reference's shapely loops.

bbox_iou / RA_to_cartesian_box / decode in train/utils.py replace the reference's
implementations (RADIal FFTRadNet utils/metrics.py, dataset/encoder.py), whose per-pair
shapely Polygon intersection inside an O(n^2) NMS took 4.7 min per confidence threshold
on the val split for an untrained model. The reference versions are copied here VERBATIM
and compared on random data; perform_nms and process_predictions_FFT are the reference's
own code on top of bbox_iou, so they are compared end to end as well.
"""
import numpy as np
import pytest
from shapely.geometry import Polygon

from detection_vis_backend.train.utils import (bbox_iou, RA_to_cartesian_box, decode,
                                               perform_nms, process_predictions_FFT)


# ---- verbatim reference implementations ------------------------------------------------

def ref_RA_to_cartesian_box(data):
    L = 4
    W = 1.8

    boxes = []
    for i in range(len(data)):

        x = np.sin(np.radians(data[i][1])) * data[i][0]
        y = np.cos(np.radians(data[i][1])) * data[i][0]

        boxes.append([x - W/2,y,x + W/2,y, x + W/2,y+L,x - W/2,y+L])

    return boxes


def ref_bbox_iou(box1, boxes):

    # currently inspected box
    box1 = box1.reshape((4,2))
    rect_1 = Polygon([(box1[0, 0], box1[0, 1]), (box1[1, 0], box1[1, 1]), (box1[2, 0], box1[2, 1]),
                      (box1[3, 0], box1[3, 1])])
    area_1 = rect_1.area

    # IoU of box1 with each of the boxes in "boxes"
    ious = np.zeros(boxes.shape[0])
    for box_id in range(boxes.shape[0]):
        box2 = boxes[box_id]
        box2 = box2.reshape((4,2))
        rect_2 = Polygon([(box2[0, 0], box2[0, 1]), (box2[1, 0], box2[1, 1]), (box2[2, 0], box2[2, 1]),
                          (box2[3, 0], box2[3, 1])])
        area_2 = rect_2.area

        # get intersection of both bounding boxes
        inter_area = rect_1.intersection(rect_2).area

        # compute IoU of the two bounding boxes
        iou = inter_area / (area_1 + area_2 - inter_area)

        ious[box_id] = iou

    return ious


def ref_perform_nms(valid_class_predictions, valid_box_predictions, nms_threshold):

    # sort the detections such that the entry with the maximum confidence score is at the top
    sorted_indices = np.argsort(valid_class_predictions)[::-1]
    sorted_box_predictions = valid_box_predictions[sorted_indices]
    sorted_class_predictions = valid_class_predictions[sorted_indices]

    for i in range(sorted_box_predictions.shape[0]):
        # get the IOUs of all boxes with the currently most certain bounding box
        try:
            ious = np.zeros((sorted_box_predictions.shape[0]))
            ious[i + 1:] = ref_bbox_iou(sorted_box_predictions[i, :], sorted_box_predictions[i + 1:, :])
        except ValueError:
            break
        except IndexError:
            break

        # eliminate all detections which have IoU > threshold
        overlap_mask = np.where(ious < nms_threshold, True, False)
        sorted_box_predictions = sorted_box_predictions[overlap_mask]
        sorted_class_predictions = sorted_class_predictions[overlap_mask]

    return sorted_class_predictions, sorted_box_predictions


def ref_decode(map, threshold):
    geometry = {"ranges": [512, 896, 1], "resolution": [0.201171875, 0.2], "size": 3}
    statistics = {"reg_mean": [0.4048094369863972, 0.3997392847799934],
                  "reg_std": [0.6968599580482511, 0.6942950877813826]}
    OUTPUT_DIM = (3, 512 // 4, 896 // 4)
    range_bins,angle_bins = np.where(map[0,:,:]>=threshold)
    coordinates = []
    for range_bin,angle_bin in zip(range_bins,angle_bins):
        R = range_bin*4*geometry['resolution'][0] + map[1,range_bin,angle_bin] * statistics['reg_std'][0] + statistics['reg_mean'][0]
        A = (angle_bin-OUTPUT_DIM[2]/2)*4*geometry['resolution'][1] + map[2,range_bin,angle_bin] * statistics['reg_std'][1] + statistics['reg_mean'][1]
        C = map[0,range_bin,angle_bin]
        coordinates.append([R,A,C])
    return coordinates


# ---- fixtures --------------------------------------------------------------------------

def _candidates(rng, n, spread=60.0):
    """(range m, angle deg, confidence) rows; ranges/angles cluster so boxes overlap."""
    r = rng.uniform(5, spread, n)
    a = rng.uniform(-40, 40, n)
    c = rng.uniform(0, 1, n)
    return np.stack((r, a, c), axis=1)


def test_bbox_iou_matches_shapely_on_random_and_degenerate_pairs():
    rng = np.random.default_rng(0)
    boxes = np.asarray(ref_RA_to_cartesian_box(_candidates(rng, 400, spread=30)))
    box1 = boxes[0]
    ref = ref_bbox_iou(box1.copy(), boxes)
    new = bbox_iou(box1, boxes)
    assert np.allclose(new, ref, atol=1e-12, rtol=0)
    # identical box -> 1, far box -> 0; a box shifted by its own width overlaps by
    # float rounding only (~1e-16) and shapely reports that same sliver
    assert bbox_iou(box1, box1[None]) == pytest.approx(1.0)
    far = box1 + np.array([500, 0] * 4)
    assert bbox_iou(box1, far[None]) == 0.0
    touch = box1 + np.array([1.8, 0] * 4)
    assert bbox_iou(box1, touch[None]) == pytest.approx(ref_bbox_iou(box1.copy(), touch[None]), abs=1e-15)
    # the decisions the metric actually takes
    for thr in (0.05, 0.5):
        assert np.array_equal(new < thr, ref < thr)
        assert np.array_equal(new >= thr, ref >= thr)


@pytest.mark.parametrize("seed,n", [(1, 50), (2, 300), (3, 800)])
def test_nms_matches_reference(seed, n):
    rng = np.random.default_rng(seed)
    cand = _candidates(rng, n, spread=40)
    boxes = np.asarray(ref_RA_to_cartesian_box(cand))
    ref_c, ref_b = ref_perform_nms(cand[:, 2].copy(), boxes.copy(), 0.05)
    new_c, new_b = perform_nms(cand[:, 2].copy(), np.asarray(RA_to_cartesian_box(cand)), 0.05)
    assert np.array_equal(new_c, ref_c)
    assert np.allclose(new_b, ref_b, atol=1e-12, rtol=0)
    assert len(new_c) < n


def test_process_predictions_matches_reference_pipeline():
    rng = np.random.default_rng(4)
    cand = _candidates(rng, 500, spread=40)
    out = process_predictions_FFT(cand, confidence_threshold=0.3)
    mask = cand[:, 2] > 0.3
    ref_c, ref_b = ref_perform_nms(cand[mask, 2], np.asarray(ref_RA_to_cartesian_box(cand[mask])), 0.05)
    assert np.array_equal(out[:, 0], ref_c)
    assert np.allclose(out[:, 1:], ref_b, atol=1e-12, rtol=0)


def test_ra_to_cartesian_box_matches_reference():
    rng = np.random.default_rng(5)
    cand = _candidates(rng, 200)
    assert np.allclose(RA_to_cartesian_box(cand), ref_RA_to_cartesian_box(cand), atol=1e-12, rtol=0)
    assert RA_to_cartesian_box(np.zeros((0, 3))) == []


def test_decode_matches_reference():
    rng = np.random.default_rng(6)
    m = np.zeros((3, 128, 224), np.float32)
    m[0] = rng.beta(0.5, 4, (128, 224))
    m[1:] = rng.normal(size=(2, 128, 224))
    ref = np.asarray(ref_decode(m, 0.05))
    new = np.asarray(decode(m, 0.05))
    assert new.shape == ref.shape
    assert np.array_equal(new, ref)
    assert decode(np.zeros((3, 128, 224), np.float32), 0.05) == []
