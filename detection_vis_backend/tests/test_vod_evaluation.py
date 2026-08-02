import os
import pytest

VOD_ROOT = "/home/kangle/dataset/ViewOfDelft"
GT_DIR = os.path.join(VOD_ROOT, "radar", "training", "label_2")


@pytest.mark.skipif(not os.path.isdir(GT_DIR), reason="VoD dataset not on this machine")
def test_vod_eval_gt_as_detections_scores_100(tmp_path):
    """Feed the evaluator ground truth as its own detections (nudged 1 cm):
    every class must score 100 AP. Validates the copied devkit code end-to-end
    (numba jit included) and pins the file format: VoD labels already carry 16
    fields (col 2 is a track id, col 16 a score of 1), and our prediction
    writer must produce exactly 16 fields, because the parser zeroes all
    scores on any other width.

    Two non-obvious requirements, learned the hard way:
    - The diagonal nudge: the devkit's CPU rotated-IoU drops degenerate
      polygon-clipping cases -- exactly coincident boxes score 0, and a shift
      along one axis still fails for boxes whose yaw aligns with it (parallel
      overlapping edges). Harmless for real predictions, fatal for a verbatim
      gt copy.
    - The full val split: R11 AP fills at most one recall column per
      detection, so a class with < 41 gts caps far below 100 (18 Cars ->
      exactly 45.45) no matter how perfect the detections are. Small-sample
      slices cannot assert ~100.

    Runtime ~75 s after the ~30 s numba jit warmup -- also the expected
    per-epoch evaluation cost during training.
    """
    from detection_vis_backend.train.vod_evaluation import Evaluation

    with open(os.path.join(VOD_ROOT, "radar", "ImageSets", "val.txt")) as f:
        frames = [l.strip() for l in f]

    pred_dir = tmp_path / "preds"
    pred_dir.mkdir()
    for name in frames:
        with open(os.path.join(GT_DIR, name + ".txt")) as f:
            lines = [l.split() for l in f if l.strip()]
        for p in lines:
            p[11] = f"{float(p[11]) + 0.010:.6f}"  # camera-frame x location
            p[13] = f"{float(p[13]) + 0.013:.6f}"  # camera-frame z location
        (pred_dir / (name + ".txt")).write_text(
            "\n".join(" ".join(p) for p in lines) + ("\n" if lines else ""))

    results = Evaluation(test_annotation_file=GT_DIR).evaluate(
        result_path=str(pred_dir), current_class=[0, 1, 2])

    assert "entire_area" in results and "roi" in results
    ea = results["entire_area"]
    for key in ("Car_3d_all", "Pedestrian_3d_all", "Cyclist_3d_all"):
        assert ea[key] > 99.0, f"{key} = {ea[key]}"
