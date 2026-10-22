#!/usr/bin/env python
"""Score the same CRUW detections through the T-RODNet authors' evaluator and through
ours, and compare.

The reference ships no trained weights, so the checkpoint-through-our-pipeline gate used
for TransRAD is impossible for T-RODNet. What can be checked is the ruler: the authors
revised the public CRUW evaluation package (cruw/eval in the reference clone), and their
paper's 83.27 / 86.98 was measured with it on evaluate/gt/<seq>.txt for four held-out
sequences. This script
  1. compares those four ground-truth files with our TRAIN_RAD_H_ANNO files byte for byte,
  2. converts one of our detection files (<seq>_rod_res.txt, as RODNet_evaluation writes
     it: frame class row col conf, grid indices) to the authors' row format (frame range
     angle class conf, metres / radians) and scores it with cruw.eval.evaluate_rod2021
     imported from the reference clone and with our evaluate_rodnet_seq / accumulate /
     summarize chain, overall and per class,
  3. exits 1 when any number differs by more than --tol points (default 0.01).

    python scripts/trodnet_scorer_check.py --results <run dir> [--seq 2019_05_23_PM1S013]

The authors' evaluator finds the frame count under <data_root>/sequences/test/<seq>/IMAGES_0,
so a data root of symlinks to our TRAIN_CAM_0 folders is built under TMP_ROOTDIR.
"""
import argparse
import contextlib
import io
import json
import os
import re
import shutil
import sys
import tempfile

import numpy as np

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)

os.environ.setdefault("TMP_ROOTDIR", "/home/kangle/dataset/tmp")
os.environ.setdefault("LOGURU_LEVEL", "ERROR")

REF_DIR = "/home/kangle/dataset/trodnet_reference/T-RODNet"
CRUW_ROOT = "/home/kangle/dataset/CRUW"
TEST_SEQS = ("2019_04_09_BMS1001", "2019_04_30_MLMS001", "2019_05_23_PM1S013", "2019_09_29_ONRD005")
CLASSES = ["pedestrian", "cyclist", "car"]


def to_reference_rows(lines, rng_grid, agl_grid, score_decimals=2):
    """Our 'frame class row col conf' rows -> the authors' 'frame range angle class conf '
    rows (output_results.py:32: "%d %.4f %.4f %s %.2f \\n"), scores clipped at 1 as their
    reader does. score_decimals=2 is their format; 4 keeps our scores intact to show how
    much of a scorer difference is only that rounding."""
    out = []
    for line in lines:
        parts = line.split()
        if len(parts) != 5:
            continue
        frame, cls, row, col, conf = parts
        conf = min(float(conf), 1.0)
        out.append("%d %.4f %.4f %s %.*f " % (int(frame), rng_grid[int(row)], agl_grid[int(col)], cls,
                                             score_decimals, conf))
    return out


def compare_ground_truth():
    identical = []
    for seq in TEST_SEQS:
        theirs = os.path.join(REF_DIR, "evaluate", "gt", seq + ".txt")
        ours = os.path.join(CRUW_ROOT, "TRAIN_RAD_H_ANNO", seq + ".txt")
        with open(theirs, "rb") as a, open(ours, "rb") as b:
            identical.append(a.read() == b.read())
    return identical


def score_with_reference(rows_by_seq):
    """rows_by_seq: {seq: converted rows}. The authors' evaluator pools every file in the
    submit dir, as it pools their four test sequences; their accumulate has no guard for
    a class absent from both ground truth and detections, so pooling several sequences
    is also what keeps it from failing on a single-class sequence."""
    sys.path.insert(0, REF_DIR)
    from cruw import CRUW
    from cruw.eval import evaluate_rod2021
    work = tempfile.mkdtemp(prefix="trodnet_scorer_", dir=os.environ["TMP_ROOTDIR"])
    try:
        seq_root = os.path.join(work, "data_root", "sequences", "test")
        submit, truth = os.path.join(work, "sub"), os.path.join(work, "gt")
        for d in (seq_root, submit, truth):
            os.makedirs(d)
        for seq, rows in rows_by_seq.items():
            os.symlink(os.path.join(CRUW_ROOT, "TRAIN_CAM_0", seq), os.path.join(seq_root, seq))
            with open(os.path.join(submit, seq + ".txt"), "w") as f:
                f.write("\n".join(rows) + "\n")
            shutil.copy(os.path.join(CRUW_ROOT, "TRAIN_RAD_H_ANNO", seq + ".txt"), os.path.join(truth, seq + ".txt"))
        dataset = CRUW(data_root=os.path.join(work, "data_root"), sensor_config_name="sensor_config_rod2021")
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            evaluate_rod2021(submit, truth, dataset)
    finally:
        shutil.rmtree(work, ignore_errors=True)
    nums = {}
    for line in buf.getvalue().splitlines():
        for key, val in re.findall(r"(A[PR]_\w+): ([0-9.]+)", line):
            nums[key] = float(val)
    return nums


def score_with_ours(results_dir, seqs, rng_grid, agl_grid):
    """The pooling RODNet_evaluation does over its sequences (evaluate.py)."""
    from detection_vis_backend.train.evaluate import evaluate_rodnet_seq, accumulate, summarize
    olsThrs = np.around(np.linspace(0.5, 0.9, int(np.round((0.9 - 0.5) / 0.05) + 1), endpoint=True), decimals=2)
    recThrs = np.around(np.linspace(0.0, 1.0, int(np.round((1.0 - 0.0) / 0.01) + 1), endpoint=True), decimals=2)
    evalImgs_all, n_frames_all = [], 0
    for seq in seqs:
        res_path = os.path.join(results_dir, seq + "_rod_res.txt")
        gt_path = os.path.join(CRUW_ROOT, "TRAIN_RAD_H_ANNO", seq + ".txt")
        n_frame = len(os.listdir(os.path.join(CRUW_ROOT, "TRAIN_CAM_0", seq, "IMAGES_0")))
        evalImgs_all.extend(evaluate_rodnet_seq(res_path, gt_path, n_frame, 3, CLASSES, rng_grid, agl_grid, olsThrs, recThrs))
        n_frames_all += n_frame
    ev = accumulate(evalImgs_all, n_frames_all, olsThrs, recThrs, 3, CLASSES)
    stats = summarize(ev, olsThrs, recThrs, 3, gl=False)
    nums = {"AP_total": stats[0] * 100, "AR_total": stats[1] * 100}
    for k, name in enumerate(CLASSES):
        p, r = ev["precision"][:, :, k], ev["recall"][:, k]
        nums["AP_" + name] = float(np.mean(p[p > -1]) * 100) if (p > -1).any() else 0.0
        nums["AR_" + name] = float(np.mean(r[r > -1]) * 100) if (r > -1).any() else 0.0
    return nums


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", required=True, help="run directory holding <seq>_rod_res.txt files")
    p.add_argument("--seqs", nargs="+", help="sequences to pool (default: every *_rod_res.txt in --results)")
    p.add_argument("--tol", type=float, default=0.2,
                   help="max allowed difference in AP/AR points; the authors' file format rounds "
                        "scores to 0.01, which reorders near-ties and moves AP by up to ~0.1")
    p.add_argument("--score-decimals", type=int, default=2,
                   help="2 = the authors' format; 4 keeps our scores intact (isolates the rounding)")
    args = p.parse_args()

    identical = compare_ground_truth()
    print(f"gt files: {'identical' if all(identical) else 'DIFFER'} ({sum(identical)}/{len(identical)})")

    from detection_vis_backend.datasets.utils import confmap2ra
    with open(os.path.join(CRUW_ROOT, "sensor_config_rod2021.json")) as f:
        radar_cfg = json.load(f)["radar_cfg"]
    rng_grid = confmap2ra(radar_cfg, name="range")
    agl_grid = confmap2ra(radar_cfg, name="angle")

    seqs = args.seqs or sorted(n[:-len("_rod_res.txt")] for n in os.listdir(args.results) if n.endswith("_rod_res.txt"))
    rows_by_seq = {}
    for seq in seqs:
        with open(os.path.join(args.results, seq + "_rod_res.txt")) as f:
            rows_by_seq[seq] = to_reference_rows(f.read().splitlines(), rng_grid, agl_grid, args.score_decimals)
    print(f"{len(seqs)} sequences, {sum(len(r) for r in rows_by_seq.values())} detections")

    theirs = score_with_reference(rows_by_seq)
    ours = score_with_ours(args.results, seqs, rng_grid, agl_grid)
    keys = ["AP_total", "AR_total"] + [f"{m}_{c}" for c in CLASSES for m in ("AP", "AR")]
    print(f"{'metric':14s} {'authors':>9s} {'ours':>9s} {'diff':>8s}")
    worst = 0.0
    for k in keys:
        a, b = theirs.get(k, float('nan')), ours.get(k, float('nan'))
        worst = max(worst, abs(a - b))
        print(f"{k:14s} {a:9.4f} {b:9.4f} {a - b:8.4f}")
    verdict = "match" if worst <= args.tol else f"MISMATCH (worst diff {worst:.4f} > {args.tol})"
    print(f"scorers: {verdict}")
    return 0 if all(identical) and worst <= args.tol else 1


if __name__ == "__main__":
    sys.exit(main())
