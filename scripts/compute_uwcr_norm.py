"""Derive UWCR input-normalization constants from OUR training split.

WHY THIS EXISTS
The RAMP-CNN reference standardizes every input view before the forward pass and
always has: train_dop.py:91 sets `Norm = True` as an unconditional local in main()
-- no flag, no config override -- and applies it to all three views at :228-234.
test.py does the same (:17, :177-189); the "finished ra normalization" lines in the
reference README's own sample output (:98-99) are that code running. Our UWCR
__getitem__ never did it: the only normalize_* call there is normalize_confmap,
which acts on the LABEL confmap, not the input.

WHY NOT JUST COPY config.py's CONSTANTS
Because they do not describe our data. The reference's mean1/std1 pair was fit to
the shipped train_test_data, whose slices sit around std 0.008 (RA) / 0.035 (RV) /
0.024 (VA). Slices computed from the public raw ADC -- by the reference's OWN
unmodified slice3d.py, not just by ours -- are roughly 5000x larger. That gap is
not a bug on our side: running reference code on the reference's own
slice_sample_data does not reproduce its own shipped train_test_data either
(RA r 0.14-0.44, RV 0.37-0.50, VA 0.95-0.97), so train_test_data is a different
vintage of the data and its statistics do not transfer. Hence: derive our own.

TRAIN SPLIT ONLY. val and test sequences are excluded -- fitting normalization
statistics on held-out data leaks it. The split rule is restated from
scripts/smoke_sweep.py build_split (the UWCR branch): test is capture date
2019_05_29, val is three named sequences, train is the remaining 14. Keep the two
in sync; SPLIT_SRC below names the authority.

The constants are computed over cached frames weighted uniformly. Training windows
(win_size 16, stride 8) overlap 50%, so a frame's true sample weight is ~2x except
at sequence edges -- a uniform mean is within a rounding error of that and is what
the reference does too.

Usage:
    .venv/bin/python scripts/compute_uwcr_norm.py [--out docs/uwcr-norm.json]
"""
import argparse
import glob
import json
import os
from multiprocessing import Pool

import numpy as np

CACHE_ROOT = os.path.join(
    os.environ.get("TMP_ROOTDIR", "/home/kangle/dataset/tmp"), "uwcr_frame_cache")

SPLIT_SRC = "scripts/smoke_sweep.py build_split(), UWCR branch"
TEST_DATE = "2019_05_29"
VAL_NAMES = {"2019_04_30_pbms002", "2019_04_30_pcms001", "2019_05_09_cm1s003"}

# One constant pair per VIEW, pooled across that view's channels -- this mirrors the
# reference, where `data[ib, :] = (data[ib, :] - mean1) / std1` hits the whole RA
# tensor including both the real and the imaginary channel.
VIEWS = {"ra": ("ra0", "ra128"), "rv": ("rv0", "rv1"), "va": ("va0", "va1")}


def split_of(seq_name):
    if seq_name[:10] == TEST_DATE:
        return "test"
    if seq_name in VAL_NAMES:
        return "val"
    return "train"


def accumulate(seq_dir):
    """Return {view: (count, sum, sumsq)} over every cached frame of one sequence.

    float64 throughout: the slices are float32 and RV/VA run to ~1e3, so a float32
    sum of squares over ~1.5e7 elements per sequence would lose most of its
    precision to accumulation error.
    """
    acc = {v: [0, 0.0, 0.0] for v in VIEWS}
    for path in sorted(glob.glob(os.path.join(seq_dir, "*.npz"))):
        with np.load(path) as z:
            for view, keys in VIEWS.items():
                for k in keys:
                    a = z[k].astype(np.float64, copy=False).ravel()
                    acc[view][0] += a.size
                    acc[view][1] += a.sum()
                    acc[view][2] += (a * a).sum()
    return os.path.basename(seq_dir), acc


def stats(count, total, totalsq):
    mean = total / count
    # E[x^2] - E[x]^2, clamped: the two terms are within ~1e-9 of each other only if
    # the view is constant, but a negative from float cancellation would nan the sqrt.
    var = max(totalsq / count - mean * mean, 0.0)
    return mean, np.sqrt(var)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=None,
                    help="uwcr_frame_cache/<hash> dir; default: the only one present")
    ap.add_argument("--out", default="docs/uwcr-norm.json")
    ap.add_argument("--jobs", type=int, default=8)
    args = ap.parse_args()

    cache = args.cache
    if cache is None:
        cands = sorted(glob.glob(os.path.join(CACHE_ROOT, "*")))
        cands = [c for c in cands if os.path.isdir(c)]
        if len(cands) != 1:
            raise SystemExit(
                "expected exactly one cache dir under %s, found %d: %s\n"
                "pass --cache explicitly. NOTE the dir name is a hash of the source of "
                "get_RD/get_RA/get_RV_VA_slice, so more than one means the slice code "
                "changed and the constants must be recomputed against the current one."
                % (CACHE_ROOT, len(cands), cands))
        cache = cands[0]

    seq_dirs = sorted(d for d in glob.glob(os.path.join(cache, "*")) if os.path.isdir(d))
    train_dirs = [d for d in seq_dirs if split_of(os.path.basename(d)) == "train"]
    if not train_dirs:
        raise SystemExit("no train sequences cached under %s" % cache)

    print("cache: %s" % cache)
    print("split: %s" % SPLIT_SRC)
    print("train sequences: %d of %d cached\n" % (len(train_dirs), len(seq_dirs)))

    with Pool(min(args.jobs, len(train_dirs))) as pool:
        results = pool.map(accumulate, train_dirs)

    print("%-24s %28s %28s %28s" % ("sequence", "RA mean/std", "RV mean/std", "VA mean/std"))
    print("-" * 112)
    per_seq = {}
    for name, acc in results:
        row = {}
        cells = []
        for view in ("ra", "rv", "va"):
            m, s = stats(*acc[view])
            row[view] = {"mean": m, "std": s}
            cells.append("%+12.4f / %11.4f" % (m, s))
        per_seq[name] = row
        print("%-24s %s" % (name, " ".join(cells)))

    totals = {v: [0, 0.0, 0.0] for v in VIEWS}
    for _, acc in results:
        for v in VIEWS:
            for i in range(3):
                totals[v][i] += acc[v][i]

    print()
    out = {
        "_about": "UWCR input-normalization constants, fit on OUR 14-sequence training "
                  "split. Derived, not copied: the reference's config.py constants "
                  "describe its shipped train_test_data, which is a different vintage "
                  "roughly 5000x smaller in scale. See scripts/compute_uwcr_norm.py.",
        "_split": SPLIT_SRC,
        "_train_sequences": sorted(per_seq),
        "_cache": os.path.basename(cache),
        "per_sequence": per_seq,
    }
    print("%-6s %14s %14s %16s" % ("view", "mean", "std", "elements"))
    print("-" * 54)
    for view in ("ra", "rv", "va"):
        c, t, tsq = totals[view]
        m, s = stats(c, t, tsq)
        out[view] = {"mean": m, "std": s, "count": c}
        print("%-6s %+14.6f %14.6f %16d" % (view, m, s, c))

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote %s" % args.out)


if __name__ == "__main__":
    main()
