#!/usr/bin/env python
"""Pre-build the RADDetDataset collapsed-spectrum cache.

DAROD consumes a 256x64 map but each frame on disk is a 256x256x64 complex RAD cube,
so an uncached epoch re-reads ~500x the data it uses (measured: 2709 s/epoch against
CARRADA's 28.9 s). This does that pass once, in parallel; afterwards RADDetDataset
serves the collapsed map straight from disk.

    python scripts/build_raddet_cache.py [--feature RD] [--workers 8]

Idempotent: existing entries are skipped, so it doubles as a repair pass.
"""
import argparse
import os
import sys
import time
from multiprocessing import Pool

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_ROOT)
sys.path.insert(0, os.path.join(REPO_ROOT, "scripts"))

os.environ.setdefault("MODEL_ROOTDIR", "/home/kangle/dataset/trained_models")
os.environ.setdefault("DATASET_ROOTDIR", "/home/kangle/dataset")
os.environ.setdefault("TMP_ROOTDIR", "/home/kangle/dataset/tmp")
os.environ.setdefault("LOGURU_LEVEL", "ERROR")
os.environ.setdefault("DATABASE_URL", "postgresql://postgres:postgres@localhost/postgres")

_ds = None
_feature = None


def _init(feature):
    """One dataset instance per worker process. RADDetDataset is a parser: the cache dir
    and _get_collapsed need no model or train config."""
    global _ds, _feature
    from smoke_sweep import fetch_datafiles, build_split
    from detection_vis_backend.datasets.dataset import DatasetFactory

    _feature = feature
    datafiles, _ = build_split("RADDetDataset", fetch_datafiles()["RADDetDataset"], "original")
    f = datafiles[0]
    _ds = DatasetFactory().get_instance(f["parse"], f["id"])


def _one(index):
    try:
        _ds._get_collapsed(index, _feature)
        return True
    except Exception as e:                       # keep going; report at the end
        return f"{index}: {type(e).__name__}: {e}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--feature", default="RD", choices=["RD", "RA"])
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args()

    # Parse once in the parent so the workers hit a warm registration.
    from smoke_sweep import fetch_datafiles, build_split
    from detection_vis_backend.jobs.tasks import _parse_datafiles
    from detection_vis_backend.datasets.dataset import DatasetFactory

    datafiles, _ = build_split("RADDetDataset", fetch_datafiles()["RADDetDataset"], "original")
    _parse_datafiles(datafiles)
    f = datafiles[0]
    ds = DatasetFactory().get_instance(f["parse"], f["id"])
    n = len(ds)
    cache_dir = os.path.join(ds.collapsed_cache_dir, args.feature)
    print(f"{n} frames -> {cache_dir}  ({args.workers} workers)", flush=True)

    t0 = time.time()
    errors = []
    done = 0
    with Pool(args.workers, initializer=_init, initargs=(args.feature,)) as pool:
        for r in pool.imap_unordered(_one, range(n), chunksize=16):
            done += 1
            if r is not True:
                errors.append(r)
            if done % 250 == 0 or done == n:
                el = time.time() - t0
                eta = el / done * (n - done)
                print(f"  {done}/{n}  elapsed {el/60:.1f}m  eta {eta/60:.1f}m  "
                      f"errors {len(errors)}", flush=True)

    cached = len(os.listdir(cache_dir)) if os.path.isdir(cache_dir) else 0
    size = sum(os.path.getsize(os.path.join(cache_dir, x)) for x in os.listdir(cache_dir)) \
        if os.path.isdir(cache_dir) else 0
    print(f"\ndone in {(time.time()-t0)/60:.1f}m: {cached}/{n} cached, {size/2**30:.2f} GiB")
    if errors:
        print(f"{len(errors)} errors, first 5:")
        for e in errors[:5]:
            print("  ", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
